"""Real pinned vllm-router binary against actual HTTP CPU workers. No proxy replacement."""
from contextlib import contextmanager, ExitStack
import json
import os
import subprocess
import sys
import time

import httpx
import pytest

from tests.helpers.support import write_model_fixture
from tests.helpers.process import ROOT, free_port


def wait_http(url, process, log, path="/ready"):
    deadline = time.monotonic() + 45
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            assert process.poll() is None, log.read_text(encoding="utf-8", errors="replace")
            try:
                if client.get(url + path).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.1)
    pytest.fail(log.read_text(encoding="utf-8", errors="replace"))


@contextmanager
def process(command, log, env=None):
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        yield child
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)


@contextmanager
def cluster(root, binary, *, pd=False, workers=1):
    model = write_model_fixture(root / "model")
    used = set()
    def port():
        value = free_port()
        while value in used:
            value = free_port()
        used.add(value)
        return value
    roles = [("p", "prefill"), ("d", "decode")] if pd else [(f"w{i}", "colocated") for i in range(workers)]
    endpoints = []
    token = "cpu-router-internal-test-token"
    env = {**os.environ, "NANO_XPU_INTERNAL_TOKEN": token}
    with ExitStack() as stack:
        for name, role in roles:
            address = port()
            command = [sys.executable, "-m", "tests.helpers.router_worker", "_worker", str(model), "--served-model-name", "cpu-contract",
                       "--port", str(address), "--distributed-port", str(port()), "--role", role, "--instance-id", name,
                       "--engine-generation", name + "-generation", "--cluster-id", "cpu-router", "--advertise-ip", "127.0.0.1"]
            if pd:
                command += ["--kv-port", str(port()), "--kv-connector", "nixl-pull"]
                if role == "decode":
                    command += ["--prefill-peer", "p=p-generation"]
            log = root / f"worker-{name}.log"
            child = stack.enter_context(process(command, log, env))
            url = f"http://127.0.0.1:{address}"
            wait_http(url, child, log)
            endpoints.append(url)
        router_port = port()
        command = [str(binary), "--host", "127.0.0.1", "--port", str(router_port), "--policy", "round_robin",
                   "--health-check-endpoint", "/ready", "--disable-retries", "--request-timeout-secs", "10",
                   "--worker-startup-timeout-secs", "30"]
        if pd:
            command += ["--vllm-pd-disaggregation", "--kv-connector", "nixl", "--prefill", endpoints[0], "--decode", endpoints[1]]
        else:
            command += ["--worker-urls", *endpoints]
        log = root / "router.log"
        child = stack.enter_context(process(command, log, {**env, "OPENAI_API_KEY": token}))
        url = f"http://127.0.0.1:{router_port}"
        wait_http(url, child, log, "/health")
        with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
            deadline = time.monotonic() + 30
            while True:
                pool = client.get("/workers").json().get("workers", [])
                if len(pool) == len(roles) and all(worker["is_healthy"] for worker in pool):
                    break
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.1)
            yield client, model


def body(text="hello", budget=7):
    return {"model": "cpu-contract", "messages": [{"role": "user", "content": text}], "max_tokens": budget, "stream": False}


def captures(model, name):
    path = model / f"capture-{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("workers", [1, 2])
def test_real_router_routes_single_and_multiple_workers(tmp_path, router_binary, workers):
    with cluster(tmp_path, router_binary, workers=workers) as (client, model):
        observed = set()
        for _ in range(workers * 4):
            result = client.post("/v1/chat/completions", json=body())
            assert result.status_code == 200, result.text
            observed.add(result.json()["choices"][0]["message"]["content"])
        assert observed == {f"CPU CONTRACT w{i}" for i in range(workers)}
        assert all(captures(model, f"w{i}") for i in range(workers))


def test_real_pd_router_preserves_nano_metadata_and_original_budget(tmp_path, router_binary):
    with cluster(tmp_path, router_binary, pd=True) as (client, model):
        result = client.post("/v1/chat/completions", json=body(budget=7))
        assert result.status_code == 200, result.text
        assert result.json()["choices"][0]["message"]["content"] == "CPU CONTRACT d"
        assert result.json()["usage"]["completion_tokens"] == 7
        assert "INTERNAL PREFILL" not in result.text and "kv_transfer_params" not in result.json()
        p, d = captures(model, "p"), captures(model, "d")
        assert len(p) == len(d) == 1
        assert p[0]["body"]["max_tokens"] == p[0]["body"]["min_tokens"] == 1
        assert p[0]["body"]["stream"] is False
        assert d[0]["body"]["max_tokens"] == 7
        assert p[0]["request_id"] == d[0]["request_id"]
        assert p[0]["request_id"].startswith("___prefill_addr_")
        assert d[0]["body"]["kv_transfer_params"] == json.loads((model / "exported.json").read_text())


@pytest.mark.parametrize("text", ["BUSY", "FAIL_PREFILL"])
def test_real_pd_router_does_not_replay_failed_prefill(tmp_path, router_binary, text):
    with cluster(tmp_path, router_binary, pd=True) as (client, model):
        result = client.post("/v1/chat/completions", json=body(text))
        assert result.status_code >= 400
        time.sleep(0.3)
        assert len(captures(model, "p")) == 1
        assert not captures(model, "d")


@pytest.mark.parametrize("text", ["BUSY_DECODE", "FAIL_DECODE"])
def test_real_pd_router_does_not_replay_failed_decode(tmp_path, router_binary, text):
    with cluster(tmp_path, router_binary, pd=True) as (client, model):
        result = client.post("/v1/chat/completions", json=body(text))
        assert result.status_code >= 400
        time.sleep(0.3)
        assert len(captures(model, "p")) == len(captures(model, "d")) == 1


def test_real_router_client_disconnect_does_not_replay_generation(tmp_path, router_binary):
    with cluster(tmp_path, router_binary) as (client, model):
        with pytest.raises(httpx.ReadTimeout):
            client.post("/v1/chat/completions", json=body("SLOW"), timeout=0.1)
        deadline = time.monotonic() + 5
        while not captures(model, "w0") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(captures(model, "w0")) == 1
        # The service retains the execution slot until synthetic computation ends.
        time.sleep(0.6)
        result = client.post("/v1/chat/completions", json=body("after disconnect"))
        assert result.status_code == 200, result.text
        assert [item["body"]["messages"][0]["content"] for item in captures(model, "w0")] == ["SLOW", "after disconnect"]
