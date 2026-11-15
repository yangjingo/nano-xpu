"""Real router HTTP path with shared StageEngine and actual CPU-array KV copies.

P/D use separate engine threads and storage in this test process. The router is a
real external binary; the CPU connector is not cross-process transport or NIXL.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import os
import threading
import time

import httpx
import numpy as np
import pytest
import uvicorn

from nanoxpu.engine.stage_engine import StageEngine
from nanoxpu.server.app import create_app
from nanoxpu.server.config import ServerConfig
from tests.helpers.cpu_kv import CPUModel, CPUHub, CPUConnector
from tests.helpers.process import free_port
from tests.helpers.cpu_kv import baseline
from tests.integration.test_router import process, wait_http


@contextmanager
def stage_http(config, hub):
    engine = StageEngine(config, CPUModel(), CPUConnector(hub), lease_seconds=30, transfer_timeout=5)
    app = create_app(config, lambda _: engine, internal_token="cpu-stage-token")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=config.port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        with httpx.Client(timeout=1, trust_env=False) as client:
            while time.monotonic() < deadline:
                assert thread.is_alive(), "CPU stage HTTP server exited during startup"
                try:
                    if client.get(f"http://127.0.0.1:{config.port}/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.05)
            else:
                pytest.fail("CPU stage HTTP server did not become ready")
        yield engine
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "CPU stage HTTP server failed to stop"


@pytest.mark.parametrize("length,budget", [(1, 1), (5, 3), (17, 5)])
def test_real_router_stage_output_matches_full_cpu_prefill(tmp_path, router_binary, length, budget):
    ports = set()
    while len(ports) < 5:
        ports.add(free_port())
    p_port, d_port, router_port, p_kv, d_kv = ports
    p_config = ServerConfig(str(tmp_path), served_model_name="cpu-stage", role="prefill", instance_id="p",
                            engine_generation="p-generation", cluster_id="cpu-stage", advertise_ip="127.0.0.1",
                            port=p_port, kv_port=p_kv, kv_connector="nixl-pull", max_num_batched_tokens=3)
    d_config = replace(p_config, port=d_port, kv_port=d_kv, role="decode", instance_id="d",
                       engine_generation="d-generation", prefill_peers=("p=p-generation",))
    text = ("abcde" * (length // 5 + 1))[:length]
    expected = baseline(text, budget)
    hub = CPUHub()
    with ExitStack() as stack:
        p = stack.enter_context(stage_http(p_config, hub))
        d = stack.enter_context(stage_http(d_config, hub))
        log = tmp_path / "stage-router.log"
        child = stack.enter_context(process([
            str(router_binary), "--host", "127.0.0.1", "--port", str(router_port), "--policy", "round_robin",
            "--vllm-pd-disaggregation", "--kv-connector", "nixl", "--prefill", f"http://127.0.0.1:{p_port}",
            "--decode", f"http://127.0.0.1:{d_port}", "--health-check-endpoint", "/ready", "--disable-retries",
            "--worker-startup-timeout-secs", "15", "--request-timeout-secs", "10"],
            log, {**os.environ, "OPENAI_API_KEY": "cpu-stage-token"}))
        url = f"http://127.0.0.1:{router_port}"
        wait_http(url, child, log, "/health")
        with httpx.Client(base_url=url, timeout=15, trust_env=False) as client:
            deadline = time.monotonic() + 15
            while True:
                pool = client.get("/workers").json().get("workers", [])
                if len(pool) == 2 and all(worker["is_healthy"] for worker in pool):
                    break
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.1)
            result = client.post("/v1/chat/completions", json={
                "model": "cpu-stage", "messages": [{"role": "user", "content": text}], "max_tokens": budget})
            assert result.status_code == 200, result.text
            assert result.json()["choices"][0]["message"]["content"] == expected
            assert result.json()["usage"] == {"prompt_tokens": length, "completion_tokens": budget, "total_tokens": length + budget}
            assert "kv_transfer_params" not in result.json()
        assert d.model.runs[0][:3] == (length - 1, 1, True)
        assert not np.shares_memory(p.model.storage, d.model.storage)
        assert not hub.regions and not p.leases.leases
        for engine in (p, d):
            assert not engine.scheduler.block_manager.used_block_ids
            assert not engine.quarantine
