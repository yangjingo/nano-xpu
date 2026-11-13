import asyncio
from contextlib import asynccontextmanager
import importlib.metadata
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import threading
import time

import httpx
import pytest

from nanovllm.cli import parse_config
from nanovllm.server.app import create_app
from nanovllm.server.config import ServerConfig
from tests.support import TestEngine


def payload(**changes):
    return {"model": "qwen-test", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8, **changes}


@asynccontextmanager
async def client_for(tmp_path, engine=None):
    config = ServerConfig(str(tmp_path), served_model_name="qwen-test")
    engine = engine or TestEngine(config)
    app = create_app(config, lambda _: engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client, app, engine


def test_imports_and_help_do_not_load_cuda():
    result = subprocess.run([sys.executable, "-c", "import nanovllm, nanovllm.server.app, sys; assert not any(x in sys.modules for x in ('torch', 'transformers', 'triton', 'flash_attn')); from nanovllm.cli import main; main(['serve', '--help'])"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--tensor-parallel-size" in result.stdout
    assert "--max-num-batched-tokens" in result.stdout


def test_console_entry_point():
    entry = next(ep for ep in importlib.metadata.distribution("nano-vllm").entry_points if ep.name == "nano-xpu")
    assert entry.value == "nanovllm.cli:main"
    assert entry.load().__name__ == "main"


@pytest.mark.parametrize("flag,value", [
    ("--tensor-parallel-size", "0"), ("--tensor-parallel-size", "9"),
    ("--max-num-batched-tokens", "0"), ("--max-num-seqs", "-1"),
    ("--max-model-len", "0"), ("--gpu-memory-utilization", "nan"),
    ("--gpu-memory-utilization", "1"), ("--port", "0"), ("--distributed-port", "65536"),
])
def test_cli_rejects_invalid_configuration(tmp_path, flag, value):
    with pytest.raises(SystemExit) as exc:
        parse_config(["serve", str(tmp_path), flag, value])
    assert exc.value.code == 2


def test_cli_missing_model(tmp_path):
    with pytest.raises(SystemExit):
        parse_config(["serve", str(tmp_path / "missing")])


def test_cli_rejects_empty_model_alias(tmp_path):
    with pytest.raises(SystemExit):
        parse_config(["serve", str(tmp_path), "--served-model-name", ""])


def test_split_configuration_is_forwarded(tmp_path):
    config = parse_config(["serve", str(tmp_path), "--tensor-parallel-size", "2", "--max-num-batched-tokens", "128", "--max-num-seqs", "16", "--distributed-port", "2345"])
    async def scenario():
        engine = TestEngine(config)
        app = create_app(config, lambda config_seen: engine if config_seen == config else None)
        async with app.router.lifespan_context(app):
            assert engine.config.tensor_parallel_size == 2
            assert engine.config.max_num_batched_tokens == 128
            assert engine.config.max_num_seqs == 16
            assert engine.config.distributed_port == 2345
    asyncio.run(scenario())


def test_routes_and_lifecycle(tmp_path):
    async def scenario():
        async with client_for(tmp_path) as (client, app, engine):
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/ready")).status_code == 200
            assert (await client.get("/v1/models")).json()["data"][0]["id"] == "qwen-test"
            for _ in range(100):
                response = await client.post("/v1/chat/completions", json=payload())
                assert response.status_code == 200
                result = response.json()
                assert result["object"] == "chat.completion"
                assert result["choices"][0]["message"]["role"] == "assistant"
                assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            assert app.state.service.active is None
        assert engine.closed == 1
        assert len({thread for _, thread in engine.calls}) == 1
        assert engine.calls[0][1] != threading.get_ident()
    asyncio.run(scenario())


@pytest.mark.parametrize("changes", [
    {"messages": []}, {"messages": [{"role": "user", "content": ["image"]}]},
    {"temperature": 0}, {"temperature": "0.6"}, {"temperature": True},
    {"stream": True}, {"stream": "false"}, {"n": 2}, {"n": True},
    {"max_tokens": 0}, {"max_tokens": True}, {"max_tokens": "8"},
    {"stop": ["end"]}, {"tools": []},
])
def test_request_validation(tmp_path, changes):
    async def scenario():
        async with client_for(tmp_path) as (client, _, engine):
            response = await client.post("/v1/chat/completions", json=payload(**changes))
            assert response.status_code == 400, response.text
            assert response.json()["error"]["code"] == "invalid_request"
            assert not engine.started.is_set()
            assert (await client.post("/v1/chat/completions", json=payload())).status_code == 200
    asyncio.run(scenario())


def test_wrong_model_and_capacity_error_are_recoverable(tmp_path):
    async def scenario():
        async with client_for(tmp_path) as (client, _, _engine):
            assert (await client.post("/v1/chat/completions", json=payload(model="unknown"))).status_code == 404
            assert (await client.post("/v1/chat/completions", json=payload(max_tokens=4096))).status_code == 400
            assert (await client.post("/v1/chat/completions", json=payload())).status_code == 200
    asyncio.run(scenario())


def test_not_ready():
    # ASGITransport without lifespan intentionally simulates pre-startup state.
    async def scenario():
        app = create_app(ServerConfig("."), TestEngine)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for method, route in (("GET", "/ready"), ("GET", "/v1/models"), ("POST", "/v1/chat/completions")):
                response = await client.request(method, route, json=payload() if method == "POST" else None)
                assert response.status_code == 503
        await app.state.service.close()
    asyncio.run(scenario())


def test_busy_and_cancelled_connection_keep_slot_until_engine_finishes(tmp_path):
    async def scenario():
        async with client_for(tmp_path) as (client, app, engine):
            engine.release.clear()
            first = asyncio.create_task(client.post("/v1/chat/completions", json=payload()))
            assert await asyncio.to_thread(engine.started.wait, 2)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            try:
                response = await client.post("/v1/chat/completions", json=payload())
                assert response.status_code == 429
                assert response.headers["Retry-After"] == "1"
                assert (await asyncio.wait_for(client.get("/health"), 1)).status_code == 200
                assert (await client.get("/ready")).status_code == 200
            finally:
                engine.release.set()
            await asyncio.wait_for(asyncio.shield(app.state.service.active), 2)
            await asyncio.sleep(0)
            assert (await client.post("/v1/chat/completions", json=payload())).status_code == 200
    asyncio.run(scenario())


def test_start_failure_is_closed(tmp_path):
    engine = TestEngine(ServerConfig(str(tmp_path)))
    engine.fail_start = True
    async def scenario():
        app = create_app(engine.config, lambda _: engine)
        with pytest.raises(RuntimeError, match="startup failure"):
            async with app.router.lifespan_context(app):
                pytest.fail("startup unexpectedly succeeded")
        assert engine.closed == 1
        assert not app.state.service.ready
    asyncio.run(scenario())


def test_fatal_generation_marks_unhealthy(tmp_path):
    async def scenario():
        async with client_for(tmp_path) as (client, app, engine):
            notifications = []
            app.state.on_fatal = lambda: notifications.append("failed")
            engine.fail_generation = True
            response = await client.post("/v1/chat/completions", json=payload())
            assert response.status_code == 500
            assert "injected generation failure" not in response.text
            assert (await client.get("/health")).status_code == 503
            assert (await client.get("/ready")).status_code == 503
            assert (await client.post("/v1/chat/completions", json=payload())).status_code == 503
            assert notifications == ["failed"]
    asyncio.run(scenario())


def test_real_http_subprocess_and_same_port_restart(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    log_path = tmp_path / "cpu-server.log"
    for iteration in range(2):
        with log_path.open("a", encoding="utf-8") as log:
            command = [
                sys.executable, "-m", "tests.cpu_server", "serve", str(tmp_path),
                "--port", str(port), "--served-model-name", "qwen3-0.6b",
            ]
            powershell = shutil.which("pwsh") if sys.platform == "win32" else None
            if powershell and iteration == 0:
                command = [powershell, "-NoProfile", "-File", "docs/rfcs/examples/serve-cpu.ps1",
                           "-Python", sys.executable, "-Port", str(port)]
            process = subprocess.Popen(command, stdout=log, stderr=log, creationflags=flags)
            try:
                deadline = time.monotonic() + 15
                with httpx.Client(base_url=url, timeout=2) as client:
                    while True:
                        assert process.poll() is None, log_path.read_text(encoding="utf-8")
                        try:
                            if client.get("/ready").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        assert time.monotonic() < deadline, log_path.read_text(encoding="utf-8")
                        time.sleep(0.1)
                    # Exercise the supplied request script's embedded Python over a real socket.
                    script = Path("docs/rfcs/examples/request-qwen.sh").read_text(encoding="utf-8")
                    source = script.split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
                    env = {**os.environ, "BASE_URL": url, "SERVED_MODEL_NAME": "qwen3-0.6b"}
                    call = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=10)
                    assert call.returncode == 0, call.stderr
                    assert "CPU TEST ENGINE" in call.stdout
                    if powershell and iteration == 0:
                        call = subprocess.run([
                            powershell, "-NoProfile", "-File", "docs/rfcs/examples/request-cpu.ps1",
                            "-BaseUrl", url, "-Shutdown",
                        ], capture_output=True, text=True, timeout=10)
                        assert call.returncode == 0, call.stderr
                        assert "CPU TEST ENGINE" in call.stdout
                    else:
                        assert client.post("/__test__/shutdown").status_code == 200
                assert process.wait(timeout=10) == 0, log_path.read_text(encoding="utf-8")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)


def test_shutdown_watchdog_bounds_stuck_process():
    source = """
import time
from nanovllm.server.app import create_app
from nanovllm.server.config import ServerConfig
from nanovllm.server.runtime import ManagedServer
from tests.support import TestEngine
config = ServerConfig('.')
server = ManagedServer(create_app(config, TestEngine), config, shutdown_timeout=0.1)
server.request_shutdown()
time.sleep(10)
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "shutdown timed out" in result.stderr
