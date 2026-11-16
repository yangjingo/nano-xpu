"""Worker role contracts. Real upstream-router forwarding is tested separately on Linux."""
from tests.helpers.http import PREFILL_KV
import asyncio
import hashlib
import json
import time

import httpx
import pytest

from nanoxpu.server.app import create_app
from nanoxpu.server.config import ServerConfig
from nanoxpu.server.engine import Completion
from nanoxpu.server.metadata import Capabilities, Fingerprints, model_fingerprints
from tests.helpers.support import TestEngine, write_model_fixture
from tests.helpers.http import payload

TOKEN = "test-internal-token-not-for-logging"
HEADERS = {"Authorization": "Bearer " + TOKEN, "X-Request-Id": "___prefill_addr_127.0.0.1:1___decode_addr_127.0.0.1:2_abc"}


def transfer(**changes):
    return {"do_remote_decode": False, "do_remote_prefill": True, "remote_engine_id": "prefill-1",
            "remote_host": "127.0.0.1", "remote_port": 24000, "remote_block_ids": [7],
            "nano": {"protocol_version": 1, "lease_id": "lease-1", "generation": "p-generation",
                     "expires_at": time.time() + 60, "cached_tokens": 4, "descriptor_digest": "a" * 64}, **changes}


class StageHTTPFixture(TestEngine):
    def __init__(self, config):
        super().__init__(config)
        self.capabilities = Capabilities(transport="cpu-test", transport_version="cpu-test-v1", connector_ready=True,
                                         layout_digest="b" * 64,
                                         fingerprints=Fingerprints(**model_fingerprints(config.model)))
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if self.config.role == "prefill":
            return Completion("INTERNAL P OUTPUT", 5, 1, "length", transfer())
        return super().complete(request)


def config_for(root, role):
    return ServerConfig(str(root), served_model_name="qwen-test", role=role, cluster_id="cluster",
                        instance_id="prefill-1" if role == "prefill" else "decode-1",
                        engine_generation="p-generation" if role == "prefill" else "d-generation",
                        kv_connector="nixl-pull", kv_port=24000,
                        prefill_peers=("prefill-1=p-generation",) if role == "decode" else ())


def run_stage(root, role, scenario, *, engine_change=lambda engine: None, token=TOKEN):
    config = config_for(write_model_fixture(root), role)
    engine = StageHTTPFixture(config)
    engine_change(engine)
    async def run():
        app = create_app(config, lambda _: engine, internal_token=token)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                await scenario(client, engine)
    asyncio.run(run())
    return engine


def test_prefill_and_decode_contracts_hide_internal_output(tmp_path):
    exported = {}
    async def prefill(client, engine):
        body = payload(max_tokens=1, min_tokens=1, kv_transfer_params=PREFILL_KV)
        result = await client.post("/v1/chat/completions", json=body, headers=HEADERS)
        assert result.status_code == 200, result.text
        exported.update(result.json()["kv_transfer_params"])
        assert engine.requests[0].request_id == HEADERS["X-Request-Id"]
        assert (await client.get("/server_info")).status_code == 401
        info = (await client.get("/server_info", headers=HEADERS)).json()
        assert info["engine_generation"] == "p-generation"
        assert info["capabilities"]["transport"] == "cpu-test"
        assert TOKEN not in json.dumps(info)
    run_stage(tmp_path, "prefill", prefill)

    async def decode(client, engine):
        result = await client.post("/v1/chat/completions", json=payload(kv_transfer_params=exported), headers=HEADERS)
        assert result.status_code == 200, result.text
        assert "kv_transfer_params" not in result.json()
        assert "INTERNAL P OUTPUT" not in result.text
        assert engine.requests[0].max_tokens == 8
        assert result.json()["usage"]["prompt_tokens"] == 5
    run_stage(tmp_path, "decode", decode)


@pytest.mark.parametrize("change", [
    {"max_tokens": 2}, {"min_tokens": 0}, {"min_tokens": True}, {"stream": True},
    {"kv_transfer_params": {**PREFILL_KV, "do_remote_decode": 1}},
    {"kv_transfer_params": {**PREFILL_KV, "do_remote_prefill": True}},
    {"kv_transfer_params": {**PREFILL_KV, "remote_host": "127.0.0.1"}},
    {"kv_transfer_params": {**PREFILL_KV, "nano": {}}},
])
def test_prefill_rejects_wrong_stage_before_engine(tmp_path, change):
    async def scenario(client, engine):
        body = {**payload(max_tokens=1, min_tokens=1, kv_transfer_params=PREFILL_KV), **change}
        response = await client.post("/v1/chat/completions", json=body, headers=HEADERS)
        assert response.status_code == 400
        assert engine.requests == []
        assert (await client.get("/ready")).status_code == 200
    run_stage(tmp_path, "prefill", scenario)


@pytest.mark.parametrize("case", ["generation", "engine", "expired", "version", "negative_block", "wrong_stage", "extra", "oversized"])
def test_decode_rejects_stale_or_invalid_metadata(tmp_path, case):
    value = transfer()
    if case == "generation":
        value["nano"]["generation"] = "old"
    elif case == "engine":
        value["remote_engine_id"] = "untrusted"
    elif case == "expired":
        value["nano"]["expires_at"] = time.time() - 1
    elif case == "version":
        value["nano"]["protocol_version"] = True
    elif case == "negative_block":
        value["remote_block_ids"] = [-1]
    elif case == "wrong_stage":
        value["do_remote_prefill"] = False
    elif case == "extra":
        value["nano"]["secret"] = TOKEN
    else:
        value["remote_block_ids"] = list(range(4097))
    async def scenario(client, engine):
        response = await client.post("/v1/chat/completions", json=payload(kv_transfer_params=value), headers=HEADERS)
        assert response.status_code == 400
        assert TOKEN not in response.text
        assert engine.requests == []
    run_stage(tmp_path, "decode", scenario)


@pytest.mark.parametrize("headers, status", [({}, 401), ({"Authorization": "Bearer wrong"}, 401),
                                             ({"Authorization": "Bearer " + TOKEN}, 400)])
def test_internal_auth_and_request_id_required(tmp_path, headers, status):
    async def scenario(client, engine):
        result = await client.post("/v1/chat/completions", json=payload(kv_transfer_params=transfer()), headers=headers)
        assert result.status_code == status
        assert not engine.requests
    run_stage(tmp_path, "decode", scenario)


@pytest.mark.parametrize("case", ["credentials", "connector", "layout"])
def test_uninitialized_pd_cannot_be_ready(tmp_path, case):
    async def scenario(client, engine):
        pytest.fail("uninitialized stage became ready")
    def change(engine):
        if case == "connector":
            engine.capabilities = engine.capabilities.model_copy(update={"connector_ready": False})
        elif case == "layout":
            engine.capabilities = engine.capabilities.model_copy(update={"layout_digest": None})
    with pytest.raises(RuntimeError, match="P/D"):
        run_stage(tmp_path, "prefill", scenario, token="" if case == "credentials" else TOKEN, engine_change=change)


def test_colocated_still_rejects_kv_fields(tmp_path):
    from tests.helpers.http import client_for
    async def scenario():
        async with client_for(tmp_path) as (client, _, engine):
            result = await client.post("/v1/chat/completions", json=payload(kv_transfer_params=transfer()))
            assert result.status_code == 400
            assert not engine.started.is_set()
    asyncio.run(scenario())


def test_final_response_cannot_leak_engine_metadata(tmp_path, caplog):
    async def scenario(client, engine):
        engine.complete = lambda request: Completion(TOKEN, 5, 1, "stop", transfer())
        response = await client.post("/v1/chat/completions", json=payload(kv_transfer_params=transfer()), headers=HEADERS)
        assert response.status_code == 500
        assert TOKEN not in response.text and TOKEN not in caplog.text
        assert (await client.get("/ready")).status_code == 503
    run_stage(tmp_path, "decode", scenario)


def test_content_fingerprints_detect_changes_and_ignore_directory(tmp_path, monkeypatch):
    # Python 3.10 is declared supported and has no hashlib.file_digest.
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    first, second = write_model_fixture(tmp_path / "one"), write_model_fixture(tmp_path / "two")
    before = model_fingerprints(first)
    assert model_fingerprints(second) == before
    (second / "model.safetensors").write_bytes(b"different weights")
    assert model_fingerprints(second)["model"] != before["model"]
    assert model_fingerprints(second)["tokenizer"] == before["tokenizer"]
    (second / "chat_template.jinja").write_text("new template", encoding="utf-8")
    assert model_fingerprints(second)["template"] != before["template"]
    (first / "model.safetensors.index.json").write_text('{"weight_map": {"weight": "missing.safetensors"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        model_fingerprints(first)


ROUTER_DEFAULTS = {"logprobs": False, "no_stop_trim": False, "ignore_eos": False, "continue_final_message": False,
                   "skip_special_tokens": True, "separate_reasoning": True, "stream_reasoning": True,
                   "return_hidden_states": False, "include_reasoning": True}


def test_pinned_router_serialized_defaults_are_accepted(tmp_path):
    async def scenario(client, engine):
        body = payload(max_tokens=1, min_tokens=1, kv_transfer_params=PREFILL_KV, **ROUTER_DEFAULTS)
        response = await client.post("/v1/chat/completions", json=body, headers=HEADERS)
        assert response.status_code == 200, response.text
        assert len(engine.requests) == 1
    run_stage(tmp_path, "prefill", scenario)


@pytest.mark.parametrize("key,neutral", ROUTER_DEFAULTS.items())
def test_router_defaults_do_not_enable_unsupported_behavior(key, neutral):
    from nanoxpu.server.schema import ChatRequest
    for value in (not neutral, int(neutral)):
        with pytest.raises(ValueError):
            ChatRequest.model_validate(payload(**{key: value}))
