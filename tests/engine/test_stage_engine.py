from tests.helpers.cpu_kv import baseline
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import threading
import time
from types import SimpleNamespace
import uuid

import httpx
import numpy as np
import pytest

from nanoxpu.engine.stage_engine import StageEngine
from nanoxpu.server.app import create_app
from nanoxpu.server.config import ServerConfig
from tests.helpers.cpu_kv import CPUModel, CPUHub, CPUConnector
from tests.helpers.http import PREFILL_KV


@asynccontextmanager
async def pair(root, *, options=None, eos_after=None, max_leases=16, blocks=32):
    hub = CPUHub()
    config = ServerConfig(str(root), served_model_name="cpu-stage", role="prefill", instance_id="p",
                          engine_generation="p-generation", cluster_id="cluster", advertise_ip="127.0.0.1",
                          kv_port=24000, kv_connector="nixl-pull", max_num_batched_tokens=3)
    decode = replace(config, role="decode", instance_id="d", engine_generation="d-generation",
                     kv_port=24001, prefill_peers=("p=p-generation",))
    p = StageEngine(config, CPUModel(blocks), CPUConnector(hub), lease_seconds=3, transfer_timeout=0.5, max_leases=max_leases)
    d = StageEngine(decode, CPUModel(blocks, eos_after=eos_after), CPUConnector(hub, **(options or {})), lease_seconds=3, transfer_timeout=0.5)
    p_app = create_app(config, lambda _: p, internal_token="test-token")
    d_app = create_app(decode, lambda _: d, internal_token="test-token")
    async with p_app.router.lifespan_context(p_app), d_app.router.lifespan_context(d_app):
        # P and D have different physical block IDs in separate arrays.
        await p_app.state.service.run(p.scheduler.block_manager.free_block_ids.rotate, 3)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=p_app), base_url="http://p") as pc:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=d_app), base_url="http://d") as dc:
                yield SimpleNamespace(p=p, d=d, pc=pc, dc=dc, p_app=p_app, d_app=d_app, hub=hub)


def request(text, budget, metadata=None):
    body = {"model": "cpu-stage", "messages": [{"role": "user", "content": text}], "max_tokens": budget}
    if metadata is None:
        body.update(max_tokens=1, min_tokens=1, kv_transfer_params=PREFILL_KV)
    else:
        body["kv_transfer_params"] = metadata
    return body


def headers():
    return {"Authorization": "Bearer test-token", "X-Request-Id": uuid.uuid4().hex}




@pytest.mark.parametrize("length,budget", [(1, 1), (1, 3), (3, 3), (4, 1), (4, 3), (5, 3), (9, 3), (17, 5)])
def test_pd_matches_full_cpu_computation_and_resumes_only_last_prompt_token(tmp_path, length, budget):
    text = "abcde" * (length // 5) + "abcde"[:length % 5]
    expected = baseline(text, budget)
    async def scenario():
        async with pair(tmp_path) as state:
            identity = headers()
            exported = await state.pc.post("/v1/chat/completions", json=request(text, 1), headers=identity)
            assert exported.status_code == 200, exported.text
            meta = exported.json()["kv_transfer_params"]
            assert meta["nano"]["cached_tokens"] == length - 1
            assert len(state.p.leases.leases) == 1
            assert state.p.scheduler.is_finished()
            result = await state.dc.post("/v1/chat/completions", json=request(text, budget, meta), headers=identity)
            assert result.status_code == 200, result.text
            assert result.json()["choices"][0]["message"]["content"] == expected
            assert result.json()["usage"] == {"prompt_tokens": length, "completion_tokens": budget, "total_tokens": length + budget}
            assert state.d.model.runs[0][:3] == (length - 1, 1, True)
            assert all(not prefill for _, _, prefill, _ in state.d.model.runs[1:])
            assert not np.shares_memory(state.p.model.storage, state.d.model.storage)
            assert not state.hub.regions and not state.p.leases.leases
            for engine in (state.p, state.d):
                assert not engine.scheduler.block_manager.used_block_ids
                assert not engine.scheduler.block_manager.hash_to_block_id
                assert not engine.quarantine
                assert all(block.ref_count == 0 for block in engine.scheduler.block_manager.blocks)
    asyncio.run(scenario())


def test_eos_uses_original_decode_budget_and_usage(tmp_path):
    async def scenario():
        async with pair(tmp_path, eos_after=2) as state:
            identity = headers()
            meta = (await state.pc.post("/v1/chat/completions", json=request("abcdef", 1), headers=identity)).json()["kv_transfer_params"]
            result = await state.dc.post("/v1/chat/completions", json=request("abcdef", 8, meta), headers=identity)
            assert result.status_code == 200
            assert result.json()["choices"][0]["finish_reason"] == "stop"
            assert result.json()["usage"]["completion_tokens"] == 2
            assert result.json()["choices"][0]["message"]["content"] == baseline("abcdef", 8, eos_after=2)
    asyncio.run(scenario())


def test_disconnect_keeps_reservation_until_copy_visible_and_generation_finishes(tmp_path):
    visible = threading.Event()
    async def scenario():
        async with pair(tmp_path, options={"visible_gate": visible}) as state:
            identity = headers()
            meta = (await state.pc.post("/v1/chat/completions", json=request("abcde", 1), headers=identity)).json()["kv_transfer_params"]
            task = asyncio.create_task(state.dc.post("/v1/chat/completions", json=request("abcde", 3, meta), headers=identity))
            try:
                deadline = time.monotonic() + 2
                while state.d.connector.last_transfer is None and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                ticket = state.d.connector.last_transfer
                assert await asyncio.to_thread(ticket.written.wait, 2)
                assert not state.d.model.runs
                assert state.d.scheduler.block_manager.used_block_ids
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert (await state.dc.post("/v1/chat/completions", json=request("abcde", 3, meta), headers=identity)).status_code == 429
                visible.set()
                await asyncio.wait_for(asyncio.shield(state.d_app.state.service.active), 3)
                assert not state.d.scheduler.block_manager.used_block_ids
                assert len(state.d.model.runs) == 3
                assert not state.p.leases.leases
            finally:
                visible.set()
                if not task.done():
                    await task
    asyncio.run(scenario())


@pytest.mark.parametrize("option", ["copy_error", "corrupt", "ack_error"])
def test_transfer_failures_do_not_forward_and_cleanup_allocations(tmp_path, option):
    async def scenario():
        async with pair(tmp_path, options={option: True}) as state:
            identity = headers()
            meta = (await state.pc.post("/v1/chat/completions", json=request("abcde", 1), headers=identity)).json()["kv_transfer_params"]
            response = await state.dc.post("/v1/chat/completions", json=request("abcde", 3, meta), headers=identity)
            assert response.status_code == 500
            assert not state.d.model.runs
            assert not state.d.scheduler.block_manager.used_block_ids
            assert not state.p.leases.leases
            assert (await state.dc.get("/ready")).status_code == 503
    asyncio.run(scenario())


def test_unproven_fence_retains_destination_until_positive_completion(tmp_path):
    visible = threading.Event()
    async def scenario():
        async with pair(tmp_path, options={"visible_gate": visible, "fence_allowed": False}) as state:
            identity = headers()
            meta = (await state.pc.post("/v1/chat/completions", json=request("abcde", 1), headers=identity)).json()["kv_transfer_params"]
            response = await state.dc.post("/v1/chat/completions", json=request("abcde", 3, meta), headers=identity)
            assert response.status_code == 500
            assert state.d.quarantine and state.d.scheduler.block_manager.used_block_ids
            assert not state.d.model.runs
            ticket = state.d.connector.last_transfer
            # Permission to fence is not completion of the copy thread. Wait for
            # its explicit signal before asking the nonblocking tick to reclaim.
            assert await asyncio.to_thread(ticket.finished.wait, 2)
            ticket.fence_allowed = True
            await state.d_app.state.service.run(state.d.tick)
            assert not state.d.quarantine and not state.d.scheduler.block_manager.used_block_ids
            assert (await state.dc.get("/ready")).status_code == 503
    asyncio.run(scenario())


def test_prompt_content_mismatch_is_rejected_before_reservation(tmp_path):
    async def scenario():
        async with pair(tmp_path) as state:
            identity = headers()
            meta = (await state.pc.post("/v1/chat/completions", json=request("abcde", 1), headers=identity)).json()["kv_transfer_params"]
            response = await state.dc.post("/v1/chat/completions", json=request("vwxyz", 3, meta), headers=identity)
            assert response.status_code == 400
            assert not state.d.model.runs and not state.d.scheduler.block_manager.used_block_ids
            assert (await state.dc.get("/ready")).status_code == 200
    asyncio.run(scenario())


def test_prefill_lease_budget_rejects_without_allocating_or_computing(tmp_path):
    async def scenario():
        async with pair(tmp_path, max_leases=1) as state:
            assert (await state.pc.post("/v1/chat/completions", json=request("abcde", 1), headers=headers())).status_code == 200
            runs = len(state.p.model.runs)
            used = set(state.p.scheduler.block_manager.used_block_ids)
            response = await state.pc.post("/v1/chat/completions", json=request("abcde", 1), headers=headers())
            assert response.status_code == 429
            assert len(state.p.model.runs) == runs
            assert state.p.scheduler.block_manager.used_block_ids == used
    asyncio.run(scenario())


def test_consumed_lease_cannot_replay_decode(tmp_path):
    async def scenario():
        async with pair(tmp_path) as state:
            identity = headers()
            meta = (await state.pc.post("/v1/chat/completions", json=request("abcde", 1), headers=identity)).json()["kv_transfer_params"]
            body = request("abcde", 3, meta)
            assert (await state.dc.post("/v1/chat/completions", json=body, headers=identity)).status_code == 200
            runs = len(state.d.model.runs)
            assert (await state.dc.post("/v1/chat/completions", json=body, headers=identity)).status_code == 400
            assert len(state.d.model.runs) == runs
            assert (await state.dc.get("/ready")).status_code == 200
    asyncio.run(scenario())


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_compute_failure_without_fence_keeps_owned_blocks(tmp_path, role):
    async def scenario():
        async with pair(tmp_path) as state:
            identity = headers()
            if role == "decode":
                meta = (await state.pc.post("/v1/chat/completions", json=request("abcde", 1), headers=identity)).json()["kv_transfer_params"]
                engine, app, client, body = state.d, state.d_app, state.dc, request("abcde", 3, meta)
            else:
                engine, app, client, body = state.p, state.p_app, state.pc, request("abcde", 1)
            engine.model.compute_error = True
            engine.model.fence_allowed = False
            result = await client.post("/v1/chat/completions", json=body, headers=identity)
            assert result.status_code == 500
            assert engine.quarantine and engine.scheduler.block_manager.used_block_ids
            assert not engine.model.closed
            engine.model.fence_allowed = True
            await app.state.service.run(engine.tick)
            assert not engine.quarantine and not engine.scheduler.block_manager.used_block_ids
    asyncio.run(scenario())
