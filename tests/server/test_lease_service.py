"""Lease events and expiration must run on the owner thread even without HTTP traffic."""
import asyncio
import threading
import time

import httpx

from nanoxpu.kv.descriptor import KVDescriptor, ModelLayout
from nanoxpu.kv.leases import LeaseTable, LeaseCapacityError
from nanoxpu.server.app import create_app
from nanoxpu.server.config import ServerConfig
from tests.helpers.support import TestEngine
from tests.helpers.http import payload


class LeaseEngine(TestEngine):
    def __init__(self, config, *, active=False, fence=True, lifetime=60):
        super().__init__(config)
        self.active_read = active
        self.fenced = fence
        self.lifetime = lifetime
        self.released = threading.Event()
        self.release_thread = None

    def start(self):
        super().start()
        layout = ModelLayout(model_fingerprint="a" * 64, tokenizer_fingerprint="b" * 64,
                             template_fingerprint="c" * 64, rope_fingerprint="d" * 64,
                             dtype="float16", layers=1, kv_heads=1, head_dim=1, block_size=4)
        descriptor = KVDescriptor(cluster_id="cluster", lease_id="lease", request_id="request", generation="generation",
                                  layout=layout, source_num_blocks=1, source_block_ids=(0,), strides_bytes=layout.strides(1),
                                  prompt_tokens=4, cached_tokens=3, expires_at=time.time() + self.lifetime,
                                  registration_id="registration", content_digest="e" * 64, prompt_digest="f" * 64)
        self.leases = LeaseTable(1, 1024)
        self.leases.export(descriptor, pin=lambda: None, release=self.release_lease, fence=lambda: self.fenced)
        if self.active_read:
            self.leases.handle("begin_read", "lease", "generation")

    def release_lease(self):
        self.release_thread = threading.get_ident()
        self.released.set()

    def tick(self):
        self.leases.tick()
        return self.leases.healthy

    def close(self):
        self.leases.close()
        super().close()


def test_periodic_expiry_without_incoming_requests(tmp_path):
    engine = LeaseEngine(ServerConfig(str(tmp_path)), lifetime=0.15)
    async def scenario():
        app = create_app(engine.config, lambda _: engine)
        async with app.router.lifespan_context(app):
            assert await asyncio.to_thread(engine.released.wait, 3)
            assert engine.release_thread == engine.calls[0][1]
            assert engine.release_thread != threading.get_ident()
            assert app.state.service.ready
            assert not engine.leases.leases
        assert engine.closed == 1
    asyncio.run(scenario())


def test_callback_ack_does_not_release_before_read_completion(tmp_path):
    engine = LeaseEngine(ServerConfig(str(tmp_path)), active=True)
    async def scenario():
        app = create_app(engine.config, lambda _: engine)
        async with app.router.lifespan_context(app):
            engine.leases.post("ack", "lease", "generation")
            await asyncio.sleep(0.25)
            assert not engine.released.is_set()
            engine.leases.post("read_finished", "lease", "generation")
            assert await asyncio.to_thread(engine.released.wait, 3)
            assert engine.release_thread == engine.calls[0][1]
    asyncio.run(scenario())


def test_unfenced_expiry_marks_service_unavailable_and_keeps_allocation(tmp_path):
    engine = LeaseEngine(ServerConfig(str(tmp_path)), active=True, fence=False, lifetime=0.15)
    async def scenario():
        app = create_app(engine.config, lambda _: engine)
        notifications = []
        app.state.on_fatal = lambda: notifications.append("failed")
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                deadline = time.monotonic() + 3
                while not app.state.service.failed and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                assert (await client.get("/ready")).status_code == 503
                assert notifications == ["failed"]
                assert not engine.released.is_set()
                assert engine.leases.leases["lease"].state == "QUARANTINED"
                # A later, positively observed transport fence permits shutdown cleanup.
                engine.fenced = True
        assert engine.released.is_set()
    asyncio.run(scenario())


def test_lease_budget_rejection_is_recoverable_http_429(tmp_path):
    engine = LeaseEngine(ServerConfig(str(tmp_path), served_model_name="qwen-test"))
    async def scenario():
        app = create_app(engine.config, lambda _: engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
                original = engine.complete
                def busy(_request):
                    raise LeaseCapacityError("lease limit")
                engine.complete = busy
                assert (await client.post("/v1/chat/completions", json=payload())).status_code == 429
                assert (await client.get("/ready")).status_code == 200
                engine.complete = original
                assert (await client.post("/v1/chat/completions", json=payload())).status_code == 200
    asyncio.run(scenario())
