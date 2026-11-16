import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import hmac
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from nanoxpu.server.engine import InvalidRequest
from nanoxpu.server.schema import ChatRequest, PrefillRequest, DecodeRequest, DecodeKV
from nanoxpu.server.metadata import Capabilities, server_info
from nanoxpu.kv.leases import LeaseCapacityError

logger = logging.getLogger(__name__)


def error(status, message, code):
    headers = {"Retry-After": "1"} if status == 429 else None
    return JSONResponse(
        status_code=status, headers=headers,
        content={"error": {"message": message, "type": "server_error" if status >= 500 else "invalid_request_error", "code": code}},
    )


class EngineService:
    def __init__(self, config, factory, on_fatal):
        self.config = config
        self.factory = factory
        self.on_fatal = on_fatal
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nano-engine")
        self.engine = None
        self.ready = False
        self.failed = False
        self.active = None
        self.capabilities = None
        self.maintenance_task = None

    async def run(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    def initialize(self):
        self.engine = self.factory(self.config)
        self.engine.start()
        capabilities = getattr(self.engine, "capabilities", None)
        if capabilities is not None:
            self.capabilities = Capabilities.model_validate(capabilities)
        if self.config.role != "colocated" and (
            self.capabilities is None or not self.capabilities.connector_ready
            or self.capabilities.transport == "none" or self.capabilities.layout_digest is None
        ):
            raise RuntimeError("P/D cannot become ready without an initialized connector and KV layout")

    async def start(self):
        await self.run(self.initialize)
        self.ready = True
        if callable(getattr(self.engine, "tick", None)):
            self.maintenance_task = asyncio.create_task(self.maintain())

    def fail(self):
        if not self.failed:
            self.failed = True
            self.ready = False
            self.on_fatal()

    async def maintain(self):
        try:
            while self.ready:
                await asyncio.sleep(0.1)
                # Allocator references stay on the same thread as start/complete/close.
                if await self.run(self.engine.tick) is False:
                    raise RuntimeError("connector quarantined an allocation")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Engine maintenance failed; restart required")
            self.fail()

    async def complete(self, request):
        try:
            result = await self.run(self.engine.complete, request)
            response = {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion", "created": int(time.time()),
                "model": self.config.served_model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": result.finish_reason}],
                "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                          "total_tokens": result.prompt_tokens + result.completion_tokens},
            }
            if self.config.role == "prefill":
                transfer = DecodeKV.model_validate(result.kv_transfer_params)
                if (transfer.remote_engine_id != self.config.instance_id
                    or transfer.nano.generation != self.config.engine_generation
                    or transfer.nano.expires_at <= time.time()):
                    raise RuntimeError("prefill returned invalid lease ownership")
                response["kv_transfer_params"] = transfer.model_dump()
            elif result.kv_transfer_params is not None:
                raise RuntimeError("internal metadata must not appear in a final completion")
            return response
        except InvalidRequest as exc:
            return error(400, str(exc), "invalid_request")
        except LeaseCapacityError:
            return error(429, "Pinned KV capacity is busy", "engine_busy")
        except Exception:
            # Exceptions may contain connector metadata; retain the failure class, never request data.
            logger.error("Engine generation failed; restart required")
            self.fail()
            return error(500, "Engine generation failed; restart the server", "engine_failed")

    def release(self, task):
        # The task, not the HTTP connection, owns the slot until computation ends.
        self.active = None
        if not task.cancelled():
            task.exception()

    async def close(self):
        self.ready = False
        try:
            if self.maintenance_task is not None:
                self.maintenance_task.cancel()
                try:
                    await self.maintenance_task
                except asyncio.CancelledError:
                    pass
                self.maintenance_task = None
            if self.active is not None:
                await asyncio.shield(self.active)
            if self.engine is not None:
                await self.run(self.engine.close)
                self.engine = None
        finally:
            self.executor.shutdown(wait=True)


def create_app(config, engine_factory, *, internal_token=None):
    token = os.environ.get("NANO_XPU_INTERNAL_TOKEN", "") if internal_token is None else internal_token

    def authorized(request):
        value = request.headers.get("authorization", "")
        return bool(token) and hmac.compare_digest(value.encode(), ("Bearer " + token).encode())

    @asynccontextmanager
    async def lifespan(app):
        try:
            if config.role != "colocated" and (not token or not config.cluster_id):
                raise RuntimeError("P/D requires deployment credentials and a cluster identity")
            await service.start()
            yield
        finally:
            await service.close()

    app = FastAPI(lifespan=lifespan)
    app.state.on_fatal = lambda: None
    service = EngineService(config, engine_factory, lambda: app.state.on_fatal())
    app.state.service = service

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        message = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in exc.errors())
        return error(400, message, "invalid_request")

    @app.get("/health")
    async def health():
        return error(503, "Engine failed", "engine_failed") if service.failed else {"status": "ok"}

    @app.get("/ready")
    async def ready():
        return {"status": "ready"} if service.ready else error(503, "Engine not ready", "engine_not_ready")

    @app.get("/v1/models")
    async def models():
        if not service.ready:
            return error(503, "Engine not ready", "engine_not_ready")
        return {"object": "list", "data": [{"id": config.served_model_name, "object": "model", "created": 0, "owned_by": "nano-xpu"}]}

    @app.get("/server_info")
    async def metadata(request: Request):
        if not authorized(request):
            return error(401, "Internal credentials required", "unauthorized")
        if not service.ready:
            return error(503, "Engine not ready", "engine_not_ready")
        return server_info(config, service.capabilities)

    @app.post("/v1/chat/completions")
    async def chat(http_request: Request):
        if not service.ready:
            return error(503, "Engine not ready", "engine_not_ready")
        if config.role != "colocated" and not authorized(http_request):
            return error(401, "Internal credentials required", "unauthorized")
        try:
            # Bound control payloads before parsing; raw block/tensor data never belongs here.
            body = bytearray()
            async for chunk in http_request.stream():
                body.extend(chunk)
                if len(body) > 1024 * 1024:
                    return error(413, "Request body exceeds 1 MiB", "request_too_large")
            schema = {"colocated": ChatRequest, "prefill": PrefillRequest, "decode": DecodeRequest}[config.role]
            request = schema.model_validate_json(body)
            if config.role != "colocated":
                request.bind_request_id(http_request.headers.get("x-request-id"))
                if len(request.kv_transfer_params.model_dump_json().encode()) > 64 * 1024:
                    raise ValueError("KV metadata exceeds 64 KiB")
            if config.role == "decode":
                transfer = request.kv_transfer_params
                peers = dict(entry.split("=", 1) for entry in config.prefill_peers)
                if peers.get(transfer.remote_engine_id) != transfer.nano.generation:
                    raise ValueError("unknown prefill instance or stale generation")
                if transfer.nano.expires_at <= time.time():
                    raise ValueError("KV lease expired")
        except (ValidationError, ValueError):
            # Do not echo inputs, credentials or remote registration metadata.
            return error(400, "Request does not match the worker role or lease identity", "invalid_request")
        if request.model != config.served_model_name:
            return error(404, "Unknown model", "model_not_found")
        if service.active is not None:
            return error(429, "Engine is busy", "engine_busy")
        # No await between checking and claiming the slot on this event loop.
        task = asyncio.create_task(service.complete(request))
        service.active = task
        task.add_done_callback(service.release)
        return await asyncio.shield(task)

    return app
