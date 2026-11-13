import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from nanovllm.server.engine import InvalidRequest
from nanovllm.server.schema import ChatRequest

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

    async def run(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    def initialize(self):
        self.engine = self.factory(self.config)
        self.engine.start()

    async def start(self):
        await self.run(self.initialize)
        self.ready = True

    async def complete(self, request):
        try:
            result = await self.run(self.engine.complete, request)
            return {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion", "created": int(time.time()),
                "model": self.config.served_model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": result.finish_reason}],
                "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                          "total_tokens": result.prompt_tokens + result.completion_tokens},
            }
        except InvalidRequest as exc:
            return error(400, str(exc), "invalid_request")
        except Exception:
            logger.exception("Engine generation failed")
            self.failed = True
            self.ready = False
            self.on_fatal()
            return error(500, "Engine generation failed; restart the server", "engine_failed")

    def release(self, task):
        # The task, not the HTTP connection, owns the slot until computation ends.
        self.active = None
        if not task.cancelled():
            task.exception()

    async def close(self):
        self.ready = False
        try:
            if self.active is not None:
                await asyncio.shield(self.active)
            if self.engine is not None:
                await self.run(self.engine.close)
                self.engine = None
        finally:
            self.executor.shutdown(wait=True)


def create_app(config, engine_factory):
    @asynccontextmanager
    async def lifespan(app):
        try:
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

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest):
        if not service.ready:
            return error(503, "Engine not ready", "engine_not_ready")
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
