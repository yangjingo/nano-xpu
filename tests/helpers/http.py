"""Shared CPU HTTP request and client fixtures."""
from contextlib import asynccontextmanager
import httpx
from nanoxpu.server.app import create_app
from nanoxpu.server.config import ServerConfig
from tests.helpers.support import TestEngine

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
