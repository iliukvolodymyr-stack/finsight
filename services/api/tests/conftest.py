import pytest
from app.infra.db import engine
from app.infra.redis import close_redis
from app.main import app
from httpx import ASGITransport, AsyncClient


@pytest.fixture(autouse=True)
async def _dispose_db_engine():
    yield
    await close_redis()
    await engine.dispose()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
