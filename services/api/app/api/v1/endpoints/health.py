import asyncio

from app.infra.db import engine
from app.infra.redis import get_redis
from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    database: str
    redis: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check() -> ReadyResponse:
    db_status = "disconnected"
    redis_status = "disconnected"

    try:
        async with asyncio.timeout(1.0):
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                db_status = "connected"
    except Exception:
        db_status = "disconnected"

    try:
        async with asyncio.timeout(1.0):
            redis = get_redis()
            if await redis.ping():
                redis_status = "connected"
    except Exception:
        redis_status = "disconnected"

    overall = "ok" if (db_status == "connected" and redis_status == "connected") else "degraded"
    return ReadyResponse(
        status=overall,
        database=db_status,
        redis=redis_status,
    )
