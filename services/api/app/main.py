from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.router import api_v1_router
from app.core.errors import (
    AppException,
    app_exception_handler,
    validation_exception_handler,
)
from app.infra.db import engine
from app.infra.redis import close_redis
from app.services.job_runner import reconcile_stale_jobs
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from finsight_core.config import get_settings
from finsight_core.logging import setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(settings.LOG_LEVEL)
    await reconcile_stale_jobs()
    yield
    await close_redis()
    await engine.dispose()


app = FastAPI(
    title="FinSight AI API",
    version="1.1.0",
    description="Intelligent financial asset analytics, risk assessment, and forecasting API",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=r"^https://[a-zA-Z0-9-]+\.app\.github\.dev$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, app_exception_handler)

app.include_router(health_router)
app.include_router(api_v1_router)
