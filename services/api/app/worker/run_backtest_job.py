import argparse
import asyncio
import uuid

from app.services.backtest_runner import execute_backtest_job
from finsight_core.config import get_settings
from finsight_core.logging import setup_logging
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


async def _run(backtest_id: uuid.UUID, job_id: uuid.UUID) -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    local_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    session_factory = async_sessionmaker(local_engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with session_factory() as session:
            await execute_backtest_job(session, backtest_id, job_id)
    finally:
        await local_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a queued FinSight backtest job")
    parser.add_argument("--backtest-id", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    asyncio.run(_run(uuid.UUID(args.backtest_id), uuid.UUID(args.job_id)))


if __name__ == "__main__":
    main()
