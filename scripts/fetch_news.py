import asyncio

from app.infra.db import AsyncSessionLocal, engine
from app.services.rss import ingest_rss_news
from finsight_core.logging import setup_logging
from loguru import logger


async def run() -> None:
    setup_logging()
    async with AsyncSessionLocal() as session:
        summary = await ingest_rss_news(session)
        logger.info(
            "RSS fetch complete: status={}, sources={}/{}, written={}, rejected={}, links={}",
            summary.status,
            summary.sources_checked - summary.sources_failed,
            summary.sources_checked,
            summary.rows_written,
            summary.rows_rejected,
            summary.links_created,
        )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run())
