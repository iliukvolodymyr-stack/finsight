import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from app.infra.models import CandleModel, IngestionRunModel, InstrumentModel
from app.services.ingestion import ingest_instrument_candles, refresh_data_gaps
from app.services.rss import ingest_rss_news
from app.worker.celery_app import celery_app
from finsight_core.config import get_settings
from finsight_ml.data import is_equity_exchange, is_quote_stale
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


def _run_with_fresh_session[T](fn: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """
    Execute an async DB coroutine inside a fresh event loop with a task-local NullPool engine.
    Avoids asyncpg 'Event loop is closed' errors across sequential Celery worker task invocations.
    """

    async def _runner() -> T:
        settings = get_settings()
        task_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
        session_factory = async_sessionmaker(
            bind=task_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        try:
            async with session_factory() as session:
                return await fn(session)
        finally:
            await task_engine.dispose()

    return asyncio.run(_runner())


async def _should_poll_instrument(
    session: AsyncSession,
    inst: InstrumentModel,
    now_utc: datetime,
) -> bool:
    if not is_equity_exchange(inst.exchange):
        return True

    latest_res = await session.execute(
        select(CandleModel.open_time)
        .where(
            CandleModel.instrument_id == inst.id,
            CandleModel.timeframe == "1h",
            CandleModel.is_closed.is_(True),
        )
        .order_by(CandleModel.open_time.desc())
        .limit(1)
    )
    latest_ts = latest_res.scalar_one_or_none()
    # Poll equities only if quote is stale (missing bars or active session needing update)
    return is_quote_stale(
        inst.exchange,
        latest_ts,
        now_utc,
        timeframe="1h",
        max_age_sec=3600,
        post_close_grace_sec=0,
    )


@celery_app.task(name="ingest.poll_closed_bars")
def poll_closed_bars() -> dict[str, Any]:
    """Poll latest closed 1h bars for active instruments (skips equities when market closed & fresh)."""

    async def _task(session: AsyncSession) -> dict[str, Any]:
        now_utc = datetime.now(UTC)
        instruments = list(
            (
                await session.execute(
                    select(InstrumentModel)
                    .where(InstrumentModel.is_active.is_(True))
                    .order_by(InstrumentModel.symbol.asc())
                )
            )
            .scalars()
            .all()
        )
        polled: list[str] = []
        skipped: list[str] = []
        errors: dict[str, str] = {}

        for inst in instruments:
            if not await _should_poll_instrument(session, inst, now_utc):
                skipped.append(inst.symbol)
                continue
            try:
                await ingest_instrument_candles(
                    session,
                    symbol=inst.symbol,
                    timeframe="1h",
                    days=2,
                    job_name="bar_poll",
                    check_gaps=False,
                )
                polled.append(inst.symbol)
            except Exception as exc:
                logger.warning("bar_poll failed for {}: {}", inst.symbol, exc)
                errors[inst.symbol] = str(exc)[:300]

        return {"polled": polled, "skipped": skipped, "errors": errors}

    return _run_with_fresh_session(_task)


@celery_app.task(name="ingest.fetch_rss_news")
def fetch_rss_news() -> dict[str, Any]:
    """Fetch all active RSS feeds and record a coverage heartbeat in ingestion_runs."""

    async def _task(session: AsyncSession) -> dict[str, Any]:
        summary = await ingest_rss_news(session)
        return {
            "sources_checked": summary.sources_checked,
            "sources_failed": summary.sources_failed,
            "rows_written": summary.rows_written,
            "rows_rejected": summary.rows_rejected,
            "links_created": summary.links_created,
            "status": summary.status,
        }

    return _run_with_fresh_session(_task)


@celery_app.task(name="ingest.check_data_gaps")
def check_data_gaps() -> dict[str, Any]:
    """Run hourly calendar-aware gap check across all active instruments (1h and 1d)."""

    async def _task(session: AsyncSession) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        instruments = list(
            (
                await session.execute(
                    select(InstrumentModel)
                    .where(InstrumentModel.is_active.is_(True))
                    .order_by(InstrumentModel.symbol.asc())
                )
            )
            .scalars()
            .all()
        )
        total_gaps = 0
        details: dict[str, dict[str, Any]] = {}

        for inst in instruments:
            for tf in ("1h", "1d"):
                report = await refresh_data_gaps(session, inst, tf)
                total_gaps += len(report.gaps)
                details[f"{inst.symbol}:{tf}"] = {
                    "completeness_pct": report.completeness_pct,
                    "gaps": len(report.gaps),
                }

        session.add(
            IngestionRunModel(
                job_name="gap_check",
                source="calendar",
                instrument_id=None,
                timeframe=None,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                rows_written=0,
                rows_rejected=0,
                gaps_found=total_gaps,
                status="ok",
            )
        )
        await session.commit()
        return {"total_gaps": total_gaps, "details": details}

    return _run_with_fresh_session(_task)
