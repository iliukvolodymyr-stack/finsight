from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from app.infra.models import CandleModel, DataGapModel, IngestionRunModel, InstrumentModel
from finsight_ml.data import (
    QualityReport,
    RawCandle,
    detect_gaps,
    fetch_binance_klines,
    fetch_yahoo_candles,
    validate_candles,
)
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

UPSERT_BATCH_SIZE = 1000


@dataclass
class BackfillSummary:
    symbol: str
    timeframe: str
    source: str
    rows_written: int
    rows_rejected: int
    unclosed_skipped: int
    expected_bars: int
    actual_bars: int
    completeness_pct: float
    gaps_found: int
    status: str


def _resolve_source_and_symbol(inst: InstrumentModel) -> tuple[str, str]:
    aliases = inst.metadata_.get("aliases", {}) if isinstance(inst.metadata_, dict) else {}
    exch = inst.exchange.lower()
    if exch == "binance":
        return "binance", str(aliases.get("binance", inst.symbol))
    return "yahoo", str(aliases.get("yahoo", inst.symbol))


async def upsert_candles(
    session: AsyncSession,
    instrument_id: UUID,
    timeframe: str,
    candles: list[RawCandle],
) -> int:
    """Idempotent chunked PostgreSQL upsert into candles hypertable."""
    if not candles:
        return 0

    now_utc = datetime.now(UTC)
    total_written = 0

    for offset in range(0, len(candles), UPSERT_BATCH_SIZE):
        chunk = candles[offset : offset + UPSERT_BATCH_SIZE]
        rows: list[dict[str, Any]] = [
            {
                "instrument_id": instrument_id,
                "timeframe": timeframe,
                "open_time": c.open_time,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "adjusted_close": c.adjusted_close,
                "volume": c.volume,
                "quote_volume": c.quote_volume,
                "trades_count": c.trades_count,
                "vwap": None,
                "is_closed": c.is_closed,
                "source": c.source,
                "ingested_at": now_utc,
            }
            for c in chunk
        ]

        stmt = pg_insert(CandleModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="pk_candles",
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "adjusted_close": stmt.excluded.adjusted_close,
                "volume": stmt.excluded.volume,
                "quote_volume": stmt.excluded.quote_volume,
                "trades_count": stmt.excluded.trades_count,
                "is_closed": stmt.excluded.is_closed,
                "source": stmt.excluded.source,
                "ingested_at": stmt.excluded.ingested_at,
            },
        )
        await session.execute(stmt)
        total_written += len(chunk)

    return total_written


async def refresh_data_gaps(
    session: AsyncSession,
    instrument: InstrumentModel,
    timeframe: str,
) -> QualityReport:
    """
    Evaluate calendar-aware completeness over stored closed candles
    and synchronize open gaps in data_gaps.
    """
    res = await session.execute(
        select(CandleModel.open_time)
        .where(
            CandleModel.instrument_id == instrument.id,
            CandleModel.timeframe == timeframe,
            CandleModel.is_closed.is_(True),
        )
        .order_by(CandleModel.open_time.asc())
    )
    observed = list(res.scalars().all())
    report = detect_gaps(observed, exchange=instrument.exchange, timeframe=timeframe)

    now_utc = datetime.now(UTC)
    # Mark previously open gaps resolved, then record current gaps
    await session.execute(
        update(DataGapModel)
        .where(
            DataGapModel.instrument_id == instrument.id,
            DataGapModel.timeframe == timeframe,
            DataGapModel.resolved_at.is_(None),
        )
        .values(resolved_at=now_utc)
    )

    for g in report.gaps:
        session.add(
            DataGapModel(
                instrument_id=instrument.id,
                timeframe=timeframe,
                gap_start=g.gap_start,
                gap_end=g.gap_end,
                bars_missing=g.bars_missing,
                detected_at=now_utc,
                resolved_at=None,
            )
        )

    return report


async def ingest_instrument_candles(
    session: AsyncSession,
    symbol: str,
    timeframe: str,
    *,
    days: int = 730,
    job_name: str = "backfill",
) -> BackfillSummary:
    """
    Fetch, validate (data gates), upsert candles, evaluate calendar gaps,
    and persist an ingestion_runs audit record (including rows_rejected).
    """
    inst_res = await session.execute(
        select(InstrumentModel).where(InstrumentModel.symbol == symbol.upper())
    )
    instrument = inst_res.scalar_one_or_none()
    if instrument is None:
        raise ValueError(f"Instrument {symbol} not found in database. Run seed.py first.")

    source, ext_symbol = _resolve_source_and_symbol(instrument)
    started_at = datetime.now(UTC)

    run_record = IngestionRunModel(
        job_name=job_name,
        source=source,
        instrument_id=instrument.id,
        timeframe=timeframe,
        started_at=started_at,
        status="running",
    )
    session.add(run_record)
    await session.flush()

    try:
        end_utc = datetime.now(UTC)
        start_utc = end_utc - timedelta(days=days)

        if source == "binance":
            raw = await fetch_binance_klines(ext_symbol, timeframe, start_utc, end_utc)
        else:
            raw = await fetch_yahoo_candles(ext_symbol, timeframe, start_utc, end_utc)

        val = validate_candles(raw, allow_unclosed=False)
        if val.rejected_count > 0:
            logger.warning(
                "Data gates rejected {} bars for {} {} (sample: {})",
                val.rejected_count,
                instrument.symbol,
                timeframe,
                val.rejection_reasons[:3],
            )

        rows_written = await upsert_candles(session, instrument.id, timeframe, val.valid)
        report = await refresh_data_gaps(session, instrument, timeframe)

        run_record.finished_at = datetime.now(UTC)
        run_record.rows_written = rows_written
        run_record.rows_rejected = val.rejected_count
        run_record.gaps_found = len(report.gaps)
        run_record.status = "ok" if val.rejected_count == 0 else "partial"
        await session.commit()

        logger.info(
            "{} {} {} [{}]: written={}, rejected={}, completeness={:.2f}% ({}/{}), gaps={}",
            job_name,
            instrument.symbol,
            timeframe,
            source,
            rows_written,
            val.rejected_count,
            report.completeness_pct,
            report.actual_bars,
            report.expected_bars,
            len(report.gaps),
        )

        return BackfillSummary(
            symbol=instrument.symbol,
            timeframe=timeframe,
            source=source,
            rows_written=rows_written,
            rows_rejected=val.rejected_count,
            unclosed_skipped=val.unclosed_skipped,
            expected_bars=report.expected_bars,
            actual_bars=report.actual_bars,
            completeness_pct=report.completeness_pct,
            gaps_found=len(report.gaps),
            status=run_record.status,
        )
    except Exception as exc:
        run_record.finished_at = datetime.now(UTC)
        run_record.status = "failed"
        run_record.error = str(exc)[:1000]
        await session.commit()
        raise
