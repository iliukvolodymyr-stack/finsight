import argparse
import asyncio

from app.infra.db import AsyncSessionLocal, engine
from app.infra.models import InstrumentModel
from app.services.ingestion import ingest_instrument_candles
from finsight_core.logging import setup_logging
from loguru import logger
from sqlalchemy import select


async def run_all(
    symbols: list[str] | None,
    timeframes: list[str],
    days: int,
) -> None:
    setup_logging()
    async with AsyncSessionLocal() as session:
        if symbols:
            target_symbols = [s.upper() for s in symbols]
        else:
            res = await session.execute(
                select(InstrumentModel.symbol)
                .where(InstrumentModel.is_active.is_(True))
                .order_by(InstrumentModel.symbol.asc())
            )
            target_symbols = list(res.scalars().all())

        if not target_symbols:
            logger.error("No active instruments found. Run scripts/seed.py first.")
            return

        logger.info(
            "Starting backfill for symbols={} timeframes={} days={}",
            target_symbols,
            timeframes,
            days,
        )
        for sym in target_symbols:
            for tf in timeframes:
                summary = await ingest_instrument_candles(
                    session,
                    symbol=sym,
                    timeframe=tf,
                    days=days,
                    job_name="backfill",
                )
                status_mark = "OK" if summary.completeness_pct >= 98.0 else "WARN(<98%)"
                logger.info(
                    "[{}] {} {}: written={}, rejected={}, completeness={:.2f}% ({}/{}), gaps={}",
                    status_mark,
                    summary.symbol,
                    summary.timeframe,
                    summary.rows_written,
                    summary.rows_rejected,
                    summary.completeness_pct,
                    summary.actual_bars,
                    summary.expected_bars,
                    summary.gaps_found,
                )

    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical OHLCV candles into TimescaleDB"
    )
    parser.add_argument("--symbols", nargs="*", help="Symbols to backfill (default: all active)")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "1d"],
        help="Timeframes to backfill (default: 1h 1d)",
    )
    parser.add_argument("--days", type=int, default=730, help="Lookback days (default: 730)")
    args = parser.parse_args()
    asyncio.run(run_all(args.symbols, args.timeframes, args.days))


if __name__ == "__main__":
    main()
