"""Empirical Triple Barrier class-balance diagnostic (training window only).

Selects barrier multiplier `k` (pt_mult = sl_mult = k) strictly by class balance
on the initial 12-month training window — NEVER by profitability or test-fold performance.
"""

import asyncio
from datetime import timedelta

import pandas as pd
from app.infra.db import AsyncSessionLocal, engine
from app.infra.models import CandleModel, InstrumentModel
from finsight_ml.labels.triple_barrier import (
    compute_triple_barrier_dataset,
    summarize_label_distribution,
)
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

console = Console()


async def _load_candles(symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    async with AsyncSessionLocal() as session:
        inst = (
            await session.execute(select(InstrumentModel).where(InstrumentModel.symbol == symbol))
        ).scalar_one()
        rows = (
            await session.execute(
                select(
                    CandleModel.open_time,
                    CandleModel.open,
                    CandleModel.high,
                    CandleModel.low,
                    CandleModel.close,
                    CandleModel.volume,
                )
                .where(
                    CandleModel.instrument_id == inst.id,
                    CandleModel.timeframe == timeframe,
                    CandleModel.is_closed.is_(True),
                )
                .order_by(CandleModel.open_time.asc())
            )
        ).all()

    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df = df.set_index("open_time")
    return df


async def main() -> None:
    try:
        for symbol in ("BTCUSDT", "ETHUSDT", "SPY"):
            df_all = await _load_candles(symbol, "1h")
            if df_all.empty:
                console.print(f"[red]No 1h candles found for {symbol}[/red]")
                continue

            # Restrict strictly to the initial 12-month training window (no test leakage!)
            train_end = df_all.index.min() + timedelta(days=365)
            df_train = df_all[df_all.index <= train_end]

            table = Table(
                title=f"{symbol} 1h — Train-Only Label Balance ({len(df_train)} bars, H=24)"
            )
            table.add_column("k", justify="right")
            table.add_column("Valid", justify="right")
            table.add_column("BUY %", justify="right")
            table.add_column("HOLD %", justify="right")
            table.add_column("SELL %", justify="right")
            table.add_column("Mean Uniq", justify="right")
            table.add_column("Warning", justify="center")

            for k in (0.75, 1.0, 1.25, 1.5, 2.0):
                tb = compute_triple_barrier_dataset(
                    df_train,
                    horizon_bars=24,
                    pt_mult=k,
                    sl_mult=k,
                    min_barrier_pct=0.003,
                    entry_on_next_open=True,
                )
                rep = summarize_label_distribution(tb)
                table.add_row(
                    f"{k:.2f}",
                    str(rep.n_valid),
                    f"{rep.buy_share * 100:.1f}%",
                    f"{rep.hold_share * 100:.1f}%",
                    f"{rep.sell_share * 100:.1f}%",
                    f"{rep.mean_uniqueness:.3f}",
                    "HOLD>70%" if rep.hold_warning else "OK",
                )
            console.print(table)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
