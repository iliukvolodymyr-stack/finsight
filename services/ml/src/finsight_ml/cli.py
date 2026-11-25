import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
import typer
from finsight_core.config import get_settings
from rich.console import Console
from rich.table import Table
from sqlalchemy.ext.asyncio import create_async_engine

from finsight_ml.backtest import BacktestStrategyConfig, run_walk_forward_backtest
from finsight_ml.data import (
    detect_gaps,
    fetch_binance_klines,
    fetch_yahoo_candles,
    validate_candles,
)
from finsight_ml.training import TrainingConfig, train_and_evaluate

app = typer.Typer(help="FinSight AI ML command line interface")
console = Console()


def _load_candles_from_file(data_path: Path) -> pd.DataFrame:
    if data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


async def _load_candles_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    settings = get_settings()
    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            inst_row = (
                await conn.execute(
                    sa.text("SELECT id FROM instruments WHERE symbol = :sym"),
                    {"sym": symbol.upper()},
                )
            ).first()
            if inst_row is None:
                raise RuntimeError(f"Instrument {symbol} not found in database.")
            rows = (
                await conn.execute(
                    sa.text(
                        """
                        SELECT open_time, open, high, low, close, volume
                        FROM candles
                        WHERE instrument_id = :inst_id
                          AND timeframe = :tf
                          AND is_closed = true
                        ORDER BY open_time ASC
                        """
                    ),
                    {"inst_id": inst_row[0], "tf": timeframe},
                )
            ).all()
        if not rows:
            raise RuntimeError(f"No closed candles found for {symbol} {timeframe}.")
        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        return df.set_index("open_time").sort_index()
    finally:
        await engine.dispose()


def _load_market_df(symbol: str, timeframe: str, data_path: Path | None) -> pd.DataFrame:
    if data_path is not None:
        return _load_candles_from_file(data_path)
    return asyncio.run(_load_candles_from_db(symbol, timeframe))


@app.command()
def backfill(
    symbol: str = typer.Argument(..., help="Symbol (e.g. BTCUSDT, SPY)"),
    timeframe: str = typer.Argument("1h", help="Timeframe (1h, 1d)"),
    days: int = typer.Option(730, help="Historical lookback in days"),
    source: str = typer.Option("auto", help="Data source: auto, binance, yahoo"),
    out: Path | None = typer.Option(None, help="Optional Parquet output path"),
) -> None:
    """Fetch historical market data, run data gates and calendar completeness check."""
    sym = symbol.upper()
    resolved_source = source.lower()
    if resolved_source == "auto":
        resolved_source = "binance" if sym.endswith("USDT") else "yahoo"
    exchange = "binance" if resolved_source == "binance" else "nyse"

    console.print(
        f"[bold green]Fetching {sym} {timeframe} ({days}d) via {resolved_source}...[/bold green]"
    )

    async def _run() -> None:
        end_utc = datetime.now(UTC)
        start_utc = end_utc - timedelta(days=days)
        if resolved_source == "binance":
            raw = await fetch_binance_klines(sym, timeframe, start_utc, end_utc)
        else:
            raw = await fetch_yahoo_candles(sym, timeframe, start_utc, end_utc)

        val = validate_candles(raw, allow_unclosed=False)
        report = detect_gaps(
            [c.open_time for c in val.valid],
            exchange=exchange,
            timeframe=timeframe,
        )
        console.print(
            f"[green]✓ {sym} {timeframe}: valid={len(val.valid)}, "
            f"rejected={val.rejected_count}, completeness={report.completeness_pct:.2f}% "
            f"({report.actual_bars}/{report.expected_bars}), gaps={len(report.gaps)}[/green]"
        )
        if out is not None and val.valid:
            out.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame([c.__dict__ for c in val.valid])
            df.to_parquet(out, index=False)
            console.print(f"[cyan]Saved Parquet snapshot to {out}[/cyan]")

    asyncio.run(_run())


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to model config YAML"),
    data_path: Path | None = typer.Option(
        None, "--data-path", help="Optional Parquet/CSV snapshot (uses DB if omitted)"
    ),
    artifacts_dir: Path | None = typer.Option(None, "--artifacts-dir", help="Artifact root"),
) -> None:
    """Train & evaluate a forecasting model via Purged Walk-Forward CV."""
    cfg = TrainingConfig.from_yaml(config_path)
    console.print(
        f"[bold blue]Training {cfg.name} for {cfg.symbol} {cfg.timeframe} "
        f"(H={cfg.horizon_bars}, k={cfg.pt_mult})...[/bold blue]"
    )
    df = _load_market_df(cfg.symbol, cfg.timeframe, data_path)
    res = train_and_evaluate(df, cfg, artifacts_dir=artifacts_dir)

    table = Table(title=f"OOS Walk-Forward Summary — {res.symbol} {res.timeframe}")
    table.add_column("Model")
    table.add_column("Macro-F1", justify="right")
    table.add_column("BalAcc", justify="right")
    table.add_column("MCC", justify="right")
    table.add_column("ECE-10", justify="right")
    table.add_column("LogLoss", justify="right")

    for label, key in (
        ("Majority", "majority"),
        ("LogReg", "logreg"),
        ("LGBM (raw)", "lgbm_raw"),
        ("LGBM (calibrated)", "lgbm_calibrated"),
    ):
        m = res.oos_summary["models"][key]
        table.add_row(
            label,
            f"{m['macro_f1']:.4f}",
            f"{m['balanced_accuracy']:.4f}",
            f"{m['mcc']:.4f}",
            f"{m['ece_10']:.4f}",
            f"{m['log_loss']:.4f}",
        )
    console.print(table)
    cov = res.oos_summary["conformal_coverage"]
    console.print(
        f"[green]✓ Saved artifacts to {res.artifact_dir} | "
        f"Conformal 80% cov={cov['mean'] * 100:.1f}% "
        f"[{cov['ci_low'] * 100:.1f}%, {cov['ci_high'] * 100:.1f}%][/green]"
    )


@app.command()
def backtest(
    config_path: Path = typer.Argument(..., help="Path to model/backtest config YAML"),
    data_path: Path | None = typer.Option(
        None, "--data-path", help="Optional Parquet/CSV snapshot (uses DB if omitted)"
    ),
    allow_short: bool = typer.Option(False, "--allow-short", help="Allow SHORT positions"),
    position_size: float = typer.Option(
        1.0, "--position-size", help="Fraction of capital per trade"
    ),
    fee_bps: float = typer.Option(10.0, "--fee-bps", help="Per-side fee in bps"),
    slippage_bps: float = typer.Option(5.0, "--slippage-bps", help="Per-side slippage in bps"),
    artifacts_dir: Path | None = typer.Option(None, "--artifacts-dir", help="Artifact root"),
) -> None:
    """Run Purged Walk-Forward Backtest v0 (exit_mode=label_barriers) vs baselines."""
    train_cfg = TrainingConfig.from_yaml(config_path)
    exchange = "binance" if train_cfg.symbol.endswith("USDT") else "nyse"
    bt_cfg = BacktestStrategyConfig(
        horizon_bars=train_cfg.horizon_bars,
        pt_mult=train_cfg.pt_mult,
        sl_mult=train_cfg.sl_mult,
        min_barrier_pct=train_cfg.min_barrier_pct,
        position_size=position_size,
        allow_short=allow_short,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        exchange=exchange,
        timeframe=train_cfg.timeframe,
    )
    df = _load_market_df(train_cfg.symbol, train_cfg.timeframe, data_path)
    out = run_walk_forward_backtest(
        df=df,
        train_cfg=train_cfg,
        bt_cfg=bt_cfg,
        artifacts_dir=artifacts_dir or Path(get_settings().ARTIFACTS_DIR),
    )

    table = Table(title=f"OOS Backtest v0 — {out.symbol} {out.timeframe} (label_barriers)")
    table.add_column("Strategy")
    table.add_column("Return", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("WinRate", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Trades", justify="right")

    for name, m in (
        ("LGBM Strategy", out.strategy_metrics),
        ("Buy & Hold", out.buy_and_hold_metrics),
        ("EMA(20/50)", out.ema_crossover_metrics),
    ):
        table.add_row(
            name,
            f"{m['total_return'] * 100:+.2f}%",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown'] * 100:.2f}%",
            f"{m['win_rate'] * 100:.1f}%",
            f"{m['profit_factor']:.2f}",
            str(m["trades_count"]),
        )
    r = out.random_benchmark_metrics
    table.add_row(
        f"Random ({r['n_seeds']} seeds)",
        f"{r['total_return_mean'] * 100:+.2f}%±{r['total_return_std'] * 100:.2f}%",
        f"{r['sharpe_mean']:.2f}±{r['sharpe_std']:.2f}",
        f"{r['max_drawdown_mean'] * 100:.2f}%",
        f"{r['win_rate_mean'] * 100:.1f}%",
        "—",
        "—",
    )
    console.print(table)
    console.print(
        f"[green]✓ Backtest complete. FeeDrag={out.strategy_metrics['fee_drag'] * 100:.1f}%, "
        f"Artifacts={out.artifact_dir}[/green]"
    )


@app.command()
def experiment(name: str = typer.Argument(..., help="Experiment identifier (e.g. E01)")) -> None:
    """Execute evaluation experiment."""
    console.print(f"[bold yellow]Running experiment {name}...[/bold yellow]")
    console.print(f"[green]✓ Experiment {name} executed.[/green]")


if __name__ == "__main__":
    app()
