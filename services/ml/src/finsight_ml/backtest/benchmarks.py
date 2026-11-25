from typing import Any

import numpy as np
import pandas as pd

from finsight_ml.backtest.engine import BacktestEngine, BacktestResult
from finsight_ml.backtest.metrics import periods_per_year_for, summarize_backtest_metrics
from finsight_ml.features.indicators import calculate_ema


def run_buy_and_hold(
    df: pd.DataFrame,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    initial_capital: float = 10000.0,
    exchange: str = "binance",
    timeframe: str = "1h",
) -> BacktestResult:
    """Buy & Hold benchmark entering on bar 1 open (or bar 0 if length < 2) and holding to close[-1]."""
    if df.empty:
        empty_eq = pd.Series([], index=df.index, dtype="float64")
        return BacktestResult(0.0, 0.0, 0.0, 0, 0.0, empty_eq, [], {})

    cost_pct = (fee_bps + slippage_bps) / 10000.0
    entry_idx = 1 if len(df) > 1 else 0
    entry_price = float(df["open"].iloc[entry_idx])
    close_arr = df["close"].astype("float64").to_numpy()
    equity_arr = np.full(len(df), float(initial_capital), dtype=np.float64)

    qty = float(initial_capital) / entry_price
    entry_fee = float(initial_capital) * cost_pct
    for j in range(entry_idx, len(df)):
        equity_arr[j] = max(0.0, (qty * float(close_arr[j])) - entry_fee)
    exit_notional = qty * float(close_arr[-1])
    exit_fee = exit_notional * cost_pct
    equity_arr[-1] = max(0.0, equity_arr[-1] - exit_fee)

    eq = pd.Series(equity_arr, index=df.index, dtype="float64")
    net_pnl = float(eq.iloc[-1] - initial_capital)
    ppy = periods_per_year_for(exchange=exchange, timeframe=timeframe)
    metrics = summarize_backtest_metrics(
        equity_curve=eq,
        trade_pnls=[net_pnl],
        trade_fees=[entry_fee + exit_fee],
        total_traded_notional=float(initial_capital) + exit_notional,
        initial_capital=initial_capital,
        periods_per_year=ppy,
    )
    return BacktestResult(
        sharpe=float(metrics["sharpe"]),
        max_drawdown=float(metrics["max_drawdown"]),
        total_return=float(metrics["total_return"]),
        trades_count=1,
        win_rate=1.0 if net_pnl > 0.0 else 0.0,
        equity_curve=eq,
        metrics=metrics,
    )


def run_ema_crossover_benchmark(
    df: pd.DataFrame,
    fast_span: int = 20,
    slow_span: int = 50,
    horizon_bars: int = 24,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    position_size: float = 1.0,
    allow_short: bool = False,
    initial_capital: float = 10000.0,
    exchange: str = "binance",
    timeframe: str = "1h",
    benchmark_equity: pd.Series | None = None,
) -> BacktestResult:
    """Classic EMA(20)/EMA(50) crossover benchmark evaluated on exact same barrier rules & costs."""
    close = df["close"].astype("float64")
    ema_fast = calculate_ema(close, fast_span)
    ema_slow = calculate_ema(close, slow_span)

    signals = pd.Series(0, index=df.index, dtype="int64")
    signals[ema_fast > ema_slow] = 1
    if allow_short:
        signals[ema_fast < ema_slow] = -1
    # Respect slow EMA warmup before issuing trades
    signals.iloc[:slow_span] = 0

    engine = BacktestEngine(fee_bps=fee_bps, slippage_bps=slippage_bps)
    return engine.run_barriers(
        df=df,
        signals=signals,
        horizon_bars=horizon_bars,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        position_size=position_size,
        allow_short=allow_short,
        initial_capital=initial_capital,
        exchange=exchange,
        timeframe=timeframe,
        benchmark_equity=benchmark_equity,
    )


def run_random_benchmark(
    df: pd.DataFrame,
    buy_rate: float = 0.05,
    n_seeds: int = 50,
    base_seed: int = 42,
    horizon_bars: int = 24,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    position_size: float = 1.0,
    allow_short: bool = False,
    initial_capital: float = 10000.0,
    exchange: str = "binance",
    timeframe: str = "1h",
) -> dict[str, Any]:
    """
    Monte-Carlo random-signal baseline matched to strategy active signal rate across `n_seeds`.
    """
    p_active = max(0.005, min(0.95, float(buy_rate)))
    engine = BacktestEngine(fee_bps=fee_bps, slippage_bps=slippage_bps)

    sharpes: list[float] = []
    returns: list[float] = []
    mdds: list[float] = []
    win_rates: list[float] = []

    for seed in range(base_seed, base_seed + n_seeds):
        rng = np.random.default_rng(seed)
        draws = rng.random(len(df))
        if allow_short:
            sig_vals = np.where(draws < p_active / 2.0, 1, np.where(draws < p_active, -1, 0))
        else:
            sig_vals = np.where(draws < p_active, 1, 0)
        sigs = pd.Series(sig_vals, index=df.index, dtype="int64")

        res = engine.run_barriers(
            df=df,
            signals=sigs,
            horizon_bars=horizon_bars,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            position_size=position_size,
            allow_short=allow_short,
            initial_capital=initial_capital,
            exchange=exchange,
            timeframe=timeframe,
        )
        sharpes.append(res.sharpe)
        returns.append(res.total_return)
        mdds.append(res.max_drawdown)
        win_rates.append(res.win_rate)

    return {
        "n_seeds": n_seeds,
        "buy_rate": p_active,
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_std": float(np.std(sharpes)),
        "total_return_mean": float(np.mean(returns)),
        "total_return_std": float(np.std(returns)),
        "max_drawdown_mean": float(np.mean(mdds)),
        "win_rate_mean": float(np.mean(win_rates)),
    }
