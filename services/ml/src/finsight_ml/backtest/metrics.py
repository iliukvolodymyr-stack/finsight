from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def periods_per_year_for(exchange: str = "binance", timeframe: str = "1h") -> int:
    """
    Exchange- and timeframe-aware annualization factor:
    - Crypto (24/7): 8760 (1h), 365 (1d)
    - NYSE (252 sessions * 7 hourly bars): 1764 (1h), 252 (1d)
    """
    ex = exchange.lower()
    tf = timeframe.lower()
    if ex in {"nyse", "nasdaq", "xnys"}:
        return 1764 if tf == "1h" else 252
    return 8760 if tf == "1h" else 365


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = float(excess_returns.std())
    if std <= 1e-12 or np.isnan(std):
        return 0.0
    return float((excess_returns.mean() / std) * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown as a positive fraction in [0, 1]."""
    if len(equity_curve) == 0:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max.replace(0.0, np.nan)
    max_dd = float(drawdown.min())
    return abs(max_dd) if not np.isnan(max_dd) else 0.0


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """Annualized Sortino ratio using RMS downside deviation over all N observations."""
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    negative_returns = np.minimum(0.0, excess_returns)
    downside_dev = float(np.sqrt(np.mean(negative_returns**2)))
    if downside_dev <= 1e-12 or np.isnan(downside_dev):
        return 0.0
    return float((excess_returns.mean() / downside_dev) * np.sqrt(periods_per_year))


def summarize_backtest_metrics(
    equity_curve: pd.Series,
    trade_pnls: Sequence[float],
    trade_fees: Sequence[float],
    total_traded_notional: float,
    initial_capital: float,
    periods_per_year: int = 8760,
    benchmark_equity: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Compute complete post-cost portfolio and trade-level metrics for Backtest v0.
    Includes `sharpe_warning` guard (Sharpe > 3.0 triggers leakage audit flag).
    """
    eq = equity_curve.astype("float64")
    n_bars = len(eq)
    returns = eq.pct_change().fillna(0.0)
    if n_bars > 0 and initial_capital > 0:
        returns.iloc[0] = (float(eq.iloc[0]) - initial_capital) / initial_capital

    total_return = (
        (float(eq.iloc[-1]) / initial_capital - 1.0) if n_bars > 0 and initial_capital > 0 else 0.0
    )
    years = n_bars / float(periods_per_year) if periods_per_year > 0 else 0.0
    final_mult = 1.0 + total_return
    cagr = (
        float(final_mult ** (1.0 / years) - 1.0)
        if years > 0.0 and final_mult > 0.0
        else (-1.0 if final_mult <= 0.0 else 0.0)
    )

    sharpe = calculate_sharpe_ratio(returns, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(returns, periods_per_year=periods_per_year)
    mdd = calculate_max_drawdown(eq)
    calmar = float(cagr / mdd) if mdd > 1e-9 else 0.0

    pnls = np.asarray(trade_pnls, dtype=np.float64)
    fees = np.asarray(trade_fees, dtype=np.float64)
    trades_count = int(len(pnls))

    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]
    net_profit = float(np.sum(pnls)) if trades_count > 0 else 0.0
    total_fees = float(np.sum(fees)) if len(fees) > 0 else 0.0
    gross_pnls = pnls + fees if len(fees) == trades_count else pnls

    gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = (
        float(gross_wins / gross_losses)
        if gross_losses > 1e-12
        else (999.0 if gross_wins > 0.0 else 0.0)
    )
    win_rate = float(len(wins) / trades_count) if trades_count > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    gross_positive_sum = float(np.sum(gross_pnls[gross_pnls > 0.0])) if trades_count > 0 else 0.0
    fee_drag = (
        float(total_fees / gross_positive_sum)
        if gross_positive_sum > 1e-12
        else (1.0 if total_fees > 0.0 else 0.0)
    )
    turnover = float(total_traded_notional / initial_capital) if initial_capital > 0 else 0.0

    alpha_vs_bh: float | None = None
    bh_return: float | None = None
    bh_sharpe: float | None = None
    if benchmark_equity is not None and len(benchmark_equity) == n_bars and initial_capital > 0:
        bh_eq = benchmark_equity.astype("float64")
        bh_ret_series = bh_eq.pct_change().fillna(0.0)
        bh_ret_series.iloc[0] = (float(bh_eq.iloc[0]) - initial_capital) / initial_capital
        bh_return = float(bh_eq.iloc[-1]) / initial_capital - 1.0
        bh_sharpe = calculate_sharpe_ratio(bh_ret_series, periods_per_year=periods_per_year)
        alpha_vs_bh = float(total_return - bh_return)

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(calmar),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "net_profit": float(net_profit),
        "total_fees": float(total_fees),
        "fee_drag": float(fee_drag),
        "turnover": float(turnover),
        "trades_count": trades_count,
        "periods_per_year": int(periods_per_year),
        "benchmark_return": bh_return,
        "benchmark_sharpe": bh_sharpe,
        "alpha_vs_benchmark": alpha_vs_bh,
        "sharpe_warning": sharpe > 3.0,
    }
