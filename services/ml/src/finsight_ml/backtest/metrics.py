import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 8760
) -> float:
    """Annualized Sharpe ratio (for 1h bars: 8760 periods/year, for 1d: 365 or 252)."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = excess_returns.std()
    if std <= 1e-9 or np.isnan(std):
        return 0.0
    return float((excess_returns.mean() / std) * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())
    return abs(max_dd) if not np.isnan(max_dd) else 0.0


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 8760
) -> float:
    """Annualized Sortino ratio."""
    excess_returns = returns - (risk_free_rate / periods_per_year)
    negative_returns = np.minimum(0.0, excess_returns)
    downside_dev = float(np.sqrt(np.mean(negative_returns**2)))
    if downside_dev <= 1e-9 or np.isnan(downside_dev):
        return 0.0
    return float((excess_returns.mean() / downside_dev) * np.sqrt(periods_per_year))
