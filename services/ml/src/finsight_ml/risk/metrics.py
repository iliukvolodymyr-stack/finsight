from typing import Any

import numpy as np
import pandas as pd

from finsight_ml.backtest.metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


def _normalize_daily_close(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64")
    idx = pd.DatetimeIndex(series.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    normalized = pd.Series(series.astype("float64").to_numpy(), index=idx.normalize())
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    return normalized.dropna()


def align_portfolio_daily_returns(
    close_series_by_symbol: dict[str, pd.Series],
    weights_by_symbol: dict[str, float],
    window_days: int = 90,
) -> pd.Series:
    """
    Build weighted historical daily portfolio simple returns over `window_days`:
    - Normalizes each asset's `1d` close history by UTC date.
    - Forward-fills non-trading days (e.g. NYSE weekends/holidays) across the common
      date range so mixed Crypto + ETF portfolios share an exact daily MTM grid.
    """
    active_syms = [
        s for s, w in weights_by_symbol.items() if w > 1e-12 and s in close_series_by_symbol
    ]
    if not active_syms:
        return pd.Series(dtype="float64")

    normalized_cols = {
        sym: _normalize_daily_close(close_series_by_symbol[sym]) for sym in active_syms
    }
    price_df = pd.DataFrame(normalized_cols).sort_index()
    if price_df.empty:
        return pd.Series(dtype="float64")

    price_df = price_df.ffill().dropna(how="any")
    if len(price_df) < 2:
        return pd.Series(dtype="float64")

    rets_df = price_df.pct_change().dropna(how="any")
    if window_days > 0 and len(rets_df) > window_days:
        rets_df = rets_df.iloc[-window_days:]

    w_raw = np.array([float(weights_by_symbol[s]) for s in active_syms], dtype=np.float64)
    w_sum = float(np.sum(w_raw))
    if w_sum <= 1e-12:
        return pd.Series(dtype="float64")
    w_norm = w_raw / w_sum

    port_rets = rets_df[active_syms].to_numpy(dtype=np.float64) @ w_norm
    return pd.Series(port_rets, index=rets_df.index, dtype="float64")


def classify_risk_level(var_95: float, annual_volatility: float) -> str:
    """Discretize portfolio risk profile into low | medium | high."""
    if var_95 >= 0.045 or annual_volatility >= 0.45:
        return "high"
    if var_95 < 0.020 and annual_volatility < 0.20:
        return "low"
    return "medium"


def compute_portfolio_risk_metrics(
    daily_returns: pd.Series,
    weights_by_symbol: dict[str, float],
    periods_per_year: int = 365,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """
    Compute Historical VaR (95%), CVaR / Expected Shortfall (95%), Annualized Volatility,
    Sharpe, Sortino, Max Drawdown, and discrete `risk_level`.
    """
    r = daily_returns.dropna().astype("float64")
    if len(r) < 2:
        return {
            "var_95": 0.0,
            "cvar_95": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "annual_volatility": 0.0,
            "risk_level": "low",
            "weights": dict(weights_by_symbol),
            "observations": int(len(r)),
            "periods_per_year": int(periods_per_year),
            "method": f"historical_1d_{len(r)}d",
        }

    r_arr = r.to_numpy(dtype=np.float64)
    tail_alpha = max(1e-4, min(0.5, 1.0 - confidence_level))
    q_tail = float(np.quantile(r_arr, tail_alpha))
    var_95 = max(0.0, -q_tail)

    tail_losses = r_arr[r_arr <= q_tail]
    cvar_95 = max(var_95, float(-np.mean(tail_losses))) if len(tail_losses) > 0 else var_95

    annual_vol = float(np.std(r_arr, ddof=1) * np.sqrt(periods_per_year))
    sharpe = calculate_sharpe_ratio(r, periods_per_year=periods_per_year)
    sortino = calculate_sortino_ratio(r, periods_per_year=periods_per_year)
    equity = (1.0 + r).cumprod()
    max_dd = calculate_max_drawdown(equity)
    level = classify_risk_level(var_95, annual_vol)

    return {
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "annual_volatility": float(annual_vol),
        "risk_level": level,
        "weights": {k: round(float(v), 6) for k, v in weights_by_symbol.items()},
        "observations": int(len(r)),
        "periods_per_year": int(periods_per_year),
        "method": f"historical_1d_{len(r)}d",
    }
