import numpy as np
import pandas as pd


def apply_triple_barrier(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon_bars: int = 24,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    volatility: pd.Series | None = None,
    min_barrier_pct: float = 0.003,
) -> pd.Series:
    """
    Triple-barrier labeling method (Marcos Lopez de Prado):
    - Upper barrier (profit taking): P_t * exp(+pt_mult * sigma_t * sqrt(H))
    - Lower barrier (stop loss): P_t * exp(-sl_mult * sigma_t * sqrt(H))
    - Time barrier: horizon_bars (H)
    - Minimum barrier width: min_barrier_pct (default 30 bps); narrower -> HOLD
    - Tail: last horizon_bars rows have insufficient future data -> NaN

    Returns float64 Series with values: 1.0 (BUY), -1.0 (SELL), 0.0 (HOLD), or NaN.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")

    n = len(close)
    labels = pd.Series(np.nan, index=close.index, dtype="float64")
    if n <= horizon_bars:
        return labels

    if volatility is None:
        log_ret = np.log(close / close.shift(1))
        volatility = log_ret.ewm(span=24, min_periods=2).std()

    horizon_scale = float(np.sqrt(horizon_bars))

    for i in range(n - horizon_bars):
        entry_price = float(close.iloc[i])
        vol = float(volatility.iloc[i])
        if np.isnan(entry_price) or entry_price <= 0 or np.isnan(vol):
            continue

        upper_width = pt_mult * vol * horizon_scale
        lower_width = sl_mult * vol * horizon_scale

        # If volatility is too low to cover round-trip trading costs, do not trade
        if max(upper_width, lower_width) < min_barrier_pct:
            labels.iloc[i] = 0.0
            continue

        upper_barrier = entry_price * float(np.exp(upper_width))
        lower_barrier = entry_price * float(np.exp(-lower_width))

        hit = 0.0
        for step in range(1, horizon_bars + 1):
            h = float(high.iloc[i + step])
            l = float(low.iloc[i + step])

            touch_up = h >= upper_barrier
            touch_down = l <= lower_barrier

            if touch_up and touch_down:
                # Conservative: when both barriers touched in same bar, prioritize stop-loss
                hit = -1.0
                break
            if touch_up:
                hit = 1.0
                break
            if touch_down:
                hit = -1.0
                break

        labels.iloc[i] = hit

    return labels
