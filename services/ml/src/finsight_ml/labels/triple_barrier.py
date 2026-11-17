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
) -> pd.Series:
    """
    Triple-barrier labeling method (Marcos Lopez de Prado):
    - Upper barrier (profit taking): +pt_mult * vol
    - Lower barrier (stop loss): -sl_mult * vol
    - Time barrier: horizon_bars
    Returns series with values: 1 (BUY), -1 (SELL), 0 (HOLD).
    """
    n = len(close)
    labels = pd.Series(0, index=close.index, dtype=int)

    if volatility is None:
        # Default daily/rolling volatility estimate
        volatility = close.pct_change().rolling(24).std().fillna(0.01)

    for i in range(n - horizon_bars):
        entry_price = close.iloc[i]
        vol = volatility.iloc[i]
        if np.isnan(vol) or vol <= 0:
            vol = 0.01

        upper_barrier = entry_price * (1.0 + pt_mult * vol)
        lower_barrier = entry_price * (1.0 - sl_mult * vol)

        hit = 0
        for step in range(1, horizon_bars + 1):
            h = high.iloc[i + step]
            l = low.iloc[i + step]

            if h >= upper_barrier:
                hit = 1
                break
            elif l <= lower_barrier:
                hit = -1
                break

        labels.iloc[i] = hit

    return labels
