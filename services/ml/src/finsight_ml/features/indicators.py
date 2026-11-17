import numpy as np
import pandas as pd


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (scale-invariant, flat -> 50.0)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    denom = avg_gain + avg_loss
    rsi = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = denom.notna()
    zero_denom = valid & (denom == 0.0)
    nonzero_denom = valid & (denom > 0.0)

    rsi.loc[zero_denom] = 50.0
    rsi.loc[nonzero_denom] = 100.0 * (avg_gain.loc[nonzero_denom] / denom.loc[nonzero_denom])
    return rsi


def calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    fast_ema = calculate_ema(series, fast)
    slow_ema = calculate_ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def calculate_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (middle, upper, lower, bandwidth)."""
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = (upper - lower) / (middle + 1e-10)
    return middle, upper, lower, bandwidth


def calculate_log_returns(series: pd.Series, periods: list[int] | None = None) -> pd.DataFrame:
    """Log returns: ln(P_t / P_{t-k})."""
    if periods is None:
        periods = [1, 3, 6, 12, 24]
    df = pd.DataFrame(index=series.index)
    for p in periods:
        df[f"logret_{p}"] = np.log(series / series.shift(p))
    return df
