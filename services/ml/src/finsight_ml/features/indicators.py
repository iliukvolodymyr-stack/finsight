import numpy as np
import pandas as pd


def safe_div(
    numerator: pd.Series,
    denominator: pd.Series | float,
    fill_zero: float = 0.0,
) -> pd.Series:
    """Scale-invariant elementwise division preserving NaNs and replacing 0-denominators."""
    num = numerator.astype("float64")
    if isinstance(denominator, int | float):
        if denominator == 0.0:
            out = pd.Series(fill_zero, index=num.index, dtype="float64")
            out[num.isna()] = np.nan
            return out
        return num / float(denominator)

    den = denominator.astype("float64")
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = num.notna() & den.notna()
    zero_den = valid & (den == 0.0)
    nonzero_den = valid & (den != 0.0)
    out.loc[zero_den] = fill_zero
    out.loc[nonzero_den] = num.loc[nonzero_den] / den.loc[nonzero_den]
    return out


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
    """MACD line, signal line, and histogram (raw price units for UI overlay)."""
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
    bandwidth = safe_div(upper - lower, middle, fill_zero=0.0)
    return middle, upper, lower, bandwidth


def calculate_log_returns(series: pd.Series, periods: list[int] | None = None) -> pd.DataFrame:
    """Log returns: ln(P_t / P_{t-k})."""
    if periods is None:
        periods = [1, 3, 6, 12, 24]
    df = pd.DataFrame(index=series.index)
    for p in periods:
        df[f"logret_{p}"] = np.log(series / series.shift(p))
    return df


def calculate_roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change: (P_t / P_{t-k}) - 1."""
    return safe_div(series, series.shift(period), fill_zero=1.0) - 1.0


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (%K, %D), scale-invariant, flat range -> 50.0."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    spread = highest_high - lowest_low
    stoch_k = 100.0 * safe_div(close - lowest_low, spread, fill_zero=0.5)
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
    return stoch_k, stoch_d


def calculate_realized_volatility(log_ret_1: pd.Series, window: int) -> pd.Series:
    """Backward rolling standard deviation of 1-bar log returns."""
    return log_ret_1.rolling(window=window, min_periods=window).std()


def calculate_obv_slope_norm(close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """
    Normalized OBV slope over `window` bars:
        sum_{k=t-window+1}^{t} sign(close_k - close_{k-1}) * volume_k
        -------------------------------------------------------------
        sum_{k=t-window+1}^{t} volume_k

    Strictly bounded in [-1, 1], stationary, and invariant to both price and volume scaling.
    """
    delta = close.diff()
    direction = np.sign(delta)
    signed_vol = direction * volume.astype("float64")
    obv_delta = signed_vol.rolling(window=window, min_periods=window).sum()
    vol_sum = volume.astype("float64").rolling(window=window, min_periods=window).sum()
    return safe_div(obv_delta, vol_sum, fill_zero=0.0)


def calculate_rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 24,
) -> pd.Series:
    """Rolling typical-price VWAP over `window` bars (falls back to rolling TP mean if volume sum == 0)."""
    typical_price = (high + low + close) / 3.0
    vol = volume.astype("float64")
    pv_sum = (typical_price * vol).rolling(window=window, min_periods=window).sum()
    v_sum = vol.rolling(window=window, min_periods=window).sum()
    tp_mean = typical_price.rolling(window=window, min_periods=window).mean()

    vwap = pd.Series(np.nan, index=close.index, dtype="float64")
    valid = pv_sum.notna() & v_sum.notna()
    has_vol = valid & (v_sum > 0.0)
    no_vol = valid & (v_sum <= 0.0)
    vwap.loc[has_vol] = pv_sum.loc[has_vol] / v_sum.loc[has_vol]
    vwap.loc[no_vol] = tp_mean.loc[no_vol]
    return vwap
