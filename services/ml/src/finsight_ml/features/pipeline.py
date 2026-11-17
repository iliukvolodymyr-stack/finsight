import pandas as pd
from finsight_ml.features.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_log_returns,
    calculate_macd,
    calculate_rsi,
)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from OHLCV dataframe.
    Guarantees no lookahead (only backward rolling calculations).
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Log returns
    log_rets = calculate_log_returns(close, [1, 3, 6, 12, 24])

    # Trend
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    ema200 = calculate_ema(close, 200)
    ema_ratio_20_50 = ema20 / (ema50 + 1e-10)
    ema_ratio_50_200 = ema50 / (ema200 + 1e-10)
    macd_line, macd_signal, macd_hist = calculate_macd(close)

    # Momentum
    rsi14 = calculate_rsi(close, 14)

    # Volatility
    atr14 = calculate_atr(high, low, close, 14)
    atr14_norm = atr14 / (close + 1e-10)
    _, _, _, bb_width = calculate_bollinger_bands(close, 20, 2.0)

    # Normalized relative distance to EMAs (stationary representation)
    norm_dist_ema20 = (close - ema20) / (atr14 + 1e-10)
    norm_dist_ema50 = (close - ema50) / (atr14 + 1e-10)
    norm_dist_ema200 = (close - ema200) / (atr14 + 1e-10)

    features = pd.DataFrame(
        {
            "logret_1": log_rets["logret_1"],
            "logret_3": log_rets["logret_3"],
            "logret_6": log_rets["logret_6"],
            "logret_12": log_rets["logret_12"],
            "logret_24": log_rets["logret_24"],
            "norm_dist_ema20": norm_dist_ema20,
            "norm_dist_ema50": norm_dist_ema50,
            "norm_dist_ema200": norm_dist_ema200,
            "ema_ratio_20_50": ema_ratio_20_50,
            "ema_ratio_50_200": ema_ratio_50_200,
            "macd_line": macd_line / (close + 1e-10),
            "macd_signal": macd_signal / (close + 1e-10),
            "macd_hist": macd_hist / (close + 1e-10),
            "rsi14": rsi14,
            "atr14_norm": atr14_norm,
            "bb_width": bb_width,
        },
        index=df.index,
    )
    return features
