from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    dtype: Literal["float64", "int64", "bool"]
    lookback: int
    description: str
    version: str = "1.0.0"


FEATURE_REGISTRY: list[FeatureDefinition] = [
    # Returns
    FeatureDefinition("logret_1", "float64", 1, "1-bar log return"),
    FeatureDefinition("logret_3", "float64", 3, "3-bar log return"),
    FeatureDefinition("logret_6", "float64", 6, "6-bar log return"),
    FeatureDefinition("logret_12", "float64", 12, "12-bar log return"),
    FeatureDefinition("logret_24", "float64", 24, "24-bar log return"),
    # Trend
    FeatureDefinition("ema_20", "float64", 20, "20-period Exponential Moving Average"),
    FeatureDefinition("ema_50", "float64", 50, "50-period Exponential Moving Average"),
    FeatureDefinition("ema_200", "float64", 200, "200-period Exponential Moving Average"),
    FeatureDefinition("ema_ratio_20_50", "float64", 50, "Ratio of EMA20 to EMA50"),
    FeatureDefinition("macd_line", "float64", 26, "MACD Line (12, 26)"),
    FeatureDefinition("macd_signal", "float64", 35, "MACD Signal Line (9)"),
    FeatureDefinition("macd_hist", "float64", 35, "MACD Histogram"),
    # Momentum
    FeatureDefinition("rsi14", "float64", 14, "14-period Relative Strength Index"),
    # Volatility
    FeatureDefinition("atr14", "float64", 14, "14-period Average True Range"),
    FeatureDefinition("bb_width", "float64", 20, "Bollinger Bandwidth (20, 2.0)"),
]


def get_feature_names() -> list[str]:
    return [f.name for f in FEATURE_REGISTRY]
