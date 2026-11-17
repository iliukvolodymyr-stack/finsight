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
    # Trend (Normalized)
    FeatureDefinition("norm_dist_ema20", "float64", 20, "Distance to EMA20 normalized by ATR14"),
    FeatureDefinition("norm_dist_ema50", "float64", 50, "Distance to EMA50 normalized by ATR14"),
    FeatureDefinition("norm_dist_ema200", "float64", 200, "Distance to EMA200 normalized by ATR14"),
    FeatureDefinition("ema_ratio_20_50", "float64", 50, "Ratio of EMA20 to EMA50"),
    FeatureDefinition("ema_ratio_50_200", "float64", 200, "Ratio of EMA50 to EMA200"),
    FeatureDefinition("macd_line", "float64", 26, "Normalized MACD Line (12, 26)"),
    FeatureDefinition("macd_signal", "float64", 35, "Normalized MACD Signal Line (9)"),
    FeatureDefinition("macd_hist", "float64", 35, "Normalized MACD Histogram"),
    # Momentum
    FeatureDefinition("rsi14", "float64", 14, "14-period Relative Strength Index"),
    # Volatility
    FeatureDefinition(
        "atr14_norm", "float64", 14, "14-period Average True Range normalized by close"
    ),
    FeatureDefinition("bb_width", "float64", 20, "Bollinger Bandwidth (20, 2.0)"),
]


def get_feature_names() -> list[str]:
    return [f.name for f in FEATURE_REGISTRY]
