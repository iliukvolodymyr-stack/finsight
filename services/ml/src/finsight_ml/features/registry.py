import hashlib
import json
from dataclasses import dataclass
from typing import Literal

FEATURE_SET_NAME = "core_market_v1"
FEATURE_SET_VERSION = 1


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    dtype: Literal["float64", "int64", "bool"]
    lookback: int
    group: Literal["returns", "trend", "momentum", "volatility", "volume", "calendar"]
    description: str
    version: str = "1.0.0"


FEATURE_REGISTRY: list[FeatureDefinition] = [
    # 1. Returns (6)
    FeatureDefinition("logret_1", "float64", 1, "returns", "1-bar log return"),
    FeatureDefinition("logret_3", "float64", 3, "returns", "3-bar log return"),
    FeatureDefinition("logret_6", "float64", 6, "returns", "6-bar log return"),
    FeatureDefinition("logret_12", "float64", 12, "returns", "12-bar log return"),
    FeatureDefinition("logret_24", "float64", 24, "returns", "24-bar log return"),
    FeatureDefinition("logret_72", "float64", 72, "returns", "72-bar log return"),
    # 2. Trend — all dimensionless / price-normalized (8)
    FeatureDefinition(
        "norm_dist_ema20", "float64", 20, "trend", "Distance to EMA20 normalized by ATR14"
    ),
    FeatureDefinition(
        "norm_dist_ema50", "float64", 50, "trend", "Distance to EMA50 normalized by ATR14"
    ),
    FeatureDefinition(
        "norm_dist_ema200", "float64", 200, "trend", "Distance to EMA200 normalized by ATR14"
    ),
    FeatureDefinition("ema_ratio_20_50", "float64", 50, "trend", "Ratio of EMA20 to EMA50"),
    FeatureDefinition("ema_ratio_50_200", "float64", 200, "trend", "Ratio of EMA50 to EMA200"),
    FeatureDefinition(
        "macd_line", "float64", 26, "trend", "MACD line (12, 26) normalized by close"
    ),
    FeatureDefinition(
        "macd_signal", "float64", 35, "trend", "MACD signal line (9) normalized by close"
    ),
    FeatureDefinition("macd_hist", "float64", 35, "trend", "MACD histogram normalized by close"),
    # 3. Momentum (4)
    FeatureDefinition("rsi14", "float64", 14, "momentum", "14-period Wilder RSI"),
    FeatureDefinition("roc_12", "float64", 12, "momentum", "12-bar Rate of Change"),
    FeatureDefinition("stoch_k_14", "float64", 14, "momentum", "14-bar Stochastic %K"),
    FeatureDefinition("stoch_d_14", "float64", 16, "momentum", "3-bar SMA of Stochastic %K"),
    # 4. Volatility (6)
    FeatureDefinition(
        "atr14_norm", "float64", 14, "volatility", "14-period ATR normalized by close"
    ),
    FeatureDefinition("bb_width", "float64", 20, "volatility", "Bollinger Bandwidth (20, 2.0)"),
    FeatureDefinition("bb_pct_b", "float64", 20, "volatility", "Bollinger %B position (20, 2.0)"),
    FeatureDefinition(
        "rv_24", "float64", 25, "volatility", "24-bar realized volatility of log returns"
    ),
    FeatureDefinition(
        "rv_168", "float64", 169, "volatility", "168-bar realized volatility of log returns"
    ),
    FeatureDefinition(
        "rv_ratio_24_168",
        "float64",
        169,
        "volatility",
        "Short-to-long realized volatility ratio (24 / 168)",
    ),
    # 5. Volume — all dimensionless / volume-normalized (4)
    FeatureDefinition("vol_zscore_24", "float64", 24, "volume", "24-bar rolling volume z-score"),
    FeatureDefinition(
        "vol_ratio_24", "float64", 24, "volume", "Current volume divided by 24-bar rolling mean"
    ),
    FeatureDefinition(
        "obv_slope_14",
        "float64",
        15,
        "volume",
        "14-bar directional OBV pressure normalized by 14-bar volume sum",
    ),
    FeatureDefinition(
        "vwap_dist_24",
        "float64",
        24,
        "volume",
        "Distance from close to 24-bar rolling VWAP normalized by ATR14",
    ),
    # 6. Calendar — cyclical UTC bar open_time encodings (4)
    FeatureDefinition("hour_sin", "float64", 1, "calendar", "Sine of UTC bar open hour"),
    FeatureDefinition("hour_cos", "float64", 1, "calendar", "Cosine of UTC bar open hour"),
    FeatureDefinition("dow_sin", "float64", 1, "calendar", "Sine of UTC day of week"),
    FeatureDefinition("dow_cos", "float64", 1, "calendar", "Cosine of UTC day of week"),
]


def get_feature_names() -> list[str]:
    return [f.name for f in FEATURE_REGISTRY]


def compute_schema_hash(registry: list[FeatureDefinition] | None = None) -> str:
    """Deterministic SHA-256 hash of feature names, dtypes, and lookbacks (train/serve contract)."""
    items = registry if registry is not None else FEATURE_REGISTRY
    payload = [
        {
            "name": f.name,
            "dtype": f.dtype,
            "lookback": f.lookback,
            "version": f.version,
        }
        for f in items
    ]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
