import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from finsight_ml.features.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_log_returns,
    calculate_macd,
    calculate_obv_slope_norm,
    calculate_realized_volatility,
    calculate_roc,
    calculate_rolling_vwap,
    calculate_rsi,
    calculate_stochastic,
    safe_div,
)
from finsight_ml.features.registry import (
    FEATURE_REGISTRY,
    FEATURE_SET_NAME,
    FEATURE_SET_VERSION,
    FeatureDefinition,
    compute_schema_hash,
)


def _extract_utc_timestamps(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract UTC DatetimeIndex from df.index or df['open_time']."""
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "open_time" in df.columns:
        ts = pd.DatetimeIndex(pd.to_datetime(df["open_time"], utc=True))
    else:
        raise ValueError("Input DataFrame must have a DatetimeIndex or an 'open_time' column.")

    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def compute_row_hash(row: pd.Series) -> str:
    """Deterministic SHA-256 hash of a single feature vector (rounded to 8 decimals)."""
    payload: dict[str, Any] = {}
    for col, val in row.items():
        fval = float(val)
        payload[str(col)] = None if np.isnan(fval) else round(fval, 8)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class FeaturePipeline:
    """
    Single train/serve point-in-time feature computation pipeline (`core_market_v1`).

    Guarantees:
    - Zero look-ahead: every feature at index t uses strictly OHLCV data <= t.
    - Scale invariance: multiplying prices by c_p > 0 or volume by c_v > 0 does not alter features.
    - Fixed column order and deterministic SHA-256 `schema_hash`.
    """

    def __init__(self, registry: list[FeatureDefinition] | None = None) -> None:
        self.registry: list[FeatureDefinition] = (
            list(registry) if registry is not None else list(FEATURE_REGISTRY)
        )
        self.name = FEATURE_SET_NAME
        self.version = FEATURE_SET_VERSION
        self.schema_hash = compute_schema_hash(self.registry)
        self.feature_names = [f.name for f in self.registry]
        self.max_lookback = max((f.lookback for f in self.registry), default=0)
        # EMA200 needs ~1500 bars for (1 - 2/201)^n < 1e-6 train/serve parity
        self.serve_warmup_bars = 1500

    def to_schema_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "schema_hash": self.schema_hash,
            "max_lookback": self.max_lookback,
            "features": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "lookback": f.lookback,
                    "group": f.group,
                    "version": f.version,
                }
                for f in self.registry
            ],
        }

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

        if df.empty:
            return pd.DataFrame(columns=self.feature_names, index=df.index, dtype="float64")

        ts_utc = _extract_utc_timestamps(df)
        close = df["close"].astype("float64")
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        volume = df["volume"].astype("float64")

        # 1. Returns (6)
        log_rets = calculate_log_returns(close, [1, 3, 6, 12, 24, 72])

        # 2. Trend (8) — all dimensionless
        ema20 = calculate_ema(close, 20)
        ema50 = calculate_ema(close, 50)
        ema200 = calculate_ema(close, 200)
        atr14 = calculate_atr(high, low, close, 14)

        norm_dist_ema20 = safe_div(close - ema20, atr14, fill_zero=0.0)
        norm_dist_ema50 = safe_div(close - ema50, atr14, fill_zero=0.0)
        norm_dist_ema200 = safe_div(close - ema200, atr14, fill_zero=0.0)
        ema_ratio_20_50 = safe_div(ema20, ema50, fill_zero=1.0)
        ema_ratio_50_200 = safe_div(ema50, ema200, fill_zero=1.0)

        macd_raw, macd_sig_raw, macd_hist_raw = calculate_macd(close)
        macd_line = safe_div(macd_raw, close, fill_zero=0.0)
        macd_signal = safe_div(macd_sig_raw, close, fill_zero=0.0)
        macd_hist = safe_div(macd_hist_raw, close, fill_zero=0.0)

        # 3. Momentum (4)
        rsi14 = calculate_rsi(close, 14)
        roc_12 = calculate_roc(close, 12)
        stoch_k_14, stoch_d_14 = calculate_stochastic(high, low, close, k_period=14, d_period=3)

        # 4. Volatility (6)
        atr14_norm = safe_div(atr14, close, fill_zero=0.0)
        _, bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
        bb_pct_b = safe_div(close - bb_lower, bb_upper - bb_lower, fill_zero=0.5)
        rv_24 = calculate_realized_volatility(log_rets["logret_1"], 24)
        rv_168 = calculate_realized_volatility(log_rets["logret_1"], 168)
        rv_ratio_24_168 = safe_div(rv_24, rv_168, fill_zero=1.0)

        # 5. Volume (4) — all scale-invariant
        vol_mean_24 = volume.rolling(window=24, min_periods=24).mean()
        vol_std_24 = volume.rolling(window=24, min_periods=24).std()
        vol_zscore_24 = safe_div(volume - vol_mean_24, vol_std_24, fill_zero=0.0)
        vol_ratio_24 = safe_div(volume, vol_mean_24, fill_zero=1.0)
        obv_slope_14 = calculate_obv_slope_norm(close, volume, window=14)
        vwap_24 = calculate_rolling_vwap(high, low, close, volume, window=24)
        vwap_dist_24 = safe_div(close - vwap_24, atr14, fill_zero=0.0)

        # 6. Calendar (4)
        hour_angle = 2.0 * np.pi * (ts_utc.hour.to_numpy(dtype=np.float64) / 24.0)
        dow_angle = 2.0 * np.pi * (ts_utc.day_of_week.to_numpy(dtype=np.float64) / 7.0)

        all_cols: dict[str, pd.Series | np.ndarray] = {
            "logret_1": log_rets["logret_1"],
            "logret_3": log_rets["logret_3"],
            "logret_6": log_rets["logret_6"],
            "logret_12": log_rets["logret_12"],
            "logret_24": log_rets["logret_24"],
            "logret_72": log_rets["logret_72"],
            "norm_dist_ema20": norm_dist_ema20,
            "norm_dist_ema50": norm_dist_ema50,
            "norm_dist_ema200": norm_dist_ema200,
            "ema_ratio_20_50": ema_ratio_20_50,
            "ema_ratio_50_200": ema_ratio_50_200,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "rsi14": rsi14,
            "roc_12": roc_12,
            "stoch_k_14": stoch_k_14,
            "stoch_d_14": stoch_d_14,
            "atr14_norm": atr14_norm,
            "bb_width": bb_width,
            "bb_pct_b": bb_pct_b,
            "rv_24": rv_24,
            "rv_168": rv_168,
            "rv_ratio_24_168": rv_ratio_24_168,
            "vol_zscore_24": vol_zscore_24,
            "vol_ratio_24": vol_ratio_24,
            "obv_slope_14": obv_slope_14,
            "vwap_dist_24": vwap_dist_24,
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "dow_sin": np.sin(dow_angle),
            "dow_cos": np.cos(dow_angle),
        }

        features = pd.DataFrame(
            {name: all_cols[name] for name in self.feature_names},
            index=df.index,
            dtype="float64",
        )
        return features


_DEFAULT_PIPELINE = FeaturePipeline()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute core_market_v1 features using the single train/serve FeaturePipeline."""
    return _DEFAULT_PIPELINE.compute(df)
