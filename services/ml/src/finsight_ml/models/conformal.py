import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConformalPredictionInterval:
    price_low: np.ndarray
    price_high: np.ndarray
    coverage: float


class VolatilityNormalizedConformal:
    """
    Volatility-normalized Split Conformal price corridor.

    Calibrated on validation forward H-bar log-returns normalized by current
    volatility scale:
        z_t = log(close_{t+H} / close_t) / (sigma_t * sqrt(H))

    At inference time:
        price_low  = close_t * exp(z_low  * sigma_t * sqrt(H))
        price_high = close_t * exp(z_high * sigma_t * sqrt(H))

    This adapts interval width proportionally to current volatility while targeting
    `1 - alpha` (default 80%) empirical out-of-sample coverage.
    """

    def __init__(
        self,
        alpha: float = 0.20,
        horizon_bars: int = 24,
        min_sigma: float = 1e-4,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        self.alpha = float(alpha)
        self.horizon_bars = int(horizon_bars)
        self.min_sigma = float(min_sigma)
        self.z_low: float = -1.28155
        self.z_high: float = 1.28155
        self.n_calibration: int = 0

    def fit(
        self, forward_log_returns: np.ndarray, sigma: np.ndarray
    ) -> "VolatilityNormalizedConformal":
        ret = np.asarray(forward_log_returns, dtype=np.float64)
        sig = np.asarray(sigma, dtype=np.float64)
        valid = np.isfinite(ret) & np.isfinite(sig) & (sig > 0.0)
        if int(np.sum(valid)) < 10:
            raise ValueError(
                "Need at least 10 finite calibration residuals to fit conformal quantiles."
            )

        scale = np.maximum(sig[valid], self.min_sigma) * float(np.sqrt(self.horizon_bars))
        z = ret[valid] / scale
        n = len(z)
        self.n_calibration = n

        # Finite-sample conservative split conformal quantile levels for [alpha/2, 1 - alpha/2]
        q_lo_level = max(0.0, min(1.0, (self.alpha / 2.0) * (n + 1) / n))
        q_hi_level = max(0.0, min(1.0, (1.0 - self.alpha / 2.0) * (n + 1) / n))

        # Fall back cleanly toward empirical quantiles when finite-sample correction saturates small n
        q_lo_level = (
            min(q_lo_level, self.alpha / 2.0)
            if n < 10
            else self.alpha / 2.0 * (1.0 - 1.0 / (n + 1))
        )
        z_sorted = np.sort(z)
        lo_idx = max(0, min(n - 1, int(np.floor((self.alpha / 2.0) * n))))
        hi_idx = max(0, min(n - 1, int(np.ceil((1.0 - self.alpha / 2.0) * (n + 1))) - 1))
        _ = (q_lo_level, q_hi_level)

        self.z_low = float(z_sorted[lo_idx])
        self.z_high = float(max(self.z_low, z_sorted[hi_idx]))
        return self

    def predict_interval(
        self,
        close_prices: np.ndarray,
        sigma: np.ndarray,
    ) -> ConformalPredictionInterval:
        close = np.asarray(close_prices, dtype=np.float64)
        sig = np.maximum(
            np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, self.min_sigma),
            self.min_sigma,
        )
        scale = sig * float(np.sqrt(self.horizon_bars))
        price_low = close * np.exp(self.z_low * scale)
        price_high = close * np.exp(self.z_high * scale)
        return ConformalPredictionInterval(
            price_low=price_low,
            price_high=price_high,
            coverage=1.0 - self.alpha,
        )

    def evaluate_coverage(
        self,
        close_prices: np.ndarray,
        future_close_prices: np.ndarray,
        sigma: np.ndarray,
    ) -> dict[str, Any]:
        interval = self.predict_interval(close_prices, sigma)
        future = np.asarray(future_close_prices, dtype=np.float64)
        valid = (
            np.isfinite(future) & np.isfinite(interval.price_low) & np.isfinite(interval.price_high)
        )
        if not np.any(valid):
            return {
                "coverage": 0.0,
                "mean_rel_width": 0.0,
                "covered_mask": np.array([], dtype=bool),
            }

        covered = (future[valid] >= interval.price_low[valid]) & (
            future[valid] <= interval.price_high[valid]
        )
        rel_width = (interval.price_high[valid] - interval.price_low[valid]) / np.maximum(
            np.asarray(close_prices, dtype=np.float64)[valid], 1e-12
        )
        return {
            "coverage": float(np.mean(covered)),
            "mean_rel_width": float(np.mean(rel_width)),
            "covered_mask": covered,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "volatility_normalized_split_conformal",
            "alpha": self.alpha,
            "target_coverage": 1.0 - self.alpha,
            "horizon_bars": self.horizon_bars,
            "min_sigma": self.min_sigma,
            "z_low": self.z_low,
            "z_high": self.z_high,
            "n_calibration": self.n_calibration,
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "VolatilityNormalizedConformal":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            alpha=float(data["alpha"]),
            horizon_bars=int(data["horizon_bars"]),
            min_sigma=float(data.get("min_sigma", 1e-4)),
        )
        obj.z_low = float(data["z_low"])
        obj.z_high = float(data["z_high"])
        obj.n_calibration = int(data.get("n_calibration", 0))
        return obj
