from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """
    Out-of-fold 3-class probability calibrator fit strictly on validation folds.

    Supported methods:
    - `temperature` (default): 1-parameter multiclass Temperature Scaling
      `softmax(log(p) / T)` optimizing NLL on val. Preserves class ranking
      (`argmax` / `Macro-F1` invariant) and resists overfitting on non-stationary
      financial validation folds.
    - `platt`: One-vs-rest Platt scaling (`LogisticRegression` on logit(p_c)).
    - `isotonic`: One-vs-rest `IsotonicRegression` (only when `n_val >= isotonic_min_samples`
      and requested explicitly via `method="isotonic"` or `"auto"`).
    """

    def __init__(
        self,
        method: Literal["temperature", "platt", "isotonic", "auto"] = "temperature",
        isotonic_min_samples: int = 1000,
    ) -> None:
        self.requested_method = method
        self.isotonic_min_samples = isotonic_min_samples
        self.method: str = "unfitted"
        self.temperature: float = 1.0
        self.calibrators: dict[int, LogisticRegression | IsotonicRegression | float] = {}

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)

    @staticmethod
    def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
        row_sums = np.sum(probs, axis=1, keepdims=True)
        p_norm = np.where(row_sums > 0.0, probs / np.maximum(row_sums, 1e-12), 1.0 / 3.0)
        logits = np.log(np.clip(p_norm, 1e-7, 1.0)) / float(temperature)
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return np.asarray(exp_l / np.sum(exp_l, axis=1, keepdims=True), dtype=np.float64)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        p_arr = np.asarray(probs, dtype=np.float64)
        y_arr = np.asarray(y_true, dtype=np.int64)
        if p_arr.ndim != 2 or p_arr.shape[1] != 3:
            raise ValueError(f"Expected probs shape (N, 3), got {p_arr.shape}")
        if len(p_arr) != len(y_arr) or len(p_arr) == 0:
            raise ValueError("probs and y_true must be non-empty and equal length")

        y_mapped = y_arr + 1
        if self.requested_method == "auto":
            resolved = "isotonic" if len(y_mapped) >= self.isotonic_min_samples else "platt"
        else:
            resolved = self.requested_method

        self.method = resolved
        self.calibrators = {}
        self.temperature = 1.0

        if resolved == "temperature":
            idx_rows = np.arange(len(y_mapped))

            def _nll(temp: float) -> float:
                cal_p = self._apply_temperature(p_arr, temp)
                chosen = np.clip(cal_p[idx_rows, y_mapped], 1e-12, 1.0)
                return float(-np.mean(np.log(chosen)))

            base_nll = _nll(1.0)
            res = minimize_scalar(_nll, bounds=(0.25, 10.0), method="bounded")
            opt_temp = float(res.x)
            self.temperature = opt_temp if float(res.fun) < base_nll else 1.0
            return self

        use_isotonic = resolved == "isotonic"
        for cls_id in (0, 1, 2):
            y_bin = (y_mapped == cls_id).astype(np.int64)
            n_pos = int(np.sum(y_bin))
            if n_pos == 0 or n_pos == len(y_bin):
                self.calibrators[cls_id] = float(np.mean(y_bin))
                continue

            if use_isotonic:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(p_arr[:, cls_id], y_bin)
                self.calibrators[cls_id] = iso
            else:
                lr = LogisticRegression(C=10.0, max_iter=500)
                lr.fit(self._logit(p_arr[:, cls_id]).reshape(-1, 1), y_bin)
                self.calibrators[cls_id] = lr

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(probs, dtype=np.float64)
        if p_arr.ndim != 2 or p_arr.shape[1] != 3:
            raise ValueError(f"Expected probs shape (N, 3), got {p_arr.shape}")
        if len(p_arr) == 0 or self.method == "unfitted":
            return p_arr.copy()

        if self.method == "temperature":
            return self._apply_temperature(p_arr, self.temperature)

        if not self.calibrators:
            return p_arr.copy()

        out = np.zeros_like(p_arr, dtype=np.float64)
        for cls_id in (0, 1, 2):
            cal = self.calibrators[cls_id]
            if isinstance(cal, float):
                out[:, cls_id] = cal
            elif isinstance(cal, IsotonicRegression):
                out[:, cls_id] = cal.predict(p_arr[:, cls_id])
            else:
                out[:, cls_id] = cal.predict_proba(self._logit(p_arr[:, cls_id]).reshape(-1, 1))[
                    :, 1
                ]

        out = np.clip(out, 0.0, 1.0)
        row_sums = np.sum(out, axis=1, keepdims=True)
        raw_sums = np.sum(p_arr, axis=1, keepdims=True)
        p_fallback = np.where(raw_sums > 0.0, p_arr / np.maximum(raw_sums, 1e-12), 1.0 / 3.0)
        normalized = np.where(row_sums > 1e-12, out / np.maximum(row_sums, 1e-12), p_fallback)
        return np.asarray(normalized, dtype=np.float64)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "requested_method": self.requested_method,
                "isotonic_min_samples": self.isotonic_min_samples,
                "method": self.method,
                "temperature": self.temperature,
                "calibrators": self.calibrators,
            },
            p,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ProbabilityCalibrator":
        data = joblib.load(path)
        obj = cls(
            method=data.get("requested_method", "temperature"),
            isotonic_min_samples=int(data.get("isotonic_min_samples", 1000)),
        )
        obj.method = str(data.get("method", "unfitted"))
        obj.temperature = float(data.get("temperature", 1.0))
        obj.calibrators = dict(data.get("calibrators", {}))
        return obj
