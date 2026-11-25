from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """
    Out-of-fold 3-class probability calibrator fit strictly on validation folds.

    Strategy (docs/04-ml-pipeline.md §6.4):
    - If n_val < 1000: one-vs-rest Platt scaling (LogisticRegression on logit(p_c))
      to prevent isotonic overfitting on small validation sets.
    - If n_val >= 1000: one-vs-rest IsotonicRegression per class.
    - Output rows are always non-negative and normalized to sum to 1.0.
    """

    def __init__(self, isotonic_min_samples: int = 1000) -> None:
        self.isotonic_min_samples = isotonic_min_samples
        self.method: str = "unfitted"
        self.calibrators: dict[int, LogisticRegression | IsotonicRegression | float] = {}

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        p_arr = np.asarray(probs, dtype=np.float64)
        y_arr = np.asarray(y_true, dtype=np.int64)
        if p_arr.ndim != 2 or p_arr.shape[1] != 3:
            raise ValueError(f"Expected probs shape (N, 3), got {p_arr.shape}")
        if len(p_arr) != len(y_arr) or len(p_arr) == 0:
            raise ValueError("probs and y_true must be non-empty and equal length")

        # Map {-1, 0, 1} -> {0, 1, 2}
        y_mapped = y_arr + 1
        use_isotonic = len(y_mapped) >= self.isotonic_min_samples
        self.method = "isotonic" if use_isotonic else "platt"
        self.calibrators = {}

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
        if len(p_arr) == 0:
            return p_arr.copy()
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
        # Safe fallback to raw probabilities if all calibrated scores hit 0 simultaneously
        raw_sums = np.sum(p_arr, axis=1, keepdims=True)
        p_fallback = np.where(raw_sums > 0.0, p_arr / np.maximum(raw_sums, 1e-12), 1.0 / 3.0)
        normalized = np.where(row_sums > 1e-12, out / np.maximum(row_sums, 1e-12), p_fallback)
        return np.asarray(normalized, dtype=np.float64)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "isotonic_min_samples": self.isotonic_min_samples,
                "method": self.method,
                "calibrators": self.calibrators,
            },
            p,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ProbabilityCalibrator":
        data = joblib.load(path)
        obj = cls(isotonic_min_samples=int(data.get("isotonic_min_samples", 1000)))
        obj.method = str(data.get("method", "unfitted"))
        obj.calibrators = dict(data.get("calibrators", {}))
        return obj
