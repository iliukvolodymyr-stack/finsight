from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from finsight_ml.models.base import BaseForecaster
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _prepare_targets(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    valid_mask = y.notna()
    X_valid = X.loc[valid_mask]
    y_valid = y.loc[valid_mask].astype(int)
    unexpected = set(y_valid.unique()) - {-1, 0, 1}
    if unexpected:
        raise ValueError(f"Unexpected label values: {unexpected}. Expected subset of {{-1, 0, 1}}.")
    y_mapped = y_valid.map({-1: 0, 0: 1, 1: 2}).to_numpy(dtype=np.int64)
    return X_valid, y_mapped, valid_mask.to_numpy(dtype=bool)


class MajorityBaseline(BaseForecaster):
    """Sanity-check baseline predicting empirical training class prior (argmax = majority)."""

    def __init__(self) -> None:
        self.class_probs = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "MajorityBaseline":
        _, y_mapped, valid_mask = _prepare_targets(X, y)
        if len(y_mapped) == 0:
            raise ValueError("Cannot fit MajorityBaseline on empty targets.")

        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64)[valid_mask]
        else:
            w = np.ones(len(y_mapped), dtype=np.float64)

        counts = np.zeros(3, dtype=np.float64)
        for cls_id in (0, 1, 2):
            counts[cls_id] = float(np.sum(w[y_mapped == cls_id]))
        total = float(np.sum(counts))
        self.class_probs = counts / total if total > 0.0 else np.array([1.0 / 3.0] * 3)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(self.class_probs, (len(X), 1))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"class_probs": self.class_probs}, p)

    @classmethod
    def load(cls, path: Path | str) -> "MajorityBaseline":
        obj = cls()
        data = joblib.load(path)
        obj.class_probs = np.asarray(data["class_probs"], dtype=np.float64)
        return obj


class LogisticRegressionBaseline(BaseForecaster):
    """
    Linear Level-0 baseline (`SimpleImputer -> StandardScaler -> LogisticRegression`).
    Scaling statistics are strictly fit inside each training fold (zero leakage).
    """

    def __init__(
        self,
        c_reg: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.c_reg = c_reg
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.classes_: np.ndarray = np.array([0, 1, 2], dtype=np.int64)
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.c_reg,
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "LogisticRegressionBaseline":
        X_valid, y_mapped, valid_mask = _prepare_targets(X, y)
        if len(np.unique(y_mapped)) < 2:
            raise ValueError(
                "Need at least 2 distinct classes in y to fit LogisticRegressionBaseline."
            )
        self.feature_names = list(X_valid.columns)
        fit_kwargs: dict[str, object] = {}
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = np.asarray(sample_weight, dtype=np.float64)[
                valid_mask
            ]
        self.pipeline.fit(X_valid.to_numpy(dtype=np.float64), y_mapped, **fit_kwargs)
        self.classes_ = np.asarray(self.pipeline.named_steps["clf"].classes_, dtype=np.int64)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = (
            X[self.feature_names].to_numpy(dtype=np.float64)
            if self.feature_names
            else X.to_numpy(dtype=np.float64)
        )
        raw_probs = np.asarray(self.pipeline.predict_proba(X_arr), dtype=np.float64)
        full_probs = np.zeros((len(X), 3), dtype=np.float64)
        for col_idx, cls_id in enumerate(self.classes_):
            full_probs[:, int(cls_id)] = raw_probs[:, col_idx]
        return full_probs

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "classes_": self.classes_,
                "c_reg": self.c_reg,
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
            },
            p,
        )

    @classmethod
    def load(cls, path: Path | str) -> "LogisticRegressionBaseline":
        data = joblib.load(path)
        obj = cls(
            c_reg=float(data.get("c_reg", 1.0)),
            max_iter=int(data.get("max_iter", 1000)),
            class_weight=data.get("class_weight", "balanced"),
            random_state=int(data.get("random_state", 42)),
        )
        obj.pipeline = data["pipeline"]
        obj.feature_names = list(data.get("feature_names", []))
        obj.classes_ = np.asarray(data.get("classes_", [0, 1, 2]), dtype=np.int64)
        return obj
