from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from finsight_ml.models.base import BaseForecaster


class LGBMForecaster(BaseForecaster):
    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 31,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            verbose=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMForecaster":
        valid_mask = y.notna()
        X_valid = X.loc[valid_mask]
        y_valid = y.loc[valid_mask].astype(int)

        unexpected = set(y_valid.unique()) - {-1, 0, 1}
        if unexpected:
            raise ValueError(
                f"Unexpected label values: {unexpected}. Expected subset of {{-1, 0, 1}}."
            )
        if y_valid.nunique() < 2:
            raise ValueError("Need at least 2 distinct classes in y to fit LGBMForecaster.")

        # y contains -1, 0, 1 -> map to 0 (SELL), 1 (HOLD), 2 (BUY)
        y_mapped = y_valid.map({-1: 0, 0: 1, 1: 2})
        self.model.fit(X_valid, y_mapped)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_probs = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        full_probs = np.zeros((len(X), 3), dtype=np.float64)
        for col_idx, cls_id in enumerate(self.model.classes_):
            full_probs[:, int(cls_id)] = raw_probs[:, col_idx]
        return full_probs

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, p)

    @classmethod
    def load(cls, path: Path | str) -> "LGBMForecaster":
        loaded_model = joblib.load(path)
        params = loaded_model.get_params()
        forecaster = cls(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 5)),
            num_leaves=int(params.get("num_leaves", 31)),
            random_state=int(params.get("random_state", 42)),
        )
        forecaster.model = loaded_model
        return forecaster
