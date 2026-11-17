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
        # y contains -1, 0, 1 -> map to 0, 1, 2 for multiclass
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})
        self.model.fit(X, y_mapped)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X))

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, p)

    @classmethod
    def load(cls, path: Path | str) -> "LGBMForecaster":
        forecaster = cls()
        forecaster.model = joblib.load(path)
        return forecaster
