from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from finsight_ml.models.base import BaseForecaster


class LGBMForecaster(BaseForecaster):
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        max_depth: int = 5,
        num_leaves: int = 31,
        min_child_samples: int = 40,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.best_iteration: int | None = None

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=1,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    @staticmethod
    def _prepare_xy(
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]:
        valid_mask = y.notna()
        X_valid = X.loc[valid_mask]
        y_valid = y.loc[valid_mask].astype(int)

        unexpected = set(y_valid.unique()) - {-1, 0, 1}
        if unexpected:
            raise ValueError(
                f"Unexpected label values: {unexpected}. Expected subset of {{-1, 0, 1}}."
            )
        y_mapped = y_valid.map({-1: 0, 0: 1, 1: 2})
        w_valid = (
            np.asarray(sample_weight, dtype=np.float64)[valid_mask.to_numpy(dtype=bool)]
            if sample_weight is not None
            else None
        )
        return X_valid, y_mapped, w_valid

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int | None = None,
    ) -> "LGBMForecaster":
        X_valid, y_mapped, w_valid = self._prepare_xy(X, y, sample_weight)
        if y_mapped.nunique() < 2:
            raise ValueError("Need at least 2 distinct classes in y to fit LGBMForecaster.")

        self.feature_names = list(X_valid.columns)
        fit_kwargs: dict[str, Any] = {}
        if w_valid is not None:
            fit_kwargs["sample_weight"] = w_valid

        if eval_set is not None:
            X_val_raw, y_val_raw = eval_set
            X_val, y_val_mapped, _ = self._prepare_xy(X_val_raw, y_val_raw)
            if len(X_val) > 0 and set(y_val_mapped.unique()).issubset(set(y_mapped.unique())):
                fit_kwargs["eval_X"] = X_val[self.feature_names]
                fit_kwargs["eval_y"] = y_val_mapped
                if early_stopping_rounds is not None and early_stopping_rounds > 0:
                    fit_kwargs["callbacks"] = [
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                    ]

        self.model.fit(X_valid[self.feature_names], y_mapped, **fit_kwargs)
        best_iter = getattr(self.model, "best_iteration_", None)
        self.best_iteration = int(best_iter) if best_iter and int(best_iter) > 0 else None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_in = X[self.feature_names] if self.feature_names else X
        raw_probs = np.asarray(self.model.predict_proba(X_in), dtype=np.float64)
        full_probs = np.zeros((len(X_in), 3), dtype=np.float64)
        for col_idx, cls_id in enumerate(self.model.classes_):
            full_probs[:, int(cls_id)] = raw_probs[:, col_idx]
        return full_probs

    def explain_top_k(
        self,
        X: pd.DataFrame,
        top_k: int = 5,
        target_signals: np.ndarray | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Native exact TreeSHAP via LightGBM `pred_contrib=True`.

        Returns per-row top-k features ranked by |contribution|.
        NOTE: contributions are on the **raw model log-odds margin scale**
        (before probability calibration) for the target/predicted class.
        """
        if X.empty:
            return []

        X_in = X[self.feature_names] if self.feature_names else X
        feat_names = list(X_in.columns)
        n_feats = len(feat_names)
        classes = np.asarray(self.model.classes_, dtype=np.int64)
        n_classes = len(classes)

        if target_signals is None:
            probs = self.predict_proba(X_in)
            target_cls_ids = np.argmax(probs, axis=1)
        else:
            # External signal in {-1, 0, 1} -> mapped {0, 1, 2}
            target_cls_ids = np.asarray(target_signals, dtype=np.int64) + 1

        raw_contrib = np.asarray(
            self.model.booster_.predict(X_in, pred_contrib=True), dtype=np.float64
        )
        cls_to_slot = {int(cls_id): slot for slot, cls_id in enumerate(classes)}

        explanations: list[list[dict[str, Any]]] = []
        for row_idx in range(len(X_in)):
            tgt_cls = int(target_cls_ids[row_idx])
            if n_classes == 2:
                # Binary booster returns shape (N, n_feats + 1) for classes_[1]
                pos_cls = int(classes[1])
                contrib_vec = raw_contrib[row_idx, :n_feats]
                if tgt_cls != pos_cls:
                    contrib_vec = -contrib_vec
            else:
                slot = cls_to_slot.get(tgt_cls)
                if slot is None:
                    slot = int(np.argmax(self.predict_proba(X_in.iloc[[row_idx]])[0]))
                    slot = cls_to_slot.get(slot, 0)
                stride = n_feats + 1
                start = slot * stride
                contrib_vec = raw_contrib[row_idx, start : start + n_feats]

            ranked_idx = np.argsort(np.abs(contrib_vec))[::-1][:top_k]
            explanations.append(
                [
                    {
                        "feature": feat_names[int(j)],
                        "contribution": round(float(contrib_vec[int(j)]), 6),
                        "scale": "raw_log_odds",
                    }
                    for j in ranked_idx
                ]
            )
        return explanations

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "best_iteration": self.best_iteration,
            },
            p,
        )

    @classmethod
    def load(cls, path: Path | str) -> "LGBMForecaster":
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            loaded_model = loaded["model"]
            feature_names = list(loaded.get("feature_names", []))
            best_iteration = loaded.get("best_iteration")
        else:
            loaded_model = loaded
            feature_names = []
            best_iteration = None

        params = loaded_model.get_params()
        forecaster = cls(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_depth=int(params.get("max_depth", 5)),
            num_leaves=int(params.get("num_leaves", 31)),
            min_child_samples=int(params.get("min_child_samples", 40)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.1)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            class_weight=params.get("class_weight", "balanced"),
            random_state=int(params.get("random_state", 42)),
        )
        forecaster.model = loaded_model
        forecaster.feature_names = feature_names
        forecaster.best_iteration = best_iteration
        return forecaster
