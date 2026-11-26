from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileIntervalPrediction:
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    price_low: np.ndarray
    price_mid: np.ndarray
    price_high: np.ndarray


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Standard asymmetric pinball (quantile) loss for quantile level `alpha`."""
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    loss = np.maximum(alpha * err, (alpha - 1.0) * err)
    return float(np.mean(loss))


def winkler_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.20,
) -> float:
    """
    Proper scoring rule for prediction intervals at nominal coverage (1 - alpha).
    Rewards narrow intervals while heavily penalizing misses by 2/alpha.
    Lower score is strictly better.
    """
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    width = hi - lo
    miss_low = np.where(y < lo, (2.0 / alpha) * (lo - y), 0.0)
    miss_high = np.where(y > hi, (2.0 / alpha) * (y - hi), 0.0)
    return float(np.mean(width + miss_low + miss_high))


def evaluate_interval_forecast(
    y_true_logret: np.ndarray,
    lower_logret: np.ndarray,
    upper_logret: np.ndarray,
    sigma: np.ndarray | None = None,
    alpha: float = 0.20,
) -> dict[str, Any]:
    """
    Evaluate interval quality overall and across volatility quartiles (Q1 calm vs Q4 storm)
    to test whether adaptive models hold up against naive persistence during regime shifts.
    """
    y = np.asarray(y_true_logret, dtype=np.float64)
    lo = np.asarray(lower_logret, dtype=np.float64)
    hi = np.asarray(upper_logret, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(valid):
        return {"n": 0, "coverage": 0.0, "winkler_score": 0.0, "mean_width_logret": 0.0}

    yv, lov, hiv = y[valid], lo[valid], hi[valid]
    covered = (yv >= lov) & (yv <= hiv)
    widths = hiv - lov
    winkler = winkler_interval_score(yv, lov, hiv, alpha=alpha)
    pb_low = pinball_loss(yv, lov, alpha=alpha / 2.0)
    pb_high = pinball_loss(yv, hiv, alpha=1.0 - alpha / 2.0)

    out: dict[str, Any] = {
        "n": int(len(yv)),
        "coverage": float(np.mean(covered)),
        "mean_width_logret": float(np.mean(widths)),
        "winkler_score": winkler,
        "pinball_mean": float(0.5 * (pb_low + pb_high)),
    }

    if sigma is not None:
        sv = np.asarray(sigma, dtype=np.float64)[valid]
        fin_sig = np.isfinite(sv)
        if int(np.sum(fin_sig)) >= 20:
            s_fin = sv[fin_sig]
            cov_fin = covered[fin_sig]
            q25, q75 = np.quantile(s_fin, [0.25, 0.75])
            out["low_vol_q1_cov"] = float(np.mean(cov_fin[s_fin <= q25]))
            out["high_vol_q4_cov"] = float(np.mean(cov_fin[s_fin >= q75]))

    return out


class NaivePersistenceInterval:
    """
    Unconditional historical quantile corridor (naive baseline without features or adaptive vol).
    Directly tests whether adaptive conformal / quantile models beat market base-rate width.
    """

    def __init__(self, alpha: float = 0.20) -> None:
        self.alpha = float(alpha)
        self.q_low: float = -0.02
        self.q_high: float = 0.02

    def fit(self, forward_log_returns: np.ndarray) -> "NaivePersistenceInterval":
        ret = np.asarray(forward_log_returns, dtype=np.float64)
        ret = ret[np.isfinite(ret)]
        if len(ret) < 10:
            raise ValueError("Need at least 10 finite returns for NaivePersistenceInterval.")
        self.q_low = float(np.quantile(ret, self.alpha / 2.0))
        self.q_high = float(np.quantile(ret, 1.0 - self.alpha / 2.0))
        return self

    def predict_logret_bounds(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.full(n, self.q_low, dtype=np.float64), np.full(n, self.q_high, dtype=np.float64)


class QuantileLGBMForecaster:
    """
    Three-model Quantile LightGBM (`alpha = 0.10, 0.50, 0.90`) with
    Conformalized Quantile Regression (CQR) margin calibration on validation folds.
    """

    def __init__(
        self,
        alpha: float = 0.20,
        n_estimators: int = 250,
        learning_rate: float = 0.03,
        max_depth: int = 5,
        num_leaves: int = 31,
        min_child_samples: int = 40,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.alpha = float(alpha)
        self.quantile_levels: tuple[float, float, float] = (
            self.alpha / 2.0,
            0.5,
            1.0 - self.alpha / 2.0,
        )
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": 1,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.feature_names: list[str] = []
        self.cqr_delta: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y_forward_logret: pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = 40,
    ) -> "QuantileLGBMForecaster":
        valid = y_forward_logret.notna() & np.isfinite(y_forward_logret.to_numpy(dtype=np.float64))
        X_tr = X.loc[valid]
        y_tr = y_forward_logret.loc[valid].astype("float64")
        if len(X_tr) < 30:
            raise ValueError("Insufficient training samples for QuantileLGBMForecaster.")

        self.feature_names = list(X_tr.columns)
        w_tr = (
            np.asarray(sample_weight, dtype=np.float64)[valid.to_numpy(dtype=bool)]
            if sample_weight is not None
            else None
        )

        eval_X_arg: pd.DataFrame | None = None
        eval_y_arg: pd.Series | None = None
        if eval_set is not None:
            Xv_raw, yv_raw = eval_set
            v_mask = yv_raw.notna() & np.isfinite(yv_raw.to_numpy(dtype=np.float64))
            if int(v_mask.sum()) >= 20:
                eval_X_arg = Xv_raw.loc[v_mask, self.feature_names]
                eval_y_arg = yv_raw.loc[v_mask].astype("float64")

        self.models = {}
        self.cqr_delta = 0.0
        for q in self.quantile_levels:
            reg = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params)
            fit_kw: dict[str, Any] = {}
            if w_tr is not None:
                fit_kw["sample_weight"] = w_tr
            if eval_X_arg is not None and eval_y_arg is not None:
                fit_kw["eval_X"] = eval_X_arg
                fit_kw["eval_y"] = eval_y_arg
                if early_stopping_rounds > 0:
                    fit_kw["callbacks"] = [
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                    ]
            reg.fit(X_tr[self.feature_names], y_tr, **fit_kw)
            self.models[q] = reg

        return self

    def calibrate_cqr(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Conformalized Quantile Regression (Romano et al., 2019):
        Adjusts raw quantile intervals by scalar `delta` on validation residuals so
        empirical coverage aligns cleanly with `1 - alpha`.
        """
        valid = y_val.notna() & np.isfinite(y_val.to_numpy(dtype=np.float64))
        if int(valid.sum()) < 20:
            self.cqr_delta = 0.0
            return 0.0
        q10, _, q90 = self._predict_raw_sorted(X_val.loc[valid])
        yv = y_val.loc[valid].to_numpy(dtype=np.float64)
        scores = np.maximum(q10 - yv, yv - q90)
        n = len(scores)
        level = min(1.0, max(0.0, (1.0 - self.alpha) * (n + 1.0) / n))
        self.cqr_delta = float(np.quantile(scores, level))
        return self.cqr_delta

    def _predict_raw_sorted(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_in = X[self.feature_names] if self.feature_names else X
        q_lo, q_mid, q_hi = self.quantile_levels
        preds = np.column_stack(
            [
                np.asarray(self.models[q_lo].predict(X_in), dtype=np.float64),
                np.asarray(self.models[q_mid].predict(X_in), dtype=np.float64),
                np.asarray(self.models[q_hi].predict(X_in), dtype=np.float64),
            ]
        )
        # Prevent quantile crossing
        preds.sort(axis=1)
        return preds[:, 0], preds[:, 1], preds[:, 2]

    def predict_logret_quantiles(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q10, q50, q90 = self._predict_raw_sorted(X)
        lo = np.minimum(q10 - self.cqr_delta, q50)
        hi = np.maximum(q90 + self.cqr_delta, q50)
        return lo, q50, hi

    def predict_prices(
        self, X: pd.DataFrame, close_prices: np.ndarray
    ) -> QuantileIntervalPrediction:
        q10, q50, q90 = self.predict_logret_quantiles(X)
        c = np.asarray(close_prices, dtype=np.float64)
        return QuantileIntervalPrediction(
            q10=q10,
            q50=q50,
            q90=q90,
            price_low=c * np.exp(q10),
            price_mid=c * np.exp(q50),
            price_high=c * np.exp(q90),
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "alpha": self.alpha,
                "quantile_levels": self.quantile_levels,
                "params": self.params,
                "models": self.models,
                "feature_names": self.feature_names,
                "cqr_delta": self.cqr_delta,
            },
            p,
        )

    @classmethod
    def load(cls, path: Path | str) -> "QuantileLGBMForecaster":
        data = joblib.load(path)
        p = data["params"]
        obj = cls(
            alpha=float(data["alpha"]),
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            num_leaves=int(p["num_leaves"]),
            min_child_samples=int(p["min_child_samples"]),
            subsample=float(p["subsample"]),
            colsample_bytree=float(p["colsample_bytree"]),
            reg_alpha=float(p["reg_alpha"]),
            reg_lambda=float(p["reg_lambda"]),
            random_state=int(p["random_state"]),
        )
        obj.models = data["models"]
        obj.feature_names = list(data["feature_names"])
        obj.cqr_delta = float(data.get("cqr_delta", 0.0))
        return obj


def compute_forward_realized_volatility(close: pd.Series, horizon_bars: int = 24) -> pd.Series:
    """
    Strictly forward-looking realized volatility target over bars [t+1 .. t+H]:
        RV_{t -> t+H} = sqrt(mean_{j=1..H} r_{t+j}^2)
    Used ONLY as regression target `y_vol` (never as a feature!).
    Last `horizon_bars` rows remain NaN.
    """
    c = close.astype("float64")
    r1 = np.log(c / c.shift(1))
    r2 = r1**2
    # At position i, future window [i+1 .. i+H] aligns with rolling mean shifted by -H
    fwd_ms = r2.rolling(window=horizon_bars, min_periods=horizon_bars).mean().shift(-horizon_bars)
    return np.sqrt(fwd_ms)


def evaluate_volatility_forecast(
    y_true_sigma: np.ndarray,
    y_pred_sigma: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate predicted 1-bar-equivalent sigma against realized forward sigma
    using RMSE, MAE, QLIKE (econometric variance scoring rule), and log-vol R^2.
    """
    yt = np.asarray(y_true_sigma, dtype=np.float64)
    yp = np.asarray(y_pred_sigma, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp) & (yt > 1e-8) & (yp > 1e-8)
    if int(np.sum(valid)) < 5:
        return {"n": 0, "rmse": 0.0, "mae": 0.0, "qlike": 0.0, "log_r2": 0.0}

    ytv, ypv = yt[valid], yp[valid]
    rmse = float(np.sqrt(np.mean((ytv - ypv) ** 2)))
    mae = float(np.mean(np.abs(ytv - ypv)))

    var_t = ytv**2
    var_p = ypv**2
    ratio = np.clip(var_t / np.maximum(var_p, 1e-16), 1e-12, 1e6)
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))

    log_t = np.log(ytv)
    log_p = np.log(ypv)
    ss_res = float(np.sum((log_t - log_p) ** 2))
    ss_tot = float(np.sum((log_t - np.mean(log_t)) ** 2))
    log_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "n": int(len(ytv)),
        "rmse": rmse,
        "mae": mae,
        "qlike": qlike,
        "log_r2": float(log_r2),
    }


class VolatilityLGBMForecaster:
    """
    LightGBM regressor predicting forward H-bar realized volatility in log-sigma space.
    Strictly positive on exp() transform and naturally robust to heavy right tails.
    """

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 40,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=1,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y_forward_sigma: pd.Series,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = 40,
    ) -> "VolatilityLGBMForecaster":
        y_arr = y_forward_sigma.to_numpy(dtype=np.float64)
        valid = np.isfinite(y_arr) & (y_arr > 1e-8)
        X_tr = X.iloc[np.flatnonzero(valid)]
        y_tr_log = np.log(y_arr[valid])
        if len(X_tr) < 30:
            raise ValueError("Insufficient samples for VolatilityLGBMForecaster.")

        self.feature_names = list(X_tr.columns)
        fit_kw: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kw["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)[valid]

        if eval_set is not None:
            Xv_raw, yv_raw = eval_set
            yv_arr = yv_raw.to_numpy(dtype=np.float64)
            v_valid = np.isfinite(yv_arr) & (yv_arr > 1e-8)
            if int(np.sum(v_valid)) >= 20:
                fit_kw["eval_X"] = Xv_raw.iloc[np.flatnonzero(v_valid)][self.feature_names]
                fit_kw["eval_y"] = np.log(yv_arr[v_valid])
                if early_stopping_rounds > 0:
                    fit_kw["callbacks"] = [
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
                    ]

        self.model.fit(X_tr[self.feature_names], y_tr_log, **fit_kw)
        return self

    def predict_sigma(self, X: pd.DataFrame) -> np.ndarray:
        X_in = X[self.feature_names] if self.feature_names else X
        log_pred = np.asarray(self.model.predict(X_in), dtype=np.float64)
        return np.asarray(np.exp(np.clip(log_pred, -12.0, 2.0)), dtype=np.float64)
