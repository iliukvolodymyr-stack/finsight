from typing import Any

import numpy as np
import pandas as pd

from finsight_ml.features.pipeline import FeaturePipeline
from finsight_ml.labels.triple_barrier import compute_ewm_volatility
from finsight_ml.models.conformal import VolatilityNormalizedConformal
from finsight_ml.models.quantile import (
    NaivePersistenceInterval,
    QuantileLGBMForecaster,
    VolatilityLGBMForecaster,
    compute_forward_realized_volatility,
    evaluate_interval_forecast,
    evaluate_volatility_forecast,
)
from finsight_ml.training.pipeline import TrainingConfig
from finsight_ml.validation.walk_forward import purged_walk_forward_splits


def evaluate_distribution_and_vol_walk_forward(
    df: pd.DataFrame,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    """
    Scientific Step-A Walk-Forward Benchmark answering interval & volatility predictability:
    1. Compares 80% prediction corridor methods on unseen OOS folds:
       - `naive_persistence`: static historical quantiles (no features / no sigma adaptation)
       - `vol_conformal`: volatility-normalized split conformal (`sigma_t * sqrt(H)`)
       - `quantile_lgbm_raw`: direct Quantile LightGBM (`0.10 / 0.50 / 0.90`)
       - `quantile_lgbm_cqr`: Conformalized Quantile Regression (`Quantile LGBM + val CQR`)
       Evaluated by Overall Coverage, Q1 Low-Vol / Q4 High-Vol Coverage, Winkler Score, and Pinball Loss.
    2. Compares 24h forward Realized Volatility forecasting:
       - `vol_persistence`: backward EWMA volatility (`sigma_t`)
       - `vol_lgbm`: LightGBM Regressor predicting `RV_{t -> t+H}`
       Evaluated by RMSE, MAE, QLIKE, and log-volatility R^2.
    """
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    pipeline = FeaturePipeline()
    feats = pipeline.compute(df)
    close = df["close"].astype("float64")
    sigma_ewm = compute_ewm_volatility(close, span=24, min_periods=24)

    future_close = close.shift(-cfg.horizon_bars)
    fwd_logret = np.log(future_close / close)
    fwd_rv = compute_forward_realized_volatility(close, horizon_bars=cfg.horizon_bars)

    # Exact H-bar horizon timestamps t1 for strict purge/embargo
    idx_series = pd.Series(df.index, index=df.index)
    t1_series = idx_series.shift(-cfg.horizon_bars)

    valid_mask = (
        (~feats.isna().any(axis=1))
        & fwd_logret.notna()
        & fwd_rv.notna()
        & sigma_ewm.notna()
        & t1_series.notna()
    )
    valid_pos = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
    if len(valid_pos) < 300:
        raise ValueError(f"Insufficient valid bars for distribution benchmark: {len(valid_pos)}")

    X = feats.iloc[valid_pos]
    y_ret = fwd_logret.iloc[valid_pos]
    y_rv = fwd_rv.iloc[valid_pos]
    sig = sigma_ewm.iloc[valid_pos].to_numpy(dtype=np.float64)
    t1 = t1_series.iloc[valid_pos]

    folds = purged_walk_forward_splits(
        timestamps=X.index,
        t1=t1,
        min_train_months=cfg.min_train_months,
        val_months=cfg.val_months,
        test_months=cfg.test_months,
        step_months=cfg.step_months,
        embargo_bars=cfg.embargo_bars,
    )
    if not folds:
        raise ValueError("Purged walk-forward produced 0 folds.")

    buckets: dict[str, list[np.ndarray]] = {
        "y_ret": [],
        "y_rv": [],
        "sig_persist": [],
        "sig_lgbm": [],
        "naive_lo": [],
        "naive_hi": [],
        "conf_lo": [],
        "conf_hi": [],
        "qlgbm_raw_lo": [],
        "qlgbm_raw_hi": [],
        "cqr_lo": [],
        "cqr_hi": [],
    }

    hp = cfg.hyperparameters
    for fold in folds:
        tr, va, te = fold.train_indices, fold.val_indices, fold.test_indices
        X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X.iloc[te]
        y_ret_tr, y_ret_va, y_ret_te = y_ret.iloc[tr], y_ret.iloc[va], y_ret.iloc[te]
        y_rv_tr, y_rv_va, y_rv_te = y_rv.iloc[tr], y_rv.iloc[va], y_rv.iloc[te]

        # 1. Naive Persistence corridor (fit on train + val)
        naive_iv = NaivePersistenceInterval(alpha=cfg.conformal_alpha).fit(
            np.concatenate([y_ret_tr.to_numpy(), y_ret_va.to_numpy()])
        )
        n_lo, n_hi = naive_iv.predict_logret_bounds(len(te))

        # 2. Volatility-normalized Conformal corridor (fit on val)
        vol_conf = VolatilityNormalizedConformal(
            alpha=cfg.conformal_alpha, horizon_bars=cfg.horizon_bars
        ).fit(y_ret_va.to_numpy(), sig[va])
        scale_te = np.maximum(sig[te], vol_conf.min_sigma) * float(np.sqrt(cfg.horizon_bars))
        c_lo = vol_conf.z_low * scale_te
        c_hi = vol_conf.z_high * scale_te

        # 3. Quantile LightGBM (0.10 / 0.50 / 0.90) + CQR calibration on val
        qlgbm = QuantileLGBMForecaster(
            alpha=cfg.conformal_alpha,
            n_estimators=int(hp.get("n_estimators", 250)),
            learning_rate=float(hp.get("learning_rate", 0.03)),
            max_depth=int(hp.get("max_depth", 5)),
            num_leaves=int(hp.get("num_leaves", 31)),
            min_child_samples=int(hp.get("min_child_samples", 40)),
            random_state=int(hp.get("random_state", 42)),
        ).fit(
            X_tr,
            y_ret_tr,
            eval_set=(X_va, y_ret_va),
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        q_raw_lo, _, q_raw_hi = qlgbm.predict_logret_quantiles(X_te)
        qlgbm.calibrate_cqr(X_va, y_ret_va)
        cqr_lo, _, cqr_hi = qlgbm.predict_logret_quantiles(X_te)

        # 4. Forward Volatility Regressor vs Persistence
        vol_lgbm = VolatilityLGBMForecaster(
            n_estimators=int(hp.get("n_estimators", 250)),
            learning_rate=float(hp.get("learning_rate", 0.03)),
            max_depth=4,
            num_leaves=15,
            min_child_samples=int(hp.get("min_child_samples", 40)),
            random_state=int(hp.get("random_state", 42)),
        ).fit(
            X_tr,
            y_rv_tr,
            eval_set=(X_va, y_rv_va),
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        pred_rv_te = vol_lgbm.predict_sigma(X_te)

        buckets["y_ret"].append(y_ret_te.to_numpy(dtype=np.float64))
        buckets["y_rv"].append(y_rv_te.to_numpy(dtype=np.float64))
        buckets["sig_persist"].append(sig[te])
        buckets["sig_lgbm"].append(pred_rv_te)
        buckets["naive_lo"].append(n_lo)
        buckets["naive_hi"].append(n_hi)
        buckets["conf_lo"].append(c_lo)
        buckets["conf_hi"].append(c_hi)
        buckets["qlgbm_raw_lo"].append(q_raw_lo)
        buckets["qlgbm_raw_hi"].append(q_raw_hi)
        buckets["cqr_lo"].append(cqr_lo)
        buckets["cqr_hi"].append(cqr_hi)

    y_ret_all = np.concatenate(buckets["y_ret"])
    y_rv_all = np.concatenate(buckets["y_rv"])
    sig_all = np.concatenate(buckets["sig_persist"])

    intervals = {
        "naive_persistence": evaluate_interval_forecast(
            y_ret_all,
            np.concatenate(buckets["naive_lo"]),
            np.concatenate(buckets["naive_hi"]),
            sigma=sig_all,
            alpha=cfg.conformal_alpha,
        ),
        "vol_conformal": evaluate_interval_forecast(
            y_ret_all,
            np.concatenate(buckets["conf_lo"]),
            np.concatenate(buckets["conf_hi"]),
            sigma=sig_all,
            alpha=cfg.conformal_alpha,
        ),
        "quantile_lgbm_raw": evaluate_interval_forecast(
            y_ret_all,
            np.concatenate(buckets["qlgbm_raw_lo"]),
            np.concatenate(buckets["qlgbm_raw_hi"]),
            sigma=sig_all,
            alpha=cfg.conformal_alpha,
        ),
        "quantile_lgbm_cqr": evaluate_interval_forecast(
            y_ret_all,
            np.concatenate(buckets["cqr_lo"]),
            np.concatenate(buckets["cqr_hi"]),
            sigma=sig_all,
            alpha=cfg.conformal_alpha,
        ),
    }

    volatility = {
        "ewm_persistence": evaluate_volatility_forecast(y_rv_all, sig_all),
        "vol_lgbm": evaluate_volatility_forecast(y_rv_all, np.concatenate(buckets["sig_lgbm"])),
    }

    return {
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "n_folds": len(folds),
        "n_oos_samples": int(len(y_ret_all)),
        "intervals": intervals,
        "volatility": volatility,
    }
