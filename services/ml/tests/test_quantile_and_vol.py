from pathlib import Path

import numpy as np
import pandas as pd
from finsight_ml.models.quantile import (
    NaivePersistenceInterval,
    QuantileLGBMForecaster,
    VolatilityLGBMForecaster,
    compute_forward_realized_volatility,
    evaluate_interval_forecast,
    winkler_interval_score,
)
from finsight_ml.training.distribution import evaluate_distribution_and_vol_walk_forward
from finsight_ml.training.pipeline import TrainingConfig


def _make_synthetic_regimes(n: int = 12000, seed: int = 71) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    # Strong volatility regimes so adaptive intervals should decisively beat naive static width
    vol = np.where((np.arange(n) // 240) % 2 == 0, 0.004, 0.020)
    rets = rng.normal(0.0, vol)
    close = 50000.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = 50000.0
    high = np.maximum(open_p, close) * (1.0 + vol * 0.4)
    low = np.maximum(1.0, np.minimum(open_p, close) * (1.0 - vol * 0.4))
    volume = rng.lognormal(6.0, 0.5, size=n)
    return pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_quantile_lgbm_cqr_and_persistence_scoring(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f1", "f2", "f3", "f4"])
    sigma = np.where(X["f1"] > 0, 0.03, 0.005)
    y = pd.Series(rng.normal(0.0, sigma), index=X.index)

    q_model = QuantileLGBMForecaster(alpha=0.20, n_estimators=40, min_child_samples=10).fit(
        X.iloc[:250], y.iloc[:250]
    )
    delta = q_model.calibrate_cqr(X.iloc[250:320], y.iloc[250:320])
    assert np.isfinite(delta)

    iv = q_model.predict_prices(X.iloc[320:], np.full(80, 100.0))
    assert np.all(iv.q10 <= iv.q50 + 1e-12)
    assert np.all(iv.q50 <= iv.q90 + 1e-12)
    assert np.all(iv.price_low <= iv.price_mid + 1e-9)
    assert np.all(iv.price_mid <= iv.price_high + 1e-9)

    save_p = tmp_path / "q_model.joblib"
    q_model.save(save_p)
    reloaded = QuantileLGBMForecaster.load(save_p)
    iv2 = reloaded.predict_prices(X.iloc[320:], np.full(80, 100.0))
    np.testing.assert_allclose(iv2.q10, iv.q10, atol=1e-12)

    # Naive baseline vs tight oracle interval scoring: Winkler score must reward sharper well-covered intervals
    naive = NaivePersistenceInterval(alpha=0.20).fit(y.iloc[:320].to_numpy())
    n_lo, n_hi = naive.predict_logret_bounds(80)
    ev_naive = evaluate_interval_forecast(
        y.iloc[320:].to_numpy(), n_lo, n_hi, sigma=sigma[320:], alpha=0.20
    )
    ev_cqr = evaluate_interval_forecast(
        y.iloc[320:].to_numpy(), iv.q10, iv.q90, sigma=sigma[320:], alpha=0.20
    )
    assert ev_naive["winkler_score"] > 0 and ev_cqr["winkler_score"] > 0
    assert winkler_interval_score(
        np.array([0.0]), np.array([-0.01]), np.array([0.01])
    ) < winkler_interval_score(np.array([0.05]), np.array([-0.01]), np.array([0.01]))


def test_forward_volatility_target_and_regressor() -> None:
    df = _make_synthetic_regimes(n=400, seed=5)
    rv_fwd = compute_forward_realized_volatility(df["close"], horizon_bars=24)
    assert rv_fwd.iloc[-24:].isna().all()
    assert (rv_fwd.dropna() > 0).all()

    X = pd.DataFrame(
        {"lag_ret": np.log(df["close"] / df["close"].shift(1)).abs().fillna(0.01)},
        index=df.index,
    )
    vol_m = VolatilityLGBMForecaster(n_estimators=25, min_child_samples=10).fit(
        X.iloc[:300],
        rv_fwd.iloc[:300],
        eval_set=(X.iloc[300:350], rv_fwd.iloc[300:350]),
        early_stopping_rounds=5,
    )
    assert int(getattr(vol_m.model, "best_iteration_", 0) or 0) > 0
    preds = vol_m.predict_sigma(X.iloc[300:350])
    assert len(preds) == 50
    assert np.all(preds > 0.0)


def test_step_a_distribution_walk_forward_beats_naive_in_regimes() -> None:
    df = _make_synthetic_regimes(n=12000, seed=71)
    cfg = TrainingConfig(
        symbol="TESTBTC",
        timeframe="1h",
        min_train_months=8,
        val_months=2,
        test_months=2,
        embargo_bars=24,
        early_stopping_rounds=20,
        hyperparameters={
            "n_estimators": 40,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 20,
            "random_state": 42,
        },
    )
    res = evaluate_distribution_and_vol_walk_forward(df, cfg)
    assert res["n_folds"] >= 3
    ivs = res["intervals"]
    # During volatile shock quartile (Q4), naive persistence severely under-covers while adaptive beats it!
    assert (
        ivs["vol_conformal"]["high_vol_q4_cov"] > ivs["naive_persistence"]["high_vol_q4_cov"] + 0.15
    )
    assert ivs["vol_conformal"]["winkler_score"] < ivs["naive_persistence"]["winkler_score"]
    # Quantile LGBM / CQR also produces much lower Winkler score than naive static persistence
    assert ivs["quantile_lgbm_cqr"]["winkler_score"] < ivs["naive_persistence"]["winkler_score"]
