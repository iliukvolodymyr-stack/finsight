from pathlib import Path

import numpy as np
import pandas as pd
from finsight_ml.models.baselines import LogisticRegressionBaseline, MajorityBaseline
from finsight_ml.models.calibration import ProbabilityCalibrator
from finsight_ml.models.conformal import VolatilityNormalizedConformal
from finsight_ml.models.lgbm import LGBMForecaster
from finsight_ml.training.pipeline import TrainingConfig, train_and_evaluate
from finsight_ml.validation.metrics import (
    apply_signal_decision_rule,
    bootstrap_coverage_ci,
    compute_classification_metrics,
)


def _make_hourly_market(n: int = 12000, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01 00:00:00", periods=n, freq="h", tz="UTC")
    # Mild volatility clustering + small drift
    vol = 0.008 + 0.004 * np.abs(np.sin(np.arange(n) / 180.0))
    rets = rng.normal(0.0, vol)
    close = 40000.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = 40000.0
    spread = np.abs(rng.normal(0.0, 0.005, size=n)) * close
    high = np.maximum(open_p, close) + spread
    low = np.maximum(1.0, np.minimum(open_p, close) - spread)
    volume = rng.lognormal(mean=5.5, sigma=0.7, size=n)
    return pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_calibrator_temperature_platt_and_isotonic_and_conformal() -> None:
    rng = np.random.default_rng(7)
    # Default Temperature Scaling preserves argmax & optimizes val NLL
    raw_temp = rng.dirichlet([3.0, 3.0, 3.0], size=400)
    y_temp = rng.choice([-1, 0, 1], size=400, p=[0.3, 0.4, 0.3])
    cal_temp = ProbabilityCalibrator().fit(raw_temp, y_temp)
    assert cal_temp.method == "temperature"
    p_temp = cal_temp.transform(raw_temp)
    assert p_temp.shape == (400, 3)
    np.testing.assert_allclose(p_temp.sum(axis=1), 1.0, atol=1e-9)
    assert np.array_equal(np.argmax(p_temp, axis=1), np.argmax(raw_temp, axis=1))

    # Small val (<1000) with method="auto" -> Platt
    raw_small = rng.dirichlet([2.0, 3.0, 2.0], size=300)
    y_small = rng.choice([-1, 0, 1], size=300, p=[0.3, 0.4, 0.3])
    cal_platt = ProbabilityCalibrator(method="auto", isotonic_min_samples=1000).fit(
        raw_small, y_small
    )
    assert cal_platt.method == "platt"
    p_platt = cal_platt.transform(raw_small)
    assert p_platt.shape == (300, 3)
    np.testing.assert_allclose(p_platt.sum(axis=1), 1.0, atol=1e-9)
    assert (p_platt >= 0.0).all()

    # Large val (>=1000) with method="auto" -> Isotonic
    raw_large = rng.dirichlet([2.0, 3.0, 2.0], size=1200)
    y_large = rng.choice([-1, 0, 1], size=1200, p=[0.3, 0.4, 0.3])
    cal_iso = ProbabilityCalibrator(method="auto", isotonic_min_samples=1000).fit(
        raw_large, y_large
    )
    assert cal_iso.method == "isotonic"
    p_iso = cal_iso.transform(raw_large)
    np.testing.assert_allclose(p_iso.sum(axis=1), 1.0, atol=1e-9)

    # Conformal volatility-normalized scales width linearly with sigma in exp space
    sig_val = rng.uniform(0.005, 0.02, size=500)
    fwd_ret = rng.normal(0.0, 1.0, size=500) * sig_val * np.sqrt(24.0)
    conf = VolatilityNormalizedConformal(alpha=0.20, horizon_bars=24).fit(fwd_ret, sig_val)
    iv_low_vol = conf.predict_interval(np.array([100.0]), np.array([0.005]))
    iv_high_vol = conf.predict_interval(np.array([100.0]), np.array([0.025]))
    w_low = float(iv_low_vol.price_high[0] - iv_low_vol.price_low[0])
    w_high = float(iv_high_vol.price_high[0] - iv_high_vol.price_low[0])
    assert w_high > 3.0 * w_low

    # Decision rule forces HOLD when margin < min_margin or max < entry_threshold
    probs = np.array(
        [
            [0.10, 0.15, 0.75],  # clear BUY
            [0.44, 0.06, 0.50],  # max < 0.55 & margin < 0.10 -> HOLD
            [0.35, 0.10, 0.55],  # exact threshold 0.55 & margin 0.20 >= 0.10 -> BUY
            [0.62, 0.20, 0.18],  # clear SELL
        ]
    )
    sigs = apply_signal_decision_rule(probs, entry_threshold=0.55, min_margin=0.10)
    assert sigs.tolist() == [1, 0, 1, -1]

    ci = bootstrap_coverage_ci(np.array([1, 1, 0, 1, 1] * 40), block_size=24, n_resamples=200)
    assert 0.0 <= ci["ci_low"] <= ci["mean"] <= ci["ci_high"] <= 1.0


def test_lgbm_native_treeshap_3class_and_2class() -> None:
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(180, 6)), columns=[f"f{i}" for i in range(6)])

    # 3-class model
    y3 = pd.Series(np.where(X["f0"] > 0.4, 1, np.where(X["f0"] < -0.4, -1, 0)))
    m3 = LGBMForecaster(n_estimators=20, min_child_samples=5, random_state=42).fit(X, y3)
    expl3 = m3.explain_top_k(X.iloc[:4], top_k=5)
    assert len(expl3) == 4
    assert len(expl3[0]) == 5
    assert all(item["scale"] == "raw_log_odds" for item in expl3[0])
    assert expl3[0][0]["feature"] in X.columns

    # 2-class subset (missing SELL) must also produce clean TreeSHAP without crashing
    y2 = pd.Series(np.where(X["f1"] > 0.0, 1, 0))
    m2 = LGBMForecaster(n_estimators=20, min_child_samples=5, random_state=42).fit(X, y2)
    expl2 = m2.explain_top_k(X.iloc[:3], top_k=4)
    assert len(expl2) == 3
    assert len(expl2[0]) == 4


def test_baselines_save_load_and_metrics() -> None:
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(rng.choice([-1, 0, 1], size=120, p=[0.25, 0.5, 0.25]))

    maj = MajorityBaseline().fit(X, y)
    lr = LogisticRegressionBaseline().fit(X, y)
    assert maj.predict_proba(X).shape == (120, 3)
    assert lr.predict_proba(X).shape == (120, 3)
    np.testing.assert_allclose(lr.predict_proba(X).sum(axis=1), 1.0, atol=1e-9)

    metrics = compute_classification_metrics(y.to_numpy(), lr.predict_proba(X))
    assert "macro_f1" in metrics and "ece_10" in metrics and "mcc" in metrics


def test_end_to_end_training_pipeline_and_artifacts(tmp_path: Path) -> None:
    # 12,000 hourly bars (~16.4 months) with 8m train / 2m val / 2m test -> >= 3 folds
    df = _make_hourly_market(n=12000, seed=101)
    cfg = TrainingConfig(
        name="lgbm_core",
        semver="0.1.0-test",
        symbol="TESTBTC",
        timeframe="1h",
        horizon_bars=24,
        status_target="production",
        pt_mult=1.0,
        sl_mult=1.0,
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

    res = train_and_evaluate(df, cfg, artifacts_dir=tmp_path)
    assert len(res.fold_reports) >= 3
    assert res.oos_summary["n_oos_samples"] > 1000
    assert (res.artifact_dir / "model.joblib").exists()
    assert (res.artifact_dir / "calibrator.joblib").exists()
    assert (res.artifact_dir / "conformal.json").exists()
    assert (res.artifact_dir / "feature_schema.json").exists()
    assert (res.artifact_dir / "metrics.json").exists()
    assert (res.artifact_dir / "dataset_manifest.json").exists()
    assert (res.artifact_dir / "model_card.md").exists()

    # Verify saved artifacts reload cleanly and produce TreeSHAP + conformal interval
    loaded_model = LGBMForecaster.load(res.artifact_dir / "model.joblib")
    loaded_cal = ProbabilityCalibrator.load(res.artifact_dir / "calibrator.joblib")
    loaded_conf = VolatilityNormalizedConformal.load(res.artifact_dir / "conformal.json")
    assert loaded_conf.horizon_bars == 24
    assert loaded_cal.method in {"temperature", "platt", "isotonic"}
    assert len(loaded_model.feature_names) == 32
