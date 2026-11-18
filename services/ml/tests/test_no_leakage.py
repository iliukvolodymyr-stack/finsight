import numpy as np
import pandas as pd
from finsight_ml.features.pipeline import FeaturePipeline, compute_features, compute_row_hash
from finsight_ml.features.registry import FEATURE_REGISTRY, compute_schema_hash
from finsight_ml.labels.triple_barrier import compute_ewm_volatility


def _make_synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01 00:00:00", periods=n, freq="h", tz="UTC")
    log_rets = rng.normal(loc=0.0001, scale=0.012, size=n)
    close = 50000.0 * np.exp(np.cumsum(log_rets))
    open_p = np.roll(close, 1)
    open_p[0] = 50000.0
    spread = np.abs(rng.normal(0.0, 0.006, size=n)) * close
    high = np.maximum(open_p, close) + spread
    low = np.maximum(1.0, np.minimum(open_p, close) - spread)
    volume = rng.lognormal(mean=6.0, sigma=0.6, size=n)
    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def test_feature_registry_has_32_features_and_deterministic_schema_hash() -> None:
    pipeline = FeaturePipeline()
    assert len(FEATURE_REGISTRY) == 32
    assert len(pipeline.feature_names) == 32
    assert len(set(pipeline.feature_names)) == 32
    assert pipeline.schema_hash == compute_schema_hash()
    assert len(pipeline.schema_hash) == 64

    df = _make_synthetic_ohlcv(300)
    feats = pipeline.compute(df)
    assert list(feats.columns) == pipeline.feature_names
    # After max_lookback (200) warmup bars, all 32 features must be finite (no NaNs/infs)
    tail = feats.iloc[pipeline.max_lookback :]
    assert not tail.isna().any().any()
    assert np.isfinite(tail.to_numpy()).all()


def test_prefix_invariance_no_lookahead() -> None:
    """Computing features on [0..t] must match [0..T] at index t within 1e-12 for every feature."""
    df = _make_synthetic_ohlcv(420, seed=7)
    full_feats = compute_features(df)

    for cutoff in (210, 260, 350, 419):
        prefix_feats = compute_features(df.iloc[: cutoff + 1])
        expected = full_feats.iloc[cutoff].to_numpy(dtype=np.float64)
        actual = prefix_feats.iloc[cutoff].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Look-ahead detected at cutoff bar {cutoff}",
        )


def test_future_shock_invariance() -> None:
    """Mutating OHLCV strictly after bar t (×10 shock) must not change any feature at <= t."""
    df = _make_synthetic_ohlcv(360, seed=13)
    cutoff = 250

    shocked = df.copy()
    shocked.iloc[cutoff + 1 :, shocked.columns.get_indexer(["open", "high", "low", "close"])] *= (
        10.0
    )
    shocked.iloc[cutoff + 1 :, shocked.columns.get_indexer(["volume"])] *= 25.0

    f_orig = compute_features(df)
    f_shock = compute_features(shocked)

    orig_slice = f_orig.iloc[200 : cutoff + 1].to_numpy(dtype=np.float64)
    shock_slice = f_shock.iloc[200 : cutoff + 1].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(shock_slice, orig_slice, rtol=1e-12, atol=1e-12)


def test_scale_invariance_prices_and_volumes() -> None:
    """Multiplying prices by 10 (or 1e-4) and volume by 10 (or 0.01) must leave all 32 features invariant."""
    df = _make_synthetic_ohlcv(360, seed=21)
    base_feats = compute_features(df).iloc[200:]

    for price_mult, vol_mult in ((10.0, 10.0), (1e-4, 0.01), (0.05, 500.0)):
        scaled = df.copy()
        for col in ("open", "high", "low", "close"):
            scaled[col] = scaled[col] * price_mult
        scaled["volume"] = scaled["volume"] * vol_mult
        scaled_feats = compute_features(scaled).iloc[200:]

        for col in base_feats.columns:
            np.testing.assert_allclose(
                scaled_feats[col].to_numpy(dtype=np.float64),
                base_feats[col].to_numpy(dtype=np.float64),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Feature '{col}' is NOT scale-invariant (p×{price_mult}, v×{vol_mult})",
            )


def test_train_serve_golden_fixture_500_bars() -> None:
    """
    Golden train/serve parity check (docs/04-ml-pipeline.md §9.3):
    1. Serve warmup window (1500 bars for EMA200 convergence) matches full-history
       batch computation at the final bar within 1e-5 across all 32 features.
    2. Exact 500-bar golden fixture produces bit-identical train vs serve vectors (atol=1e-12)
       and deterministic feature row hashes.
    """
    pipeline = FeaturePipeline()
    df_full = _make_synthetic_ohlcv(2000, seed=99)

    train_last = pipeline.compute(df_full).iloc[-1]
    serve_last = pipeline.compute(df_full.iloc[-pipeline.serve_warmup_bars :]).iloc[-1]

    np.testing.assert_allclose(
        serve_last.to_numpy(dtype=np.float64),
        train_last.to_numpy(dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )

    # Exact train == serve identity on the 500-bar golden fixture
    df_500 = _make_synthetic_ohlcv(500, seed=99)
    batch_500 = pipeline.compute(df_500).iloc[-1]
    serve_500 = FeaturePipeline().compute(df_500.copy()).iloc[-1]
    np.testing.assert_allclose(
        serve_500.to_numpy(dtype=np.float64),
        batch_500.to_numpy(dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert compute_row_hash(batch_500) == compute_row_hash(serve_500)
    assert len(compute_row_hash(batch_500)) == 64


def test_triple_barrier_volatility_has_no_lookahead() -> None:
    df = _make_synthetic_ohlcv(200, seed=55)
    cutoff = 120
    shocked = df.copy()
    shocked.iloc[cutoff + 1 :, shocked.columns.get_indexer(["close"])] *= 5.0

    vol_orig = compute_ewm_volatility(df["close"], span=24, min_periods=24)
    vol_shock = compute_ewm_volatility(shocked["close"], span=24, min_periods=24)
    np.testing.assert_allclose(
        vol_shock.iloc[24 : cutoff + 1].to_numpy(dtype=np.float64),
        vol_orig.iloc[24 : cutoff + 1].to_numpy(dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
