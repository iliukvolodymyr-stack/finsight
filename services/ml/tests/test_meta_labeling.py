import numpy as np
import pandas as pd
import pytest
from finsight_ml.features.context_v2 import ContextFeatureBuilder
from finsight_ml.training.meta_labeling import (
    MetaLabelConfig,
    compute_net_trade_outcomes,
    run_v1_vs_v2_meta_walk_forward,
)


def _make_trend_regime_ohlcv(n: int = 2400, seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly data spanning ~3.3 months with alternating trend and chop regimes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01 00:00:00", periods=n, freq="h", tz="UTC")
    # Slow regime drift + noise
    phase = np.linspace(0, 8 * np.pi, n)
    drift = 0.0005 * np.sin(phase)
    vol = np.where((np.arange(n) // 300) % 2 == 0, 0.008, 0.015)
    rets = rng.normal(loc=drift, scale=vol, size=n)
    close = 50000.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = 50000.0
    spread = np.abs(rng.normal(0.0, 0.003, size=n)) * close
    high = np.maximum(open_p, close) + spread
    low = np.maximum(1.0, np.minimum(open_p, close) - spread)
    volume = rng.lognormal(mean=6.0, sigma=0.4, size=n)
    return pd.DataFrame(
        {
            "open_time": dates,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_compute_net_trade_outcomes_deducts_round_trip_cost() -> None:
    df = _make_trend_regime_ohlcv(n=120, seed=7)
    vol = pd.Series(0.01, index=df.index)
    sides = np.zeros(len(df), dtype=np.int64)
    sides[30] = 1
    sides[60] = -1

    net_zero_fee, _ = compute_net_trade_outcomes(
        df,
        sides,
        vol,
        horizon_bars=12,
        pt_mult=1.0,
        sl_mult=1.0,
        min_barrier_pct=0.003,
        round_trip_cost_pct=0.0,
    )
    net_30bps, y_30bps = compute_net_trade_outcomes(
        df,
        sides,
        vol,
        horizon_bars=12,
        pt_mult=1.0,
        sl_mult=1.0,
        min_barrier_pct=0.003,
        round_trip_cost_pct=0.003,
    )
    assert np.isnan(net_30bps[0])
    assert net_zero_fee[30] - net_30bps[30] == pytest.approx(0.003, abs=1e-12)
    assert net_zero_fee[60] - net_30bps[60] == pytest.approx(0.003, abs=1e-12)
    assert y_30bps[30] in (0.0, 1.0)


def test_walk_forward_meta_labeling_runs_end_to_end() -> None:
    df = _make_trend_regime_ohlcv(n=2600, seed=99)
    builder = ContextFeatureBuilder(include_base_v1=True)
    X_v2 = builder.build(df).features

    cfg = MetaLabelConfig(
        horizon_bars=12,
        embargo_bars=12,
        min_train_months=1,
        val_months=1,
        test_months=1,
        step_months=1,
        inner_oof_folds=2,
        min_val_trades=4,
    )
    res = run_v1_vs_v2_meta_walk_forward(
        df,
        X_v2,
        symbol="BTCUSDT",
        timeframe="1h",
        cfg=cfg,
        primary_params={
            "n_estimators": 40,
            "learning_rate": 0.08,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 20,
            "class_weight": "balanced",
            "random_state": 42,
        },
    )
    assert res.n_folds >= 1
    assert res.oos_bars > 100
    assert 0.0 <= res.primary_metrics["balanced_accuracy"] <= 1.0
    assert len(res.primary_backtest.equity_curve) == len(res.meta_vol_backtest.equity_curve)
    assert np.isfinite(res.meta_vol_backtest.sharpe)
    assert res.meta_vol_backtest.max_drawdown >= 0.0
