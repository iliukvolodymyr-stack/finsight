import numpy as np
import pandas as pd
from finsight_ml.backtest.engine import BacktestEngine
from finsight_ml.features.pipeline import compute_features
from finsight_ml.labels.triple_barrier import apply_triple_barrier
from finsight_ml.models.lgbm import LGBMForecaster


def test_feature_pipeline_smoke():
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 150, n),
            "high": np.linspace(102, 152, n),
            "low": np.linspace(98, 148, n),
            "close": np.linspace(101, 151, n),
            "volume": np.ones(n) * 1000,
        },
        index=dates,
    )
    features = compute_features(df)
    assert "logret_1" in features.columns
    assert "rsi14" in features.columns
    assert "bb_width" in features.columns
    assert len(features) == n


def test_triple_barrier_smoke():
    n = 60
    close = pd.Series(np.linspace(100, 130, n))
    high = close + 1.0
    low = close - 1.0
    labels = apply_triple_barrier(close, high, low, horizon_bars=10)
    assert len(labels) == n
    assert set(labels.unique()).issubset({-1, 0, 1})


def test_lgbm_smoke():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f2": [0.1, 0.2, 0.1, 0.4, 0.5, 0.6]})
    y = pd.Series([0, 1, 0, -1, 1, 0])
    model = LGBMForecaster(n_estimators=10)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_backtest_smoke():
    n = 100
    signals = pd.Series([1 if i % 10 < 5 else 0 for i in range(n)])
    open_prices = pd.Series(np.linspace(100, 120, n))
    close_prices = open_prices + 0.5

    engine = BacktestEngine(fee_bps=10, slippage_bps=5)
    result = engine.run(signals, open_prices, close_prices)
    assert result.trades_count >= 0
    assert len(result.equity_curve) == n
