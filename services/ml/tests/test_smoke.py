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
    # Last horizon_bars rows must remain unlabeled (NaN) to avoid look-ahead/tail bias
    assert labels.iloc[-10:].isna().all()
    assert set(labels.dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_triple_barrier_scaling_min_width_and_tiebreak():
    # horizon=4 -> sqrt(4)=2; pt=sl=2.0, vol=0.01 -> width = 0.04
    # upper = 100 * exp(0.04) ≈ 104.081, lower = 100 * exp(-0.04) ≈ 96.079
    close = pd.Series([100.0] * 10)
    vol = pd.Series([0.01] * 10)

    # Move up to 103 (< 104.081) -> should stay HOLD (0.0)
    high_hold = pd.Series([100.0, 103.0, 102.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    low_hold = pd.Series([100.0] * 10)
    labels_hold = apply_triple_barrier(close, high_hold, low_hold, horizon_bars=4, volatility=vol)
    assert labels_hold.iloc[0] == 0.0

    # Move up to 105 (>= 104.081) -> BUY (1.0)
    high_buy = pd.Series([100.0, 105.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    labels_buy = apply_triple_barrier(close, high_buy, low_hold, horizon_bars=4, volatility=vol)
    assert labels_buy.iloc[0] == 1.0

    # Both barriers touched on bar 1 -> conservative stop-loss (-1.0)
    low_both = pd.Series([100.0, 95.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    labels_both = apply_triple_barrier(close, high_buy, low_both, horizon_bars=4, volatility=vol)
    assert labels_both.iloc[0] == -1.0

    # Tiny volatility below min_barrier_pct (0.003) -> forced HOLD (0.0)
    tiny_vol = pd.Series([0.0001] * 10)
    labels_tiny = apply_triple_barrier(
        close, high_buy, low_hold, horizon_bars=4, volatility=tiny_vol, min_barrier_pct=0.003
    )
    assert labels_tiny.iloc[0] == 0.0


def test_lgbm_smoke():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f2": [0.1, 0.2, 0.1, 0.4, 0.5, 0.6]})
    y = pd.Series([0, 1, 0, -1, 1, 0])
    model = LGBMForecaster(n_estimators=10)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_lgbm_missing_class_preserves_3col_contract():
    # Only HOLD (0) and BUY (1) present, plus a NaN tail label; SELL (-1) absent
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    y = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, np.nan])
    model = LGBMForecaster(n_estimators=10)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (7, 3)
    # Column 0 is SELL -> must be all zeros since SELL never appeared in train
    assert np.allclose(probs[:, 0], 0.0)
    # Columns 1 (HOLD) + 2 (BUY) must sum to 1.0
    assert np.allclose(probs[:, 1] + probs[:, 2], 1.0)


def test_backtest_smoke():
    n = 100
    signals = pd.Series([1 if i % 10 < 5 else 0 for i in range(n)])
    open_prices = pd.Series(np.linspace(100, 120, n))
    close_prices = open_prices + 0.5

    engine = BacktestEngine(fee_bps=10, slippage_bps=5)
    result = engine.run(signals, open_prices, close_prices)
    assert result.trades_count >= 0
    assert len(result.equity_curve) == n


def test_backtest_direct_reversal_and_close_mtm():
    # Bar 0: signal +1 -> LONG enters at Bar 1 open (100), exits at Bar 2 open (110) => +10%
    # Bar 1: signal -1 -> SHORT enters at Bar 2 open (110), MTM at Bar 2 close (99) => +10%
    signals = pd.Series([1, -1, -1])
    open_prices = pd.Series([100.0, 100.0, 110.0])
    close_prices = pd.Series([100.0, 110.0, 99.0])

    engine = BacktestEngine(fee_bps=10, slippage_bps=5)
    res_both_win = engine.run(signals, open_prices, close_prices)
    assert res_both_win.trades_count == 2
    assert res_both_win.win_rate == 1.0
    assert res_both_win.total_return > 0.19

    # Now make the SHORT lose (Bar 2 close = 121 => -10% on short), LONG still won (+10%)
    close_prices_loss = pd.Series([100.0, 110.0, 121.0])
    res_split = engine.run(signals, open_prices, close_prices_loss)
    assert res_split.trades_count == 2
    assert res_split.win_rate == 0.5
