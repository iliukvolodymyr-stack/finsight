from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from finsight_ml.risk import (
    TransactionRecord,
    align_portfolio_daily_returns,
    compute_correlation_matrix,
    compute_portfolio_risk_metrics,
    recompute_positions,
    valuate_positions,
)


def test_position_ledger_weighted_avg_sell_and_over_sell_guard() -> None:
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    txs = [
        TransactionRecord("inst-btc", "BTCUSDT", "BUY", 1.0, 80000.0, 80.0, t0),
        TransactionRecord(
            "inst-btc", "BTCUSDT", "BUY", 1.0, 100000.0, 100.0, t0 + timedelta(days=1)
        ),
        TransactionRecord(
            "inst-btc", "BTCUSDT", "SELL", 0.5, 110000.0, 55.0, t0 + timedelta(days=2)
        ),
        TransactionRecord("inst-eth", "ETHUSDT", "BUY", 10.0, 3000.0, 30.0, t0 + timedelta(days=1)),
    ]

    states, realized_pnl, total_fees = recompute_positions(txs)
    btc = states["inst-btc"]
    # Avg entry = (80k + 100k) / 2 = 90k; remaining qty = 1.5
    assert np.isclose(btc.avg_price, 90000.0)
    assert np.isclose(btc.quantity, 1.5)
    # Realized PnL = (110k - 90k) * 0.5 - 55 = 10,000 - 55 = 9,945
    assert np.isclose(realized_pnl, 9945.0)
    assert np.isclose(total_fees, 80.0 + 100.0 + 55.0 + 30.0)

    valuation = valuate_positions(
        states,
        current_prices={"inst-btc": 95000.0, "inst-eth": 3200.0},
        asset_classes={"inst-btc": "crypto", "inst-eth": "crypto"},
        total_realized_pnl=realized_pnl,
        total_fees=total_fees,
    )
    # BTC MV = 1.5 * 95k = 142,500; ETH MV = 10 * 3.2k = 32,000 -> total 174,500
    assert np.isclose(valuation.total_market_value, 174500.0)
    assert np.isclose(sum(valuation.weights_by_symbol.values()), 1.0)
    assert np.isclose(valuation.total_unrealized_pnl, (95000 - 90000) * 1.5 + (3200 - 3000) * 10)

    # Selling more than held must fail cleanly
    bad_txs = [
        *txs,
        TransactionRecord("inst-eth", "ETHUSDT", "SELL", 10.5, 3200.0, 0.0, t0 + timedelta(days=3)),
    ]
    with pytest.raises(ValueError, match="Insufficient quantity"):
        recompute_positions(bad_txs)


def test_historical_var_cvar_invariant_and_risk_levels() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    btc_rets = rng.normal(0.001, 0.03, size=120)
    eth_rets = rng.normal(0.0005, 0.04, size=120)
    btc_close = pd.Series(90000.0 * np.exp(np.cumsum(btc_rets)), index=dates)
    eth_close = pd.Series(3000.0 * np.exp(np.cumsum(eth_rets)), index=dates)

    port_rets = align_portfolio_daily_returns(
        {"BTCUSDT": btc_close, "ETHUSDT": eth_close},
        {"BTCUSDT": 0.7, "ETHUSDT": 0.3},
        window_days=90,
    )
    assert len(port_rets) == 90

    risk = compute_portfolio_risk_metrics(
        port_rets,
        {"BTCUSDT": 0.7, "ETHUSDT": 0.3},
        periods_per_year=365,
    )
    assert risk["var_95"] > 0.0
    assert risk["cvar_95"] >= risk["var_95"]
    assert risk["annual_volatility"] > 0.0
    assert 0.0 <= risk["max_drawdown"] <= 1.0
    assert risk["risk_level"] in {"low", "medium", "high"}

    # Calm bond-like return series -> risk_level == 'low'
    calm_rets = pd.Series(rng.normal(0.0002, 0.003, size=90), index=dates[-90:])
    calm_risk = compute_portfolio_risk_metrics(calm_rets, {"BOND": 1.0}, periods_per_year=252)
    assert calm_risk["risk_level"] == "low"


def test_correlation_matrix_aligns_crypto_and_nyse_schedules() -> None:
    # 100 calendar days for crypto vs weekdays-only for SPY
    all_days = pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC")
    weekdays = all_days[all_days.day_of_week < 5]

    # Make SPY and BTC positively correlated on shared weekdays
    rng = np.random.default_rng(15)
    common_shock = rng.normal(0.0, 0.015, size=len(all_days))
    btc_close = pd.Series(
        90000.0 * np.exp(np.cumsum(common_shock + rng.normal(0.0, 0.01, size=100))), index=all_days
    )
    spy_base = btc_close.loc[weekdays] / 180.0
    spy_close = pd.Series(
        spy_base.to_numpy() * np.exp(np.cumsum(rng.normal(0.0, 0.003, size=len(weekdays)))),
        index=weekdays,
    )

    corr = compute_correlation_matrix({"BTCUSDT": btc_close, "SPY": spy_close}, window_days=90)
    assert corr["symbols"] == ["BTCUSDT", "SPY"]
    mat = corr["matrix"]
    assert mat[0][0] == 1.0 and mat[1][1] == 1.0
    assert np.isclose(mat[0][1], mat[1][0], atol=1e-9)
    assert -1.0 <= mat[0][1] <= 1.0
    assert corr["observations"] <= len(weekdays) - 1
