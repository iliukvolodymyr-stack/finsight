import numpy as np
import pandas as pd
from finsight_ml.labels.triple_barrier import (
    compute_sample_uniqueness,
    compute_triple_barrier_dataset,
    summarize_label_distribution,
)


def test_triple_barrier_next_open_gap_and_vertical_exit() -> None:
    # 8 bars: horizon=3, pt=sl=1.0, sigma=0.02 -> width = 0.02 * sqrt(3) ≈ 0.03464
    # For entry=100: upper ≈ 103.525, lower ≈ 96.595
    dates = pd.date_range("2026-01-01 00:00:00", periods=8, freq="h", tz="UTC")
    vol = pd.Series([0.02] * 8, index=dates)

    df = pd.DataFrame(
        {
            # Bar 0 decision enters at Bar 1 open = 100.0; Bar 2 gaps up to open=105.0 -> gap TP!
            # Bar 1 decision enters at Bar 2 open = 105.0; Bar 3 gaps down to open=95.0 -> gap SL!
            # Bar 2 decision enters at Bar 3 open = 95.0; stays inside barriers on bars 3,4,5 -> vertical at bar 5 close!
            "open": [100.0, 100.0, 105.0, 95.0, 95.5, 95.8, 96.0, 96.0],
            "high": [100.5, 101.0, 106.0, 96.0, 96.5, 96.8, 96.5, 96.5],
            "low": [99.5, 99.0, 104.5, 94.5, 94.5, 94.8, 95.5, 95.5],
            "close": [100.0, 100.5, 105.5, 95.2, 95.7, 96.1, 96.2, 96.2],
        },
        index=dates,
    )

    tb = compute_triple_barrier_dataset(
        df,
        horizon_bars=3,
        pt_mult=1.0,
        sl_mult=1.0,
        volatility=vol,
        entry_on_next_open=True,
    )

    # Bar 0: entry at Bar 1 open (100.0), Bar 2 gaps open at 105.0 >= upper -> PT at open[2]
    assert tb["label"].iloc[0] == 1.0
    assert tb["entry_price"].iloc[0] == 100.0
    assert tb["exit_reason"].iloc[0] == "pt"
    assert tb["exit_price"].iloc[0] == 105.0
    assert tb["t1"].iloc[0] == dates[2]

    # Bar 1: entry at Bar 2 open (105.0), Bar 3 gaps open at 95.0 <= lower -> SL at open[3]
    assert tb["label"].iloc[1] == -1.0
    assert tb["entry_price"].iloc[1] == 105.0
    assert tb["exit_reason"].iloc[1] == "sl"
    assert tb["exit_price"].iloc[1] == 95.0
    assert tb["t1"].iloc[1] == dates[3]

    # Bar 2: entry at Bar 3 open (95.0), bars 3, 4, 5 stay inside [91.77, 98.35] -> vertical at bar 5 close
    assert tb["label"].iloc[2] == 0.0
    assert tb["entry_price"].iloc[2] == 95.0
    assert tb["exit_reason"].iloc[2] == "vertical"
    assert tb["exit_price"].iloc[2] == 96.1
    assert tb["t1"].iloc[2] == dates[5]

    # Last 3 (horizon_bars) rows must be unlabeled NaN / NaT
    assert tb["label"].iloc[-3:].isna().all()
    assert tb["t1"].iloc[-3:].isna().all()
    assert tb["sample_weight"].iloc[-3:].isna().all()


def test_sample_uniqueness_exact_values() -> None:
    # Event 0 spans bars [1, 2], Event 1 spans bars [2, 3], Event 2 spans bars [5, 5] (disjoint)
    # Concurrency per bar: bar1=1, bar2=2, bar3=1, bar5=1
    # Uniqueness:
    #   Event 0: mean(1/1, 1/2) = 0.75
    #   Event 1: mean(1/2, 1/1) = 0.75
    #   Event 2: mean(1/1)      = 1.00
    entries = np.array([1.0, 2.0, 5.0, np.nan])
    exits = np.array([2.0, 3.0, 5.0, np.nan])

    u, w = compute_sample_uniqueness(entries, exits, n_bars=8)
    np.testing.assert_allclose(u[:3], [0.75, 0.75, 1.0], rtol=1e-12, atol=1e-12)
    assert np.isnan(u[3])
    assert np.isnan(w[3])
    assert np.isclose(np.nanmean(w), 1.0, atol=1e-12)
    # Disjoint event gets strictly higher weight than overlapping events
    assert w[2] > w[0]


def test_label_distribution_report_and_hold_warning() -> None:
    dates = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    # 8 HOLD, 1 BUY, 1 SELL -> hold_share = 0.80 > 0.70 -> warning True
    tb_skewed = pd.DataFrame(
        {
            "label": [0.0] * 8 + [1.0, -1.0],
            "exit_reason": ["vertical"] * 8 + ["pt", "sl"],
            "uniqueness": [0.5] * 10,
        },
        index=dates,
    )
    rep_skewed = summarize_label_distribution(tb_skewed)
    assert rep_skewed.n_valid == 10
    assert rep_skewed.hold_share == 0.8
    assert rep_skewed.hold_warning is True
    assert rep_skewed.exit_reason_counts == {"vertical": 8, "pt": 1, "sl": 1}

    # Balanced: 3 BUY, 4 HOLD, 3 SELL -> hold_share = 0.40 -> warning False
    tb_bal = pd.DataFrame(
        {
            "label": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            "exit_reason": ["pt"] * 3 + ["vertical"] * 4 + ["sl"] * 3,
            "uniqueness": [0.4] * 10,
        },
        index=dates,
    )
    rep_bal = summarize_label_distribution(tb_bal)
    assert rep_bal.hold_warning is False
    assert np.isclose(rep_bal.buy_share, 0.3)
    assert np.isclose(rep_bal.hold_share, 0.4)
    assert np.isclose(rep_bal.sell_share, 0.3)
