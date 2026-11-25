from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from finsight_ml.backtest import (
    BacktestEngine,
    BacktestStrategyConfig,
    periods_per_year_for,
    run_buy_and_hold,
    run_ema_crossover_benchmark,
    run_walk_forward_backtest,
    validate_backtest_no_in_sample_leak,
)
from finsight_ml.cli import app as ml_cli
from finsight_ml.labels.triple_barrier import (
    compute_ewm_volatility,
    compute_triple_barrier_dataset,
)
from finsight_ml.training import TrainingConfig
from typer.testing import CliRunner


def _make_hourly_ohlcv(n: int = 100, seed: int = 33) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01 00:00:00", periods=n, freq="h", tz="UTC")
    vol = 0.008 + 0.004 * np.abs(np.sin(np.arange(n) / 180.0))
    rets = rng.normal(0.0, vol)
    close = 30000.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = 30000.0
    spread = np.abs(rng.normal(0.0, 0.006, size=n)) * close
    high = np.maximum(open_p, close) + spread
    low = np.maximum(1.0, np.minimum(open_p, close) - spread)
    volume = rng.lognormal(mean=6.0, sigma=0.5, size=n)
    return pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_backtest_barriers_exact_1to1_parity_with_triple_barrier_labels() -> None:
    df = _make_hourly_ohlcv(n=250, seed=19)
    horizon = 12
    pt_mult = 1.0
    sl_mult = 1.0
    min_barrier = 0.003

    sigma = compute_ewm_volatility(df["close"], span=24, min_periods=24)
    tb = compute_triple_barrier_dataset(
        df,
        horizon_bars=horizon,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        volatility=sigma,
        min_barrier_pct=min_barrier,
        entry_on_next_open=True,
    )

    # Issue non-overlapping BUY signals safely spaced after each expected t1
    signals = pd.Series(0, index=df.index, dtype="int64")
    pos = 30
    chosen_signal_positions: list[int] = []
    while pos < len(df) - horizon - 2:
        row = tb.iloc[pos]
        if pd.notna(row["label"]) and row["exit_reason"] != "low_vol":
            signals.iloc[pos] = 1
            chosen_signal_positions.append(pos)
            pos = int(row["exit_pos"]) + 1
        else:
            pos += 1

    engine = BacktestEngine(fee_bps=10.0, slippage_bps=5.0)
    res = engine.run_barriers(
        df=df,
        signals=signals,
        horizon_bars=horizon,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        volatility=sigma,
        min_barrier_pct=min_barrier,
        allow_short=False,
    )

    assert len(res.trades) == len(chosen_signal_positions)
    assert len(res.trades) >= 3

    for trade, sig_pos in zip(res.trades, chosen_signal_positions, strict=True):
        expected = tb.iloc[sig_pos]
        assert trade.side == "LONG"
        assert trade.entry_at == df.index[sig_pos + 1]
        assert np.isclose(trade.entry_price, float(expected["entry_price"]))
        assert trade.exit_at == expected["t1"]
        assert np.isclose(trade.exit_price, float(expected["exit_price"]))
        assert trade.exit_reason == expected["exit_reason"]


def test_long_flat_ignores_short_and_no_pyramiding() -> None:
    df = _make_hourly_ohlcv(n=120, seed=7)
    sigma = pd.Series(0.01, index=df.index)

    # Only -1 signals in long/flat spot mode -> 0 trades
    only_shorts = pd.Series(-1, index=df.index, dtype="int64")
    engine = BacktestEngine(fee_bps=10.0, slippage_bps=5.0)
    res_flat = engine.run_barriers(
        df, only_shorts, horizon_bars=10, volatility=sigma, allow_short=False
    )
    assert res_flat.trades_count == 0
    assert res_flat.total_return == 0.0

    # With allow_short=True, continuous -1 triggers sequential non-overlapping SHORT trades
    res_short = engine.run_barriers(
        df, only_shorts, horizon_bars=10, volatility=sigma, allow_short=True
    )
    assert res_short.trades_count >= 1
    assert all(t.side == "SHORT" for t in res_short.trades)
    for t1, t2 in zip(res_short.trades, res_short.trades[1:], strict=False):
        assert t2.entry_pos > t1.exit_pos


def test_in_sample_guard_and_exchange_periods() -> None:
    assert periods_per_year_for("binance", "1h") == 8760
    assert periods_per_year_for("binance", "1d") == 365
    assert periods_per_year_for("nyse", "1h") == 1764
    assert periods_per_year_for("nyse", "1d") == 252

    train_end = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    # 12h after train_data_end violates 24h embargo for static_model
    with pytest.raises(ValueError, match="In-sample leakage guard"):
        validate_backtest_no_in_sample_leak(
            mode="static_model",
            period_start=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
            train_data_end=train_end,
            embargo_bars=24,
            timeframe="1h",
        )

    # >= 24h after train_data_end is allowed
    validate_backtest_no_in_sample_leak(
        mode="static_model",
        period_start=datetime(2026, 10, 2, 0, 0, tzinfo=UTC),
        train_data_end=train_end,
        embargo_bars=24,
        timeframe="1h",
    )
    # walk_forward_retrain is always allowed over full history
    validate_backtest_no_in_sample_leak(
        mode="walk_forward_retrain",
        period_start=datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
        train_data_end=train_end,
    )


def test_benchmarks_and_walk_forward_backtest_end_to_end(tmp_path: Path) -> None:
    df = _make_hourly_ohlcv(n=12000, seed=42)
    bh = run_buy_and_hold(df.iloc[-1000:])
    ema = run_ema_crossover_benchmark(df.iloc[-1000:], benchmark_equity=bh.equity_curve)
    assert bh.trades_count == 1
    assert "alpha_vs_benchmark" in ema.metrics

    cfg = TrainingConfig(
        name="lgbm_core",
        semver="0.1.0-bt",
        symbol="TESTBTC",
        timeframe="1h",
        horizon_bars=24,
        pt_mult=1.0,
        sl_mult=1.0,
        min_train_months=8,
        val_months=2,
        test_months=2,
        embargo_bars=24,
        early_stopping_rounds=20,
        hyperparameters={
            "n_estimators": 30,
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 20,
            "random_state": 42,
        },
    )
    bt_cfg = BacktestStrategyConfig(
        horizon_bars=24,
        pt_mult=1.0,
        sl_mult=1.0,
        position_size=0.5,
        allow_short=False,
        fee_bps=10.0,
        slippage_bps=5.0,
        random_seeds=5,
    )

    out = run_walk_forward_backtest(
        df=df,
        train_cfg=cfg,
        bt_cfg=bt_cfg,
        artifacts_dir=tmp_path,
        run_id="test_run",
    )
    assert len(out.fold_summaries) >= 3
    assert out.artifact_dir is not None
    assert (out.artifact_dir / "equity.parquet").exists()
    assert (out.artifact_dir / "trades.parquet").exists()
    assert (out.artifact_dir / "metrics.json").exists()
    assert (out.artifact_dir / "config.yaml").exists()
    assert out.random_benchmark_metrics["n_seeds"] == 5

    # CLI smoke test using offline Parquet snapshot
    pq_path = tmp_path / "testbtc_1h.parquet"
    df_reset = df.reset_index().rename(columns={"index": "open_time"})
    df_reset.to_parquet(pq_path, index=False)

    cfg_path = tmp_path / "testbtc.yaml"
    cfg_path.write_text(
        """
name: lgbm_core
version: 0.1.0-cli
symbol: TESTBTC
timeframe: 1h
horizon_bars: 24
status_target: candidate
labeling:
  pt_mult: 1.0
  sl_mult: 1.0
  min_barrier_pct: 0.003
validation:
  min_train_months: 8
  val_months: 2
  test_months: 2
  embargo_bars: 24
hyperparameters:
  n_estimators: 25
  max_depth: 3
  num_leaves: 8
  min_child_samples: 20
""".strip(),
        encoding="utf-8",
    )

    runner = CliRunner()
    res_train = runner.invoke(
        ml_cli,
        ["train", str(cfg_path), "--data-path", str(pq_path), "--artifacts-dir", str(tmp_path)],
    )
    assert res_train.exit_code == 0, res_train.output
    assert "OOS Walk-Forward Summary" in res_train.output

    res_bt = runner.invoke(
        ml_cli,
        ["backtest", str(cfg_path), "--data-path", str(pq_path), "--artifacts-dir", str(tmp_path)],
    )
    assert res_bt.exit_code == 0, res_bt.output
    assert "OOS Backtest v0" in res_bt.output
