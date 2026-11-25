import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from finsight_core.config import get_settings

from finsight_ml.backtest.benchmarks import (
    run_buy_and_hold,
    run_ema_crossover_benchmark,
    run_random_benchmark,
)
from finsight_ml.backtest.engine import BacktestEngine, BacktestTrade
from finsight_ml.features.pipeline import FeaturePipeline
from finsight_ml.labels.triple_barrier import compute_ewm_volatility, compute_triple_barrier_dataset
from finsight_ml.models.calibration import ProbabilityCalibrator
from finsight_ml.models.lgbm import LGBMForecaster
from finsight_ml.training.pipeline import TrainingConfig
from finsight_ml.validation.metrics import (
    apply_signal_decision_rule,
    tune_signal_thresholds_on_val,
)
from finsight_ml.validation.walk_forward import purged_walk_forward_splits


@dataclass
class BacktestStrategyConfig:
    entry_threshold: float | None = None
    min_margin: float = 0.10
    exit_mode: str = "label_barriers"
    horizon_bars: int = 24
    pt_mult: float = 1.0
    sl_mult: float = 1.0
    min_barrier_pct: float = 0.003
    position_size: float = 1.0
    allow_short: bool = False
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_capital: float = 10000.0
    exchange: str = "binance"
    timeframe: str = "1h"
    random_seeds: int = 50


@dataclass
class WalkForwardBacktestOutput:
    symbol: str
    timeframe: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    strategy_metrics: dict[str, Any]
    buy_and_hold_metrics: dict[str, Any]
    ema_crossover_metrics: dict[str, Any]
    random_benchmark_metrics: dict[str, Any]
    fold_summaries: list[dict[str, Any]]
    trades: list[BacktestTrade]
    equity_df: pd.DataFrame
    acceptance_a3: dict[str, Any]
    artifact_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)


def run_walk_forward_backtest(
    df: pd.DataFrame,
    train_cfg: TrainingConfig,
    bt_cfg: BacktestStrategyConfig | None = None,
    artifacts_dir: Path | str | None = None,
    run_id: str | None = None,
) -> WalkForwardBacktestOutput:
    """
    Execute strict out-of-sample walk-forward backtest (`exit_mode='label_barriers'`):
    - Each fold retrains LightGBM + ProbabilityCalibrator strictly on purged train/val.
    - Signal thresholds are either fixed from `bt_cfg.entry_threshold` or tuned on val.
    - Unseen test folds are stitched chronologically and simulated end-to-end with
      next-open execution and realistic 10+5 bps costs, alongside Buy & Hold,
      EMA(20/50) crossover, and Random benchmarks on the exact same OOS period.
    """
    if bt_cfg is None:
        bt_cfg = BacktestStrategyConfig(
            horizon_bars=train_cfg.horizon_bars,
            pt_mult=train_cfg.pt_mult,
            sl_mult=train_cfg.sl_mult,
            min_barrier_pct=train_cfg.min_barrier_pct,
            exchange="binance" if train_cfg.symbol.endswith("USDT") else "nyse",
            timeframe=train_cfg.timeframe,
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input OHLCV DataFrame must have a UTC DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    feats = FeaturePipeline().compute(df)
    sigma = compute_ewm_volatility(df["close"], span=24, min_periods=24)
    tb = compute_triple_barrier_dataset(
        df,
        horizon_bars=train_cfg.horizon_bars,
        pt_mult=train_cfg.pt_mult,
        sl_mult=train_cfg.sl_mult,
        volatility=sigma,
        min_barrier_pct=train_cfg.min_barrier_pct,
        entry_on_next_open=True,
    )

    valid_mask = (~feats.isna().any(axis=1)) & tb["label"].notna() & sigma.notna()
    valid_indices = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
    if len(valid_indices) < 200:
        raise ValueError(f"Insufficient valid labeled bars: {len(valid_indices)}")

    df_valid = df.iloc[valid_indices]
    X_valid = feats.iloc[valid_indices]
    y_valid = tb["label"].iloc[valid_indices].astype(int)
    t1_valid = tb["t1"].iloc[valid_indices]
    weights_valid = tb["sample_weight"].iloc[valid_indices].to_numpy(dtype=np.float64)

    folds = purged_walk_forward_splits(
        timestamps=df_valid.index,
        t1=t1_valid,
        min_train_months=train_cfg.min_train_months,
        val_months=train_cfg.val_months,
        test_months=train_cfg.test_months,
        step_months=train_cfg.step_months,
        embargo_bars=train_cfg.embargo_bars,
    )
    if not folds:
        raise ValueError("Purged walk-forward produced 0 folds for backtest.")

    oos_sig_series = pd.Series(0, index=df.index, dtype="int64")
    oos_conf_series = pd.Series(np.nan, index=df.index, dtype="float64")
    oos_fold_series = pd.Series(pd.NA, index=df.index, dtype="Int64")

    engine = BacktestEngine(fee_bps=bt_cfg.fee_bps, slippage_bps=bt_cfg.slippage_bps)
    fold_summaries: list[dict[str, Any]] = []
    hp = train_cfg.hyperparameters

    for fold in folds:
        tr_idx, va_idx, te_idx = fold.train_indices, fold.val_indices, fold.test_indices
        X_tr, y_tr, w_tr = X_valid.iloc[tr_idx], y_valid.iloc[tr_idx], weights_valid[tr_idx]
        X_va, y_va = X_valid.iloc[va_idx], y_valid.iloc[va_idx]
        X_te = X_valid.iloc[te_idx]

        lgbm = LGBMForecaster(
            n_estimators=int(hp.get("n_estimators", 300)),
            learning_rate=float(hp.get("learning_rate", 0.03)),
            max_depth=int(hp.get("max_depth", 5)),
            num_leaves=int(hp.get("num_leaves", 31)),
            min_child_samples=int(hp.get("min_child_samples", 40)),
            subsample=float(hp.get("subsample", 0.8)),
            colsample_bytree=float(hp.get("colsample_bytree", 0.8)),
            reg_alpha=float(hp.get("reg_alpha", 0.1)),
            reg_lambda=float(hp.get("reg_lambda", 1.0)),
            class_weight=hp.get("class_weight", "balanced"),
            random_state=int(hp.get("random_state", 42)),
        ).fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=train_cfg.early_stopping_rounds,
        )

        cal = ProbabilityCalibrator().fit(lgbm.predict_proba(X_va), y_va.to_numpy(dtype=np.int64))
        cal_va = cal.transform(lgbm.predict_proba(X_va))
        cal_te = cal.transform(lgbm.predict_proba(X_te))

        if bt_cfg.entry_threshold is None:
            thr, margin = tune_signal_thresholds_on_val(
                cal_va,
                y_va.to_numpy(dtype=np.int64),
                min_coverage=train_cfg.min_signal_coverage,
            )
        else:
            thr = float(bt_cfg.entry_threshold)
            margin = float(bt_cfg.min_margin)

        te_sigs = apply_signal_decision_rule(cal_te, entry_threshold=thr, min_margin=margin)
        te_conf = np.max(cal_te, axis=1)
        te_ts = X_te.index

        oos_sig_series.loc[te_ts] = te_sigs
        oos_conf_series.loc[te_ts] = te_conf
        oos_fold_series.loc[te_ts] = fold.fold_no

        # Per-fold isolated backtest vs Buy & Hold on the same fold test slice
        fold_df = df.loc[fold.test_start : fold.test_end]
        fold_bh = run_buy_and_hold(
            fold_df,
            fee_bps=bt_cfg.fee_bps,
            slippage_bps=bt_cfg.slippage_bps,
            initial_capital=bt_cfg.initial_capital,
            exchange=bt_cfg.exchange,
            timeframe=bt_cfg.timeframe,
        )
        fold_bt = engine.run_barriers(
            df=fold_df,
            signals=oos_sig_series.loc[fold_df.index],
            horizon_bars=bt_cfg.horizon_bars,
            pt_mult=bt_cfg.pt_mult,
            sl_mult=bt_cfg.sl_mult,
            volatility=sigma.loc[fold_df.index],
            min_barrier_pct=bt_cfg.min_barrier_pct,
            position_size=bt_cfg.position_size,
            allow_short=bt_cfg.allow_short,
            initial_capital=bt_cfg.initial_capital,
            exchange=bt_cfg.exchange,
            timeframe=bt_cfg.timeframe,
            confidence=oos_conf_series.loc[fold_df.index],
            fold_map=oos_fold_series.loc[fold_df.index],
            benchmark_equity=fold_bh.equity_curve,
        )
        fold_summaries.append(
            {
                "fold_no": fold.fold_no,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "val_start": fold.val_start.isoformat(),
                "val_end": fold.val_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "thresholds": {"entry_threshold": thr, "min_margin": margin},
                "metrics": {
                    **fold_bt.metrics,
                    "beats_bh_sharpe": fold_bt.sharpe > fold_bh.sharpe,
                    "beats_bh_return": fold_bt.total_return > fold_bh.total_return,
                },
            }
        )

    # Full contiguous OOS window across all test folds
    oos_start = folds[0].test_start
    oos_end = folds[-1].test_end
    df_oos = df.loc[oos_start:oos_end]

    bh_res = run_buy_and_hold(
        df_oos,
        fee_bps=bt_cfg.fee_bps,
        slippage_bps=bt_cfg.slippage_bps,
        initial_capital=bt_cfg.initial_capital,
        exchange=bt_cfg.exchange,
        timeframe=bt_cfg.timeframe,
    )
    strat_res = engine.run_barriers(
        df=df_oos,
        signals=oos_sig_series.loc[df_oos.index],
        horizon_bars=bt_cfg.horizon_bars,
        pt_mult=bt_cfg.pt_mult,
        sl_mult=bt_cfg.sl_mult,
        volatility=sigma.loc[df_oos.index],
        min_barrier_pct=bt_cfg.min_barrier_pct,
        position_size=bt_cfg.position_size,
        allow_short=bt_cfg.allow_short,
        initial_capital=bt_cfg.initial_capital,
        exchange=bt_cfg.exchange,
        timeframe=bt_cfg.timeframe,
        confidence=oos_conf_series.loc[df_oos.index],
        fold_map=oos_fold_series.loc[df_oos.index],
        benchmark_equity=bh_res.equity_curve,
    )
    ema_res = run_ema_crossover_benchmark(
        df_oos,
        horizon_bars=bt_cfg.horizon_bars,
        pt_mult=bt_cfg.pt_mult,
        sl_mult=bt_cfg.sl_mult,
        fee_bps=bt_cfg.fee_bps,
        slippage_bps=bt_cfg.slippage_bps,
        position_size=bt_cfg.position_size,
        allow_short=bt_cfg.allow_short,
        initial_capital=bt_cfg.initial_capital,
        exchange=bt_cfg.exchange,
        timeframe=bt_cfg.timeframe,
        benchmark_equity=bh_res.equity_curve,
    )

    active_rate = float(np.mean(oos_sig_series.loc[df_oos.index].to_numpy() != 0))
    rand_metrics = run_random_benchmark(
        df_oos,
        buy_rate=max(0.01, active_rate),
        n_seeds=bt_cfg.random_seeds,
        horizon_bars=bt_cfg.horizon_bars,
        pt_mult=bt_cfg.pt_mult,
        sl_mult=bt_cfg.sl_mult,
        fee_bps=bt_cfg.fee_bps,
        slippage_bps=bt_cfg.slippage_bps,
        position_size=bt_cfg.position_size,
        allow_short=bt_cfg.allow_short,
        initial_capital=bt_cfg.initial_capital,
        exchange=bt_cfg.exchange,
        timeframe=bt_cfg.timeframe,
    )

    equity_df = pd.DataFrame(
        {
            "timestamp": df_oos.index,
            "strategy_equity": strat_res.equity_curve.to_numpy(dtype=np.float64),
            "benchmark_equity": bh_res.equity_curve.to_numpy(dtype=np.float64),
            "ema_equity": ema_res.equity_curve.to_numpy(dtype=np.float64),
        }
    )

    folds_beating_bh = sum(1 for f in fold_summaries if f["metrics"]["beats_bh_sharpe"])
    n_folds = len(fold_summaries)
    acceptance_a3 = {
        "sharpe_gt_zero": strat_res.sharpe > 0.0,
        "folds_beating_bh_sharpe": f"{folds_beating_bh}/{n_folds}",
        "beats_bh_half_folds": folds_beating_bh >= (n_folds / 2.0),
    }

    warnings: list[str] = []
    if strat_res.metrics.get("sharpe_warning"):
        warnings.append("SHARPE_GT_3_CHECK_LEAKAGE")
    if strat_res.metrics.get("fee_drag", 0.0) > 0.50:
        warnings.append("HIGH_FEE_DRAG_GT_50PCT")

    out_dir: Path | None = None
    if artifacts_dir is not None or run_id is not None:
        base_dir = Path(artifacts_dir or get_settings().ARTIFACTS_DIR)
        folder = run_id or f"{train_cfg.symbol.lower()}_{train_cfg.timeframe}_wf"
        out_dir = base_dir / "backtests" / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        equity_df.to_parquet(out_dir / "equity.parquet", index=False)
        trades_rows = [
            {
                "seq": t.seq,
                "side": t.side,
                "entry_at": t.entry_at.isoformat(),
                "exit_at": t.exit_at.isoformat(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "fee": t.fee,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "confidence": t.confidence,
                "fold_no": t.fold_no,
            }
            for t in strat_res.trades
        ]
        pd.DataFrame(trades_rows).to_parquet(out_dir / "trades.parquet", index=False)
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "strategy": strat_res.metrics,
                    "buy_and_hold": bh_res.metrics,
                    "ema_crossover": ema_res.metrics,
                    "random_benchmark": rand_metrics,
                    "folds": fold_summaries,
                    "acceptance_a3": acceptance_a3,
                    "warnings": warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {"training": asdict(train_cfg), "backtest": asdict(bt_cfg)}, sort_keys=False
            ),
            encoding="utf-8",
        )

    return WalkForwardBacktestOutput(
        symbol=train_cfg.symbol,
        timeframe=train_cfg.timeframe,
        period_start=oos_start,
        period_end=oos_end,
        strategy_metrics=strat_res.metrics,
        buy_and_hold_metrics=bh_res.metrics,
        ema_crossover_metrics=ema_res.metrics,
        random_benchmark_metrics=rand_metrics,
        fold_summaries=fold_summaries,
        trades=strat_res.trades,
        equity_df=equity_df,
        acceptance_a3=acceptance_a3,
        artifact_dir=out_dir,
        warnings=warnings,
    )
