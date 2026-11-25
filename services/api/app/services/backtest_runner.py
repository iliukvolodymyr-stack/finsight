import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from app.infra.models import (
    BacktestFoldModel,
    BacktestModel,
    BacktestTradeModel,
    CandleModel,
    InstrumentModel,
    JobModel,
    ModelVersionModel,
)
from finsight_core.config import get_settings
from finsight_ml.backtest import (
    BacktestEngine,
    BacktestStrategyConfig,
    run_buy_and_hold,
    run_ema_crossover_benchmark,
    run_walk_forward_backtest,
    validate_backtest_no_in_sample_leak,
)
from finsight_ml.features.pipeline import FeaturePipeline
from finsight_ml.labels.triple_barrier import compute_ewm_volatility
from finsight_ml.models.calibration import ProbabilityCalibrator
from finsight_ml.models.lgbm import LGBMForecaster
from finsight_ml.training import TrainingConfig
from finsight_ml.validation.metrics import apply_signal_decision_rule
from loguru import logger
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession


async def _update_job(
    session: AsyncSession,
    job_id: uuid.UUID,
    **fields: Any,
) -> None:
    job = await session.get(JobModel, job_id)
    if job is not None:
        for k, v in fields.items():
            setattr(job, k, v)
        await session.commit()


async def execute_backtest_job(
    session: AsyncSession,
    backtest_id: uuid.UUID,
    job_id: uuid.UUID,
) -> None:
    """Execute queued backtest (walk_forward_retrain or static_model) and persist results."""
    bt = await session.get(BacktestModel, backtest_id)
    if bt is None:
        await _update_job(
            session,
            job_id,
            status="failed",
            error=f"Backtest {backtest_id} not found",
            finished_at=datetime.now(UTC),
        )
        return

    now_start = datetime.now(UTC)
    bt.status = "running"
    await _update_job(
        session,
        job_id,
        status="running",
        progress=0.1,
        started_at=now_start,
    )

    try:
        scope = bt.instrument_scope or []
        if not scope:
            raise ValueError("Backtest instrument_scope cannot be empty")
        target_inst_id = uuid.UUID(str(scope[0]))
        inst = await session.get(InstrumentModel, target_inst_id)
        if inst is None:
            raise ValueError(f"Instrument {target_inst_id} not found")

        cfg_dict = bt.strategy_config or {}
        timeframe = str(cfg_dict.get("timeframe", "1h"))
        horizon_bars = int(cfg_dict.get("max_holding_bars", 24))
        pt_mult = float(cfg_dict.get("pt_mult", 1.0))
        sl_mult = float(cfg_dict.get("sl_mult", 1.0))
        entry_thr = cfg_dict.get("entry_threshold")
        min_margin = float(cfg_dict.get("min_margin", 0.10))
        position_size = float(cfg_dict.get("position_size", 1.0))
        allow_short = bool(cfg_dict.get("allow_short", False))
        fee_bps = float(cfg_dict.get("fee_bps", 10.0))
        slippage_bps = float(cfg_dict.get("slippage_bps", 5.0))

        warmup_bars = 1600 if bt.mode == "static_model" else 0
        bar_hours = 1 if timeframe == "1h" else 24
        fetch_from = bt.period_start - timedelta(hours=warmup_bars * bar_hours)

        rows = (
            await session.execute(
                select(
                    CandleModel.open_time,
                    CandleModel.open,
                    CandleModel.high,
                    CandleModel.low,
                    CandleModel.close,
                    CandleModel.volume,
                )
                .where(
                    CandleModel.instrument_id == inst.id,
                    CandleModel.timeframe == timeframe,
                    CandleModel.is_closed.is_(True),
                    CandleModel.open_time >= fetch_from,
                    CandleModel.open_time <= bt.period_end,
                )
                .order_by(CandleModel.open_time.asc())
            )
        ).all()

        if len(rows) < 250:
            raise ValueError(
                f"Insufficient closed candles ({len(rows)}) for {inst.symbol} {timeframe} backtest."
            )

        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time").sort_index()

        await _update_job(session, job_id, progress=0.25)

        settings = get_settings()
        artifacts_root = Path(settings.ARTIFACTS_DIR)
        bt_dir = artifacts_root / "backtests" / str(bt.id)
        bt_dir.mkdir(parents=True, exist_ok=True)

        # Clean prior folds/trades if re-run
        await session.execute(
            delete(BacktestFoldModel).where(BacktestFoldModel.backtest_id == bt.id)
        )
        await session.execute(
            delete(BacktestTradeModel).where(BacktestTradeModel.backtest_id == bt.id)
        )

        if bt.mode == "static_model":
            mv = (
                await session.execute(
                    select(ModelVersionModel)
                    .where(
                        ModelVersionModel.instrument_id == inst.id,
                        ModelVersionModel.timeframe == timeframe,
                        ModelVersionModel.task == "direction_3class",
                        ModelVersionModel.status == "production",
                    )
                    .order_by(desc(ModelVersionModel.promoted_at))
                    .limit(1)
                )
            ).scalar_one_or_none()
            if mv is None:
                raise ValueError(f"No production model available for {inst.symbol} {timeframe}")

            validate_backtest_no_in_sample_leak(
                mode="static_model",
                period_start=bt.period_start,
                train_data_end=mv.train_data_end,
                embargo_bars=horizon_bars,
                timeframe=timeframe,
            )

            art_dir = Path(mv.artifact_uri)
            model = LGBMForecaster.load(art_dir / "model.joblib")
            calibrator = ProbabilityCalibrator.load(art_dir / "calibrator.joblib")

            feats = FeaturePipeline().compute(df)
            sigma = compute_ewm_volatility(df["close"], span=24, min_periods=24)
            mask = (
                (df.index >= bt.period_start)
                & (df.index <= bt.period_end)
                & (~feats.isna().any(axis=1))
            )
            df_eval = df.loc[mask]
            X_eval = feats.loc[mask]

            cal_probs = calibrator.transform(model.predict_proba(X_eval))
            thr = float(
                entry_thr
                if entry_thr is not None
                else (mv.params or {}).get("entry_threshold", 0.55)
            )
            sigs = apply_signal_decision_rule(cal_probs, entry_threshold=thr, min_margin=min_margin)
            sig_series = pd.Series(sigs, index=df_eval.index, dtype="int64")
            conf_series = pd.Series(np.max(cal_probs, axis=1), index=df_eval.index, dtype="float64")

            bh_res = run_buy_and_hold(
                df_eval,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                exchange=inst.exchange,
                timeframe=timeframe,
            )
            ema_res = run_ema_crossover_benchmark(
                df_eval,
                horizon_bars=horizon_bars,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                position_size=position_size,
                allow_short=allow_short,
                exchange=inst.exchange,
                timeframe=timeframe,
                benchmark_equity=bh_res.equity_curve,
            )
            strat_res = BacktestEngine(fee_bps=fee_bps, slippage_bps=slippage_bps).run_barriers(
                df=df_eval,
                signals=sig_series,
                horizon_bars=horizon_bars,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
                volatility=sigma.loc[df_eval.index],
                position_size=position_size,
                allow_short=allow_short,
                exchange=inst.exchange,
                timeframe=timeframe,
                confidence=conf_series,
                benchmark_equity=bh_res.equity_curve,
            )

            equity_df = pd.DataFrame(
                {
                    "timestamp": df_eval.index,
                    "strategy_equity": strat_res.equity_curve.to_numpy(dtype=np.float64),
                    "benchmark_equity": bh_res.equity_curve.to_numpy(dtype=np.float64),
                    "ema_equity": ema_res.equity_curve.to_numpy(dtype=np.float64),
                }
            )
            equity_path = bt_dir / "equity.parquet"
            equity_df.to_parquet(equity_path, index=False)

            metrics_payload: dict[str, Any] = {
                "strategy": strat_res.metrics,
                "buy_and_hold": bh_res.metrics,
                "ema_crossover": ema_res.metrics,
                "model_version_id": str(mv.id),
            }
            trades_list = strat_res.trades
        else:
            train_cfg = TrainingConfig(
                symbol=inst.symbol,
                timeframe=timeframe,
                horizon_bars=horizon_bars,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
                min_train_months=int(cfg_dict.get("min_train_months", 12)),
                val_months=int(cfg_dict.get("val_months", 2)),
                test_months=int(cfg_dict.get("test_months", 2)),
                embargo_bars=horizon_bars,
            )
            bt_cfg = BacktestStrategyConfig(
                entry_threshold=float(entry_thr) if entry_thr is not None else None,
                min_margin=min_margin,
                horizon_bars=horizon_bars,
                pt_mult=pt_mult,
                sl_mult=sl_mult,
                position_size=position_size,
                allow_short=allow_short,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                exchange=inst.exchange,
                timeframe=timeframe,
                random_seeds=20,
            )
            out = run_walk_forward_backtest(
                df=df,
                train_cfg=train_cfg,
                bt_cfg=bt_cfg,
                artifacts_dir=artifacts_root,
                run_id=str(bt.id),
            )
            equity_path = (out.artifact_dir or bt_dir) / "equity.parquet"
            metrics_payload = {
                "strategy": out.strategy_metrics,
                "buy_and_hold": out.buy_and_hold_metrics,
                "ema_crossover": out.ema_crossover_metrics,
                "random_benchmark": out.random_benchmark_metrics,
                "acceptance_a3": out.acceptance_a3,
                "warnings": out.warnings,
            }
            for f in out.fold_summaries:
                session.add(
                    BacktestFoldModel(
                        backtest_id=bt.id,
                        fold_no=int(f["fold_no"]),
                        train_start=datetime.fromisoformat(f["train_start"]),
                        train_end=datetime.fromisoformat(f["train_end"]),
                        val_start=datetime.fromisoformat(f["val_start"]),
                        val_end=datetime.fromisoformat(f["val_end"]),
                        test_start=datetime.fromisoformat(f["test_start"]),
                        test_end=datetime.fromisoformat(f["test_end"]),
                        metrics=f["metrics"],
                    )
                )
            trades_list = out.trades

        await _update_job(session, job_id, progress=0.85)

        for t in trades_list:
            session.add(
                BacktestTradeModel(
                    backtest_id=bt.id,
                    seq=t.seq,
                    instrument_id=inst.id,
                    side=t.side,
                    entry_at=pd.Timestamp(t.entry_at).to_pydatetime(),
                    entry_price=t.entry_price,
                    exit_at=pd.Timestamp(t.exit_at).to_pydatetime(),
                    exit_price=t.exit_price,
                    quantity=t.quantity,
                    fee=t.fee,
                    pnl=t.pnl,
                    exit_reason=t.exit_reason,
                    confidence=t.confidence,
                    fold_no=t.fold_no,
                )
            )

        report_path = bt_dir / "metrics.json"
        report_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        finished = datetime.now(UTC)
        bt.status = "done"
        bt.metrics = metrics_payload
        bt.equity_curve_uri = str(equity_path)
        bt.report_uri = str(report_path)
        bt.finished_at = finished
        await _update_job(
            session,
            job_id,
            status="done",
            progress=1.0,
            result={"backtest_id": str(bt.id)},
            finished_at=finished,
        )
        await session.commit()
    except Exception as exc:
        logger.exception("Backtest job failed (backtest_id={}): {}", backtest_id, exc)
        await session.rollback()
        finished_err = datetime.now(UTC)
        bt_err = await session.get(BacktestModel, backtest_id)
        if bt_err is not None:
            bt_err.status = "failed"
            bt_err.error = str(exc)
            bt_err.finished_at = finished_err
        job_err = await session.get(JobModel, job_id)
        if job_err is not None:
            job_err.status = "failed"
            job_err.error = str(exc)
            job_err.finished_at = finished_err
        await session.commit()
