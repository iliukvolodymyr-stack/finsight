"""Train coursework LightGBM 1h models on TimescaleDB data, register versions,
run OOS walk-forward backtests, and seed initial predictions for the latest bars.

Usage (once Docker TimescaleDB is running):
    py -3 -m uv run python scripts/train_models.py
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from app.infra.db import AsyncSessionLocal, engine
from app.infra.models import (
    CandleModel,
    FeatureSetModel,
    InstrumentModel,
    ModelVersionModel,
    PredictionModel,
    new_uuid7,
)
from finsight_core.logging import setup_logging
from finsight_ml.backtest import BacktestStrategyConfig, run_walk_forward_backtest
from finsight_ml.features.pipeline import FeaturePipeline, compute_row_hash
from finsight_ml.labels.triple_barrier import compute_ewm_volatility
from finsight_ml.models.calibration import ProbabilityCalibrator
from finsight_ml.models.conformal import VolatilityNormalizedConformal
from finsight_ml.models.lgbm import LGBMForecaster
from finsight_ml.training import TrainingConfig, train_and_evaluate
from finsight_ml.validation.metrics import apply_signal_decision_rule
from loguru import logger
from rich.console import Console
from rich.table import Table
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

console = Console()

MODEL_CONFIGS = [
    Path("configs/models/lgbm_btc_1h.yaml"),
    Path("configs/models/lgbm_eth_1h.yaml"),
    Path("configs/models/lgbm_spy_1h.yaml"),
]


async def _load_symbol_candles(symbol: str, timeframe: str) -> tuple[InstrumentModel, pd.DataFrame]:
    async with AsyncSessionLocal() as session:
        inst = (
            await session.execute(select(InstrumentModel).where(InstrumentModel.symbol == symbol))
        ).scalar_one()
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
                )
                .order_by(CandleModel.open_time.asc())
            )
        ).all()

    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return inst, df.set_index("open_time").sort_index()


def _normalized_decimal_probs(
    p_buy: float, p_sell: float, p_hold: float
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """Round probabilities to 4 decimals while preserving exact sum == 1.0000 for DB constraint."""
    b = round(float(p_buy), 4)
    s = round(float(p_sell), 4)
    h = round(1.0 - b - s, 4)
    if h < 0.0:
        h = 0.0
        s = round(max(0.0, 1.0 - b), 4)
    conf = round(max(b, s, h), 4)
    return Decimal(f"{b:.4f}"), Decimal(f"{s:.4f}"), Decimal(f"{h:.4f}"), Decimal(f"{conf:.4f}")


async def _register_model_and_predictions(
    inst: InstrumentModel,
    df: pd.DataFrame,
    cfg: TrainingConfig,
    train_res_metrics: dict,
    train_result,
    seed_tail_bars: int = 48,
) -> None:
    pipeline = FeaturePipeline()
    async with AsyncSessionLocal() as session:
        # 1. Ensure feature_set exists
        fs = (
            await session.execute(
                select(FeatureSetModel).where(
                    FeatureSetModel.name == pipeline.name,
                    FeatureSetModel.version == pipeline.version,
                )
            )
        ).scalar_one_or_none()
        if fs is None:
            fs = FeatureSetModel(
                id=new_uuid7(),
                name=pipeline.name,
                version=pipeline.version,
                schema_hash=pipeline.schema_hash,
            )
            session.add(fs)
            await session.flush()

        # 2. Archive previous production version if promoting
        if cfg.status_target == "production":
            await session.execute(
                update(ModelVersionModel)
                .where(
                    ModelVersionModel.name == train_result.model_name,
                    ModelVersionModel.task == "direction_3class",
                    ModelVersionModel.instrument_id == inst.id,
                    ModelVersionModel.timeframe == cfg.timeframe,
                    ModelVersionModel.status == "production",
                )
                .values(status="archived")
            )

        existing_mv = (
            await session.execute(
                select(ModelVersionModel).where(
                    ModelVersionModel.name == train_result.model_name,
                    ModelVersionModel.task == "direction_3class",
                    ModelVersionModel.instrument_id == inst.id,
                    ModelVersionModel.timeframe == cfg.timeframe,
                    ModelVersionModel.semver == cfg.semver,
                )
            )
        ).scalar_one_or_none()

        if existing_mv is None:
            mv_id = new_uuid7()
            mv = ModelVersionModel(
                id=mv_id,
                name=train_result.model_name,
                task="direction_3class",
                algo="lightgbm",
                semver=cfg.semver,
                status=cfg.status_target,
                params={
                    **cfg.hyperparameters,
                    "entry_threshold": train_result.entry_threshold,
                    "min_margin": train_result.min_margin,
                    "pt_mult": cfg.pt_mult,
                    "sl_mult": cfg.sl_mult,
                    "horizon_bars": cfg.horizon_bars,
                },
                metrics=train_res_metrics,
                artifact_uri=str(train_result.artifact_dir),
                instrument_id=inst.id,
                timeframe=cfg.timeframe,
                feature_set_id=fs.id,
                dataset_hash=train_result.dataset_hash,
                train_data_end=train_result.train_data_end,
                promoted_at=datetime.now(UTC) if cfg.status_target == "production" else None,
            )
            session.add(mv)
            await session.flush()
        else:
            mv = existing_mv
            mv.status = cfg.status_target
            mv.metrics = train_res_metrics
            mv.artifact_uri = str(train_result.artifact_dir)
            mv.dataset_hash = train_result.dataset_hash
            mv.train_data_end = train_result.train_data_end
            mv.promoted_at = datetime.now(UTC) if cfg.status_target == "production" else None
            await session.flush()

        # 3. Generate recent inference predictions (chart markers + Forecast panel)
        model = LGBMForecaster.load(train_result.artifact_dir / "model.joblib")
        calibrator = ProbabilityCalibrator.load(train_result.artifact_dir / "calibrator.joblib")
        conformal = VolatilityNormalizedConformal.load(train_result.artifact_dir / "conformal.json")

        warmup_df = df.iloc[-(pipeline.serve_warmup_bars + seed_tail_bars) :]
        feats_all = pipeline.compute(warmup_df)
        sigma_all = compute_ewm_volatility(warmup_df["close"], span=24, min_periods=24)

        tail_idx = warmup_df.index[-seed_tail_bars:]
        X_tail = feats_all.loc[tail_idx]
        raw_probs = model.predict_proba(X_tail)
        cal_probs = calibrator.transform(raw_probs)
        sigs = apply_signal_decision_rule(
            cal_probs,
            entry_threshold=train_result.entry_threshold,
            min_margin=train_result.min_margin,
        )
        closes_tail = warmup_df["close"].loc[tail_idx].to_numpy(dtype=np.float64)
        sigs_vol = sigma_all.loc[tail_idx].to_numpy(dtype=np.float64)
        intervals = conformal.predict_interval(closes_tail, sigs_vol)
        explanations = model.explain_top_k(X_tail, top_k=5, target_signals=sigs)

        sig_map = {1: "BUY", 0: "HOLD", -1: "SELL"}
        pred_rows = []
        for i, as_of_ts in enumerate(tail_idx):
            p_sell_raw, p_hold_raw, p_buy_raw = cal_probs[i]
            p_buy_d, p_sell_d, p_hold_d, conf_d = _normalized_decimal_probs(
                p_buy_raw, p_sell_raw, p_hold_raw
            )
            pred_rows.append(
                {
                    "id": new_uuid7(),
                    "instrument_id": inst.id,
                    "timeframe": cfg.timeframe,
                    "as_of": as_of_ts.to_pydatetime(),
                    "horizon_bars": cfg.horizon_bars,
                    "model_version_id": mv.id,
                    "p_buy": p_buy_d,
                    "p_sell": p_sell_d,
                    "p_hold": p_hold_d,
                    "signal": sig_map[int(sigs[i])],
                    "confidence": conf_d,
                    "price_low": float(intervals.price_low[i]),
                    "price_high": float(intervals.price_high[i]),
                    "features_hash": compute_row_hash(X_tail.iloc[i]),
                    "component_versions": {train_result.model_name: str(mv.id)},
                    "explanation": explanations[i],
                }
            )

        stmt = pg_insert(PredictionModel).values(pred_rows)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["instrument_id", "timeframe", "as_of", "model_version_id"]
        )
        await session.execute(stmt)
        await session.commit()
        logger.info(
            "Registered {} v{} (status={}) and seeded {} predictions.",
            train_result.model_name,
            cfg.semver,
            cfg.status_target,
            len(pred_rows),
        )


async def main() -> None:
    setup_logging()
    summary_table = Table(title="Sprint C2 — OOS Model & Backtest Checkpoint")
    summary_table.add_column("Asset")
    summary_table.add_column("Status")
    summary_table.add_column("Macro-F1", justify="right")
    summary_table.add_column("Majority F1", justify="right")
    summary_table.add_column("ECE-10", justify="right")
    summary_table.add_column("Conf 80% Cov", justify="right")
    summary_table.add_column("OOS Return", justify="right")
    summary_table.add_column("OOS Sharpe", justify="right")
    summary_table.add_column("B&H Sharpe", justify="right")

    try:
        for cfg_path in MODEL_CONFIGS:
            cfg = TrainingConfig.from_yaml(cfg_path)
            inst, df = await _load_symbol_candles(cfg.symbol, cfg.timeframe)
            console.print(
                f"\n[bold cyan]=== {cfg.symbol} {cfg.timeframe} ({len(df)} bars) ===[/bold cyan]"
            )

            train_res = train_and_evaluate(df, cfg)
            bt_cfg = BacktestStrategyConfig(
                horizon_bars=cfg.horizon_bars,
                pt_mult=cfg.pt_mult,
                sl_mult=cfg.sl_mult,
                min_barrier_pct=cfg.min_barrier_pct,
                position_size=1.0,
                allow_short=False,
                exchange=inst.exchange,
                timeframe=cfg.timeframe,
                random_seeds=50,
            )
            bt_out = run_walk_forward_backtest(
                df, cfg, bt_cfg, run_id=f"c2_{cfg.symbol.lower()}_1h"
            )

            combined_metrics = {
                "oos_classification": train_res.oos_summary,
                "acceptance_checks": train_res.acceptance_checks,
                "oos_backtest": {
                    "strategy": bt_out.strategy_metrics,
                    "buy_and_hold": bt_out.buy_and_hold_metrics,
                    "ema_crossover": bt_out.ema_crossover_metrics,
                    "random_benchmark": bt_out.random_benchmark_metrics,
                    "acceptance_a3": bt_out.acceptance_a3,
                },
            }
            await _register_model_and_predictions(inst, df, cfg, combined_metrics, train_res)

            lgbm_m = train_res.oos_summary["models"]["lgbm_calibrated"]
            maj_m = train_res.oos_summary["models"]["majority"]
            cov_m = train_res.oos_summary["conformal_coverage"]
            summary_table.add_row(
                f"{cfg.symbol} {cfg.timeframe}",
                cfg.status_target,
                f"{lgbm_m['macro_f1']:.4f}",
                f"{maj_m['macro_f1']:.4f}",
                f"{lgbm_m['ece_10']:.4f}",
                f"{cov_m['mean'] * 100:.1f}%",
                f"{bt_out.strategy_metrics['total_return'] * 100:+.2f}%",
                f"{bt_out.strategy_metrics['sharpe']:.2f}",
                f"{bt_out.buy_and_hold_metrics['sharpe']:.2f}",
            )

        console.print()
        console.print(summary_table)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
