import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from app.core.errors import AppException
from app.infra.models import (
    CandleModel,
    FeatureSetModel,
    InstrumentModel,
    ModelVersionModel,
    PredictionModel,
    new_uuid7,
)
from finsight_ml.features.pipeline import FeaturePipeline, compute_row_hash
from finsight_ml.labels.triple_barrier import compute_ewm_volatility
from finsight_ml.models.calibration import ProbabilityCalibrator
from finsight_ml.models.conformal import VolatilityNormalizedConformal
from finsight_ml.models.lgbm import LGBMForecaster
from finsight_ml.validation.metrics import apply_signal_decision_rule
from sqlalchemy import and_, desc, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession


def serialize_prediction_row(
    pred: PredictionModel,
    inst: InstrumentModel,
    mv: ModelVersionModel,
) -> dict[str, Any]:
    return {
        "id": str(pred.id),
        "instrument_id": str(pred.instrument_id),
        "symbol": inst.symbol,
        "timeframe": pred.timeframe,
        "as_of": pred.as_of.isoformat().replace("+00:00", "Z"),
        "horizon_bars": pred.horizon_bars,
        "signal": pred.signal,
        "confidence": float(pred.confidence),
        "probabilities": {
            "buy": float(pred.p_buy),
            "sell": float(pred.p_sell),
            "hold": float(pred.p_hold),
        },
        "price_range": {
            "low": float(pred.price_low) if pred.price_low is not None else 0.0,
            "high": float(pred.price_high) if pred.price_high is not None else 0.0,
            "coverage": 0.8,
        },
        "model": {
            "id": str(mv.id),
            "name": mv.name,
            "version": mv.semver,
            "status": mv.status,
        },
        "component_versions": dict(pred.component_versions or {}),
        "explanation": list(pred.explanation or []),
        "explanation_scale": "raw_log_odds",
    }


def _model_not_ready(symbol: str, timeframe: str, detail: str | None = None) -> AppException:
    return AppException(
        status_code=503,
        code="MODEL_NOT_READY",
        title="Model not ready",
        detail=detail or f"No active production model available for {symbol} {timeframe}.",
    )


async def get_production_model_for_scope(
    session: AsyncSession,
    instrument: InstrumentModel,
    timeframe: str,
) -> tuple[ModelVersionModel, FeatureSetModel]:
    stmt = (
        select(ModelVersionModel, FeatureSetModel)
        .join(FeatureSetModel, FeatureSetModel.id == ModelVersionModel.feature_set_id)
        .where(
            ModelVersionModel.instrument_id == instrument.id,
            ModelVersionModel.timeframe == timeframe,
            ModelVersionModel.task == "direction_3class",
            ModelVersionModel.status == "production",
        )
        .order_by(desc(ModelVersionModel.promoted_at), desc(ModelVersionModel.trained_at))
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        # Check if a candidate exists so we can surface an informative detail message
        cand = (
            await session.execute(
                select(ModelVersionModel.semver, ModelVersionModel.status)
                .where(
                    ModelVersionModel.instrument_id == instrument.id,
                    ModelVersionModel.timeframe == timeframe,
                    ModelVersionModel.task == "direction_3class",
                )
                .limit(1)
            )
        ).first()
        if cand is not None:
            raise _model_not_ready(
                instrument.symbol,
                timeframe,
                f"Only experimental '{cand[1]}' model (v{cand[0]}) exists for "
                f"{instrument.symbol} {timeframe}; no production model is promoted.",
            )
        raise _model_not_ready(instrument.symbol, timeframe)
    return row[0], row[1]


async def get_latest_predictions(
    session: AsyncSession,
    instruments: list[InstrumentModel] | None,
    timeframe: str = "1h",
) -> list[dict[str, Any]]:
    subq = (
        select(
            PredictionModel.instrument_id,
            func.max(PredictionModel.as_of).label("max_as_of"),
        )
        .where(PredictionModel.timeframe == timeframe)
        .group_by(PredictionModel.instrument_id)
        .subquery()
    )

    stmt = (
        select(PredictionModel, InstrumentModel, ModelVersionModel)
        .join(
            subq,
            and_(
                PredictionModel.instrument_id == subq.c.instrument_id,
                PredictionModel.as_of == subq.c.max_as_of,
            ),
        )
        .join(InstrumentModel, InstrumentModel.id == PredictionModel.instrument_id)
        .join(ModelVersionModel, ModelVersionModel.id == PredictionModel.model_version_id)
        .where(
            PredictionModel.timeframe == timeframe,
            ModelVersionModel.status == "production",
        )
        .order_by(InstrumentModel.symbol.asc())
    )
    if instruments:
        stmt = stmt.where(PredictionModel.instrument_id.in_([i.id for i in instruments]))

    rows = (await session.execute(stmt)).all()
    if instruments and not rows:
        # Ensure we surface 503 MODEL_NOT_READY when a specific instrument has no production prediction
        await get_production_model_for_scope(session, instruments[0], timeframe)
        raise _model_not_ready(
            instruments[0].symbol,
            timeframe,
            f"Production model exists for {instruments[0].symbol} {timeframe}, "
            "but no predictions have been generated yet.",
        )

    return [serialize_prediction_row(p, inst, mv) for p, inst, mv in rows]


async def get_prediction_history(
    session: AsyncSession,
    instrument: InstrumentModel,
    timeframe: str = "1h",
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    limit: int = 300,
) -> list[dict[str, Any]]:
    stmt = (
        select(PredictionModel, InstrumentModel, ModelVersionModel)
        .join(InstrumentModel, InstrumentModel.id == PredictionModel.instrument_id)
        .join(ModelVersionModel, ModelVersionModel.id == PredictionModel.model_version_id)
        .where(
            PredictionModel.instrument_id == instrument.id,
            PredictionModel.timeframe == timeframe,
            ModelVersionModel.status == "production",
        )
        .order_by(desc(PredictionModel.as_of))
        .limit(limit)
    )
    if from_dt is not None:
        stmt = stmt.where(PredictionModel.as_of >= from_dt)
    if to_dt is not None:
        stmt = stmt.where(PredictionModel.as_of <= to_dt)

    rows = list((await session.execute(stmt)).all())
    rows.reverse()
    return [serialize_prediction_row(p, inst, mv) for p, inst, mv in rows]


async def get_prediction_by_id(
    session: AsyncSession,
    pred_id: uuid.UUID,
) -> dict[str, Any] | None:
    stmt = (
        select(PredictionModel, InstrumentModel, ModelVersionModel)
        .join(InstrumentModel, InstrumentModel.id == PredictionModel.instrument_id)
        .join(ModelVersionModel, ModelVersionModel.id == PredictionModel.model_version_id)
        .where(PredictionModel.id == pred_id)
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    return serialize_prediction_row(row[0], row[1], row[2])


async def run_inference_for_instrument(
    session: AsyncSession,
    instrument: InstrumentModel,
    timeframe: str = "1h",
) -> dict[str, Any]:
    mv, fs = await get_production_model_for_scope(session, instrument, timeframe)
    pipeline = FeaturePipeline()
    if fs.schema_hash != pipeline.schema_hash:
        raise _model_not_ready(
            instrument.symbol,
            timeframe,
            f"Feature schema mismatch (model expects {fs.schema_hash[:12]}, "
            f"runtime has {pipeline.schema_hash[:12]}).",
        )

    artifact_dir = Path(mv.artifact_uri)
    model_path = artifact_dir / "model.joblib"
    cal_path = artifact_dir / "calibrator.joblib"
    conf_path = artifact_dir / "conformal.json"
    if not (model_path.exists() and cal_path.exists() and conf_path.exists()):
        raise _model_not_ready(
            instrument.symbol,
            timeframe,
            f"Model artifact files missing at {artifact_dir}.",
        )

    warmup_limit = pipeline.serve_warmup_bars + 50
    candle_rows = (
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
                CandleModel.instrument_id == instrument.id,
                CandleModel.timeframe == timeframe,
                CandleModel.is_closed.is_(True),
            )
            .order_by(desc(CandleModel.open_time))
            .limit(warmup_limit)
        )
    ).all()
    if len(candle_rows) < pipeline.max_lookback + 10:
        raise _model_not_ready(
            instrument.symbol,
            timeframe,
            f"Insufficient closed candles ({len(candle_rows)}) for feature warmup.",
        )

    candle_rows = list(reversed(candle_rows))
    df = pd.DataFrame(candle_rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()

    feats = pipeline.compute(df)
    sigma = compute_ewm_volatility(df["close"], span=24, min_periods=24)
    last_ts = df.index[-1]
    X_last = feats.iloc[[-1]]
    if X_last.isna().any().any() or pd.isna(sigma.iloc[-1]):
        raise _model_not_ready(
            instrument.symbol,
            timeframe,
            "Latest feature vector contains NaN values.",
        )

    model = LGBMForecaster.load(model_path)
    calibrator = ProbabilityCalibrator.load(cal_path)
    conformal = VolatilityNormalizedConformal.load(conf_path)

    raw_probs = model.predict_proba(X_last)
    cal_probs = calibrator.transform(raw_probs)

    entry_thr = float((mv.params or {}).get("entry_threshold", 0.55))
    min_margin = float((mv.params or {}).get("min_margin", 0.10))
    horizon_bars = int((mv.params or {}).get("horizon_bars", 24))
    sig_int = int(
        apply_signal_decision_rule(cal_probs, entry_threshold=entry_thr, min_margin=min_margin)[0]
    )
    sig_str = {1: "BUY", 0: "HOLD", -1: "SELL"}[sig_int]

    close_val = np.array([float(df["close"].iloc[-1])], dtype=np.float64)
    sigma_val = np.array([float(sigma.iloc[-1])], dtype=np.float64)
    interval = conformal.predict_interval(close_val, sigma_val)
    expl = model.explain_top_k(X_last, top_k=5, target_signals=np.array([sig_int]))[0]

    p_sell_raw, p_hold_raw, p_buy_raw = cal_probs[0]
    p_buy_r = round(float(p_buy_raw), 4)
    p_sell_r = round(float(p_sell_raw), 4)
    p_hold_r = round(1.0 - p_buy_r - p_sell_r, 4)
    if p_hold_r < 0.0:
        p_hold_r = 0.0
        p_sell_r = round(max(0.0, 1.0 - p_buy_r), 4)
    conf_r = round(max(p_buy_r, p_sell_r, p_hold_r), 4)
    features_hash = compute_row_hash(X_last.iloc[0])

    stmt = (
        pg_insert(PredictionModel)
        .values(
            id=new_uuid7(),
            instrument_id=instrument.id,
            timeframe=timeframe,
            as_of=last_ts.to_pydatetime(),
            horizon_bars=horizon_bars,
            model_version_id=mv.id,
            p_buy=p_buy_r,
            p_sell=p_sell_r,
            p_hold=p_hold_r,
            signal=sig_str,
            confidence=conf_r,
            price_low=float(interval.price_low[0]),
            price_high=float(interval.price_high[0]),
            features_hash=features_hash,
            component_versions={mv.name: str(mv.id)},
            explanation=expl,
        )
        .on_conflict_do_nothing(
            index_elements=["instrument_id", "timeframe", "as_of", "model_version_id"]
        )
    )
    await session.execute(stmt)
    await session.commit()

    stored = (
        await session.execute(
            select(PredictionModel).where(
                PredictionModel.instrument_id == instrument.id,
                PredictionModel.timeframe == timeframe,
                PredictionModel.as_of == last_ts.to_pydatetime(),
                PredictionModel.model_version_id == mv.id,
            )
        )
    ).scalar_one()
    return serialize_prediction_row(stored, instrument, mv)
