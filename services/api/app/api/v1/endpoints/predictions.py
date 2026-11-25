import uuid

from app.core.deps import CurrentUser, get_current_user, get_db_session
from app.core.errors import AppException, NotFoundException
from app.infra.models import InstrumentModel
from app.services.inference import (
    get_latest_predictions,
    get_prediction_by_id,
    get_prediction_history,
    run_inference_for_instrument,
)
from app.services.market_queries import parse_iso_utc, resolve_instrument
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/predictions", tags=["Predictions"])


class PriceRange(BaseModel):
    low: float
    high: float
    coverage: float = 0.8


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    status: str = "production"


class ExplanationItem(BaseModel):
    feature: str
    contribution: float
    scale: str = "raw_log_odds"


class PredictionResponse(BaseModel):
    id: str
    instrument_id: str
    symbol: str
    timeframe: str
    as_of: str
    horizon_bars: int
    signal: str
    confidence: float
    probabilities: dict[str, float]
    price_range: PriceRange
    model: ModelInfo
    component_versions: dict[str, str] = Field(default_factory=dict)
    explanation: list[ExplanationItem] = Field(default_factory=list)
    explanation_scale: str = "raw_log_odds"


class RunInferenceRequest(BaseModel):
    instrument_id: str
    timeframe: str = "1h"


@router.get("", response_model=list[PredictionResponse])
async def list_predictions(
    instrument_ids: list[str] | None = Query(None),
    timeframe: str = Query("1h"),
    latest: bool = Query(True),
    session: AsyncSession = Depends(get_db_session),
) -> list[PredictionResponse]:
    _ = latest
    instruments: list[InstrumentModel] | None = None
    if instrument_ids:
        instruments = [await resolve_instrument(session, item) for item in instrument_ids]

    rows = await get_latest_predictions(session, instruments=instruments, timeframe=timeframe)
    return [PredictionResponse(**r) for r in rows]


@router.get("/history", response_model=list[PredictionResponse])
async def list_prediction_history(
    instrument_id: str = Query(...),
    timeframe: str = Query("1h"),
    from_ts: str | None = Query(None, alias="from"),
    to_ts: str | None = Query(None, alias="to"),
    limit: int = Query(300, ge=1, le=2000),
    session: AsyncSession = Depends(get_db_session),
) -> list[PredictionResponse]:
    inst = await resolve_instrument(session, instrument_id)
    from_dt = parse_iso_utc(from_ts, "from") if from_ts else None
    to_dt = parse_iso_utc(to_ts, "to") if to_ts else None
    rows = await get_prediction_history(
        session,
        instrument=inst,
        timeframe=timeframe,
        from_dt=from_dt,
        to_dt=to_dt,
        limit=limit,
    )
    return [PredictionResponse(**r) for r in rows]


@router.post("/run", response_model=PredictionResponse)
async def trigger_inference(
    req: RunInferenceRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PredictionResponse:
    if current_user.role != "admin":
        raise AppException(
            status_code=403,
            code="FORBIDDEN",
            title="Forbidden",
            detail="Only admin users can trigger manual inference.",
        )
    inst = await resolve_instrument(session, req.instrument_id)
    res = await run_inference_for_instrument(session, inst, timeframe=req.timeframe)
    return PredictionResponse(**res)


@router.get("/{id}", response_model=PredictionResponse)
async def get_prediction(
    id: str,
    session: AsyncSession = Depends(get_db_session),
) -> PredictionResponse:
    try:
        pred_uuid = uuid.UUID(id)
    except ValueError as exc:
        raise NotFoundException(f"Prediction {id} not found") from exc

    row = await get_prediction_by_id(session, pred_uuid)
    if row is None:
        raise NotFoundException(f"Prediction {id} not found")
    return PredictionResponse(**row)
