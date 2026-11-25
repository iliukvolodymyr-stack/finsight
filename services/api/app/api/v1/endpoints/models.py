from typing import Any

from app.core.deps import get_db_session
from app.infra.models import InstrumentModel, ModelVersionModel
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/models", tags=["Models"])


class ModelVersionResponse(BaseModel):
    id: str
    name: str
    task: str
    algo: str
    semver: str
    status: str
    instrument_id: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    train_data_end: str
    trained_at: str
    promoted_at: str | None = None
    metrics: dict[str, Any]
    notes: str | None = None


@router.get("", response_model=list[ModelVersionResponse])
async def list_models(
    status_filter: str | None = Query(None, alias="status"),
    session: AsyncSession = Depends(get_db_session),
) -> list[ModelVersionResponse]:
    stmt = (
        select(ModelVersionModel, InstrumentModel.symbol)
        .outerjoin(InstrumentModel, InstrumentModel.id == ModelVersionModel.instrument_id)
        .order_by(desc(ModelVersionModel.trained_at))
    )
    if status_filter:
        stmt = stmt.where(ModelVersionModel.status == status_filter)

    rows = (await session.execute(stmt)).all()
    return [
        ModelVersionResponse(
            id=str(mv.id),
            name=mv.name,
            task=mv.task,
            algo=mv.algo,
            semver=mv.semver,
            status=mv.status,
            instrument_id=str(mv.instrument_id) if mv.instrument_id else None,
            symbol=sym,
            timeframe=mv.timeframe,
            train_data_end=mv.train_data_end.isoformat().replace("+00:00", "Z"),
            trained_at=mv.trained_at.isoformat().replace("+00:00", "Z"),
            promoted_at=mv.promoted_at.isoformat().replace("+00:00", "Z")
            if mv.promoted_at
            else None,
            metrics=dict(mv.metrics or {}),
            notes=mv.notes,
        )
        for mv, sym in rows
    ]
