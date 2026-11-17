from typing import Any

from app.core.deps import CurrentUser, get_current_user
from fastapi import APIRouter, Depends, Header, status
from pydantic import BaseModel

router = APIRouter(prefix="/backtests", tags=["Backtests"])


class CreateBacktestRequest(BaseModel):
    name: str
    instrument_ids: list[str]
    timeframe: str = "1h"
    period: dict[str, str]
    mode: str = "walk_forward_retrain"
    strategy: dict[str, Any]
    costs: dict[str, float] = {"fee_bps": 10.0, "slippage_bps": 5.0}


class CreateBacktestResponse(BaseModel):
    job_id: str
    estimated_seconds: int = 60


class BacktestSummaryResponse(BaseModel):
    id: str
    name: str
    status: str
    sharpe: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    trades_count: int | None = None


@router.post("", response_model=CreateBacktestResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_backtest(
    req: CreateBacktestRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    current_user: CurrentUser = Depends(get_current_user),
) -> CreateBacktestResponse:
    return CreateBacktestResponse(
        job_id="job_01933c00-0000-7000-8000-000000000001",
        estimated_seconds=60,
    )


@router.get("", response_model=list[BacktestSummaryResponse])
async def list_backtests(
    current_user: CurrentUser = Depends(get_current_user),
) -> list[BacktestSummaryResponse]:
    return [
        BacktestSummaryResponse(
            id="01933c00-0000-7000-8000-000000000001",
            name="BTC 1h LGBM walk-forward",
            status="completed",
            sharpe=1.82,
            max_drawdown=0.125,
            win_rate=0.584,
            trades_count=342,
        )
    ]
