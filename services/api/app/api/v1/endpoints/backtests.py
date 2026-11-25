import contextlib
import json
import uuid
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
from app.core.deps import CurrentUser, get_current_user, get_db_session
from app.core.errors import AppException, NotFoundException
from app.infra.models import (
    BacktestFoldModel,
    BacktestModel,
    BacktestTradeModel,
    InstrumentModel,
    JobModel,
    new_uuid7,
)
from app.infra.redis import get_redis
from app.services.inference import get_production_model_for_scope
from app.services.job_runner import get_job_runner
from app.services.market_queries import resolve_instrument
from fastapi import APIRouter, Depends, Header, Query, status
from finsight_ml.backtest import validate_backtest_no_in_sample_leak
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/backtests", tags=["Backtests"])


class BacktestPeriod(BaseModel):
    from_date: str = Field(..., alias="from")
    to_date: str = Field(..., alias="to")

    model_config = {"populate_by_name": True}


class BacktestStrategyParams(BaseModel):
    entry_threshold: float | None = Field(default=0.55, ge=0.33, le=0.95)
    min_margin: float = Field(default=0.10, ge=0.0, le=0.50)
    exit_mode: str = "label_barriers"
    position_size: float = Field(default=1.0, gt=0.0, le=1.0)
    max_holding_bars: int = Field(default=24, ge=1, le=240)
    pt_mult: float = Field(default=1.0, gt=0.1, le=5.0)
    sl_mult: float = Field(default=1.0, gt=0.1, le=5.0)
    allow_short: bool = False


class BacktestCosts(BaseModel):
    fee_bps: float = Field(default=10.0, ge=0.0, le=100.0)
    slippage_bps: float = Field(default=5.0, ge=0.0, le=100.0)


class CreateBacktestRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    instrument_ids: list[str] = Field(..., min_length=1, max_length=10)
    timeframe: str = "1h"
    period: BacktestPeriod
    mode: str = "walk_forward_retrain"
    strategy: BacktestStrategyParams = Field(default_factory=BacktestStrategyParams)
    costs: BacktestCosts = Field(default_factory=BacktestCosts)


class CreateBacktestResponse(BaseModel):
    job_id: str
    backtest_id: str
    estimated_seconds: int = 90


class BacktestSummaryResponse(BaseModel):
    id: str
    name: str
    mode: str
    status: str
    timeframe: str
    period_start: str
    period_end: str
    created_at: str
    finished_at: str | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    total_return: float | None = None
    profit_factor: float | None = None
    trades_count: int | None = None


class BacktestDetailResponse(BacktestSummaryResponse):
    instrument_scope: list[str]
    strategy_config: dict[str, Any]
    metrics: dict[str, Any]
    error: str | None = None


class EquityPoint(BaseModel):
    t: str
    strategy: float
    benchmark: float
    ema_crossover: float | None = None


class BacktestEquityResponse(BaseModel):
    backtest_id: str
    items: list[EquityPoint]


class BacktestTradeItem(BaseModel):
    seq: int
    side: str
    entry_at: str
    entry_price: float
    exit_at: str
    exit_price: float
    quantity: float
    fee: float
    pnl: float
    exit_reason: str
    confidence: float | None = None
    fold_no: int | None = None


class BacktestTradesResponse(BaseModel):
    backtest_id: str
    total: int
    items: list[BacktestTradeItem]


class BacktestFoldItem(BaseModel):
    fold_no: int
    train_start: str
    train_end: str
    val_start: str | None = None
    val_end: str | None = None
    test_start: str
    test_end: str
    metrics: dict[str, Any]


def _parse_period_bound(value: str, end_of_day: bool = False) -> datetime:
    if len(value) == 10 and value[4] == "-" and value[7] == "-":
        d = datetime.strptime(value, "%Y-%m-%d").date()
        t = time(23, 59, 59) if end_of_day else time(0, 0, 0)
        return datetime.combine(d, t, tzinfo=UTC)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _to_summary(bt: BacktestModel) -> BacktestSummaryResponse:
    strat = (bt.metrics or {}).get("strategy", {})
    cfg = bt.strategy_config or {}
    return BacktestSummaryResponse(
        id=str(bt.id),
        name=bt.name,
        mode=bt.mode,
        status=bt.status,
        timeframe=str(cfg.get("timeframe", "1h")),
        period_start=bt.period_start.isoformat().replace("+00:00", "Z"),
        period_end=bt.period_end.isoformat().replace("+00:00", "Z"),
        created_at=bt.created_at.isoformat().replace("+00:00", "Z"),
        finished_at=bt.finished_at.isoformat().replace("+00:00", "Z") if bt.finished_at else None,
        sharpe=strat.get("sharpe"),
        max_drawdown=strat.get("max_drawdown"),
        win_rate=strat.get("win_rate"),
        total_return=strat.get("total_return"),
        profit_factor=strat.get("profit_factor"),
        trades_count=strat.get("trades_count"),
    )


async def _get_owned_backtest(
    session: AsyncSession,
    bt_id_str: str,
    user_id: uuid.UUID,
) -> BacktestModel:
    try:
        bt_id = uuid.UUID(bt_id_str)
    except ValueError as exc:
        raise NotFoundException(f"Backtest {bt_id_str} not found") from exc

    res = await session.execute(
        select(BacktestModel).where(BacktestModel.id == bt_id, BacktestModel.user_id == user_id)
    )
    bt = res.scalar_one_or_none()
    if bt is None:
        raise NotFoundException(f"Backtest {bt_id_str} not found")
    return bt


@router.post("", response_model=CreateBacktestResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_backtest(
    req: CreateBacktestRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> CreateBacktestResponse:
    if req.timeframe not in ("1h", "1d"):
        raise AppException(
            status_code=422,
            code="VALIDATION_ERROR",
            title="Invalid timeframe",
            detail="Backtest timeframe must be '1h' or '1d'.",
        )
    if req.mode not in ("static_model", "walk_forward_retrain"):
        raise AppException(
            status_code=422,
            code="VALIDATION_ERROR",
            title="Invalid mode",
            detail="Backtest mode must be 'static_model' or 'walk_forward_retrain'.",
        )

    period_start = _parse_period_bound(req.period.from_date, end_of_day=False)
    period_end = _parse_period_bound(req.period.to_date, end_of_day=True)
    if (period_end - period_start).days < 364:
        raise AppException(
            status_code=422,
            code="VALIDATION_ERROR",
            title="Period too short",
            detail="Backtest period must cover at least 12 months (365 days).",
        )

    user_uuid = uuid.UUID(current_user.id)
    idemp_cache_key = (
        f"finsight:idemp:backtest:{user_uuid}:{idempotency_key}" if idempotency_key else None
    )
    if idemp_cache_key:
        with contextlib.suppress(Exception):
            cached = await get_redis().get(idemp_cache_key)
            if cached:
                return CreateBacktestResponse(**json.loads(cached))

    resolved_insts: list[InstrumentModel] = [
        await resolve_instrument(session, item) for item in req.instrument_ids
    ]

    if req.mode == "static_model":
        mv, _ = await get_production_model_for_scope(session, resolved_insts[0], req.timeframe)
        try:
            validate_backtest_no_in_sample_leak(
                mode=req.mode,
                period_start=period_start,
                train_data_end=mv.train_data_end,
                embargo_bars=req.strategy.max_holding_bars,
                timeframe=req.timeframe,
            )
        except ValueError as exc:
            raise AppException(
                status_code=422,
                code="IN_SAMPLE_LEAKAGE_BLOCKED",
                title="In-sample static backtest blocked",
                detail=str(exc),
            ) from exc

    bt_id = new_uuid7()
    job_id = new_uuid7()
    strategy_cfg = {
        **req.strategy.model_dump(),
        "timeframe": req.timeframe,
        "fee_bps": req.costs.fee_bps,
        "slippage_bps": req.costs.slippage_bps,
    }

    bt = BacktestModel(
        id=bt_id,
        user_id=user_uuid,
        name=req.name,
        strategy_config=strategy_cfg,
        instrument_scope=[str(i.id) for i in resolved_insts],
        period_start=period_start,
        period_end=period_end,
        mode=req.mode,
        status="queued",
    )
    job = JobModel(
        id=job_id,
        user_id=user_uuid,
        kind="backtest",
        status="queued",
        progress=0.0,
        payload={"backtest_id": str(bt_id), "mode": req.mode},
    )
    session.add(bt)
    session.add(job)
    await session.commit()

    await get_job_runner().enqueue_backtest(bt_id, job_id)
    est_sec = 90 if req.mode == "walk_forward_retrain" else 15
    resp = CreateBacktestResponse(
        job_id=str(job_id), backtest_id=str(bt_id), estimated_seconds=est_sec
    )

    if idemp_cache_key:
        with contextlib.suppress(Exception):
            await get_redis().set(idemp_cache_key, resp.model_dump_json(), ex=86400)

    return resp


@router.get("", response_model=list[BacktestSummaryResponse])
async def list_backtests(
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[BacktestSummaryResponse]:
    user_uuid = uuid.UUID(current_user.id)
    rows = (
        (
            await session.execute(
                select(BacktestModel)
                .where(BacktestModel.user_id == user_uuid)
                .order_by(desc(BacktestModel.created_at))
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    return [_to_summary(bt) for bt in rows]


@router.get("/{id}", response_model=BacktestDetailResponse)
async def get_backtest_detail(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> BacktestDetailResponse:
    bt = await _get_owned_backtest(session, id, uuid.UUID(current_user.id))
    base = _to_summary(bt)
    return BacktestDetailResponse(
        **base.model_dump(),
        instrument_scope=[str(x) for x in (bt.instrument_scope or [])],
        strategy_config=dict(bt.strategy_config or {}),
        metrics=dict(bt.metrics or {}),
        error=bt.error,
    )


@router.get("/{id}/equity", response_model=BacktestEquityResponse)
async def get_backtest_equity(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> BacktestEquityResponse:
    bt = await _get_owned_backtest(session, id, uuid.UUID(current_user.id))
    if not bt.equity_curve_uri or not Path(bt.equity_curve_uri).exists():
        return BacktestEquityResponse(backtest_id=str(bt.id), items=[])

    df = pd.read_parquet(bt.equity_curve_uri)
    # Downsample gracefully if > 1500 points for snappy chart rendering
    step = max(1, len(df) // 1500)
    sampled = df.iloc[::step] if step > 1 else df
    items = [
        EquityPoint(
            t=pd.Timestamp(row["timestamp"]).isoformat().replace("+00:00", "Z"),
            strategy=round(float(row["strategy_equity"]), 2),
            benchmark=round(float(row["benchmark_equity"]), 2),
            ema_crossover=(
                round(float(row["ema_equity"]), 2)
                if "ema_equity" in df.columns and pd.notna(row["ema_equity"])
                else None
            ),
        )
        for _, row in sampled.iterrows()
    ]
    return BacktestEquityResponse(backtest_id=str(bt.id), items=items)


@router.get("/{id}/trades", response_model=BacktestTradesResponse)
async def get_backtest_trades(
    id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> BacktestTradesResponse:
    bt = await _get_owned_backtest(session, id, uuid.UUID(current_user.id))
    total = int(
        (
            await session.execute(
                select(func.count())
                .select_from(BacktestTradeModel)
                .where(BacktestTradeModel.backtest_id == bt.id)
            )
        ).scalar_one()
    )
    rows = (
        (
            await session.execute(
                select(BacktestTradeModel)
                .where(BacktestTradeModel.backtest_id == bt.id)
                .order_by(BacktestTradeModel.seq.asc())
                .offset(offset)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    items = [
        BacktestTradeItem(
            seq=t.seq,
            side=t.side,
            entry_at=t.entry_at.isoformat().replace("+00:00", "Z"),
            entry_price=t.entry_price,
            exit_at=t.exit_at.isoformat().replace("+00:00", "Z"),
            exit_price=t.exit_price,
            quantity=t.quantity,
            fee=t.fee,
            pnl=t.pnl,
            exit_reason=t.exit_reason,
            confidence=t.confidence,
            fold_no=t.fold_no,
        )
        for t in rows
    ]
    return BacktestTradesResponse(backtest_id=str(bt.id), total=total, items=items)


@router.get("/{id}/folds", response_model=list[BacktestFoldItem])
async def get_backtest_folds(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[BacktestFoldItem]:
    bt = await _get_owned_backtest(session, id, uuid.UUID(current_user.id))
    rows = (
        (
            await session.execute(
                select(BacktestFoldModel)
                .where(BacktestFoldModel.backtest_id == bt.id)
                .order_by(BacktestFoldModel.fold_no.asc())
            )
        )
        .scalars()
        .all()
    )
    return [
        BacktestFoldItem(
            fold_no=f.fold_no,
            train_start=f.train_start.isoformat().replace("+00:00", "Z"),
            train_end=f.train_end.isoformat().replace("+00:00", "Z"),
            val_start=f.val_start.isoformat().replace("+00:00", "Z") if f.val_start else None,
            val_end=f.val_end.isoformat().replace("+00:00", "Z") if f.val_end else None,
            test_start=f.test_start.isoformat().replace("+00:00", "Z"),
            test_end=f.test_end.isoformat().replace("+00:00", "Z"),
            metrics=dict(f.metrics or {}),
        )
        for f in rows
    ]
