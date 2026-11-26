import uuid
from datetime import UTC, datetime
from typing import Literal

from app.core.deps import CurrentUser, get_current_user, get_db_session
from app.core.errors import AppException, NotFoundException
from app.infra.models import InstrumentModel, PortfolioTransactionModel
from app.services.market_queries import parse_iso_utc, resolve_instrument
from app.services.portfolio import (
    _get_owned_portfolio,
    add_portfolio_transaction,
    cancel_portfolio_transaction,
    compute_and_store_portfolio_risk,
    compute_portfolio_correlation_view,
    compute_portfolio_pnl_breakdown,
    compute_portfolio_valuation,
    create_portfolio_record,
    list_portfolios_summary,
)
from fastapi import APIRouter, Depends, Header, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


class PortfolioResponse(BaseModel):
    id: str
    name: str
    base_currency: str = "USD"
    created_at: str
    updated_at: str | None = None
    total_value: float = 0.0
    total_cost_basis: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    positions_count: int = 0


class CreatePortfolioRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    base_currency: str = Field(default="USD", min_length=2, max_length=10)


class UpdatePortfolioRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    base_currency: str | None = Field(default=None, min_length=2, max_length=10)


class PositionItem(BaseModel):
    instrument_id: str
    symbol: str
    asset_class: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    weight: float


class PortfolioPositionsResponse(BaseModel):
    portfolio_id: str
    total_market_value: float
    total_cost_basis: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_fees: float
    items: list[PositionItem]


class CreateTransactionRequest(BaseModel):
    instrument_id: str
    side: Literal["BUY", "SELL"]
    quantity: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    fee: float = Field(default=0.0, ge=0.0)
    executed_at: str | None = None
    note: str | None = Field(default=None, max_length=500)


class TransactionResponse(BaseModel):
    id: str
    portfolio_id: str
    instrument_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    executed_at: str
    note: str | None = None


class RiskSnapshotResponse(BaseModel):
    portfolio_id: str | None = None
    as_of: str | None = None
    var_95: float
    cvar_95: float
    sharpe: float
    sortino: float
    max_drawdown: float
    annual_volatility: float
    risk_level: str
    weights: dict[str, float] = Field(default_factory=dict)
    observations: int = 0
    method: str = "historical_1d_90d"


class PnlBreakdownItem(BaseModel):
    key: str
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float


class PortfolioPnlResponse(BaseModel):
    portfolio_id: str
    period: str
    group_by: str
    total_market_value: float
    total_cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    total_fees: float
    breakdown: list[PnlBreakdownItem]


class CorrelationResponse(BaseModel):
    portfolio_id: str
    timeframe: str
    window: str
    symbols: list[str]
    matrix: list[list[float]]
    observations: int


def _parse_uuid(value: str, label: str = "Portfolio") -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise NotFoundException(f"{label} {value} not found") from exc


def _parse_window_days(window: str) -> int:
    clean = window.strip().lower()
    if clean.endswith("d") and clean[:-1].isdigit():
        days = int(clean[:-1])
        if 7 <= days <= 730:
            return days
    raise AppException(
        status_code=422,
        code="VALIDATION_ERROR",
        title="Invalid correlation window",
        detail="Window must be like '30d', '90d', or '180d' (between 7d and 730d).",
    )


@router.get("", response_model=list[PortfolioResponse])
async def list_portfolios(
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[PortfolioResponse]:
    rows = await list_portfolios_summary(session, uuid.UUID(current_user.id))
    return [PortfolioResponse(**r) for r in rows]


@router.post("", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    req: CreatePortfolioRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PortfolioResponse:
    port = await create_portfolio_record(
        session, uuid.UUID(current_user.id), name=req.name, base_currency=req.base_currency
    )
    return PortfolioResponse(
        id=str(port.id),
        name=port.name,
        base_currency=port.base_currency,
        created_at=port.created_at.isoformat().replace("+00:00", "Z"),
        updated_at=port.updated_at.isoformat().replace("+00:00", "Z"),
    )


@router.get("/{id}", response_model=PortfolioResponse)
async def get_portfolio(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PortfolioResponse:
    port_id = _parse_uuid(id)
    user_id = uuid.UUID(current_user.id)
    port = await _get_owned_portfolio(session, user_id, port_id)
    val = await compute_portfolio_valuation(session, user_id, port_id)
    return PortfolioResponse(
        id=str(port.id),
        name=port.name,
        base_currency=port.base_currency,
        created_at=port.created_at.isoformat().replace("+00:00", "Z"),
        updated_at=port.updated_at.isoformat().replace("+00:00", "Z"),
        total_value=round(val.total_market_value, 2),
        total_cost_basis=round(val.total_cost_basis, 2),
        unrealized_pnl=round(val.total_unrealized_pnl, 2),
        realized_pnl=round(val.total_realized_pnl, 2),
        positions_count=len(val.positions),
    )


@router.patch("/{id}", response_model=PortfolioResponse)
async def update_portfolio(
    id: str,
    req: UpdatePortfolioRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PortfolioResponse:
    port_id = _parse_uuid(id)
    user_id = uuid.UUID(current_user.id)
    port = await _get_owned_portfolio(session, user_id, port_id)
    if req.name is not None:
        port.name = req.name.strip()
    if req.base_currency is not None:
        currency = req.base_currency.upper().strip()
        if currency != "USD":
            raise AppException(
                status_code=422,
                code="UNSUPPORTED_CURRENCY",
                title="Unsupported base currency",
                detail="Currently only 'USD' base currency is supported (no FX rate conversion yet).",
            )
        port.base_currency = currency
    port.updated_at = datetime.now(UTC)
    await session.commit()
    return await get_portfolio(id=id, current_user=current_user, session=session)


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    port_id = _parse_uuid(id)
    port = await _get_owned_portfolio(session, uuid.UUID(current_user.id), port_id)
    port.archived_at = datetime.now(UTC)
    await session.commit()


@router.get("/{id}/positions", response_model=PortfolioPositionsResponse)
async def get_portfolio_positions(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PortfolioPositionsResponse:
    port_id = _parse_uuid(id)
    val = await compute_portfolio_valuation(session, uuid.UUID(current_user.id), port_id)
    return PortfolioPositionsResponse(
        portfolio_id=str(port_id),
        total_market_value=round(val.total_market_value, 2),
        total_cost_basis=round(val.total_cost_basis, 2),
        total_unrealized_pnl=round(val.total_unrealized_pnl, 2),
        total_realized_pnl=round(val.total_realized_pnl, 2),
        total_fees=round(val.total_fees, 2),
        items=[
            PositionItem(
                instrument_id=p.instrument_id,
                symbol=p.symbol,
                asset_class=p.asset_class,
                quantity=round(p.quantity, 8),
                avg_price=round(p.avg_price, 4),
                current_price=round(p.current_price, 4),
                market_value=round(p.market_value, 2),
                cost_basis=round(p.cost_basis, 2),
                unrealized_pnl=round(p.unrealized_pnl, 2),
                unrealized_pnl_pct=round(p.unrealized_pnl_pct, 4),
                realized_pnl=round(p.realized_pnl, 2),
                weight=round(p.weight, 4),
            )
            for p in val.positions
        ],
    )


@router.get("/{id}/transactions", response_model=list[TransactionResponse])
async def list_portfolio_transactions(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[TransactionResponse]:
    port_id = _parse_uuid(id)
    await _get_owned_portfolio(session, uuid.UUID(current_user.id), port_id)
    rows = (
        await session.execute(
            select(PortfolioTransactionModel, InstrumentModel.symbol)
            .join(InstrumentModel, InstrumentModel.id == PortfolioTransactionModel.instrument_id)
            .where(PortfolioTransactionModel.portfolio_id == port_id)
            .order_by(desc(PortfolioTransactionModel.executed_at))
            .limit(200)
        )
    ).all()
    return [
        TransactionResponse(
            id=str(tx.id),
            portfolio_id=str(tx.portfolio_id),
            instrument_id=str(tx.instrument_id),
            symbol=sym,
            side=tx.side,
            quantity=float(tx.quantity),
            price=float(tx.price),
            fee=float(tx.fee),
            executed_at=tx.executed_at.isoformat().replace("+00:00", "Z"),
            note=tx.note,
        )
        for tx, sym in rows
    ]


@router.post(
    "/{id}/transactions", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED
)
async def create_portfolio_transaction(
    id: str,
    req: CreateTransactionRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> TransactionResponse:
    port_id = _parse_uuid(id)
    inst = await resolve_instrument(session, req.instrument_id)
    exec_dt = (
        parse_iso_utc(req.executed_at, "executed_at") if req.executed_at else datetime.now(UTC)
    )

    tx = await add_portfolio_transaction(
        session=session,
        user_id=uuid.UUID(current_user.id),
        portfolio_id=port_id,
        instrument=inst,
        side=req.side,
        quantity=req.quantity,
        price=req.price,
        fee=req.fee,
        executed_at=exec_dt,
        note=req.note,
        idempotency_key=idempotency_key,
    )
    return TransactionResponse(
        id=str(tx.id),
        portfolio_id=str(tx.portfolio_id),
        instrument_id=str(tx.instrument_id),
        symbol=inst.symbol,
        side=tx.side,
        quantity=float(tx.quantity),
        price=float(tx.price),
        fee=float(tx.fee),
        executed_at=tx.executed_at.isoformat().replace("+00:00", "Z"),
        note=tx.note,
    )


@router.delete("/{id}/transactions/{tx_id}", response_model=TransactionResponse)
async def delete_portfolio_transaction(
    id: str,
    tx_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> TransactionResponse:
    port_id = _parse_uuid(id)
    t_uuid = _parse_uuid(tx_id, label="Transaction")
    rev = await cancel_portfolio_transaction(session, uuid.UUID(current_user.id), port_id, t_uuid)
    inst = await session.get(InstrumentModel, rev.instrument_id)
    return TransactionResponse(
        id=str(rev.id),
        portfolio_id=str(rev.portfolio_id),
        instrument_id=str(rev.instrument_id),
        symbol=inst.symbol if inst else "",
        side=rev.side,
        quantity=float(rev.quantity),
        price=float(rev.price),
        fee=float(rev.fee),
        executed_at=rev.executed_at.isoformat().replace("+00:00", "Z"),
        note=rev.note,
    )


@router.get("/{id}/risk", response_model=RiskSnapshotResponse)
async def get_portfolio_risk(
    id: str,
    window: str = Query("90d"),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> RiskSnapshotResponse:
    port_id = _parse_uuid(id)
    window_days = _parse_window_days(window)
    res = await compute_and_store_portfolio_risk(
        session, uuid.UUID(current_user.id), port_id, window_days=window_days
    )
    return RiskSnapshotResponse(**res)


@router.get("/{id}/pnl", response_model=PortfolioPnlResponse)
async def get_portfolio_pnl(
    id: str,
    period: str = Query("30d"),
    group_by: str = Query("instrument"),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> PortfolioPnlResponse:
    port_id = _parse_uuid(id)
    if group_by not in ("instrument", "asset_class"):
        raise AppException(
            status_code=422,
            code="VALIDATION_ERROR",
            title="Invalid group_by",
            detail="group_by must be 'instrument' or 'asset_class'.",
        )
    res = await compute_portfolio_pnl_breakdown(
        session, uuid.UUID(current_user.id), port_id, period=period, group_by=group_by
    )
    return PortfolioPnlResponse(**res)


@router.get("/{id}/correlation", response_model=CorrelationResponse)
async def get_portfolio_correlation(
    id: str,
    window: str = Query("90d"),
    timeframe: str = Query("1d"),
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> CorrelationResponse:
    port_id = _parse_uuid(id)
    window_days = _parse_window_days(window)
    res = await compute_portfolio_correlation_view(
        session,
        uuid.UUID(current_user.id),
        port_id,
        window_days=window_days,
        timeframe=timeframe,
    )
    return CorrelationResponse(**res)
