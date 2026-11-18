from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.deps import CurrentUser, get_current_user, get_db_session
from app.core.errors import AppException, NotFoundException, RateLimitedException
from app.infra.models import CandleModel, InstrumentModel
from app.infra.redis import get_redis
from app.services.ingestion import ingest_instrument_candles
from app.services.market_queries import resolve_instrument
from fastapi import APIRouter, Depends, Query, Response, status
from finsight_core.time import to_iso_utc
from finsight_ml.data import is_quote_stale
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/instruments", tags=["Instruments"])

REFRESH_COOLDOWN_SEC = 60
REFRESH_LOCK_TTL_SEC = 60


class InstrumentResponse(BaseModel):
    id: str
    symbol: str
    name: str
    asset_class: str
    exchange: str
    quote_currency: str
    is_active: bool
    metadata: dict[str, Any] = {}


class QuoteResponse(BaseModel):
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    ts: str
    stale: bool = False


class RefreshResponse(BaseModel):
    symbol: str
    timeframe: str
    status: str
    rows_written: int
    rows_rejected: int


def _to_instrument_response(inst: InstrumentModel) -> InstrumentResponse:
    return InstrumentResponse(
        id=str(inst.id),
        symbol=inst.symbol,
        name=inst.name,
        asset_class=inst.asset_class,
        exchange=inst.exchange,
        quote_currency=inst.quote_currency,
        is_active=inst.is_active,
        metadata=inst.metadata_ if isinstance(inst.metadata_, dict) else {},
    )


@router.get("", response_model=list[InstrumentResponse])
async def list_instruments(
    response: Response,
    asset_class: str | None = Query(None),
    q: str | None = Query(None),
    is_active: bool = Query(True),
    session: AsyncSession = Depends(get_db_session),
) -> list[InstrumentResponse]:
    stmt = select(InstrumentModel).where(InstrumentModel.is_active.is_(is_active))
    if asset_class:
        stmt = stmt.where(func.lower(InstrumentModel.asset_class) == asset_class.lower())
    if q:
        pattern = f"%{q.strip()}%"
        stmt = stmt.where(
            or_(
                InstrumentModel.symbol.ilike(pattern),
                InstrumentModel.name.ilike(pattern),
            )
        )
    stmt = stmt.order_by(InstrumentModel.symbol.asc())
    res = await session.execute(stmt)
    items = list(res.scalars().all())
    response.headers["Cache-Control"] = "public, max-age=300"
    return [_to_instrument_response(inst) for inst in items]


@router.get("/{id}", response_model=InstrumentResponse)
async def get_instrument(
    id: str,
    session: AsyncSession = Depends(get_db_session),
) -> InstrumentResponse:
    inst = await resolve_instrument(session, id)
    return _to_instrument_response(inst)


@router.get("/{id}/quote", response_model=QuoteResponse)
async def get_quote(
    id: str,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
) -> QuoteResponse:
    inst = await resolve_instrument(session, id)

    # Prefer latest closed 1h candle; fallback to 1d if 1h not available
    used_tf = "1h"
    latest_res = await session.execute(
        select(CandleModel)
        .where(
            CandleModel.instrument_id == inst.id,
            CandleModel.timeframe == "1h",
            CandleModel.is_closed.is_(True),
        )
        .order_by(CandleModel.open_time.desc())
        .limit(1)
    )
    latest = latest_res.scalar_one_or_none()

    if latest is None:
        used_tf = "1d"
        latest_res = await session.execute(
            select(CandleModel)
            .where(
                CandleModel.instrument_id == inst.id,
                CandleModel.timeframe == "1d",
                CandleModel.is_closed.is_(True),
            )
            .order_by(CandleModel.open_time.desc())
            .limit(1)
        )
        latest = latest_res.scalar_one_or_none()

    if latest is None:
        raise NotFoundException(f"No market data available for {inst.symbol}")

    cutoff_24h = latest.open_time - timedelta(hours=24)

    # Reference candle ~24h ago (for 1d: previous bar)
    ref_res = await session.execute(
        select(CandleModel.close)
        .where(
            CandleModel.instrument_id == inst.id,
            CandleModel.timeframe == used_tf,
            CandleModel.is_closed.is_(True),
            CandleModel.open_time <= cutoff_24h,
        )
        .order_by(CandleModel.open_time.desc())
        .limit(1)
    )
    ref_close = ref_res.scalar_one_or_none()

    if ref_close is not None and ref_close > 0:
        change_24h = round(((latest.close - ref_close) / ref_close) * 100.0, 4)
    else:
        change_24h = 0.0

    if used_tf == "1h":
        vol_res = await session.execute(
            select(func.coalesce(func.sum(CandleModel.volume), 0.0)).where(
                CandleModel.instrument_id == inst.id,
                CandleModel.timeframe == "1h",
                CandleModel.is_closed.is_(True),
                CandleModel.open_time > cutoff_24h,
                CandleModel.open_time <= latest.open_time,
            )
        )
        volume_24h = float(vol_res.scalar_one())
    else:
        volume_24h = float(latest.volume)

    stale = is_quote_stale(
        inst.exchange,
        latest.open_time,
        datetime.now(UTC),
        timeframe=used_tf,
    )

    response.headers["Cache-Control"] = "public, max-age=2"
    return QuoteResponse(
        symbol=inst.symbol,
        price=float(latest.close),
        change_24h=change_24h,
        volume_24h=round(volume_24h, 4),
        ts=to_iso_utc(latest.open_time),
        stale=stale,
    )


@router.post(
    "/{id}/refresh",
    response_model=RefreshResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def refresh_instrument(
    id: str,
    timeframe: str = Query("1h", pattern="^(1h|1d)$"),
    session: AsyncSession = Depends(get_db_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> RefreshResponse:
    """Force incremental candle refresh with Redis distributed lock and 60s cooldown."""
    inst = await resolve_instrument(session, id)
    redis = get_redis()

    cooldown_key = f"ratelimit:refresh:{inst.id}:{timeframe}"
    lock_key = f"lock:refresh:{inst.id}:{timeframe}"

    ttl = await redis.ttl(cooldown_key)
    if ttl and ttl > 0:
        raise RateLimitedException(
            retry_after=int(ttl),
            detail=f"Refresh for {inst.symbol} ({timeframe}) is rate-limited. Retry in {ttl}s.",
        )

    acquired = await redis.set(lock_key, current_user.id, nx=True, ex=REFRESH_LOCK_TTL_SEC)
    if not acquired:
        raise AppException(
            status_code=409,
            code="CONFLICT",
            title="Refresh already in progress",
            detail=f"A refresh for {inst.symbol} ({timeframe}) is already running.",
        )

    try:
        await redis.set(cooldown_key, "1", ex=REFRESH_COOLDOWN_SEC)
        lookback_days = 3 if timeframe == "1h" else 14
        summary = await ingest_instrument_candles(
            session,
            symbol=inst.symbol,
            timeframe=timeframe,
            days=lookback_days,
            job_name="manual_refresh",
            check_gaps=False,
        )
        return RefreshResponse(
            symbol=summary.symbol,
            timeframe=summary.timeframe,
            status=summary.status,
            rows_written=summary.rows_written,
            rows_rejected=summary.rows_rejected,
        )
    finally:
        await redis.delete(lock_key)
