import math

import pandas as pd
from app.core.deps import get_db_session
from app.core.errors import AppException
from app.infra.models import CandleModel
from app.services.market_queries import parse_iso_utc, resolve_instrument
from fastapi import APIRouter, Depends, Query
from finsight_core.schemas import ErrorDetail
from finsight_core.time import to_iso_utc
from finsight_ml.features.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/instruments", tags=["Candles & Indicators"])

INDICATOR_WARMUP_BARS = 500


class CandleItem(BaseModel):
    t: str
    o: float
    h: float
    l: float
    c: float
    v: float
    adjusted_close: float | None = None
    closed: bool = True


class CandleListResponse(BaseModel):
    instrument_id: str
    symbol: str
    timeframe: str
    items: list[CandleItem]
    next_cursor: str | None = None
    has_more: bool = False


class IndicatorPoint(BaseModel):
    t: str
    ema20: float | None = None
    ema50: float | None = None
    ema200: float | None = None
    rsi14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    bb_upper: float | None = None
    bb_mid: float | None = None
    bb_lower: float | None = None
    atr14: float | None = None


class IndicatorsResponse(BaseModel):
    instrument_id: str
    symbol: str
    timeframe: str
    items: list[IndicatorPoint]


def _clean_float(val: object) -> float | None:
    if val is None:
        return None
    f = float(val)  # type: ignore[arg-type]
    if not math.isfinite(f):
        return None
    return f


@router.get("/{id}/candles", response_model=CandleListResponse)
async def get_candles(
    id: str,
    timeframe: str = Query("1h", pattern="^(1m|15m|1h|4h|1d)$"),
    from_time: str | None = Query(None, alias="from"),
    to_time: str | None = Query(None, alias="to"),
    limit: int = Query(1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
) -> CandleListResponse:
    inst = await resolve_instrument(session, id)

    from_dt = parse_iso_utc(from_time, "from") if from_time else None
    to_dt = parse_iso_utc(to_time, "to") if to_time else None
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise AppException(
            status_code=422,
            code="VALIDATION_ERROR",
            title="Request validation failed",
            detail="'from' timestamp must be <= 'to' timestamp",
            errors=[ErrorDetail(field="from", message="must be <= to", code="range")],
        )

    stmt = select(CandleModel).where(
        CandleModel.instrument_id == inst.id,
        CandleModel.timeframe == timeframe,
    )
    if from_dt is not None:
        stmt = stmt.where(CandleModel.open_time >= from_dt)
    if to_dt is not None:
        stmt = stmt.where(CandleModel.open_time <= to_dt)

    if from_dt is None:
        # Default chart view: latest `limit` bars returned in ascending chronological order.
        # next_cursor points to the (limit+1)-th older bar so `to=next_cursor` has zero overlap.
        stmt = stmt.order_by(CandleModel.open_time.desc()).limit(limit + 1)
        rows = list((await session.execute(stmt)).scalars().all())
        has_more = len(rows) > limit
        next_cursor = to_iso_utc(rows[limit].open_time) if has_more else None
        rows = rows[:limit]
        rows.reverse()
    else:
        # Forward pagination: next_cursor points to the (limit+1)-th bar so `from=next_cursor`
        # starts cleanly on the next bar with zero overlap.
        stmt = stmt.order_by(CandleModel.open_time.asc()).limit(limit + 1)
        rows = list((await session.execute(stmt)).scalars().all())
        has_more = len(rows) > limit
        next_cursor = to_iso_utc(rows[limit].open_time) if has_more else None
        rows = rows[:limit]

    items = [
        CandleItem(
            t=to_iso_utc(c.open_time),
            o=float(c.open),
            h=float(c.high),
            l=float(c.low),
            c=float(c.close),
            v=float(c.volume),
            adjusted_close=float(c.adjusted_close) if c.adjusted_close is not None else None,
            closed=bool(c.is_closed),
        )
        for c in rows
    ]

    return CandleListResponse(
        instrument_id=str(inst.id),
        symbol=inst.symbol,
        timeframe=timeframe,
        items=items,
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get("/{id}/indicators", response_model=IndicatorsResponse)
async def get_indicators(
    id: str,
    timeframe: str = Query("1h", pattern="^(1m|15m|1h|4h|1d)$"),
    from_time: str | None = Query(None, alias="from"),
    to_time: str | None = Query(None, alias="to"),
    limit: int = Query(1000, ge=1, le=5000),
    session: AsyncSession = Depends(get_db_session),
) -> IndicatorsResponse:
    """
    Compute technical indicators on closed bars only using finsight_ml.features.indicators,
    pre-loading INDICATOR_WARMUP_BARS (500) bars prior to the window so EMA200 is warmed up.
    """
    inst = await resolve_instrument(session, id)

    from_dt = parse_iso_utc(from_time, "from") if from_time else None
    to_dt = parse_iso_utc(to_time, "to") if to_time else None
    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise AppException(
            status_code=422,
            code="VALIDATION_ERROR",
            title="Request validation failed",
            detail="'from' timestamp must be <= 'to' timestamp",
            errors=[ErrorDetail(field="from", message="must be <= to", code="range")],
        )

    if from_dt is None:
        stmt = select(CandleModel).where(
            CandleModel.instrument_id == inst.id,
            CandleModel.timeframe == timeframe,
            CandleModel.is_closed.is_(True),
        )
        if to_dt is not None:
            stmt = stmt.where(CandleModel.open_time <= to_dt)
        stmt = stmt.order_by(CandleModel.open_time.desc()).limit(limit + INDICATOR_WARMUP_BARS)
        all_rows = list((await session.execute(stmt)).scalars().all())
        all_rows.reverse()
        warmup_len = max(0, len(all_rows) - limit)
    else:
        target_stmt = select(CandleModel).where(
            CandleModel.instrument_id == inst.id,
            CandleModel.timeframe == timeframe,
            CandleModel.is_closed.is_(True),
            CandleModel.open_time >= from_dt,
        )
        if to_dt is not None:
            target_stmt = target_stmt.where(CandleModel.open_time <= to_dt)
        target_stmt = target_stmt.order_by(CandleModel.open_time.asc()).limit(limit)
        target_rows = list((await session.execute(target_stmt)).scalars().all())

        if not target_rows:
            return IndicatorsResponse(
                instrument_id=str(inst.id),
                symbol=inst.symbol,
                timeframe=timeframe,
                items=[],
            )

        warmup_stmt = (
            select(CandleModel)
            .where(
                CandleModel.instrument_id == inst.id,
                CandleModel.timeframe == timeframe,
                CandleModel.is_closed.is_(True),
                CandleModel.open_time < target_rows[0].open_time,
            )
            .order_by(CandleModel.open_time.desc())
            .limit(INDICATOR_WARMUP_BARS)
        )
        warmup_rows = list((await session.execute(warmup_stmt)).scalars().all())
        warmup_rows.reverse()
        all_rows = warmup_rows + target_rows
        warmup_len = len(warmup_rows)

    if not all_rows:
        return IndicatorsResponse(
            instrument_id=str(inst.id),
            symbol=inst.symbol,
            timeframe=timeframe,
            items=[],
        )

    close_s = pd.Series([float(c.close) for c in all_rows], dtype="float64")
    high_s = pd.Series([float(c.high) for c in all_rows], dtype="float64")
    low_s = pd.Series([float(c.low) for c in all_rows], dtype="float64")

    ema20 = calculate_ema(close_s, 20)
    ema50 = calculate_ema(close_s, 50)
    ema200 = calculate_ema(close_s, 200)
    rsi14 = calculate_rsi(close_s, 14)
    macd_line, macd_sig, macd_hist = calculate_macd(close_s)
    bb_mid, bb_upper, bb_lower, _ = calculate_bollinger_bands(close_s, 20, 2.0)
    atr14 = calculate_atr(high_s, low_s, close_s, 14)

    items: list[IndicatorPoint] = []
    for idx in range(warmup_len, len(all_rows)):
        items.append(
            IndicatorPoint(
                t=to_iso_utc(all_rows[idx].open_time),
                ema20=_clean_float(ema20.iloc[idx]),
                ema50=_clean_float(ema50.iloc[idx]),
                ema200=_clean_float(ema200.iloc[idx]),
                rsi14=_clean_float(rsi14.iloc[idx]),
                macd=_clean_float(macd_line.iloc[idx]),
                macd_signal=_clean_float(macd_sig.iloc[idx]),
                macd_hist=_clean_float(macd_hist.iloc[idx]),
                bb_upper=_clean_float(bb_upper.iloc[idx]),
                bb_mid=_clean_float(bb_mid.iloc[idx]),
                bb_lower=_clean_float(bb_lower.iloc[idx]),
                atr14=_clean_float(atr14.iloc[idx]),
            )
        )

    return IndicatorsResponse(
        instrument_id=str(inst.id),
        symbol=inst.symbol,
        timeframe=timeframe,
        items=items,
    )
