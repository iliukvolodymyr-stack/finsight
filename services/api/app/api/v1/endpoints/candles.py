from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/instruments", tags=["Candles & Indicators"])


class CandleItem(BaseModel):
    t: str
    o: float
    h: float
    l: float
    c: float
    v: float
    closed: bool = True


class CandleListResponse(BaseModel):
    instrument_id: str
    symbol: str
    timeframe: str
    items: list[CandleItem]
    next_cursor: str | None = None
    has_more: bool = False


class IndicatorsResponse(BaseModel):
    instrument_id: str
    symbol: str
    timeframe: str
    items: list[dict[str, float | str | None]]


@router.get("/{id}/candles", response_model=CandleListResponse)
async def get_candles(
    id: str,
    timeframe: str = Query("1h", pattern="^(1m|15m|1h|4h|1d)$"),
    from_time: str | None = Query(None, alias="from"),
    to_time: str | None = Query(None, alias="to"),
    limit: int = Query(1000, ge=1, le=5000),
) -> CandleListResponse:
    # Skeleton mock candles for frontend development
    sample_candles = [
        CandleItem(
            t="2026-11-17T12:00:00Z",
            o=91000.0,
            h=91350.0,
            l=90950.0,
            c=91250.0,
            v=420.5,
            closed=True,
        ),
        CandleItem(
            t="2026-11-17T13:00:00Z",
            o=91250.5,
            h=91400.0,
            l=91180.5,
            c=91372.0,
            v=812.44,
            closed=True,
        ),
    ]
    return CandleListResponse(
        instrument_id=id,
        symbol="BTCUSDT",
        timeframe=timeframe,
        items=sample_candles,
        next_cursor=None,
        has_more=False,
    )


@router.get("/{id}/indicators", response_model=IndicatorsResponse)
async def get_indicators(
    id: str,
    timeframe: str = Query("1h", pattern="^(1m|15m|1h|4h|1d)$"),
) -> IndicatorsResponse:
    return IndicatorsResponse(
        instrument_id=id,
        symbol="BTCUSDT",
        timeframe=timeframe,
        items=[
            {
                "t": "2026-11-17T13:00:00Z",
                "ema20": 91100.0,
                "ema50": 90800.0,
                "rsi14": 62.4,
                "macd": 150.2,
            }
        ],
    )
