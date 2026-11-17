from typing import Any

from app.core.deps import get_db_session
from app.core.errors import NotFoundException
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/instruments", tags=["Instruments"])


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


@router.get("", response_model=list[InstrumentResponse])
async def list_instruments(
    asset_class: str | None = Query(None),
    q: str | None = Query(None),
    is_active: bool = Query(True),
    session: AsyncSession = Depends(get_db_session),
) -> list[InstrumentResponse]:
    # In Sprint C0 skeleton, return default mock/db instruments
    return [
        InstrumentResponse(
            id="7f0c0000-0000-7000-8000-000000000001",
            symbol="BTCUSDT",
            name="Bitcoin",
            asset_class="crypto",
            exchange="binance",
            quote_currency="USDT",
            is_active=True,
            metadata={"sector": "layer1"},
        ),
        InstrumentResponse(
            id="7f0c0000-0000-7000-8000-000000000002",
            symbol="ETHUSDT",
            name="Ethereum",
            asset_class="crypto",
            exchange="binance",
            quote_currency="USDT",
            is_active=True,
            metadata={"sector": "smart_contracts"},
        ),
        InstrumentResponse(
            id="7f0c0000-0000-7000-8000-000000000003",
            symbol="SPY",
            name="SPDR S&P 500 ETF Trust",
            asset_class="etf",
            exchange="nyse",
            quote_currency="USD",
            is_active=True,
            metadata={"sector": "index"},
        ),
    ]


@router.get("/{id}", response_model=InstrumentResponse)
async def get_instrument(id: str) -> InstrumentResponse:
    if id == "7f0c0000-0000-7000-8000-000000000001" or id == "BTCUSDT":
        return InstrumentResponse(
            id="7f0c0000-0000-7000-8000-000000000001",
            symbol="BTCUSDT",
            name="Bitcoin",
            asset_class="crypto",
            exchange="binance",
            quote_currency="USDT",
            is_active=True,
        )
    raise NotFoundException(f"Instrument {id} not found")


@router.get("/{id}/quote", response_model=QuoteResponse)
async def get_quote(id: str) -> QuoteResponse:
    return QuoteResponse(
        symbol="BTCUSDT",
        price=91372.0,
        change_24h=1.84,
        volume_24h=28114.2,
        ts="2026-11-17T14:00:00Z",
        stale=False,
    )
