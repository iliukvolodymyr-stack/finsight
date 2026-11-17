import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_instruments(client: AsyncClient):
    response = await client.get("/api/v1/instruments")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) >= 1
    symbols = [item["symbol"] for item in items]
    assert "BTCUSDT" in symbols


@pytest.mark.asyncio
async def test_get_candles(client: AsyncClient):
    response = await client.get("/api/v1/instruments/BTCUSDT/candles?timeframe=1h")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["timeframe"] == "1h"
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_predictions(client: AsyncClient):
    response = await client.get("/api/v1/predictions?timeframe=1h")
    assert response.status_code == 200
    items = response.json()
    assert len(items) >= 1
    assert items[0]["signal"] in ("BUY", "SELL", "HOLD")
