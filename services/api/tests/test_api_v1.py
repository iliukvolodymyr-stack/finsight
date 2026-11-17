import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_instruments(client: AsyncClient):
    response = await client.get("/api/v1/instruments")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) == 3
    symbols = [item["symbol"] for item in items]
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols
    assert "SPY" in symbols


@pytest.mark.asyncio
async def test_instruments_filtering_and_lookup(client: AsyncClient):
    res_etf = await client.get("/api/v1/instruments?asset_class=etf")
    assert res_etf.status_code == 200
    assert [i["symbol"] for i in res_etf.json()] == ["SPY"]

    res_eth = await client.get("/api/v1/instruments/ETHUSDT")
    assert res_eth.status_code == 200
    assert res_eth.json()["symbol"] == "ETHUSDT"

    res_quote = await client.get("/api/v1/instruments/ETHUSDT/quote")
    assert res_quote.status_code == 200
    assert res_quote.json()["symbol"] == "ETHUSDT"

    res_missing = await client.get("/api/v1/instruments/UNKNOWN")
    assert res_missing.status_code == 404
    assert "application/problem+json" in res_missing.headers["content-type"]
    assert res_missing.json()["code"] == "NOT_FOUND"


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
