from unittest.mock import AsyncMock

import pytest
from app.api.v1.endpoints import instruments as instruments_ep
from app.infra.redis import get_redis
from app.services.ingestion import BackfillSummary
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_instruments(client: AsyncClient):
    response = await client.get("/api/v1/instruments")
    assert response.status_code == 200
    assert "max-age=300" in response.headers.get("cache-control", "")
    items = response.json()
    assert isinstance(items, list)
    assert len(items) == 3
    symbols = [item["symbol"] for item in items]
    assert symbols == ["BTCUSDT", "ETHUSDT", "SPY"]


@pytest.mark.asyncio
async def test_instruments_filtering_and_lookup(client: AsyncClient):
    res_etf = await client.get("/api/v1/instruments?asset_class=etf")
    assert res_etf.status_code == 200
    assert [i["symbol"] for i in res_etf.json()] == ["SPY"]

    res_q = await client.get("/api/v1/instruments?q=ether")
    assert res_q.status_code == 200
    assert [i["symbol"] for i in res_q.json()] == ["ETHUSDT"]

    res_eth = await client.get("/api/v1/instruments/ETHUSDT")
    assert res_eth.status_code == 200
    eth_data = res_eth.json()
    assert eth_data["symbol"] == "ETHUSDT"

    # Lookup by UUID must match symbol lookup
    res_by_uuid = await client.get(f"/api/v1/instruments/{eth_data['id']}")
    assert res_by_uuid.status_code == 200
    assert res_by_uuid.json()["symbol"] == "ETHUSDT"

    res_missing = await client.get("/api/v1/instruments/UNKNOWN")
    assert res_missing.status_code == 404
    assert "application/problem+json" in res_missing.headers["content-type"]
    assert res_missing.json()["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_get_quote_from_timescaledb(client: AsyncClient):
    res_btc = await client.get("/api/v1/instruments/BTCUSDT/quote")
    assert res_btc.status_code == 200
    assert "max-age=2" in res_btc.headers.get("cache-control", "")
    btc = res_btc.json()
    assert btc["symbol"] == "BTCUSDT"
    assert btc["price"] > 0
    assert btc["volume_24h"] > 0
    assert isinstance(btc["stale"], bool)

    res_spy = await client.get("/api/v1/instruments/SPY/quote")
    assert res_spy.status_code == 200
    spy = res_spy.json()
    assert spy["symbol"] == "SPY"
    assert spy["price"] > 0


@pytest.mark.asyncio
async def test_get_candles_and_adjusted_close(client: AsyncClient):
    response = await client.get("/api/v1/instruments/BTCUSDT/candles?timeframe=1h&limit=50")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTCUSDT"
    assert data["timeframe"] == "1h"
    assert len(data["items"]) == 50
    assert data["has_more"] is True
    # Must be strictly chronologically ascending
    times = [c["t"] for c in data["items"]]
    assert times == sorted(times)
    assert data["items"][0]["adjusted_close"] is None

    # SPY 1d must populate adjusted_close; SPY 1h must keep it None
    res_spy_1d = await client.get("/api/v1/instruments/SPY/candles?timeframe=1d&limit=10")
    assert res_spy_1d.status_code == 200
    spy_1d_items = res_spy_1d.json()["items"]
    assert len(spy_1d_items) == 10
    assert all(c["adjusted_close"] is not None for c in spy_1d_items)

    res_spy_1h = await client.get("/api/v1/instruments/SPY/candles?timeframe=1h&limit=10")
    assert res_spy_1h.status_code == 200
    assert all(c["adjusted_close"] is None for c in res_spy_1h.json()["items"])

    # Invalid range (from > to) -> 422 RFC 9457
    res_bad = await client.get(
        "/api/v1/instruments/BTCUSDT/candles?timeframe=1h"
        "&from=2026-06-10T00:00:00Z&to=2026-06-01T00:00:00Z"
    )
    assert res_bad.status_code == 422
    assert res_bad.json()["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_indicators_pre_from_ema200_warmup(client: AsyncClient):
    window_from = "2026-06-01T00:00:00Z"
    res_ind = await client.get(
        f"/api/v1/instruments/BTCUSDT/indicators?timeframe=1h&from={window_from}&limit=10"
    )
    assert res_ind.status_code == 200
    ind_data = res_ind.json()
    assert ind_data["symbol"] == "BTCUSDT"
    assert len(ind_data["items"]) == 10
    first = ind_data["items"][0]
    assert first["t"] == window_from

    # With 250-bar warmup prior to `from`, RSI14, BB, ATR14, and EMA200 are all ready on bar 0
    assert first["rsi14"] is not None and 0.0 <= first["rsi14"] <= 100.0
    assert first["bb_upper"] is not None and first["bb_lower"] is not None
    assert first["atr14"] is not None and first["atr14"] > 0
    assert first["ema200"] is not None

    # Verify EMA200 is NOT cold-started equal to the bar 0 close price
    res_c = await client.get(
        f"/api/v1/instruments/BTCUSDT/candles?timeframe=1h&from={window_from}&limit=1"
    )
    bar0_close = res_c.json()["items"][0]["c"]
    assert abs(first["ema200"] - bar0_close) > 1.0


@pytest.mark.asyncio
async def test_refresh_rate_limit_and_lock(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    # Resolve BTCUSDT UUID to clean up Redis keys deterministically
    inst = (await client.get("/api/v1/instruments/BTCUSDT")).json()
    cooldown_key = f"ratelimit:refresh:{inst['id']}:1h"
    lock_key = f"lock:refresh:{inst['id']}:1h"
    redis = get_redis()
    await redis.delete(cooldown_key, lock_key)

    mock_ingest = AsyncMock(
        return_value=BackfillSummary(
            symbol="BTCUSDT",
            timeframe="1h",
            source="binance",
            rows_written=5,
            rows_rejected=0,
            unclosed_skipped=1,
            expected_bars=0,
            actual_bars=5,
            completeness_pct=100.0,
            gaps_found=0,
            status="ok",
        )
    )
    monkeypatch.setattr(instruments_ep, "ingest_instrument_candles", mock_ingest)

    try:
        r1 = await client.post("/api/v1/instruments/BTCUSDT/refresh?timeframe=1h")
        assert r1.status_code == 202
        body = r1.json()
        assert body["symbol"] == "BTCUSDT"
        assert body["rows_written"] == 5
        assert mock_ingest.await_count == 1

        # Immediate second request must hit 429 RATE_LIMITED with Retry-After header
        r2 = await client.post("/api/v1/instruments/BTCUSDT/refresh?timeframe=1h")
        assert r2.status_code == 429
        assert int(r2.headers["retry-after"]) > 0
        assert "application/problem+json" in r2.headers["content-type"]
        assert r2.json()["code"] == "RATE_LIMITED"
        # Upstream ingest must NOT have been called a second time
        assert mock_ingest.await_count == 1
    finally:
        await redis.delete(cooldown_key, lock_key)


@pytest.mark.asyncio
async def test_predictions(client: AsyncClient):
    response = await client.get("/api/v1/predictions?timeframe=1h")
    assert response.status_code == 200
    items = response.json()
    assert len(items) >= 1
    assert items[0]["signal"] in ("BUY", "SELL", "HOLD")
