from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from app.api.v1.endpoints import instruments as instruments_ep
from app.core.deps import CurrentUser, get_current_user
from app.infra.db import AsyncSessionLocal
from app.infra.models import CandleModel, InstrumentModel, new_uuid7
from app.infra.redis import get_redis
from app.main import app
from app.services.ingestion import BackfillSummary
from finsight_ml.features.indicators import calculate_ema
from httpx import AsyncClient
from sqlalchemy import delete


@pytest.fixture
async def synthetic_market_data() -> AsyncIterator[dict[str, str]]:
    """
    Create isolated synthetic test instruments & candles (600 1h crypto bars + 15 1d/1h ETF bars)
    so API tests never depend on a manual live backfill and pass cleanly on an empty CI database.
    """
    crypto_id = new_uuid7()
    etf_id = new_uuid7()
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start_1h = now_hour - timedelta(hours=599)

    async with AsyncSessionLocal() as session:
        session.add_all(
            [
                InstrumentModel(
                    id=crypto_id,
                    symbol="TESTBTC",
                    name="Test Bitcoin",
                    asset_class="crypto",
                    exchange="binance",
                    quote_currency="USDT",
                    is_active=False,
                    metadata_={"synthetic": True},
                ),
                InstrumentModel(
                    id=etf_id,
                    symbol="TESTSPY",
                    name="Test S&P 500 ETF",
                    asset_class="etf",
                    exchange="nyse",
                    quote_currency="USD",
                    is_active=False,
                    metadata_={"synthetic": True},
                ),
            ]
        )
        await session.flush()

        # 600 hourly closed bars trending smoothly from 10_000 to 15_990
        crypto_candles = [
            CandleModel(
                instrument_id=crypto_id,
                timeframe="1h",
                open_time=start_1h + timedelta(hours=i),
                open=10000.0 + i * 10.0,
                high=10050.0 + i * 10.0,
                low=9950.0 + i * 10.0,
                close=10010.0 + i * 10.0,
                adjusted_close=None,
                volume=100.0 + (i % 10),
                is_closed=True,
                source="binance",
            )
            for i in range(600)
        ]

        # 15 daily bars (with adjusted_close) + 15 hourly bars (adjusted_close=None) for TESTSPY
        base_day = datetime(2026, 10, 1, 13, 30, 0, tzinfo=UTC)
        etf_1d = [
            CandleModel(
                instrument_id=etf_id,
                timeframe="1d",
                open_time=base_day + timedelta(days=i),
                open=500.0 + i,
                high=505.0 + i,
                low=498.0 + i,
                close=502.0 + i,
                adjusted_close=(502.0 + i) * 0.98,
                volume=1_000_000.0,
                is_closed=True,
                source="yahoo",
            )
            for i in range(15)
        ]
        etf_1h = [
            CandleModel(
                instrument_id=etf_id,
                timeframe="1h",
                open_time=base_day + timedelta(hours=i),
                open=500.0 + i * 0.5,
                high=502.0 + i * 0.5,
                low=499.0 + i * 0.5,
                close=501.0 + i * 0.5,
                adjusted_close=None,
                volume=150_000.0,
                is_closed=True,
                source="yahoo",
            )
            for i in range(15)
        ]

        session.add_all(crypto_candles + etf_1d + etf_1h)
        await session.commit()

    try:
        yield {
            "crypto_id": str(crypto_id),
            "crypto_symbol": "TESTBTC",
            "etf_id": str(etf_id),
            "etf_symbol": "TESTSPY",
            "bar_550_iso": (start_1h + timedelta(hours=550)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(InstrumentModel).where(InstrumentModel.id.in_([crypto_id, etf_id]))
            )
            await session.commit()


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
async def test_get_quote_from_timescaledb(
    client: AsyncClient,
    synthetic_market_data: dict[str, str],
):
    sym = synthetic_market_data["crypto_symbol"]
    res_btc = await client.get(f"/api/v1/instruments/{sym}/quote")
    assert res_btc.status_code == 200
    assert "max-age=2" in res_btc.headers.get("cache-control", "")
    btc = res_btc.json()
    assert btc["symbol"] == sym
    assert btc["price"] == 10010.0 + 599 * 10.0
    assert btc["volume_24h"] > 0
    assert btc["stale"] is False


@pytest.mark.asyncio
async def test_get_candles_non_overlapping_pagination_and_adjusted_close(
    client: AsyncClient,
    synthetic_market_data: dict[str, str],
):
    sym = synthetic_market_data["crypto_symbol"]
    etf = synthetic_market_data["etf_symbol"]

    # Backward pagination (from=None): page 1 + page 2 via `to=next_cursor` must have ZERO overlap
    p1 = (await client.get(f"/api/v1/instruments/{sym}/candles?timeframe=1h&limit=10")).json()
    assert len(p1["items"]) == 10
    assert p1["has_more"] is True
    assert p1["next_cursor"] is not None

    p2 = (
        await client.get(
            f"/api/v1/instruments/{sym}/candles?timeframe=1h&to={p1['next_cursor']}&limit=10"
        )
    ).json()
    assert len(p2["items"]) == 10
    p1_times = [c["t"] for c in p1["items"]]
    p2_times = [c["t"] for c in p2["items"]]
    assert p1_times == sorted(p1_times)
    assert p2_times == sorted(p2_times)
    assert set(p1_times).isdisjoint(set(p2_times))
    assert p2_times[-1] < p1_times[0]

    # Forward pagination (from set) with limit=1 must advance without infinite loop or duplicates
    f1 = (
        await client.get(
            f"/api/v1/instruments/{sym}/candles?timeframe=1h&from={p2_times[0]}&limit=1"
        )
    ).json()
    assert len(f1["items"]) == 1
    assert f1["has_more"] is True
    f2 = (
        await client.get(
            f"/api/v1/instruments/{sym}/candles?timeframe=1h&from={f1['next_cursor']}&limit=1"
        )
    ).json()
    assert len(f2["items"]) == 1
    assert f2["items"][0]["t"] > f1["items"][0]["t"]

    # ETF 1d populates adjusted_close; ETF 1h leaves adjusted_close as None
    res_etf_1d = await client.get(f"/api/v1/instruments/{etf}/candles?timeframe=1d&limit=10")
    assert res_etf_1d.status_code == 200
    assert all(c["adjusted_close"] is not None for c in res_etf_1d.json()["items"])

    res_etf_1h = await client.get(f"/api/v1/instruments/{etf}/candles?timeframe=1h&limit=10")
    assert res_etf_1h.status_code == 200
    assert all(c["adjusted_close"] is None for c in res_etf_1h.json()["items"])

    # Invalid range (from > to) -> 422 RFC 9457
    res_bad = await client.get(
        f"/api/v1/instruments/{sym}/candles?timeframe=1h"
        "&from=2026-06-10T00:00:00Z&to=2026-06-01T00:00:00Z"
    )
    assert res_bad.status_code == 422
    assert res_bad.json()["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_indicators_500_bar_ema200_warmup_accuracy(
    client: AsyncClient,
    synthetic_market_data: dict[str, str],
):
    sym = synthetic_market_data["crypto_symbol"]
    window_from = synthetic_market_data["bar_550_iso"]

    res_ind = await client.get(
        f"/api/v1/instruments/{sym}/indicators?timeframe=1h&from={window_from}&limit=10"
    )
    assert res_ind.status_code == 200
    items = res_ind.json()["items"]
    assert len(items) == 10
    first = items[0]
    assert first["t"] == window_from

    # Compare bar 550 EMA200 from API against exact full 600-bar series EMA200
    full_closes = pd.Series([10010.0 + i * 10.0 for i in range(600)], dtype="float64")
    expected_ema200_at_550 = float(calculate_ema(full_closes, 200).iloc[550])
    assert first["ema200"] is not None
    # With 500-bar warmup, residual error vs full-history EMA200 is tiny (< 5.0 out of ~14,500)
    assert abs(first["ema200"] - expected_ema200_at_550) < 5.0
    assert first["rsi14"] is not None and 0.0 <= first["rsi14"] <= 100.0
    assert first["bb_upper"] is not None and first["bb_lower"] is not None
    assert first["atr14"] is not None and first["atr14"] > 0


@pytest.mark.asyncio
async def test_refresh_rbac_lock_and_rate_limit(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
):
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
        # 1. Non-admin user must receive 403 FORBIDDEN
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(
            id="u1", email="user@finsight.local", role="user"
        )
        r_forbidden = await client.post("/api/v1/instruments/BTCUSDT/refresh?timeframe=1h")
        assert r_forbidden.status_code == 403
        assert r_forbidden.json()["code"] == "FORBIDDEN"
        app.dependency_overrides.clear()

        # 2. Existing lock held by another worker -> 409 CONFLICT and lock NOT deleted
        await redis.set(lock_key, "other-worker-token", ex=60)
        r_conflict = await client.post("/api/v1/instruments/BTCUSDT/refresh?timeframe=1h")
        assert r_conflict.status_code == 409
        assert r_conflict.json()["code"] == "CONFLICT"
        assert await redis.get(lock_key) == "other-worker-token"
        await redis.delete(lock_key)

        # 3. Normal admin refresh -> 202 Accepted
        r1 = await client.post("/api/v1/instruments/BTCUSDT/refresh?timeframe=1h")
        assert r1.status_code == 202
        assert r1.json()["rows_written"] == 5
        assert mock_ingest.await_count == 1
        assert await redis.get(lock_key) is None  # lock released cleanly

        # 4. Immediate repeat -> 429 RATE_LIMITED with Retry-After
        r2 = await client.post("/api/v1/instruments/BTCUSDT/refresh?timeframe=1h")
        assert r2.status_code == 429
        assert int(r2.headers["retry-after"]) > 0
        assert "application/problem+json" in r2.headers["content-type"]
        assert r2.json()["code"] == "RATE_LIMITED"
        assert mock_ingest.await_count == 1
    finally:
        app.dependency_overrides.clear()
        await redis.delete(cooldown_key, lock_key)


@pytest.mark.asyncio
async def test_predictions(client: AsyncClient):
    response = await client.get("/api/v1/predictions?timeframe=1h")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    for item in items:
        assert item["signal"] in ("BUY", "SELL", "HOLD")
