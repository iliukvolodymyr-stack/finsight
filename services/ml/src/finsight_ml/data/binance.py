import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
from finsight_core.config import get_settings

from finsight_ml.data.types import RawCandle

BINANCE_MAX_LIMIT = 1000


def parse_binance_klines(
    payload: list[list[Any]],
    *,
    now: datetime | None = None,
) -> list[RawCandle]:
    """
    Parse raw Binance /api/v3/klines JSON array into RawCandle objects.
    Binance kline format:
    [0: open_time_ms, 1: open, 2: high, 3: low, 4: close, 5: volume,
     6: close_time_ms, 7: quote_asset_volume, 8: number_of_trades, ...]
    """
    ref_now = (now or datetime.now(UTC)).astimezone(UTC)
    now_ms = int(ref_now.timestamp() * 1000)
    candles: list[RawCandle] = []

    for row in payload:
        if len(row) < 9:
            continue
        open_ms = int(row[0])
        close_ms = int(row[6])
        open_time = datetime.fromtimestamp(open_ms / 1000.0, tz=UTC)
        is_closed = close_ms < now_ms

        candles.append(
            RawCandle(
                open_time=open_time,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                adjusted_close=None,
                quote_volume=float(row[7]),
                trades_count=int(row[8]),
                is_closed=is_closed,
                source="binance",
            )
        )

    return candles


async def fetch_binance_klines(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    *,
    base_url: str | None = None,
    max_retries: int = 3,
    client: httpx.AsyncClient | None = None,
) -> list[RawCandle]:
    """
    Fetch historical klines from Binance REST API with automatic 1000-bar pagination
    and exponential backoff on transient HTTP / 429 errors.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware UTC datetimes")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    api_base = (base_url or get_settings().BINANCE_REST).rstrip("/")
    url = f"{api_base}/api/v3/klines"

    start_ms = int(start.astimezone(UTC).timestamp() * 1000)
    end_ms = int(end.astimezone(UTC).timestamp() * 1000)

    all_candles: list[RawCandle] = []
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=20.0)

    try:
        cursor_ms = start_ms
        while cursor_ms <= end_ms:
            params: dict[str, str | int] = {
                "symbol": symbol.upper(),
                "interval": timeframe,
                "startTime": cursor_ms,
                "endTime": end_ms,
                "limit": BINANCE_MAX_LIMIT,
            }

            payload: list[list[Any]] | None = None
            for attempt in range(max_retries):
                try:
                    resp = await http.get(url, params=params)
                except httpx.TransportError:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(min(1.0 * (2**attempt), 10.0))
                    continue

                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt == max_retries - 1:
                        resp.raise_for_status()
                    retry_after = float(resp.headers.get("Retry-After", 1.0 * (2**attempt)))
                    await asyncio.sleep(min(retry_after, 10.0))
                    continue
                resp.raise_for_status()
                payload = resp.json()
                break

            if not payload:
                break

            batch = parse_binance_klines(payload)
            all_candles.extend(batch)

            last_open_ms = int(payload[-1][0])
            if len(payload) < BINANCE_MAX_LIMIT or last_open_ms <= cursor_ms:
                break

            cursor_ms = last_open_ms + 1
            await asyncio.sleep(0.05)
    finally:
        if owns_client:
            await http.aclose()

    return all_candles
