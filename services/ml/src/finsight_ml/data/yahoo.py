import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from finsight_ml.data.calendar import get_nyse_calendar, is_nyse_regular_bar_time
from finsight_ml.data.types import RawCandle

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FinSightAI/1.1"}
YAHOO_1H_MAX_DAYS = 720


def _retry_delay(resp: httpx.Response, default_sec: float) -> float:
    raw = resp.headers.get("Retry-After")
    if raw is not None:
        try:
            return min(max(float(raw), 0.0), 10.0)
        except ValueError:
            pass
    return min(default_sec, 10.0)


def _to_float_or_nan(val: Any) -> float:
    if val is None:
        return float("nan")
    return float(val)


def _bar_is_closed(open_time: datetime, timeframe: str, ref_now: datetime) -> bool:
    if timeframe == "1d":
        cal = get_nyse_calendar()
        import pandas as pd

        day = pd.Timestamp(open_time.date())
        if cal.is_session(day):
            session_close: datetime = cal.session_close(day).to_pydatetime().astimezone(UTC)
            return bool(ref_now >= session_close)
        return ref_now >= open_time + timedelta(days=1)

    if timeframe == "1h":
        cal = get_nyse_calendar()
        import pandas as pd

        day = pd.Timestamp(open_time.date())
        bar_end = open_time + timedelta(hours=1)
        if cal.is_session(day):
            session_close = cal.session_close(day).to_pydatetime().astimezone(UTC)
            bar_end = min(bar_end, session_close)
        return ref_now >= bar_end

    return ref_now >= open_time + timedelta(hours=1)


def parse_yahoo_chart(
    payload: dict[str, Any],
    timeframe: str,
    *,
    now: datetime | None = None,
) -> list[RawCandle]:
    """
    Parse Yahoo Finance v8 /chart JSON payload into RawCandle objects.
    - Preserves raw OHLC separately from adjusted_close (populated on 1d only).
    - Filters synthetic post-close snapshot ticks on 1h (e.g. 16:00 ET V=0)
      using the XNYS exchange calendar.
    - Null OHLC values are mapped to NaN so data gates count them in rows_rejected.
    """
    ref_now = (now or datetime.now(UTC)).astimezone(UTC)
    chart = payload.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        return []

    res = results[0]
    timestamps = res.get("timestamp") or []
    indicators = res.get("indicators") or {}
    quotes = (indicators.get("quote") or [{}])[0]
    adj_list = (indicators.get("adjclose") or [{}])[0].get("adjclose")

    opens = quotes.get("open") or []
    highs = quotes.get("high") or []
    lows = quotes.get("low") or []
    closes = quotes.get("close") or []
    volumes = quotes.get("volume") or []

    candles: list[RawCandle] = []
    for i, ts in enumerate(timestamps):
        if ts is None:
            continue
        open_time = datetime.fromtimestamp(int(ts), tz=UTC)

        # Yahoo appends a synthetic 16:00 ET (volume=0) tick at session_close on 1h charts
        if timeframe == "1h" and not is_nyse_regular_bar_time(open_time):
            continue

        o = _to_float_or_nan(opens[i] if i < len(opens) else None)
        h = _to_float_or_nan(highs[i] if i < len(highs) else None)
        l = _to_float_or_nan(lows[i] if i < len(lows) else None)
        c = _to_float_or_nan(closes[i] if i < len(closes) else None)
        v = _to_float_or_nan(volumes[i] if i < len(volumes) else None)

        adj_close: float | None = None
        if (
            timeframe == "1d"
            and adj_list is not None
            and i < len(adj_list)
            and adj_list[i] is not None
        ):
            adj_close = float(adj_list[i])

        candles.append(
            RawCandle(
                open_time=open_time,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
                adjusted_close=adj_close,
                quote_volume=None,
                trades_count=None,
                is_closed=_bar_is_closed(open_time, timeframe, ref_now),
                source="yahoo",
            )
        )

    return candles


async def fetch_yahoo_candles(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    *,
    max_retries: int = 3,
    client: httpx.AsyncClient | None = None,
) -> list[RawCandle]:
    """
    Fetch OHLCV candles from Yahoo Finance chart API.
    Note: Yahoo enforces a ~730-day rolling window for 1h bars; start is clamped to 720d.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware UTC datetimes")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    now_utc = datetime.now(UTC)
    start_utc = start.astimezone(UTC)
    end_utc = min(end.astimezone(UTC), now_utc)

    if timeframe == "1h":
        min_start = now_utc - timedelta(days=YAHOO_1H_MAX_DAYS)
        if start_utc < min_start:
            start_utc = min_start

    url = YAHOO_CHART_URL.format(symbol=symbol.upper())
    params: dict[str, str | int] = {
        "interval": timeframe,
        "period1": int(start_utc.timestamp()),
        "period2": int(end_utc.timestamp()),
    }

    owns_client = client is None
    http = client or httpx.AsyncClient(headers=YAHOO_HEADERS, timeout=20.0)
    try:
        payload: dict[str, Any] = {}
        for attempt in range(max_retries):
            try:
                resp = await http.get(url, params=params)
            except httpx.TransportError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(min(1.5 * (2**attempt), 10.0))
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == max_retries - 1:
                    resp.raise_for_status()
                await asyncio.sleep(_retry_delay(resp, 1.5 * (2**attempt)))
                continue
            resp.raise_for_status()
            payload = resp.json()
            break
    finally:
        if owns_client:
            await http.aclose()

    return parse_yahoo_chart(payload, timeframe, now=now_utc)
