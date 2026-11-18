from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import exchange_calendars as xcals
import pandas as pd

_TF_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}

_EQUITY_EXCHANGES = {"nyse", "nasdaq", "xnys", "arca"}


@lru_cache(maxsize=1)
def get_nyse_calendar() -> Any:
    """Cached NYSE (XNYS) trading calendar including holidays and early closes."""
    return xcals.get_calendar("XNYS")


def is_equity_exchange(exchange: str) -> bool:
    return exchange.strip().lower() in _EQUITY_EXCHANGES


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("Naive datetime is not allowed; expected tz-aware UTC datetime.")
    return dt.astimezone(UTC)


def _ceil_utc_grid(dt: datetime, delta: timedelta) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    step_sec = int(delta.total_seconds())
    elapsed = (dt - epoch).total_seconds()
    steps = int(-(-elapsed // step_sec))
    return epoch + timedelta(seconds=steps * step_sec)


def expected_bar_times(
    exchange: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    """
    Generate expected bar open_times (UTC) within [start, end] inclusive.
    - Crypto (24/7): uniform UTC grid.
    - NYSE/US equities: XNYS calendar accounting for weekends, holidays,
      early-close half-days, and Yahoo 1h :30 session-open alignment.
    """
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    if start_utc > end_utc:
        return []

    if timeframe not in _TF_DELTAS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if not is_equity_exchange(exchange):
        delta = _TF_DELTAS[timeframe]
        cur = _ceil_utc_grid(start_utc, delta)
        out: list[datetime] = []
        while cur <= end_utc:
            out.append(cur)
            cur += delta
        return out

    cal = get_nyse_calendar()
    # Clamp range to calendar bounds to avoid OutOfBoundsDatetime
    cal_first = cal.first_session
    cal_last = cal.last_session
    start_day = max(pd.Timestamp(start_utc.date()), cal_first)
    end_day = min(pd.Timestamp(end_utc.date()), cal_last)
    if start_day > end_day:
        return []

    sessions = cal.sessions_in_range(start_day, end_day)
    result: list[datetime] = []

    for session in sessions:
        session_open = cal.session_open(session).to_pydatetime().astimezone(UTC)
        session_close = cal.session_close(session).to_pydatetime().astimezone(UTC)

        if timeframe == "1d":
            if start_utc <= session_open <= end_utc:
                result.append(session_open)
        elif timeframe == "1h":
            # Yahoo 1h bars start at session_open (:30 ET) and step +1h while < session_close.
            # Normal day (9:30-16:00 ET): 7 bars (last is 15:30-16:00 partial 30m bar).
            # Half day (9:30-13:00 ET): 4 bars (last is 12:30-13:00 partial 30m bar).
            cur = session_open
            while cur < session_close:
                if start_utc <= cur <= end_utc:
                    result.append(cur)
                cur += timedelta(hours=1)
        else:
            delta = _TF_DELTAS[timeframe]
            cur = session_open
            while cur < session_close:
                if start_utc <= cur <= end_utc:
                    result.append(cur)
                cur += delta

    return result


def is_nyse_regular_bar_time(open_time: datetime) -> bool:
    """
    Return True if open_time falls strictly inside [session_open, session_close)
    on an XNYS trading day. Filters synthetic post-close tick bars (e.g. 16:00 ET V=0).
    """
    dt_utc = _ensure_utc(open_time)
    cal = get_nyse_calendar()
    day = pd.Timestamp(dt_utc.date())
    if not cal.is_session(day):
        return False
    session_open: datetime = cal.session_open(day).to_pydatetime().astimezone(UTC)
    session_close: datetime = cal.session_close(day).to_pydatetime().astimezone(UTC)
    return bool(session_open <= dt_utc < session_close)


def is_quote_stale(
    exchange: str,
    last_bar_time: datetime | None,
    now: datetime,
    *,
    max_age_sec: int = 7200,
) -> bool:
    """
    Determine whether the latest stored quote/bar is stale.
    - Crypto (24/7): stale if age > max_age_sec.
    - NYSE/US equities: NOT stale on weekends, holidays, or after-hours
      as long as we have a bar from the most recently completed trading session.
    """
    if last_bar_time is None:
        return True

    now_utc = _ensure_utc(now)
    last_utc = _ensure_utc(last_bar_time)

    if not is_equity_exchange(exchange):
        return (now_utc - last_utc).total_seconds() > max_age_sec

    cal = get_nyse_calendar()
    now_ts = pd.Timestamp(now_utc).floor("min")

    # If NYSE is currently in a regular trading session, apply live staleness check
    # (allow a small grace window right after open before first 1h bar closes)
    try:
        in_session = bool(cal.is_open_on_minute(now_ts, ignore_breaks=True))
    except Exception:
        in_session = False

    if in_session:
        day = pd.Timestamp(now_utc.date())
        session_open = cal.session_open(day).to_pydatetime().astimezone(UTC)
        # During the first bar after open, previous session close is still fresh
        if (now_utc - session_open).total_seconds() < max_age_sec:
            prev_session = cal.previous_session(day)
            prev_open: datetime = cal.session_open(prev_session).to_pydatetime().astimezone(UTC)
            return bool(last_utc < prev_open)
        return (now_utc - last_utc).total_seconds() > max_age_sec

    # Market is closed (weekend, holiday, pre-market, or after-hours):
    # Find the most recent session whose open has already passed.
    today = pd.Timestamp(now_utc.date())
    if cal.is_session(today):
        today_open: datetime = cal.session_open(today).to_pydatetime().astimezone(UTC)
        latest_session = today if now_utc >= today_open else cal.previous_session(today)
    else:
        latest_session = cal.date_to_session(today, direction="previous")

    latest_session_open: datetime = cal.session_open(latest_session).to_pydatetime().astimezone(UTC)
    return bool(last_utc < latest_session_open)
