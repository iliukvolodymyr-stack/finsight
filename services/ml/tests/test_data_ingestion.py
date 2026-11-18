from datetime import UTC, datetime

import httpx
import pytest
from finsight_ml.data import (
    RawCandle,
    detect_gaps,
    expected_bar_times,
    fetch_binance_klines,
    fetch_yahoo_candles,
    is_quote_stale,
    parse_binance_klines,
    parse_yahoo_chart,
    validate_candles,
)


def test_parse_binance_klines_fixture():
    # Recorded Binance /api/v3/klines format:
    # [open_ms, open, high, low, close, volume, close_ms, quote_vol, trades, ...]
    ref_now = datetime(2026, 11, 17, 14, 30, 0, tzinfo=UTC)
    t1 = int(datetime(2026, 11, 17, 13, 0, 0, tzinfo=UTC).timestamp() * 1000)
    t1_close = int(datetime(2026, 11, 17, 13, 59, 59, 999000, tzinfo=UTC).timestamp() * 1000)
    t2 = int(datetime(2026, 11, 17, 14, 0, 0, tzinfo=UTC).timestamp() * 1000)
    t2_close = int(datetime(2026, 11, 17, 14, 59, 59, 999000, tzinfo=UTC).timestamp() * 1000)

    fixture = [
        [t1, "91000.0", "91500.0", "90800.0", "91250.5", "412.5", t1_close, "37600000.0", 12450],
        [t2, "91250.5", "91400.0", "91100.0", "91372.0", "210.1", t2_close, "19180000.0", 6200],
    ]

    candles = parse_binance_klines(fixture, now=ref_now)
    assert len(candles) == 2
    assert candles[0].open_time == datetime(2026, 11, 17, 13, 0, 0, tzinfo=UTC)
    assert candles[0].is_closed is True
    assert candles[0].adjusted_close is None
    assert candles[0].trades_count == 12450
    assert candles[1].is_closed is False

    # Data gates drop unclosed bar without counting it as corrupt
    val = validate_candles(candles, allow_unclosed=False)
    assert len(val.valid) == 1
    assert val.rejected_count == 0
    assert val.unclosed_skipped == 1


def test_parse_yahoo_1h_filters_synthetic_post_close_tick():
    # Nov 17, 2026 (EST): regular session 14:30 - 21:00 UTC
    # 20:30 UTC is the valid last partial 30m bar; 21:00 UTC (V=0) is Yahoo's synthetic close tick
    ts_valid = int(datetime(2026, 11, 17, 20, 30, 0, tzinfo=UTC).timestamp())
    ts_synth = int(datetime(2026, 11, 17, 21, 0, 0, tzinfo=UTC).timestamp())
    fixture = {
        "chart": {
            "result": [
                {
                    "timestamp": [ts_valid, ts_synth],
                    "indicators": {
                        "quote": [
                            {
                                "open": [730.1, 732.44],
                                "high": [732.5, 732.44],
                                "low": [729.4, 732.44],
                                "close": [732.4, 732.44],
                                "volume": [11174915, 0],
                            }
                        ]
                    },
                }
            ]
        }
    }

    ref_now = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
    candles = parse_yahoo_chart(fixture, "1h", now=ref_now)
    assert len(candles) == 1
    assert candles[0].open_time == datetime(2026, 11, 17, 20, 30, 0, tzinfo=UTC)
    assert candles[0].adjusted_close is None
    assert candles[0].is_closed is True


def test_parse_yahoo_1d_separates_raw_ohlc_and_adjusted_close_and_counts_nulls():
    ts1 = int(datetime(2025, 11, 17, 14, 30, 0, tzinfo=UTC).timestamp())
    ts2 = int(datetime(2025, 11, 18, 14, 30, 0, tzinfo=UTC).timestamp())
    fixture = {
        "chart": {
            "result": [
                {
                    "timestamp": [ts1, ts2],
                    "indicators": {
                        "quote": [
                            {
                                "open": [588.0, None],
                                "high": [592.0, None],
                                "low": [586.5, None],
                                "close": [590.5, None],
                                "volume": [45000000, None],
                            }
                        ],
                        "adjclose": [{"adjclose": [582.15, None]}],
                    },
                }
            ]
        }
    }

    ref_now = datetime(2025, 11, 20, 12, 0, 0, tzinfo=UTC)
    candles = parse_yahoo_chart(fixture, "1d", now=ref_now)
    val = validate_candles(candles, allow_unclosed=False)
    assert len(val.valid) == 1
    assert val.rejected_count == 1
    # Raw OHLC preserved untouched, adjusted_close stored in dedicated field
    assert val.valid[0].close == 590.5
    assert val.valid[0].adjusted_close == 582.15


def test_data_gates_rejects_bad_bars_and_deduplicates():
    t1 = datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 11, 17, 13, 0, 0, tzinfo=UTC)
    naive = datetime(2026, 11, 17, 14, 0, 0)  # naive -> must be rejected

    candles = [
        RawCandle(t2, 100.0, 105.0, 99.0, 102.0, 50.0),  # out-of-order valid
        RawCandle(t1, 100.0, 102.0, 98.0, 101.0, 10.0),  # valid
        RawCandle(t1, 100.0, 103.0, 98.0, 102.0, 15.0),  # duplicate t1 (latest wins)
        RawCandle(naive, 100.0, 102.0, 98.0, 101.0, 10.0),  # rejected: naive dt
        RawCandle(t2, 100.0, 95.0, 98.0, 99.0, 10.0),  # rejected: high < low
        RawCandle(t2, 100.0, 105.0, 99.0, 101.0, -5.0),  # rejected: negative volume
        RawCandle(t2, 100.0, 105.0, 101.0, 102.0, 10.0),  # rejected: low > open
    ]

    res = validate_candles(candles)
    assert res.rejected_count == 4
    assert len(res.rejection_reasons) == 4
    assert len(res.valid) == 2
    assert res.valid[0].open_time == t1
    assert res.valid[0].close == 102.0  # latest duplicate won
    assert res.valid[1].open_time == t2


def test_gap_detector_spy_thanksgiving_and_black_friday_half_day():
    # Nov 26 (Wed): 7 bars; Nov 27 (Thu Thanksgiving): 0 bars;
    # Nov 28 (Fri Black Friday early close 13:00 ET): 4 bars;
    # Nov 29-30 (Weekend): 0 bars; Dec 1 (Mon): 7 bars => Total expected = 18 bars
    start = datetime(2025, 11, 26, 14, 30, 0, tzinfo=UTC)
    end = datetime(2025, 12, 1, 20, 30, 0, tzinfo=UTC)

    expected = expected_bar_times("nyse", "1h", start, end)
    assert len(expected) == 18

    # Full calendar coverage -> 100%, 0 gaps despite holiday, half-day, and weekend
    full_report = detect_gaps(expected, exchange="nyse", timeframe="1h", start=start, end=end)
    assert full_report.expected_bars == 18
    assert full_report.actual_bars == 18
    assert full_report.completeness_pct == 100.0
    assert full_report.gaps == []

    # Drop 1 bar on Black Friday (15:30 UTC / 10:30 ET)
    missing_bar = datetime(2025, 11, 28, 15, 30, 0, tzinfo=UTC)
    partial = [t for t in expected if t != missing_bar]
    gap_report = detect_gaps(partial, exchange="nyse", timeframe="1h", start=start, end=end)
    assert gap_report.expected_bars == 18
    assert gap_report.actual_bars == 17
    assert len(gap_report.gaps) == 1
    assert gap_report.gaps[0].gap_start == missing_bar
    assert gap_report.gaps[0].bars_missing == 1


def test_gap_detector_crypto_24_7():
    start = datetime(2026, 11, 17, 0, 0, 0, tzinfo=UTC)
    end = datetime(2026, 11, 17, 9, 0, 0, tzinfo=UTC)
    expected = expected_bar_times("binance", "1h", start, end)
    assert len(expected) == 10

    # Remove hours 3, 4, 5 (contiguous 3-bar gap)
    observed = [t for t in expected if t.hour not in (3, 4, 5)]
    report = detect_gaps(observed, exchange="binance", timeframe="1h", start=start, end=end)
    assert report.expected_bars == 10
    assert report.actual_bars == 7
    assert report.completeness_pct == 70.0
    assert len(report.gaps) == 1
    assert report.gaps[0].bars_missing == 3


def test_is_quote_stale_respects_spy_weekends_and_holidays():
    # Friday Nov 21, 2025 last 1h bar at 20:30 UTC (15:30 ET)
    friday_open = datetime(2025, 11, 21, 14, 30, 0, tzinfo=UTC)
    friday_last = datetime(2025, 11, 21, 20, 30, 0, tzinfo=UTC)
    sunday_noon = datetime(2025, 11, 23, 12, 0, 0, tzinfo=UTC)

    # SPY on Sunday with Friday's closing 1h bar -> NOT stale
    assert is_quote_stale("nyse", friday_last, sunday_noon, timeframe="1h") is False
    # SPY on Sunday with only Friday's 14:30 morning bar (missed 6 hours!) -> STALE for 1h
    assert is_quote_stale("nyse", friday_open, sunday_noon, timeframe="1h") is True
    # But for 1d timeframe, Friday 14:30 UTC IS the valid daily bar -> NOT stale
    assert is_quote_stale("nyse", friday_open, sunday_noon, timeframe="1d") is False
    # Crypto on Sunday with Friday's bar -> STALE
    assert is_quote_stale("binance", friday_last, sunday_noon) is True

    # Black Friday half-day (Nov 28, 2025: 14:30 - 18:00 UTC, last 1h bar = 17:30 UTC)
    bf_last = datetime(2025, 11, 28, 17, 30, 0, tzinfo=UTC)
    bf_early = datetime(2025, 11, 28, 15, 30, 0, tzinfo=UTC)
    sat_noon = datetime(2025, 11, 29, 12, 0, 0, tzinfo=UTC)
    assert is_quote_stale("nyse", bf_last, sat_noon, timeframe="1h") is False
    assert is_quote_stale("nyse", bf_early, sat_noon, timeframe="1h") is True

    # Post-close grace window (21:05 UTC = 5 min after close vs 21:20 UTC = 20 min after)
    fri_1930 = datetime(2025, 11, 21, 19, 30, 0, tzinfo=UTC)
    assert is_quote_stale("nyse", fri_1930, datetime(2025, 11, 21, 21, 5, 0, tzinfo=UTC)) is False
    assert is_quote_stale("nyse", fri_1930, datetime(2025, 11, 21, 21, 20, 0, tzinfo=UTC)) is True

    # Thanksgiving Thursday Nov 27, 2025 with Wednesday's bar -> NOT stale
    wed_last = datetime(2025, 11, 26, 20, 30, 0, tzinfo=UTC)
    thanksgiving = datetime(2025, 11, 27, 17, 0, 0, tzinfo=UTC)
    assert is_quote_stale("nyse", wed_last, thanksgiving) is False

    # Mid-session & post-close grace for 1d: Tuesday's daily bar stays fresh on Wednesday
    tue_daily = datetime(2025, 11, 18, 14, 30, 0, tzinfo=UTC)
    wed_mid_session = datetime(2025, 11, 19, 18, 0, 0, tzinfo=UTC)
    wed_post_close_grace = datetime(2025, 11, 19, 21, 5, 0, tzinfo=UTC)
    wed_after_grace = datetime(2025, 11, 19, 21, 20, 0, tzinfo=UTC)
    assert is_quote_stale("nyse", tue_daily, wed_mid_session, timeframe="1d") is False
    assert is_quote_stale("nyse", tue_daily, wed_post_close_grace, timeframe="1d") is False
    assert is_quote_stale("nyse", tue_daily, wed_after_grace, timeframe="1d") is True

    # Missing Friday's session completely when checked on Sunday -> STALE
    thursday_last = datetime(2025, 11, 20, 20, 30, 0, tzinfo=UTC)
    assert is_quote_stale("nyse", thursday_last, sunday_noon) is True


@pytest.mark.asyncio
async def test_adapters_raise_on_exhausted_retries_and_retry_transport_errors():
    start = datetime(2026, 11, 17, 0, 0, 0, tzinfo=UTC)
    end = datetime(2026, 11, 17, 2, 0, 0, tzinfo=UTC)

    # 1. Persistent 503 must raise HTTPStatusError instead of returning empty []
    always_503 = httpx.MockTransport(lambda req: httpx.Response(503, headers={"Retry-After": "0"}))
    async with httpx.AsyncClient(transport=always_503) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await fetch_binance_klines("BTCUSDT", "1h", start, end, max_retries=2, client=client)
        with pytest.raises(httpx.HTTPStatusError):
            await fetch_yahoo_candles("SPY", "1h", start, end, max_retries=2, client=client)

    # 2. Transient TransportError on attempt 1 succeeds on attempt 2
    calls = {"n": 0}

    def flaky_handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("transient network error", request=req)
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(flaky_handler)) as client:
        res = await fetch_binance_klines("BTCUSDT", "1h", start, end, max_retries=2, client=client)
        assert res == []
        assert calls["n"] == 2
