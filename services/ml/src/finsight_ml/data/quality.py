import math
from datetime import UTC, datetime

from finsight_ml.data.calendar import expected_bar_times
from finsight_ml.data.types import GapRecord, QualityReport, RawCandle, ValidationResult


def _is_finite_pos(val: float) -> bool:
    return math.isfinite(val) and val > 0.0


def validate_candles(
    candles: list[RawCandle],
    *,
    allow_unclosed: bool = False,
) -> ValidationResult:
    """
    Apply strict data gates to raw candles:
    - Timestamps must be tz-aware (converted to UTC)
    - OHLC must be finite and > 0; volume finite and >= 0
    - OHLC invariants: low <= open, close <= high and low <= high
    - Optional adjusted_close must be finite and > 0
    - Bad bars are counted in rejected_count (never silently dropped)
    - Deduplicates by open_time (latest wins) and sorts strictly monotonically ascending
    """
    result = ValidationResult()
    by_time: dict[datetime, RawCandle] = {}

    for idx, c in enumerate(candles):
        if not allow_unclosed and not c.is_closed:
            result.unclosed_skipped += 1
            continue

        reason: str | None = None
        if c.open_time.tzinfo is None:
            reason = f"row[{idx}]: naive open_time {c.open_time!r}"
        elif not (
            _is_finite_pos(c.open)
            and _is_finite_pos(c.high)
            and _is_finite_pos(c.low)
            and _is_finite_pos(c.close)
        ):
            reason = f"row[{idx}] ({c.open_time.isoformat()}): non-positive or non-finite OHLC"
        elif not (math.isfinite(c.volume) and c.volume >= 0.0):
            reason = f"row[{idx}] ({c.open_time.isoformat()}): invalid volume={c.volume}"
        elif c.high < c.low or c.high < c.open or c.high < c.close:
            reason = f"row[{idx}] ({c.open_time.isoformat()}): high invariant violated"
        elif c.low > c.open or c.low > c.close:
            reason = f"row[{idx}] ({c.open_time.isoformat()}): low invariant violated"
        elif c.adjusted_close is not None and not _is_finite_pos(c.adjusted_close):
            reason = f"row[{idx}] ({c.open_time.isoformat()}): invalid adjusted_close"

        if reason is not None:
            result.rejected_count += 1
            if len(result.rejection_reasons) < 20:
                result.rejection_reasons.append(reason)
            continue

        open_utc = c.open_time.astimezone(UTC)
        normalized = RawCandle(
            open_time=open_utc,
            open=float(c.open),
            high=float(c.high),
            low=float(c.low),
            close=float(c.close),
            volume=float(c.volume),
            adjusted_close=float(c.adjusted_close) if c.adjusted_close is not None else None,
            quote_volume=float(c.quote_volume) if c.quote_volume is not None else None,
            trades_count=int(c.trades_count) if c.trades_count is not None else None,
            is_closed=c.is_closed,
            source=c.source,
        )
        by_time[open_utc] = normalized

    result.valid = [by_time[t] for t in sorted(by_time.keys())]
    return result


def detect_gaps(
    observed_times: list[datetime],
    exchange: str,
    timeframe: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> QualityReport:
    """
    Detect missing bars relative to the exchange calendar (24/7 for crypto,
    XNYS with holidays & half-days for US equities/SPY).
    """
    utc_observed = sorted(dt.astimezone(UTC) for dt in observed_times if dt.tzinfo is not None)

    if not utc_observed:
        if start is None or end is None:
            return QualityReport(expected_bars=0, actual_bars=0, completeness_pct=100.0, gaps=[])
        expected = expected_bar_times(exchange, timeframe, start, end)
        empty_gaps = (
            [GapRecord(gap_start=expected[0], gap_end=expected[-1], bars_missing=len(expected))]
            if expected
            else []
        )
        pct = 0.0 if expected else 100.0
        return QualityReport(
            expected_bars=len(expected), actual_bars=0, completeness_pct=pct, gaps=empty_gaps
        )

    window_start = start.astimezone(UTC) if start is not None else utc_observed[0]
    window_end = end.astimezone(UTC) if end is not None else utc_observed[-1]
    expected = expected_bar_times(exchange, timeframe, window_start, window_end)

    if not expected:
        return QualityReport(
            expected_bars=0, actual_bars=len(utc_observed), completeness_pct=100.0, gaps=[]
        )

    if timeframe == "1d":
        observed_keys = {dt.date() for dt in utc_observed}
        missing = [exp for exp in expected if exp.date() not in observed_keys]
    else:
        observed_min_keys = {dt.replace(second=0, microsecond=0) for dt in utc_observed}
        missing = [
            exp for exp in expected if exp.replace(second=0, microsecond=0) not in observed_min_keys
        ]

    # Group consecutive missing expected slots into contiguous GapRecord intervals
    gaps: list[GapRecord] = []
    if missing:
        missing_set = set(missing)
        gap_start: datetime | None = None
        gap_end: datetime | None = None
        count = 0

        for exp in expected:
            if exp in missing_set:
                if gap_start is None:
                    gap_start = exp
                gap_end = exp
                count += 1
            elif gap_start is not None and gap_end is not None:
                gaps.append(GapRecord(gap_start=gap_start, gap_end=gap_end, bars_missing=count))
                gap_start = None
                gap_end = None
                count = 0

        if gap_start is not None and gap_end is not None:
            gaps.append(GapRecord(gap_start=gap_start, gap_end=gap_end, bars_missing=count))

    actual_matched = len(expected) - len(missing)
    completeness = (actual_matched / len(expected)) * 100.0
    return QualityReport(
        expected_bars=len(expected),
        actual_bars=actual_matched,
        completeness_pct=round(completeness, 4),
        gaps=gaps,
    )
