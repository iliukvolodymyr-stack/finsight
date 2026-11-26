from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RawCandle:
    """Normalized market bar before DB persistence (all timestamps tz-aware UTC)."""

    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float | None = None
    quote_volume: float | None = None
    trades_count: int | None = None
    is_closed: bool = True
    source: str = "binance"


@dataclass(frozen=True)
class FundingRateRecord:
    """Normalized Binance Futures perpetual funding settlement record (UTC)."""

    funding_time: datetime
    funding_rate: float
    mark_price: float | None = None


@dataclass
class ValidationResult:
    """Outcome of data-gate checks; bad bars are counted and never dropped silently."""

    valid: list[RawCandle] = field(default_factory=list)
    rejected_count: int = 0
    unclosed_skipped: int = 0
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GapRecord:
    """Contiguous missing interval relative to the expected exchange calendar."""

    gap_start: datetime
    gap_end: datetime
    bars_missing: int


@dataclass(frozen=True)
class QualityReport:
    """Completeness report computed strictly against expected calendar bars."""

    expected_bars: int
    actual_bars: int
    completeness_pct: float
    gaps: list[GapRecord]
