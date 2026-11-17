from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current time in UTC with timezone awareness."""
    return datetime.now(UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime object is timezone-aware and set to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_iso_utc(dt: datetime) -> str:
    """Format datetime as ISO 8601 string with UTC indicator."""
    return ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")
