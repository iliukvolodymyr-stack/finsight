import uuid
from datetime import UTC, datetime

from app.core.errors import AppException, NotFoundException
from app.infra.models import InstrumentModel
from finsight_core.schemas import ErrorDetail
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


async def resolve_instrument(session: AsyncSession, id_or_symbol: str) -> InstrumentModel:
    """Resolve an instrument by UUID or case-insensitive symbol."""
    parsed_uuid: uuid.UUID | None = None
    try:
        parsed_uuid = uuid.UUID(id_or_symbol)
    except ValueError:
        parsed_uuid = None

    if parsed_uuid is not None:
        stmt = select(InstrumentModel).where(InstrumentModel.id == parsed_uuid)
    else:
        stmt = select(InstrumentModel).where(
            func.upper(InstrumentModel.symbol) == id_or_symbol.upper()
        )

    res = await session.execute(stmt)
    inst = res.scalar_one_or_none()
    if inst is None:
        raise NotFoundException(f"Instrument {id_or_symbol} not found")
    return inst


def parse_iso_utc(value: str, field: str) -> datetime:
    """Parse an ISO-8601 timestamp string and enforce timezone-aware UTC."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AppException(
            status_code=422,
            code="VALIDATION_ERROR",
            title="Request validation failed",
            detail=f"Invalid ISO-8601 timestamp for '{field}': {value}",
            errors=[ErrorDetail(field=field, message="must be ISO-8601 UTC", code="datetime")],
        ) from exc

    if dt.tzinfo is None:
        raise AppException(
            status_code=422,
            code="VALIDATION_ERROR",
            title="Request validation failed",
            detail=f"Timestamp '{field}' must include timezone (e.g. Z or +00:00)",
            errors=[ErrorDetail(field=field, message="timezone required", code="datetime_tz")],
        )

    return dt.astimezone(UTC)
