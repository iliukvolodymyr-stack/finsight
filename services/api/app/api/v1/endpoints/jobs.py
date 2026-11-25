import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.deps import CurrentUser, get_current_user, get_db_session
from app.core.errors import NotFoundException
from app.infra.models import JobModel
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/jobs", tags=["Jobs"])


class JobStatusResponse(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


def _serialize_job(job: JobModel) -> JobStatusResponse:
    return JobStatusResponse(
        id=str(job.id),
        kind=job.kind,
        status=job.status,
        progress=float(job.progress),
        result=dict(job.result) if job.result is not None else None,
        error=job.error,
        created_at=job.created_at.isoformat().replace("+00:00", "Z"),
        started_at=job.started_at.isoformat().replace("+00:00", "Z") if job.started_at else None,
        finished_at=job.finished_at.isoformat().replace("+00:00", "Z") if job.finished_at else None,
    )


@router.get("/{id}", response_model=JobStatusResponse)
async def get_job_status(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> JobStatusResponse:
    try:
        job_uuid = uuid.UUID(id)
    except ValueError as exc:
        raise NotFoundException(f"Job {id} not found") from exc

    job = await session.get(JobModel, job_uuid)
    if job is None or (job.user_id is not None and str(job.user_id) != current_user.id):
        raise NotFoundException(f"Job {id} not found")
    return _serialize_job(job)


@router.delete("/{id}", response_model=JobStatusResponse)
async def cancel_job(
    id: str,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> JobStatusResponse:
    try:
        job_uuid = uuid.UUID(id)
    except ValueError as exc:
        raise NotFoundException(f"Job {id} not found") from exc

    job = await session.get(JobModel, job_uuid)
    if job is None or (job.user_id is not None and str(job.user_id) != current_user.id):
        raise NotFoundException(f"Job {id} not found")
    if job.status in ("queued", "running"):
        job.status = "cancelled"
        job.finished_at = datetime.now(UTC)
        await session.commit()
    return _serialize_job(job)
