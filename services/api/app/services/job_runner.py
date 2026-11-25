import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.infra.db import AsyncSessionLocal
from app.infra.models import BacktestModel, JobModel
from finsight_core.config import get_settings
from loguru import logger
from sqlalchemy import update

_API_ROOT = Path(__file__).resolve().parents[2]


class JobRunner(Protocol):
    """Abstract background job dispatcher (Subprocess in coursework, Celery in D0)."""

    async def enqueue_backtest(self, backtest_id: uuid.UUID, job_id: uuid.UUID) -> None: ...


class SubprocessJobRunner:
    """
    Runs heavy ML backtest jobs in a detached OS subprocess (`app.worker.run_backtest_job`)
    so the FastAPI event loop stays responsive and `--reload` does not abort running work.
    """

    def __init__(self, logs_dir: Path | None = None) -> None:
        settings = get_settings()
        base = logs_dir or (Path(settings.ARTIFACTS_DIR).parent / "logs" / "jobs")
        self.logs_dir = base

    async def enqueue_backtest(self, backtest_id: uuid.UUID, job_id: uuid.UUID) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{job_id}.log"
        cmd = [
            sys.executable,
            "-m",
            "app.worker.run_backtest_job",
            "--backtest-id",
            str(backtest_id),
            "--job-id",
            str(job_id),
        ]
        with open(log_path, "ab") as log_file:
            subprocess.Popen(
                cmd,
                cwd=str(_API_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        logger.info("Spawned subprocess backtest job_id={} backtest_id={}", job_id, backtest_id)


async def reconcile_stale_jobs() -> int:
    """
    Mark any `queued`/`running` jobs left over from a previous process crash/restart as `failed`.
    Returns the number of reconciled jobs, or 0 if DB is unreachable at startup.
    """
    try:
        now = datetime.now(UTC)
        async with AsyncSessionLocal() as session:
            res_jobs = await session.execute(
                update(JobModel)
                .where(JobModel.status.in_(["queued", "running"]))
                .values(
                    status="failed",
                    error="Interrupted by API restart",
                    finished_at=now,
                )
            )
            await session.execute(
                update(BacktestModel)
                .where(BacktestModel.status.in_(["queued", "running"]))
                .values(
                    status="failed",
                    error="Interrupted by API restart",
                    finished_at=now,
                )
            )
            await session.commit()
            count = int(getattr(res_jobs, "rowcount", 0) or 0)
            if count > 0:
                logger.warning("Reconciled {} stale interrupted jobs on startup.", count)
            return count
    except Exception as exc:
        logger.warning("Skipping stale job reconciliation on startup: {}", exc)
        return 0


_DEFAULT_RUNNER: JobRunner | None = None


def get_job_runner() -> JobRunner:
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = SubprocessJobRunner()
    return _DEFAULT_RUNNER
