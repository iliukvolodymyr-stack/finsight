from celery import Celery
from finsight_core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "finsight_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "ingest.*": {"queue": "ingest"},
        "features.*": {"queue": "ingest"},
        "inference.*": {"queue": "inference"},
        "nlp.*": {"queue": "nlp"},
        "alerts.*": {"queue": "alerts"},
    },
)


@celery_app.task(name="healthcheck")
def worker_healthcheck() -> str:
    return "ok"
