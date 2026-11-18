from celery import Celery
from finsight_core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "finsight_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.worker.tasks"],
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
    beat_schedule={
        "poll-closed-bars-1m": {
            "task": "ingest.poll_closed_bars",
            "schedule": 60.0,
            "options": {"queue": "ingest"},
        },
        "fetch-rss-news-5m": {
            "task": "ingest.fetch_rss_news",
            "schedule": 300.0,
            "options": {"queue": "ingest"},
        },
        "check-data-gaps-1h": {
            "task": "ingest.check_data_gaps",
            "schedule": 3600.0,
            "options": {"queue": "ingest"},
        },
    },
)


@celery_app.task(name="healthcheck")
def worker_healthcheck() -> str:
    return "ok"
