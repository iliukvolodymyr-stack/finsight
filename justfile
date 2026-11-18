set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set dotenv-load := true

default:
    @just --list

bootstrap:
    pnpm install
    py -3 -m uv sync

up:
    docker compose -f infra/compose/docker-compose.dev.yml up -d

down:
    docker compose -f infra/compose/docker-compose.dev.yml down

migrate:
    Push-Location services/api; py -3 -m uv run alembic upgrade head; Pop-Location

migration msg:
    Push-Location services/api; py -3 -m uv run alembic revision --autogenerate -m "{{msg}}"; Pop-Location

seed:
    py -3 -m uv run python scripts/seed.py

dev-api:
    py -3 -m uv run uvicorn services.api.app.main:app --reload --port 8000

dev-worker:
    py -3 -m uv run celery -A services.api.app.worker.celery_app worker -l INFO -P solo -Q default,ingest,inference,nlp,alerts

dev-beat:
    py -3 -m uv run celery -A services.api.app.worker.celery_app beat -l INFO

dev-streamer:
    py -3 -m uv run python -m finsight_streamer

dev-desktop:
    pnpm --filter desktop dev

dev-mobile:
    pnpm --filter mobile start

gen-openapi:
    py -3 -m uv run python -m services.api.app.scripts.openapi

gen-client:
    npx openapi-typescript openapi.json -o packages/api-client/src/schema.d.ts

test:
    py -3 -m uv run pytest
    pnpm test

test-py:
    py -3 -m uv run pytest

test-js:
    pnpm test

lint:
    py -3 -m uv run ruff check .
    pnpm lint

typecheck:
    py -3 -m uv run mypy services/api/app services/ml/src
    pnpm typecheck

ci:
    py -3 -m uv run ruff check .
    py -3 -m uv run ruff format --check .
    py -3 -m uv run pytest
    pnpm typecheck
    pnpm test
