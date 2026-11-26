#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] || cp .env.example .env

echo "[1/5] Starting PostgreSQL (TimescaleDB) + Redis in Docker..."
docker compose -f infra/compose/docker-compose.dev.yml up -d postgres redis

echo "[2/5] Waiting for Postgres to become ready..."
for _ in $(seq 1 30); do
  if docker compose -f infra/compose/docker-compose.dev.yml exec -T postgres pg_isready -U finsight -d finsight >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "[3/5] Running Alembic migrations and seeding demo instruments/portfolios..."
(cd services/api && uv run alembic upgrade head)
uv run python scripts/seed.py

echo "[4/5] Backfilling 2y OHLCV market data + training initial coursework models (if needed)..."
uv run python scripts/backfill.py
uv run python scripts/train_models.py

echo "[5/5] Starting FastAPI (:8000) and Vite Desktop UI (:5173)..."
(cd services/api && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000) &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

pnpm --filter desktop dev -- --host 0.0.0.0
