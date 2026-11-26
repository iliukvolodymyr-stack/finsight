#!/usr/bin/env bash
set -euo pipefail

[ -f .env ] || cp .env.example .env

export PATH="$HOME/.local/bin:$PATH"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' is not available (Codespace is still in recovery mode)."
  echo "Run: Cmd/Ctrl + Shift + P -> 'Codespaces: Rebuild Container' (or 'Full Rebuild'), then re-run this script."
  exit 1
fi

if ! dpkg -s libgomp1 >/dev/null 2>&1; then
  echo "[0/5] Installing libgomp1 (for LightGBM)..."
  sudo apt-get update && sudo apt-get install -y --no-install-recommends libgomp1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[0/5] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v pnpm >/dev/null 2>&1; then
  echo "[0/5] Installing pnpm@10..."
  npm install -g pnpm@10 || sudo npm install -g pnpm@10
fi

echo "[1/5] Syncing Python & Node dependencies (if needed)..."
uv python install 3.12
uv sync --all-packages --group dev
pnpm install --frozen-lockfile

echo "[2/5] Starting PostgreSQL (TimescaleDB) + Redis in Docker..."
docker compose -f infra/compose/docker-compose.dev.yml up -d postgres redis

echo "[3/5] Waiting for Postgres to become ready..."
for _ in $(seq 1 30); do
  if docker compose -f infra/compose/docker-compose.dev.yml exec -T postgres pg_isready -U finsight -d finsight >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

echo "[4/5] Running Alembic migrations, seeding, backfill, and initial training..."
(cd services/api && uv run alembic upgrade head)
uv run python scripts/seed.py
uv run python scripts/backfill.py
uv run python scripts/train_models.py

echo "[5/5] Starting FastAPI (:8000) and Vite Desktop UI (:5173)..."
(cd services/api && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000) &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

pnpm --filter desktop dev -- --host 0.0.0.0
