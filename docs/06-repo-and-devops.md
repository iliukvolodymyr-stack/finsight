# FinSight AI — Структура репозиторію та DevOps

Версія: 1.1 · Інструменти: pnpm + Turborepo, uv, Docker, GitHub Actions

---

## 1. Структура monorepo

```
finsight-ai/
├─ apps/
│  ├─ desktop/                     # Tauri v2 + React + TS + Vite
│  │  ├─ src/
│  │  │  ├─ app/                   # router, providers, layout
│  │  │  ├─ features/              # market, forecast, portfolio, news, risk, alerts, settings
│  │  │  ├─ components/            # спільні UI-примітиви
│  │  │  ├─ lib/                   # api-клієнт, ws, formatters, hooks
│  │  │  └─ styles/
│  │  ├─ src-tauri/                # Rust-обгортка, конфіг вікна, CSP, updater
│  │  ├─ index.html
│  │  ├─ vite.config.ts
│  │  └─ package.json
│  └─ mobile/                      # Expo + React Native + expo-router
│     ├─ app/                      # (auth), (tabs): signals, chart, portfolio, digest
│     ├─ src/
│     │  ├─ features/
│     │  ├─ components/
│     │  └─ lib/                   # secure-store, push, api
│     ├─ app.json
│     └─ package.json
│
├─ packages/
│  ├─ api-client/                  # ЗГЕНЕРОВАНО з OpenAPI + тонка обгортка
│  │  ├─ src/schema.d.ts           # не редагувати вручну
│  │  ├─ src/client.ts             # baseURL, auth, refresh, ApiError
│  │  └─ package.json
│  ├─ shared/                      # Zod-схеми форм, форматери, константи, типи UI
│  ├─ ui/                          # (опційно) спільні React-примітиви для web
│  └─ config/                      # eslint, tsconfig, tailwind preset
│
├─ services/
│  ├─ api/                         # FastAPI
│  │  ├─ app/
│  │  │  ├─ main.py
│  │  │  ├─ core/                  # config, security, logging, errors, deps
│  │  │  ├─ api/v1/                # routers + request/response schemas
│  │  │  ├─ domain/                # entities, ports
│  │  │  ├─ services/              # use cases
│  │  │  ├─ infra/                 # SQLAlchemy, repos, adapters
│  │  │  └─ worker/                # Celery app + tasks
│  │  ├─ migrations/               # Alembic
│  │  ├─ tests/
│  │  └─ pyproject.toml
│  ├─ ml/                          # finsight_ml (бібліотека + CLI)
│  │  ├─ src/finsight_ml/
│  │  │  ├─ features/              # registry, pipeline, indicators
│  │  │  ├─ labels/                # triple barrier
│  │  │  ├─ models/                # lgbm, deep, meta, calibration
│  │  │  ├─ backtest/              # engine, metrics, report
│  │  │  ├─ registry/              # artifacts, promotion
│  │  │  └─ cli.py
│  │  ├─ tests/
│  │  └─ pyproject.toml
│  ├─ streamer/                    # довгоживучий сервіс біржових WS
│  │  ├─ src/finsight_streamer/
│  │  └─ pyproject.toml
│  └─ py-core/                     # finsight_core: config, logging, time, schemas
│     ├─ src/finsight_core/
│     └─ pyproject.toml
│
├─ infra/
│  ├─ docker/                      # Dockerfile.api, Dockerfile.worker, Dockerfile.streamer
│  ├─ compose/
│  │  ├─ docker-compose.dev.yml    # postgres, redis (+ minio з D0)
│  │  └─ docker-compose.prod.yml   # усі сервіси
│  └─ caddy/Caddyfile
│
├─ configs/
│  ├─ instruments.yaml             # юніверс
│  ├─ models/                      # lgbm_btc_1h.yaml, deep_core.yaml, meta.yaml
│  └─ experiments/                 # E01_baseline.yaml ... E12_load.yaml
│
├─ notebooks/                      # дослідження, не production-код
├─ scripts/                        # seed, demo_snapshot, backup, restore
├─ docs/                           # ця документація + ADR
├─ .github/workflows/
├─ turbo.json
├─ pnpm-workspace.yaml
├─ package.json
├─ pyproject.toml                  # uv workspace
├─ uv.lock
├─ justfile
├─ .npmrc                          # node-linker=hoisted (Expo + pnpm workspaces)
├─ .agent/                         # правила/скіли/workflows для Antigravity
├─ .env.example
└─ .gitattributes / .gitignore
```

### 1.1. Правила монорепо

- `packages/api-client` — **генерований**; будь-яка ручна правка заборонена (CI перевіряє diff).
- `packages/shared` — лише те, що справді спільне між desktop і mobile: форматери чисел/дат, Zod-схеми
  форм, константи (таймфрейми, кольори сигналів). Бізнес-логіка — на сервері.
- Python-пакети не імпортують один одного напряму, крім `finsight_core` та `finsight_ml`.
- `notebooks/` не імпортуються у production; код із ноутбука переноситься в `finsight_ml` і покривається тестами.

---

## 2. Інструменти та версії

| Інструмент | Версія | Роль |
|---|---|---|
| Node.js | 22 LTS | JS/TS |
| pnpm | 10.x | пакетний менеджер (workspaces) |
| Turborepo | 2.x | кеш і паралелізм JS-задач |
| TypeScript | 5.6+ | типи |
| Vite | 6.x | збірка desktop |
| Tauri | 2.x | desktop-обгортка |
| Expo | актуальний stable SDK | mobile |
| Python | 3.12 | бекенд і ML |
| uv | 0.5+ | залежності та venv |
| ruff | 0.7+ | лінтер + форматер |
| mypy | 1.13+ | типи (strict для domain) |
| pytest | 8.x | тести |
| Docker | 27+ | інфраструктура |

Фіксація: `pnpm-lock.yaml`, `uv.lock`, `packageManager` у `package.json`, `requires-python = ">=3.12,<3.13"`.

---

## 3. Команди (`justfile`)

```make
just bootstrap        # pnpm install + uv sync + pre-commit install
just up               # docker compose -f infra/compose/docker-compose.dev.yml up -d
just down
just migrate          # alembic upgrade head
just migration "msg"  # alembic revision --autogenerate -m "msg"
just seed             # демо-інструменти, портфелі, новини

just dev-api          # uvicorn --reload :8000
just dev-worker       # celery worker
just dev-beat         # celery beat
just dev-streamer     # python -m finsight_streamer
just dev-desktop      # pnpm --filter desktop tauri dev
just dev-mobile       # pnpm --filter mobile expo start

just gen-openapi      # python -m app.scripts.openapi > openapi.json
just gen-client       # openapi-typescript → packages/api-client
just check-api        # CI: генерація не дає diff

just test             # pytest + vitest
just test-py
just test-js
just e2e              # playwright
just lint             # ruff check + eslint + prettier --check
just typecheck        # mypy + tsc --noEmit
just ci               # lint + typecheck + test + build

just backfill BTCUSDT 1h
just train configs/models/lgbm_btc_1h.yaml
just experiment E03
just backtest configs/experiments/E09_walkforward.yaml
just demo-snapshot    # офлайн-набір для захисту
just deploy           # лише через CI, локально — заборонено
```

### 3.1. Windows: налаштування `just`

За замовчуванням `just` шукає `sh`, якого на чистому Windows немає. На початку `justfile`:

```make
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set dotenv-load := true
```

Рецепти пишемо як однорядкові виклики (`uv run ...`, `pnpm ...`), без bash-ізмів, щоб вони однаково працювали в PowerShell, WSL і CI.
Якщо репозиторій лежить у WSL-файловій системі, `just` запускається з WSL і це обмеження не діє.

---

## 4. Змінні середовища

`.env.example` (реальні значення — ніколи в git):

```dotenv
# App
APP_ENV=local                  # local | staging | prod | demo
DEMO_MODE=false
LOG_LEVEL=INFO
API_BASE_URL=http://localhost:8000
CORS_ORIGINS=http://localhost:5173,http://tauri.localhost,tauri://localhost   # http://tauri.localhost — Windows

# Auth
AUTH_MODE=single_user          # single_user (курсова) | jwt (диплом)
JWT_SECRET=change-me-32-chars-min
ACCESS_TOKEN_TTL_MIN=15
REFRESH_TOKEN_TTL_DAYS=30

# Data
DATABASE_URL=postgresql+asyncpg://finsight:finsight@localhost:5432/finsight
REDIS_URL=redis://localhost:6379/0
STORAGE_BACKEND=local          # local (курсова) | s3 (диплом)
ARTIFACTS_DIR=./var/artifacts
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio123
S3_BUCKET_MODELS=models
S3_BUCKET_DATASETS=datasets
S3_BUCKET_REPORTS=reports

# Market data
BINANCE_REST=https://api.binance.com
BINANCE_WS=wss://stream.binance.com:9443
ALPHAVANTAGE_API_KEY=
YAHOO_ENABLED=true

# News / LLM
CRYPTOPANIC_API_KEY=
TELEGRAM_API_ID=
TELEGRAM_API_HASH=
OPENROUTER_API_KEY=
# Slug'и нижче — плейсхолдери: звірити з каталогом OpenRouter до першого запуску (стартова перевірка в llm/healthcheck)
LLM_SENTIMENT_MODEL=deepseek/deepseek-v4.1-flash
LLM_ASSISTANT_MODEL=google/gemini-3.8-flash
LLM_DAILY_BUDGET_USD=2.00
LLM_CACHE_TTL_DAYS=30

# Mobile
EXPO_ACCESS_TOKEN=

# Observability
SENTRY_DSN=
METRICS_TOKEN=
```

Правила:
- Конфіг читається через Pydantic Settings; відсутня обов'язкова змінна → падіння на старті з чітким повідомленням.
- Секрети ніколи не потрапляють у клієнтський бандл (лише публічний `API_BASE_URL`).
- `DEMO_MODE=true` блокує будь-які зовнішні виклики та записи, окрім читання снапшоту.
- `AUTH_MODE=single_user` дозволений лише при `APP_ENV` ∈ {local, demo}; в іншому разі падіння на старті.

---

## 5. CI/CD

### 5.1. Pull request (`ci.yml`)

```yaml
jobs:
  js:
    - pnpm install --frozen-lockfile
    - pnpm lint && pnpm typecheck
    - pnpm test
    - pnpm --filter desktop build
  python:
    - uv sync --frozen
    - ruff check . && ruff format --check .
    - mypy services/api/app services/ml/src
    - pytest -q --cov=app --cov=finsight_ml --cov-fail-under=70
  contract:
    - just gen-openapi && git diff --exit-code openapi.json
    - just gen-client && git diff --exit-code packages/api-client
  db:
    - alembic upgrade head
    - alembic check          # env.py: include_object ігнорує службові індекси/таблиці TimescaleDB
  docker:
    - docker build -f infra/docker/Dockerfile.api .
    - docker build -f infra/docker/Dockerfile.worker .
  security:
    - pip-audit
    - pnpm audit --prod
    - gitleaks detect
```

### 5.2. Main (`deploy.yml`)

1. Build & push образів у GHCR (теги: `sha`, `latest`).
2. SSH на VPS: `docker compose pull`, `alembic upgrade head`, `up -d`, healthcheck.
3. Smoke-тест: `/health`, `/ready`, один запит `/instruments`.
4. Rollback: попередній тег зберігається у `deploy/.last_good`.

### 5.3. Nightly (`nightly.yml`)

- Перевірка свіжості даних по всіх інструментах.
- `pip-audit`, `trivy image`, `npm audit`.
- Тренувальний smoke на 1 активі / 1 фолді (перевірка, що пайплайн не зламався).
- Перевірка бекапу: restore у тимчасову БД + `COUNT(*)` по ключових таблицях.

### 5.4. E2E (`e2e.yml`, на PR з тегом `e2e` або вночі)

- Підняти compose, застосувати міграції, засіяти демо-дані.
- Playwright: відкрити desktop-web, дочекатися графіка, перемкнути TF, відкрити прогноз, створити бектест.

---

## 6. Тестування

| Рівень | Що покриває | Інструмент | Цільове покриття |
|---|---|---|---|
| Unit (Python) | domain, services, features, labels, metrics | pytest | ≥ 70% `app/`, ≥ 80% `finsight_ml` |
| Property-based | triple barrier, VaR, метрики | hypothesis | ключові інваріанти |
| Integration | API + PG + Redis (testcontainers) | pytest + httpx AsyncClient | усі роутери |
| Contract | OpenAPI snapshot, TS-клієнт компілюється | CI diff | 100% ендпоінтів |
| ML validation | leakage, parity train/serve, reproducibility | pytest | обов'язково |
| E2E | 5 критичних сценаріїв desktop | Playwright | 100% сценаріїв демо |
| Mobile | екрани, навігація | Jest + RNTL | smoke |
| Load | 50 RPS читання, 10 WS | k6 / Locust | p95 у межах SLO |
| Security | ізоляція даних, rate limit, auth | pytest | усі захищені роутери |

**Обов'язкові тести перед захистом:**
1. `test_no_leakage.py` — point-in-time.
2. `test_feature_parity.py` — однакові ознаки в train і serve.
3. `test_backtest_no_lookahead.py` — виконання на next open.
4. `test_authz_isolation.py` — чужий портфель → 404.
5. `test_reproducibility.py` — той самий `dataset_hash` → ті самі метрики.
6. `test_label_backtest_consistency.py` — бар'єри бектесту збігаються з бар'єрами розмітки (за `exit_mode=label_barriers`).
7. `test_news_availability.py` — жодна новина з `available_at > t` не потрапляє в ознаки.
8. `test_fold_scoped_fit.py` — HMM, Isolation Forest і scaler навчаються лише на train-фолді.

---

## 7. Стандарти коду

### Python
- ruff (line-length 100), import order, `from __future__ import annotations`.
- mypy strict для `domain/` і `finsight_ml`; для infra — `disallow_untyped_defs`.
- Заборонено: `print`, `except Exception: pass`, `datetime.utcnow()`, naive datetime, `eval`.
- Публічні функції — docstring із призначенням, аргументами, винятками.
- SQL — лише через SQLAlchemy Core/ORM або параметризовані запити.

### TypeScript
- `strict: true`, `noUncheckedIndexedAccess`.
- ESLint flat config + Prettier; імпорти через `@/`.
- Компоненти — функціональні, без `any`; стани — TanStack Query (сервер) + Zustand (UI).
- Числа форматувати через `Intl.NumberFormat` у `packages/shared`.

### Git
- Conventional Commits: `feat(forecast): add meta-learner`, `fix(api): ...`.
- Гілки: `feat/...`, `fix/...`, `chore/...`; merge у `main` лише через PR.
- Теги: `v0.1.0-coursework`, `v0.5.0-beta`, `v1.0.0-thesis`.
- `.gitattributes`: `* text=auto eol=lf`, бінарники — `binary`.

---

## 8. Windows-специфіка

| Проблема | Рішення |
|---|---|
| TA-Lib не збирається | Не залежимо від нього: власні індикатори (pandas/numpy); TA-Lib — лише опційний «оракул» у тестах (conda-forge) |
| Довгі шляхи | Увімкнути LongPathsEnabled; тримати репо близько до кореня |
| Docker повільний | WSL2 backend, код у WSL-файловій системі для швидкого hot reload |
| Tauri потребує MSVC + WebView2 | Встановити Build Tools і WebView2 Runtime (інструкція в README) |
| CRLF ламає pre-commit | `.gitattributes` + `core.autocrlf=false` |
| Порт зайнятий | `just dev-api PORT=8001`, конфіг у `.env` |
| Celery prefork не працює | Локально `-P solo` (або `-P threads`); prefork — лише в Linux-контейнерах |
| Metro (Expo) не бачить symlink-пакети pnpm | `.npmrc`: `node-linker=hoisted` |
| `just` не знаходить `sh` | `set windows-shell` (див. 3.1) або запуск із WSL |

---

## 9. Бекапи та відновлення

| Що | Як | Частота | Зберігання |
|---|---|---|---|
| PostgreSQL | `pg_dump -Fc` + WAL-архів | щодня / щогодини | S3/MinIO, 14 днів |
| MinIO (моделі, звіти) | `mc mirror` | щодня | інший бакет/диск |
| Конфіги та секрети | зашифрований архів | при зміні | поза репозиторієм |
| Код | Git + GitHub | кожен пуш | — |

Перевірка відновлення: щотижня restore у `staging`-БД, smoke-тест, фіксація часу відновлення (RTO ≤ 30 хв).
