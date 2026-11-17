# FinSight AI

Кросплатформна інтелектуальна система аналізу, оцінки ризиків та прогнозування динаміки фінансових активів.

Формат «2-в-1»: десктопний клієнт (Tauri + React) і мобільний компаньйон (Expo + React Native)
над єдиним аналітичним та ML-бекендом (FastAPI + Celery + PostgreSQL/TimescaleDB + Redis).

## Призначення

- Курсова робота → дипломний проєкт, спеціальність 122 «Комп'ютерні науки».
- Єдина кодова база (monorepo), єдиний API-контракт, спільна типізація для обох клієнтів.

## Структура документації

| Документ | Зміст |
|---|---|
| [docs/00-review-changelog.md](docs/00-review-changelog.md) | Результати рев'ю v1.0 → v1.1: що виправлено і чому, відкриті питання |
| [docs/01-architecture.md](docs/01-architecture.md) | Архітектура системи, ключові рішення (ADR), потоки даних, стек, SLO, безпека, розгортання |
| [docs/02-data-model.md](docs/02-data-model.md) | Схема БД, TimescaleDB-гіпертаблиці, індекси, feature store, point-in-time правила |
| [docs/03-api-and-contracts.md](docs/03-api-and-contracts.md) | REST/WebSocket контракти, авторизація, асинхронні задачі, генерація TS-клієнта |
| [docs/04-ml-pipeline.md](docs/04-ml-pipeline.md) | Дані → ознаки → розмітка → моделі → ансамбль → бектест → реєстр → моніторинг |
| [docs/05-roadmap.md](docs/05-roadmap.md) | Скоуп, етапи, спринти, WBS з оцінками, віхи, ризики, сценарій захисту |
| [docs/06-repo-and-devops.md](docs/06-repo-and-devops.md) | Структура monorepo, інструменти, CI/CD, тестування, змінні середовища |

## Архітектура в одному абзаці

Модульний моноліт на FastAPI (REST + WebSocket) із фоновими Celery-воркерами та окремим
довгоживучим сервісом `market-streamer` для біржових WebSocket-потоків. Дані (OHLCV,
новини, сентимент, прогнози, ризик-метрики) зберігаються в PostgreSQL 16 + TimescaleDB;
Redis відповідає за кеш, pub/sub і черги. ML-пайплайн (LightGBM + 1D-CNN/BiLSTM-Attention,
HMM, Isolation Forest, оптимізація Марковіца) виконується виключно на сервері; клієнти —
тонкі, отримують типові DTO через згенерований з OpenAPI TypeScript-клієнт.

## Швидкий старт (після реалізації Sprint 0)

Передумови: Node.js 22 LTS, Python 3.12 + [uv](https://docs.astral.sh/uv/), Docker Desktop (WSL2),
Rust + WebView2 для Tauri, [just](https://github.com/casey/just).

```bash
just bootstrap       # pnpm install + uv sync + pre-commit install
just up              # postgres, redis (+ minio з етапу D0)
just migrate
just seed
# далі — у окремих терміналах:
just dev-api
just dev-worker      # на Windows Celery запускається з -P solo (див. docs/06, розділ 8)
just dev-desktop
```

Єдиної команди `just dev` немає навмисно: на Windows надійніше запускати процеси в окремих терміналах.

## Статус

- [x] Архітектурне проєктування
- [x] План розробки (v1.1)
- [ ] Узгодження документації з науковим керівником (docs/05, розділ 12)
- [ ] Sprint 0: каркас monorepo, CI, docker-compose, OpenAPI-скелет
- [ ] ...
