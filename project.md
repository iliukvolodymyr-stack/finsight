# FinSight AI — правила проєкту (завжди діють)

## Джерело істини
Читай перед будь-якою задачею: `docs/01-architecture.md`, `docs/02-data-model.md`, `docs/03-api-and-contracts.md`,
`docs/04-ml-pipeline.md`, `docs/05-roadmap.md`, `docs/06-repo-and-devops.md`.
Не змінюй архітектуру, схему БД чи API-контракт мовчки: запропонуй зміну окремо (ADR або діф до docs/) і дочекайся підтвердження.

## Структура
Monorepo: `apps/desktop`, `apps/mobile`, `packages/{api-client,shared,config}`, `services/{api,ml,streamer,py-core}`,
`infra/`, `configs/`, `scripts/`, `docs/`. Python — uv workspace, JS — pnpm + Turborepo. Єдина точка входу — `just`.
`packages/api-client` генерується з OpenAPI, вручну не редагується.

## Поточний етап: КУРСОВА (до 25.12.2026)
Дозволено: Binance (BTCUSDT, ETHUSDT) + SPY, 1h/1d, beat-опитування барів, індикатори, ~30 ознак, triple barrier,
LightGBM, бектест, VaR/Sharpe/MaxDD, desktop як web (Vite).
НЕ робити без явної команди: streamer, MinIO, Tauri-обгортка, auth (`AUTH_MODE=single_user`), mobile, NLP/LLM,
deep-модель, Марковіц, асистент. Збір сирих RSS-новин (без LLM) — дозволено з C1.

## Python
- 3.12, типи обов'язкові, ruff + mypy (strict для `domain/` і `finsight_ml`), pytest.
- Заборонено: `print`, `except Exception: pass`, `datetime.utcnow()`, naive datetime, `eval`, сирий SQL-конкатенацією.
- Конфіг лише через Pydantic Settings; секрети не в коді.
- Зміни схеми — лише Alembic. Індикатори — власні (pandas/numpy), без TA-Lib у runtime.
- Windows: Celery локально з `-P solo`; `just` з `windows-shell`.

## ML (критично)
- Жодного look-ahead: ознаки в момент t — лише дані з `available_at <= t`.
- `train_test_split(shuffle=True)` та KFold без purge заборонені. Валідація — purged walk-forward з embargo.
- HMM, Isolation Forest, scaler — навчаються окремо в кожному фолді лише на train.
- Новини: `available_at`, не `published_at`; live і backfill звітуються окремо.
- Бектест: сигнал на закритті t, виконання на open t+1, комісія 10 bps + проковзування 5 bps,
  виходи = бар'єри розмітки (`exit_mode=label_barriers`), завжди з baseline (buy&hold, EMA-кросовер, random).
- Критерії приймання A1–A7 фіксуються до експериментів; гіперпараметри на test-фолдах не підбираємо.
- Sharpe > 3 або accuracy > 70% — вважай витоком, поки не доведено інше.

## Процес
- Задача: план (артефакт) → підтвердження → реалізація → тести → короткий звіт. Не коміть із червоними тестами.
- Conventional Commits; гілки `feat/`, `fix/`, `chore/`; merge у `main` через PR.
- Один агент — одна директорія (`services/api`, `services/ml`, `apps/desktop`). Агенти не редагують чужі зони.
- Неоднозначність у вимогах — питай, не вгадуй.
- Пояснюй ключові ML-рішення в коментарях/ADR: автор має захищати їх усно.
