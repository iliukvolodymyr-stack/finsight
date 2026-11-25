import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from app.api.v1.endpoints import backtests as backtests_ep
from app.core.deps import CurrentUser, get_current_user
from app.infra.db import AsyncSessionLocal
from app.infra.models import (
    BacktestFoldModel,
    BacktestModel,
    BacktestTradeModel,
    CandleModel,
    FeatureSetModel,
    InstrumentModel,
    JobModel,
    ModelVersionModel,
    PredictionModel,
    new_uuid7,
)
from app.main import app
from finsight_ml.features.pipeline import FeaturePipeline
from finsight_ml.models.calibration import ProbabilityCalibrator
from finsight_ml.models.conformal import VolatilityNormalizedConformal
from finsight_ml.models.lgbm import LGBMForecaster
from httpx import AsyncClient
from sqlalchemy import delete


@pytest.fixture
async def ml_api_fixture(tmp_path: Path) -> AsyncIterator[dict[str, str]]:
    """
    Isolated DB + artifact fixture for Sprint C2 predictions, models, jobs, and backtest APIs.
    """
    inst_prod_id = new_uuid7()
    inst_cand_id = new_uuid7()
    fs_id = new_uuid7()
    mv_prod_id = new_uuid7()
    mv_cand_id = new_uuid7()
    pred_id = new_uuid7()
    bt_id = new_uuid7()
    job_id = new_uuid7()

    sym_prod = f"MLBTC{inst_prod_id.hex[:4].upper()}"
    sym_cand = f"MLSPY{inst_cand_id.hex[:4].upper()}"
    pipeline = FeaturePipeline()

    # Build a tiny genuine LightGBM + Calibrator + Conformal artifact in tmp_path
    n_bars = 1600
    rng = np.random.default_rng(17)
    start_dt = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    dates = pd.date_range(start_dt, periods=n_bars, freq="h", tz="UTC")
    close = 50000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.006, size=n_bars)))
    open_p = np.roll(close, 1)
    open_p[0] = 50000.0
    high = np.maximum(open_p, close) * 1.003
    low = np.minimum(open_p, close) * 0.997
    vol = rng.lognormal(5.0, 0.4, size=n_bars)
    df_syn = pd.DataFrame(
        {"open": open_p, "high": high, "low": low, "close": close, "volume": vol}, index=dates
    )

    feats = pipeline.compute(df_syn).iloc[220:500]
    y_syn = pd.Series(rng.choice([-1, 0, 1], size=len(feats), p=[0.3, 0.4, 0.3]), index=feats.index)
    model = LGBMForecaster(n_estimators=15, min_child_samples=5, random_state=42).fit(feats, y_syn)
    cal = ProbabilityCalibrator().fit(model.predict_proba(feats), y_syn.to_numpy(dtype=np.int64))
    conf = VolatilityNormalizedConformal(alpha=0.20, horizon_bars=24).fit(
        rng.normal(0.0, 0.02, size=100),
        np.full(100, 0.005),
    )

    art_dir = tmp_path / "models" / sym_prod / "0.1.0"
    model.save(art_dir / "model.joblib")
    cal.save(art_dir / "calibrator.joblib")
    conf.save(art_dir / "conformal.json")

    equity_path = tmp_path / "bt_equity.parquet"
    pd.DataFrame(
        {
            "timestamp": dates[-20:],
            "strategy_equity": np.linspace(10000.0, 10800.0, 20),
            "benchmark_equity": np.linspace(10000.0, 10400.0, 20),
            "ema_equity": np.linspace(10000.0, 10200.0, 20),
        }
    ).to_parquet(equity_path, index=False)

    demo_user_id = uuid.UUID("00000000-0000-7000-8000-000000000001")
    train_data_end = dates[-25].to_pydatetime()

    try:
        async with AsyncSessionLocal() as session:
            session.add_all(
                [
                    InstrumentModel(
                        id=inst_prod_id,
                        symbol=sym_prod,
                        name="Synthetic ML BTC",
                        asset_class="crypto",
                        exchange="binance",
                        quote_currency="USDT",
                        is_active=False,
                    ),
                    InstrumentModel(
                        id=inst_cand_id,
                        symbol=sym_cand,
                        name="Synthetic ML SPY",
                        asset_class="etf",
                        exchange="nyse",
                        quote_currency="USD",
                        is_active=False,
                    ),
                    FeatureSetModel(
                        id=fs_id,
                        name=f"core_test_{fs_id.hex[:6]}",
                        version=99,
                        schema_hash=pipeline.schema_hash,
                    ),
                ]
            )
            await session.flush()

            session.add_all(
                [
                    CandleModel(
                        instrument_id=inst_prod_id,
                        timeframe="1h",
                        open_time=dates[i].to_pydatetime(),
                        open=float(open_p[i]),
                        high=float(high[i]),
                        low=float(low[i]),
                        close=float(close[i]),
                        volume=float(vol[i]),
                        is_closed=True,
                        source="binance",
                    )
                    for i in range(n_bars)
                ]
            )

            session.add_all(
                [
                    ModelVersionModel(
                        id=mv_prod_id,
                        name=f"lgbm_test_{sym_prod.lower()}",
                        task="direction_3class",
                        algo="lightgbm",
                        semver="0.1.0",
                        status="production",
                        params={"entry_threshold": 0.45, "min_margin": 0.05, "horizon_bars": 24},
                        metrics={
                            "oos_classification": {
                                "models": {"lgbm_calibrated": {"macro_f1": 0.48}}
                            }
                        },
                        artifact_uri=str(art_dir),
                        instrument_id=inst_prod_id,
                        timeframe="1h",
                        feature_set_id=fs_id,
                        dataset_hash="a" * 64,
                        train_data_end=train_data_end,
                        promoted_at=datetime.now(UTC),
                    ),
                    ModelVersionModel(
                        id=mv_cand_id,
                        name=f"lgbm_test_{sym_cand.lower()}",
                        task="direction_3class",
                        algo="lightgbm",
                        semver="0.1.0",
                        status="candidate",
                        params={"entry_threshold": 0.55, "min_margin": 0.10, "horizon_bars": 24},
                        metrics={},
                        artifact_uri=str(art_dir),
                        instrument_id=inst_cand_id,
                        timeframe="1h",
                        feature_set_id=fs_id,
                        dataset_hash="b" * 64,
                        train_data_end=train_data_end,
                    ),
                ]
            )
            await session.flush()

            session.add(
                PredictionModel(
                    id=pred_id,
                    instrument_id=inst_prod_id,
                    timeframe="1h",
                    as_of=dates[-5].to_pydatetime(),
                    horizon_bars=24,
                    model_version_id=mv_prod_id,
                    p_buy=Decimal("0.6200"),
                    p_sell=Decimal("0.1500"),
                    p_hold=Decimal("0.2300"),
                    signal="BUY",
                    confidence=Decimal("0.6200"),
                    price_low=49100.0,
                    price_high=51500.0,
                    features_hash="c" * 64,
                    component_versions={"lgbm_core": str(mv_prod_id)},
                    explanation=[
                        {"feature": "rsi14", "contribution": 0.21, "scale": "raw_log_odds"}
                    ],
                )
            )

            session.add(
                BacktestModel(
                    id=bt_id,
                    user_id=demo_user_id,
                    name="Test BTC WF",
                    strategy_config={"timeframe": "1h", "exit_mode": "label_barriers"},
                    instrument_scope=[str(inst_prod_id)],
                    period_start=dates[0].to_pydatetime(),
                    period_end=dates[-1].to_pydatetime(),
                    mode="walk_forward_retrain",
                    status="done",
                    metrics={"strategy": {"sharpe": 1.15, "total_return": 0.08, "trades_count": 1}},
                    equity_curve_uri=str(equity_path),
                    finished_at=datetime.now(UTC),
                )
            )
            await session.flush()

            session.add(
                BacktestFoldModel(
                    backtest_id=bt_id,
                    fold_no=1,
                    train_start=dates[0].to_pydatetime(),
                    train_end=dates[1000].to_pydatetime(),
                    val_start=dates[1024].to_pydatetime(),
                    val_end=dates[1200].to_pydatetime(),
                    test_start=dates[1224].to_pydatetime(),
                    test_end=dates[-1].to_pydatetime(),
                    metrics={"sharpe": 1.15, "total_return": 0.08, "trades_count": 1},
                )
            )
            session.add(
                BacktestTradeModel(
                    backtest_id=bt_id,
                    seq=1,
                    instrument_id=inst_prod_id,
                    side="LONG",
                    entry_at=dates[-10].to_pydatetime(),
                    entry_price=50000.0,
                    exit_at=dates[-2].to_pydatetime(),
                    exit_price=50800.0,
                    quantity=0.2,
                    fee=15.0,
                    pnl=145.0,
                    exit_reason="pt",
                    confidence=0.62,
                    fold_no=1,
                )
            )
            session.add(
                JobModel(
                    id=job_id,
                    user_id=demo_user_id,
                    kind="backtest",
                    status="done",
                    progress=1.0,
                    result={"backtest_id": str(bt_id)},
                )
            )
            await session.commit()

        yield {
            "sym_prod": sym_prod,
            "sym_cand": sym_cand,
            "inst_prod_id": str(inst_prod_id),
            "pred_id": str(pred_id),
            "bt_id": str(bt_id),
            "job_id": str(job_id),
            "train_data_end": train_data_end.isoformat(),
        }
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(delete(JobModel).where(JobModel.id == job_id))
            await session.execute(delete(BacktestModel).where(BacktestModel.id == bt_id))
            await session.execute(
                delete(PredictionModel).where(PredictionModel.instrument_id == inst_prod_id)
            )
            await session.execute(
                delete(ModelVersionModel).where(ModelVersionModel.id.in_([mv_prod_id, mv_cand_id]))
            )
            await session.execute(delete(FeatureSetModel).where(FeatureSetModel.id == fs_id))
            await session.execute(
                delete(InstrumentModel).where(InstrumentModel.id.in_([inst_prod_id, inst_cand_id]))
            )
            await session.commit()


@pytest.mark.asyncio
async def test_predictions_history_run_and_model_not_ready(
    client: AsyncClient, ml_api_fixture: dict[str, str]
) -> None:
    sym_prod = ml_api_fixture["sym_prod"]
    sym_cand = ml_api_fixture["sym_cand"]

    # 1. Latest prediction returns seeded prediction + explanation
    r_latest = await client.get(
        "/api/v1/predictions", params={"instrument_ids": [sym_prod], "timeframe": "1h"}
    )
    assert r_latest.status_code == 200
    items = r_latest.json()
    assert len(items) == 1
    assert items[0]["signal"] == "BUY"
    assert items[0]["explanation_scale"] == "raw_log_odds"

    # 2. Candidate-only symbol raises 503 MODEL_NOT_READY
    r_not_ready = await client.get(
        "/api/v1/predictions", params={"instrument_ids": [sym_cand], "timeframe": "1h"}
    )
    assert r_not_ready.status_code == 503
    assert r_not_ready.json()["code"] == "MODEL_NOT_READY"

    # 3. History + get-by-id
    r_hist = await client.get(
        "/api/v1/predictions/history", params={"instrument_id": sym_prod, "timeframe": "1h"}
    )
    assert r_hist.status_code == 200
    assert len(r_hist.json()) >= 1

    r_detail = await client.get(f"/api/v1/predictions/{ml_api_fixture['pred_id']}")
    assert r_detail.status_code == 200
    assert r_detail.json()["id"] == ml_api_fixture["pred_id"]

    # 4. Trigger real inference on latest candle
    r_run = await client.post(
        "/api/v1/predictions/run", json={"instrument_id": sym_prod, "timeframe": "1h"}
    )
    assert r_run.status_code == 200
    ran = r_run.json()
    assert ran["symbol"] == sym_prod
    assert ran["signal"] in {"BUY", "SELL", "HOLD"}
    assert len(ran["explanation"]) == 5

    # 5. Non-admin gets 403 on POST /predictions/run
    async def _non_admin() -> CurrentUser:
        return CurrentUser(id="00000000-0000-7000-8000-000000000001", email="u@test", role="user")

    app.dependency_overrides[get_current_user] = _non_admin
    try:
        r_forbidden = await client.post(
            "/api/v1/predictions/run", json={"instrument_id": sym_prod, "timeframe": "1h"}
        )
        assert r_forbidden.status_code == 403
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_backtests_jobs_models_and_in_sample_guard(
    client: AsyncClient,
    ml_api_fixture: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sym_prod = ml_api_fixture["sym_prod"]
    bt_id = ml_api_fixture["bt_id"]
    job_id = ml_api_fixture["job_id"]

    # List models
    r_models = await client.get("/api/v1/models")
    assert r_models.status_code == 200
    assert any(m["symbol"] == sym_prod and m["status"] == "production" for m in r_models.json())

    # List & inspect existing backtest, equity, trades, folds, and job
    r_bt_list = await client.get("/api/v1/backtests")
    assert r_bt_list.status_code == 200
    assert any(b["id"] == bt_id for b in r_bt_list.json())

    assert (await client.get(f"/api/v1/backtests/{bt_id}")).status_code == 200
    r_eq = await client.get(f"/api/v1/backtests/{bt_id}/equity")
    assert r_eq.status_code == 200 and len(r_eq.json()["items"]) == 20
    r_tr = await client.get(f"/api/v1/backtests/{bt_id}/trades")
    assert r_tr.status_code == 200 and r_tr.json()["total"] == 1
    r_fd = await client.get(f"/api/v1/backtests/{bt_id}/folds")
    assert r_fd.status_code == 200 and len(r_fd.json()) == 1
    r_jb = await client.get(f"/api/v1/jobs/{job_id}")
    assert r_jb.status_code == 200 and r_jb.json()["status"] == "done"

    # Period < 12 months rejected (422)
    r_short = await client.post(
        "/api/v1/backtests",
        json={
            "name": "Too short",
            "instrument_ids": [sym_prod],
            "timeframe": "1h",
            "period": {"from": "2026-01-01", "to": "2026-03-01"},
            "mode": "walk_forward_retrain",
        },
    )
    assert r_short.status_code == 422

    # static_model overlapping train_data_end blocked (422 IN_SAMPLE_LEAKAGE_BLOCKED)
    r_leak = await client.post(
        "/api/v1/backtests",
        json={
            "name": "Leaky static",
            "instrument_ids": [sym_prod],
            "timeframe": "1h",
            "period": {"from": "2025-01-01", "to": "2026-10-15"},
            "mode": "static_model",
        },
    )
    assert r_leak.status_code == 422
    assert r_leak.json()["code"] == "IN_SAMPLE_LEAKAGE_BLOCKED"

    # Valid walk_forward_retrain enqueues via JobRunner and cleans up created DB rows
    mock_runner = AsyncMock()
    monkeypatch.setattr(backtests_ep, "get_job_runner", lambda: mock_runner)
    r_ok = await client.post(
        "/api/v1/backtests",
        json={
            "name": "Clean WF request",
            "instrument_ids": [sym_prod],
            "timeframe": "1h",
            "period": {"from": "2025-01-01", "to": "2026-06-01"},
            "mode": "walk_forward_retrain",
        },
    )
    assert r_ok.status_code == 202
    created = r_ok.json()
    mock_runner.enqueue_backtest.assert_awaited_once()

    async with AsyncSessionLocal() as session:
        await session.execute(delete(JobModel).where(JobModel.id == uuid.UUID(created["job_id"])))
        await session.execute(
            delete(BacktestModel).where(BacktestModel.id == uuid.UUID(created["backtest_id"]))
        )
        await session.commit()
