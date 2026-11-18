from datetime import UTC, datetime
from decimal import Decimal

import pytest
from app.infra.db import AsyncSessionLocal
from app.infra.models import (
    FeatureSetModel,
    InstrumentModel,
    ModelVersionModel,
    PredictionModel,
    new_uuid7,
)
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError


@pytest.mark.asyncio
async def test_model_versions_single_production_per_instrument_invariant() -> None:
    fs_id = new_uuid7()
    inst_btc_id = new_uuid7()
    inst_eth_id = new_uuid7()
    m1_id = new_uuid7()
    m2_id = new_uuid7()
    m3_id = new_uuid7()
    m4_id = new_uuid7()
    model_name = f"lgbm_core_{fs_id.hex[:6]}"
    train_end = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

    try:
        async with AsyncSessionLocal() as session:
            session.add_all(
                [
                    InstrumentModel(
                        id=inst_btc_id,
                        symbol=f"TBTC{inst_btc_id.hex[:4].upper()}",
                        name="Test BTC",
                        asset_class="crypto",
                        exchange="binance",
                        quote_currency="USDT",
                    ),
                    InstrumentModel(
                        id=inst_eth_id,
                        symbol=f"TETH{inst_eth_id.hex[:4].upper()}",
                        name="Test ETH",
                        asset_class="crypto",
                        exchange="binance",
                        quote_currency="USDT",
                    ),
                    FeatureSetModel(
                        id=fs_id,
                        name=f"test_fs_{fs_id.hex[:8]}",
                        version=1,
                        schema_hash="a" * 64,
                    ),
                ]
            )
            await session.flush()

            # Production model for BTC 1h
            session.add(
                ModelVersionModel(
                    id=m1_id,
                    name=model_name,
                    task="direction_3class",
                    algo="lightgbm",
                    semver="0.1.0",
                    status="production",
                    artifact_uri="local://test/btc/0.1.0",
                    instrument_id=inst_btc_id,
                    timeframe="1h",
                    feature_set_id=fs_id,
                    dataset_hash="b" * 64,
                    train_data_end=train_end,
                )
            )
            # Same model name & semver for ETH 1h in production is allowed
            session.add(
                ModelVersionModel(
                    id=m2_id,
                    name=model_name,
                    task="direction_3class",
                    algo="lightgbm",
                    semver="0.1.0",
                    status="production",
                    artifact_uri="local://test/eth/0.1.0",
                    instrument_id=inst_eth_id,
                    timeframe="1h",
                    feature_set_id=fs_id,
                    dataset_hash="b" * 64,
                    train_data_end=train_end,
                )
            )
            # Candidate alongside production for BTC 1h is allowed
            session.add(
                ModelVersionModel(
                    id=m3_id,
                    name=model_name,
                    task="direction_3class",
                    algo="lightgbm",
                    semver="0.2.0",
                    status="candidate",
                    artifact_uri="local://test/btc/0.2.0",
                    instrument_id=inst_btc_id,
                    timeframe="1h",
                    feature_set_id=fs_id,
                    dataset_hash="b" * 64,
                    train_data_end=train_end,
                )
            )
            await session.commit()

        # Second production version for same (name, task, instrument_id, timeframe) must fail
        async with AsyncSessionLocal() as session:
            session.add(
                ModelVersionModel(
                    id=m4_id,
                    name=model_name,
                    task="direction_3class",
                    algo="lightgbm",
                    semver="0.3.0",
                    status="production",
                    artifact_uri="local://test/btc/0.3.0",
                    instrument_id=inst_btc_id,
                    timeframe="1h",
                    feature_set_id=fs_id,
                    dataset_hash="b" * 64,
                    train_data_end=train_end,
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(ModelVersionModel).where(
                    ModelVersionModel.id.in_([m1_id, m2_id, m3_id, m4_id])
                )
            )
            await session.execute(delete(FeatureSetModel).where(FeatureSetModel.id == fs_id))
            await session.execute(
                delete(InstrumentModel).where(InstrumentModel.id.in_([inst_btc_id, inst_eth_id]))
            )
            await session.commit()


@pytest.mark.asyncio
async def test_predictions_probability_sum_check_constraint() -> None:
    inst_id = new_uuid7()
    fs_id = new_uuid7()
    mv_id = new_uuid7()
    pred_ok_id = new_uuid7()
    pred_bad_id = new_uuid7()
    now = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)

    try:
        async with AsyncSessionLocal() as session:
            session.add(
                InstrumentModel(
                    id=inst_id,
                    symbol=f"TST{inst_id.hex[:6].upper()}",
                    name="Test Instrument",
                    asset_class="crypto",
                    exchange="binance",
                    quote_currency="USDT",
                )
            )
            session.add(
                FeatureSetModel(
                    id=fs_id,
                    name=f"test_fs_{fs_id.hex[:8]}",
                    version=1,
                    schema_hash="c" * 64,
                )
            )
            await session.flush()
            session.add(
                ModelVersionModel(
                    id=mv_id,
                    name=f"test_mv_{mv_id.hex[:8]}",
                    task="direction_3class",
                    algo="lightgbm",
                    semver="0.1.0",
                    status="candidate",
                    artifact_uri="local://test/0.1.0",
                    instrument_id=inst_id,
                    timeframe="1h",
                    feature_set_id=fs_id,
                    dataset_hash="d" * 64,
                    train_data_end=now,
                )
            )
            await session.flush()

            # Valid probabilities summing to 1.0000
            session.add(
                PredictionModel(
                    id=pred_ok_id,
                    instrument_id=inst_id,
                    timeframe="1h",
                    as_of=now,
                    horizon_bars=24,
                    model_version_id=mv_id,
                    p_buy=Decimal("0.6000"),
                    p_sell=Decimal("0.1500"),
                    p_hold=Decimal("0.2500"),
                    signal="BUY",
                    confidence=Decimal("0.6000"),
                    price_low=90000.0,
                    price_high=95000.0,
                    features_hash="e" * 64,
                    component_versions={"lgbm_core": str(mv_id)},
                    explanation=[{"feature": "rsi14", "contribution": 0.12}],
                )
            )
            await session.commit()

        # Invalid probability sum (1.5000) must fail DB CHECK constraint
        async with AsyncSessionLocal() as session:
            session.add(
                PredictionModel(
                    id=pred_bad_id,
                    instrument_id=inst_id,
                    timeframe="1h",
                    as_of=datetime(2026, 11, 18, 13, 0, tzinfo=UTC),
                    horizon_bars=24,
                    model_version_id=mv_id,
                    p_buy=Decimal("0.5000"),
                    p_sell=Decimal("0.5000"),
                    p_hold=Decimal("0.5000"),
                    signal="BUY",
                    confidence=Decimal("0.5000"),
                    features_hash="e" * 64,
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(PredictionModel).where(PredictionModel.id.in_([pred_ok_id, pred_bad_id]))
            )
            await session.execute(delete(ModelVersionModel).where(ModelVersionModel.id == mv_id))
            await session.execute(delete(FeatureSetModel).where(FeatureSetModel.id == fs_id))
            await session.execute(delete(InstrumentModel).where(InstrumentModel.id == inst_id))
            await session.commit()
