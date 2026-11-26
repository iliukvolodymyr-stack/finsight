import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from app.infra.db import AsyncSessionLocal
from app.infra.models import (
    InstrumentModel,
    PortfolioModel,
    PortfolioPositionModel,
    PortfolioTransactionModel,
    RiskSnapshotModel,
    new_uuid7,
)
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError


@pytest.mark.asyncio
async def test_portfolio_constraints_and_cascade() -> None:
    user_id = uuid.UUID("00000000-0000-7000-8000-000000000001")
    port_id = new_uuid7()
    inst_id = new_uuid7()
    tx_id = new_uuid7()
    snap_id = new_uuid7()
    now = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)

    try:
        async with AsyncSessionLocal() as session:
            session.add(
                InstrumentModel(
                    id=inst_id,
                    symbol=f"PRT{inst_id.hex[:5].upper()}",
                    name="Portfolio Test Asset",
                    asset_class="crypto",
                    exchange="binance",
                    quote_currency="USDT",
                    is_active=False,
                )
            )
            session.add(
                PortfolioModel(
                    id=port_id,
                    user_id=user_id,
                    name="Schema Test Portfolio",
                    base_currency="USD",
                )
            )
            await session.flush()

            session.add(
                PortfolioPositionModel(
                    portfolio_id=port_id,
                    instrument_id=inst_id,
                    quantity=Decimal("0.5000000000"),
                    avg_price=Decimal("85000.0000000000"),
                )
            )
            session.add(
                PortfolioTransactionModel(
                    id=tx_id,
                    portfolio_id=port_id,
                    instrument_id=inst_id,
                    side="BUY",
                    quantity=Decimal("0.5000000000"),
                    price=Decimal("85000.0000000000"),
                    fee=Decimal("42.50"),
                    executed_at=now,
                )
            )
            session.add(
                RiskSnapshotModel(
                    id=snap_id,
                    portfolio_id=port_id,
                    as_of=now,
                    var_95=0.041,
                    cvar_95=0.056,
                    sharpe=1.25,
                    sortino=1.60,
                    max_drawdown=0.11,
                    annual_volatility=0.35,
                    risk_level="medium",
                    weights={"TEST": 1.0},
                )
            )
            await session.commit()

        # Invalid transaction side must violate check constraint
        async with AsyncSessionLocal() as session:
            session.add(
                PortfolioTransactionModel(
                    id=new_uuid7(),
                    portfolio_id=port_id,
                    instrument_id=inst_id,
                    side="MARGIN",
                    quantity=Decimal("1.0"),
                    price=Decimal("100.0"),
                    fee=Decimal("0"),
                    executed_at=now,
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()
            await session.rollback()
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(delete(PortfolioModel).where(PortfolioModel.id == port_id))
            await session.execute(delete(InstrumentModel).where(InstrumentModel.id == inst_id))
            await session.commit()
