import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from app.infra.db import AsyncSessionLocal
from app.infra.models import CandleModel, InstrumentModel, PortfolioModel, new_uuid7
from httpx import AsyncClient
from sqlalchemy import delete


@pytest.fixture
async def portfolio_api_fixture() -> AsyncIterator[dict[str, str]]:
    """
    Isolated DB fixture with 120 daily candles for 2 crypto assets and 1 ETF,
    allowing full end-to-end testing of portfolio CRUD, transactions, VaR/CVaR risk,
    correlation matrix, PnL, and causal HMM regime endpoints.
    """
    btc_id = new_uuid7()
    eth_id = new_uuid7()
    sym_btc = f"PBTC{btc_id.hex[:4].upper()}"
    sym_eth = f"PETH{eth_id.hex[:4].upper()}"

    rng = np.random.default_rng(77)
    n_days = 120
    start_day = datetime(2026, 7, 1, 0, 0, tzinfo=UTC)
    btc_closes = 80000.0 * np.exp(np.cumsum(rng.normal(0.001, 0.025, size=n_days)))
    eth_closes = 3000.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.032, size=n_days)))

    created_port_ids: list[uuid.UUID] = []

    try:
        async with AsyncSessionLocal() as session:
            session.add_all(
                [
                    InstrumentModel(
                        id=btc_id,
                        symbol=sym_btc,
                        name="Portfolio Test BTC",
                        asset_class="crypto",
                        exchange="binance",
                        quote_currency="USDT",
                        is_active=False,
                    ),
                    InstrumentModel(
                        id=eth_id,
                        symbol=sym_eth,
                        name="Portfolio Test ETH",
                        asset_class="crypto",
                        exchange="binance",
                        quote_currency="USDT",
                        is_active=False,
                    ),
                ]
            )
            await session.flush()

            candles = []
            for i in range(n_days):
                dt = start_day + timedelta(days=i)
                bc = float(btc_closes[i])
                ec = float(eth_closes[i])
                candles.append(
                    CandleModel(
                        instrument_id=btc_id,
                        timeframe="1d",
                        open_time=dt,
                        open=bc * 0.99,
                        high=bc * 1.01,
                        low=bc * 0.98,
                        close=bc,
                        volume=500.0,
                        is_closed=True,
                        source="binance",
                    )
                )
                candles.append(
                    CandleModel(
                        instrument_id=eth_id,
                        timeframe="1d",
                        open_time=dt,
                        open=ec * 0.99,
                        high=ec * 1.01,
                        low=ec * 0.98,
                        close=ec,
                        volume=2000.0,
                        is_closed=True,
                        source="binance",
                    )
                )
            session.add_all(candles)
            await session.commit()

        yield {"sym_btc": sym_btc, "sym_eth": sym_eth, "btc_id": str(btc_id), "eth_id": str(eth_id)}
    finally:
        async with AsyncSessionLocal() as session:
            if created_port_ids:
                await session.execute(
                    delete(PortfolioModel).where(PortfolioModel.id.in_(created_port_ids))
                )
            await session.execute(
                delete(InstrumentModel).where(InstrumentModel.id.in_([btc_id, eth_id]))
            )
            await session.commit()


@pytest.mark.asyncio
async def test_portfolios_crud_transactions_risk_correlation_and_regime(
    client: AsyncClient, portfolio_api_fixture: dict[str, str]
) -> None:
    sym_btc = portfolio_api_fixture["sym_btc"]
    sym_eth = portfolio_api_fixture["sym_eth"]

    # 1. Create portfolio
    r_create = await client.post(
        "/api/v1/portfolios", json={"name": "C3 Risk Test", "base_currency": "USD"}
    )
    assert r_create.status_code == 201
    port_id = r_create.json()["id"]

    try:
        # 2. Add BUY transactions (idempotent)
        tx_payload = {
            "instrument_id": sym_btc,
            "side": "BUY",
            "quantity": 0.5,
            "price": 75000.0,
            "fee": 25.0,
            "executed_at": "2026-10-01T12:00:00Z",
        }
        r_tx1 = await client.post(
            f"/api/v1/portfolios/{port_id}/transactions",
            json=tx_payload,
            headers={"Idempotency-Key": f"key-{port_id}-1"},
        )
        assert r_tx1.status_code == 201
        tx1_id = r_tx1.json()["id"]

        # Replay with same Idempotency-Key returns same transaction id
        r_tx1_dup = await client.post(
            f"/api/v1/portfolios/{port_id}/transactions",
            json=tx_payload,
            headers={"Idempotency-Key": f"key-{port_id}-1"},
        )
        assert r_tx1_dup.status_code == 201
        assert r_tx1_dup.json()["id"] == tx1_id

        r_tx2 = await client.post(
            f"/api/v1/portfolios/{port_id}/transactions",
            json={
                "instrument_id": sym_eth,
                "side": "BUY",
                "quantity": 10.0,
                "price": 2800.0,
                "fee": 14.0,
                "executed_at": "2026-10-02T12:00:00Z",
            },
        )
        assert r_tx2.status_code == 201

        # Overselling must fail with 422 INSUFFICIENT_POSITION
        r_oversell = await client.post(
            f"/api/v1/portfolios/{port_id}/transactions",
            json={
                "instrument_id": sym_btc,
                "side": "SELL",
                "quantity": 5.0,
                "price": 80000.0,
            },
        )
        assert r_oversell.status_code == 422
        assert r_oversell.json()["code"] == "INSUFFICIENT_POSITION"

        # 3. Positions, PnL, Risk (VaR/CVaR), Correlation
        r_pos = await client.get(f"/api/v1/portfolios/{port_id}/positions")
        assert r_pos.status_code == 200
        assert len(r_pos.json()["items"]) == 2

        r_pnl = await client.get(
            f"/api/v1/portfolios/{port_id}/pnl", params={"group_by": "instrument"}
        )
        assert r_pnl.status_code == 200
        assert len(r_pnl.json()["breakdown"]) == 2

        r_risk = await client.get(f"/api/v1/portfolios/{port_id}/risk")
        assert r_risk.status_code == 200
        risk = r_risk.json()
        assert risk["var_95"] >= 0.0
        assert risk["cvar_95"] >= risk["var_95"]
        assert risk["risk_level"] in {"low", "medium", "high"}

        r_corr = await client.get(f"/api/v1/portfolios/{port_id}/correlation")
        assert r_corr.status_code == 200
        corr = r_corr.json()
        assert set(corr["symbols"]) == {sym_btc, sym_eth}
        assert len(corr["matrix"]) == 2

        # 4. Regime endpoint (1d series with 120 bars -> valid causal HMM output)
        r_reg = await client.get(
            f"/api/v1/instruments/{sym_btc}/regime", params={"timeframe": "1d", "days": 60}
        )
        assert r_reg.status_code == 200
        assert r_reg.json()["regime"] in {"bear", "range", "bull"}

        # 5. Soft delete archives portfolio
        r_del = await client.delete(f"/api/v1/portfolios/{port_id}")
        assert r_del.status_code == 204
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(PortfolioModel).where(PortfolioModel.id == uuid.UUID(port_id))
            )
            await session.commit()
