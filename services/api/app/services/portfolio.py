import math
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, cast

import pandas as pd
from app.core.errors import AppException, NotFoundException
from app.infra.models import (
    CandleModel,
    InstrumentModel,
    PortfolioModel,
    PortfolioPositionModel,
    PortfolioTransactionModel,
    RiskSnapshotModel,
    new_uuid7,
)
from finsight_ml.risk import (
    PortfolioValuation,
    TransactionRecord,
    align_portfolio_daily_returns,
    compute_correlation_matrix,
    compute_portfolio_risk_metrics,
    recompute_positions,
    valuate_positions,
)
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession


async def _get_owned_portfolio(
    session: AsyncSession,
    user_id: uuid.UUID,
    portfolio_id: uuid.UUID,
) -> PortfolioModel:
    stmt = select(PortfolioModel).where(
        PortfolioModel.id == portfolio_id,
        PortfolioModel.user_id == user_id,
        PortfolioModel.archived_at.is_(None),
    )
    port = (await session.execute(stmt)).scalar_one_or_none()
    if port is None:
        raise NotFoundException(f"Portfolio {portfolio_id} not found")
    return port


async def _load_portfolio_tx_records(
    session: AsyncSession,
    portfolio_id: uuid.UUID,
) -> tuple[
    list[PortfolioTransactionModel], dict[uuid.UUID, InstrumentModel], list[TransactionRecord]
]:
    stmt = (
        select(PortfolioTransactionModel, InstrumentModel)
        .join(InstrumentModel, InstrumentModel.id == PortfolioTransactionModel.instrument_id)
        .where(PortfolioTransactionModel.portfolio_id == portfolio_id)
        .order_by(PortfolioTransactionModel.executed_at.asc())
    )
    rows = (await session.execute(stmt)).all()
    tx_models: list[PortfolioTransactionModel] = []
    inst_map: dict[uuid.UUID, InstrumentModel] = {}
    records: list[TransactionRecord] = []

    for tx, inst in rows:
        tx_models.append(tx)
        inst_map[inst.id] = inst
        records.append(
            TransactionRecord(
                instrument_id=str(inst.id),
                symbol=inst.symbol,
                side=cast(Literal["BUY", "SELL"], tx.side),
                quantity=float(tx.quantity),
                price=float(tx.price),
                fee=float(tx.fee),
                executed_at=tx.executed_at,
            )
        )
    return tx_models, inst_map, records


async def _latest_market_prices(
    session: AsyncSession,
    instrument_ids: list[uuid.UUID],
) -> dict[str, float]:
    if not instrument_ids:
        return {}

    cutoff = datetime.now(UTC) - timedelta(days=30)
    rows = (
        await session.execute(
            select(
                CandleModel.instrument_id,
                CandleModel.timeframe,
                CandleModel.open_time,
                CandleModel.close,
            )
            .where(
                CandleModel.instrument_id.in_(instrument_ids),
                CandleModel.timeframe.in_(["1h", "1d"]),
                CandleModel.is_closed.is_(True),
                CandleModel.open_time >= cutoff,
            )
            .order_by(desc(CandleModel.open_time))
        )
    ).all()

    best_ts: dict[str, tuple[datetime, int]] = {}
    prices: dict[str, float] = {}
    tf_prio = {"1h": 2, "1d": 1}
    for iid, tf, open_ts, close_val in rows:
        price = float(close_val)
        if not math.isfinite(price) or price <= 0.0:
            continue
        key = str(iid)
        prio = tf_prio.get(str(tf), 0)
        cur = (open_ts, prio)
        if key not in best_ts or cur > best_ts[key]:
            best_ts[key] = cur
            prices[key] = price

    # Fallback without 30-day cutoff if an inactive instrument only has older historical bars
    missing = [iid for iid in instrument_ids if str(iid) not in prices]
    for iid in missing:
        row = (
            await session.execute(
                select(CandleModel.close)
                .where(
                    CandleModel.instrument_id == iid,
                    CandleModel.timeframe.in_(["1h", "1d"]),
                    CandleModel.is_closed.is_(True),
                )
                .order_by(desc(CandleModel.open_time))
                .limit(1)
            )
        ).first()
        if row is not None and float(row[0]) > 0.0:
            prices[str(iid)] = float(row[0])
    return prices


async def _sync_positions_table(
    session: AsyncSession,
    portfolio_id: uuid.UUID,
    positions_map: dict[str, Any],
) -> None:
    await session.execute(
        delete(PortfolioPositionModel).where(PortfolioPositionModel.portfolio_id == portfolio_id)
    )
    for iid_str, state in positions_map.items():
        if state.quantity > 1e-12:
            session.add(
                PortfolioPositionModel(
                    portfolio_id=portfolio_id,
                    instrument_id=uuid.UUID(iid_str),
                    quantity=Decimal(f"{state.quantity:.10f}"),
                    avg_price=Decimal(f"{state.avg_price:.10f}"),
                    opened_at=state.opened_at,
                    updated_at=state.updated_at,
                )
            )


async def list_portfolios_summary(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> list[dict[str, Any]]:
    ports = (
        (
            await session.execute(
                select(PortfolioModel)
                .where(PortfolioModel.user_id == user_id, PortfolioModel.archived_at.is_(None))
                .order_by(PortfolioModel.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    out: list[dict[str, Any]] = []
    for p in ports:
        val = await compute_portfolio_valuation(session, user_id, p.id)
        out.append(
            {
                "id": str(p.id),
                "name": p.name,
                "base_currency": p.base_currency,
                "created_at": p.created_at.isoformat().replace("+00:00", "Z"),
                "updated_at": p.updated_at.isoformat().replace("+00:00", "Z"),
                "total_value": round(val.total_market_value, 2),
                "total_cost_basis": round(val.total_cost_basis, 2),
                "unrealized_pnl": round(val.total_unrealized_pnl, 2),
                "realized_pnl": round(val.total_realized_pnl, 2),
                "positions_count": len(val.positions),
            }
        )
    return out


async def create_portfolio_record(
    session: AsyncSession,
    user_id: uuid.UUID,
    name: str,
    base_currency: str = "USD",
) -> PortfolioModel:
    currency = base_currency.upper().strip()
    if currency != "USD":
        raise AppException(
            status_code=422,
            code="UNSUPPORTED_CURRENCY",
            title="Unsupported base currency",
            detail="Currently only 'USD' base currency is supported (no FX rate conversion yet).",
        )
    now = datetime.now(UTC)
    port = PortfolioModel(
        id=new_uuid7(),
        user_id=user_id,
        name=name.strip(),
        base_currency=currency,
        created_at=now,
        updated_at=now,
    )
    session.add(port)
    await session.commit()
    await session.refresh(port)
    return port


async def compute_portfolio_valuation(
    session: AsyncSession,
    user_id: uuid.UUID,
    portfolio_id: uuid.UUID,
) -> PortfolioValuation:
    await _get_owned_portfolio(session, user_id, portfolio_id)
    _, inst_map, records = await _load_portfolio_tx_records(session, portfolio_id)
    if not records:
        return valuate_positions({}, {}, {})

    states, realized_pnl, total_fees = recompute_positions(records)
    active_ids = [uuid.UUID(k) for k, v in states.items() if v.quantity > 1e-12]
    prices = await _latest_market_prices(session, active_ids)
    ac_map = {str(iid): inst.asset_class for iid, inst in inst_map.items()}
    return valuate_positions(
        states,
        current_prices=prices,
        asset_classes=ac_map,
        total_realized_pnl=realized_pnl,
        total_fees=total_fees,
    )


async def add_portfolio_transaction(
    session: AsyncSession,
    user_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    instrument: InstrumentModel,
    side: Literal["BUY", "SELL"],
    quantity: float,
    price: float,
    fee: float,
    executed_at: datetime,
    note: str | None = None,
    idempotency_key: str | None = None,
) -> PortfolioTransactionModel:
    await _get_owned_portfolio(session, user_id, portfolio_id)

    if idempotency_key:
        existing = (
            await session.execute(
                select(PortfolioTransactionModel).where(
                    PortfolioTransactionModel.portfolio_id == portfolio_id,
                    PortfolioTransactionModel.client_idempotency_key == idempotency_key,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

    _, _, existing_records = await _load_portfolio_tx_records(session, portfolio_id)
    proposed = TransactionRecord(
        instrument_id=str(instrument.id),
        symbol=instrument.symbol,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        executed_at=executed_at,
    )

    try:
        updated_states, _, _ = recompute_positions([*existing_records, proposed])
    except ValueError as exc:
        raise AppException(
            status_code=422,
            code="INSUFFICIENT_POSITION",
            title="Invalid transaction",
            detail=str(exc),
        ) from exc

    tx = PortfolioTransactionModel(
        id=new_uuid7(),
        portfolio_id=portfolio_id,
        instrument_id=instrument.id,
        side=side,
        quantity=Decimal(f"{quantity:.10f}"),
        price=Decimal(f"{price:.10f}"),
        fee=Decimal(f"{fee:.10f}"),
        executed_at=executed_at,
        note=note,
        client_idempotency_key=idempotency_key,
    )
    session.add(tx)
    await _sync_positions_table(session, portfolio_id, updated_states)
    await session.commit()
    await session.refresh(tx)
    return tx


async def cancel_portfolio_transaction(
    session: AsyncSession,
    user_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    tx_id: uuid.UUID,
) -> PortfolioTransactionModel:
    await _get_owned_portfolio(session, user_id, portfolio_id)
    orig = (
        await session.execute(
            select(PortfolioTransactionModel).where(
                PortfolioTransactionModel.id == tx_id,
                PortfolioTransactionModel.portfolio_id == portfolio_id,
            )
        )
    ).scalar_one_or_none()
    if orig is None:
        raise NotFoundException(f"Transaction {tx_id} not found")

    inst = await session.get(InstrumentModel, orig.instrument_id)
    if inst is None:
        raise NotFoundException(f"Instrument {orig.instrument_id} not found")

    reverse_side: Literal["BUY", "SELL"] = "SELL" if orig.side == "BUY" else "BUY"
    exec_ts = max(datetime.now(UTC), orig.executed_at + timedelta(seconds=1))
    return await add_portfolio_transaction(
        session=session,
        user_id=user_id,
        portfolio_id=portfolio_id,
        instrument=inst,
        side=reverse_side,
        quantity=float(orig.quantity),
        price=float(orig.price),
        fee=0.0,
        executed_at=exec_ts,
        note=f"Compensating reversal for tx {tx_id}",
        idempotency_key=f"cancel:{tx_id}",
    )


async def _load_daily_closes_by_symbol(
    session: AsyncSession,
    instruments: list[InstrumentModel],
    lookback_days: int,
    timeframe: str = "1d",
) -> dict[str, pd.Series]:
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    out: dict[str, pd.Series] = {}
    for inst in instruments:
        rows = (
            await session.execute(
                select(CandleModel.open_time, CandleModel.close)
                .where(
                    CandleModel.instrument_id == inst.id,
                    CandleModel.timeframe == timeframe,
                    CandleModel.is_closed.is_(True),
                    CandleModel.open_time >= cutoff,
                )
                .order_by(CandleModel.open_time.asc())
            )
        ).all()
        if rows:
            idx = pd.DatetimeIndex([r[0] for r in rows])
            vals = [float(r[1]) for r in rows]
            out[inst.symbol] = pd.Series(vals, index=idx, dtype="float64")
    return out


async def compute_and_store_portfolio_risk(
    session: AsyncSession,
    user_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    window_days: int = 90,
) -> dict[str, Any]:
    val = await compute_portfolio_valuation(session, user_id, portfolio_id)
    if not val.positions:
        return {
            "portfolio_id": str(portfolio_id),
            "as_of": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "var_95": 0.0,
            "cvar_95": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "annual_volatility": 0.0,
            "risk_level": "low",
            "weights": {},
            "observations": 0,
            "method": f"historical_1d_{window_days}d",
        }

    inst_ids = [uuid.UUID(p.instrument_id) for p in val.positions]
    insts = list(
        (await session.execute(select(InstrumentModel).where(InstrumentModel.id.in_(inst_ids))))
        .scalars()
        .all()
    )
    closes = await _load_daily_closes_by_symbol(session, insts, lookback_days=window_days + 45)
    port_rets = align_portfolio_daily_returns(
        closes, val.weights_by_symbol, window_days=window_days
    )
    has_crypto = any(p.asset_class == "crypto" for p in val.positions)
    ppy = 365 if has_crypto else 252
    risk = compute_portfolio_risk_metrics(port_rets, val.weights_by_symbol, periods_per_year=ppy)

    as_of_dt = port_rets.index[-1].to_pydatetime() if not port_rets.empty else datetime.now(UTC)
    session.add(
        RiskSnapshotModel(
            id=new_uuid7(),
            portfolio_id=portfolio_id,
            as_of=as_of_dt,
            var_95=float(risk["var_95"]),
            cvar_95=float(risk["cvar_95"]),
            sharpe=float(risk["sharpe"]),
            sortino=float(risk["sortino"]),
            max_drawdown=float(risk["max_drawdown"]),
            annual_volatility=float(risk["annual_volatility"]),
            risk_level=str(risk["risk_level"]),
            weights=dict(risk["weights"]),
            method=str(risk["method"]),
        )
    )
    await session.commit()

    return {
        "portfolio_id": str(portfolio_id),
        "as_of": as_of_dt.isoformat().replace("+00:00", "Z"),
        **risk,
    }


async def compute_portfolio_correlation_view(
    session: AsyncSession,
    user_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    window_days: int = 90,
    timeframe: str = "1d",
) -> dict[str, Any]:
    val = await compute_portfolio_valuation(session, user_id, portfolio_id)
    held_ids = [uuid.UUID(p.instrument_id) for p in val.positions]
    if len(held_ids) >= 2:
        insts = list(
            (
                await session.execute(
                    select(InstrumentModel)
                    .where(InstrumentModel.id.in_(held_ids))
                    .order_by(InstrumentModel.symbol.asc())
                )
            )
            .scalars()
            .all()
        )
    else:
        insts = list(
            (
                await session.execute(
                    select(InstrumentModel)
                    .where(InstrumentModel.is_active.is_(True))
                    .order_by(InstrumentModel.symbol.asc())
                )
            )
            .scalars()
            .all()
        )

    closes = await _load_daily_closes_by_symbol(
        session, insts, lookback_days=window_days + 45, timeframe=timeframe
    )
    corr = compute_correlation_matrix(closes, window_days=window_days)
    return {
        "portfolio_id": str(portfolio_id),
        "timeframe": timeframe,
        "window": f"{window_days}d",
        **corr,
    }


async def compute_portfolio_pnl_breakdown(
    session: AsyncSession,
    user_id: uuid.UUID,
    portfolio_id: uuid.UUID,
    period: str = "30d",
    group_by: str = "instrument",
) -> dict[str, Any]:
    val = await compute_portfolio_valuation(session, user_id, portfolio_id)
    buckets: dict[str, dict[str, float]] = {}

    for pos in val.positions:
        key = pos.symbol if group_by == "instrument" else pos.asset_class
        entry = buckets.setdefault(
            key,
            {"market_value": 0.0, "cost_basis": 0.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0},
        )
        entry["market_value"] += pos.market_value
        entry["cost_basis"] += pos.cost_basis
        entry["unrealized_pnl"] += pos.unrealized_pnl
        entry["realized_pnl"] += pos.realized_pnl

    breakdown = [
        {
            "key": k,
            "market_value": round(v["market_value"], 2),
            "cost_basis": round(v["cost_basis"], 2),
            "unrealized_pnl": round(v["unrealized_pnl"], 2),
            "realized_pnl": round(v["realized_pnl"], 2),
        }
        for k, v in buckets.items()
    ]
    return {
        "portfolio_id": str(portfolio_id),
        "period": period,
        "group_by": group_by,
        "total_market_value": round(val.total_market_value, 2),
        "total_cost_basis": round(val.total_cost_basis, 2),
        "unrealized_pnl": round(val.total_unrealized_pnl, 2),
        "realized_pnl": round(val.total_realized_pnl, 2),
        "total_fees": round(val.total_fees, 2),
        "breakdown": breakdown,
    }
