from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class TransactionRecord:
    instrument_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float
    price: float
    fee: float
    executed_at: datetime


@dataclass(frozen=True)
class PositionState:
    instrument_id: str
    symbol: str
    quantity: float
    avg_price: float
    cost_basis: float
    realized_pnl: float
    total_fees: float
    opened_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ValuatedPosition:
    instrument_id: str
    symbol: str
    asset_class: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    weight: float
    opened_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PortfolioValuation:
    positions: list[ValuatedPosition]
    total_market_value: float
    total_cost_basis: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_fees: float
    weights_by_symbol: dict[str, float]


def recompute_positions(
    transactions: Sequence[TransactionRecord],
) -> tuple[dict[str, PositionState], float, float]:
    """
    Deterministic weighted-average cost-basis spot position ledger.

    Raises ValueError if any SELL exceeds currently held spot quantity.
    Returns `(positions_by_instrument_id, total_realized_pnl, total_fees)`.
    """
    ordered = sorted(transactions, key=lambda t: t.executed_at)
    qty_map: dict[str, float] = {}
    avg_map: dict[str, float] = {}
    sym_map: dict[str, str] = {}
    realized_map: dict[str, float] = {}
    fees_map: dict[str, float] = {}
    opened_map: dict[str, datetime] = {}
    updated_map: dict[str, datetime] = {}

    total_realized = 0.0
    total_fees = 0.0

    for tx in ordered:
        if tx.quantity <= 0 or tx.price <= 0 or tx.fee < 0:
            raise ValueError("Transaction must have quantity > 0, price > 0, and fee >= 0.")

        iid = tx.instrument_id
        sym_map[iid] = tx.symbol
        prev_qty = qty_map.get(iid, 0.0)
        prev_avg = avg_map.get(iid, 0.0)
        fees_map[iid] = fees_map.get(iid, 0.0) + float(tx.fee)
        total_fees += float(tx.fee)
        updated_map[iid] = tx.executed_at

        if tx.side == "BUY":
            if prev_qty <= 1e-12:
                opened_map[iid] = tx.executed_at
                new_qty = float(tx.quantity)
                new_avg = float(tx.price)
            else:
                new_qty = prev_qty + float(tx.quantity)
                new_avg = ((prev_qty * prev_avg) + (float(tx.quantity) * float(tx.price))) / new_qty
            qty_map[iid] = new_qty
            avg_map[iid] = new_avg
            realized_map.setdefault(iid, 0.0)
        elif tx.side == "SELL":
            if float(tx.quantity) > prev_qty + 1e-9:
                raise ValueError(
                    f"Insufficient quantity to SELL {tx.quantity} of {tx.symbol} "
                    f"(held={prev_qty:.8f})."
                )
            sell_qty = min(float(tx.quantity), prev_qty)
            pnl_delta = ((float(tx.price) - prev_avg) * sell_qty) - float(tx.fee)
            realized_map[iid] = realized_map.get(iid, 0.0) + pnl_delta
            total_realized += pnl_delta

            rem_qty = prev_qty - sell_qty
            if rem_qty <= 1e-12:
                qty_map[iid] = 0.0
                avg_map[iid] = 0.0
            else:
                qty_map[iid] = rem_qty
        else:
            raise ValueError(f"Unsupported side: {tx.side}")

    states: dict[str, PositionState] = {}
    for iid, q in qty_map.items():
        avg_p = avg_map[iid]
        states[iid] = PositionState(
            instrument_id=iid,
            symbol=sym_map[iid],
            quantity=q,
            avg_price=avg_p,
            cost_basis=q * avg_p,
            realized_pnl=realized_map.get(iid, 0.0),
            total_fees=fees_map.get(iid, 0.0),
            opened_at=opened_map.get(iid, updated_map[iid]),
            updated_at=updated_map[iid],
        )

    return states, float(total_realized), float(total_fees)


def valuate_positions(
    positions: dict[str, PositionState],
    current_prices: dict[str, float],
    asset_classes: dict[str, str] | None = None,
    total_realized_pnl: float = 0.0,
    total_fees: float = 0.0,
) -> PortfolioValuation:
    """Mark active positions to market and compute normalized portfolio weights."""
    ac_map = asset_classes or {}
    active = [p for p in positions.values() if p.quantity > 1e-12]

    raw_values: list[tuple[PositionState, float, float]] = []
    total_mv = 0.0
    total_cb = 0.0
    for pos in active:
        mkt_price = float(current_prices.get(pos.instrument_id, pos.avg_price))
        mv = pos.quantity * mkt_price
        raw_values.append((pos, mkt_price, mv))
        total_mv += mv
        total_cb += pos.cost_basis

    valuated: list[ValuatedPosition] = []
    weights_by_symbol: dict[str, float] = {}
    total_unr = 0.0

    for pos, mkt_price, mv in sorted(raw_values, key=lambda x: x[2], reverse=True):
        unr = (mkt_price - pos.avg_price) * pos.quantity
        unr_pct = (unr / pos.cost_basis) if pos.cost_basis > 1e-12 else 0.0
        w = (mv / total_mv) if total_mv > 1e-12 else 0.0
        total_unr += unr
        weights_by_symbol[pos.symbol] = float(w)
        valuated.append(
            ValuatedPosition(
                instrument_id=pos.instrument_id,
                symbol=pos.symbol,
                asset_class=ac_map.get(pos.instrument_id, "unknown"),
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                current_price=mkt_price,
                market_value=mv,
                cost_basis=pos.cost_basis,
                unrealized_pnl=unr,
                unrealized_pnl_pct=unr_pct,
                realized_pnl=pos.realized_pnl,
                weight=w,
                opened_at=pos.opened_at,
                updated_at=pos.updated_at,
            )
        )

    return PortfolioValuation(
        positions=valuated,
        total_market_value=float(total_mv),
        total_cost_basis=float(total_cb),
        total_unrealized_pnl=float(total_unr),
        total_realized_pnl=float(total_realized_pnl),
        total_fees=float(total_fees),
        weights_by_symbol=weights_by_symbol,
    )
