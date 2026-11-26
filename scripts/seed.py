import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from app.infra.db import AsyncSessionLocal
from app.infra.models import InstrumentModel, NewsSourceModel, PortfolioModel, UserModel, new_uuid7
from app.services.portfolio import add_portfolio_transaction
from finsight_core.logging import setup_logging
from loguru import logger
from sqlalchemy import select

DEFAULT_NEWS_SOURCES: list[dict[str, object]] = [
    {
        "kind": "rss",
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "trust_score": 0.9,
    },
    {
        "kind": "rss",
        "name": "Cointelegraph",
        "url": "https://cointelegraph.com/rss",
        "trust_score": 0.85,
    },
    {
        "kind": "rss",
        "name": "Decrypt",
        "url": "https://decrypt.co/feed",
        "trust_score": 0.85,
    },
    {
        "kind": "rss",
        "name": "CNBC Markets",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
        "trust_score": 0.9,
    },
    {
        "kind": "rss",
        "name": "MarketWatch",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "trust_score": 0.85,
    },
    {
        "kind": "rss",
        "name": "Yahoo Finance SPY",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
        "trust_score": 0.8,
    },
]


async def seed() -> None:
    setup_logging()
    logger.info("Seeding initial data...")

    config_path = Path("configs/instruments.yaml")
    if not config_path.exists():
        logger.error("configs/instruments.yaml not found")
        return

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    instruments = data.get("instruments", [])

    try:
        async with AsyncSessionLocal() as session:
            # 1. Seed demo user for AUTH_MODE=single_user
            demo_user_id = uuid.UUID("00000000-0000-7000-8000-000000000001")
            user_res = await session.execute(select(UserModel).where(UserModel.id == demo_user_id))
            user = user_res.scalar_one_or_none()
            if not user:
                user = UserModel(
                    id=demo_user_id,
                    email="demo@finsight.local",
                    password_hash="mock_argon2_hash",
                    display_name="Demo User",
                    role="admin",
                    is_active=True,
                )
                session.add(user)
                logger.info("Added default demo user.")

            # 2. Seed instruments
            for item in instruments:
                sym = item["symbol"]
                inst_res = await session.execute(
                    select(InstrumentModel).where(InstrumentModel.symbol == sym)
                )
                inst = inst_res.scalar_one_or_none()
                if not inst:
                    inst = InstrumentModel(
                        id=new_uuid7(),
                        symbol=sym,
                        name=item["name"],
                        asset_class=item["asset_class"],
                        exchange=item["exchange"],
                        quote_currency=item.get("quote_currency", "USDT"),
                        tick_size=item.get("tick_size"),
                        lot_size=item.get("lot_size"),
                        is_active=item.get("is_active", True),
                        metadata_=item.get("metadata", {}),
                    )
                    session.add(inst)
                    logger.info("Added instrument: {} (active={})", sym, inst.is_active)

            # 3. Seed verified RSS news sources
            for src in DEFAULT_NEWS_SOURCES:
                src_name = str(src["name"])
                src_res = await session.execute(
                    select(NewsSourceModel).where(NewsSourceModel.name == src_name)
                )
                existing_src = src_res.scalar_one_or_none()
                if not existing_src:
                    session.add(
                        NewsSourceModel(
                            id=new_uuid7(),
                            kind=str(src["kind"]),
                            name=src_name,
                            url=str(src["url"]),
                            is_active=True,
                            trust_score=float(src["trust_score"]),  # type: ignore[arg-type]
                        )
                    )
                    logger.info("Added news source: {}", src_name)

            await session.flush()

            # 4. Seed 2 demo portfolios idempotently
            sym_map: dict[str, InstrumentModel] = {
                inst.symbol: inst
                for inst in (await session.execute(select(InstrumentModel))).scalars().all()
            }
            demo_portfolios = [
                {
                    "name": "Main Crypto Portfolio",
                    "txs": [
                        ("BTCUSDT", "BUY", 0.35, 68500.0, 23.98, "seed-main-btc-1"),
                        ("ETHUSDT", "BUY", 4.5, 2650.0, 11.93, "seed-main-eth-1"),
                    ],
                },
                {
                    "name": "Balanced Multi-Asset",
                    "txs": [
                        ("BTCUSDT", "BUY", 0.15, 72000.0, 10.80, "seed-bal-btc-1"),
                        ("ETHUSDT", "BUY", 2.0, 2800.0, 5.60, "seed-bal-eth-1"),
                        ("SPY", "BUY", 25.0, 545.0, 1.00, "seed-bal-spy-1"),
                    ],
                },
            ]
            exec_dt = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)

            for p_spec in demo_portfolios:
                p_name = str(p_spec["name"])
                port = (
                    await session.execute(
                        select(PortfolioModel).where(
                            PortfolioModel.user_id == demo_user_id,
                            PortfolioModel.name == p_name,
                            PortfolioModel.archived_at.is_(None),
                        )
                    )
                ).scalar_one_or_none()
                if port is None:
                    port = PortfolioModel(
                        id=new_uuid7(),
                        user_id=demo_user_id,
                        name=p_name,
                        base_currency="USD",
                    )
                    session.add(port)
                    await session.flush()
                    logger.info("Added demo portfolio: {}", p_name)

                for sym, side, qty, price, fee, key in p_spec["txs"]:  # type: ignore[attr-defined]
                    target_inst = sym_map.get(str(sym))
                    if target_inst is not None:
                        await add_portfolio_transaction(
                            session=session,
                            user_id=demo_user_id,
                            portfolio_id=port.id,
                            instrument=target_inst,
                            side=side,  # type: ignore[arg-type]
                            quantity=float(qty),
                            price=float(price),
                            fee=float(fee),
                            executed_at=exec_dt,
                            note="Initial demo seed position",
                            idempotency_key=str(key),
                        )

            await session.commit()
        logger.info("Seeding complete.")
    except Exception as exc:
        logger.warning("Could not seed database (is PostgreSQL container running?): {}", exc)


if __name__ == "__main__":
    asyncio.run(seed())
