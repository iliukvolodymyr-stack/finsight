import asyncio
import uuid
from pathlib import Path

import yaml
from app.infra.db import AsyncSessionLocal
from app.infra.models import InstrumentModel, NewsSourceModel, UserModel, new_uuid7
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

            await session.commit()
        logger.info("Seeding complete.")
    except Exception as exc:
        logger.warning("Could not seed database (is PostgreSQL container running?): {}", exc)


if __name__ == "__main__":
    asyncio.run(seed())
