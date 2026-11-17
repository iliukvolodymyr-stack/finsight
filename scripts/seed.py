import asyncio
from pathlib import Path

import uuid_utils as uuid
import yaml
from app.infra.db import AsyncSessionLocal
from app.infra.models import InstrumentModel, UserModel
from finsight_core.logging import setup_logging
from loguru import logger
from sqlalchemy import select


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
                    id=uuid.uuid7(),
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
                logger.info("Added instrument: {}", sym)

        await session.commit()
    logger.info("Seeding complete.")


if __name__ == "__main__":
    asyncio.run(seed())
