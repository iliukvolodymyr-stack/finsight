"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-17 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enable TimescaleDB extension if available in postgres installation
    op.execute(
        """
        DO $$
        BEGIN
            BEGIN
                CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
            EXCEPTION WHEN OTHERS THEN
                RAISE NOTICE 'TimescaleDB extension not available, skipping.';
            END;
        END
        $$;
        """
    )

    # Users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False, server_default="user"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_users_email", "users", ["email"])

    # Instruments table
    op.create_table(
        "instruments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("symbol", sa.String(length=30), nullable=False, unique=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("asset_class", sa.String(length=20), nullable=False),
        sa.Column("exchange", sa.String(length=30), nullable=False),
        sa.Column("quote_currency", sa.String(length=10), nullable=False, server_default="USDT"),
        sa.Column("tick_size", sa.Numeric(precision=28, scale=10), nullable=True),
        sa.Column("lot_size", sa.Numeric(precision=28, scale=10), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("idx_instruments_symbol", "instruments", ["symbol"])
    op.create_index("idx_instruments_asset_class", "instruments", ["asset_class"])

    # Candles table
    op.create_table(
        "candles",
        sa.Column("instrument_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("trades_count", sa.Integer(), nullable=True),
        sa.Column("vwap", sa.Float(), nullable=True),
        sa.Column("is_closed", sa.Boolean(), nullable=False, server_default="true"),
        sa.PrimaryKeyConstraint("instrument_id", "timeframe", "open_time", name="pk_candles"),
    )
    op.create_index(
        "idx_candles_symbol_tf_time",
        "candles",
        ["instrument_id", "timeframe", "open_time"],
    )

    # Convert candles to TimescaleDB hypertable if timescaledb is present
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                PERFORM create_hypertable('candles', 'open_time', if_not_exists => TRUE);
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.drop_table("candles")
    op.drop_table("instruments")
    op.drop_table("users")
