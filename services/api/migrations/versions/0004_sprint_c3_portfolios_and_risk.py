"""sprint c3 portfolios, transactions, risk snapshots, and market regimes tables

Revision ID: 0004_sprint_c3_portfolios
Revises: 0003_sprint_c2_ml_and_backtests
Create Date: 2026-11-26 13:25:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_sprint_c3_portfolios"
down_revision: str | None = "0003_sprint_c2_ml_and_backtests"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. portfolios
    op.create_table(
        "portfolios",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("base_currency", sa.String(length=10), nullable=False, server_default="USD"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_portfolios_user_archived", "portfolios", ["user_id", "archived_at"])

    # 2. portfolio_positions (materialized aggregate recalculated inside tx)
    op.create_table(
        "portfolio_positions",
        sa.Column(
            "portfolio_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("quantity", sa.Numeric(precision=28, scale=10), nullable=False),
        sa.Column("avg_price", sa.Numeric(precision=28, scale=10), nullable=False),
        sa.Column(
            "opened_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("portfolio_id", "instrument_id", name="pk_portfolio_positions"),
        sa.CheckConstraint("quantity >= 0", name="ck_portfolio_positions_qty_nonneg"),
        sa.CheckConstraint("avg_price >= 0", name="ck_portfolio_positions_price_nonneg"),
    )

    # 3. portfolio_transactions (source of truth for PnL and positions)
    op.create_table(
        "portfolio_transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=28, scale=10), nullable=False),
        sa.Column("price", sa.Numeric(precision=28, scale=10), nullable=False),
        sa.Column(
            "fee",
            sa.Numeric(precision=28, scale=10),
            nullable=False,
            server_default="0",
        ),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("client_idempotency_key", sa.String(length=120), nullable=True),
        sa.CheckConstraint("side IN ('BUY','SELL')", name="ck_portfolio_transactions_side"),
        sa.CheckConstraint("quantity > 0", name="ck_portfolio_transactions_qty_pos"),
        sa.CheckConstraint("price > 0", name="ck_portfolio_transactions_price_pos"),
        sa.CheckConstraint("fee >= 0", name="ck_portfolio_transactions_fee_nonneg"),
        sa.UniqueConstraint(
            "portfolio_id",
            "client_idempotency_key",
            name="uq_portfolio_transactions_idempotency",
        ),
    )
    op.create_index(
        "idx_portfolio_transactions_port_exec",
        "portfolio_transactions",
        ["portfolio_id", "executed_at"],
    )

    # 4. risk_snapshots
    op.create_table(
        "risk_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "portfolio_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("portfolios.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("var_95", sa.Float(), nullable=False),
        sa.Column("cvar_95", sa.Float(), nullable=False),
        sa.Column("sharpe", sa.Float(), nullable=False),
        sa.Column("sortino", sa.Float(), nullable=False),
        sa.Column("max_drawdown", sa.Float(), nullable=False),
        sa.Column("annual_volatility", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column(
            "weights",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "method",
            sa.String(length=50),
            nullable=False,
            server_default="historical_1d_90d",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "risk_level IN ('low','medium','high')",
            name="ck_risk_snapshots_level",
        ),
    )
    op.create_index("idx_risk_snapshots_port_asof", "risk_snapshots", ["portfolio_id", "as_of"])

    # 5. market_regimes (causal HMM baseline regime states)
    op.create_table(
        "market_regimes",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("regime", sa.String(length=20), nullable=False),
        sa.Column(
            "proba",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "model_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("model_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint(
            "instrument_id",
            "timeframe",
            "as_of",
            "model_version_id",
            name="pk_market_regimes",
        ),
        sa.CheckConstraint(
            "regime IN ('bull','bear','range','high_vol')",
            name="ck_market_regimes_regime",
        ),
    )
    op.create_index(
        "idx_market_regimes_inst_tf_asof",
        "market_regimes",
        ["instrument_id", "timeframe", "as_of"],
    )


def downgrade() -> None:
    op.drop_table("market_regimes")
    op.drop_table("risk_snapshots")
    op.drop_table("portfolio_transactions")
    op.drop_table("portfolio_positions")
    op.drop_table("portfolios")
