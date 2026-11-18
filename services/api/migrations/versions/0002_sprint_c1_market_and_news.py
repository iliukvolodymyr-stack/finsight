"""sprint c1 market data quality and rss news tables

Revision ID: 0002_sprint_c1_market_and_news
Revises: 0001_initial_schema
Create Date: 2026-11-18 13:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_sprint_c1_market_and_news"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Extend candles table with adjusted_close, quote_volume, source, ingested_at, FK and OHLC checks
    op.add_column("candles", sa.Column("adjusted_close", sa.Float(), nullable=True))
    op.add_column("candles", sa.Column("quote_volume", sa.Float(), nullable=True))
    op.add_column(
        "candles",
        sa.Column("source", sa.String(length=30), nullable=False, server_default="binance"),
    )
    op.add_column(
        "candles",
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_foreign_key(
        "fk_candles_instrument_id",
        "candles",
        "instruments",
        ["instrument_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_check_constraint(
        "ck_candles_timeframe",
        "candles",
        "timeframe IN ('1m','15m','1h','4h','1d')",
    )
    op.create_check_constraint("ck_candles_high_gte_low", "candles", "high >= low")
    op.create_check_constraint(
        "ck_candles_high_gte_oc", "candles", "high >= open AND high >= close"
    )
    op.create_check_constraint("ck_candles_low_lte_oc", "candles", "low <= open AND low <= close")
    op.create_check_constraint("ck_candles_volume_nonneg", "candles", "volume >= 0")

    # 2. ingestion_runs (including rows_rejected and heartbeat support)
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("job_name", sa.String(length=50), nullable=False),
        sa.Column("source", sa.String(length=30), nullable=False),
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("timeframe", sa.String(length=10), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rows_written", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rows_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gaps_found", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="running"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running','ok','partial','failed')",
            name="ck_ingestion_runs_status",
        ),
    )
    op.create_index(
        "idx_ingestion_runs_job_started",
        "ingestion_runs",
        ["job_name", "started_at"],
    )

    # 3. data_gaps
    op.create_table(
        "data_gaps",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("gap_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gap_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bars_missing", sa.Integer(), nullable=False),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_data_gaps_inst_tf_start",
        "data_gaps",
        ["instrument_id", "timeframe", "gap_start"],
    )

    # 4. news_sources
    op.create_table(
        "news_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(length=20), nullable=False, server_default="rss"),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("url", sa.Text(), nullable=False, unique=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("trust_score", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # 5. news_articles (with available_at and availability_basis from day 1)
    op.create_table(
        "news_articles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("news_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("url_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False, server_default=""),
        sa.Column("language", sa.String(length=10), nullable=False, server_default="en"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "availability_basis",
            sa.String(length=30),
            nullable=False,
            server_default="live",
        ),
        sa.Column("dedup_cluster_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "raw",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.CheckConstraint(
            "availability_basis IN ('live','backfill_published_proxy')",
            name="ck_news_articles_availability_basis",
        ),
    )
    op.create_index("idx_news_articles_content_hash", "news_articles", ["content_hash"])
    op.create_index("idx_news_articles_published_at", "news_articles", ["published_at"])
    op.create_index("idx_news_articles_available_at", "news_articles", ["available_at"])
    op.create_index("idx_news_articles_dedup_cluster", "news_articles", ["dedup_cluster_id"])

    # 6. news_article_instruments
    op.create_table(
        "news_article_instruments",
        sa.Column(
            "article_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("news_articles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "relevance",
            sa.Numeric(precision=4, scale=3),
            nullable=False,
            server_default="1.000",
        ),
        sa.PrimaryKeyConstraint("article_id", "instrument_id", name="pk_news_article_instruments"),
    )


def downgrade() -> None:
    op.drop_table("news_article_instruments")
    op.drop_table("news_articles")
    op.drop_table("news_sources")
    op.drop_table("data_gaps")
    op.drop_table("ingestion_runs")

    op.drop_constraint("ck_candles_volume_nonneg", "candles", type_="check")
    op.drop_constraint("ck_candles_low_lte_oc", "candles", type_="check")
    op.drop_constraint("ck_candles_high_gte_oc", "candles", type_="check")
    op.drop_constraint("ck_candles_high_gte_low", "candles", type_="check")
    op.drop_constraint("ck_candles_timeframe", "candles", type_="check")
    op.drop_constraint("fk_candles_instrument_id", "candles", type_="foreignkey")
    op.drop_column("candles", "ingested_at")
    op.drop_column("candles", "source")
    op.drop_column("candles", "quote_volume")
    op.drop_column("candles", "adjusted_close")
