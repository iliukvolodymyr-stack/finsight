import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import uuid_utils
from app.infra.db import Base
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column


def new_uuid7() -> uuid.UUID:
    """Generate a standard library uuid.UUID backed by UUIDv7 (time-ordered)."""
    return uuid.UUID(bytes=uuid_utils.uuid7().bytes)


class UserModel(Base):
    __tablename__ = "users"
    __table_args__ = (Index("idx_users_email", "email"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class InstrumentModel(Base):
    __tablename__ = "instruments"
    __table_args__ = (
        Index("idx_instruments_symbol", "symbol"),
        Index("idx_instruments_asset_class", "asset_class"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    symbol: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(20), nullable=False)
    exchange: Mapped[str] = mapped_column(String(30), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(10), default="USDT", nullable=False)
    tick_size: Mapped[float | None] = mapped_column(Numeric(28, 10), nullable=True)
    lot_size: Mapped[float | None] = mapped_column(Numeric(28, 10), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class CandleModel(Base):
    __tablename__ = "candles"
    __table_args__ = (
        PrimaryKeyConstraint("instrument_id", "timeframe", "open_time", name="pk_candles"),
        Index("idx_candles_symbol_tf_time", "instrument_id", "timeframe", "open_time"),
        CheckConstraint("timeframe IN ('1m','15m','1h','4h','1d')", name="ck_candles_timeframe"),
        CheckConstraint("high >= low", name="ck_candles_high_gte_low"),
        CheckConstraint("high >= open AND high >= close", name="ck_candles_high_gte_oc"),
        CheckConstraint("low <= open AND low <= close", name="ck_candles_low_lte_oc"),
        CheckConstraint("volume >= 0", name="ck_candles_volume_nonneg"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    quote_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    trades_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vwap: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_closed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    source: Mapped[str] = mapped_column(String(30), default="binance", nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class IngestionRunModel(Base):
    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running','ok','partial','failed')", name="ck_ingestion_runs_status"
        ),
        Index("idx_ingestion_runs_job_started", "job_name", "started_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="SET NULL"),
        nullable=True,
    )
    timeframe: Mapped[str | None] = mapped_column(String(10), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rows_written: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    gaps_found: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="running", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class DataGapModel(Base):
    __tablename__ = "data_gaps"
    __table_args__ = (
        Index("idx_data_gaps_inst_tf_start", "instrument_id", "timeframe", "gap_start"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    gap_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    gap_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    bars_missing: Mapped[int] = mapped_column(Integer, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class NewsSourceModel(Base):
    __tablename__ = "news_sources"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    kind: Mapped[str] = mapped_column(String(20), default="rss", nullable=False)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    trust_score: Mapped[float] = mapped_column(Float, default=0.8, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class NewsArticleModel(Base):
    __tablename__ = "news_articles"
    __table_args__ = (
        CheckConstraint(
            "availability_basis IN ('live','backfill_published_proxy')",
            name="ck_news_articles_availability_basis",
        ),
        Index("idx_news_articles_content_hash", "content_hash"),
        Index("idx_news_articles_published_at", "published_at"),
        Index("idx_news_articles_available_at", "available_at"),
        Index("idx_news_articles_dedup_cluster", "dedup_cluster_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("news_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, default="", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    availability_basis: Mapped[str] = mapped_column(String(30), default="live", nullable=False)
    dedup_cluster_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)


class NewsArticleInstrumentModel(Base):
    __tablename__ = "news_article_instruments"
    __table_args__ = (
        PrimaryKeyConstraint("article_id", "instrument_id", name="pk_news_article_instruments"),
    )

    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("news_articles.id", ondelete="CASCADE"),
        nullable=False,
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    relevance: Mapped[float] = mapped_column(Numeric(4, 3), default=1.0, nullable=False)


class FeatureSetModel(Base):
    __tablename__ = "feature_sets"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_feature_sets_name_version"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    code_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class FeatureModel(Base):
    __tablename__ = "features"
    __table_args__ = (
        PrimaryKeyConstraint(
            "instrument_id", "timeframe", "as_of", "feature_set_id", name="pk_features"
        ),
        Index("idx_features_as_of", "as_of"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_set_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("feature_sets.id", ondelete="CASCADE"),
        nullable=False,
    )
    values: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    quality_flags: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)


class LabelModel(Base):
    __tablename__ = "labels"
    __table_args__ = (
        PrimaryKeyConstraint(
            "instrument_id", "timeframe", "as_of", "label_set_version", name="pk_labels"
        ),
        CheckConstraint("label IN (-1, 0, 1)", name="ck_labels_label"),
        CheckConstraint(
            "exit_reason IN ('pt','sl','vertical','low_vol')",
            name="ck_labels_exit_reason",
        ),
        Index("idx_labels_t1", "instrument_id", "timeframe", "t1"),
    )

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label_set_version: Mapped[str] = mapped_column(String(50), nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str] = mapped_column(String(20), nullable=False)
    t1: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    barrier_up: Mapped[float | None] = mapped_column(Float, nullable=True)
    barrier_down: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)


class ModelVersionModel(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        CheckConstraint(
            "task IN ('direction_3class','price_quantile','regime')",
            name="ck_model_versions_task",
        ),
        CheckConstraint(
            "status IN ('candidate','production','archived','failed')",
            name="ck_model_versions_status",
        ),
        Index(
            "uq_model_versions_scope_semver",
            "name",
            "task",
            "instrument_id",
            "timeframe",
            "semver",
            unique=True,
            postgresql_nulls_not_distinct=True,
        ),
        Index(
            "uq_model_versions_prod_scope",
            "name",
            "task",
            "instrument_id",
            "timeframe",
            unique=True,
            postgresql_where=text("status = 'production'"),
            postgresql_nulls_not_distinct=True,
        ),
        Index("idx_model_versions_inst_tf_status", "instrument_id", "timeframe", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False)
    algo: Mapped[str] = mapped_column(String(50), nullable=False)
    semver: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="candidate", nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="SET NULL"),
        nullable=True,
    )
    timeframe: Mapped[str | None] = mapped_column(String(10), nullable=True)
    feature_set_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("feature_sets.id", ondelete="RESTRICT"),
        nullable=False,
    )
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    train_data_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class PredictionModel(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("signal IN ('BUY','SELL','HOLD')", name="ck_predictions_signal"),
        CheckConstraint(
            "abs((p_buy + p_sell + p_hold) - 1.0) < 0.001",
            name="ck_predictions_prob_sum",
        ),
        CheckConstraint(
            "p_buy >= 0 AND p_sell >= 0 AND p_hold >= 0",
            name="ck_predictions_prob_nonneg",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_predictions_confidence_range",
        ),
        UniqueConstraint(
            "instrument_id",
            "timeframe",
            "as_of",
            "model_version_id",
            name="uq_predictions_inst_tf_asof_model",
        ),
        Index("idx_predictions_inst_tf_asof", "instrument_id", "timeframe", "as_of"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    horizon_bars: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    p_buy: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    p_sell: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    p_hold: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    signal: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    price_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    features_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    component_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    explanation: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class BacktestModel(Base):
    __tablename__ = "backtests"
    __table_args__ = (
        CheckConstraint(
            "mode IN ('static_model','walk_forward_retrain')",
            name="ck_backtests_mode",
        ),
        CheckConstraint(
            "status IN ('queued','running','done','failed','cancelled')",
            name="ck_backtests_status",
        ),
        Index("idx_backtests_user_created", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    strategy_config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    instrument_scope: Mapped[list[Any]] = mapped_column(JSONB, default=list, nullable=False)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mode: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    equity_curve_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BacktestFoldModel(Base):
    __tablename__ = "backtest_folds"
    __table_args__ = (PrimaryKeyConstraint("backtest_id", "fold_no", name="pk_backtest_folds"),)

    backtest_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("backtests.id", ondelete="CASCADE"),
        nullable=False,
    )
    fold_no: Mapped[int] = mapped_column(Integer, nullable=False)
    train_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    train_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    val_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    val_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    test_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)


class BacktestTradeModel(Base):
    __tablename__ = "backtest_trades"
    __table_args__ = (
        PrimaryKeyConstraint("backtest_id", "seq", name="pk_backtest_trades"),
        CheckConstraint("side IN ('LONG','SHORT')", name="ck_backtest_trades_side"),
        Index("idx_backtest_trades_inst_entry", "backtest_id", "instrument_id", "entry_at"),
    )

    backtest_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("backtests.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
    )
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=False)
    exit_reason: Mapped[str] = mapped_column(String(30), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    fold_no: Mapped[int | None] = mapped_column(Integer, nullable=True)


class JobModel(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('backtest','optimize','report','retrain','inference')",
            name="ck_jobs_kind",
        ),
        CheckConstraint(
            "status IN ('queued','running','done','failed','cancelled')",
            name="ck_jobs_status",
        ),
        CheckConstraint(
            "progress >= 0.0 AND progress <= 1.0",
            name="ck_jobs_progress_range",
        ),
        Index("idx_jobs_user_created", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid7)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    kind: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
