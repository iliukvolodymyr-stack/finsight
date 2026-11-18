"""sprint c2 ml registry, predictions, backtests, and jobs tables

Revision ID: 0003_sprint_c2_ml_and_backtests
Revises: 0002_sprint_c1_market_and_news
Create Date: 2026-11-18 15:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_sprint_c2_ml_and_backtests"
down_revision: str | None = "0002_sprint_c1_market_and_news"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. feature_sets
    op.create_table(
        "feature_sets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_hash", sa.String(length=64), nullable=False),
        sa.Column("code_commit", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("name", "version", name="uq_feature_sets_name_version"),
    )

    # 2. features (point-in-time feature log)
    op.create_table(
        "features",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "feature_set_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("feature_sets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("values", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "quality_flags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.PrimaryKeyConstraint(
            "instrument_id", "timeframe", "as_of", "feature_set_id", name="pk_features"
        ),
    )
    op.create_index("idx_features_as_of", "features", ["as_of"])

    # 3. labels (training-only triple-barrier labels with t1 for purge/embargo)
    op.create_table(
        "labels",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label_set_version", sa.String(length=50), nullable=False),
        sa.Column("label", sa.Integer(), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("exit_reason", sa.String(length=20), nullable=False),
        sa.Column("t1", sa.DateTime(timezone=True), nullable=False),
        sa.Column("barrier_up", sa.Float(), nullable=True),
        sa.Column("barrier_down", sa.Float(), nullable=True),
        sa.Column("sample_weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.PrimaryKeyConstraint(
            "instrument_id", "timeframe", "as_of", "label_set_version", name="pk_labels"
        ),
        sa.CheckConstraint("label IN (-1, 0, 1)", name="ck_labels_label"),
        sa.CheckConstraint(
            "exit_reason IN ('pt','sl','vertical','low_vol')",
            name="ck_labels_exit_reason",
        ),
    )
    op.create_index("idx_labels_t1", "labels", ["instrument_id", "timeframe", "t1"])

    # 4. model_versions
    op.create_table(
        "model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("task", sa.String(length=50), nullable=False),
        sa.Column("algo", sa.String(length=50), nullable=False),
        sa.Column("semver", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="candidate"),
        sa.Column(
            "params",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column(
            "feature_set_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("feature_sets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "task IN ('direction_3class','price_quantile','regime')",
            name="ck_model_versions_task",
        ),
        sa.CheckConstraint(
            "status IN ('candidate','production','archived','failed')",
            name="ck_model_versions_status",
        ),
        sa.UniqueConstraint("name", "task", "semver", name="uq_model_versions_name_task_semver"),
    )
    op.create_index(
        "uq_model_versions_prod_name_task",
        "model_versions",
        ["name", "task"],
        unique=True,
        postgresql_where=sa.text("status = 'production'"),
    )

    # 5. predictions
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("horizon_bars", sa.Integer(), nullable=False),
        sa.Column(
            "model_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("p_buy", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("p_sell", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("p_hold", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("signal", sa.String(length=10), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("price_low", sa.Float(), nullable=True),
        sa.Column("price_high", sa.Float(), nullable=True),
        sa.Column("features_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "component_versions",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "explanation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("signal IN ('BUY','SELL','HOLD')", name="ck_predictions_signal"),
        sa.CheckConstraint(
            "abs((p_buy + p_sell + p_hold) - 1.0) < 0.001",
            name="ck_predictions_prob_sum",
        ),
        sa.CheckConstraint(
            "p_buy >= 0 AND p_sell >= 0 AND p_hold >= 0",
            name="ck_predictions_prob_nonneg",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_predictions_confidence_range",
        ),
        sa.UniqueConstraint(
            "instrument_id",
            "timeframe",
            "as_of",
            "model_version_id",
            name="uq_predictions_inst_tf_asof_model",
        ),
    )
    op.create_index(
        "idx_predictions_inst_tf_asof",
        "predictions",
        ["instrument_id", "timeframe", "as_of"],
    )

    # 6. backtests
    op.create_table(
        "backtests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column(
            "strategy_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "instrument_scope",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mode", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("equity_curve_uri", sa.Text(), nullable=True),
        sa.Column("report_uri", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "mode IN ('static_model','walk_forward_retrain')",
            name="ck_backtests_mode",
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','done','failed','cancelled')",
            name="ck_backtests_status",
        ),
    )
    op.create_index("idx_backtests_user_created", "backtests", ["user_id", "created_at"])

    # 7. backtest_folds
    op.create_table(
        "backtest_folds",
        sa.Column(
            "backtest_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("backtests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("fold_no", sa.Integer(), nullable=False),
        sa.Column("train_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("train_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("val_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("val_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("test_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "model_version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("model_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.PrimaryKeyConstraint("backtest_id", "fold_no", name="pk_backtest_folds"),
    )

    # 8. backtest_trades
    op.create_table(
        "backtest_trades",
        sa.Column(
            "backtest_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("backtests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("entry_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("exit_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("fee", sa.Float(), nullable=False),
        sa.Column("pnl", sa.Float(), nullable=False),
        sa.Column("exit_reason", sa.String(length=30), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("fold_no", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("backtest_id", "seq", name="pk_backtest_trades"),
        sa.CheckConstraint("side IN ('LONG','SHORT')", name="ck_backtest_trades_side"),
    )
    op.create_index(
        "idx_backtest_trades_inst_entry",
        "backtest_trades",
        ["backtest_id", "instrument_id", "entry_at"],
    )

    # 9. jobs
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('backtest','optimize','report','retrain','inference')",
            name="ck_jobs_kind",
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','done','failed','cancelled')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint(
            "progress >= 0.0 AND progress <= 1.0",
            name="ck_jobs_progress_range",
        ),
    )
    op.create_index("idx_jobs_user_created", "jobs", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("backtest_trades")
    op.drop_table("backtest_folds")
    op.drop_table("backtests")
    op.drop_table("predictions")
    op.drop_table("model_versions")
    op.drop_table("labels")
    op.drop_table("features")
    op.drop_table("feature_sets")
