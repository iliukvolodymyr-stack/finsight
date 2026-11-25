import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from finsight_core.config import get_settings

from finsight_ml.features.pipeline import FeaturePipeline
from finsight_ml.labels.triple_barrier import (
    compute_ewm_volatility,
    compute_triple_barrier_dataset,
    summarize_label_distribution,
)
from finsight_ml.models.baselines import LogisticRegressionBaseline, MajorityBaseline
from finsight_ml.models.calibration import ProbabilityCalibrator
from finsight_ml.models.conformal import VolatilityNormalizedConformal
from finsight_ml.models.lgbm import LGBMForecaster
from finsight_ml.validation.metrics import (
    apply_signal_decision_rule,
    bootstrap_coverage_ci,
    compute_classification_metrics,
    tune_signal_thresholds_on_val,
)
from finsight_ml.validation.walk_forward import WalkForwardFold, purged_walk_forward_splits


@dataclass
class TrainingConfig:
    name: str = "lgbm_core"
    semver: str = "0.1.0"
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    horizon_bars: int = 24
    status_target: str = "production"
    pt_mult: float = 1.0
    sl_mult: float = 1.0
    min_barrier_pct: float = 0.003
    entry_on_next_open: bool = True
    min_train_months: int = 12
    val_months: int = 2
    test_months: int = 2
    step_months: int | None = None
    embargo_bars: int = 24
    early_stopping_rounds: int = 50
    min_signal_coverage: float = 0.10
    conformal_alpha: float = 0.20
    hyperparameters: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        labeling = raw.get("labeling", {})
        validation = raw.get("validation", {})
        default_hp = cls().hyperparameters
        hp = {**default_hp, **raw.get("hyperparameters", {})}
        return cls(
            name=str(raw.get("name", "lgbm_core")),
            semver=str(raw.get("version", "0.1.0")),
            symbol=str(raw["symbol"]).upper(),
            timeframe=str(raw.get("timeframe", "1h")),
            horizon_bars=int(raw.get("horizon_bars", labeling.get("horizon_bars", 24))),
            status_target=str(raw.get("status_target", "candidate")),
            pt_mult=float(labeling.get("pt_mult", 1.0)),
            sl_mult=float(labeling.get("sl_mult", 1.0)),
            min_barrier_pct=float(labeling.get("min_barrier_pct", 0.003)),
            entry_on_next_open=bool(labeling.get("entry_on_next_open", True)),
            min_train_months=int(validation.get("min_train_months", 12)),
            val_months=int(validation.get("val_months", 2)),
            test_months=int(validation.get("test_months", 2)),
            step_months=(
                int(validation["step_months"])
                if validation.get("step_months") is not None
                else None
            ),
            embargo_bars=int(validation.get("embargo_bars", 24)),
            early_stopping_rounds=int(raw.get("early_stopping_rounds", 50)),
            min_signal_coverage=float(raw.get("min_signal_coverage", 0.10)),
            conformal_alpha=float(raw.get("conformal_alpha", 0.20)),
            hyperparameters=hp,
        )


@dataclass
class TrainingResult:
    symbol: str
    timeframe: str
    model_name: str
    semver: str
    status_target: str
    artifact_dir: Path
    dataset_hash: str
    schema_hash: str
    train_data_end: datetime
    entry_threshold: float
    min_margin: float
    fold_reports: list[dict[str, Any]]
    oos_summary: dict[str, Any]
    acceptance_checks: dict[str, Any]


def _build_lgbm(cfg: TrainingConfig) -> LGBMForecaster:
    hp = cfg.hyperparameters
    return LGBMForecaster(
        n_estimators=int(hp.get("n_estimators", 300)),
        learning_rate=float(hp.get("learning_rate", 0.03)),
        max_depth=int(hp.get("max_depth", 5)),
        num_leaves=int(hp.get("num_leaves", 31)),
        min_child_samples=int(hp.get("min_child_samples", 40)),
        subsample=float(hp.get("subsample", 0.8)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.8)),
        reg_alpha=float(hp.get("reg_alpha", 0.1)),
        reg_lambda=float(hp.get("reg_lambda", 1.0)),
        class_weight=hp.get("class_weight", "balanced"),
        random_state=int(hp.get("random_state", 42)),
    )


def _compute_dataset_hash(
    symbol: str,
    timeframe: str,
    schema_hash: str,
    df_valid: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    manifest = {
        "symbol": symbol,
        "timeframe": timeframe,
        "schema_hash": schema_hash,
        "row_count": int(len(df_valid)),
        "start_ts": df_valid.index[0].isoformat(),
        "end_ts": df_valid.index[-1].isoformat(),
        "close_sum_rounded": round(float(df_valid["close"].sum()), 4),
    }
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), manifest


def _evaluate_fold(
    fold: WalkForwardFold,
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    close_arr: np.ndarray,
    future_close_arr: np.ndarray,
    fwd_logret_arr: np.ndarray,
    sigma_arr: np.ndarray,
    cfg: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tr_idx = fold.train_indices
    va_idx = fold.val_indices
    te_idx = fold.test_indices

    X_tr, y_tr, w_tr = X.iloc[tr_idx], y.iloc[tr_idx], weights[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
    y_va_np = y_va.to_numpy(dtype=np.int64)
    y_te_np = y_te.to_numpy(dtype=np.int64)

    # 1. Level-0 Baselines fit strictly on train
    maj = MajorityBaseline().fit(X_tr, y_tr, sample_weight=w_tr)
    maj_probs_te = maj.predict_proba(X_te)

    logreg = LogisticRegressionBaseline(
        random_state=int(cfg.hyperparameters.get("random_state", 42))
    ).fit(X_tr, y_tr, sample_weight=w_tr)
    logreg_probs_te = logreg.predict_proba(X_te)

    # 2. LightGBM fit strictly on train (early stopping on val)
    lgbm = _build_lgbm(cfg).fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    raw_probs_va = lgbm.predict_proba(X_va)
    raw_probs_te = lgbm.predict_proba(X_te)

    # 3. Calibration + threshold selection strictly on val
    calibrator = ProbabilityCalibrator().fit(raw_probs_va, y_va_np)
    cal_probs_va = calibrator.transform(raw_probs_va)
    cal_probs_te = calibrator.transform(raw_probs_te)

    entry_thr, min_margin = tune_signal_thresholds_on_val(
        cal_probs_va,
        y_va_np,
        min_coverage=cfg.min_signal_coverage,
    )

    # 4. Volatility-normalized Conformal corridor fit strictly on val
    conformal = VolatilityNormalizedConformal(
        alpha=cfg.conformal_alpha,
        horizon_bars=cfg.horizon_bars,
    ).fit(fwd_logret_arr[va_idx], sigma_arr[va_idx])
    conf_eval_te = conformal.evaluate_coverage(
        close_prices=close_arr[te_idx],
        future_close_prices=future_close_arr[te_idx],
        sigma=sigma_arr[te_idx],
    )

    maj_metrics = compute_classification_metrics(
        y_te_np, maj_probs_te, entry_threshold=0.0, min_margin=0.0
    )
    logreg_metrics = compute_classification_metrics(
        y_te_np, logreg_probs_te, entry_threshold=0.45, min_margin=0.05
    )
    lgbm_raw_metrics = compute_classification_metrics(
        y_te_np, raw_probs_te, entry_threshold=entry_thr, min_margin=min_margin
    )
    lgbm_cal_metrics = compute_classification_metrics(
        y_te_np, cal_probs_te, entry_threshold=entry_thr, min_margin=min_margin
    )

    fold_report: dict[str, Any] = {
        "fold_no": fold.fold_no,
        "train_start": fold.train_start.isoformat(),
        "train_end": fold.train_end.isoformat(),
        "val_start": fold.val_start.isoformat(),
        "val_end": fold.val_end.isoformat(),
        "test_start": fold.test_start.isoformat(),
        "test_end": fold.test_end.isoformat(),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "purged_from_train": fold.purged_from_train,
        "purged_from_val": fold.purged_from_val,
        "calibration_method": calibrator.method,
        "best_iteration": lgbm.best_iteration,
        "tuned_thresholds": {"entry_threshold": entry_thr, "min_margin": min_margin},
        "conformal": {
            "z_low": conformal.z_low,
            "z_high": conformal.z_high,
            "test_coverage": conf_eval_te["coverage"],
            "mean_rel_width": conf_eval_te["mean_rel_width"],
        },
        "models": {
            "majority": maj_metrics,
            "logreg": logreg_metrics,
            "lgbm_raw": lgbm_raw_metrics,
            "lgbm_calibrated": lgbm_cal_metrics,
        },
    }

    oos_arrays = {
        "y_true": y_te_np,
        "maj_probs": maj_probs_te,
        "logreg_probs": logreg_probs_te,
        "lgbm_raw_probs": raw_probs_te,
        "lgbm_cal_probs": cal_probs_te,
        "lgbm_signals": apply_signal_decision_rule(
            cal_probs_te, entry_threshold=entry_thr, min_margin=min_margin
        ),
        "conformal_covered": conf_eval_te["covered_mask"].astype(np.float64),
    }
    return fold_report, oos_arrays


def _render_model_card(result: TrainingResult, label_report: dict[str, Any]) -> str:
    oos_lgbm = result.oos_summary["models"]["lgbm_calibrated"]
    oos_logreg = result.oos_summary["models"]["logreg"]
    oos_maj = result.oos_summary["models"]["majority"]
    cov = result.oos_summary["conformal_coverage"]
    is_low_power = result.oos_summary["n_total_valid_samples"] < 5000

    low_power_note = (
        "\n> **LOW STATISTICAL POWER WARNING:** Dataset contains fewer than 5,000 labeled bars "
        "(e.g. SPY 1h with overnight gaps). Treat metrics as experimental candidate evidence only.\n"
        if is_low_power
        else ""
    )

    lines = [
        f"# Model Card — {result.model_name} ({result.symbol} {result.timeframe} v{result.semver})",
        "",
        f"- **Target Status:** `{result.status_target}`",
        f"- **Horizon:** `{result.oos_summary['horizon_bars']}` bars (`entry_on_next_open=True`)",
        f"- **Trained At:** {datetime.now(UTC).isoformat()}",
        f"- **Train Data End:** {result.train_data_end.isoformat()}",
        f"- **Dataset Hash:** `{result.dataset_hash}`",
        f"- **Feature Schema Hash:** `{result.schema_hash}`",
        low_power_note,
        "## Out-of-Sample Walk-Forward Summary",
        "",
        f"- Folds evaluated: **{len(result.fold_reports)}** (expanding purged walk-forward, embargo={result.oos_summary['embargo_bars']} bars)",
        f"- Total OOS test samples: **{result.oos_summary['n_oos_samples']}**",
        f"- Majority Baseline Macro-F1: **{oos_maj['macro_f1']:.4f}** (Acc={oos_maj['accuracy']:.4f})",
        f"- LogisticRegression Baseline Macro-F1: **{oos_logreg['macro_f1']:.4f}** (BalAcc={oos_logreg['balanced_accuracy']:.4f}, MCC={oos_logreg['mcc']:.4f})",
        f"- LightGBM Calibrated Macro-F1 (argmax): **{oos_lgbm['macro_f1']:.4f}** (Signal Macro-F1={oos_lgbm['signal_macro_f1']:.4f})",
        f"- LightGBM Calibrated BalAcc / MCC: **{oos_lgbm['balanced_accuracy']:.4f}** / **{oos_lgbm['mcc']:.4f}**",
        f"- Calibration (OOS test ECE / LogLoss / Brier): **{oos_lgbm['ece_10']:.4f}** / **{oos_lgbm['log_loss']:.4f}** / **{oos_lgbm['brier_score']:.4f}**",
        f"- Signal Coverage & Distribution: **{oos_lgbm['coverage'] * 100:.1f}%** active ({oos_lgbm['signal_share']})",
        f"- Volatility-Normalized Conformal 80% Coverage: **{cov['mean'] * 100:.1f}%** (95% block-bootstrap CI: [{cov['ci_low'] * 100:.1f}%, {cov['ci_high'] * 100:.1f}%])",
        "",
        "## Label Distribution (Full History)",
        f"- Valid labeled rows: {label_report['n_valid']} | BUY={label_report['buy_share'] * 100:.1f}%, HOLD={label_report['hold_share'] * 100:.1f}%, SELL={label_report['sell_share'] * 100:.1f}% | Mean uniqueness={label_report['mean_uniqueness']:.3f}",
        "",
        "## Methodological Caveats & Limitations",
        "- Early stopping, probability calibration, and signal thresholds are fit on validation folds; all classification and ECE metrics above are reported strictly on unseen test folds.",
        "- Native LightGBM TreeSHAP explanations (`pred_contrib=True`) reflect **raw model log-odds contributions** prior to probability calibration.",
        "- Conformal intervals normalize forward returns by `sigma_t * sqrt(H)`. Because 24h triple-barrier windows overlap, standard finite-sample exchangeability does not hold strictly; a moving-block bootstrap CI (block=24) is reported alongside mean coverage.",
        "- Educational coursework research artifact only; not financial or investment advice.",
    ]
    return "\n".join(lines) + "\n"


def train_and_evaluate(
    df: pd.DataFrame,
    cfg: TrainingConfig,
    artifacts_dir: Path | str | None = None,
) -> TrainingResult:
    """
    Execute full leakage-free Purged Walk-Forward training, calibration, conformal
    corridor fitting, TreeSHAP validation, and artifact persistence.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input OHLCV DataFrame must have a UTC DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    pipeline = FeaturePipeline()
    feats = pipeline.compute(df)
    sigma = compute_ewm_volatility(df["close"], span=24, min_periods=24)
    tb = compute_triple_barrier_dataset(
        df,
        horizon_bars=cfg.horizon_bars,
        pt_mult=cfg.pt_mult,
        sl_mult=cfg.sl_mult,
        volatility=sigma,
        min_barrier_pct=cfg.min_barrier_pct,
        entry_on_next_open=cfg.entry_on_next_open,
    )

    future_close = df["close"].astype("float64").shift(-cfg.horizon_bars)
    fwd_logret = np.log(future_close / df["close"].astype("float64"))

    valid_mask = (~feats.isna().any(axis=1)) & tb["label"].notna() & sigma.notna()
    valid_indices = np.flatnonzero(valid_mask.to_numpy(dtype=bool))
    if len(valid_indices) < 200:
        raise ValueError(
            f"Insufficient valid samples after warmup/labeling for {cfg.symbol} {cfg.timeframe}: {len(valid_indices)}"
        )

    df_valid = df.iloc[valid_indices]
    X_valid = feats.iloc[valid_indices]
    y_valid = tb["label"].iloc[valid_indices].astype(int)
    t1_valid = tb["t1"].iloc[valid_indices]
    weights_valid = tb["sample_weight"].iloc[valid_indices].to_numpy(dtype=np.float64)
    sigma_valid = sigma.iloc[valid_indices].to_numpy(dtype=np.float64)
    close_valid = df_valid["close"].to_numpy(dtype=np.float64)
    future_close_valid = future_close.iloc[valid_indices].to_numpy(dtype=np.float64)
    fwd_logret_valid = fwd_logret.iloc[valid_indices].to_numpy(dtype=np.float64)

    dataset_hash, dataset_manifest = _compute_dataset_hash(
        cfg.symbol, cfg.timeframe, pipeline.schema_hash, df_valid
    )
    label_rep = asdict(summarize_label_distribution(tb.iloc[valid_indices]))

    folds = purged_walk_forward_splits(
        timestamps=df_valid.index,
        t1=t1_valid,
        min_train_months=cfg.min_train_months,
        val_months=cfg.val_months,
        test_months=cfg.test_months,
        step_months=cfg.step_months,
        embargo_bars=cfg.embargo_bars,
    )
    if not folds:
        raise ValueError(
            f"Purged walk-forward produced 0 valid folds for {cfg.symbol} {cfg.timeframe}."
        )

    fold_reports: list[dict[str, Any]] = []
    oos_buckets: dict[str, list[np.ndarray]] = {
        "y_true": [],
        "maj_probs": [],
        "logreg_probs": [],
        "lgbm_raw_probs": [],
        "lgbm_cal_probs": [],
        "lgbm_signals": [],
        "conformal_covered": [],
    }

    for fold in folds:
        rep, arrays = _evaluate_fold(
            fold=fold,
            X=X_valid,
            y=y_valid,
            weights=weights_valid,
            close_arr=close_valid,
            future_close_arr=future_close_valid,
            fwd_logret_arr=fwd_logret_valid,
            sigma_arr=sigma_valid,
            cfg=cfg,
        )
        fold_reports.append(rep)
        for k, arr in arrays.items():
            oos_buckets[k].append(arr)

    agg_y = np.concatenate(oos_buckets["y_true"])
    agg_maj = np.vstack(oos_buckets["maj_probs"])
    agg_logreg = np.vstack(oos_buckets["logreg_probs"])
    agg_raw = np.vstack(oos_buckets["lgbm_raw_probs"])
    agg_cal = np.vstack(oos_buckets["lgbm_cal_probs"])
    agg_sigs = np.concatenate(oos_buckets["lgbm_signals"])
    agg_cov = np.concatenate(oos_buckets["conformal_covered"])

    median_thr = float(np.median([f["tuned_thresholds"]["entry_threshold"] for f in fold_reports]))
    median_margin = float(np.median([f["tuned_thresholds"]["min_margin"] for f in fold_reports]))

    oos_cal_metrics = compute_classification_metrics(
        agg_y, agg_cal, entry_threshold=median_thr, min_margin=median_margin
    )
    # Override signal_macro_f1 & signal_share with actual fold-by-fold val-tuned OOS signals
    oos_cal_metrics["coverage"] = float(np.mean(agg_sigs != 0))
    oos_cal_metrics["signal_share"] = {
        "BUY": float(np.mean(agg_sigs == 1)),
        "HOLD": float(np.mean(agg_sigs == 0)),
        "SELL": float(np.mean(agg_sigs == -1)),
    }

    cov_ci = bootstrap_coverage_ci(agg_cov, block_size=cfg.horizon_bars, n_resamples=1000, seed=42)

    oos_maj_metrics = compute_classification_metrics(
        agg_y, agg_maj, entry_threshold=0.0, min_margin=0.0
    )
    oos_logreg_metrics = compute_classification_metrics(
        agg_y, agg_logreg, entry_threshold=0.45, min_margin=0.05
    )
    oos_raw_metrics = compute_classification_metrics(
        agg_y, agg_raw, entry_threshold=median_thr, min_margin=median_margin
    )

    oos_summary: dict[str, Any] = {
        "n_total_valid_samples": int(len(df_valid)),
        "n_oos_samples": int(len(agg_y)),
        "n_folds": int(len(fold_reports)),
        "horizon_bars": cfg.horizon_bars,
        "embargo_bars": cfg.embargo_bars,
        "conformal_coverage": cov_ci,
        "models": {
            "majority": oos_maj_metrics,
            "logreg": oos_logreg_metrics,
            "lgbm_raw": oos_raw_metrics,
            "lgbm_calibrated": oos_cal_metrics,
        },
    }

    # Check Acceptance Criteria A2, A4, A5 (+ leakage guard)
    folds_beating_maj = sum(
        1
        for f in fold_reports
        if f["models"]["lgbm_calibrated"]["macro_f1"] >= f["models"]["majority"]["macro_f1"] + 0.05
    )
    acceptance_checks = {
        "A1_leakage_guard_ok": not oos_cal_metrics["leakage_warning"],
        "A2_beats_majority_by_005_folds": f"{folds_beating_maj}/{len(fold_reports)}",
        "A4_ece_le_005": oos_cal_metrics["ece_10"] <= 0.05,
        "A5_conformal_cov_075_085": 0.75 <= cov_ci["mean"] <= 0.85,
    }

    # Refit final deployable model on full labeled history up to train_data_end
    # Split chronologically: last 15% (at least 150 samples) for purged out-of-sample calibration & conformal
    n_all = len(df_valid)
    cal_len = max(150, int(0.15 * n_all))
    cal_start_pos = max(100, n_all - cal_len)
    cal_start_ts = df_valid.index[cal_start_pos]
    t1_ts = pd.DatetimeIndex(t1_valid.to_numpy())
    if t1_ts.tz is None:
        t1_ts = t1_ts.tz_localize("UTC")

    pos_all = np.arange(n_all, dtype=np.int64)
    final_train_mask = (t1_ts < cal_start_ts) & (pos_all <= cal_start_pos - cfg.embargo_bars - 1)
    final_tr_idx = np.flatnonzero(final_train_mask)
    final_cal_idx = np.arange(cal_start_pos, n_all, dtype=np.int64)

    final_lgbm = _build_lgbm(cfg).fit(
        X_valid.iloc[final_tr_idx],
        y_valid.iloc[final_tr_idx],
        sample_weight=weights_valid[final_tr_idx],
        eval_set=(X_valid.iloc[final_cal_idx], y_valid.iloc[final_cal_idx]),
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    final_cal_raw = final_lgbm.predict_proba(X_valid.iloc[final_cal_idx])
    final_cal_y = y_valid.iloc[final_cal_idx].to_numpy(dtype=np.int64)
    final_calibrator = ProbabilityCalibrator().fit(final_cal_raw, final_cal_y)
    final_cal_probs = final_calibrator.transform(final_cal_raw)
    final_thr, final_margin = tune_signal_thresholds_on_val(
        final_cal_probs, final_cal_y, min_coverage=cfg.min_signal_coverage
    )
    final_conformal = VolatilityNormalizedConformal(
        alpha=cfg.conformal_alpha,
        horizon_bars=cfg.horizon_bars,
    ).fit(fwd_logret_valid[final_cal_idx], sigma_valid[final_cal_idx])

    base_dir = Path(artifacts_dir or get_settings().ARTIFACTS_DIR)
    model_slug = f"{cfg.name}_{cfg.symbol.lower()}_{cfg.timeframe}"
    artifact_dir = base_dir / "models" / model_slug / cfg.semver
    artifact_dir.mkdir(parents=True, exist_ok=True)

    final_lgbm.save(artifact_dir / "model.joblib")
    final_calibrator.save(artifact_dir / "calibrator.joblib")
    final_conformal.save(artifact_dir / "conformal.json")
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(pipeline.to_schema_dict(), indent=2), encoding="utf-8"
    )
    (artifact_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_hash": dataset_hash,
                **dataset_manifest,
                "label_distribution": label_rep,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "config.yaml").write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8"
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(
            {
                "oos_summary": oos_summary,
                "folds": fold_reports,
                "acceptance_checks": acceptance_checks,
                "deployable_thresholds": {
                    "entry_threshold": final_thr,
                    "min_margin": final_margin,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_data_end = df_valid.index[-1].to_pydatetime()
    result = TrainingResult(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        model_name=model_slug,
        semver=cfg.semver,
        status_target=cfg.status_target,
        artifact_dir=artifact_dir,
        dataset_hash=dataset_hash,
        schema_hash=pipeline.schema_hash,
        train_data_end=train_data_end,
        entry_threshold=final_thr,
        min_margin=final_margin,
        fold_reports=fold_reports,
        oos_summary=oos_summary,
        acceptance_checks=acceptance_checks,
    )
    (artifact_dir / "model_card.md").write_text(
        _render_model_card(result, label_rep), encoding="utf-8"
    )
    return result
