from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

# Canonical 3-class contract across all models:
# col 0 -> SELL (-1), col 1 -> HOLD (0), col 2 -> BUY (+1)
CANONICAL_LABELS: tuple[int, int, int] = (-1, 0, 1)
LABEL_NAMES: dict[int, str] = {-1: "SELL", 0: "HOLD", 1: "BUY"}


def probs_to_argmax_labels(probs: np.ndarray) -> np.ndarray:
    """Map [P(SELL), P(HOLD), P(BUY)] array (N, 3) to {-1, 0, 1} argmax predictions."""
    col_idx = np.argmax(probs, axis=1)
    return (col_idx - 1).astype(np.int64)


def apply_signal_decision_rule(
    probs: np.ndarray,
    entry_threshold: float = 0.55,
    min_margin: float = 0.10,
) -> np.ndarray:
    """
    Convert calibrated probabilities [P(SELL), P(HOLD), P(BUY)] to {-1, 0, 1} trading signals.

    Rule (docs/04-ml-pipeline.md §6.4):
    - Take argmax class.
    - Force HOLD (0) if max(p) < entry_threshold OR |p_buy - p_sell| < min_margin.
    """
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Expected probs shape (N, 3), got {probs.shape}")

    raw_preds = probs_to_argmax_labels(probs)
    max_p = np.max(probs, axis=1)
    directional_margin = np.abs(probs[:, 2] - probs[:, 0])

    active_mask = (raw_preds != 0) & (max_p >= entry_threshold) & (directional_margin >= min_margin)
    signals = np.zeros(len(probs), dtype=np.int64)
    signals[active_mask] = raw_preds[active_mask]
    return signals


def compute_ece(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Top-label Expected Calibration Error (ECE) over `n_bins` confidence buckets.
    Directly measures how closely UI `confidence = max(p)` matches realized accuracy.
    """
    if len(y_true) == 0:
        return 0.0

    y_pred = probs_to_argmax_labels(probs)
    conf = np.max(probs, axis=1)
    correct = (y_pred == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y_true))

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_acc = float(np.mean(correct[in_bin]))
            bin_conf = float(np.mean(conf[in_bin]))
            ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between one-hot class matrix and predicted 3-class probabilities."""
    if len(y_true) == 0:
        return 0.0
    y_idx = y_true.astype(np.int64) + 1
    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(len(y_true)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def tune_signal_thresholds_on_val(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    min_coverage: float = 0.10,
    threshold_grid: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60),
    margin_grid: tuple[float, ...] = (0.05, 0.10, 0.15),
) -> tuple[float, float]:
    """
    Select `(entry_threshold, min_margin)` strictly on validation fold by Macro-F1
    while maintaining at least `min_coverage` non-HOLD predictions.
    """
    best_thr = 0.55
    best_margin = 0.10
    best_score = -1.0

    for thr in threshold_grid:
        for margin in margin_grid:
            sigs = apply_signal_decision_rule(val_probs, entry_threshold=thr, min_margin=margin)
            coverage = float(np.mean(sigs != 0))
            if coverage < min_coverage:
                continue
            score = float(
                f1_score(
                    val_y, sigs, labels=list(CANONICAL_LABELS), average="macro", zero_division=0
                )
            )
            if score > best_score:
                best_score = score
                best_thr = float(thr)
                best_margin = float(margin)

    # Fallback: if no combo hits min_coverage, pick the loosest threshold/margin pair
    if best_score < 0.0:
        return float(min(threshold_grid)), float(min(margin_grid))
    return best_thr, best_margin


def compute_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    entry_threshold: float = 0.55,
    min_margin: float = 0.10,
) -> dict[str, Any]:
    """Full classification, probability calibration, and signal distribution metrics."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.asarray(probs, dtype=np.float64)
    if len(y_arr) == 0:
        return {"n_samples": 0}

    row_sums = np.sum(p_arr, axis=1, keepdims=True)
    p_norm = np.where(row_sums > 0.0, p_arr / np.maximum(row_sums, 1e-12), 1.0 / 3.0)
    p_clip = np.clip(p_norm, 1e-7, 1.0 - 1e-7)
    p_clip = p_clip / np.sum(p_clip, axis=1, keepdims=True)

    argmax_preds = probs_to_argmax_labels(p_norm)
    signals = apply_signal_decision_rule(
        p_norm, entry_threshold=entry_threshold, min_margin=min_margin
    )

    acc = float(accuracy_score(y_arr, argmax_preds))
    bal_acc = float(balanced_accuracy_score(y_arr, argmax_preds))
    macro_f1 = float(
        f1_score(
            y_arr, argmax_preds, labels=list(CANONICAL_LABELS), average="macro", zero_division=0
        )
    )
    signal_macro_f1 = float(
        f1_score(y_arr, signals, labels=list(CANONICAL_LABELS), average="macro", zero_division=0)
    )
    mcc = float(matthews_corrcoef(y_arr, argmax_preds))
    ll = float(log_loss(y_arr, p_clip, labels=list(CANONICAL_LABELS)))
    brier = compute_multiclass_brier(y_arr, p_norm)
    ece = compute_ece(y_arr, p_norm, n_bins=10)
    cm = confusion_matrix(y_arr, argmax_preds, labels=list(CANONICAL_LABELS)).tolist()

    prec, rec, f1_per_cls, sup = precision_recall_fscore_support(
        y_arr, argmax_preds, labels=list(CANONICAL_LABELS), zero_division=0
    )
    per_class = {
        LABEL_NAMES[lbl]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_per_cls[i]),
            "support": int(sup[i]),
        }
        for i, lbl in enumerate(CANONICAL_LABELS)
    }

    coverage = float(np.mean(signals != 0))
    signal_share = {
        "BUY": float(np.mean(signals == 1)),
        "HOLD": float(np.mean(signals == 0)),
        "SELL": float(np.mean(signals == -1)),
    }

    return {
        "n_samples": len(y_arr),
        "macro_f1": macro_f1,
        "signal_macro_f1": signal_macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "mcc": mcc,
        "log_loss": ll,
        "brier_score": brier,
        "ece_10": ece,
        "coverage": coverage,
        "signal_share": signal_share,
        "per_class": per_class,
        "confusion_matrix": cm,
        "entry_threshold": float(entry_threshold),
        "min_margin": float(min_margin),
        "leakage_warning": acc > 0.70,
    }


def bootstrap_coverage_ci(
    covered: np.ndarray,
    block_size: int = 24,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """
    Moving-block bootstrap confidence interval for conformal interval coverage.
    Accounts for overlapping H-bar horizon autocorrelation (non-exchangeability caveat).
    """
    arr = np.asarray(covered, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    mean_cov = float(np.mean(arr))
    b = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / b))
    max_start = max(1, n - b + 1)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        starts = rng.integers(0, max_start, size=n_blocks)
        sampled = np.concatenate([arr[s : s + b] for s in starts])[:n]
        boot_means[r] = float(np.mean(sampled))

    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {"mean": mean_cov, "ci_low": lo, "ci_high": hi}
