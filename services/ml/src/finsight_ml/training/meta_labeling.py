from dataclasses import dataclass, field
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from finsight_ml.backtest.benchmarks import run_buy_and_hold
from finsight_ml.backtest.engine import BacktestEngine, BacktestResult
from finsight_ml.features.context_v2 import fit_transform_causal_hmm_fold
from finsight_ml.labels.triple_barrier import (
    compute_ewm_volatility,
    compute_triple_barrier_dataset,
    evaluate_barrier_path,
)
from finsight_ml.models.calibration import ProbabilityCalibrator
from finsight_ml.models.lgbm import LGBMForecaster
from finsight_ml.models.quantile import (
    VolatilityLGBMForecaster,
    compute_forward_realized_volatility,
)
from finsight_ml.validation.metrics import (
    apply_signal_decision_rule,
    compute_classification_metrics,
    tune_signal_thresholds_on_val,
)
from finsight_ml.validation.walk_forward import purged_walk_forward_splits


@dataclass(frozen=True)
class MetaLabelConfig:
    """Configuration for 2-stage López de Prado Meta-Labeling + Volatility Sizing."""

    horizon_bars: int = 24
    pt_mult: float = 1.0
    sl_mult: float = 1.0
    min_barrier_pct: float = 0.003
    embargo_bars: int = 24
    min_train_months: int = 10
    val_months: int = 2
    test_months: int = 2
    step_months: int = 2
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    allow_short: bool = False
    use_hmm: bool = True
    use_vol_sizing: bool = True
    min_position_size: float = 0.25
    max_position_size: float = 1.0
    inner_oof_folds: int = 3
    meta_threshold_grid: tuple[float, ...] = (0.48, 0.50, 0.52, 0.55, 0.58, 0.62)
    min_val_trades: int = 8


@dataclass
class WalkForwardComparisonResult:
    symbol: str
    timeframe: str
    oos_bars: int
    n_folds: int
    primary_metrics: dict[str, Any]
    meta_metrics: dict[str, Any]
    primary_backtest: BacktestResult
    meta_backtest: BacktestResult
    meta_vol_backtest: BacktestResult
    buy_hold_return: float
    buy_hold_sharpe: float
    fold_details: list[dict[str, Any]] = field(default_factory=list)


def compute_net_trade_outcomes(
    df: pd.DataFrame,
    sides: np.ndarray,
    vol_series: pd.Series,
    *,
    horizon_bars: int,
    pt_mult: float,
    sl_mult: float,
    min_barrier_pct: float,
    round_trip_cost_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate exact next-open triple-barrier net return for each proposed directional `side in {-1, 0, +1}`.
    Returns `(net_returns, binary_profitable_mask)` aligned to `df.index`.
    """
    n = len(df)
    open_arr = df["open"].astype("float64").to_numpy()
    high_arr = df["high"].astype("float64").to_numpy()
    low_arr = df["low"].astype("float64").to_numpy()
    close_arr = df["close"].astype("float64").to_numpy()
    sigma_arr = vol_series.reindex(df.index).astype("float64").to_numpy()

    net_rets = np.full(n, np.nan, dtype=np.float64)
    y_bin = np.full(n, np.nan, dtype=np.float64)
    h_scale = float(np.sqrt(horizon_bars))

    for j in range(n - 1):
        side = int(sides[j])
        if side not in (-1, 1):
            continue
        vol = float(sigma_arr[j])
        entry_pos = j + 1
        entry_price = float(open_arr[entry_pos])
        if (
            np.isnan(vol)
            or entry_price <= 0.0
            or max(pt_mult, sl_mult) * vol * h_scale < min_barrier_pct
        ):
            continue

        if side == 1:
            b_up = entry_price * float(np.exp(pt_mult * vol * h_scale))
            b_down = entry_price * float(np.exp(-sl_mult * vol * h_scale))
            c_side: Literal["LONG", "SHORT"] = "LONG"
        else:
            b_up = entry_price * float(np.exp(sl_mult * vol * h_scale))
            b_down = entry_price * float(np.exp(-pt_mult * vol * h_scale))
            c_side = "SHORT"

        out = evaluate_barrier_path(
            entry_price=entry_price,
            barrier_up=b_up,
            barrier_down=b_down,
            open_arr=open_arr,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
            entry_pos=entry_pos,
            horizon_bars=horizon_bars,
            check_gap_open=True,
            conservative_side=c_side,
        )
        gross = float(side) * (out.exit_price - entry_price) / entry_price
        net = gross - round_trip_cost_pct
        net_rets[j] = net
        y_bin[j] = 1.0 if net > 0.0 else 0.0

    return net_rets, y_bin


def _build_meta_feature_frame(
    X_base: pd.DataFrame,
    primary_probs: np.ndarray,
    proposed_side: np.ndarray,
    ewm_vol: pd.Series,
    pred_vol: np.ndarray | None = None,
    hmm_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble meta-classifier input matrix strictly from causal bar-t values."""
    meta = X_base.copy()
    meta["primary_p_sell"] = primary_probs[:, 0]
    meta["primary_p_hold"] = primary_probs[:, 1]
    meta["primary_p_buy"] = primary_probs[:, 2]
    meta["primary_dir_margin"] = primary_probs[:, 2] - primary_probs[:, 0]
    meta["primary_abs_margin"] = np.abs(primary_probs[:, 2] - primary_probs[:, 0])
    meta["primary_max_dir_p"] = np.maximum(primary_probs[:, 0], primary_probs[:, 2])
    meta["proposed_side"] = proposed_side.astype(np.float64)

    ewm = ewm_vol.reindex(X_base.index).astype("float64").to_numpy()
    if pred_vol is not None:
        meta["pred_vol_24"] = pred_vol
        meta["pred_vol_ratio"] = np.where(ewm > 1e-12, pred_vol / np.maximum(ewm, 1e-12), 1.0)
    if hmm_df is not None:
        aligned = hmm_df.reindex(X_base.index)
        for col in aligned.columns:
            meta[col] = aligned[col].to_numpy(dtype=np.float64)
    return meta


def _train_oof_primary_probs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray | None,
    *,
    n_splits: int,
    embargo_bars: int,
    lgbm_params: dict[str, Any],
) -> np.ndarray:
    """
    Generate strictly purged out-of-fold primary probability predictions inside `X_train`
    so the Stage 2 meta-model never trains on in-sample overfit probabilities.
    """
    n = len(X_train)
    oof_probs = np.full((n, 3), np.nan, dtype=np.float64)
    block_size = n // (n_splits + 1)
    if block_size < 120:
        return oof_probs

    for k in range(1, n_splits + 1):
        val_start = k * block_size
        val_end = (k + 1) * block_size if k < n_splits else n
        tr_end = max(0, val_start - embargo_bars)
        if tr_end < 120 or (val_end - val_start) < 40:
            continue

        X_tr = X_train.iloc[:tr_end]
        y_tr = y_train.iloc[:tr_end]
        w_tr = w_train[:tr_end] if w_train is not None else None
        X_va = X_train.iloc[val_start:val_end]
        y_va = y_train.iloc[val_start:val_end]

        if y_tr.dropna().nunique() < 2:
            continue

        model = LGBMForecaster(**lgbm_params)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=w_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=40,
        )
        raw = model.predict_proba(X_va)
        cal = ProbabilityCalibrator()
        cal.fit(raw, y_va.to_numpy(dtype=np.int64))
        oof_probs[val_start:val_end] = cal.transform(raw)

    return oof_probs


def _propose_side_from_probs(probs: np.ndarray, *, allow_short: bool) -> np.ndarray:
    """Propose directional side from calibrated probs: +1 if P(BUY) > P(SELL) (else -1 if allow_short)."""
    p_sell = probs[:, 0]
    p_hold = probs[:, 1]
    p_buy = probs[:, 2]
    sides = np.zeros(len(probs), dtype=np.int64)
    if allow_short:
        buy_mask = (p_buy >= p_sell) & (p_buy > p_hold * 0.5)
        sell_mask = (p_sell > p_buy) & (p_sell > p_hold * 0.5)
        sides[buy_mask] = 1
        sides[sell_mask] = -1
    else:
        buy_mask = (p_buy > p_sell) & (p_buy >= 0.25)
        sides[buy_mask] = 1
    return sides


def run_v1_vs_v2_meta_walk_forward(
    df: pd.DataFrame,
    X_features: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str = "1h",
    cfg: MetaLabelConfig | None = None,
    primary_params: dict[str, Any] | None = None,
) -> WalkForwardComparisonResult:
    """
    Strictly out-of-sample walk-forward comparison between:
    1) Primary LGBM v1-style directional classifier (tuned on val)
    2) Stage-2 Meta-Labeled LGBM filtering net-negative / cost-dominated signals
    3) Stage-2 Meta-Labeled LGBM + causal VolatilityLGBM position sizing
    """
    c = cfg or MetaLabelConfig(embargo_bars=24 if timeframe == "1h" else 6)
    lgbm_hp: dict[str, Any] = primary_params or {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "class_weight": "balanced",
        "random_state": 42,
    }

    work_df = df.reset_index(drop=True)
    X_work = X_features.reset_index(drop=True)
    timestamps = pd.DatetimeIndex(pd.to_datetime(work_df["open_time"], utc=True))

    tb_input = work_df.copy()
    tb_input.index = timestamps
    tb_df = compute_triple_barrier_dataset(
        tb_input,
        horizon_bars=c.horizon_bars,
        pt_mult=c.pt_mult,
        sl_mult=c.sl_mult,
        min_barrier_pct=c.min_barrier_pct,
        entry_on_next_open=True,
    ).reset_index(drop=True)
    weights_all = tb_df["sample_weight"].to_numpy(dtype=np.float64)
    ewm_vol = compute_ewm_volatility(work_df["close"], span=24, min_periods=24)
    fwd_vol = compute_forward_realized_volatility(work_df["close"], horizon_bars=c.horizon_bars)

    valid_mask = (
        X_work.notna().all(axis=1) & tb_df["label"].notna() & ewm_vol.notna() & tb_df["t1"].notna()
    )

    folds = purged_walk_forward_splits(
        timestamps=timestamps,
        t1=tb_df["t1"],
        valid_mask=valid_mask,
        min_train_months=c.min_train_months,
        val_months=c.val_months,
        test_months=c.test_months,
        step_months=c.step_months,
        embargo_bars=c.embargo_bars,
    )
    if not folds:
        raise ValueError("Insufficient history to generate walk-forward folds.")

    oos_positions: list[int] = []
    oos_y_true: list[np.ndarray] = []
    oos_prim_probs: list[np.ndarray] = []
    oos_prim_sigs: list[np.ndarray] = []
    oos_meta_sigs: list[np.ndarray] = []
    oos_meta_confs: list[np.ndarray] = []
    oos_vol_sizes: list[np.ndarray] = []
    oos_fold_ids: list[np.ndarray] = []
    fold_details: list[dict[str, Any]] = []

    round_trip_cost = 2.0 * (c.fee_bps + c.slippage_bps) / 10_000.0
    exchange = "binance" if symbol.upper().endswith("USDT") else "nyse"

    for fold in folds:
        tr_pos = fold.train_indices[valid_mask.iloc[fold.train_indices].to_numpy()]
        va_pos = fold.val_indices[valid_mask.iloc[fold.val_indices].to_numpy()]
        te_pos = fold.test_indices[valid_mask.iloc[fold.test_indices].to_numpy()]
        if len(tr_pos) < 300 or len(va_pos) < 50 or len(te_pos) < 30:
            continue

        # Optional per-fold causal HMM fit ONLY on train close, forward-filtered on val/test
        hmm_tr: pd.DataFrame | None = None
        hmm_va: pd.DataFrame | None = None
        hmm_te: pd.DataFrame | None = None
        if c.use_hmm:
            tr_end_raw = int(tr_pos[-1])
            te_end_raw = int(te_pos[-1])
            hmm_tr_full, hmm_eval = fit_transform_causal_hmm_fold(
                work_df["close"].iloc[: tr_end_raw + 1],
                work_df["close"].iloc[tr_end_raw + 1 : te_end_raw + 1],
                vol_span=24,
                n_iter=25,
            )
            assert hmm_eval is not None
            hmm_tr = hmm_tr_full.reindex(tr_pos)
            hmm_va = hmm_eval.reindex(va_pos)
            hmm_te = hmm_eval.reindex(te_pos)

        X_tr_df = X_work.iloc[tr_pos].copy()
        X_va_df = X_work.iloc[va_pos].copy()
        X_te_df = X_work.iloc[te_pos].copy()
        if hmm_tr is not None and hmm_va is not None and hmm_te is not None:
            for col in hmm_tr.columns:
                X_tr_df[col] = hmm_tr[col].to_numpy(dtype=np.float64)
                X_va_df[col] = hmm_va[col].to_numpy(dtype=np.float64)
                X_te_df[col] = hmm_te[col].to_numpy(dtype=np.float64)

        y_tr = tb_df["label"].iloc[tr_pos].astype(np.int64)
        y_va = tb_df["label"].iloc[va_pos].astype(np.int64)
        y_te = tb_df["label"].iloc[te_pos].astype(np.int64)
        w_tr = weights_all[tr_pos]

        # Volatility forecaster strictly on train (early stop on val fwd_vol)
        vol_tr_mask = fwd_vol.iloc[tr_pos].notna().to_numpy()
        vol_va_mask = fwd_vol.iloc[va_pos].notna().to_numpy()
        vol_reg = VolatilityLGBMForecaster(random_state=42)
        vol_reg.fit(
            X_tr_df.iloc[vol_tr_mask],
            fwd_vol.iloc[tr_pos[vol_tr_mask]],
            eval_set=(X_va_df.iloc[vol_va_mask], fwd_vol.iloc[va_pos[vol_va_mask]])
            if vol_va_mask.sum() >= 20
            else None,
            early_stopping_rounds=40,
        )
        pred_vol_tr = vol_reg.predict_sigma(X_tr_df)
        pred_vol_va = vol_reg.predict_sigma(X_va_df)
        pred_vol_te = vol_reg.predict_sigma(X_te_df)

        # Primary Stage-1 model + OOF train probabilities for leak-free Meta training
        oof_tr_probs = _train_oof_primary_probs(
            X_tr_df,
            y_tr,
            w_tr,
            n_splits=c.inner_oof_folds,
            embargo_bars=c.embargo_bars,
            lgbm_params=lgbm_hp,
        )
        primary = LGBMForecaster(**lgbm_hp)
        primary.fit(
            X_tr_df,
            y_tr,
            sample_weight=w_tr,
            eval_set=(X_va_df, y_va),
            early_stopping_rounds=40,
        )
        cal = ProbabilityCalibrator()
        cal.fit(primary.predict_proba(X_va_df), y_va.to_numpy(dtype=np.int64))
        va_probs = cal.transform(primary.predict_proba(X_va_df))
        te_probs = cal.transform(primary.predict_proba(X_te_df))

        # Primary baseline decision rule tuned on val
        best_thr, best_margin = tune_signal_thresholds_on_val(
            va_probs,
            y_va.to_numpy(dtype=np.int64),
            min_coverage=0.10,
        )
        te_prim_sig = apply_signal_decision_rule(
            te_probs, entry_threshold=best_thr, min_margin=best_margin
        )
        if not c.allow_short:
            te_prim_sig = np.where(te_prim_sig == 1, 1, 0)

        # Stage-2 Meta-Labeling (only train on OOF train rows where OOF probs are finite)
        oof_finite = np.isfinite(oof_tr_probs).all(axis=1)
        oof_sides = np.zeros(len(tr_pos), dtype=np.int64)
        oof_sides[oof_finite] = _propose_side_from_probs(
            oof_tr_probs[oof_finite], allow_short=c.allow_short
        )

        # Compute exact net returns on full work_df for proposed train sides & val sides
        all_train_sides = np.zeros(len(work_df), dtype=np.int64)
        all_train_sides[tr_pos] = oof_sides
        va_sides = _propose_side_from_probs(va_probs, allow_short=c.allow_short)
        all_train_sides[va_pos] = va_sides
        te_sides = _propose_side_from_probs(te_probs, allow_short=c.allow_short)

        net_rets_all, y_meta_all = compute_net_trade_outcomes(
            work_df,
            all_train_sides,
            ewm_vol,
            horizon_bars=c.horizon_bars,
            pt_mult=c.pt_mult,
            sl_mult=c.sl_mult,
            min_barrier_pct=c.min_barrier_pct,
            round_trip_cost_pct=round_trip_cost,
        )

        tr_meta_idx = np.where(oof_finite & (oof_sides != 0) & np.isfinite(y_meta_all[tr_pos]))[0]
        va_meta_idx = np.where((va_sides != 0) & np.isfinite(y_meta_all[va_pos]))[0]

        X_meta_tr = _build_meta_feature_frame(
            X_tr_df.iloc[tr_meta_idx],
            oof_tr_probs[tr_meta_idx],
            oof_sides[tr_meta_idx],
            ewm_vol,
            pred_vol=pred_vol_tr[tr_meta_idx],
        )
        y_meta_tr = y_meta_all[tr_pos[tr_meta_idx]].astype(np.int64)

        X_meta_va_all = _build_meta_feature_frame(
            X_va_df, va_probs, va_sides, ewm_vol, pred_vol=pred_vol_va
        )
        X_meta_te_all = _build_meta_feature_frame(
            X_te_df, te_probs, te_sides, ewm_vol, pred_vol=pred_vol_te
        )

        if len(X_meta_tr) >= 60 and len(np.unique(y_meta_tr)) == 2:
            meta_clf = lgb.LGBMClassifier(
                n_estimators=180,
                learning_rate=0.03,
                max_depth=3,
                num_leaves=8,
                min_child_samples=35,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            meta_clf.fit(X_meta_tr, y_meta_tr)
            va_meta_p = np.asarray(meta_clf.predict_proba(X_meta_va_all), dtype=np.float64)[:, 1]
            te_meta_p = np.asarray(meta_clf.predict_proba(X_meta_te_all), dtype=np.float64)[:, 1]

            # Select meta threshold on val maximizing total validation trade net Sharpe (with floor)
            best_meta_thr = 0.52
            best_val_score = -1e9
            va_nets = np.nan_to_num(net_rets_all[va_pos], nan=0.0)
            for thr in c.meta_threshold_grid:
                taken = (va_sides != 0) & (va_meta_p >= thr)
                n_taken = int(taken.sum())
                if n_taken < min(c.min_val_trades, max(5, len(va_meta_idx) // 4)):
                    continue
                r_taken = va_nets[taken]
                mean_r = float(np.mean(r_taken))
                std_r = float(np.std(r_taken)) + 1e-6
                score = mean_r / std_r * np.sqrt(n_taken)
                if score > best_val_score:
                    best_val_score = score
                    best_meta_thr = thr
        else:
            best_meta_thr = 0.50
            te_meta_p = np.where(te_sides != 0, 0.55, 0.0)

        te_meta_sig = np.where((te_sides != 0) & (te_meta_p >= best_meta_thr), te_sides, 0)

        # Causal volatility-targeted sizing: reference target_vol strictly from train fold median
        target_vol = float(np.nanmedian(pred_vol_tr))
        raw_sizes = target_vol / np.maximum(pred_vol_te, 1e-6)
        te_vol_sizes = (
            np.clip(raw_sizes, c.min_position_size, c.max_position_size)
            if c.use_vol_sizing
            else np.ones(len(te_pos), dtype=np.float64)
        )

        fold_prim_acc = float(
            np.mean(
                compute_classification_metrics(y_te.to_numpy(dtype=np.int64), te_probs)[
                    "balanced_accuracy"
                ]
            )
        )
        fold_details.append(
            {
                "fold_number": fold.fold_no,
                "train_n": len(tr_pos),
                "val_n": len(va_pos),
                "test_n": len(te_pos),
                "primary_thr": round(best_thr, 3),
                "meta_thr": round(best_meta_thr, 3),
                "primary_bal_acc": round(fold_prim_acc, 4),
                "primary_trades_proposed": int((te_prim_sig != 0).sum()),
                "meta_signals": int((te_meta_sig != 0).sum()),
            }
        )

        oos_positions.extend(te_pos.tolist())
        oos_y_true.append(y_te.to_numpy(dtype=np.int64))
        oos_prim_probs.append(te_probs)
        oos_prim_sigs.append(te_prim_sig)
        oos_meta_sigs.append(te_meta_sig)
        oos_meta_confs.append(te_meta_p)
        oos_vol_sizes.append(te_vol_sizes)
        oos_fold_ids.append(np.full(len(te_pos), fold.fold_no, dtype=np.int64))

    if not oos_positions:
        raise ValueError("All walk-forward folds had insufficient valid rows.")

    # Stitch contiguous OOS test slice sorted chronologically without duplicates
    pos_arr = np.asarray(oos_positions, dtype=np.int64)
    _, uniq_idx = np.unique(pos_arr, return_index=True)
    uniq_idx = uniq_idx[np.argsort(pos_arr[uniq_idx])]
    sel_pos = pos_arr[uniq_idx]

    y_all_oos = np.concatenate(oos_y_true)[uniq_idx]
    probs_all_oos = np.vstack(oos_prim_probs)[uniq_idx]
    prim_sig_oos = np.concatenate(oos_prim_sigs)[uniq_idx]
    meta_sig_oos = np.concatenate(oos_meta_sigs)[uniq_idx]
    meta_conf_oos = np.concatenate(oos_meta_confs)[uniq_idx]
    vol_size_oos = np.concatenate(oos_vol_sizes)[uniq_idx]
    fold_ids_oos = np.concatenate(oos_fold_ids)[uniq_idx]

    oos_start = int(sel_pos[0])
    oos_end = int(sel_pos[-1])
    bt_df = work_df.iloc[oos_start : oos_end + 1].copy()
    bt_df.index = pd.to_datetime(bt_df["open_time"], utc=True)
    bt_vol = ewm_vol.iloc[oos_start : oos_end + 1].copy()
    bt_vol.index = bt_df.index

    s_prim = pd.Series(0, index=bt_df.index, dtype=np.int64)
    s_meta = pd.Series(0, index=bt_df.index, dtype=np.int64)
    s_conf = pd.Series(np.nan, index=bt_df.index, dtype=np.float64)
    s_size = pd.Series(1.0, index=bt_df.index, dtype=np.float64)
    s_fold = pd.Series(pd.NA, index=bt_df.index, dtype="Int64")

    rel = sel_pos - oos_start
    s_prim.iloc[rel] = prim_sig_oos
    s_meta.iloc[rel] = meta_sig_oos
    s_conf.iloc[rel] = meta_conf_oos
    s_size.iloc[rel] = vol_size_oos
    s_fold.iloc[rel] = fold_ids_oos

    engine = BacktestEngine(fee_bps=c.fee_bps, slippage_bps=c.slippage_bps)
    prim_bt = engine.run_barriers(
        bt_df,
        s_prim,
        horizon_bars=c.horizon_bars,
        pt_mult=c.pt_mult,
        sl_mult=c.sl_mult,
        volatility=bt_vol,
        min_barrier_pct=c.min_barrier_pct,
        position_size=1.0,
        allow_short=c.allow_short,
        exchange=exchange,
        timeframe=timeframe,
        fold_map=s_fold,
    )
    meta_bt = engine.run_barriers(
        bt_df,
        s_meta,
        horizon_bars=c.horizon_bars,
        pt_mult=c.pt_mult,
        sl_mult=c.sl_mult,
        volatility=bt_vol,
        min_barrier_pct=c.min_barrier_pct,
        position_size=1.0,
        allow_short=c.allow_short,
        exchange=exchange,
        timeframe=timeframe,
        confidence=s_conf,
        fold_map=s_fold,
    )
    meta_vol_bt = engine.run_barriers(
        bt_df,
        s_meta,
        horizon_bars=c.horizon_bars,
        pt_mult=c.pt_mult,
        sl_mult=c.sl_mult,
        volatility=bt_vol,
        min_barrier_pct=c.min_barrier_pct,
        position_size=1.0,
        position_sizes=s_size,
        allow_short=c.allow_short,
        exchange=exchange,
        timeframe=timeframe,
        confidence=s_conf,
        fold_map=s_fold,
    )

    # Buy & hold reference over exact same bt_df span using proper timeframe annualization
    bh_bt = run_buy_and_hold(
        bt_df,
        fee_bps=c.fee_bps,
        slippage_bps=c.slippage_bps,
        exchange=exchange,
        timeframe=timeframe,
    )

    prim_cls_metrics = compute_classification_metrics(y_all_oos, probs_all_oos)
    # Meta-filtered directional accuracy on active trades only
    meta_active = meta_sig_oos != 0
    if meta_active.sum() > 0:
        dir_hit = float(np.mean(meta_sig_oos[meta_active] == y_all_oos[meta_active]))
        meta_summary: dict[str, Any] = {
            "active_signals": int(meta_active.sum()),
            "signal_coverage": float(np.mean(meta_active)),
            "exact_barrier_hit_rate": dir_hit,
            "mean_meta_conf": float(np.mean(meta_conf_oos[meta_active])),
        }
    else:
        meta_summary = {
            "active_signals": 0,
            "signal_coverage": 0.0,
            "exact_barrier_hit_rate": 0.0,
            "mean_meta_conf": 0.0,
        }

    return WalkForwardComparisonResult(
        symbol=symbol,
        timeframe=timeframe,
        oos_bars=len(sel_pos),
        n_folds=len(fold_details),
        primary_metrics=prim_cls_metrics,
        meta_metrics=meta_summary,
        primary_backtest=prim_bt,
        meta_backtest=meta_bt,
        meta_vol_backtest=meta_vol_bt,
        buy_hold_return=bh_bt.total_return,
        buy_hold_sharpe=bh_bt.sharpe,
        fold_details=fold_details,
    )
