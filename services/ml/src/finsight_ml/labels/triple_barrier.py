from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarrierOutcome:
    label: float
    exit_pos: int
    exit_price: float
    exit_reason: str


@dataclass(frozen=True)
class LabelDistributionReport:
    n_total: int
    n_valid: int
    n_unlabeled: int
    buy_count: int
    hold_count: int
    sell_count: int
    buy_share: float
    hold_share: float
    sell_share: float
    exit_reason_counts: dict[str, int]
    mean_uniqueness: float
    hold_warning: bool


def compute_ewm_volatility(
    close: pd.Series,
    span: int = 24,
    min_periods: int | None = None,
) -> pd.Series:
    """Strictly backward EWMA standard deviation of 1-bar log returns."""
    mp = span if min_periods is None else min_periods
    log_ret = np.log(close.astype("float64") / close.astype("float64").shift(1))
    return log_ret.ewm(span=span, min_periods=mp, adjust=False).std()


def evaluate_barrier_path(
    entry_price: float,
    barrier_up: float,
    barrier_down: float,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    entry_pos: int,
    horizon_bars: int,
    check_gap_open: bool = True,
) -> BarrierOutcome:
    """
    Shared deterministic barrier evaluator used by both Triple Barrier labeling
    and BacktestEngine(exit_mode='label_barriers') so label exits and backtest
    exits match 1-to-1.

    Scans bars j = entry_pos .. entry_pos + horizon_bars - 1 (exact H bars):
    - On bars after entry (j > entry_pos), checks gap-open breaches first when
      `check_gap_open=True`.
    - Checks intrabar high/low touches with conservative SL-first tie-break if
      both barriers are hit within the same bar.
    - If neither barrier is touched within H bars, exits at vertical barrier
      on close[entry_pos + horizon_bars - 1].
    """
    end_pos = min(entry_pos + horizon_bars - 1, len(close_arr) - 1)

    for j in range(entry_pos, end_pos + 1):
        if check_gap_open and j > entry_pos:
            o_j = float(open_arr[j])
            if o_j <= barrier_down:
                return BarrierOutcome(-1.0, j, o_j, "sl")
            if o_j >= barrier_up:
                return BarrierOutcome(1.0, j, o_j, "pt")

        h_j = float(high_arr[j])
        l_j = float(low_arr[j])
        touch_up = h_j >= barrier_up
        touch_down = l_j <= barrier_down

        if touch_up and touch_down:
            return BarrierOutcome(-1.0, j, barrier_down, "sl")
        if touch_up:
            return BarrierOutcome(1.0, j, barrier_up, "pt")
        if touch_down:
            return BarrierOutcome(-1.0, j, barrier_down, "sl")

    return BarrierOutcome(0.0, end_pos, float(close_arr[end_pos]), "vertical")


def compute_sample_uniqueness(
    entry_positions: np.ndarray,
    exit_positions: np.ndarray,
    n_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lopez de Prado average sample uniqueness in O(N) via difference array + prefix sums.

    For each valid event i spanning [entry_pos_i, exit_pos_i]:
        c_t = number of active events overlapping bar t
        u_i = mean_{t in [entry_pos_i, exit_pos_i]} (1 / c_t) in (0, 1]
        sample_weight_i = u_i / mean(u)  (mean = 1.0 over valid samples)
    """
    n = len(entry_positions)
    uniqueness = np.full(n, np.nan, dtype=np.float64)
    weights = np.full(n, np.nan, dtype=np.float64)

    valid_mask = (~np.isnan(entry_positions)) & (~np.isnan(exit_positions))
    if not np.any(valid_mask) or n_bars <= 0:
        return uniqueness, weights

    diff = np.zeros(n_bars + 1, dtype=np.int64)
    valid_indices = np.flatnonzero(valid_mask)
    starts = entry_positions[valid_indices].astype(np.int64)
    ends = exit_positions[valid_indices].astype(np.int64)

    for s, e in zip(starts, ends, strict=True):
        s_clamped = max(0, min(int(s), n_bars - 1))
        e_clamped = max(s_clamped, min(int(e), n_bars - 1))
        diff[s_clamped] += 1
        diff[e_clamped + 1] -= 1

    concurrency = np.cumsum(diff[:-1])
    inv_conc = np.where(concurrency > 0, 1.0 / np.maximum(concurrency, 1), 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(inv_conc)))

    for idx, s, e in zip(valid_indices, starts, ends, strict=True):
        s_clamped = max(0, min(int(s), n_bars - 1))
        e_clamped = max(s_clamped, min(int(e), n_bars - 1))
        span_len = e_clamped - s_clamped + 1
        uniqueness[idx] = float((prefix[e_clamped + 1] - prefix[s_clamped]) / span_len)

    mean_u = float(np.nanmean(uniqueness))
    if mean_u > 0.0:
        weights[valid_indices] = uniqueness[valid_indices] / mean_u
    return uniqueness, weights


def compute_triple_barrier_dataset(
    df: pd.DataFrame,
    horizon_bars: int = 24,
    pt_mult: float = 1.25,
    sl_mult: float = 1.25,
    volatility: pd.Series | None = None,
    min_barrier_pct: float = 0.003,
    entry_on_next_open: bool = True,
    vol_span: int = 24,
) -> pd.DataFrame:
    """
    Compute full Triple Barrier labeling dataset aligned with next-open execution.

    Decision at close of bar t:
    - Volatility sigma_t is computed strictly using data <= t (zero look-ahead).
    - Entry occurs at open[t+1] when `entry_on_next_open=True` (100% aligned with
      BacktestEngine next-open execution) or close[t] when `entry_on_next_open=False`.
    - Barriers: entry * exp(+pt_mult * sigma_t * sqrt(H)) and entry * exp(-sl_mult * sigma_t * sqrt(H)).
    - Scanned across bars [t+1 .. t+H] (exact H holding bars).
    - Last `horizon_bars` rows remain unlabeled (NaN) to avoid tail truncation bias.
    """
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    if pt_mult <= 0 or sl_mult <= 0:
        raise ValueError("pt_mult and sl_mult must be > 0")

    required_cols = {"high", "low", "close"}
    if entry_on_next_open:
        required_cols.add("open")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

    n = len(df)
    close = df["close"].astype("float64")
    high_arr = df["high"].astype("float64").to_numpy()
    low_arr = df["low"].astype("float64").to_numpy()
    close_arr = close.to_numpy()
    open_arr = df["open"].astype("float64").to_numpy() if "open" in df.columns else close_arr

    if volatility is None:
        sigma_series = compute_ewm_volatility(close, span=vol_span, min_periods=vol_span)
    else:
        sigma_series = volatility.reindex(df.index).astype("float64")
    sigma_arr = sigma_series.to_numpy()

    labels = np.full(n, np.nan, dtype=np.float64)
    entry_prices = np.full(n, np.nan, dtype=np.float64)
    exit_prices = np.full(n, np.nan, dtype=np.float64)
    exit_reasons: list[str | None] = [None] * n
    barrier_ups = np.full(n, np.nan, dtype=np.float64)
    barrier_downs = np.full(n, np.nan, dtype=np.float64)
    entry_positions = np.full(n, np.nan, dtype=np.float64)
    exit_positions = np.full(n, np.nan, dtype=np.float64)
    t1_values: list[object] = [pd.NaT if isinstance(df.index, pd.DatetimeIndex) else None] * n

    horizon_scale = float(np.sqrt(horizon_bars))
    idx_values = df.index

    for i in range(max(0, n - horizon_bars)):
        entry_pos = i + 1
        entry_price = float(open_arr[entry_pos]) if entry_on_next_open else float(close_arr[i])
        vol = float(sigma_arr[i])
        if np.isnan(entry_price) or entry_price <= 0.0 or np.isnan(vol):
            continue

        entry_prices[i] = entry_price
        entry_positions[i] = float(entry_pos)

        upper_width = pt_mult * vol * horizon_scale
        lower_width = sl_mult * vol * horizon_scale

        if max(upper_width, lower_width) < min_barrier_pct:
            labels[i] = 0.0
            exit_prices[i] = entry_price
            exit_reasons[i] = "low_vol"
            exit_positions[i] = float(entry_pos)
            t1_values[i] = idx_values[entry_pos]
            continue

        b_up = entry_price * float(np.exp(upper_width))
        b_down = entry_price * float(np.exp(-lower_width))
        barrier_ups[i] = b_up
        barrier_downs[i] = b_down

        outcome = evaluate_barrier_path(
            entry_price=entry_price,
            barrier_up=b_up,
            barrier_down=b_down,
            open_arr=open_arr,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
            entry_pos=entry_pos,
            horizon_bars=horizon_bars,
            check_gap_open=entry_on_next_open,
        )
        labels[i] = outcome.label
        exit_prices[i] = outcome.exit_price
        exit_reasons[i] = outcome.exit_reason
        exit_positions[i] = float(outcome.exit_pos)
        t1_values[i] = idx_values[outcome.exit_pos]

    uniqueness, sample_weight = compute_sample_uniqueness(entry_positions, exit_positions, n)

    if isinstance(df.index, pd.DatetimeIndex):
        t1_idx = pd.DatetimeIndex(t1_values)
        if df.index.tz is not None and t1_idx.tz is None:
            t1_idx = t1_idx.tz_localize(df.index.tz)
        t1_col: pd.Series = pd.Series(t1_idx, index=df.index)
    else:
        t1_col = pd.Series(t1_values, index=df.index)

    return pd.DataFrame(
        {
            "label": labels,
            "entry_price": entry_prices,
            "exit_price": exit_prices,
            "exit_reason": pd.Series(exit_reasons, index=df.index, dtype="object"),
            "t1": t1_col,
            "entry_pos": entry_positions,
            "exit_pos": exit_positions,
            "barrier_up": barrier_ups,
            "barrier_down": barrier_downs,
            "sigma": sigma_arr,
            "uniqueness": uniqueness,
            "sample_weight": sample_weight,
        },
        index=df.index,
    )


def summarize_label_distribution(tb_df: pd.DataFrame) -> LabelDistributionReport:
    """Summarize Triple Barrier class balance, exit reasons, and mean uniqueness."""
    n_total = len(tb_df)
    valid = tb_df[tb_df["label"].notna()]
    n_valid = len(valid)
    n_unlabeled = n_total - n_valid

    buy_count = int((valid["label"] == 1.0).sum())
    hold_count = int((valid["label"] == 0.0).sum())
    sell_count = int((valid["label"] == -1.0).sum())

    denom = float(n_valid) if n_valid > 0 else 1.0
    buy_share = buy_count / denom
    hold_share = hold_count / denom
    sell_share = sell_count / denom

    reasons_series = valid["exit_reason"].value_counts()
    exit_reason_counts = {str(k): int(v) for k, v in reasons_series.items()}
    mean_u = float(valid["uniqueness"].mean()) if n_valid > 0 else 0.0

    return LabelDistributionReport(
        n_total=n_total,
        n_valid=n_valid,
        n_unlabeled=n_unlabeled,
        buy_count=buy_count,
        hold_count=hold_count,
        sell_count=sell_count,
        buy_share=buy_share,
        hold_share=hold_share,
        sell_share=sell_share,
        exit_reason_counts=exit_reason_counts,
        mean_uniqueness=mean_u,
        hold_warning=hold_share > 0.70,
    )


def apply_triple_barrier(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon_bars: int = 24,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    volatility: pd.Series | None = None,
    min_barrier_pct: float = 0.003,
    open_prices: pd.Series | None = None,
) -> pd.Series:
    """
    Convenience wrapper returning just the float64 label Series (-1.0, 0.0, 1.0, NaN).
    When `open_prices` is provided, aligns entry to open[t+1]; otherwise uses close[t].
    """
    data: dict[str, pd.Series] = {"high": high, "low": low, "close": close}
    entry_next_open = open_prices is not None
    if open_prices is not None:
        data["open"] = open_prices
    df = pd.DataFrame(data, index=close.index)

    if volatility is None:
        vol_span = 24
        min_p = min(2, horizon_bars) if len(close) < 60 else 24
        volatility = compute_ewm_volatility(close, span=vol_span, min_periods=min_p)

    res = compute_triple_barrier_dataset(
        df,
        horizon_bars=horizon_bars,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        volatility=volatility,
        min_barrier_pct=min_barrier_pct,
        entry_on_next_open=entry_next_open,
    )
    return res["label"]
