from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_no: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purged_from_train: int
    purged_from_val: int


def _to_utc_index(timestamps: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(timestamps)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def purged_walk_forward_splits(
    timestamps: pd.DatetimeIndex,
    t1: pd.Series,
    valid_mask: pd.Series | None = None,
    min_train_months: int = 12,
    val_months: int = 2,
    test_months: int = 2,
    step_months: int | None = None,
    embargo_bars: int = 24,
    min_train_samples: int = 100,
    min_val_samples: int = 20,
    min_test_samples: int = 20,
) -> list[WalkForwardFold]:
    """
    Expanding-window Purged Walk-Forward splitter with horizon purging and embargo.

    Rules (docs/04-ml-pipeline.md §5):
    - Strictly chronological: train < val < test (never shuffles).
    - Purges from train any observation whose label exit time t1 >= val_start
      (or whose positional index is within `embargo_bars` before val_start).
    - Purges from val any observation whose label exit time t1 >= test_start
      (or whose positional index is within `embargo_bars` before test_start).
    - Default `step_months = test_months` yields contiguous, non-overlapping OOS test folds.
    """
    if min_train_months < 1 or val_months < 1 or test_months < 1:
        raise ValueError("min_train_months, val_months, and test_months must be >= 1")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")

    step = test_months if step_months is None else step_months
    if step < 1:
        raise ValueError("step_months must be >= 1")

    ts = _to_utc_index(timestamps)
    t1_ts = _to_utc_index(t1.to_numpy())
    if len(ts) != len(t1_ts):
        raise ValueError("timestamps and t1 must have identical length")
    if len(ts) == 0:
        return []
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order")

    if valid_mask is None:
        base_valid = ~pd.isna(t1_ts)
    else:
        base_valid = valid_mask.to_numpy(dtype=bool) & (~pd.isna(t1_ts))

    t_min = ts[0]
    t_max = ts[-1]

    folds: list[WalkForwardFold] = []
    fold_no = 1
    offset_months = 0

    while True:
        train_end_bound = t_min + pd.DateOffset(months=min_train_months + offset_months)
        val_end_bound = train_end_bound + pd.DateOffset(months=val_months)
        test_end_bound = val_end_bound + pd.DateOffset(months=test_months)

        if val_end_bound > t_max:
            break

        # Allow the final fold if at least 50% of the nominal test window is present
        nominal_test_ns = (test_end_bound - val_end_bound).value
        available_test_ns = (t_max - val_end_bound).value
        if test_end_bound > t_max and available_test_ns < int(0.5 * nominal_test_ns):
            break

        effective_test_end = min(test_end_bound, t_max + pd.Timedelta(nanoseconds=1))

        raw_train_mask = base_valid & (ts < train_end_bound)
        raw_val_mask = base_valid & (ts >= train_end_bound) & (ts < val_end_bound)
        raw_test_mask = base_valid & (ts >= val_end_bound) & (ts < effective_test_end)

        if not np.any(raw_val_mask) or not np.any(raw_test_mask):
            offset_months += step
            continue

        first_val_pos = int(np.flatnonzero(ts >= train_end_bound)[0])
        first_test_pos = int(np.flatnonzero(ts >= val_end_bound)[0])

        positions = np.arange(len(ts), dtype=np.int64)
        purged_train_mask = (
            raw_train_mask
            & (t1_ts < train_end_bound)
            & (positions <= first_val_pos - embargo_bars - 1)
        )
        purged_val_mask = (
            raw_val_mask
            & (t1_ts < val_end_bound)
            & (positions <= first_test_pos - embargo_bars - 1)
        )

        train_idx = np.flatnonzero(purged_train_mask)
        val_idx = np.flatnonzero(purged_val_mask)
        test_idx = np.flatnonzero(raw_test_mask)

        if (
            len(train_idx) >= min_train_samples
            and len(val_idx) >= min_val_samples
            and len(test_idx) >= min_test_samples
        ):
            folds.append(
                WalkForwardFold(
                    fold_no=fold_no,
                    train_indices=train_idx,
                    val_indices=val_idx,
                    test_indices=test_idx,
                    train_start=ts[train_idx[0]],
                    train_end=ts[train_idx[-1]],
                    val_start=ts[val_idx[0]],
                    val_end=ts[val_idx[-1]],
                    test_start=ts[test_idx[0]],
                    test_end=ts[test_idx[-1]],
                    purged_from_train=int(np.sum(raw_train_mask) - len(train_idx)),
                    purged_from_val=int(np.sum(raw_val_mask) - len(val_idx)),
                )
            )
            fold_no += 1

        if test_end_bound >= t_max:
            break
        offset_months += step

    return folds
