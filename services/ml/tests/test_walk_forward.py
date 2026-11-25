import numpy as np
import pandas as pd
from finsight_ml.validation.walk_forward import purged_walk_forward_splits


def test_purged_walk_forward_strict_non_overlap_and_embargo() -> None:
    # 24 months of hourly bars (~17,520 bars) with 24h horizons
    dates = pd.date_range("2024-01-01 00:00:00", periods=17520, freq="h", tz="UTC")
    # Every label exits 24 bars later (t1 = t + 24h)
    t1 = pd.Series(dates + pd.Timedelta(hours=24), index=dates)

    folds = purged_walk_forward_splits(
        timestamps=dates,
        t1=t1,
        min_train_months=12,
        val_months=2,
        test_months=2,
        step_months=2,
        embargo_bars=24,
    )

    # Months [14..16), [16..18), [18..20), [20..22), [22..24) -> 5 folds
    assert len(folds) == 5

    prev_test_indices: set[int] = set()
    for fold in folds:
        tr = fold.train_indices
        va = fold.val_indices
        te = fold.test_indices

        # Strict index ordering
        assert tr.max() < va.min()
        assert va.max() < te.min()

        # Embargo gap respected (>= 24 bars between train->val and val->test)
        assert int(va.min() - tr.max()) > 24
        assert int(te.min() - va.max()) > 24

        # Zero label-horizon leakage across fold boundaries
        assert pd.DatetimeIndex(t1.iloc[tr]).max() < dates[va.min()]
        assert pd.DatetimeIndex(t1.iloc[va]).max() < dates[te.min()]

        # Purge counters recorded accurately
        assert fold.purged_from_train >= 24
        assert fold.purged_from_val >= 24

        # Test folds must be completely disjoint across the walk-forward sequence
        cur_test = set(te.tolist())
        assert prev_test_indices.isdisjoint(cur_test)
        prev_test_indices |= cur_test


def test_purged_walk_forward_handles_irregular_nyse_schedule() -> None:
    # Simulated trading-hours-only schedule (~7 bars/day, 5 days/week for ~24 months)
    all_hours = pd.date_range("2024-01-02 14:30:00", periods=17500, freq="h", tz="UTC")
    nyse_mask = (all_hours.day_of_week < 5) & (all_hours.hour >= 14) & (all_hours.hour <= 20)
    nyse_dates = all_hours[nyse_mask]
    # Exit 24 trading bars ahead (pos + 24)
    t1_vals = [nyse_dates[min(i + 24, len(nyse_dates) - 1)] for i in range(len(nyse_dates))]
    t1 = pd.Series(t1_vals, index=nyse_dates)
    t1.iloc[-24:] = pd.NaT

    folds = purged_walk_forward_splits(
        timestamps=nyse_dates,
        t1=t1,
        min_train_months=12,
        val_months=2,
        test_months=2,
        embargo_bars=24,
    )
    assert len(folds) >= 4
    for fold in folds:
        assert pd.DatetimeIndex(t1.iloc[fold.train_indices]).max() < nyse_dates[fold.val_indices[0]]
        assert pd.DatetimeIndex(t1.iloc[fold.val_indices]).max() < nyse_dates[fold.test_indices[0]]
        assert np.intersect1d(fold.train_indices, fold.test_indices).size == 0
