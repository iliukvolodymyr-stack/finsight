from typing import Any

import numpy as np
import pandas as pd

from finsight_ml.risk.metrics import _normalize_daily_close


def compute_correlation_matrix(
    close_series_by_symbol: dict[str, pd.Series],
    window_days: int = 90,
) -> dict[str, Any]:
    """
    Compute cross-asset Pearson correlation matrix of daily log returns over `window_days`.

    Each pair (or the joint intersection when aligning mixed schedules such as
    24/7 crypto + NYSE ETFs) aligns on common active session timestamps BEFORE
    differencing so multi-day gaps (e.g. Fri -> Mon) represent identical time horizons.
    """
    symbols = [sym for sym, s in close_series_by_symbol.items() if not s.empty]
    if not symbols:
        return {"symbols": [], "matrix": [], "window_days": window_days, "observations": 0}

    norm_map = {sym: _normalize_daily_close(close_series_by_symbol[sym]) for sym in symbols}
    joined = pd.DataFrame(norm_map).sort_index().dropna(how="any")
    if len(joined) < 3:
        identity = np.eye(len(symbols), dtype=np.float64).tolist()
        return {
            "symbols": symbols,
            "matrix": identity,
            "window_days": window_days,
            "observations": max(0, len(joined) - 1),
        }

    # Keep window_days + 1 closes so differencing produces up to window_days returns
    if window_days > 0 and len(joined) > window_days + 1:
        joined = joined.iloc[-(window_days + 1) :]

    log_rets = np.log(joined / joined.shift(1)).dropna(how="any")
    if len(log_rets) < 2:
        identity = np.eye(len(symbols), dtype=np.float64).tolist()
        return {
            "symbols": symbols,
            "matrix": identity,
            "window_days": window_days,
            "observations": int(len(log_rets)),
        }

    corr_df = log_rets[symbols].corr(method="pearson").fillna(0.0)
    # Enforce exact unit diagonal & symmetry (explicit writable copy for pandas CoW / numpy 2)
    mat = np.array(corr_df.to_numpy(dtype=np.float64), copy=True)
    np.fill_diagonal(mat, 1.0)
    mat = np.clip(0.5 * (mat + mat.T), -1.0, 1.0)
    mat_list = [[round(float(val), 4) for val in row] for row in mat]

    return {
        "symbols": symbols,
        "matrix": mat_list,
        "window_days": int(window_days),
        "observations": int(len(log_rets)),
    }
