from typing import Any

import numpy as np
import pandas as pd
from app.infra.models import CandleModel, InstrumentModel
from finsight_ml.regime import CausalGaussianHMM, build_regime_observations
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession


async def compute_instrument_regime(
    session: AsyncSession,
    instrument: InstrumentModel,
    timeframe: str = "1h",
    days: int = 30,
) -> dict[str, Any]:
    """
    Causal 3-state Gaussian HMM regime baseline (`bear`, `range`, `bull`).

    Fits strictly on the initial 70% historical slice of fetched candles and emits
    strictly forward-filtered probabilities `P(s_t | x_{1:t})` over the tail window.
    """
    bars_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(timeframe, 1)
    fetch_limit = max(600, days * bars_per_day + 500)
    rows = (
        await session.execute(
            select(CandleModel.open_time, CandleModel.close)
            .where(
                CandleModel.instrument_id == instrument.id,
                CandleModel.timeframe == timeframe,
                CandleModel.is_closed.is_(True),
            )
            .order_by(desc(CandleModel.open_time))
            .limit(fetch_limit)
        )
    ).all()

    if len(rows) < 80:
        return {
            "symbol": instrument.symbol,
            "timeframe": timeframe,
            "regime": "range",
            "probabilities": {"bear": 0.3333, "range": 0.3334, "bull": 0.3333},
            "as_of": None,
            "method": "causal_gaussian_hmm_3state",
            "history": [],
        }

    ordered = list(reversed(rows))
    idx = pd.DatetimeIndex([r[0] for r in ordered])
    close = pd.Series([float(r[1]) for r in ordered], index=idx, dtype="float64")
    obs = build_regime_observations(close, vol_span=24).dropna()

    if len(obs) < 50:
        return {
            "symbol": instrument.symbol,
            "timeframe": timeframe,
            "regime": "range",
            "probabilities": {"bear": 0.3333, "range": 0.3334, "bull": 0.3333},
            "as_of": None,
            "method": "causal_gaussian_hmm_3state",
            "history": [],
        }

    fit_end = max(40, int(0.70 * len(obs)))
    hmm = CausalGaussianHMM(n_iter=30).fit(obs.iloc[:fit_end])
    filtered = hmm.filter_causal(obs)

    tail_bars = min(len(obs), max(24, days * bars_per_day))
    step = max(1, tail_bars // 180)

    start_i = len(obs) - tail_bars
    history = []
    for pos in range(start_i, len(obs), step):
        p_vec = filtered.proba[pos]
        history.append(
            {
                "t": obs.index[pos].isoformat().replace("+00:00", "Z"),
                "regime": filtered.regime[pos],
                "probabilities": {
                    "bear": round(float(p_vec[0]), 4),
                    "range": round(float(p_vec[1]), 4),
                    "bull": round(float(p_vec[2]), 4),
                },
            }
        )

    last_pos = len(obs) - 1
    p_last = np.asarray(filtered.proba[last_pos], dtype=np.float64)
    return {
        "symbol": instrument.symbol,
        "timeframe": timeframe,
        "regime": filtered.regime[last_pos],
        "probabilities": {
            "bear": round(float(p_last[0]), 4),
            "range": round(float(p_last[1]), 4),
            "bull": round(float(p_last[2]), 4),
        },
        "as_of": obs.index[last_pos].isoformat().replace("+00:00", "Z"),
        "method": "causal_gaussian_hmm_3state",
        "history": history,
    }
