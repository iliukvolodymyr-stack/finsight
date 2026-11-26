import numpy as np
import pandas as pd
from finsight_ml.regime import CausalGaussianHMM, build_regime_observations


def _make_three_regime_series() -> pd.Series:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2026-01-01", periods=450, freq="h", tz="UTC")
    # Bear (-0.4%/h), Range (0%), Bull (+0.4%/h)
    r_bear = rng.normal(-0.004, 0.008, size=150)
    r_range = rng.normal(0.000, 0.003, size=150)
    r_bull = rng.normal(+0.004, 0.007, size=150)
    rets = np.concatenate([r_bear, r_range, r_bull])
    close = 50000.0 * np.exp(np.cumsum(rets))
    return pd.Series(close, index=dates)


def test_hmm_causal_filtering_has_zero_lookahead() -> None:
    close = _make_three_regime_series()
    obs = build_regime_observations(close, vol_span=24)

    hmm = CausalGaussianHMM(n_iter=25).fit(obs.iloc[:300])
    out_full = hmm.filter_causal(obs)

    # Mutate future close prices (bar 320 onwards) by 5x -> probabilities at <= 319 must be identical!
    close_shocked = close.copy()
    close_shocked.iloc[320:] *= 5.0
    obs_shocked = build_regime_observations(close_shocked, vol_span=24)
    out_shocked = hmm.filter_causal(obs_shocked)

    np.testing.assert_allclose(
        out_shocked.proba[:320],
        out_full.proba[:320],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(out_full.proba.sum(axis=1), 1.0, atol=1e-9)
    assert set(out_full.regime).issubset({"bear", "range", "bull"})
    # Ordered means guarantee mu_bear <= mu_range <= mu_bull
    assert hmm.means[0, 0] <= hmm.means[1, 0] <= hmm.means[2, 0]
