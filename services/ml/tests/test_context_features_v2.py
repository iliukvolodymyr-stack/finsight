from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd
import pytest
from finsight_ml.data import fetch_binance_funding_rates, parse_binance_funding_rates
from finsight_ml.features.context_v2 import (
    ContextFeatureBuilder,
    fit_transform_causal_hmm_fold,
    resample_ohlcv,
)


def _make_ohlcv(n: int = 800, seed: int = 11, with_micro: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01 00:00:00", periods=n, freq="h", tz="UTC")
    rets = rng.normal(0.0001, 0.01, size=n)
    close = 50000.0 * np.exp(np.cumsum(rets))
    open_p = np.roll(close, 1)
    open_p[0] = 50000.0
    spread = np.abs(rng.normal(0.0, 0.005, size=n)) * close
    high = np.maximum(open_p, close) + spread
    low = np.maximum(1.0, np.minimum(open_p, close) - spread)
    volume = rng.lognormal(mean=6.0, sigma=0.5, size=n)
    data: dict[str, object] = {
        "open_time": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    if with_micro:
        data["quote_volume"] = volume * close
        data["trades_count"] = rng.integers(500, 15000, size=n)
    return pd.DataFrame(data)


def _make_funding(dates: pd.DatetimeIndex, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ftimes = dates[::8]
    rates = rng.normal(0.0001, 0.00025, size=len(ftimes))
    return pd.DataFrame({"funding_time": ftimes, "funding_rate": rates})


def test_parse_and_fetch_binance_funding_rates_paginates() -> None:
    t0 = int(datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp() * 1000)
    t1 = int(datetime(2025, 1, 1, 8, 0, 0, tzinfo=UTC).timestamp() * 1000)
    fixture = [
        {
            "symbol": "BTCUSDT",
            "fundingTime": t0,
            "fundingRate": "0.00010000",
            "markPrice": "94000.5",
        },
        {"symbol": "BTCUSDT", "fundingTime": t1, "fundingRate": "-0.00005000", "markPrice": ""},
    ]
    records = parse_binance_funding_rates(fixture)
    assert len(records) == 2
    assert records[0].funding_time == datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert records[0].funding_rate == pytest.approx(0.0001)
    assert records[0].mark_price == pytest.approx(94000.5)
    assert records[1].mark_price is None

    async def _run() -> list[object]:
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=fixture)

        transport = httpx.MockTransport(_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_binance_funding_rates(
                "BTCUSDT",
                datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2025, 1, 1, 16, 0, tzinfo=UTC),
                client=client,
            )

    import asyncio

    fetched = asyncio.run(_run())
    assert len(fetched) == 2


def test_resample_ohlcv_4h_drops_partial_tail() -> None:
    df = _make_ohlcv(n=25)  # 6 complete 4h bars + 1 trailing 1h bar
    res = resample_ohlcv(df, rule="4h")
    assert len(res) == 6
    assert res["close"].iloc[0] == pytest.approx(df["close"].iloc[3])
    assert res["high"].iloc[0] == pytest.approx(df["high"].iloc[:4].max())
    assert res["low"].iloc[0] == pytest.approx(df["low"].iloc[:4].min())
    assert res["volume"].iloc[0] == pytest.approx(df["volume"].iloc[:4].sum())


def test_context_v2_future_shock_and_prefix_invariance() -> None:
    btc = _make_ohlcv(900, seed=1)
    eth = _make_ohlcv(900, seed=2)
    spy = _make_ohlcv(900, seed=3).iloc[::3].reset_index(drop=True)  # sparser macro timestamps
    funding = _make_funding(pd.DatetimeIndex(btc["open_time"]))

    builder = ContextFeatureBuilder(include_base_v1=True)
    full = builder.build(
        btc,
        funding_df=funding,
        peer_df=eth,
        peer_name="eth",
        macro_df=spy,
        macro_name="spy",
    ).features

    cutoff = 720
    cutoff_open = btc["open_time"].iloc[cutoff]
    cutoff_close = cutoff_open + pd.Timedelta(hours=1)

    # Prefix invariance strictly using bars up to cutoff
    pref = builder.build(
        btc.iloc[: cutoff + 1].copy(),
        funding_df=funding[funding["funding_time"] <= cutoff_open].copy(),
        peer_df=eth[eth["open_time"] <= cutoff_open].copy(),
        peer_name="eth",
        macro_df=spy[spy["open_time"] + pd.Timedelta(hours=1) <= cutoff_close].copy(),
        macro_name="spy",
    ).features

    # Drop columns with any warmup NaN only before max_lookback (720 > 600 so all finite except spy_age if sparse)
    f_exp = full.iloc[650 : cutoff + 1].to_numpy(dtype=np.float64)
    f_act = pref.iloc[650 : cutoff + 1].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(f_act, f_exp, rtol=1e-11, atol=1e-11, equal_nan=True)

    # Future shock test: 10x price/funding shock strictly after cutoff bar_close_time
    btc_shock = btc.copy()
    btc_shock.iloc[
        cutoff + 1 :, btc_shock.columns.get_indexer(["open", "high", "low", "close"])
    ] *= 10.0
    eth_shock = eth.copy()
    eth_shock.loc[eth_shock["open_time"] > cutoff_open, ["open", "high", "low", "close"]] *= 10.0
    funding_shock = funding.copy()
    funding_shock.loc[funding_shock["funding_time"] > cutoff_open, "funding_rate"] *= 50.0
    spy_shock = spy.copy()
    spy_shock.loc[spy_shock["open_time"] + pd.Timedelta(hours=1) > cutoff_close, "close"] *= 5.0

    shocked = builder.build(
        btc_shock,
        funding_df=funding_shock,
        peer_df=eth_shock,
        peer_name="eth",
        macro_df=spy_shock,
        macro_name="spy",
    ).features

    np.testing.assert_allclose(
        shocked.iloc[650 : cutoff + 1].to_numpy(dtype=np.float64),
        full.iloc[650 : cutoff + 1].to_numpy(dtype=np.float64),
        rtol=1e-11,
        atol=1e-11,
        equal_nan=True,
    )


def test_causal_hmm_fold_helper_has_no_eval_lookahead() -> None:
    df = _make_ohlcv(600, seed=99)
    train_close = df["close"].iloc[:400]
    eval_close = df["close"].iloc[400:]

    train_hmm, eval_hmm = fit_transform_causal_hmm_fold(
        train_close, eval_close, vol_span=24, n_iter=20
    )
    assert train_hmm.shape == (400, 4)
    assert eval_hmm is not None
    assert eval_hmm.shape == (200, 4)

    # Mutating future eval bars after position 50 must not alter probabilities at <= 50
    eval_shock = eval_close.copy()
    eval_shock.iloc[51:] *= 5.0
    _, eval_hmm_shock = fit_transform_causal_hmm_fold(
        train_close, eval_shock, vol_span=24, n_iter=20
    )
    assert eval_hmm_shock is not None
    np.testing.assert_allclose(
        eval_hmm_shock.iloc[:51].to_numpy(dtype=np.float64),
        eval_hmm.iloc[:51].to_numpy(dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
