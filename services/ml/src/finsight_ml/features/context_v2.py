from dataclasses import dataclass

import numpy as np
import pandas as pd
from finsight_ml.features.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_realized_volatility,
    calculate_rsi,
    safe_div,
)
from finsight_ml.features.pipeline import FeaturePipeline, _extract_utc_timestamps
from finsight_ml.regime.hmm import CausalGaussianHMM, build_regime_observations


def _logret(close: pd.Series, period: int) -> pd.Series:
    """Backward log return `ln(close_t / close_{t-period})`."""
    c = close.astype("float64")
    return pd.Series(np.log(c / c.shift(period)), index=close.index, dtype="float64")


def infer_bar_duration(timestamps: pd.DatetimeIndex) -> pd.Timedelta:
    """Infer nominal bar duration from median positive timestamp step (defaults to 1h)."""
    if len(timestamps) < 2:
        return pd.Timedelta(hours=1)
    diffs = timestamps.to_series().diff().dropna()
    pos = diffs[diffs > pd.Timedelta(0)]
    if pos.empty:
        return pd.Timedelta(hours=1)
    med = pd.Timedelta(pos.median())
    return med if med > pd.Timedelta(0) else pd.Timedelta(hours=1)


def resample_ohlcv(
    df: pd.DataFrame,
    rule: str = "4h",
    *,
    min_bars_per_bucket: int | None = None,
) -> pd.DataFrame:
    """
    Deterministically resample lower-timeframe OHLCV bars into closed higher-timeframe bars.
    Buckets with fewer than `min_bars_per_bucket` observations (e.g. partial last bucket)
    are dropped when inferred cleanly, avoiding incomplete-candle lookahead bias.
    """
    if df.empty:
        return df.copy()

    ts = _extract_utc_timestamps(df)
    work = df.copy()
    work.index = ts
    work = work.sort_index()

    base_dur = infer_bar_duration(work.index)
    target_dur = pd.Timedelta(rule)
    expected_count = max(1, int(round(target_dur / base_dur)))
    req_count = expected_count if min_bars_per_bucket is None else max(1, min_bars_per_bucket)

    agg_spec: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "quote_volume" in work.columns:
        agg_spec["quote_volume"] = "sum"
    if "trades_count" in work.columns:
        agg_spec["trades_count"] = "sum"

    res = work.resample(rule, label="left", closed="left").agg(agg_spec)
    counts = work["close"].resample(rule, label="left", closed="left").count()
    res = res.loc[counts >= req_count].dropna(subset=["open", "high", "low", "close"])
    res = res.copy()
    res.insert(0, "open_time", res.index)
    return res.reset_index(drop=True)


def _kaufman_er(close: pd.Series, window: int) -> pd.Series:
    """Kaufman Efficiency Ratio in [0, 1]: |close_t - close_{t-n}| / sum(|diff|, n)."""
    net_move = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window=window, min_periods=window).sum()
    return safe_div(net_move, path, fill_zero=0.0).clip(0.0, 1.0)


def _donchian_position(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Relative close position within rolling [low_min, high_max] channel in [0, 1]."""
    roll_hi = high.rolling(window=window, min_periods=window).max()
    roll_lo = low.rolling(window=window, min_periods=window).min()
    return safe_div(close - roll_lo, roll_hi - roll_lo, fill_zero=0.5).clip(0.0, 1.0)


def _completed_htf_features(
    base_close_times: pd.Series,
    work_df: pd.DataFrame,
    rule: str,
    prefix: str,
    *,
    bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """
    Compute strictly causal higher-timeframe (`4h` or `24h`) features aligned by
    `htf_close_time <= base_close_time` so a base bar at `t` NEVER observes an
    unfinished higher-timeframe bar.
    """
    target_dur = pd.Timedelta(rule)
    expected_bars = max(1, int(round(target_dur / bar_duration)))
    htf = resample_ohlcv(work_df, rule=rule, min_bars_per_bucket=expected_bars)

    out_cols = [
        f"{prefix}_ret_1",
        f"{prefix}_ret_3",
        f"{prefix}_ema_ratio_12_26",
        f"{prefix}_rsi14",
        f"{prefix}_donchian_20",
    ]
    if len(htf) < 30:
        return pd.DataFrame(np.nan, index=base_close_times.index, columns=out_cols)

    htf_close_ts = pd.to_datetime(htf["open_time"], utc=True) + target_dur
    c = htf["close"].astype("float64")
    h = htf["high"].astype("float64")
    lo = htf["low"].astype("float64")
    ema12 = calculate_ema(c, 12)
    ema26 = calculate_ema(c, 26)

    htf_feats = pd.DataFrame(
        {
            "_htf_close_time": htf_close_ts,
            f"{prefix}_ret_1": _logret(c, 1),
            f"{prefix}_ret_3": _logret(c, 3),
            f"{prefix}_ema_ratio_12_26": safe_div(ema12, ema26, fill_zero=1.0),
            f"{prefix}_rsi14": calculate_rsi(c, 14),
            f"{prefix}_donchian_20": _donchian_position(h, lo, c, 20),
        }
    ).sort_values("_htf_close_time")

    probe = pd.DataFrame(
        {"_base_close_time": pd.to_datetime(base_close_times, utc=True)},
        index=base_close_times.index,
    )
    probe_sorted = probe.sort_values("_base_close_time")
    merged = pd.merge_asof(
        probe_sorted,
        htf_feats,
        left_on="_base_close_time",
        right_on="_htf_close_time",
        direction="backward",
    )
    merged.index = probe_sorted.index
    return merged.loc[base_close_times.index, out_cols]


def _align_funding_features(
    open_times: pd.Series,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Strictly backward alignment of 8h perpetual funding settlements (`funding_time <= open_time`).
    """
    cols = ["funding_rate_bps", "funding_ma_3", "funding_delta", "funding_zscore_30"]
    if funding_df.empty:
        return pd.DataFrame(np.nan, index=open_times.index, columns=cols)

    f = funding_df.copy()
    f["_funding_time"] = pd.to_datetime(f["funding_time"], utc=True)
    f = f.sort_values("_funding_time").drop_duplicates("_funding_time", keep="last")

    rate_bps = f["funding_rate"].astype("float64") * 10_000.0
    mu30 = rate_bps.rolling(window=30, min_periods=5).mean()
    sd30 = rate_bps.rolling(window=30, min_periods=5).std(ddof=0)

    f_feats = pd.DataFrame(
        {
            "_funding_time": f["_funding_time"],
            "funding_rate_bps": rate_bps,
            "funding_ma_3": rate_bps.rolling(window=3, min_periods=1).mean(),
            "funding_delta": rate_bps - rate_bps.shift(1),
            "funding_zscore_30": safe_div(rate_bps - mu30, sd30, fill_zero=0.0),
        }
    )

    probe = pd.DataFrame(
        {"_open_time": pd.to_datetime(open_times, utc=True)}, index=open_times.index
    )
    probe_sorted = probe.sort_values("_open_time")
    merged = pd.merge_asof(
        probe_sorted,
        f_feats,
        left_on="_open_time",
        right_on="_funding_time",
        direction="backward",
        tolerance=pd.Timedelta(hours=24),
    )
    merged.index = probe_sorted.index
    return merged.loc[open_times.index, cols]


def _align_cross_asset_features(
    target_close_times: pd.Series,
    target_ret1: pd.Series,
    cross_df: pd.DataFrame,
    prefix: str,
    *,
    is_equity_macro: bool,
) -> pd.DataFrame:
    """
    Align peer/macro asset bars strictly by `cross_bar_close <= target_bar_close` so
    inter-asset correlations, lead/lag returns, and relative strength have zero lookahead.
    """
    cols = (
        [
            f"{prefix}_ret_1",
            f"{prefix}_ret_6",
            f"{prefix}_ret_24",
            f"{prefix}_rv_24",
            f"{prefix}_age_hours",
        ]
        if is_equity_macro
        else [
            f"{prefix}_ret_1",
            f"{prefix}_ret_6",
            f"{prefix}_ret_24",
            f"{prefix}_rel_strength_24",
            f"{prefix}_corr_48",
            f"{prefix}_beta_72",
        ]
    )
    if cross_df.empty:
        return pd.DataFrame(np.nan, index=target_close_times.index, columns=cols)

    c_ts = _extract_utc_timestamps(cross_df)
    c_dur = infer_bar_duration(c_ts)
    c_close_ts = c_ts + c_dur
    c_close = cross_df["close"].astype("float64").reset_index(drop=True)

    c_ret1 = _logret(c_close, 1)
    c_feats = pd.DataFrame(
        {
            "_cross_close_time": pd.Series(c_close_ts).reset_index(drop=True),
            f"{prefix}_ret_1": c_ret1,
            f"{prefix}_ret_6": _logret(c_close, 6),
            f"{prefix}_ret_24": _logret(c_close, 24),
            f"{prefix}_rv_24": calculate_realized_volatility(c_ret1, 24),
        }
    ).sort_values("_cross_close_time")

    probe = pd.DataFrame(
        {
            "_target_close_time": pd.to_datetime(target_close_times, utc=True),
            "_target_ret1": target_ret1.astype("float64"),
        },
        index=target_close_times.index,
    )
    probe_sorted = probe.sort_values("_target_close_time")
    merged = pd.merge_asof(
        probe_sorted,
        c_feats,
        left_on="_target_close_time",
        right_on="_cross_close_time",
        direction="backward",
        tolerance=pd.Timedelta(days=5) if is_equity_macro else pd.Timedelta(hours=12),
    )
    merged.index = probe_sorted.index
    merged = merged.loc[target_close_times.index]

    if is_equity_macro:
        age_h = (
            merged["_target_close_time"] - merged["_cross_close_time"]
        ).dt.total_seconds() / 3600.0
        merged[f"{prefix}_age_hours"] = age_h.clip(lower=0.0)
        return merged[cols]

    # Relative strength & dynamic correlation/beta on aligned target timeline
    t_ret24 = target_ret1.rolling(window=24, min_periods=24).sum()
    merged[f"{prefix}_rel_strength_24"] = t_ret24 - merged[f"{prefix}_ret_24"]

    peer_r = merged[f"{prefix}_ret_1"]
    targ_r = merged["_target_ret1"]
    merged[f"{prefix}_corr_48"] = targ_r.rolling(window=48, min_periods=24).corr(peer_r).fillna(0.0)
    cov_72 = targ_r.rolling(window=72, min_periods=30).cov(peer_r)
    var_72 = peer_r.rolling(window=72, min_periods=30).var()
    merged[f"{prefix}_beta_72"] = safe_div(cov_72, var_72, fill_zero=1.0).clip(-5.0, 5.0)
    return merged[cols]


def fit_transform_causal_hmm_fold(
    train_close: pd.Series,
    eval_close: pd.Series | None = None,
    *,
    vol_span: int = 24,
    n_iter: int = 35,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Fit `CausalGaussianHMM` strictly on `train_close` ONLY, then apply strictly
    causal forward filtering `alpha_t = P(s_t | x_{1:t})` to `train_close` and optionally
    `eval_close` (seeded seamlessly by the preceding `train_close` history without ever
    re-fitting or backward-smoothing on `eval_close`).
    """
    hmm = CausalGaussianHMM(n_iter=n_iter)
    train_obs = build_regime_observations(train_close, vol_span=vol_span)
    hmm.fit(train_obs)
    train_res = hmm.filter_causal(train_obs)
    cols = ["hmm_p_bear", "hmm_p_range", "hmm_p_bull", "hmm_entropy"]

    def _to_df(proba: np.ndarray, idx: pd.Index) -> pd.DataFrame:
        p = np.clip(proba, 1e-9, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        entropy = -np.sum(p * np.log(p), axis=1) / np.log(3.0)
        return pd.DataFrame(
            {
                "hmm_p_bear": p[:, 0],
                "hmm_p_range": p[:, 1],
                "hmm_p_bull": p[:, 2],
                "hmm_entropy": entropy,
            },
            index=idx,
            columns=cols,
        )

    train_hmm = _to_df(train_res.proba, train_close.index)
    if eval_close is None or eval_close.empty:
        return train_hmm, None

    # Prepend a bounded causal warmup tail from train_close so ewm_vol and HMM alpha_t
    # do not restart cold at eval_close[0], while never touching future eval bars at step t.
    warmup = min(len(train_close), max(vol_span * 6, 200))
    combined_close = pd.concat([train_close.iloc[-warmup:], eval_close])
    combined_obs = build_regime_observations(combined_close, vol_span=vol_span)
    combined_res = hmm.filter_causal(combined_obs)
    eval_proba = combined_res.proba[warmup:]
    eval_hmm = _to_df(eval_proba, eval_close.index)
    return train_hmm, eval_hmm


@dataclass(frozen=True)
class ContextFeatureResult:
    features: pd.DataFrame
    feature_names: list[str]
    max_lookback: int


class ContextFeatureBuilder:
    """
    Point-in-time `core_market_v2` causal feature builder combining:
    - Base 32 `core_market_v1` indicators
    - Trend efficiency (Kaufman ER), Donchian breakout position, risk-adjusted returns (Sharpe-t)
    - OHLCV microstructure proxies (CLV, intrabar range/ATR, average trade size z-score)
    - Strictly completed higher-timeframe (`4h`, `24h`) trend & momentum features
    - Optional perpetual futures funding rate dynamics (`funding_time <= open_time`)
    - Optional peer/macro cross-asset lead-lag, relative strength, corr, beta (`cross_close <= target_close`)
    """

    def __init__(self, include_base_v1: bool = True) -> None:
        self.include_base_v1 = include_base_v1
        self.base_pipeline = FeaturePipeline()

    def build(
        self,
        df: pd.DataFrame,
        *,
        funding_df: pd.DataFrame | None = None,
        peer_df: pd.DataFrame | None = None,
        peer_name: str = "peer",
        macro_df: pd.DataFrame | None = None,
        macro_name: str = "macro",
        bar_duration: pd.Timedelta | None = None,
    ) -> ContextFeatureResult:
        if df.empty:
            return ContextFeatureResult(features=pd.DataFrame(), feature_names=[], max_lookback=200)

        ts = _extract_utc_timestamps(df)
        work = df.copy()
        work.index = pd.RangeIndex(len(work))
        dt = bar_duration if bar_duration is not None else infer_bar_duration(ts)

        open_times = pd.Series(ts, index=work.index)
        close_times = open_times + dt

        parts: list[pd.DataFrame] = []
        if self.include_base_v1:
            base = self.base_pipeline.compute(work).reset_index(drop=True)
            parts.append(base)

        c = work["close"].astype("float64")
        h = work["high"].astype("float64")
        lo = work["low"].astype("float64")
        atr14 = calculate_atr(h, lo, c, 14)
        ret1 = _logret(c, 1)
        ret24 = _logret(c, 24)
        ret72 = _logret(c, 72)
        rv24 = calculate_realized_volatility(ret1, 24)
        rv72 = calculate_realized_volatility(ret1, 72)

        clv = safe_div((c - lo) - (h - c), h - lo, fill_zero=0.0).clip(-1.0, 1.0)
        extra: dict[str, pd.Series] = {
            "er_24": _kaufman_er(c, 24),
            "er_72": _kaufman_er(c, 72),
            "er_168": _kaufman_er(c, 168),
            "donchian_pos_24": _donchian_position(h, lo, c, 24),
            "donchian_pos_72": _donchian_position(h, lo, c, 72),
            "donchian_pos_168": _donchian_position(h, lo, c, 168),
            "sharpe_24": safe_div(ret24, rv24 * np.sqrt(24.0), fill_zero=0.0).clip(-10.0, 10.0),
            "sharpe_72": safe_div(ret72, rv72 * np.sqrt(72.0), fill_zero=0.0).clip(-10.0, 10.0),
            "bar_clv": clv,
            "clv_ewm_12": clv.ewm(span=12, min_periods=12, adjust=False).mean(),
            "range_atr_ratio": safe_div(h - lo, atr14, fill_zero=1.0).clip(0.0, 10.0),
        }

        if "quote_volume" in work.columns and "trades_count" in work.columns:
            qv = work["quote_volume"].astype("float64")
            tc = work["trades_count"].astype("float64").clip(lower=1.0)
            avg_sz = np.log1p((qv / tc).clip(lower=0.0))
            mu_sz = avg_sz.rolling(window=24, min_periods=24).mean()
            sd_sz = avg_sz.rolling(window=24, min_periods=24).std(ddof=0)
            extra["avg_trade_size_z24"] = safe_div(avg_sz - mu_sz, sd_sz, fill_zero=0.0).clip(
                -8.0, 8.0
            )

        parts.append(pd.DataFrame(extra, index=work.index))

        # Strictly completed higher-timeframe bars (only include 4h HTF if base bar < 4h)
        work_with_ts = work.copy()
        work_with_ts["open_time"] = open_times
        if dt < pd.Timedelta(hours=4):
            parts.append(
                _completed_htf_features(
                    close_times, work_with_ts, rule="4h", prefix="htf4", bar_duration=dt
                )
            )
        if dt < pd.Timedelta(hours=24):
            parts.append(
                _completed_htf_features(
                    close_times, work_with_ts, rule="24h", prefix="htf24", bar_duration=dt
                )
            )

        if funding_df is not None:
            parts.append(_align_funding_features(open_times, funding_df))

        if peer_df is not None:
            parts.append(
                _align_cross_asset_features(
                    close_times, ret1, peer_df, prefix=peer_name, is_equity_macro=False
                )
            )

        if macro_df is not None:
            parts.append(
                _align_cross_asset_features(
                    close_times, ret1, macro_df, prefix=macro_name, is_equity_macro=True
                )
            )

        out = pd.concat(parts, axis=1)
        if isinstance(df.index, pd.DatetimeIndex):
            out.index = ts
        else:
            out.index = df.index

        return ContextFeatureResult(
            features=out,
            feature_names=list(out.columns),
            max_lookback=max(self.base_pipeline.max_lookback, 200),
        )
