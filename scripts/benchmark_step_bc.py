from pathlib import Path

import pandas as pd
from finsight_ml.features.context_v2 import ContextFeatureBuilder, resample_ohlcv
from finsight_ml.features.pipeline import FeaturePipeline
from finsight_ml.training.meta_labeling import (
    MetaLabelConfig,
    WalkForwardComparisonResult,
    run_v1_vs_v2_meta_walk_forward,
)
from rich.console import Console
from rich.table import Table


def _print_row(
    table: Table,
    label: str,
    bt,
    cls_str: str,
    hit_str: str,
    fee_usd: float,
) -> None:
    table.add_row(
        label,
        cls_str,
        hit_str,
        f"{bt.trades_count}",
        f"{bt.win_rate * 100:.1f}%",
        f"{bt.total_return * 100:+.2f}%",
        f"{bt.sharpe:+.2f}",
        f"{bt.max_drawdown * 100:.2f}%",
        f"${fee_usd:,.0f}",
    )


def _summarize(
    console: Console,
    title: str,
    v1_res: WalkForwardComparisonResult,
    v2_res: WalkForwardComparisonResult,
) -> None:
    tbl = Table(title=title)
    tbl.add_column("Pipeline / Variant", style="cyan", no_wrap=True)
    tbl.add_column("BalAcc / F1", justify="right", no_wrap=True)
    tbl.add_column("Active Hit%", justify="right", no_wrap=True)
    tbl.add_column("Trades", justify="right", no_wrap=True)
    tbl.add_column("Win%", justify="right", no_wrap=True)
    tbl.add_column("Net Ret", justify="right", no_wrap=True)
    tbl.add_column("Sharpe", justify="right", no_wrap=True)
    tbl.add_column("MaxDD", justify="right", no_wrap=True)
    tbl.add_column("TotalFees", justify="right", no_wrap=True)

    for name, bt, cls_m, hit_rate in (
        (
            "1) v1 Base Features (Primary)",
            v1_res.primary_backtest,
            f"{v1_res.primary_metrics['balanced_accuracy'] * 100:.1f}% / {v1_res.primary_metrics['macro_f1']:.3f}",
            "-",
        ),
        (
            "2) v2 Context+Funding+HMM (Primary)",
            v2_res.primary_backtest,
            f"{v2_res.primary_metrics['balanced_accuracy'] * 100:.1f}% / {v2_res.primary_metrics['macro_f1']:.3f}",
            "-",
        ),
        (
            "3) v2 + Stage-2 Meta-Labeling",
            v2_res.meta_backtest,
            f"cov={v2_res.meta_metrics['signal_coverage'] * 100:.1f}%",
            f"{v2_res.meta_metrics['exact_barrier_hit_rate'] * 100:.1f}%",
        ),
        (
            "4) v2 + Meta-Label + Vol-Sizing",
            v2_res.meta_vol_backtest,
            f"cov={v2_res.meta_metrics['signal_coverage'] * 100:.1f}%",
            f"{v2_res.meta_metrics['exact_barrier_hit_rate'] * 100:.1f}%",
        ),
    ):
        fees_usd = float(bt.metrics.get("total_fees", 0.0))
        _print_row(tbl, name, bt, cls_m, hit_rate, fees_usd)

    tbl.add_row(
        "Benchmark: Buy & Hold (same OOS)",
        "-",
        "-",
        "1",
        "-",
        f"{v2_res.buy_hold_return * 100:+.2f}%",
        f"{v2_res.buy_hold_sharpe:+.2f}",
        "-",
        "$30",
    )
    console.print(tbl)


def main() -> None:
    console = Console(width=140)
    data_dir = Path("var/data")
    btc = pd.read_parquet(data_dir / "btcusdt_1h.parquet")
    eth = pd.read_parquet(data_dir / "ethusdt_1h.parquet")
    spy = pd.read_parquet(data_dir / "spy_1h.parquet")
    btc_fund = pd.read_parquet(data_dir / "btcusdt_funding.parquet")
    eth_fund = pd.read_parquet(data_dir / "ethusdt_funding.parquet")

    v1_pipe = FeaturePipeline()
    v2_builder = ContextFeatureBuilder(include_base_v1=True)

    btc_4h = resample_ohlcv(btc, rule="4h")
    eth_4h = resample_ohlcv(eth, rule="4h")

    # 1. BTCUSDT 4h Long-Only Spot (horizon=12 bars = 48h, min_barrier=1.0%)
    print("[1/4] BTCUSDT 4h Spot (allow_short=False, H=48h, min_barrier=1.0%)...", flush=True)
    X_btc4_v1 = v1_pipe.compute(btc_4h)
    X_btc4_v2 = v2_builder.build(
        btc_4h, funding_df=btc_fund, peer_df=eth_4h, peer_name="eth", macro_df=spy, macro_name="spy"
    ).features
    cfg_4h_spot = MetaLabelConfig(
        horizon_bars=12,
        pt_mult=1.5,
        sl_mult=1.0,
        min_barrier_pct=0.010,
        embargo_bars=12,
        min_train_months=10,
        val_months=2,
        test_months=2,
        step_months=2,
        allow_short=False,
        min_val_trades=6,
    )
    r_btc_spot_v1 = run_v1_vs_v2_meta_walk_forward(
        btc_4h,
        X_btc4_v1,
        symbol="BTCUSDT",
        timeframe="4h",
        cfg=MetaLabelConfig(**{**cfg_4h_spot.__dict__, "use_hmm": False}),
    )
    r_btc_spot_v2 = run_v1_vs_v2_meta_walk_forward(
        btc_4h, X_btc4_v2, symbol="BTCUSDT", timeframe="4h", cfg=cfg_4h_spot
    )
    _summarize(
        console,
        f"BTCUSDT 4h SPOT (Long-Only, H=48h, {r_btc_spot_v2.oos_bars} bars, 30bps RT)",
        r_btc_spot_v1,
        r_btc_spot_v2,
    )

    # 2. BTCUSDT 4h Perpetual Long/Short (allow_short=True, H=48h, min_barrier=1.0%)
    print("[2/4] BTCUSDT 4h Perp Long/Short (allow_short=True, H=48h)...", flush=True)
    cfg_4h_perp = MetaLabelConfig(**{**cfg_4h_spot.__dict__, "allow_short": True})
    r_btc_perp_v1 = run_v1_vs_v2_meta_walk_forward(
        btc_4h,
        X_btc4_v1,
        symbol="BTCUSDT",
        timeframe="4h",
        cfg=MetaLabelConfig(**{**cfg_4h_perp.__dict__, "use_hmm": False}),
    )
    r_btc_perp_v2 = run_v1_vs_v2_meta_walk_forward(
        btc_4h, X_btc4_v2, symbol="BTCUSDT", timeframe="4h", cfg=cfg_4h_perp
    )
    _summarize(
        console,
        f"BTCUSDT 4h PERP (Long/Short, H=48h, {r_btc_perp_v2.oos_bars} bars, 30bps RT)",
        r_btc_perp_v1,
        r_btc_perp_v2,
    )

    # 3. ETHUSDT 4h Spot (allow_short=False, H=48h)
    print("[3/4] ETHUSDT 4h Spot (allow_short=False, H=48h)...", flush=True)
    X_eth4_v1 = v1_pipe.compute(eth_4h)
    X_eth4_v2 = v2_builder.build(
        eth_4h, funding_df=eth_fund, peer_df=btc_4h, peer_name="btc", macro_df=spy, macro_name="spy"
    ).features
    r_eth_spot_v1 = run_v1_vs_v2_meta_walk_forward(
        eth_4h,
        X_eth4_v1,
        symbol="ETHUSDT",
        timeframe="4h",
        cfg=MetaLabelConfig(**{**cfg_4h_spot.__dict__, "use_hmm": False}),
    )
    r_eth_spot_v2 = run_v1_vs_v2_meta_walk_forward(
        eth_4h, X_eth4_v2, symbol="ETHUSDT", timeframe="4h", cfg=cfg_4h_spot
    )
    _summarize(
        console,
        f"ETHUSDT 4h SPOT (Long-Only, H=48h, {r_eth_spot_v2.oos_bars} bars, 30bps RT)",
        r_eth_spot_v1,
        r_eth_spot_v2,
    )

    # 4. ETHUSDT 4h Perpetual Long/Short (allow_short=True, H=48h)
    print("[4/4] ETHUSDT 4h Perp Long/Short (allow_short=True, H=48h)...", flush=True)
    r_eth_perp_v1 = run_v1_vs_v2_meta_walk_forward(
        eth_4h,
        X_eth4_v1,
        symbol="ETHUSDT",
        timeframe="4h",
        cfg=MetaLabelConfig(**{**cfg_4h_perp.__dict__, "use_hmm": False}),
    )
    r_eth_perp_v2 = run_v1_vs_v2_meta_walk_forward(
        eth_4h, X_eth4_v2, symbol="ETHUSDT", timeframe="4h", cfg=cfg_4h_perp
    )
    _summarize(
        console,
        f"ETHUSDT 4h PERP (Long/Short, H=48h, {r_eth_perp_v2.oos_bars} bars, 30bps RT)",
        r_eth_perp_v1,
        r_eth_perp_v2,
    )


if __name__ == "__main__":
    main()
