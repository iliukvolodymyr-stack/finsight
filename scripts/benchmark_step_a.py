"""Step A Scientific Benchmark: Naive Persistence vs Adaptive Conformal vs Quantile LGBM + Vol Forecasting.

Runs strictly out-of-sample Purged Walk-Forward on real local 1h Parquet datasets (`BTCUSDT`, `ETHUSDT`, `SPY`).
"""

from pathlib import Path

import pandas as pd
from finsight_ml.training import TrainingConfig
from finsight_ml.training.distribution import evaluate_distribution_and_vol_walk_forward
from rich.console import Console
from rich.table import Table

console = Console()

TARGETS = [
    ("configs/models/lgbm_btc_1h.yaml", "var/data/btcusdt_1h.parquet"),
    ("configs/models/lgbm_eth_1h.yaml", "var/data/ethusdt_1h.parquet"),
    ("configs/models/lgbm_spy_1h.yaml", "var/data/spy_1h.parquet"),
]


def _load_pq(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time").sort_index()


def main() -> None:
    for cfg_path, pq_path in TARGETS:
        if not Path(pq_path).exists():
            continue
        cfg = TrainingConfig.from_yaml(cfg_path)
        df = _load_pq(pq_path)
        res = evaluate_distribution_and_vol_walk_forward(df, cfg)

        iv_table = Table(
            title=f"80% Corridor Benchmark — {cfg.symbol} {cfg.timeframe} ({res['n_oos_samples']} OOS, {res['n_folds']} folds)"
        )
        iv_table.add_column("Method")
        iv_table.add_column("Coverage", justify="right")
        iv_table.add_column("Q1 Calm Cov", justify="right")
        iv_table.add_column("Q4 High-Vol Cov", justify="right")
        iv_table.add_column("Width (logret)", justify="right")
        iv_table.add_column("Winkler ↓", justify="right")
        iv_table.add_column("Pinball ↓", justify="right")

        for label, key in (
            ("Naive Persistence (static q10/q90)", "naive_persistence"),
            ("Vol-Normalized Conformal (our v1)", "vol_conformal"),
            ("Quantile LGBM (0.10/0.50/0.90)", "quantile_lgbm_raw"),
            ("Quantile LGBM + CQR", "quantile_lgbm_cqr"),
        ):
            m = res["intervals"][key]
            iv_table.add_row(
                label,
                f"{m['coverage'] * 100:.1f}%",
                f"{m.get('low_vol_q1_cov', 0.0) * 100:.1f}%",
                f"{m.get('high_vol_q4_cov', 0.0) * 100:.1f}%",
                f"{m['mean_width_logret'] * 100:.2f}%",
                f"{m['winkler_score']:.5f}",
                f"{m['pinball_mean']:.5f}",
            )
        console.print(iv_table)

        vol_table = Table(title=f"24h Forward Volatility Forecast — {cfg.symbol} {cfg.timeframe}")
        vol_table.add_column("Model")
        vol_table.add_column("RMSE ↓", justify="right")
        vol_table.add_column("MAE ↓", justify="right")
        vol_table.add_column("QLIKE ↓", justify="right")
        vol_table.add_column("Log-Vol R² ↑", justify="right")

        for label, key in (
            ("EWMA Persistence (sigma_t)", "ewm_persistence"),
            ("LightGBM Vol Regressor", "vol_lgbm"),
        ):
            vm = res["volatility"][key]
            vol_table.add_row(
                label,
                f"{vm['rmse']:.6f}",
                f"{vm['mae']:.6f}",
                f"{vm['qlike']:.4f}",
                f"{vm['log_r2']:.4f}",
            )
        console.print(vol_table)


if __name__ == "__main__":
    main()
