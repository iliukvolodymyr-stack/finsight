from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from finsight_ml.backtest.metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    periods_per_year_for,
    summarize_backtest_metrics,
)
from finsight_ml.labels.triple_barrier import (
    compute_ewm_volatility,
    evaluate_barrier_path,
)


@dataclass(frozen=True)
class BacktestTrade:
    seq: int
    side: Literal["LONG", "SHORT"]
    entry_pos: int
    exit_pos: int
    entry_at: datetime | pd.Timestamp
    exit_at: datetime | pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    fee: float
    pnl: float
    exit_reason: str
    confidence: float | None = None
    fold_no: int | None = None


@dataclass
class BacktestResult:
    sharpe: float
    max_drawdown: float
    total_return: float
    trades_count: int
    win_rate: float
    equity_curve: pd.Series
    trades: list[BacktestTrade] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def validate_backtest_no_in_sample_leak(
    mode: str,
    period_start: datetime,
    train_data_end: datetime | None,
    embargo_bars: int = 24,
    timeframe: str = "1h",
) -> None:
    """
    Prevent silent in-sample evaluation in `static_model` mode:
    reject if period_start < train_data_end + embargo.
    """
    if mode != "static_model" or train_data_end is None:
        return
    bar_delta = timedelta(hours=embargo_bars) if timeframe == "1h" else timedelta(days=embargo_bars)
    min_allowed = train_data_end + bar_delta
    if period_start < min_allowed:
        raise ValueError(
            f"In-sample leakage guard: static_model period_start ({period_start.isoformat()}) "
            f"must be >= train_data_end + embargo ({min_allowed.isoformat()}). "
            "Use mode='walk_forward_retrain' to evaluate historical OOS folds."
        )


class BacktestEngine:
    def __init__(self, fee_bps: float = 10.0, slippage_bps: float = 5.0) -> None:
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)
        self.total_cost_pct = (self.fee_bps + self.slippage_bps) / 10000.0

    def run(
        self,
        signals: pd.Series,
        open_prices: pd.Series,
        close_prices: pd.Series,
    ) -> BacktestResult:
        """
        Simple bar-by-bar vectorized next-open backtest (kept for signal reversals & unit checks).
        """
        pos = signals.shift(1).fillna(0.0).astype(float)
        next_exit_price = open_prices.astype(float).shift(-1)
        if len(next_exit_price) > 0:
            next_exit_price.iloc[-1] = float(close_prices.iloc[-1])
        price_returns = (
            (next_exit_price - open_prices.astype(float)) / open_prices.astype(float)
        ).fillna(0.0)
        strategy_returns = (pos * price_returns).fillna(0.0)

        pos_prev_series = pos.shift(1).fillna(0.0)
        pos_changes = (pos - pos_prev_series).abs()
        costs = pos_changes * self.total_cost_pct
        net_returns = (strategy_returns - costs).fillna(0.0)

        equity = (1.0 + net_returns).cumprod()
        sharpe = calculate_sharpe_ratio(net_returns)
        mdd = calculate_max_drawdown(equity)
        tot_ret = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0

        trade_returns: list[float] = []
        current_trade_return = 1.0
        in_trade = False

        for r, c, p, p_prev in zip(strategy_returns, costs, pos, pos_prev_series, strict=False):
            if p != 0 and p_prev == 0:
                in_trade = True
                entry_cost = abs(p) * self.total_cost_pct
                current_trade_return = 1.0 + r - entry_cost
            elif in_trade:
                if p == p_prev:
                    current_trade_return *= 1.0 + r
                elif p == 0:
                    exit_cost = abs(p_prev) * self.total_cost_pct
                    current_trade_return *= 1.0 - exit_cost
                    trade_returns.append(current_trade_return - 1.0)
                    in_trade = False
                    current_trade_return = 1.0
                elif (p > 0) != (p_prev > 0):
                    exit_cost = abs(p_prev) * self.total_cost_pct
                    current_trade_return *= 1.0 - exit_cost
                    trade_returns.append(current_trade_return - 1.0)
                    entry_cost = abs(p) * self.total_cost_pct
                    current_trade_return = 1.0 + r - entry_cost
                else:
                    current_trade_return *= 1.0 + r - c

        if in_trade:
            trade_returns.append(current_trade_return - 1.0)

        trades_count = len(trade_returns)
        winning_trades = sum(1 for tr in trade_returns if tr > 0)
        win_rate = winning_trades / trades_count if trades_count > 0 else 0.0

        return BacktestResult(
            sharpe=sharpe,
            max_drawdown=mdd,
            total_return=tot_ret,
            trades_count=trades_count,
            win_rate=win_rate,
            equity_curve=equity,
        )

    def run_barriers(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        horizon_bars: int = 24,
        pt_mult: float = 1.0,
        sl_mult: float = 1.0,
        volatility: pd.Series | None = None,
        min_barrier_pct: float = 0.003,
        position_size: float = 1.0,
        position_sizes: pd.Series | None = None,
        allow_short: bool = False,
        initial_capital: float = 10000.0,
        exchange: str = "binance",
        timeframe: str = "1h",
        confidence: pd.Series | None = None,
        fold_map: pd.Series | None = None,
        benchmark_equity: pd.Series | None = None,
    ) -> BacktestResult:
        """
        Backtest v0 (`exit_mode='label_barriers'`):
        - Signal computed at close of bar t enters at open of bar t+1.
        - Fixed +/-k * sigma_t * sqrt(H) TP/SL barriers and H-bar vertical barrier
          evaluated via the exact same `evaluate_barrier_path` as Triple Barrier labeling.
        - Spot long/flat by default (`allow_short=False`), single open position max.
        - Full round-trip transaction costs (`fee_bps + slippage_bps` per side).
        """
        if df.empty:
            empty_eq = pd.Series([], index=df.index, dtype="float64")
            return BacktestResult(0.0, 0.0, 0.0, 0, 0.0, empty_eq, [], {})
        if not 0.0 < position_size <= 1.0:
            raise ValueError("position_size must be in (0, 1]")

        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")

        open_arr = df["open"].astype("float64").to_numpy()
        high_arr = df["high"].astype("float64").to_numpy()
        low_arr = df["low"].astype("float64").to_numpy()
        close_arr = df["close"].astype("float64").to_numpy()
        sig_arr = signals.reindex(df.index).fillna(0).to_numpy(dtype=np.int64)

        if volatility is None:
            vol_series = compute_ewm_volatility(df["close"], span=24, min_periods=24)
        else:
            vol_series = volatility.reindex(df.index).astype("float64")
        sigma_arr = vol_series.to_numpy()

        conf_arr = (
            confidence.reindex(df.index).to_numpy(dtype=np.float64)
            if confidence is not None
            else None
        )
        size_arr = (
            np.clip(
                position_sizes.reindex(df.index).fillna(position_size).to_numpy(dtype=np.float64),
                0.01,
                1.0,
            )
            if position_sizes is not None
            else None
        )
        fold_arr = fold_map.reindex(df.index).to_numpy() if fold_map is not None else None

        n = len(df)
        idx = df.index
        horizon_scale = float(np.sqrt(horizon_bars))
        equity_arr = np.full(n, float(initial_capital), dtype=np.float64)

        capital = float(initial_capital)
        total_traded_notional = 0.0
        trades: list[BacktestTrade] = []
        seq = 1

        # Active position state across bars
        in_pos = False
        pos_side: Literal["LONG", "SHORT"] = "LONG"
        pos_dir = 1
        pos_entry_pos = -1
        pos_exit_pos = -1
        pos_entry_price = 0.0
        pos_exit_price = 0.0
        pos_exit_reason = ""
        pos_qty = 0.0
        pos_entry_fee = 0.0
        pos_conf: float | None = None
        pos_fold: int | None = None

        for j in range(n):
            # 1. If a scheduled trade exits on bar j, realize PnL (if exit_pos == j)
            if in_pos and j == pos_exit_pos:
                exit_notional = pos_qty * pos_exit_price
                exit_fee = exit_notional * self.total_cost_pct
                gross_pnl = float(pos_dir) * pos_qty * (pos_exit_price - pos_entry_price)
                net_pnl = gross_pnl - pos_entry_fee - exit_fee
                capital = max(0.0, capital + net_pnl)
                total_traded_notional += exit_notional

                trades.append(
                    BacktestTrade(
                        seq=seq,
                        side=pos_side,
                        entry_pos=pos_entry_pos,
                        exit_pos=pos_exit_pos,
                        entry_at=idx[pos_entry_pos],
                        exit_at=idx[pos_exit_pos],
                        entry_price=pos_entry_price,
                        exit_price=pos_exit_price,
                        quantity=pos_qty,
                        fee=pos_entry_fee + exit_fee,
                        pnl=net_pnl,
                        exit_reason=pos_exit_reason,
                        confidence=pos_conf,
                        fold_no=pos_fold,
                    )
                )
                seq += 1
                in_pos = False
                equity_arr[j] = capital
            elif in_pos:
                # Mark-to-market open position at bar j close
                unrealized = float(pos_dir) * pos_qty * (float(close_arr[j]) - pos_entry_price)
                equity_arr[j] = max(0.0, capital - pos_entry_fee + unrealized)
            else:
                equity_arr[j] = capital

            # 2. Check signal at close of bar j to enter at open of bar j + 1
            if not in_pos and j < n - 1 and capital > 0.0:
                sig = int(sig_arr[j])
                if sig == 1 or (allow_short and sig == -1):
                    vol = float(sigma_arr[j])
                    entry_pos = j + 1
                    entry_price = float(open_arr[entry_pos])
                    if (
                        not np.isnan(vol)
                        and entry_price > 0.0
                        and max(pt_mult, sl_mult) * vol * horizon_scale >= min_barrier_pct
                    ):
                        pos_side = "LONG" if sig == 1 else "SHORT"
                        pos_dir = 1 if sig == 1 else -1

                        if pos_side == "LONG":
                            b_up = entry_price * float(np.exp(pt_mult * vol * horizon_scale))
                            b_down = entry_price * float(np.exp(-sl_mult * vol * horizon_scale))
                        else:
                            # Symmetric TP/SL around entry for short trades
                            b_up = entry_price * float(np.exp(sl_mult * vol * horizon_scale))
                            b_down = entry_price * float(np.exp(-pt_mult * vol * horizon_scale))

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
                            check_gap_open=True,
                            conservative_side=pos_side,
                        )

                        bar_size = (
                            float(size_arr[j]) if size_arr is not None else float(position_size)
                        )
                        notional = capital * bar_size
                        pos_qty = notional / entry_price
                        pos_entry_fee = notional * self.total_cost_pct
                        total_traded_notional += notional

                        pos_entry_pos = entry_pos
                        pos_exit_pos = outcome.exit_pos
                        pos_entry_price = entry_price
                        pos_exit_price = outcome.exit_price
                        pos_exit_reason = outcome.exit_reason
                        if outcome.exit_pos == n - 1 and outcome.exit_reason == "vertical":
                            pos_exit_reason = (
                                "end_of_period"
                                if (pos_exit_pos - pos_entry_pos + 1) < horizon_bars
                                else "vertical"
                            )

                        pos_conf = (
                            float(conf_arr[j])
                            if conf_arr is not None and np.isfinite(conf_arr[j])
                            else None
                        )
                        pos_fold = (
                            int(fold_arr[j])
                            if fold_arr is not None and pd.notna(fold_arr[j])
                            else None
                        )
                        in_pos = True

        equity_series = pd.Series(equity_arr, index=idx, dtype="float64")
        ppy = periods_per_year_for(exchange=exchange, timeframe=timeframe)
        metrics = summarize_backtest_metrics(
            equity_curve=equity_series,
            trade_pnls=[t.pnl for t in trades],
            trade_fees=[t.fee for t in trades],
            total_traded_notional=total_traded_notional,
            initial_capital=initial_capital,
            periods_per_year=ppy,
            benchmark_equity=benchmark_equity,
        )
        return BacktestResult(
            sharpe=float(metrics["sharpe"]),
            max_drawdown=float(metrics["max_drawdown"]),
            total_return=float(metrics["total_return"]),
            trades_count=int(metrics["trades_count"]),
            win_rate=float(metrics["win_rate"]),
            equity_curve=equity_series,
            trades=trades,
            metrics=metrics,
        )
