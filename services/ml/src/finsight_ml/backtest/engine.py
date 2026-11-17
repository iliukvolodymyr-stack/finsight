from dataclasses import dataclass

import pandas as pd
from finsight_ml.backtest.metrics import calculate_max_drawdown, calculate_sharpe_ratio


@dataclass
class BacktestResult:
    sharpe: float
    max_drawdown: float
    total_return: float
    trades_count: int
    win_rate: float
    equity_curve: pd.Series


class BacktestEngine:
    def __init__(self, fee_bps: float = 10.0, slippage_bps: float = 5.0):
        self.total_cost_pct = (fee_bps + slippage_bps) / 10000.0

    def run(
        self,
        signals: pd.Series,
        open_prices: pd.Series,
        close_prices: pd.Series,
    ) -> BacktestResult:
        """
        Execute backtest:
        - Signal generated at close of bar t (signals.iloc[t])
        - Position entered/exited at open of bar t+1 (open_prices.iloc[t+1])
        - Fee + slippage applied on every position change
        """
        # Align signals to execution at open of next bar
        pos = signals.shift(1).fillna(0.0).astype(float)
        next_exit_price = open_prices.astype(float).shift(-1)
        if len(next_exit_price) > 0:
            next_exit_price.iloc[-1] = float(close_prices.iloc[-1])
        price_returns = (
            (next_exit_price - open_prices.astype(float)) / open_prices.astype(float)
        ).fillna(0.0)
        strategy_returns = (pos * price_returns).fillna(0.0)

        # Deduct transaction costs on position changes
        pos_prev_series = pos.shift(1).fillna(0.0)
        pos_changes = (pos - pos_prev_series).abs()
        costs = pos_changes * self.total_cost_pct
        net_returns = (strategy_returns - costs).fillna(0.0)

        equity = (1.0 + net_returns).cumprod()
        sharpe = calculate_sharpe_ratio(net_returns)
        mdd = calculate_max_drawdown(equity)
        tot_ret = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0

        # Calculate discrete trade-level metrics (round trips)
        trade_returns: list[float] = []
        current_trade_return = 1.0
        in_trade = False

        for r, c, p, p_prev in zip(strategy_returns, costs, pos, pos_prev_series, strict=False):
            if p != 0 and p_prev == 0:
                # Fresh entry from flat
                in_trade = True
                entry_cost = abs(p) * self.total_cost_pct
                current_trade_return = 1.0 + r - entry_cost
            elif in_trade:
                if p == p_prev:
                    # Holding unchanged position
                    current_trade_return *= 1.0 + r
                elif p == 0:
                    # Exit to flat
                    exit_cost = abs(p_prev) * self.total_cost_pct
                    current_trade_return *= 1.0 - exit_cost
                    trade_returns.append(current_trade_return - 1.0)
                    in_trade = False
                    current_trade_return = 1.0
                elif (p > 0) != (p_prev > 0):
                    # Direct reversal (+1 -> -1 or -1 -> +1):
                    # close old trade at bar open, open new trade with bar return r
                    exit_cost = abs(p_prev) * self.total_cost_pct
                    current_trade_return *= 1.0 - exit_cost
                    trade_returns.append(current_trade_return - 1.0)

                    entry_cost = abs(p) * self.total_cost_pct
                    current_trade_return = 1.0 + r - entry_cost
                else:
                    # Scaling same-side position
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
