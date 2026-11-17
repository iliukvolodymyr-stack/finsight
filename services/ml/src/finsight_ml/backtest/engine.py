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
        pos = signals.shift(1).fillna(0)
        price_returns = (open_prices.shift(-1) - open_prices) / open_prices
        strategy_returns = pos * price_returns

        # Deduct transaction costs on position changes
        pos_changes = pos.diff().abs().fillna(0)
        costs = pos_changes * self.total_cost_pct
        net_returns = (strategy_returns - costs).fillna(0)

        equity = (1.0 + net_returns).cumprod()
        sharpe = calculate_sharpe_ratio(net_returns)
        mdd = calculate_max_drawdown(equity)
        tot_ret = float(equity.iloc[-1] - 1.0) if len(equity) > 0 else 0.0

        trades_count = int((pos_changes > 0).sum())
        winning_trades = int((strategy_returns > costs).sum())
        win_rate = winning_trades / trades_count if trades_count > 0 else 0.0

        return BacktestResult(
            sharpe=sharpe,
            max_drawdown=mdd,
            total_return=tot_ret,
            trades_count=trades_count,
            win_rate=win_rate,
            equity_curve=equity,
        )
