import numpy as np
import pandas as pd
from finsight_ml.features.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_log_returns,
    calculate_macd,
    calculate_rsi,
)


def test_calculate_ema():
    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    ema = calculate_ema(series, span=3)
    assert len(ema) == len(series)
    assert not ema.isna().any()
    assert ema.iloc[-1] > ema.iloc[0]


def test_calculate_rsi():
    prices = pd.Series([100.0 + i + (i % 2) * 2 for i in range(30)])
    rsi = calculate_rsi(prices, period=14)
    assert len(rsi) == len(prices)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_calculate_macd():
    prices = pd.Series(np.linspace(100, 200, 50))
    macd_line, signal_line, macd_hist = calculate_macd(prices)
    assert len(macd_line) == len(prices)
    assert len(signal_line) == len(prices)
    assert len(macd_hist) == len(prices)


def test_calculate_atr():
    n = 30
    high = pd.Series(np.linspace(105, 205, n))
    low = pd.Series(np.linspace(95, 195, n))
    close = pd.Series(np.linspace(100, 200, n))
    atr = calculate_atr(high, low, close, period=14)
    assert len(atr) == n
    assert (atr.dropna() > 0).all()


def test_calculate_log_returns():
    prices = pd.Series([100.0, 102.0, 101.0, 105.0])
    rets = calculate_log_returns(prices, periods=[1])
    assert "logret_1" in rets.columns
    assert np.isclose(rets["logret_1"].iloc[1], np.log(102.0 / 100.0))
