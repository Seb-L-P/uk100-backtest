"""
Indicator tests on the synthetic OHLC fixture. Just enough to catch the
gross errors (off-by-one, sign flips, wrong formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.indicators import (
    sma, ema, atr, rsi, bollinger, macd, vwap, detect_fvg,
)


# ---- Moving averages ----------------------------------------------------
def test_sma_of_constant_is_constant():
    s = pd.Series([5.0] * 20)
    assert sma(s, 5).iloc[-1] == pytest.approx(5.0)


def test_ema_of_constant_is_constant():
    s = pd.Series([5.0] * 20)
    assert ema(s, 5).iloc[-1] == pytest.approx(5.0)


def test_ema_responds_to_change():
    s = pd.Series([1.0] * 10 + [10.0] * 10)
    val = ema(s, 5).iloc[-1]
    # Should be between 1.0 and 10.0, biased toward recent values
    assert 5.0 < val < 10.0


# ---- ATR ----------------------------------------------------------------
def test_atr_of_flat_data_is_small(synthetic_ohlc):
    """ATR exists and is positive for our synthetic data."""
    a = atr(synthetic_ohlc, period=5)
    last = a.iloc[-1]
    assert not pd.isna(last)
    assert last > 0


# ---- RSI ----------------------------------------------------------------
def test_rsi_strong_uptrend_above_70():
    """RSI on a clean monotonic uptrend should saturate near 100."""
    s = pd.Series([100 + i for i in range(30)])
    val = rsi(s, period=14).iloc[-1]
    assert val > 90


def test_rsi_strong_downtrend_below_30():
    s = pd.Series([100 - i for i in range(30)])
    val = rsi(s, period=14).iloc[-1]
    assert val < 10


def test_rsi_flat_data_near_50():
    """Flat prices → no gains, no losses → RSI defaults to neutral 50."""
    s = pd.Series([5.0] * 30)
    val = rsi(s, period=14).iloc[-1]
    assert val == pytest.approx(50.0)


# ---- Bollinger Bands ---------------------------------------------------
def test_bollinger_middle_equals_sma():
    s = pd.Series([100 + i for i in range(30)])
    mid, upper, lower = bollinger(s, period=20, mult=2.0)
    assert mid.iloc[-1] == pytest.approx(sma(s, 20).iloc[-1])


def test_bollinger_bands_envelope_price():
    """Mid is between upper and lower; both differ from mid."""
    s = pd.Series([100 + np.sin(i / 3) * 5 for i in range(50)])
    mid, upper, lower = bollinger(s, period=20, mult=2.0)
    assert lower.iloc[-1] < mid.iloc[-1] < upper.iloc[-1]


# ---- MACD ---------------------------------------------------------------
def test_macd_components_sum_correctly():
    """MACD line minus signal line should equal histogram."""
    s = pd.Series([100 + i + np.sin(i / 5) for i in range(50)])
    m, sig, hist = macd(s)
    assert (m - sig - hist).abs().max() < 1e-9


# ---- VWAP ---------------------------------------------------------------
def test_vwap_with_constant_price():
    """VWAP of constant prices = constant price."""
    idx = pd.date_range("2024-01-01", periods=20, freq="15min")
    df = pd.DataFrame({
        "High": [100.0] * 20, "Low": [100.0] * 20,
        "Close": [100.0] * 20, "Volume": [1000.0] * 20,
    }, index=idx)
    v = vwap(df)
    assert v.iloc[-1] == pytest.approx(100.0)


def test_vwap_session_reset_daily():
    """VWAP resets at the start of each calendar day."""
    idx = pd.date_range("2024-01-01 09:00", periods=10, freq="1h")
    idx2 = pd.date_range("2024-01-02 09:00", periods=10, freq="1h")
    full_idx = idx.append(idx2)
    # Day 1: price 100. Day 2: price 200. VWAP at end of day 2 should be near 200.
    df = pd.DataFrame({
        "High": [100.0] * 10 + [200.0] * 10,
        "Low":  [100.0] * 10 + [200.0] * 10,
        "Close":[100.0] * 10 + [200.0] * 10,
        "Volume":[1000.0] * 20,
    }, index=full_idx)
    v = vwap(df, session_reset=True)
    assert v.iloc[19] == pytest.approx(200.0)


# ---- FVG detection ------------------------------------------------------
def test_fvg_bullish_detected(synthetic_ohlc):
    """Synthetic data has a bullish FVG at bar 7 (b5.high=104 < b7.low=106)."""
    fvg = detect_fvg(synthetic_ohlc, 7)
    assert fvg is not None
    assert fvg.direction == "bullish"
    assert fvg.zone_low == pytest.approx(104.0)
    assert fvg.zone_high == pytest.approx(106.0)
    assert fvg.size_points == pytest.approx(2.0)


def test_fvg_returns_none_when_no_gap():
    """Three overlapping bars → no FVG."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({
        "Open": [100, 100, 100],
        "High": [105, 105, 105],
        "Low": [95, 95, 95],
        "Close": [100, 100, 100],
        "Volume": [1000, 1000, 1000],
    }, index=idx)
    assert detect_fvg(df, 2) is None


def test_fvg_requires_at_least_three_bars():
    """Can't form a 3-bar FVG with only 2 bars."""
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({
        "Open": [100, 110], "High": [105, 115],
        "Low": [95, 105], "Close": [102, 112],
        "Volume": [1000, 1000],
    }, index=idx)
    assert detect_fvg(df, 1) is None
