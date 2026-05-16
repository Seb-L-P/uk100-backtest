"""
Tests for the 10 new indicators added to backtest/indicators.py.

Each test uses hand-crafted or synthetic data with known expected behaviour.
Not exhaustive — just enough to catch obvious bugs (off-by-one, NaN handling,
sign flips).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.indicators import (
    stochastic, adx, williams_r, obv, roc,
    keltner_channels, mfi, parabolic_sar, heikin_ashi, pivot_points_classic,
)


@pytest.fixture
def trending_up_df():
    """50 bars of monotonic uptrend."""
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    closes = np.linspace(100, 200, 50)
    highs = closes + 1
    lows = closes - 1
    return pd.DataFrame({
        "Open": closes - 0.5, "High": highs, "Low": lows, "Close": closes,
        "Volume": [1000] * 50,
    }, index=idx)


@pytest.fixture
def trending_down_df():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    closes = np.linspace(200, 100, 50)
    highs = closes + 1
    lows = closes - 1
    return pd.DataFrame({
        "Open": closes + 0.5, "High": highs, "Low": lows, "Close": closes,
        "Volume": [1000] * 50,
    }, index=idx)


@pytest.fixture
def oscillating_df():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    closes = 100 + 5 * np.sin(np.arange(50) / 3.0)
    highs = closes + 1
    lows = closes - 1
    return pd.DataFrame({
        "Open": closes, "High": highs, "Low": lows, "Close": closes,
        "Volume": [1000] * 50,
    }, index=idx)


# ---- Stochastic ---------------------------------------------------------
def test_stochastic_high_in_uptrend(trending_up_df):
    """On a clean uptrend, %K should be high (near 100)."""
    k, d = stochastic(trending_up_df, k_period=14)
    assert k.iloc[-1] > 70


def test_stochastic_low_in_downtrend(trending_down_df):
    k, _ = stochastic(trending_down_df, k_period=14)
    assert k.iloc[-1] < 30


# ---- ADX ----------------------------------------------------------------
def test_adx_high_in_trend(trending_up_df):
    """ADX should be high in a strong, persistent trend."""
    adx_line, plus_di, minus_di = adx(trending_up_df, period=14)
    # Strong trend → ADX > 30 usually
    assert adx_line.iloc[-1] > 25
    # +DI should dominate in uptrend
    assert plus_di.iloc[-1] > minus_di.iloc[-1]


def test_adx_higher_di_minus_in_downtrend(trending_down_df):
    _, plus_di, minus_di = adx(trending_down_df, period=14)
    assert minus_di.iloc[-1] > plus_di.iloc[-1]


# ---- Williams %R -------------------------------------------------------
def test_williams_r_in_range(oscillating_df):
    """Should always be in [-100, 0]."""
    wr = williams_r(oscillating_df, period=14)
    assert (wr >= -100).all()
    assert (wr <= 0).all()


# ---- OBV ----------------------------------------------------------------
def test_obv_rises_in_uptrend(trending_up_df):
    """All up bars + constant volume → OBV should be monotonically rising."""
    obv_series = obv(trending_up_df)
    assert obv_series.iloc[-1] > obv_series.iloc[10]


def test_obv_falls_in_downtrend(trending_down_df):
    obv_series = obv(trending_down_df)
    assert obv_series.iloc[-1] < obv_series.iloc[10]


def test_obv_no_volume_column_returns_zeros():
    """If Volume column is missing, OBV defaults to zeros."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"Open": [1, 2, 3, 4, 5], "High": [1, 2, 3, 4, 5],
                       "Low": [1, 2, 3, 4, 5], "Close": [1, 2, 3, 4, 5]}, index=idx)
    assert (obv(df) == 0).all()


# ---- ROC ----------------------------------------------------------------
def test_roc_positive_in_uptrend(trending_up_df):
    r = roc(trending_up_df["Close"], period=10)
    assert r.iloc[-1] > 0


def test_roc_negative_in_downtrend(trending_down_df):
    r = roc(trending_down_df["Close"], period=10)
    assert r.iloc[-1] < 0


# ---- Keltner Channels --------------------------------------------------
def test_keltner_bands_envelope_price(oscillating_df):
    mid, upper, lower = keltner_channels(oscillating_df)
    # In the tail, mid should be between upper and lower
    assert lower.iloc[-1] < mid.iloc[-1] < upper.iloc[-1]


# ---- MFI ----------------------------------------------------------------
def test_mfi_in_range(trending_up_df):
    """MFI must be in [0, 100]."""
    m = mfi(trending_up_df)
    assert (m >= 0).all()
    assert (m <= 100).all()


def test_mfi_high_in_uptrend(trending_up_df):
    """All up days with volume → MFI saturates near 100."""
    m = mfi(trending_up_df)
    assert m.iloc[-1] > 80


# ---- Parabolic SAR -----------------------------------------------------
def test_psar_below_price_in_uptrend(trending_up_df):
    """In an established uptrend, SAR sits below price."""
    sar = parabolic_sar(trending_up_df)
    # Skip first few bars (still initialising)
    assert sar.iloc[-1] < trending_up_df["Close"].iloc[-1]


# ---- Heikin Ashi -------------------------------------------------------
def test_heikin_ashi_columns():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Open":  [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
        "High":  [101, 103, 102, 104, 106, 105, 107, 109, 108, 110],
        "Low":   [99,  101, 100, 102, 104, 103, 105, 107, 106, 108],
        "Close": [102, 101, 103, 105, 104, 106, 108, 107, 109, 110],
    }, index=idx)
    ha = heikin_ashi(df)
    assert list(ha.columns) == ["HA_Open", "HA_High", "HA_Low", "HA_Close"]
    # HA_Close[0] should be average of OHLC[0]
    assert ha["HA_Close"].iloc[0] == pytest.approx((100 + 101 + 99 + 102) / 4)


# ---- Pivot Points ------------------------------------------------------
def test_pivot_points_columns_and_ordering():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Open":  range(100, 110), "High": range(105, 115),
        "Low":   range(95, 105), "Close": range(102, 112),
    }, index=idx)
    pivots = pivot_points_classic(df)
    assert set(pivots.columns) == {"P", "R1", "R2", "R3", "S1", "S2", "S3"}
    # Sanity: S3 < S2 < S1 < P < R1 < R2 < R3
    row = pivots.iloc[-1]
    assert row["S3"] < row["S2"] < row["S1"] < row["P"]
    assert row["P"] < row["R1"] < row["R2"] < row["R3"]
