"""
Tests for to_higher_timeframe — resampling base data to a higher timeframe
without look-ahead.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.indicators import to_higher_timeframe


@pytest.fixture
def fifteenmin_bars():
    """Two hours of 15-minute bars (8 bars), 14:00 → 15:45."""
    idx = pd.date_range("2024-01-01 14:00", periods=8, freq="15min")
    return pd.DataFrame({
        "Open":   [100, 102, 104, 106, 108, 110, 112, 114],
        "High":   [101, 103, 105, 107, 109, 111, 113, 115],
        "Low":    [ 99, 101, 103, 105, 107, 109, 111, 113],
        "Close":  [102, 104, 106, 108, 110, 112, 114, 116],
        "Volume": [100, 200, 150, 300, 250, 100, 400, 200],
        "Spread": [1.0, 1.5, 2.0, 1.2, 1.8, 2.5, 1.1, 1.0],
    }, index=idx)


def test_resample_15min_to_1h_aggregates_correctly(fifteenmin_bars):
    """Four 15m bars should aggregate into one 1h bar with correct OHLC."""
    # Pass a complete 8 bars (2 full hours): 14:00-15:00 and 15:00-16:00
    # Last base bar at 15:45 means 15:00 HTF bar is not complete (extends to 16:00)
    # So with include_partial=False, only 14:00 HTF bar remains.
    htf = to_higher_timeframe(fifteenmin_bars, "1h", include_partial=False)
    assert len(htf) == 1
    row = htf.iloc[0]
    # 14:00-14:45 bars: Open=100 (from 14:00 bar), High=max(101,103,105,107)=107,
    # Low=min(99,101,103,105)=99, Close=108 (from 14:45 bar)
    assert row["Open"] == pytest.approx(100)
    assert row["High"] == pytest.approx(107)
    assert row["Low"] == pytest.approx(99)
    assert row["Close"] == pytest.approx(108)
    # Volume sum: 100+200+150+300 = 750
    assert row["Volume"] == pytest.approx(750)
    # Spread average: (1.0+1.5+2.0+1.2)/4 = 1.425
    assert row["Spread"] == pytest.approx(1.425)


def test_include_partial_keeps_in_progress_htf_bar(fifteenmin_bars):
    """With include_partial=True, the still-forming 15:00 HTF bar should appear."""
    htf = to_higher_timeframe(fifteenmin_bars, "1h", include_partial=True)
    assert len(htf) == 2
    # 15:00 bar should reflect bars 4-7 (15:00, 15:15, 15:30, 15:45)
    row = htf.iloc[1]
    assert row["Open"] == pytest.approx(108)   # 15:00 open
    assert row["High"] == pytest.approx(115)   # max of 109..115
    assert row["Low"] == pytest.approx(107)
    assert row["Close"] == pytest.approx(116)  # 15:45 close


def test_drop_partial_when_only_a_few_bars(fifteenmin_bars):
    """If we only have 3 bars (less than a full hour), no completed HTF bar."""
    partial = fifteenmin_bars.iloc[:3]   # 14:00, 14:15, 14:30
    htf = to_higher_timeframe(partial, "1h", include_partial=False)
    # No complete 1h bar exists yet — 14:00 HTF bar covers 14:00-15:00,
    # but base data only goes to 14:30
    assert len(htf) == 0


def test_volume_omitted_when_absent():
    """If input has no Volume column, output shouldn't either."""
    idx = pd.date_range("2024-01-01 09:00", periods=8, freq="15min")
    df = pd.DataFrame({
        "Open":  [100] * 8, "High": [101] * 8,
        "Low":   [99] * 8,  "Close": [100] * 8,
    }, index=idx)
    htf = to_higher_timeframe(df, "1h", include_partial=True)
    assert "Volume" not in htf.columns
    assert "Spread" not in htf.columns


def test_resample_to_daily():
    """A full day of 15m bars should aggregate to one daily bar."""
    # UK session: 08:00 → 16:30, but let's just do 9 hours of 15m = 36 bars
    idx = pd.date_range("2024-01-01 08:00", periods=36, freq="15min")
    df = pd.DataFrame({
        "Open":  range(100, 136),
        "High":  range(101, 137),
        "Low":   range(99, 135),
        "Close": range(100, 136),
        "Volume": [10] * 36,
    }, index=idx)
    # Last bar at 16:45, 1D bar ends at next-day 00:00 — so not complete.
    htf_partial = to_higher_timeframe(df, "1d", include_partial=True)
    assert len(htf_partial) == 1
    assert htf_partial.iloc[0]["Open"] == 100
    assert htf_partial.iloc[0]["Close"] == 135
    assert htf_partial.iloc[0]["Volume"] == 360


def test_unknown_interval_raises():
    idx = pd.date_range("2024-01-01", periods=4, freq="15min")
    df = pd.DataFrame({
        "Open": [1, 2, 3, 4], "High": [1, 2, 3, 4],
        "Low": [1, 2, 3, 4], "Close": [1, 2, 3, 4],
    }, index=idx)
    with pytest.raises(ValueError):
        to_higher_timeframe(df, "13m")


def test_no_look_ahead_in_default_mode():
    """
    The critical safety test: the HTF bar shown must never use data from
    AFTER the last visible base bar.
    """
    # 6 15m bars: 14:00 → 15:15. Currently at 15:15 (a bar that started at 15:00 + 15min)
    idx = pd.date_range("2024-01-01 14:00", periods=6, freq="15min")
    df = pd.DataFrame({
        "Open":   [100, 102, 104, 106, 108, 110],
        "High":   [101, 103, 105, 107, 109, 111],
        "Low":    [99,  101, 103, 105, 107, 109],
        "Close":  [102, 104, 106, 108, 110, 112],
    }, index=idx)
    # 14:00 1h bar: covers 14:00-15:00. Last base = 15:15. 15:15 >= 15:00,
    # so 14:00 bar IS complete and should appear.
    # 15:00 1h bar: covers 15:00-16:00. Last base = 15:15 < 16:00, NOT complete.
    htf = to_higher_timeframe(df, "1h", include_partial=False)
    assert len(htf) == 1   # only the 14:00 bar
    assert htf.iloc[0]["Open"] == pytest.approx(100)
    assert htf.iloc[0]["Close"] == pytest.approx(108)   # close of the 14:45 bar
