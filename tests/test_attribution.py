"""
Tests for the attribution module — verify per-slice aggregations are correct.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.attribution import (
    by_hour_of_day, by_day_of_week, by_month,
    by_session_phase, by_side, by_exit_reason,
    equity_drawdown_series, monthly_return_heatmap_data,
)


@pytest.fixture
def synth_trades() -> pd.DataFrame:
    """8 trades across different hours, days, and outcomes for predictable slicing."""
    return pd.DataFrame([
        # Two winning longs Mon 09:00 and Tue 09:00 (hour 9)
        {"entry_time": pd.Timestamp("2024-01-01 09:00"),  # Monday
         "side": "long", "net_pnl_gbp": 100.0, "exit_reason": "target"},
        {"entry_time": pd.Timestamp("2024-01-02 09:00"),  # Tuesday
         "side": "long", "net_pnl_gbp": 50.0, "exit_reason": "target"},
        # Two losing shorts Mon 14:00 and Tue 14:00 (hour 14)
        {"entry_time": pd.Timestamp("2024-01-01 14:00"),
         "side": "short", "net_pnl_gbp": -60.0, "exit_reason": "stop"},
        {"entry_time": pd.Timestamp("2024-01-02 14:00"),
         "side": "short", "net_pnl_gbp": -40.0, "exit_reason": "stop"},
        # Mixed at hour 11
        {"entry_time": pd.Timestamp("2024-01-03 11:00"),  # Wednesday
         "side": "long", "net_pnl_gbp": 75.0, "exit_reason": "session_end"},
        {"entry_time": pd.Timestamp("2024-01-03 11:00"),
         "side": "short", "net_pnl_gbp": -25.0, "exit_reason": "stop"},
        # Big win Friday
        {"entry_time": pd.Timestamp("2024-01-05 13:00"),  # Friday
         "side": "long", "net_pnl_gbp": 200.0, "exit_reason": "target"},
        # Loss in February
        {"entry_time": pd.Timestamp("2024-02-01 10:00"),
         "side": "long", "net_pnl_gbp": -80.0, "exit_reason": "stop"},
    ])


def test_by_hour_of_day_basic(synth_trades):
    df = by_hour_of_day(synth_trades)
    # Should have rows for hours 9, 10, 11, 13, 14
    assert set(df["hour"]) == {9, 10, 11, 13, 14}
    # Hour 9: 2 trades both wins, total +£150
    row_9 = df[df["hour"] == 9].iloc[0]
    assert row_9["trades"] == 2
    assert row_9["wins"] == 2
    assert row_9["total_pnl_gbp"] == pytest.approx(150.0)
    # Hour 14: 2 trades both losses, total -£100
    row_14 = df[df["hour"] == 14].iloc[0]
    assert row_14["trades"] == 2
    assert row_14["wins"] == 0
    assert row_14["total_pnl_gbp"] == pytest.approx(-100.0)


def test_by_day_of_week_basic(synth_trades):
    df = by_day_of_week(synth_trades)
    # Monday: 2 trades, one win (+100), one loss (-60), total +£40
    row_mon = df[df["day"] == "Mon"].iloc[0]
    assert row_mon["trades"] == 2
    assert row_mon["total_pnl_gbp"] == pytest.approx(40.0)
    # Friday: 1 trade, win
    row_fri = df[df["day"] == "Fri"].iloc[0]
    assert row_fri["trades"] == 1
    assert row_fri["total_pnl_gbp"] == pytest.approx(200.0)


def test_by_month_basic(synth_trades):
    df = by_month(synth_trades)
    # January: 7 trades, total = 100+50-60-40+75-25+200 = 300
    row_jan = df[df["month"] == "2024-01"].iloc[0]
    assert row_jan["trades"] == 7
    assert row_jan["total_pnl_gbp"] == pytest.approx(300.0)
    # February: 1 trade, -80
    row_feb = df[df["month"] == "2024-02"].iloc[0]
    assert row_feb["trades"] == 1
    assert row_feb["total_pnl_gbp"] == pytest.approx(-80.0)


def test_by_side(synth_trades):
    df = by_side(synth_trades)
    long_row = df[df["side"] == "long"].iloc[0]
    short_row = df[df["side"] == "short"].iloc[0]
    # Longs: 100+50+75+200-80 = 345
    assert long_row["total_pnl_gbp"] == pytest.approx(345.0)
    # Shorts: -60-40-25 = -125
    assert short_row["total_pnl_gbp"] == pytest.approx(-125.0)


def test_by_exit_reason(synth_trades):
    df = by_exit_reason(synth_trades)
    # All targets are wins
    target_row = df[df["exit_reason"] == "target"].iloc[0]
    assert target_row["wins"] == target_row["trades"]
    # All stops are losses
    stop_row = df[df["exit_reason"] == "stop"].iloc[0]
    assert stop_row["wins"] == 0


def test_by_session_phase():
    """Verify first_hour / middle / last_hour bucketing."""
    trades = pd.DataFrame([
        {"entry_time": pd.Timestamp("2024-01-01 08:15"),
         "side": "long", "net_pnl_gbp": 10.0, "exit_reason": "x"},
        {"entry_time": pd.Timestamp("2024-01-01 12:00"),
         "side": "long", "net_pnl_gbp": 20.0, "exit_reason": "x"},
        {"entry_time": pd.Timestamp("2024-01-01 15:30"),
         "side": "long", "net_pnl_gbp": -5.0, "exit_reason": "x"},
    ])
    df = by_session_phase(trades, open_hour=8, close_hour=16, phase_duration_hours=1.0)
    assert df[df["phase"] == "first_hour"]["trades"].iloc[0] == 1
    assert df[df["phase"] == "middle"]["trades"].iloc[0] == 1
    assert df[df["phase"] == "last_hour"]["trades"].iloc[0] == 1


def test_empty_trades_returns_empty_df():
    empty = pd.DataFrame(columns=["entry_time", "side", "net_pnl_gbp", "exit_reason"])
    # Should not crash, returns empty-ish DataFrame
    assert by_hour_of_day(empty).empty
    assert by_day_of_week(empty).empty
    assert by_month(empty).empty


def test_equity_drawdown_series():
    equity = pd.Series([100, 110, 105, 120, 90], index=pd.date_range("2024-01-01", periods=5))
    dd = equity_drawdown_series(equity)
    # Peak: 100 → 110 → 120
    # Drawdowns: 0, 0, -4.5% (105 vs 110), 0 (new peak), -25% (90 vs 120)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(0.0)
    assert dd.iloc[2] == pytest.approx(-4.545454, abs=0.001)  # 105/110 - 1
    assert dd.iloc[3] == pytest.approx(0.0)
    assert dd.iloc[4] == pytest.approx(-25.0)
