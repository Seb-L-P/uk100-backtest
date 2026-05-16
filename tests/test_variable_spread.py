"""
Tests for the variable-spread feature.

Verify:
  1. When a position is opened with `entry_spread_pts`, that value is stored.
  2. When closed with both entry and exit spreads, the cost uses (entry+exit)/2.
  3. When opened/closed with no spread info, falls back to config.spread_points.
  4. Engine extracts `Spread` from data DataFrame and passes through correctly.
"""
from __future__ import annotations

from datetime import datetime
import pandas as pd
import pytest

from backtest.broker import Broker
from backtest.engine import run_backtest, Signal, Strategy
from config import COSTS


def test_entry_spread_stored_on_open():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    pos = broker.open("long", 1.0, t, 100.0, entry_spread_pts=2.0)
    assert pos.entry_spread_pts == pytest.approx(2.0)


def test_close_uses_average_of_entry_and_exit_spread():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    broker.open("long", 1.0, t, 100.0, entry_spread_pts=1.0)
    trade = broker.close(t1, 100.0, reason="signal", exit_spread_pts=3.0)
    # Average spread = (1.0 + 3.0) / 2 = 2.0; cost = 2.0 * stake 1.0 = 2.0
    assert trade.spread_cost_gbp == pytest.approx(2.0)


def test_close_falls_back_to_config_when_no_spread_info():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    broker.open("long", 1.0, t, 100.0)  # no entry spread
    trade = broker.close(t1, 100.0, reason="signal")  # no exit spread
    # Should use config.spread_points (1.5) * stake (1.0) = 1.5
    assert trade.spread_cost_gbp == pytest.approx(COSTS.spread_points * 1.0)


def test_close_uses_entry_spread_only_when_exit_unavailable():
    """When only the entry spread is known, use that for the round trip."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    broker.open("long", 1.0, t, 100.0, entry_spread_pts=2.5)
    trade = broker.close(t1, 100.0, reason="signal")  # no exit_spread
    assert trade.spread_cost_gbp == pytest.approx(2.5 * 1.0)


def test_engine_propagates_spread_from_data():
    """When data has a Spread column, the engine passes it to broker on entry/exit."""
    # Synthetic 5-bar DataFrame with a fixed Spread of 2.0 throughout
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105],
        "Low": [99, 100, 101, 102, 103],
        "Close": [101, 102, 103, 104, 105],
        "Volume": [1000] * 5,
        "Spread": [2.0] * 5,
    }, index=idx)

    class _OneShotLong(Strategy):
        def __init__(self):
            self.fired = False
        def on_bar(self, history, broker):
            i = len(history) - 1
            if i == 1 and not self.fired:
                self.fired = True
                return Signal(action="open_long", stake_per_point=1.0)
            return Signal(action="noop")

    result = run_backtest(data, _OneShotLong(), warmup_bars=1)
    assert len(result.trades_df) == 1
    # Spread cost = (2.0 entry + 2.0 exit) / 2 * stake 1.0 = 2.0
    # NOT the config 1.5 — the per-bar spread overrode it.
    assert result.trades_df.iloc[0]["spread_cost_gbp"] == pytest.approx(2.0)


def test_engine_falls_back_when_no_spread_column():
    """yfinance-style data (no Spread column) → engine uses config.spread_points."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105],
        "Low": [99, 100, 101, 102, 103],
        "Close": [101, 102, 103, 104, 105],
        "Volume": [1000] * 5,
        # No Spread column
    }, index=idx)

    class _OneShotLong(Strategy):
        def __init__(self):
            self.fired = False
        def on_bar(self, history, broker):
            i = len(history) - 1
            if i == 1 and not self.fired:
                self.fired = True
                return Signal(action="open_long", stake_per_point=1.0)
            return Signal(action="noop")

    result = run_backtest(data, _OneShotLong(), warmup_bars=1)
    assert len(result.trades_df) == 1
    # Falls back to config.spread_points (1.5) * stake 1.0 = 1.5
    assert result.trades_df.iloc[0]["spread_cost_gbp"] == pytest.approx(COSTS.spread_points)
