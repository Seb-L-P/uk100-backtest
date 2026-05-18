"""
Tests for variable slippage modelling.

When the bar's bid/ask spread is known (IG data), slippage scales as
  slippage = max(min_slippage_points, slip_spread_multiplier × spread)

When unknown (yfinance), falls back to the flat `slippage_points` config.
"""
from __future__ import annotations

from datetime import datetime
import pytest

from backtest.broker import Broker
from config import COSTS


# ---- Cost-model helper ------------------------------------------------
def test_effective_slippage_falls_back_when_no_spread():
    assert COSTS.effective_slippage_pts(None) == pytest.approx(COSTS.slippage_points)
    assert COSTS.effective_slippage_pts(0) == pytest.approx(COSTS.slippage_points)


def test_effective_slippage_scales_with_spread():
    # spread of 2pt × multiplier 0.5 = 1pt slippage (above min floor of 0.2)
    s = COSTS.effective_slippage_pts(2.0)
    assert s == pytest.approx(2.0 * COSTS.slip_spread_multiplier)


def test_effective_slippage_respects_min_floor():
    # Tiny spread (0.1pt) × 0.5 = 0.05pt — below min floor of 0.2 → clamped to 0.2
    s = COSTS.effective_slippage_pts(0.1)
    assert s == pytest.approx(COSTS.min_slippage_points)


def test_effective_slippage_grows_with_wide_spread():
    # Wide news-bar spread of 5pt → slippage = 2.5pt
    s = COSTS.effective_slippage_pts(5.0)
    assert s == pytest.approx(2.5)


# ---- Broker stop close uses variable slippage -------------------------
def test_stop_close_uses_variable_slippage_when_spread_known():
    """A stop closed with exit_spread_pts=4.0 should use scaled slippage."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    broker.open("long", 1.0, t, 100.0, entry_spread_pts=4.0)
    trade = broker.close(t1, 95.0, reason="stop", exit_spread_pts=4.0)

    # Slippage at spread=4.0 → 4.0 × 0.5 = 2.0pt × stake 1.0 = £2.0
    assert trade.slippage_cost_gbp == pytest.approx(2.0)


def test_stop_close_falls_back_to_fixed_slippage_no_spread():
    """When exit_spread_pts is None, broker uses the fixed config slippage."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    broker.open("long", 1.0, t, 100.0)
    trade = broker.close(t1, 95.0, reason="stop")  # no exit_spread

    # Fall back to fixed 0.5pt × stake 1.0 = £0.5
    assert trade.slippage_cost_gbp == pytest.approx(COSTS.slippage_points)


def test_signal_exit_has_zero_slippage_regardless():
    """Signal exits (not 'stop' or 'market') get zero slippage — variable doesn't apply."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    broker.open("long", 1.0, t, 100.0)
    trade = broker.close(t1, 105.0, reason="signal", exit_spread_pts=4.0)
    assert trade.slippage_cost_gbp == 0.0


# ---- Stop-order fills use variable slippage ---------------------------
def test_stop_buy_order_uses_variable_slippage():
    """When a stop buy fires, fill price = trigger + variable slippage."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="long", order_type="stop", trigger_price=100.0,
        stake_per_point=1.0, time=t,
    )
    # Wide bar: spread 4pt → slippage 2pt
    bar = {"Open": 99, "High": 101, "Low": 99, "Close": 100.5, "Spread": 4.0}
    opened = broker.check_pending_orders(t, bar, bar_spread=4.0)

    # Stop fills at 100 + 2.0pt slippage = 102
    assert len(opened) == 1
    assert opened[0].entry_price == pytest.approx(102.0)


def test_stop_sell_order_uses_variable_slippage():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="short", order_type="stop", trigger_price=100.0,
        stake_per_point=1.0, time=t,
    )
    bar = {"Open": 101, "High": 101, "Low": 99, "Close": 99.5, "Spread": 4.0}
    opened = broker.check_pending_orders(t, bar, bar_spread=4.0)
    # Stop fills at 100 - 2.0pt = 98
    assert opened[0].entry_price == pytest.approx(98.0)


def test_limit_orders_never_apply_slippage():
    """
    Limit orders never get slippage. They fill at the trigger when the bar
    reaches the trigger from the unfavourable side, OR at bar.Open when the
    bar opens past the trigger in the FAVOURABLE direction (better price —
    you'd never pay the trigger when the market opens better).
    """
    # --- Case 1: bar wandered DOWN to the long limit from above ---
    # Open above the limit means we need bar.Low to reach the trigger.
    # Fill should be exactly at the trigger, with no slippage.
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="long", order_type="limit", trigger_price=100.0,
        stake_per_point=1.0, time=t,
    )
    bar = {"Open": 102, "High": 103, "Low": 99, "Close": 101, "Spread": 4.0}
    opened = broker.check_pending_orders(t, bar, bar_spread=4.0)
    assert opened[0].entry_price == pytest.approx(100.0)

    # --- Case 2: bar opens BELOW the long limit (favourable) ---
    # The market opened at a price BETTER than our limit — a real broker
    # would fill us immediately at the gap-open price (99), not at the
    # limit (100). No slippage applied either way.
    broker = Broker()
    broker.place_pending_order(
        side="long", order_type="limit", trigger_price=100.0,
        stake_per_point=1.0, time=t,
    )
    bar = {"Open": 99, "High": 101, "Low": 95, "Close": 100, "Spread": 4.0}
    opened = broker.check_pending_orders(t, bar, bar_spread=4.0)
    assert opened[0].entry_price == pytest.approx(99.0)
