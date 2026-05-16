"""
Tests for pending limit / stop orders.

Covers placement, trigger conditions, fill prices (limit at trigger, stop with
slippage), expiry, cancellation, and end-to-end engine flow.
"""
from __future__ import annotations

from datetime import datetime
import pandas as pd
import pytest

from backtest.broker import Broker, PendingOrder
from backtest.engine import run_backtest, Signal, Strategy
from config import COSTS


# ---- Placement + identity ----------------------------------------------
def test_place_limit_returns_pending_order():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    order = broker.place_pending_order(
        side="long", order_type="limit",
        trigger_price=100.0, stake_per_point=1.0, time=t,
    )
    assert isinstance(order, PendingOrder)
    assert order.id.startswith("o")
    assert len(broker.pending_orders) == 1


def test_unknown_order_type_raises():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    with pytest.raises(ValueError):
        broker.place_pending_order(
            side="long", order_type="market",  # market is direct, not pending
            trigger_price=100.0, stake_per_point=1.0, time=t,
        )


# ---- Triggering: limit orders -----------------------------------------
def test_limit_buy_fills_at_trigger_price_when_bar_low_reaches():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    order = broker.place_pending_order(
        side="long", order_type="limit",
        trigger_price=100.0, stake_per_point=1.0, time=t,
    )
    # Bar range: 99–101, so low=99 ≤ trigger 100 → should fill at 100
    bar = {"Open": 101, "High": 101, "Low": 99, "Close": 100}
    opened = broker.check_pending_orders(t, bar)
    assert len(opened) == 1
    assert opened[0].entry_price == pytest.approx(100.0)
    assert broker.pending_orders == []


def test_limit_buy_does_not_fill_if_low_above_trigger():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="long", order_type="limit",
        trigger_price=100.0, stake_per_point=1.0, time=t,
    )
    bar = {"Open": 105, "High": 106, "Low": 101, "Close": 104}
    opened = broker.check_pending_orders(t, bar)
    assert opened == []
    assert len(broker.pending_orders) == 1  # still pending


def test_limit_sell_fills_when_high_reaches_trigger():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="short", order_type="limit",
        trigger_price=100.0, stake_per_point=1.0, time=t,
    )
    # Need high ≥ 100; bar 99-101 satisfies
    bar = {"Open": 99, "High": 101, "Low": 99, "Close": 100}
    opened = broker.check_pending_orders(t, bar)
    assert len(opened) == 1
    assert opened[0].entry_price == pytest.approx(100.0)


# ---- Triggering: stop orders ------------------------------------------
def test_stop_buy_fills_at_trigger_plus_slippage():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="long", order_type="stop",
        trigger_price=100.0, stake_per_point=1.0, time=t,
    )
    bar = {"Open": 99, "High": 101, "Low": 99, "Close": 100.5}
    opened = broker.check_pending_orders(t, bar)
    assert len(opened) == 1
    # Stop buy fills at trigger + slippage (0.5 in config)
    assert opened[0].entry_price == pytest.approx(100.0 + COSTS.slippage_points)


def test_stop_sell_fills_at_trigger_minus_slippage():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="short", order_type="stop",
        trigger_price=100.0, stake_per_point=1.0, time=t,
    )
    bar = {"Open": 101, "High": 101, "Low": 99, "Close": 99.5}
    opened = broker.check_pending_orders(t, bar)
    assert len(opened) == 1
    assert opened[0].entry_price == pytest.approx(100.0 - COSTS.slippage_points)


# ---- Expiry + cancellation --------------------------------------------
def test_expires_after_bars():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="long", order_type="limit",
        trigger_price=50.0, stake_per_point=1.0, time=t,
        expires_after_bars=2,
    )
    # Bar 1: too far from trigger, not filled. bars_alive becomes 1.
    bar = {"Open": 100, "High": 101, "Low": 99, "Close": 100}
    broker.check_pending_orders(t, bar)
    assert len(broker.pending_orders) == 1
    # Bar 2: bars_alive becomes 2, still alive (≤ expires).
    broker.check_pending_orders(t, bar)
    assert len(broker.pending_orders) == 1
    # Bar 3: bars_alive becomes 3 (> 2) → expires.
    broker.check_pending_orders(t, bar)
    assert broker.pending_orders == []


def test_cancel_pending_order_by_id():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    order = broker.place_pending_order(
        side="long", order_type="limit",
        trigger_price=100.0, stake_per_point=1.0, time=t,
    )
    assert broker.cancel_pending_order(order.id) is True
    assert broker.pending_orders == []


def test_cancel_pending_order_unknown_returns_false():
    broker = Broker()
    assert broker.cancel_pending_order("nonexistent") is False


def test_cancel_all_pending():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(side="long", order_type="limit",
                               trigger_price=100.0, stake_per_point=1.0, time=t)
    broker.place_pending_order(side="short", order_type="limit",
                               trigger_price=110.0, stake_per_point=1.0, time=t)
    assert broker.cancel_all_pending() == 2
    assert broker.pending_orders == []


# ---- Triggered order carries through stop_loss + take_profit ----------
def test_filled_order_keeps_stop_and_target():
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    broker.place_pending_order(
        side="long", order_type="limit",
        trigger_price=100.0, stake_per_point=1.0, time=t,
        stop_loss=95.0, take_profit=110.0,
    )
    bar = {"Open": 101, "High": 101, "Low": 99, "Close": 100}
    opened = broker.check_pending_orders(t, bar)
    assert opened[0].stop_loss == pytest.approx(95.0)
    assert opened[0].take_profit == pytest.approx(110.0)


# ---- End-to-end engine flow -------------------------------------------
def test_engine_executes_limit_order_strategy():
    """
    Strategy places a limit-buy order at bar 1. The order should sit pending,
    then fill on a later bar when price drops to the limit, then close cleanly
    at end of backtest.
    """
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    # Bars: high prices, then a dip that touches our limit at 100
    data = pd.DataFrame({
        "Open":  [105, 106, 107, 108, 102, 110, 111, 112, 113, 114],
        "High":  [106, 107, 108, 109, 103, 111, 112, 113, 114, 115],
        "Low":   [104, 105, 106, 107, 100, 109, 110, 111, 112, 113],  # bar 4 dips to 100
        "Close": [106, 107, 108, 108, 103, 110, 111, 112, 113, 114],
        "Volume": [1000] * 10,
    }, index=idx)

    class _LimitBuyStrategy(Strategy):
        def __init__(self):
            self.placed = False
        def on_bar(self, history, broker):
            if not self.placed:
                self.placed = True
                return Signal(
                    action="open_long",
                    order_type="limit",
                    limit_price=100.0,
                    stake_per_point=1.0,
                )
            return Signal(action="noop")

    result = run_backtest(data, _LimitBuyStrategy(), warmup_bars=1)
    # Should have exactly 1 trade: limit filled at 100 on bar 4, EOD closed at 114 on bar 9
    assert len(result.trades_df) == 1
    trade = result.trades_df.iloc[0]
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["side"] == "long"


def test_engine_cancel_all_orders_via_signal():
    """Strategy can cancel its own pending orders via a cancel_all_orders signal."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame({
        "Open": [100, 100, 100, 100, 100],
        "High": [101, 101, 101, 101, 101],
        "Low":  [99, 99, 99, 99, 99],
        "Close": [100, 100, 100, 100, 100],
        "Volume": [1000] * 5,
    }, index=idx)

    class _PlaceThenCancel(Strategy):
        def __init__(self):
            self.step = 0
        def on_bar(self, history, broker):
            self.step += 1
            if self.step == 1:
                return Signal(action="open_long", order_type="limit",
                              limit_price=50.0, stake_per_point=1.0)  # won't fill
            if self.step == 2:
                return Signal(action="cancel_all_orders")
            return Signal(action="noop")

    result = run_backtest(data, _PlaceThenCancel(), warmup_bars=1)
    # No trades because the order was cancelled before it could fill
    assert len(result.trades_df) == 0
