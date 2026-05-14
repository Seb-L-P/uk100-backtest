"""
Tests for the new multi-position / scale-out / trailing-stop capabilities.

Existing single-position tests in test_broker.py still pass via the
backward-compat .position property + the close() dispatching that detects
legacy (time, price) calling conventions.
"""
from __future__ import annotations

from datetime import datetime
import pytest

from backtest.broker import Broker, OpenPosition
from backtest.exits import atr_trailing, breakeven_after_R, combine
from config import AccountConfig, COSTS


@pytest.fixture
def multi_broker() -> Broker:
    """Broker that allows multiple concurrent positions for these tests."""
    cfg = AccountConfig(
        starting_balance_gbp=100_000.0,  # bigger float for multi-position notional
        max_concurrent_positions=5,
    )
    return Broker(account=cfg)


# ---- Multi-position basics ---------------------------------------------
def test_open_two_positions(multi_broker):
    """Two open() calls produce two distinct positions."""
    t = datetime(2024, 1, 1, 9, 0)
    p1 = multi_broker.open("long", 1.0, t, 100.0)
    p2 = multi_broker.open("short", 0.5, t, 100.0)
    assert len(multi_broker.positions) == 2
    assert p1.id != p2.id
    assert p1.side == "long"
    assert p2.side == "short"


def test_position_property_returns_none_with_multi(multi_broker):
    """Backward-compat .position returns None when more than 1 is open."""
    t = datetime(2024, 1, 1, 9, 0)
    multi_broker.open("long", 1.0, t, 100.0)
    multi_broker.open("long", 1.0, t, 100.0)
    assert multi_broker.position is None


def test_position_property_returns_single_position(multi_broker):
    """With exactly 1 open, .position gives it back (legacy strategies still work)."""
    t = datetime(2024, 1, 1, 9, 0)
    p = multi_broker.open("long", 1.0, t, 100.0)
    assert multi_broker.position is p


def test_close_specific_position(multi_broker):
    """close(position_id, time, price, reason=) closes only that position."""
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    p1 = multi_broker.open("long", 1.0, t, 100.0)
    p2 = multi_broker.open("long", 0.5, t, 100.0)
    multi_broker.close(p1.id, t1, 105.0, reason="signal")
    assert len(multi_broker.positions) == 1
    assert multi_broker.positions[0].id == p2.id


def test_close_all(multi_broker):
    """close_all closes every open position and returns trades."""
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    multi_broker.open("long", 1.0, t, 100.0)
    multi_broker.open("short", 1.0, t, 100.0)
    trades = multi_broker.close_all(t1, 100.0, reason="eod")
    assert len(trades) == 2
    assert multi_broker.positions == []


def test_leverage_cap_includes_existing_positions(multi_broker):
    """Opening a 2nd position that breaches total leverage should raise."""
    t = datetime(2024, 1, 1, 9, 0)
    # 100k account, 20x cap = 2M max notional
    multi_broker.open("long", 10.0, t, 100_000.0)  # 1M notional, OK
    with pytest.raises(ValueError):
        multi_broker.open("long", 11.0, t, 100_000.0)  # +1.1M = 2.1M, breaches


# ---- Backward-compat close (legacy single-position calls) ----------------
def test_legacy_close_with_one_position():
    """Legacy broker.close(time, price, reason=) still works with one position."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    broker.open("long", 1.0, t, 100.0)
    trade = broker.close(t1, 110.0, reason="signal")
    assert trade.gross_pnl_gbp == pytest.approx(10.0)


def test_legacy_close_fails_with_zero_or_multi(multi_broker):
    """Legacy close call should refuse if there's no unique position."""
    t = datetime(2024, 1, 1, 9, 0)
    multi_broker.open("long", 1.0, t, 100.0)
    multi_broker.open("long", 0.5, t, 100.0)
    with pytest.raises(RuntimeError):
        multi_broker.close(t, 100.0, reason="x")  # ambiguous


# ---- Scale-out (partial exits) -----------------------------------------
def test_scale_out_half():
    """Closing 50% of a position records a half-stake trade and leaves half open."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    pos = broker.open("long", 2.0, t, 100.0)
    trade = broker.scale_out(pos, t1, 110.0, fraction=0.5, reason="1R")

    # Trade was for 1.0 stake (half of 2.0), gross = (110-100)*1.0 = 10
    assert trade.stake_per_point == pytest.approx(1.0)
    assert trade.gross_pnl_gbp == pytest.approx(10.0)
    # Position still open with 1.0 remaining
    assert len(broker.positions) == 1
    assert broker.positions[0].remaining_stake_per_point == pytest.approx(1.0)


def test_scale_out_then_close_full():
    """After scaling out 50%, closing the rest produces a second trade for the remainder."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    t2 = datetime(2024, 1, 1, 11, 0)
    pos = broker.open("long", 2.0, t, 100.0)
    broker.scale_out(pos, t1, 110.0, fraction=0.5, reason="1R")
    trade2 = broker.close(pos.id, t2, 120.0, reason="target")

    # Second trade: 1.0 stake, gross = (120-100)*1.0 = 20
    assert trade2.stake_per_point == pytest.approx(1.0)
    assert trade2.gross_pnl_gbp == pytest.approx(20.0)
    # Position fully closed
    assert broker.positions == []
    # Two trades recorded total
    assert len(broker.trades) == 2


def test_scale_out_invalid_fraction(multi_broker):
    """fraction must be in (0, 1]."""
    t = datetime(2024, 1, 1, 9, 0)
    pos = multi_broker.open("long", 1.0, t, 100.0)
    with pytest.raises(ValueError):
        multi_broker.scale_out(pos, t, 100.0, fraction=0.0)
    with pytest.raises(ValueError):
        multi_broker.scale_out(pos, t, 100.0, fraction=1.5)


def test_pnl_identity_after_scale_out(multi_broker):
    """sum(trade net_pnl) + remaining unrealized should reconcile."""
    t = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    t2 = datetime(2024, 1, 1, 11, 0)
    starting = multi_broker.balance
    pos = multi_broker.open("long", 4.0, t, 100.0)
    multi_broker.scale_out(pos, t1, 110.0, fraction=0.5, reason="1R")
    multi_broker.close(pos.id, t2, 120.0, reason="target")
    realised = sum(tr.net_pnl_gbp for tr in multi_broker.trades)
    assert multi_broker.balance == pytest.approx(starting + realised, abs=0.01)


# ---- Trailing stops -----------------------------------------------------
def test_trailing_stop_ratchets_up_for_long():
    """ATR trailing stop on a long should never lower the stop."""
    import pandas as pd
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    # Build a minimal history for atr_trailing's initial snapshot
    history = pd.DataFrame({
        "Open": [100, 101, 102, 103, 104],
        "High": [101, 102, 103, 104, 105],
        "Low": [99, 100, 101, 102, 103],
        "Close": [101, 102, 103, 104, 105],
        "Volume": [1000] * 5,
    }, index=pd.date_range("2024-01-01", periods=5, freq="D"))
    trail = atr_trailing(history, atr_period=3, mult=2.0)

    pos = broker.open("long", 1.0, t, 105.0, stop_loss=100.0, trailing_stop_fn=trail)
    initial_stop = pos.stop_loss

    # Mark a bunch of bars with rising prices
    for close in [106, 108, 110, 112, 115]:
        broker.mark(datetime(2024, 1, 2, 9, 0), {
            "Open": close - 1, "High": close + 1, "Low": close - 1.5, "Close": close,
        })

    # Stop should have moved UP (ratcheted), never down
    assert pos.stop_loss > initial_stop


def test_trailing_stop_does_not_widen():
    """Trailing function returning a worse stop is silently ignored."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)

    # Trail function that always returns a stop of 50 (way below entry)
    def bad_trail(position, bar):
        return 50.0

    pos = broker.open("long", 1.0, t, 100.0, stop_loss=95.0, trailing_stop_fn=bad_trail)
    broker.mark(t, {"Open": 100, "High": 101, "Low": 99, "Close": 100})
    # Stop should still be 95, NOT 50
    assert pos.stop_loss == pytest.approx(95.0)


def test_breakeven_after_R():
    """Stop moves to entry after price travels 1R; doesn't move further."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    risk_pts = 5.0  # entry 100, stop 95
    trail = breakeven_after_R(risk_pts=risk_pts, move_to_R=1.0)
    pos = broker.open("long", 1.0, t, 100.0, stop_loss=95.0, trailing_stop_fn=trail)

    # First bar: high = 102 (only 0.4R) — stop unchanged
    broker.mark(t, {"Open": 100, "High": 102, "Low": 99, "Close": 101})
    assert pos.stop_loss == pytest.approx(95.0)

    # Second bar: high = 106 (1.2R) — stop should ratchet to 100 (entry)
    broker.mark(t, {"Open": 101, "High": 106, "Low": 100, "Close": 105})
    assert pos.stop_loss == pytest.approx(100.0)

    # Third bar: high goes higher, but breakeven function returns None now
    broker.mark(t, {"Open": 105, "High": 110, "Low": 104, "Close": 108})
    # Stop stays at 100 (no further movement from breakeven function)
    assert pos.stop_loss == pytest.approx(100.0)


def test_combine_chains_trailing_functions():
    """combine() runs functions in order; first non-None wins."""
    broker = Broker()
    t = datetime(2024, 1, 1, 9, 0)
    risk_pts = 5.0
    breakeven = breakeven_after_R(risk_pts=risk_pts, move_to_R=1.0)

    # Hand-built ATR trail that always returns close - 2
    def simple_atr(position, bar):
        return bar["Close"] - 2.0

    combined = combine(breakeven, simple_atr)

    pos = broker.open("long", 1.0, t, 100.0, stop_loss=95.0, trailing_stop_fn=combined)
    # First bar: small move — breakeven returns None, simple_atr returns 99
    broker.mark(t, {"Open": 100, "High": 101, "Low": 99, "Close": 101})
    # Stop wants to be 99 but original was 95 — ratchets up to 99
    assert pos.stop_loss == pytest.approx(99.0)

    # Big move bar — breakeven fires, returns 100. simple_atr would have returned 108.
    # combine returns first non-None → 100.
    broker.mark(t, {"Open": 101, "High": 110, "Low": 100, "Close": 110})
    assert pos.stop_loss == pytest.approx(100.0)
