"""
Broker accounting tests. The most important tests in the suite — if these
fail, every backtest is suspect.
"""
from __future__ import annotations

from datetime import datetime
import pytest

from backtest.broker import Broker
from config import CostModel, AccountConfig


@pytest.fixture
def broker() -> Broker:
    """Fresh broker with default cost model."""
    return Broker()


def test_open_then_close_long_profitable(broker):
    """Open long, close at higher price → net positive after costs."""
    t0 = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 11, 0)
    broker.open("long", stake_per_point=2.0, time=t0, price=100.0)
    trade = broker.close(t1, 110.0, reason="signal")

    # Gross: (110 - 100) * 2.0 = 20.0
    assert trade.gross_pnl_gbp == pytest.approx(20.0)
    # Spread cost: 1.5pt * 2.0 = 3.0
    assert trade.spread_cost_gbp == pytest.approx(3.0)
    # Signal exits don't get slippage
    assert trade.slippage_cost_gbp == 0.0
    # Same-day close → financing = 0
    assert trade.financing_cost_gbp == 0.0
    # Net = gross - spread - slippage - financing = 20 - 3 - 0 - 0 = 17.0
    assert trade.net_pnl_gbp == pytest.approx(17.0)


def test_open_then_close_short_profitable(broker):
    """Short at 100, cover at 90 → 10pt gain × stake."""
    t0 = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 11, 0)
    broker.open("short", stake_per_point=1.0, time=t0, price=100.0)
    trade = broker.close(t1, 90.0, reason="signal")

    assert trade.gross_pnl_gbp == pytest.approx(10.0)
    assert trade.spread_cost_gbp == pytest.approx(1.5)
    assert trade.net_pnl_gbp == pytest.approx(8.5)


def test_stop_exit_includes_slippage(broker):
    """Stop fills incur slippage cost."""
    t0 = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 10, 0)
    broker.open("long", stake_per_point=1.0, time=t0, price=100.0, stop_loss=95.0)
    trade = broker.close(t1, 95.0, reason="stop")

    # Gross: (95 - 100) * 1.0 = -5
    # Spread: 1.5; slippage: 0.5
    # Net: -5 - 1.5 - 0.5 = -7.0
    assert trade.gross_pnl_gbp == pytest.approx(-5.0)
    assert trade.slippage_cost_gbp == pytest.approx(0.5)
    assert trade.net_pnl_gbp == pytest.approx(-7.0)


def test_balance_updates_correctly(broker):
    """After a trade, balance moves by net P&L."""
    starting = broker.balance
    t0 = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 11, 0)
    broker.open("long", 2.0, t0, 100.0)
    trade = broker.close(t1, 110.0, reason="signal")
    assert broker.balance == pytest.approx(starting + trade.net_pnl_gbp)


def test_cannot_open_with_position_already_open(broker):
    """Should refuse to open if a position is already open."""
    t0 = datetime(2024, 1, 1, 9, 0)
    broker.open("long", 1.0, t0, 100.0)
    with pytest.raises(RuntimeError):
        broker.open("long", 1.0, t0, 100.0)


def test_cannot_close_when_no_position(broker):
    """Should refuse to close if no position is open."""
    t0 = datetime(2024, 1, 1, 9, 0)
    with pytest.raises(RuntimeError):
        broker.close(t0, 100.0, reason="signal")


def test_leverage_cap_enforced(broker):
    """A stake exceeding leverage cap should raise."""
    t0 = datetime(2024, 1, 1, 9, 0)
    # Account has £10k starting balance, leverage cap 20x → max notional £200k.
    # Stake £5/pt × price 100,000 = £500k notional (way over).
    with pytest.raises(ValueError):
        broker.open("long", stake_per_point=5.0, time=t0, price=100_000.0)


def test_overnight_financing_charged(broker):
    """A position held overnight accrues financing."""
    t0 = datetime(2024, 1, 1, 14, 0)
    t1 = datetime(2024, 1, 2, 14, 0)  # 1 day later
    broker.open("long", 1.0, t0, 1000.0)
    # Mark on a later bar to trigger financing
    broker.mark(t1, {"Open": 1000.0, "High": 1010.0, "Low": 990.0, "Close": 1000.0})
    trade = broker.close(t1, 1000.0, reason="signal")

    # Financing for 1 day on notional £1000 long at (SONIA 5.25% + 2.5%) / 365
    # = 1000 * 0.0775 / 365 ≈ £0.2123
    assert trade.financing_cost_gbp == pytest.approx(0.2123, abs=0.01)


def test_pnl_accounting_identity_per_trade(broker):
    """net_pnl == gross - spread - slippage - financing, always."""
    t0 = datetime(2024, 1, 1, 9, 0)
    t1 = datetime(2024, 1, 1, 11, 0)
    broker.open("short", 3.0, t0, 200.0)
    trade = broker.close(t1, 180.0, reason="stop")
    expected = (trade.gross_pnl_gbp - trade.spread_cost_gbp
                - trade.slippage_cost_gbp - trade.financing_cost_gbp)
    assert trade.net_pnl_gbp == pytest.approx(expected)
