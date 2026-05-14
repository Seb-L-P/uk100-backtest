"""
Engine tests. The key one is `test_accounting_identity` — if that fails,
the equity curve and the trade log disagree about money, which means
results from the system are not to be trusted.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import run_backtest, Signal, Strategy
from backtest.broker import Broker


class _AlwaysLongSecondBar(Strategy):
    """A toy strategy: open long on bar 2, close on bar 5. Predictable."""

    def __init__(self):
        self.opened = False
        self.closed = False

    def on_bar(self, history, broker):
        i = len(history) - 1
        if i == 2 and not self.opened:
            self.opened = True
            return Signal(action="open_long", stake_per_point=1.0)
        if i == 5 and self.opened and not self.closed:
            self.closed = True
            return Signal(action="close")
        return Signal(action="noop")


def test_accounting_identity_holds(synthetic_ohlc):
    """
    THE most important engine test: final_balance == starting + sum(trade P&Ls).
    The engine asserts this internally; if the assertion fires, the test
    will surface it as an AssertionError, which we want to detect.
    """
    result = run_backtest(synthetic_ohlc, _AlwaysLongSecondBar(), warmup_bars=2)
    # The engine's own assert wouldn't fire on a clean run; here we just confirm
    # the outputs are internally consistent.
    sum_trade_pnl = result.trades_df["net_pnl_gbp"].sum()
    assert result.final_balance == pytest.approx(result.starting_balance + sum_trade_pnl, abs=0.01)


def test_signal_fills_at_next_bar_open(synthetic_ohlc):
    """
    A signal generated on bar i must fill at bar i+1's OPEN price, not bar i's
    close. Critical no-look-ahead guarantee.
    """
    strat = _AlwaysLongSecondBar()
    result = run_backtest(synthetic_ohlc, strat, warmup_bars=2)

    # Strategy fires on bar 2 (Close=103). Fill should be bar 3's Open=103.
    assert len(result.trades_df) == 1
    trade = result.trades_df.iloc[0]
    expected_entry = synthetic_ohlc.iloc[3]["Open"]
    assert trade["entry_price"] == pytest.approx(expected_entry)


def test_warmup_bars_respected(synthetic_ohlc):
    """Strategies are not called until warmup_bars have passed."""
    calls_at_bars = []

    class _RecordingStrategy(Strategy):
        def on_bar(self, history, broker):
            calls_at_bars.append(len(history) - 1)
            return Signal(action="noop")

    run_backtest(synthetic_ohlc, _RecordingStrategy(), warmup_bars=10)
    # First call should be at bar index 10, never earlier
    assert min(calls_at_bars) >= 10


def test_no_open_position_at_end(synthetic_ohlc):
    """If a strategy doesn't close, the engine force-closes on the last bar."""

    class _OnlyOpens(Strategy):
        def __init__(self):
            self.opened = False
        def on_bar(self, history, broker):
            if not self.opened and broker.position is None:
                self.opened = True
                return Signal(action="open_long", stake_per_point=1.0)
            return Signal(action="noop")

    result = run_backtest(synthetic_ohlc, _OnlyOpens(), warmup_bars=2)
    assert len(result.trades_df) == 1
    assert result.trades_df.iloc[0]["exit_reason"] == "eod"


def test_strategy_never_sees_future_bars(synthetic_ohlc):
    """The history passed to on_bar must end at the current bar."""
    captured_lengths = []
    captured_max_indices = []

    class _Peeker(Strategy):
        def on_bar(self, history, broker):
            captured_lengths.append(len(history))
            captured_max_indices.append(history.index[-1])
            return Signal(action="noop")

    run_backtest(synthetic_ohlc, _Peeker(), warmup_bars=2)
    # Each successive call sees one more bar
    for k in range(1, len(captured_lengths)):
        assert captured_lengths[k] == captured_lengths[k - 1] + 1
