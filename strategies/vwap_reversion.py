"""
VWAP mean reversion — day-trade.

Rules:
  1. Compute session-anchored VWAP (resets each day).
  2. When price stretches above VWAP by more than `entry_stretch_pts` →
     enter SHORT (fade back to VWAP).
  3. When price stretches below by more than `entry_stretch_pts` → enter LONG.
  4. Stop: ATR-based, beyond the entry bar's extreme.
  5. Exit: when price touches VWAP again (target = VWAP).
  6. UK session only, flat by close, one position at a time, max one trade
     per day to avoid revenge trading.

Why it might work:
  - VWAP is widely used by institutional algos as a benchmark — large
    flows mean-revert around it.
  - Intraday FTSE 100 often oscillates around VWAP in non-news sessions.

Why it might not:
  - On trending days price stays one side of VWAP all day.
  - Yahoo's volume is exchange volume, not broker flow — VWAP is approximate.
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import vwap as vwap_indicator, atr
from strategies._helpers import risk_based_stake, in_session


class VwapReversion:
    def __init__(
        self,
        entry_stretch_pts: float = 15.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        stop_buffer_pts: float = 1.0,
        max_trades_per_day: int = 1,
        session_open: time = time(8, 30),
        session_close: time = time(15, 0),
        flat_by: time = time(15, 30),
    ):
        self.entry_stretch_pts = entry_stretch_pts
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.stop_buffer_pts = stop_buffer_pts
        self.max_trades_per_day = max_trades_per_day
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by

        # Day state
        self._day_date = None
        self._trades_today = 0

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        bar = history.iloc[i]
        ts = history.index[i]
        now = ts.time()

        # Day reset
        if ts.date() != self._day_date:
            self._day_date = ts.date()
            self._trades_today = 0

        if i < self.atr_period + 1:
            return Signal(action="noop")

        # Force-flat near end of day
        if broker.position is not None and now >= self.flat_by:
            return Signal(action="close", reason="session_end")

        cur_close = float(bar["Close"])
        cur_vwap = float(vwap_indicator(history).iloc[-1])

        # Exit: in a position and price has reverted to VWAP
        if broker.position is not None:
            if broker.position.side == "long" and cur_close >= cur_vwap:
                return Signal(action="close", reason="vwap_target")
            if broker.position.side == "short" and cur_close <= cur_vwap:
                return Signal(action="close", reason="vwap_target")
            return Signal(action="noop")

        if not in_session(now, self.session_open, self.session_close):
            return Signal(action="noop")
        if self._trades_today >= self.max_trades_per_day:
            return Signal(action="noop")

        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")

        stretch = cur_close - cur_vwap
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])

        if stretch > self.entry_stretch_pts:
            stop = bar_high + self.atr_stop_mult * atr_now + self.stop_buffer_pts
            risk = stop - cur_close
            if risk <= 0:
                return Signal(action="noop")
            stake = risk_based_stake(broker.balance, risk, price=cur_close)
            self._trades_today += 1
            return Signal(action="open_short", stake_per_point=stake,
                          stop_loss=stop, take_profit=cur_vwap,
                          reason=f"VWAP stretch +{stretch:.1f}pts")
        if -stretch > self.entry_stretch_pts:
            stop = bar_low - self.atr_stop_mult * atr_now - self.stop_buffer_pts
            risk = cur_close - stop
            if risk <= 0:
                return Signal(action="noop")
            stake = risk_based_stake(broker.balance, risk, price=cur_close)
            self._trades_today += 1
            return Signal(action="open_long", stake_per_point=stake,
                          stop_loss=stop, take_profit=cur_vwap,
                          reason=f"VWAP stretch {stretch:.1f}pts")

        return Signal(action="noop")
