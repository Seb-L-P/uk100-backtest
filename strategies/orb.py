"""
Opening Range Breakout (ORB) — day-trade only.

Rules:
  1. Define the "opening range" as the high and low of the first
     `opening_range_bars` bars of the trading day (e.g. 2 bars on 15m =
     first 30 minutes).
  2. After the opening range is set, wait for price to break above the high
     (go LONG) or below the low (go SHORT).
  3. Stop: on the OTHER side of the opening range (long stops at OR low,
     short stops at OR high), with an ATR-relative cushion (`stop_buffer_atr_mult`).
  4. Target: `r_target` × risk distance.
  5. One trade max per day (don't keep retrying after a stop-out).
  6. Flat by `flat_by` time regardless.
  7. UK session: opens at 08:00 LSE.

Why it might work:
  - Volatility is concentrated in the first 30 minutes of cash open.
  - A break of that range = the day has "picked a direction".
  - Well-documented historical edge in US indices (less so in UK).

Why it might not:
  - The edge has been arbitraged away in liquid indices.
  - Fakeouts at the OR are common in chop.
  - Spread + slippage eat small breakouts.
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from strategies._helpers import risk_based_stake, in_session, is_first_bar_of_day


class OpeningRangeBreakout:
    def __init__(
        self,
        opening_range_bars: int = 2,        # 2 bars on 15m = first 30 min
        r_target: float = 1.5,
        stop_buffer_atr_mult: float = 0.2,
        atr_period: int = 14,
        session_open: time = time(8, 0),    # FTSE 100 cash open
        session_close: time = time(15, 30), # latest entry
        flat_by: time = time(16, 0),
    ):
        self.opening_range_bars = int(opening_range_bars)
        self.r_target = r_target
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.atr_period = int(atr_period)
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by

        # Daily state — reset every new trading day
        self._day_date = None
        self._day_bars_seen = 0
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._traded_today = False

    def _reset_day(self):
        self._day_bars_seen = 0
        self._or_high = None
        self._or_low = None
        self._traded_today = False

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        bar = history.iloc[i]
        ts = history.index[i]
        now = ts.time()

        # Reset state at start of each new day
        if ts.date() != self._day_date:
            self._reset_day()
            self._day_date = ts.date()

        self._day_bars_seen += 1

        # Force-flat near end of day
        if broker.position is not None and now >= self.flat_by:
            return Signal(action="close", reason="session_end")

        # Skip if outside session, or already traded today, or already in a position
        if not in_session(now, self.session_open, self.session_close):
            return Signal(action="noop")
        if broker.position is not None:
            return Signal(action="noop")
        if self._traded_today:
            return Signal(action="noop")

        # Build the opening range as the first N bars
        if self._day_bars_seen <= self.opening_range_bars:
            # Track the running high/low across the OR window
            high_so_far = float(bar["High"])
            low_so_far = float(bar["Low"])
            self._or_high = high_so_far if self._or_high is None else max(self._or_high, high_so_far)
            self._or_low = low_so_far if self._or_low is None else min(self._or_low, low_so_far)
            return Signal(action="noop")

        # OR window is closed; look for a break
        if self._or_high is None or self._or_low is None:
            return Signal(action="noop")

        from strategies._helpers import atr_threshold
        stop_buffer = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)
        close = float(bar["Close"])
        if close > self._or_high:
            # Bullish breakout
            entry = close
            stop = self._or_low - stop_buffer
            risk = entry - stop
            target = entry + self.r_target * risk
            stake = risk_based_stake(broker.balance, risk, price=entry)
            self._traded_today = True
            return Signal(action="open_long", stake_per_point=stake,
                          stop_loss=stop, take_profit=target,
                          reason=f"ORB up: OR={self._or_low:.1f}-{self._or_high:.1f}")
        if close < self._or_low:
            entry = close
            stop = self._or_high + stop_buffer
            risk = stop - entry
            target = entry - self.r_target * risk
            stake = risk_based_stake(broker.balance, risk, price=entry)
            self._traded_today = True
            return Signal(action="open_short", stake_per_point=stake,
                          stop_loss=stop, take_profit=target,
                          reason=f"ORB down: OR={self._or_low:.1f}-{self._or_high:.1f}")

        return Signal(action="noop")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """
        For ensemble polling. Recomputes today's opening range from scratch
        (stateless) — finds today's bars, takes first N as the OR, checks if
        current close breaks it.
        """
        i = len(history) - 1
        if i < self.opening_range_bars + 1:
            return "none"
        today = history.index[i].date()
        # Find the first N bars of today's session
        today_mask = history.index.date == today
        today_bars = history.loc[today_mask]
        if len(today_bars) <= self.opening_range_bars:
            return "none"
        or_bars = today_bars.iloc[:self.opening_range_bars]
        or_high = float(or_bars["High"].max())
        or_low = float(or_bars["Low"].min())
        cur_close = float(history.iloc[i]["Close"])
        if cur_close > or_high:
            return "long"
        if cur_close < or_low:
            return "short"
        return "none"
