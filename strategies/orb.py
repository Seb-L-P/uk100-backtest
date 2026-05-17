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
        # IDs of the buy-stop + sell-stop orders armed once the OR closes.
        # First-fill-wins: when one triggers, the other is cancelled.
        self._stop_up_id: str | None = None
        self._stop_dn_id: str | None = None
        self._stops_armed = False

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

        # Force-flat near end of day: cancel pending OR-break orders + close
        if now >= self.flat_by:
            for oid in (self._stop_up_id, self._stop_dn_id):
                if oid is not None:
                    broker.cancel_pending_order(oid)
            self._stop_up_id = self._stop_dn_id = None
            if broker.position is not None:
                return Signal(action="close", reason="session_end")

        # Outside session / already traded → nothing to do
        if not in_session(now, self.session_open, self.session_close):
            return Signal(action="noop")
        if broker.position is not None or self._traded_today:
            return Signal(action="noop")

        # First-fill-wins: if one of our pending stops filled and is no longer
        # in the broker's pending list, cancel its sibling so we don't open
        # the opposite trade later.
        live = {o.id for o in broker.pending_orders}
        if self._stops_armed:
            up_dead = self._stop_up_id is not None and self._stop_up_id not in live
            dn_dead = self._stop_dn_id is not None and self._stop_dn_id not in live
            if up_dead and self._stop_dn_id in live:
                broker.cancel_pending_order(self._stop_dn_id)
                self._stop_dn_id = None
                self._traded_today = True
            elif dn_dead and self._stop_up_id in live:
                broker.cancel_pending_order(self._stop_up_id)
                self._stop_up_id = None
                self._traded_today = True

        # Build the opening range as the first N bars
        if self._day_bars_seen <= self.opening_range_bars:
            high_so_far = float(bar["High"])
            low_so_far = float(bar["Low"])
            self._or_high = high_so_far if self._or_high is None else max(self._or_high, high_so_far)
            self._or_low = low_so_far if self._or_low is None else min(self._or_low, low_so_far)
            return Signal(action="noop")

        # OR closed — arm two STOP orders the first time we reach this point.
        # Fill price = trigger price, so stop/target geometry is preserved.
        if self._stops_armed:
            return Signal(action="noop")
        if self._or_high is None or self._or_low is None:
            return Signal(action="noop")

        from strategies._helpers import atr_threshold
        stop_buffer = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)

        # Bullish stop-buy at OR high
        entry_up = self._or_high
        stop_up = self._or_low - stop_buffer
        risk_up = entry_up - stop_up
        target_up = entry_up + self.r_target * risk_up
        stake_up = risk_based_stake(broker.balance, risk_up, price=entry_up)

        # Bearish stop-sell at OR low
        entry_dn = self._or_low
        stop_dn = self._or_high + stop_buffer
        risk_dn = stop_dn - entry_dn
        target_dn = entry_dn - self.r_target * risk_dn
        stake_dn = risk_based_stake(broker.balance, risk_dn, price=entry_dn)

        if risk_up <= 0 or risk_dn <= 0 or stake_up <= 0 or stake_dn <= 0:
            return Signal(action="noop")

        order_up = broker.place_pending_order(
            side="long", order_type="stop",
            trigger_price=entry_up, stake_per_point=stake_up,
            time=ts, stop_loss=stop_up, take_profit=target_up,
            expires_after_bars=None,  # cancelled at flat_by
        )
        order_dn = broker.place_pending_order(
            side="short", order_type="stop",
            trigger_price=entry_dn, stake_per_point=stake_dn,
            time=ts, stop_loss=stop_dn, take_profit=target_dn,
            expires_after_bars=None,
        )
        self._stop_up_id = order_up.id
        self._stop_dn_id = order_dn.id
        self._stops_armed = True
        return Signal(action="noop",
                      reason=f"ORB stops armed: range={self._or_low:.1f}-{self._or_high:.1f}")

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
