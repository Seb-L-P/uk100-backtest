"""
Overnight-range breakout — day-trade.

When a market trades 24h (futures, crypto) or has extended-hours data, the
"overnight" range (high/low from after-close to pre-open) often acts as a
key intraday level once the cash session opens. Sharp algos and human
traders alike hunt the break of this range.

Even on assets without true 24h data, the PRIOR session's high/low from a
calendar day ago gives a similar structural level — "yesterday's range".

This strategy:
  1. At session_open each day, look back to the bars BEFORE today's first
     session bar in our dataset. Take their high (overnight high) and low
     (overnight low) — bounded by `lookback_bars`.
  2. Arm pending STOP orders at the overnight high (long) and overnight
     low (short).
  3. First-fill-wins, the other is cancelled (like ORB and inside-bar).
  4. Stops past the opposite extreme, target R-multiple.
  5. Cancel pending orders by `flat_by` and close any open trade.

Why it might work:
  - Gap-and-go is a real intraday phenomenon when news/overnight flow has
    repositioned the market.
  - The range break has cleaner geometry than an in-bar momentum entry —
    you're catching a confirmed move past a well-watched level.

Why it might not:
  - On instruments without real overnight data (Yahoo intraday only covers
    cash hours for US/UK stocks/indices), the "overnight range" is
    artificial — really just yesterday's last hour vs today's first bars.
  - Fakeouts at the level are common.
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from strategies._helpers import risk_based_stake, in_session, atr_threshold


class OvernightRangeBreakout:
    def __init__(
        self,
        lookback_bars: int = 30,
        stop_buffer_atr_mult: float = 0.3,
        r_target: float = 2.0,
        atr_period: int = 14,
        session_open: time = time(8, 0),
        session_close: time = time(15, 30),
        flat_by: time = time(16, 0),
        max_age_bars: int = 12,
    ):
        self.lookback_bars = int(lookback_bars)
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.r_target = r_target
        self.atr_period = int(atr_period)
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by
        self.max_age_bars = int(max_age_bars)

        self._day_date = None
        self._armed = False
        self._buy_id: str | None = None
        self._sell_id: str | None = None

    def _reset_day(self):
        self._armed = False
        self._buy_id = None
        self._sell_id = None

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        ts = history.index[i]
        now = ts.time()
        today = ts.date()

        if today != self._day_date:
            self._reset_day()
            self._day_date = today

        if now >= self.flat_by:
            for oid in (self._buy_id, self._sell_id):
                if oid is not None:
                    broker.cancel_pending_order(oid)
            self._buy_id = self._sell_id = None
            if broker.position is not None:
                return Signal(action="close", reason="session_end")
            return Signal(action="noop")

        if not in_session(now, self.session_open, self.session_close):
            return Signal(action="noop")

        # First-fill-wins housekeeping
        live = {o.id for o in broker.pending_orders}
        buy_dead = self._buy_id is not None and self._buy_id not in live
        sell_dead = self._sell_id is not None and self._sell_id not in live
        if buy_dead and sell_dead:
            self._buy_id = self._sell_id = None
        elif buy_dead and self._sell_id in live:
            broker.cancel_pending_order(self._sell_id)
            self._sell_id = self._buy_id = None
        elif sell_dead and self._buy_id in live:
            broker.cancel_pending_order(self._buy_id)
            self._buy_id = self._sell_id = None

        if broker.position is not None or self._armed:
            return Signal(action="noop")

        # Only arm the orders ONCE per day, on the first IN-SESSION bar of
        # today. With continuous 24h data, the first bar of the calendar day
        # is midnight — but we want to arm at session_open, using the bars
        # BEFORE session open (including the previous calendar day's tail
        # AND any overnight bars of today) as the lookback.
        today_mask = history.index.date == today
        if not today_mask.any():
            return Signal(action="noop")

        # In-session today: first bar where date == today AND time >= session_open.
        in_session_today = today_mask & (
            pd.Index([t.time() for t in history.index]) >= self.session_open
        )
        first_in_session = in_session_today.argmax() if in_session_today.any() else -1
        if first_in_session <= 0 or i != first_in_session:
            return Signal(action="noop")

        # Lookback covers `lookback_bars` bars ending just before first_in_session.
        start = max(0, first_in_session - self.lookback_bars)
        prior = history.iloc[start:first_in_session]
        if len(prior) < 3:
            return Signal(action="noop")

        on_high = float(prior["High"].max())
        on_low = float(prior["Low"].min())
        cur = float(history.iloc[i]["Close"])
        # Don't arm a stop that's already in-the-money (price has already gapped
        # past the level by the time we run).
        if cur >= on_high or cur <= on_low:
            return Signal(action="noop")

        buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)
        entry_up, entry_dn = on_high, on_low
        stop_up = on_low - buf
        risk_up = entry_up - stop_up
        target_up = entry_up + self.r_target * risk_up
        stop_dn = on_high + buf
        risk_dn = stop_dn - entry_dn
        target_dn = entry_dn - self.r_target * risk_dn
        if risk_up <= 0 or risk_dn <= 0:
            return Signal(action="noop")

        stake_up = risk_based_stake(broker.balance, risk_up, price=entry_up)
        stake_dn = risk_based_stake(broker.balance, risk_dn, price=entry_dn)
        if stake_up <= 0 or stake_dn <= 0:
            return Signal(action="noop")

        buy = broker.place_pending_order(
            side="long", order_type="stop",
            trigger_price=entry_up, stake_per_point=stake_up,
            time=ts, stop_loss=stop_up, take_profit=target_up,
            expires_after_bars=self.max_age_bars,
        )
        sell = broker.place_pending_order(
            side="short", order_type="stop",
            trigger_price=entry_dn, stake_per_point=stake_dn,
            time=ts, stop_loss=stop_dn, take_profit=target_dn,
            expires_after_bars=self.max_age_bars,
        )
        self._buy_id, self._sell_id = buy.id, sell.id
        self._armed = True
        return Signal(action="noop",
                      reason=f"overnight range armed: {on_low:.1f}-{on_high:.1f}")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """Direction implied by whichever side of the prior-day range we
        broke first today."""
        i = len(history) - 1
        today = history.index[i].date()
        today_mask = history.index.date == today
        first_today = today_mask.argmax() if today_mask.any() else -1
        if first_today <= 0:
            return "none"
        start = max(0, first_today - self.lookback_bars)
        prior = history.iloc[start:first_today]
        if len(prior) < 3:
            return "none"
        on_high = float(prior["High"].max())
        on_low = float(prior["Low"].min())
        cur = float(history.iloc[-1]["Close"])
        if cur > on_high:
            return "long"
        if cur < on_low:
            return "short"
        return "none"
