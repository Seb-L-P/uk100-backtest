"""
Pivot Reversal — day-trade.

Classic floor-trader pivot points (P, R1/R2, S1/S2) define daily horizontal
support/resistance derived from YESTERDAY's H/L/C. Many intraday traders
watch these levels because so many other traders watch them — a self-
fulfilling source of liquidity.

Rules:
  1. At the start of each trading day, compute today's pivot levels from
     yesterday's H/L/C (`pivot_points_classic` in indicators.py — already
     look-ahead safe, it uses .shift(1)).
  2. Arm a LIMIT BUY at S1 with stop past S2.
     Arm a LIMIT SELL at R1 with stop past R2.
  3. Target: the central pivot P.
  4. Pending orders expire after `max_age_bars` (typically a few hours).
  5. Force-flat by `flat_by`.

Why limits (not stops)?
  We WANT to fade the move into S1/R1. Stop orders would do the opposite —
  they'd trigger as price punched THROUGH the level. Limit orders are the
  right primitive for "buy weakness, sell strength".

Why it might work:
  - Pivots are widely watched. Limit clusters form at them.
  - The risk:reward geometry is built into the levels (S1 → P is roughly
    1R away on most days, sometimes 2R).
  - Setup is mechanical and gives clean targets.

Why it might not:
  - When price gaps through S1 or trends hard, the limit never fills (good)
    OR fills and immediately stops out (bad).
  - Pivots derived from a calendar day can be unhelpful on assets that trade
    24/5 (crypto, futures) — the "session" is fuzzy.
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import pivot_points_classic
from strategies._helpers import risk_based_stake, in_session, atr_threshold


class PivotReversal:
    def __init__(
        self,
        stop_buffer_atr_mult: float = 0.3,
        atr_period: int = 14,
        max_age_bars: int = 20,
        session_open: time = time(8, 0),
        session_close: time = time(15, 30),
        flat_by: time = time(16, 0),
        target_at: str = "P",  # which pivot level to target — P, R1, S1, etc.
    ):
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.atr_period = int(atr_period)
        self.max_age_bars = int(max_age_bars)
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by
        self.target_at = target_at

        # Daily state
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

        # Force flat at end of day
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

        # One pair of orders per day, one trade max
        if broker.position is not None or self._armed:
            return Signal(action="noop")

        # Need at least one prior day of bars to have pivots for today
        pivots = pivot_points_classic(history)
        # Find today's pivot row. pivot_points_classic is indexed by midnight
        # of each calendar day; row for today is what we need.
        today_ts = pd.Timestamp(today)
        if today_ts not in pivots.index:
            # Try matching by date if the index is timezone-aware
            mask = pivots.index.date == today
            if not mask.any():
                return Signal(action="noop")
            row = pivots[mask].iloc[0]
        else:
            row = pivots.loc[today_ts]

        try:
            P = float(row["P"])
            R1 = float(row["R1"])
            R2 = float(row["R2"])
            S1 = float(row["S1"])
            S2 = float(row["S2"])
        except (KeyError, ValueError):
            return Signal(action="noop")

        stop_buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)

        # Long: buy S1 limit, stop past S2, target P
        long_entry = S1
        long_stop = S2 - stop_buf
        long_target = float(row.get(self.target_at, P))
        long_risk = long_entry - long_stop

        # Short: sell R1 limit, stop past R2, target P
        short_entry = R1
        short_stop = R2 + stop_buf
        short_target = float(row.get(self.target_at, P))
        short_risk = short_stop - short_entry

        # Need positive risk and target on the correct side
        if long_risk <= 0 or short_risk <= 0:
            return Signal(action="noop")
        if not (long_target > long_entry):
            return Signal(action="noop")
        if not (short_target < short_entry):
            return Signal(action="noop")

        cur_price = float(history.iloc[-1]["Close"])
        # Don't arm a limit that's already in-the-money — would fill instantly
        # at a worse price than what we wanted (the open of next bar).
        if cur_price <= long_entry or cur_price >= short_entry:
            return Signal(action="noop")

        stake_long = risk_based_stake(broker.balance, long_risk, price=long_entry)
        stake_short = risk_based_stake(broker.balance, short_risk, price=short_entry)
        if stake_long <= 0 or stake_short <= 0:
            return Signal(action="noop")

        buy = broker.place_pending_order(
            side="long", order_type="limit",
            trigger_price=long_entry, stake_per_point=stake_long,
            time=ts, stop_loss=long_stop, take_profit=long_target,
            expires_after_bars=self.max_age_bars,
        )
        sell = broker.place_pending_order(
            side="short", order_type="limit",
            trigger_price=short_entry, stake_per_point=stake_short,
            time=ts, stop_loss=short_stop, take_profit=short_target,
            expires_after_bars=self.max_age_bars,
        )
        self._buy_id, self._sell_id = buy.id, sell.id
        self._armed = True
        return Signal(action="noop",
                      reason=f"pivots armed: S1={S1:.1f} R1={R1:.1f} P={P:.1f}")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """For ensemble polling — direction implied by recent pivot interaction."""
        i = len(history) - 1
        if i < 2:
            return "none"
        pivots = pivot_points_classic(history)
        today = history.index[i].date()
        mask = pivots.index.date == today
        if not mask.any():
            return "none"
        row = pivots[mask].iloc[0]
        cur = float(history.iloc[-1]["Close"])
        prev = float(history.iloc[-2]["Close"])
        S1 = float(row["S1"])
        R1 = float(row["R1"])
        # Touched S1 from above and closed above → fading the dip → long
        if prev <= S1 < cur:
            return "long"
        if prev >= R1 > cur:
            return "short"
        return "none"
