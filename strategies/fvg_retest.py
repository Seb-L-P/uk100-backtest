"""
Fair Value Gap (FVG) retest strategy — day-trade only, LIMIT-order version.

This is the correct simulation of how an SMC trader actually trades FVGs:

  1. On each bar, detect any new 3-bar FVG.
  2. If the FVG is valid size, place a LIMIT order at the FVG's near edge
     (top for bullish, bottom for bearish). The order sits pending.
  3. The order fills ONLY IF price retraces back to the near edge — exactly
     how a real limit order works.
  4. Stop / target / position size are computed at order-placement time and
     carried through to the fill.
  5. If an FVG expires (stale) or gets fully filled by price action, its
     pending order is cancelled.
  6. All pending orders cancelled at session close to avoid carrying over.

This differs from the OLD (market-order) version which fired whenever a bar
TOUCHED the FVG, then filled at next bar's OPEN — often at a worse price than
the FVG edge. The limit-order version produces more realistic, more honest
backtests for SMC strategies.
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import detect_fvg, FVG


class FvgRetest:
    def __init__(
        self,
        min_gap_atr_mult: float = 0.5,
        max_gap_atr_mult: float = 5.0,
        max_age_bars: int = 30,
        stop_buffer_atr_mult: float = 0.2,
        atr_period: int = 14,
        r_target: float = 2.0,
        session_open: time = time(9, 0),
        session_close: time = time(15, 0),
        flat_by: time = time(15, 30),
    ):
        # Thresholds expressed as ATR multiples — scale-invariant across
        # instruments (FTSE, AAPL, BTC) and volatility regimes.
        self.min_gap_atr_mult = min_gap_atr_mult
        self.max_gap_atr_mult = max_gap_atr_mult
        self.max_age_bars = int(max_age_bars)
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.atr_period = int(atr_period)
        self.r_target = r_target
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by

        # Rolling list of unfilled FVGs (each with a pending limit order)
        self._open_fvgs: list[FVG] = []
        # Map FVG (by its creator_bar_index) → pending order id
        self._fvg_to_order_id: dict[int, str] = {}
        self._last_processed_index: int = -1

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        bar = history.iloc[i]
        now = history.index[i].time()
        bar_low = float(bar["Low"])
        bar_high = float(bar["High"])

        # ---- 1. Sync order tracking: drop entries whose orders are gone ---
        # If a tracked order is no longer in broker.pending_orders, it either
        # filled (good — became a position) or expired (good — broker cleaned up).
        # Either way, we no longer need to track it.
        live_order_ids = {o.id for o in broker.pending_orders}
        self._fvg_to_order_id = {
            k: v for k, v in self._fvg_to_order_id.items() if v in live_order_ids
        }

        # ---- 2. If we just got filled into a position, cancel all OTHER
        # pending orders. "First-fill-wins" — once we're in a trade, the
        # remaining pending FVG orders are no longer relevant.
        if broker.position is not None and self._fvg_to_order_id:
            for order_id in list(self._fvg_to_order_id.values()):
                broker.cancel_pending_order(order_id)
            self._fvg_to_order_id.clear()

        # ---- 3. Update FVG list (once per bar) ---------------------------
        if i != self._last_processed_index:
            self._last_processed_index = i

            # Drop stale and price-filled FVGs; cancel their pending orders.
            kept: list[FVG] = []
            for f in self._open_fvgs:
                age = i - f.creator_bar_index
                if age > self.max_age_bars or f.is_filled_by(bar_low, bar_high):
                    order_id = self._fvg_to_order_id.pop(f.creator_bar_index, None)
                    if order_id is not None:
                        broker.cancel_pending_order(order_id)
                else:
                    kept.append(f)
            self._open_fvgs = kept

            # Place a limit order only if we're flat (no position, no pending order).
            # This keeps the "one shot at a time" intent of the original strategy
            # and avoids generating dozens of orders that can't fill.
            in_session = self.session_open <= now <= self.session_close
            is_flat = (broker.position is None and not self._fvg_to_order_id)
            new_fvg = detect_fvg(history, i)
            # Scale gap-size thresholds by current ATR
            from strategies._helpers import atr_threshold
            min_gap = atr_threshold(history, self.min_gap_atr_mult, self.atr_period)
            max_gap = atr_threshold(history, self.max_gap_atr_mult, self.atr_period,
                                     fallback_pts=1e9)
            if (new_fvg is not None
                    and min_gap <= new_fvg.size_points <= max_gap
                    and in_session and is_flat):
                self._open_fvgs.append(new_fvg)
                self._place_limit_for_fvg(new_fvg, broker, history.index[i],
                                          history=history)

        # ---- 4. Force-flat at end of session ------------------------------
        if now >= self.flat_by:
            # Cancel every pending order we placed
            for order_id in list(self._fvg_to_order_id.values()):
                broker.cancel_pending_order(order_id)
            self._fvg_to_order_id.clear()
            # Close any open position
            if broker.position is not None:
                return Signal(action="close", reason="session_end")

        return Signal(action="noop")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """
        For ensemble polling. Stateless: scans recent bars for unfilled FVGs
        that are touched by the current bar. Returns the direction of the
        most-recent such FVG. Ignores limit-order semantics — ensembles take
        market-priced entries based on intent.
        """
        from strategies._helpers import atr_threshold
        i = len(history) - 1
        if i < 2:
            return "none"
        bar = history.iloc[i]
        bar_low, bar_high = float(bar["Low"]), float(bar["High"])

        # ATR-scaled gap thresholds (computed once per call)
        min_gap = atr_threshold(history, self.min_gap_atr_mult, self.atr_period)
        max_gap = atr_threshold(history, self.max_gap_atr_mult, self.atr_period,
                                 fallback_pts=1e9)

        best: FVG | None = None
        # Walk back up to max_age_bars and check each candidate FVG.
        for j in range(max(2, i - self.max_age_bars), i + 1):
            fvg = detect_fvg(history, j)
            if fvg is None:
                continue
            if not (min_gap <= fvg.size_points <= max_gap):
                continue
            # Has the FVG already been fully filled by intermediate bars?
            after = history.iloc[fvg.creator_bar_index + 1: i + 1]
            if fvg.direction == "bullish":
                if not after.empty and bool((after["Low"] <= fvg.zone_low).any()):
                    continue
            else:
                if not after.empty and bool((after["High"] >= fvg.zone_high).any()):
                    continue
            # Is this FVG touched by the current bar?
            if not fvg.is_touched_by(bar_low, bar_high):
                continue
            # Prefer the most-recent qualifying FVG (freshest signal)
            if best is None or fvg.creator_bar_index > best.creator_bar_index:
                best = fvg
        if best is None:
            return "none"
        return "long" if best.direction == "bullish" else "short"

    def _place_limit_for_fvg(self, fvg: FVG, broker: Broker, time,
                              history: pd.DataFrame | None = None) -> None:
        """Compute entry/stop/target for the FVG and place the limit order."""
        from strategies._helpers import risk_based_stake, atr_threshold

        # ATR-scaled stop buffer
        if history is not None:
            stop_buffer = atr_threshold(history, self.stop_buffer_atr_mult,
                                         self.atr_period, fallback_pts=0.5)
        else:
            stop_buffer = 0.5

        if fvg.direction == "bullish":
            entry = fvg.near_edge                       # zone_high
            stop = fvg.far_edge - stop_buffer           # below zone_low
            risk_pts = entry - stop
            target = entry + self.r_target * risk_pts
            side = "long"
        else:
            entry = fvg.near_edge                       # zone_low
            stop = fvg.far_edge + stop_buffer           # above zone_high
            risk_pts = stop - entry
            target = entry - self.r_target * risk_pts
            side = "short"

        if risk_pts <= 0:
            return

        stake = risk_based_stake(broker.balance, risk_pts, price=entry)
        if stake <= 0:
            return

        # Place limit order; expire when the FVG itself would expire.
        order = broker.place_pending_order(
            side=side,
            order_type="limit",
            trigger_price=entry,
            stake_per_point=stake,
            time=time,
            stop_loss=stop,
            take_profit=target,
            expires_after_bars=self.max_age_bars,
        )
        self._fvg_to_order_id[fvg.creator_bar_index] = order.id
