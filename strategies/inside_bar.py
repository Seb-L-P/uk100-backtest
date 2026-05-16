"""
Inside-bar breakout.

An "inside bar" has high <= prev_high AND low >= prev_low. It's a
consolidation signal — the next bar that breaks out of the inside bar's
range often continues in the breakout direction.

Place pending stop orders at the inside bar's high (buy stop) and low (sell
stop). Whichever fills first becomes the trade; cancel the other.
"""
from __future__ import annotations
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import atr
from strategies._helpers import risk_based_stake, atr_threshold


class InsideBarBreakout:
    def __init__(self, atr_period: int = 14, atr_stop_mult: float = 1.5,
                 r_target: float = 2.0, min_inside_range_atr_mult: float = 0.3,
                 trigger_buffer_atr_mult: float = 0.05,
                 max_age_bars: int = 5):
        self.atr_period = int(atr_period)
        self.atr_stop_mult = atr_stop_mult
        self.r_target = r_target
        self.min_inside_range_atr_mult = min_inside_range_atr_mult
        self.trigger_buffer_atr_mult = trigger_buffer_atr_mult
        self.max_age_bars = int(max_age_bars)

        self._buy_stop_id: str | None = None
        self._sell_stop_id: str | None = None
        self._inside_bar_index: int = -1

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1

        # Cancel orders that have aged out
        if (self._inside_bar_index >= 0
                and i - self._inside_bar_index > self.max_age_bars):
            if self._buy_stop_id:
                broker.cancel_pending_order(self._buy_stop_id)
                self._buy_stop_id = None
            if self._sell_stop_id:
                broker.cancel_pending_order(self._sell_stop_id)
                self._sell_stop_id = None
            self._inside_bar_index = -1

        # If one side filled, cancel the other (first-fill-wins)
        live = {o.id for o in broker.pending_orders}
        if (self._buy_stop_id and self._buy_stop_id not in live
                and self._sell_stop_id and self._sell_stop_id in live):
            broker.cancel_pending_order(self._sell_stop_id)
            self._sell_stop_id = None
            self._buy_stop_id = None
            self._inside_bar_index = -1
        elif (self._sell_stop_id and self._sell_stop_id not in live
              and self._buy_stop_id and self._buy_stop_id in live):
            broker.cancel_pending_order(self._buy_stop_id)
            self._buy_stop_id = None
            self._sell_stop_id = None
            self._inside_bar_index = -1

        # Only one pair of stops outstanding at a time
        if (broker.position is not None
                or self._buy_stop_id is not None
                or self._sell_stop_id is not None):
            return Signal(action="noop")

        if i < 2:
            return Signal(action="noop")
        prev = history.iloc[-2]
        curr = history.iloc[-1]
        is_inside = (curr["High"] <= prev["High"] and curr["Low"] >= prev["Low"])
        if not is_inside:
            return Signal(action="noop")
        inside_range = float(curr["High"]) - float(curr["Low"])
        min_range = atr_threshold(history, self.min_inside_range_atr_mult, self.atr_period)
        if inside_range < min_range:
            return Signal(action="noop")

        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")

        stop_pts = self.atr_stop_mult * atr_now
        time = history.index[i]

        trig_buf = atr_threshold(history, self.trigger_buffer_atr_mult, self.atr_period)
        buy_trigger = float(curr["High"]) + trig_buf
        sell_trigger = float(curr["Low"]) - trig_buf
        stake_buy = risk_based_stake(broker.balance, stop_pts, price=buy_trigger)
        stake_sell = risk_based_stake(broker.balance, stop_pts, price=sell_trigger)

        buy = broker.place_pending_order(
            side="long", order_type="stop", trigger_price=buy_trigger,
            stake_per_point=stake_buy, time=time,
            stop_loss=buy_trigger - stop_pts,
            take_profit=buy_trigger + self.r_target * stop_pts,
            expires_after_bars=self.max_age_bars,
        )
        sell = broker.place_pending_order(
            side="short", order_type="stop", trigger_price=sell_trigger,
            stake_per_point=stake_sell, time=time,
            stop_loss=sell_trigger + stop_pts,
            take_profit=sell_trigger - self.r_target * stop_pts,
            expires_after_bars=self.max_age_bars,
        )
        self._buy_stop_id, self._sell_stop_id = buy.id, sell.id
        self._inside_bar_index = i
        return Signal(action="noop")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """
        Ensemble polling: directional bias on most recent inside-bar break.
        If current bar breaks above an inside bar formed within max_age_bars,
        return "long" (and "short" for the down break). Else "none".
        """
        i = len(history) - 1
        if i < 2:
            return "none"
        # Look back for an inside bar in the recent window
        for j in range(max(2, i - self.max_age_bars), i + 1):
            prev = history.iloc[j - 1]
            inside = history.iloc[j]
            if (inside["High"] <= prev["High"] and inside["Low"] >= prev["Low"]
                    and float(inside["High"]) - float(inside["Low"]) >=
                        atr_threshold(history, self.min_inside_range_atr_mult, self.atr_period)):
                cur = history.iloc[-1]
                if float(cur["Close"]) > float(inside["High"]):
                    return "long"
                if float(cur["Close"]) < float(inside["Low"]):
                    return "short"
        return "none"
