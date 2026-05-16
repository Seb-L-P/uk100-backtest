"""
Engulfing candle reversal strategy.

Bullish engulfing: a green bar that fully engulfs the previous red bar's body.
Bearish engulfing: a red bar that fully engulfs the previous green bar's body.

Take entries on the close of the engulfing bar. Stop on the other side of the
engulfing pattern. Optionally require recent trend in the opposite direction
(reversal context).

Single-bar / 2-bar pattern, easy to overtrade — adjustable trend filter +
min size threshold to keep signal quality up.
"""
from __future__ import annotations
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import ema, atr
from strategies._helpers import risk_based_stake, atr_threshold


class EngulfingReversal:
    def __init__(self, trend_ema_period: int = 20, min_body_atr_mult: float = 0.5,
                 r_target: float = 2.0, stop_buffer_atr_mult: float = 0.1,
                 atr_period: int = 14, require_trend_context: bool = True):
        self.trend_ema_period = int(trend_ema_period)
        self.min_body_atr_mult = min_body_atr_mult
        self.r_target = r_target
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.atr_period = int(atr_period)
        self.require_trend_context = bool(require_trend_context)

    def _detect(self, history: pd.DataFrame):
        if len(history) < self.trend_ema_period + 3:
            return None
        prev = history.iloc[-2]
        curr = history.iloc[-1]
        prev_body = prev["Close"] - prev["Open"]
        curr_body = curr["Close"] - curr["Open"]

        min_body = atr_threshold(history, self.min_body_atr_mult, self.atr_period)

        # Bullish engulfing: prev red, curr green, curr body engulfs prev body
        if (prev_body < 0 and curr_body > 0
                and curr["Open"] <= prev["Close"]
                and curr["Close"] >= prev["Open"]
                and abs(curr_body) >= min_body):
            if self.require_trend_context:
                ema_s = ema(history["Close"], self.trend_ema_period)
                # Reversal context = recent downtrend (EMA slope down)
                if not (ema_s.iloc[-1] < ema_s.iloc[-5]):
                    return None
            return "long", float(prev["Low"])

        # Bearish engulfing: prev green, curr red, curr body engulfs prev
        if (prev_body > 0 and curr_body < 0
                and curr["Open"] >= prev["Close"]
                and curr["Close"] <= prev["Open"]
                and abs(curr_body) >= min_body):
            if self.require_trend_context:
                ema_s = ema(history["Close"], self.trend_ema_period)
                if not (ema_s.iloc[-1] > ema_s.iloc[-5]):
                    return None
            return "short", float(prev["High"])
        return None

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        det = self._detect(history)
        if det is None:
            return Signal(action="noop")
        direction, stop_anchor = det
        price = float(history["Close"].iloc[-1])
        stop_buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)
        if direction == "long":
            stop = stop_anchor - stop_buf
            risk = price - stop
        else:
            stop = stop_anchor + stop_buf
            risk = stop - price
        if risk <= 0:
            return Signal(action="noop")
        target = (price + self.r_target * risk if direction == "long"
                  else price - self.r_target * risk)
        stake = risk_based_stake(broker.balance, risk, price=price)
        return Signal(
            action="open_long" if direction == "long" else "open_short",
            stake_per_point=stake, stop_loss=stop, take_profit=target,
            reason=f"{direction} engulfing",
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        det = self._detect(history)
        return det[0] if det else "none"
