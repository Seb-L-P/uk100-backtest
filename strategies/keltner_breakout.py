"""
Keltner Channel breakout — momentum/trend continuation.

Keltner Channels are centred on EMA (not SMA) and use a multiple of ATR for
band width (not std dev). The result: smoother bands that don't expand
during volatility squeezes the way Bollinger Bands do, and that ride trends
better.

When the close BREAKS OUT of the upper or lower channel, take it as
momentum confirmation and enter in the direction of the break. This is the
opposite philosophy to BB-reversion: we believe a Keltner break is the
START of a move, not the end of one.

Rules:
  1. Compute Keltner (EMA period, ATR period, multiplier).
  2. If close[-1] > upper[-1] AND close[-2] <= upper[-2] → fresh upper
     break → go long.
  3. Symmetric for lower break → short.
  4. Stop = middle EMA (treats EMA as the trend baseline). Optional ATR
     buffer.
  5. Target = R-multiple.

Why it might work:
  - ATR-based bands adapt to volatility automatically — same threshold
     works across instruments.
  - Trend-continuation logic complements all the reversion strategies in
     the library (provides diversification when ensembled).

Why it might not:
  - Breakouts have many false starts.
  - Mid-EMA stop is wide on big-trend days, expensive on chop.
"""
from __future__ import annotations

import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import keltner_channels, atr
from strategies._helpers import risk_based_stake, atr_threshold


class KeltnerBreakout:
    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        mult: float = 2.0,
        stop_buffer_atr_mult: float = 0.2,
        r_target: float = 2.0,
    ):
        self.ema_period = int(ema_period)
        self.atr_period = int(atr_period)
        self.mult = float(mult)
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.r_target = float(r_target)

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        if len(history) < self.ema_period + self.atr_period + 5:
            return Signal(action="noop")

        if broker.position is not None:
            return Signal(action="noop")

        mid, upper, lower = keltner_channels(
            history, self.ema_period, self.atr_period, self.mult
        )
        c_now = float(history["Close"].iloc[-1])
        c_prev = float(history["Close"].iloc[-2])
        u_now, u_prev = float(upper.iloc[-1]), float(upper.iloc[-2])
        l_now, l_prev = float(lower.iloc[-1]), float(lower.iloc[-2])
        mid_now = float(mid.iloc[-1])

        broke_up = c_now > u_now and c_prev <= u_prev
        broke_dn = c_now < l_now and c_prev >= l_prev
        if not (broke_up or broke_dn):
            return Signal(action="noop")

        buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)
        price = c_now

        if broke_up:
            stop = mid_now - buf
            risk = price - stop
            if risk <= 0:
                return Signal(action="noop")
            target = price + self.r_target * risk
            stake = risk_based_stake(broker.balance, risk, price=price)
            if stake <= 0:
                return Signal(action="noop")
            return Signal(
                action="open_long", stake_per_point=stake,
                stop_loss=stop, take_profit=target,
                reason=f"keltner break upper @ {u_now:.2f}",
            )
        else:
            stop = mid_now + buf
            risk = stop - price
            if risk <= 0:
                return Signal(action="noop")
            target = price - self.r_target * risk
            stake = risk_based_stake(broker.balance, risk, price=price)
            if stake <= 0:
                return Signal(action="noop")
            return Signal(
                action="open_short", stake_per_point=stake,
                stop_loss=stop, take_profit=target,
                reason=f"keltner break lower @ {l_now:.2f}",
            )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        if len(history) < self.ema_period + self.atr_period + 5:
            return "none"
        mid, upper, lower = keltner_channels(
            history, self.ema_period, self.atr_period, self.mult
        )
        c_now = float(history["Close"].iloc[-1])
        c_prev = float(history["Close"].iloc[-2])
        if c_now > float(upper.iloc[-1]) and c_prev <= float(upper.iloc[-2]):
            return "long"
        if c_now < float(lower.iloc[-1]) and c_prev >= float(lower.iloc[-2]):
            return "short"
        return "none"
