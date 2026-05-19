"""
ADX-filtered trend follower.

Classic trend strategies (SMA crossover, EMA stack) get murdered in chop. The
ADX indicator measures TREND STRENGTH regardless of direction — common
heuristic is "ADX > 25 = trending, ADX < 20 = ranging".

This strategy gates a simple SMA crossover behind an ADX threshold:
  - Only takes the cross if ADX > `adx_threshold`.
  - +DI > -DI bias confirms longs; -DI > +DI confirms shorts.

If the ADX filter actually adds skill, this should outperform plain SMA
crossover net of its rejected trades. Useful as the cleanest example of
"filter a baseline strategy with a regime indicator".

Why it might work:
  - Trend strategies need trends. ADX directly measures that.
  - Filter is mechanical, no curve-fitting per asset.

Why it might not:
  - ADX is laggy — by the time it crosses 25, the trend may be exhausted.
  - "Chop vs trend" is a fuzzy continuum, not a binary regime.
  - Adds another tunable knob, easy to over-optimise.
"""
from __future__ import annotations

import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import sma, atr, adx
from strategies._helpers import risk_based_stake


class AdxTrend:
    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        r_target: float = 2.0,
    ):
        self.fast = int(fast)
        self.slow = int(slow)
        self.adx_period = int(adx_period)
        self.adx_threshold = float(adx_threshold)
        self.atr_period = int(atr_period)
        self.atr_stop_mult = atr_stop_mult
        self.r_target = r_target

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        if len(history) < max(self.slow, self.adx_period) + 5:
            return Signal(action="noop")

        close = history["Close"]
        fast_sma = sma(close, self.fast)
        slow_sma = sma(close, self.slow)

        f_now, f_prev = float(fast_sma.iloc[-1]), float(fast_sma.iloc[-2])
        s_now, s_prev = float(slow_sma.iloc[-1]), float(slow_sma.iloc[-2])
        cross_up = f_prev <= s_prev and f_now > s_now
        cross_down = f_prev >= s_prev and f_now < s_now

        # Exit on adverse cross even if no entry triggers
        if broker.position is not None:
            if broker.position.side == "long" and cross_down:
                return Signal(action="close", reason="bearish cross")
            if broker.position.side == "short" and cross_up:
                return Signal(action="close", reason="bullish cross")
            return Signal(action="noop")

        if not (cross_up or cross_down):
            return Signal(action="noop")

        # ADX gate
        adx_line, plus_di, minus_di = adx(history, self.adx_period)
        adx_val = float(adx_line.iloc[-1])
        plus_now = float(plus_di.iloc[-1])
        minus_now = float(minus_di.iloc[-1])
        if pd.isna(adx_val) or adx_val < self.adx_threshold:
            return Signal(action="noop")

        # Directional confirmation: +DI bias for longs, -DI bias for shorts
        if cross_up and plus_now <= minus_now:
            return Signal(action="noop")
        if cross_down and minus_now <= plus_now:
            return Signal(action="noop")

        atr_val = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return Signal(action="noop")

        price = float(history["Close"].iloc[-1])
        stop_pts = self.atr_stop_mult * atr_val
        stake = risk_based_stake(broker.balance, stop_pts, price=price)
        if stake <= 0:
            return Signal(action="noop")

        if cross_up:
            return Signal(
                action="open_long",
                stake_per_point=stake,
                stop_loss=price - stop_pts,
                take_profit=price + self.r_target * stop_pts,
                reason=f"cross_up ADX={adx_val:.1f} +DI={plus_now:.1f}",
            )
        else:
            return Signal(
                action="open_short",
                stake_per_point=stake,
                stop_loss=price + stop_pts,
                take_profit=price - self.r_target * stop_pts,
                reason=f"cross_down ADX={adx_val:.1f} -DI={minus_now:.1f}",
            )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        if len(history) < max(self.slow, self.adx_period) + 5:
            return "none"
        close = history["Close"]
        fast_sma = sma(close, self.fast)
        slow_sma = sma(close, self.slow)
        f_now, f_prev = float(fast_sma.iloc[-1]), float(fast_sma.iloc[-2])
        s_now, s_prev = float(slow_sma.iloc[-1]), float(slow_sma.iloc[-2])
        cross_up = f_prev <= s_prev and f_now > s_now
        cross_down = f_prev >= s_prev and f_now < s_now
        adx_line, plus_di, minus_di = adx(history, self.adx_period)
        adx_val = float(adx_line.iloc[-1])
        if adx_val < self.adx_threshold:
            return "none"
        if cross_up and float(plus_di.iloc[-1]) > float(minus_di.iloc[-1]):
            return "long"
        if cross_down and float(minus_di.iloc[-1]) > float(plus_di.iloc[-1]):
            return "short"
        return "none"
