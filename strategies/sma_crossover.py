"""
SMA crossover — pipeline smoke test, NOT a real strategy.

Rules:
  - Go long when fast SMA crosses above slow SMA
  - Reverse to short when fast crosses below
  - Position sized to risk `risk_per_trade_pct` of equity, stop at 2 * ATR(14)

This exists purely to prove the engine runs end-to-end with real data and
real costs. It is well-known that simple SMA crossover loses money after costs
on most markets. If our backtest shows it making money, something is wrong
with our cost model. If it loses money, the pipeline is honest.
"""
from __future__ import annotations

import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from config import ACCOUNT


class SmaCrossover:
    def __init__(self, fast: int = 20, slow: int = 50, atr_period: int = 14,
                 atr_stop_mult: float = 2.0):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        if len(history) < max(self.slow, self.atr_period) + 2:
            return Signal(action="noop")

        close = history["Close"]
        fast_sma = close.rolling(self.fast).mean()
        slow_sma = close.rolling(self.slow).mean()

        # Detect crossover on the most recent COMPLETED bar
        f_now, f_prev = fast_sma.iloc[-1], fast_sma.iloc[-2]
        s_now, s_prev = slow_sma.iloc[-1], slow_sma.iloc[-2]
        cross_up = f_prev <= s_prev and f_now > s_now
        cross_down = f_prev >= s_prev and f_now < s_now

        if not (cross_up or cross_down):
            return Signal(action="noop")

        # ATR for stop sizing
        atr = _atr(history, self.atr_period).iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return Signal(action="noop")

        price = float(history["Close"].iloc[-1])
        stop_distance_pts = self.atr_stop_mult * float(atr)

        # Risk-based position sizing: risk_per_trade_pct of equity on stop distance,
        # capped by leverage to avoid over-sizing on high-priced indices.
        from strategies._helpers import risk_based_stake
        stake_per_point = risk_based_stake(broker.balance, stop_distance_pts, price=price)

        if cross_up:
            return Signal(
                action="open_long",
                stake_per_point=stake_per_point,
                stop_loss=price - stop_distance_pts,
                reason=f"fast({self.fast})>slow({self.slow})",
            )
        else:  # cross_down
            return Signal(
                action="open_short",
                stake_per_point=stake_per_point,
                stop_loss=price + stop_distance_pts,
                reason=f"fast({self.fast})<slow({self.slow})",
            )


def _atr(history: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = history["High"], history["Low"], history["Close"]
    prev_close = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
