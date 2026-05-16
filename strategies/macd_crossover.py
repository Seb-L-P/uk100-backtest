"""
MACD signal-line crossover. Classic momentum strategy.

Long when MACD line crosses above its signal line.
Short when MACD line crosses below its signal line.
Reverse on opposite signal; ATR-based stops.

Works best in trending markets; whipsaws badly in chop.
"""
from __future__ import annotations
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import macd, atr
from strategies._helpers import risk_based_stake


class MacdCrossover:
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 atr_period: int = 14, atr_stop_mult: float = 2.0):
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)
        self.atr_period = int(atr_period)
        self.atr_stop_mult = atr_stop_mult

    def _cross(self, history: pd.DataFrame):
        if len(history) < max(self.slow, self.atr_period) + 5:
            return None
        m, sig, _ = macd(history["Close"], self.fast, self.slow, self.signal)
        m_now, m_prev = float(m.iloc[-1]), float(m.iloc[-2])
        s_now, s_prev = float(sig.iloc[-1]), float(sig.iloc[-2])
        if m_prev <= s_prev and m_now > s_now:
            return "long"
        if m_prev >= s_prev and m_now < s_now:
            return "short"
        return None

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        direction = self._cross(history)
        if direction is None:
            return Signal(action="noop")
        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")
        price = float(history["Close"].iloc[-1])
        stop_pts = self.atr_stop_mult * atr_now
        if direction == "long":
            stop = price - stop_pts
        else:
            stop = price + stop_pts
        stake = risk_based_stake(broker.balance, stop_pts, price=price)
        return Signal(
            action="open_long" if direction == "long" else "open_short",
            stake_per_point=stake, stop_loss=stop,
            reason=f"MACD cross {direction}",
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        return self._cross(history) or "none"
