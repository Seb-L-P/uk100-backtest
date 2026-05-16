"""
Stochastic %K crosses %D in oversold/overbought zones.

Long when %K crosses above %D AND %D is below `oversold` (default 30).
Short when %K crosses below %D AND %D is above `overbought` (default 70).
ATR-based stop, ATR-based target (no fixed R-multiple, just exit when
%K crosses back through %D in the opposite direction).
"""
from __future__ import annotations
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import stochastic, atr
from strategies._helpers import risk_based_stake


class StochasticCrossover:
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3,
                 oversold: float = 30.0, overbought: float = 70.0,
                 atr_period: int = 14, atr_stop_mult: float = 2.0):
        self.k_period = int(k_period)
        self.d_period = int(d_period)
        self.smooth_k = int(smooth_k)
        self.oversold = oversold
        self.overbought = overbought
        self.atr_period = int(atr_period)
        self.atr_stop_mult = atr_stop_mult

    def _signal(self, history: pd.DataFrame):
        if len(history) < self.k_period + self.smooth_k + 5:
            return None
        k, d = stochastic(history, self.k_period, self.d_period, self.smooth_k)
        k_now, k_prev = float(k.iloc[-1]), float(k.iloc[-2])
        d_now, d_prev = float(d.iloc[-1]), float(d.iloc[-2])
        if k_prev <= d_prev and k_now > d_now and d_prev < self.oversold:
            return "long"
        if k_prev >= d_prev and k_now < d_now and d_prev > self.overbought:
            return "short"
        return None

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        direction = self._signal(history)
        if direction is None:
            return Signal(action="noop")
        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")
        price = float(history["Close"].iloc[-1])
        stop_pts = self.atr_stop_mult * atr_now
        stop = price - stop_pts if direction == "long" else price + stop_pts
        stake = risk_based_stake(broker.balance, stop_pts, price=price)
        return Signal(
            action="open_long" if direction == "long" else "open_short",
            stake_per_point=stake, stop_loss=stop,
            reason=f"Stoch %K x %D in {'OS' if direction=='long' else 'OB'}",
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        return self._signal(history) or "none"
