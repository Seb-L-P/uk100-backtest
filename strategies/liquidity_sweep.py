"""
Liquidity Sweep — day-trade.

Concept: stops cluster just past recent swing highs and swing lows. When price
"sweeps" through one of those clusters — pokes past the level, triggers the
stops, then immediately reverses — it's a signal that the move was a stop-hunt,
not a real breakout. Enter against the sweep direction.

Rules:
  1. Track the trailing swing high (highest high of last `swing_lookback` bars
     not including current) and swing low (lowest low of same).
  2. SWEEP HIGH (bearish signal): current bar's HIGH exceeds the trailing
     swing high by at least `sweep_min_atr_mult * ATR` AND the close is BELOW the swing
     high (rejection). Enter SHORT at close.
  3. SWEEP LOW (bullish signal): current bar's LOW pierces below trailing
     swing low by at least `sweep_min_atr_mult * ATR` AND close is ABOVE the swing low.
     Enter LONG at close.
  4. Stop: just past the sweep extreme (the high/low of the sweep bar) + buffer.
  5. Target: `r_target` × risk.
  6. UK session only. Flat by close.

Why it might work:
  - Real institutional money runs stops to fill orders. The fingerprint is
    "spike past level, immediate reversal".
  - The setup has clean math: failed breakout = momentum reversal.

Why it might not:
  - Real sweeps are easier to identify in hindsight than in real-time.
  - "Pierce + close back" is also just a wick — happens in chop without any
    institutional driver.
  - Multiple swing definitions exist; ours is naive (rolling lookback).
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from strategies._helpers import risk_based_stake, in_session, trailing_swing


class LiquiditySweep:
    def __init__(
        self,
        swing_lookback: int = 20,
        sweep_min_atr_mult: float = 0.3,    # ATR-relative pierce distance
        stop_buffer_atr_mult: float = 0.2,  # ATR-relative stop buffer
        atr_period: int = 14,
        r_target: float = 2.0,
        session_open: time = time(8, 30),
        session_close: time = time(15, 0),
        flat_by: time = time(15, 30),
    ):
        self.swing_lookback = int(swing_lookback)
        self.sweep_min_atr_mult = sweep_min_atr_mult
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.atr_period = int(atr_period)
        self.r_target = r_target
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        bar = history.iloc[i]
        ts = history.index[i]
        now = ts.time()

        if i < self.swing_lookback:
            return Signal(action="noop")

        # Force-flat near end of day
        if broker.position is not None and now >= self.flat_by:
            return Signal(action="close", reason="session_end")

        if not in_session(now, self.session_open, self.session_close):
            return Signal(action="noop")
        if broker.position is not None:
            return Signal(action="noop")

        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])
        bar_close = float(bar["Close"])

        swing_high = trailing_swing(history, i, self.swing_lookback, "high")
        swing_low = trailing_swing(history, i, self.swing_lookback, "low")

        # ATR-scaled thresholds (sweep distance + stop buffer)
        from strategies._helpers import atr_threshold
        sweep_min = atr_threshold(history, self.sweep_min_atr_mult, self.atr_period)
        stop_buffer = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)

        # ---- Sweep HIGH = bearish signal -----------------------------------
        if (bar_high > swing_high + sweep_min) and (bar_close < swing_high):
            entry = bar_close
            stop = bar_high + stop_buffer
            risk = stop - entry
            if risk <= 0:
                return Signal(action="noop")
            target = entry - self.r_target * risk
            stake = risk_based_stake(broker.balance, risk, price=entry)
            return Signal(action="open_short", stake_per_point=stake,
                          stop_loss=stop, take_profit=target,
                          reason=f"sweep_high@{swing_high:.1f}, pierce={bar_high - swing_high:.1f}pts")

        # ---- Sweep LOW = bullish signal -----------------------------------
        if (bar_low < swing_low - sweep_min) and (bar_close > swing_low):
            entry = bar_close
            stop = bar_low - stop_buffer
            risk = entry - stop
            if risk <= 0:
                return Signal(action="noop")
            target = entry + self.r_target * risk
            stake = risk_based_stake(broker.balance, risk, price=entry)
            return Signal(action="open_long", stake_per_point=stake,
                          stop_loss=stop, take_profit=target,
                          reason=f"sweep_low@{swing_low:.1f}, pierce={swing_low - bar_low:.1f}pts")

        return Signal(action="noop")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """For ensemble polling."""
        from strategies._helpers import atr_threshold
        i = len(history) - 1
        if i < self.swing_lookback:
            return "none"
        bar = history.iloc[i]
        bar_high, bar_low, bar_close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        swing_high = trailing_swing(history, i, self.swing_lookback, "high")
        swing_low = trailing_swing(history, i, self.swing_lookback, "low")
        sweep_min = atr_threshold(history, self.sweep_min_atr_mult, self.atr_period)
        if (bar_high > swing_high + sweep_min) and (bar_close < swing_high):
            return "short"
        if (bar_low < swing_low - sweep_min) and (bar_close > swing_low):
            return "long"
        return "none"
