"""
RSI mean reversion — classic oversold/overbought system.

Rules:
  1. Compute Wilder's RSI(period).
  2. When RSI dips below `oversold` (default 30) AND turns back up → LONG.
  3. When RSI rises above `overbought` (default 70) AND turns back down → SHORT.
  4. Stop: ATR-based, beyond the entry bar's extreme.
  5. Exit: when RSI crosses back through the `exit_level` (default 50).
  6. No session filter — works on any timeframe.

Why it might work:
  - Markets do mean-revert in non-trending regimes.
  - RSI extremes mark exhaustion points in ranging markets.
  - One of the oldest documented edges in technical analysis.

Why it might not:
  - The edge has degraded substantially since the 80s/90s.
  - In trending markets RSI can stay oversold/overbought for many bars.
  - On intraday charts, RSI mean-reverts too quickly to give actionable signals.
"""
from __future__ import annotations

import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import rsi, atr
from strategies._helpers import risk_based_stake, atr_threshold


class RsiReversion:
    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_level: float = 50.0,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        stop_buffer_atr_mult: float = 0.1,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.stop_buffer_atr_mult = stop_buffer_atr_mult

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        warmup = max(self.rsi_period, self.atr_period) + 3
        if i < warmup:
            return Signal(action="noop")

        rsi_series = rsi(history["Close"], self.rsi_period)
        cur_rsi = float(rsi_series.iloc[-1])
        prev_rsi = float(rsi_series.iloc[-2])

        # Exit: in a position and RSI has crossed back through exit_level
        if broker.position is not None:
            if broker.position.side == "long" and prev_rsi < self.exit_level <= cur_rsi:
                return Signal(action="close", reason="rsi_crossed_mid")
            if broker.position.side == "short" and prev_rsi > self.exit_level >= cur_rsi:
                return Signal(action="close", reason="rsi_crossed_mid")
            return Signal(action="noop")

        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")

        cur_close = float(history["Close"].iloc[-1])
        bar = history.iloc[i]
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])

        stop_buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)

        # Entry: RSI hooked back from extreme
        # Long when RSI was below oversold and is now turning up
        if prev_rsi < self.oversold and cur_rsi > prev_rsi:
            stop = bar_low - self.atr_stop_mult * atr_now - stop_buf
            risk = cur_close - stop
            if risk <= 0:
                return Signal(action="noop")
            stake = risk_based_stake(broker.balance, risk, price=cur_close)
            return Signal(action="open_long", stake_per_point=stake,
                          stop_loss=stop,
                          reason=f"RSI hook up from {prev_rsi:.1f}")
        if prev_rsi > self.overbought and cur_rsi < prev_rsi:
            stop = bar_high + self.atr_stop_mult * atr_now + stop_buf
            risk = stop - cur_close
            if risk <= 0:
                return Signal(action="noop")
            stake = risk_based_stake(broker.balance, risk, price=cur_close)
            return Signal(action="open_short", stake_per_point=stake,
                          stop_loss=stop,
                          reason=f"RSI hook down from {prev_rsi:.1f}")

        return Signal(action="noop")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """For ensemble polling."""
        if len(history) < max(self.rsi_period, self.atr_period) + 3:
            return "none"
        rsi_series = rsi(history["Close"], self.rsi_period)
        cur_rsi = float(rsi_series.iloc[-1])
        prev_rsi = float(rsi_series.iloc[-2])
        if prev_rsi < self.oversold and cur_rsi > prev_rsi:
            return "long"
        if prev_rsi > self.overbought and cur_rsi < prev_rsi:
            return "short"
        return "none"
