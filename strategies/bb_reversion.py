"""
Bollinger Band reversion — classic mean reversion.

Rules:
  1. Compute Bollinger Bands (period, mult).
  2. When close crosses ABOVE the upper band → enter SHORT (fade the move).
  3. When close crosses BELOW the lower band → enter LONG (buy the dip).
  4. Stop: ATR-based, beyond the breakout extreme + buffer.
  5. Exit: when close touches the middle band (target = mid SMA).
  6. Optional filter: only trade in normal/low volatility regime — high vol
     often means a real trend, not a reversion setup.

Why it might work:
  - Bollinger Bands quantify "stretched" — closes beyond 2σ are statistically
    rare in stable markets.
  - The middle SMA is a natural mean-reversion target.

Why it might not:
  - In a trending market, BB breakouts are CONTINUATION signals, not reversal.
  - Sharp moves often blow through the band and keep going.
  - Without a regime filter, this is a coin flip.
"""
from __future__ import annotations

import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import bollinger, atr, volatility_regime
from strategies._helpers import risk_based_stake, atr_threshold


class BollingerReversion:
    def __init__(
        self,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        stop_buffer_atr_mult: float = 0.1,
        require_low_vol_regime: bool = False,
    ):
        self.bb_period = bb_period
        self.bb_mult = bb_mult
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.require_low_vol_regime = require_low_vol_regime

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        warmup = max(self.bb_period, self.atr_period) + 2
        if i < warmup:
            return Signal(action="noop")

        close = history["Close"]
        mid, upper, lower = bollinger(close, self.bb_period, self.bb_mult)
        cur_close = float(close.iloc[-1])
        cur_mid = float(mid.iloc[-1])
        cur_upper = float(upper.iloc[-1])
        cur_lower = float(lower.iloc[-1])

        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")

        # ---- Exit: if in a position, exit at mid-band touch ---------------
        if broker.position is not None:
            if broker.position.side == "long" and cur_close >= cur_mid:
                return Signal(action="close", reason="bb_mid_target")
            if broker.position.side == "short" and cur_close <= cur_mid:
                return Signal(action="close", reason="bb_mid_target")
            return Signal(action="noop")

        # ---- Optional regime filter ---------------------------------------
        if self.require_low_vol_regime:
            regime = volatility_regime(history, self.atr_period).iloc[-1]
            if regime == 1:  # high vol — skip
                return Signal(action="noop")

        # ---- Entry: close beyond a band ----------------------------------
        bar = history.iloc[i]
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])

        stop_buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)

        if cur_close > cur_upper:
            # Short the stretch
            stop = bar_high + self.atr_stop_mult * atr_now + stop_buf
            risk = stop - cur_close
            if risk <= 0:
                return Signal(action="noop")
            target = cur_mid
            stake = risk_based_stake(broker.balance, risk, price=cur_close)
            return Signal(action="open_short", stake_per_point=stake,
                          stop_loss=stop, take_profit=target,
                          reason=f"BB upper={cur_upper:.1f}, close={cur_close:.1f}")
        if cur_close < cur_lower:
            stop = bar_low - self.atr_stop_mult * atr_now - stop_buf
            risk = cur_close - stop
            if risk <= 0:
                return Signal(action="noop")
            target = cur_mid
            stake = risk_based_stake(broker.balance, risk, price=cur_close)
            return Signal(action="open_long", stake_per_point=stake,
                          stop_loss=stop, take_profit=target,
                          reason=f"BB lower={cur_lower:.1f}, close={cur_close:.1f}")

        return Signal(action="noop")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """For ensemble polling."""
        if len(history) < max(self.bb_period, self.atr_period) + 2:
            return "none"
        close = history["Close"]
        _mid, upper, lower = bollinger(close, self.bb_period, self.bb_mult)
        cur_close = float(close.iloc[-1])
        if self.require_low_vol_regime:
            regime = volatility_regime(history, self.atr_period).iloc[-1]
            if regime == 1:
                return "none"
        if cur_close > float(upper.iloc[-1]):
            return "short"
        if cur_close < float(lower.iloc[-1]):
            return "long"
        return "none"
