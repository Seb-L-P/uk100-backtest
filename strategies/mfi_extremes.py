"""
MFI extremes — mean reversion gated by volume.

The Money Flow Index is RSI's volume-weighted cousin. Where RSI just looks
at price deltas, MFI weights each delta by that bar's volume (so a 1pt move
on heavy volume counts more than a 1pt move on light volume).

The setup: when MFI prints an extreme value AND THEN hooks back, the move
likely had genuine volume behind it — so the reversal is more likely to
have legs than a pure RSI hook. Symmetrical with the rsi_revert strategy
but should fire on a different subset of bars (volume-aware).

Rules:
  1. Compute MFI(period).
  2. LONG: MFI dipped below `oversold`, then this bar prints MFI above the
     prior bar (hook up).
  3. SHORT: mirror — MFI rose above `overbought`, then hooks down.
  4. Exit at MFI crossing back through 50 OR stop/take-profit.
  5. ATR-based stops, R-multiple target.

Why it might work:
  - Volume confirmation filters out low-quality RSI signals (the kind that
    fire on thin overnight prints).
  - Same volume mechanic that pros watch (volume-confirmed exhaustion).

Why it might not:
  - Yahoo's volume is exchange volume on the underlying components, not
    your broker's CFD flow. The signal quality is decent but not
    institutional-grade.
  - "Hook" detection on a single-bar delta is noisy. Could require a
    sustained hook (2-3 bars) for cleaner signals.
"""
from __future__ import annotations

import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import mfi, atr
from strategies._helpers import risk_based_stake, atr_threshold


class MfiExtremes:
    def __init__(
        self,
        mfi_period: int = 14,
        oversold: float = 20.0,
        overbought: float = 80.0,
        exit_level: float = 50.0,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        stop_buffer_atr_mult: float = 0.1,
        r_target: float = 2.0,
        lookback_for_extreme: int = 5,
    ):
        self.mfi_period = int(mfi_period)
        self.oversold = float(oversold)
        self.overbought = float(overbought)
        self.exit_level = float(exit_level)
        self.atr_period = int(atr_period)
        self.atr_stop_mult = atr_stop_mult
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.r_target = r_target
        self.lookback_for_extreme = int(lookback_for_extreme)

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        if len(history) < self.mfi_period + self.lookback_for_extreme + 2:
            return Signal(action="noop")

        m = mfi(history, self.mfi_period)
        m_now = float(m.iloc[-1])
        m_prev = float(m.iloc[-2])

        # Manage existing position: exit on MFI crossing back through 50
        if broker.position is not None:
            if broker.position.side == "long" and m_prev < self.exit_level <= m_now:
                return Signal(action="close", reason=f"mfi exit (>{self.exit_level:.0f})")
            if broker.position.side == "short" and m_prev > self.exit_level >= m_now:
                return Signal(action="close", reason=f"mfi exit (<{self.exit_level:.0f})")
            return Signal(action="noop")

        # Look back N bars for an extreme reading, then require a hook this bar
        recent = m.iloc[-(self.lookback_for_extreme + 1):-1]
        long_setup = recent.min() <= self.oversold and m_now > m_prev
        short_setup = recent.max() >= self.overbought and m_now < m_prev
        if not (long_setup or short_setup):
            return Signal(action="noop")

        atr_val = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return Signal(action="noop")

        price = float(history["Close"].iloc[-1])
        buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)
        stop_pts = self.atr_stop_mult * atr_val + buf

        stake = risk_based_stake(broker.balance, stop_pts, price=price)
        if stake <= 0:
            return Signal(action="noop")

        if long_setup:
            return Signal(
                action="open_long",
                stake_per_point=stake,
                stop_loss=price - stop_pts,
                take_profit=price + self.r_target * stop_pts,
                reason=f"mfi oversold hook ({m_prev:.0f}→{m_now:.0f})",
            )
        else:
            return Signal(
                action="open_short",
                stake_per_point=stake,
                stop_loss=price + stop_pts,
                take_profit=price - self.r_target * stop_pts,
                reason=f"mfi overbought hook ({m_prev:.0f}→{m_now:.0f})",
            )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        if len(history) < self.mfi_period + self.lookback_for_extreme + 2:
            return "none"
        m = mfi(history, self.mfi_period)
        m_now = float(m.iloc[-1])
        m_prev = float(m.iloc[-2])
        recent = m.iloc[-(self.lookback_for_extreme + 1):-1]
        if recent.min() <= self.oversold and m_now > m_prev:
            return "long"
        if recent.max() >= self.overbought and m_now < m_prev:
            return "short"
        return "none"
