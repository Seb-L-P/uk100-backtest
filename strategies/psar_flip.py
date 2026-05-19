"""
Parabolic SAR flip — trend follower.

Welles Wilder's Parabolic SAR alternates between two regimes:
  - Bullish: SAR sits below price, ratcheting up each bar.
  - Bearish: SAR sits above price, ratcheting down.

When the bar's range crosses SAR, the indicator FLIPS — that's both the
reversal signal AND the trailing stop. Designed to "stop and reverse" (the
literal meaning of SAR).

Rules:
  1. Detect a SAR flip on the most recent bar (sign change vs the bar before).
  2. Open in the new flip direction at next bar's open.
  3. Stop = current SAR value (with a small buffer).
  4. Exit on the next SAR flip (no fixed take-profit — let it trend).
  5. Optional: ADX gate to skip flips in chop.

Why it might work:
  - SAR is a clean implementation of trend trailing.
  - Always-in-the-market style; catches all big swings, even if it gives
    back on minor ones.

Why it might not:
  - Famously bad in chop. AF=0.02 default is slow on intraday.
  - "Stop and reverse on every flip" generates more trades than it should
    on intraday TFs — costs add up.
  - Same-bar flip-then-flip-again is common in noisy markets.
"""
from __future__ import annotations

import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import parabolic_sar, atr, adx
from strategies._helpers import risk_based_stake, atr_threshold


class ParabolicSarFlip:
    def __init__(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.2,
        stop_buffer_atr_mult: float = 0.1,
        atr_period: int = 14,
        adx_min: float = 0.0,           # 0 = no filter; e.g. 20 to skip chop
        adx_period: int = 14,
    ):
        self.af_start = float(af_start)
        self.af_step = float(af_step)
        self.af_max = float(af_max)
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.atr_period = int(atr_period)
        self.adx_min = float(adx_min)
        self.adx_period = int(adx_period)

    def _sar_direction(self, history: pd.DataFrame) -> pd.Series:
        sar = parabolic_sar(history, self.af_start, self.af_step, self.af_max)
        # Positive when bullish (sar below close), negative when bearish
        return (history["Close"] - sar)

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        if len(history) < max(20, self.atr_period + 2, self.adx_period + 2):
            return Signal(action="noop")

        diff = self._sar_direction(history)
        d_now = float(diff.iloc[-1])
        d_prev = float(diff.iloc[-2])
        flipped_bull = d_prev <= 0 and d_now > 0
        flipped_bear = d_prev >= 0 and d_now < 0

        # Exit-on-flip: if we hold a long and SAR flips bearish, close;
        # mirror for short. The new-direction open is handled below as a
        # separate entry — they'll both run in this same bar.
        if broker.position is not None:
            if broker.position.side == "long" and flipped_bear:
                return Signal(action="close", reason="sar flip bear")
            if broker.position.side == "short" and flipped_bull:
                return Signal(action="close", reason="sar flip bull")
            return Signal(action="noop")

        if not (flipped_bull or flipped_bear):
            return Signal(action="noop")

        # Optional ADX gate
        if self.adx_min > 0:
            adx_line, _, _ = adx(history, self.adx_period)
            if float(adx_line.iloc[-1]) < self.adx_min:
                return Signal(action="noop")

        sar = parabolic_sar(history, self.af_start, self.af_step, self.af_max)
        sar_now = float(sar.iloc[-1])
        price = float(history["Close"].iloc[-1])
        buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)

        if flipped_bull:
            # SAR is now below price → use it as the stop
            stop = sar_now - buf
            stop_pts = price - stop
        else:
            stop = sar_now + buf
            stop_pts = stop - price

        if stop_pts <= 0:
            return Signal(action="noop")

        stake = risk_based_stake(broker.balance, stop_pts, price=price)
        if stake <= 0:
            return Signal(action="noop")

        return Signal(
            action="open_long" if flipped_bull else "open_short",
            stake_per_point=stake,
            stop_loss=stop,
            reason=f"sar flip {'bull' if flipped_bull else 'bear'} @ {sar_now:.2f}",
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        if len(history) < 20:
            return "none"
        diff = self._sar_direction(history)
        d_now = float(diff.iloc[-1])
        d_prev = float(diff.iloc[-2])
        if d_prev <= 0 and d_now > 0:
            return "long"
        if d_prev >= 0 and d_now < 0:
            return "short"
        return "none"
