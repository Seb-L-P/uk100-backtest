"""
Triple EMA stack — multi-MA trend alignment.

Compute three EMAs (fast, mid, slow). Long when EMAs are stacked up
(fast > mid > slow). Short when stacked down (fast < mid < slow). Exit when
the stack breaks (e.g., fast crosses below mid in an up-stacked trade).

Filters out chop by requiring three timeframes of MA agreement; trades less
frequently than a 2-MA crossover but with cleaner trend context.
"""
from __future__ import annotations
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import ema, atr
from strategies._helpers import risk_based_stake


class TripleEma:
    def __init__(self, fast: int = 9, mid: int = 21, slow: int = 50,
                 atr_period: int = 14, atr_stop_mult: float = 2.0):
        self.fast = int(fast)
        self.mid = int(mid)
        self.slow = int(slow)
        self.atr_period = int(atr_period)
        self.atr_stop_mult = atr_stop_mult

    def _stack(self, history: pd.DataFrame):
        if len(history) < self.slow + 2:
            return None, None
        f = ema(history["Close"], self.fast).iloc[-1]
        m = ema(history["Close"], self.mid).iloc[-1]
        s = ema(history["Close"], self.slow).iloc[-1]
        f_p = ema(history["Close"], self.fast).iloc[-2]
        m_p = ema(history["Close"], self.mid).iloc[-2]
        # Fresh transition into a stacked state — current is stacked, prior wasn't
        cur_up = f > m > s
        cur_dn = f < m < s
        prev_up = f_p > m_p
        prev_dn = f_p < m_p
        return cur_up, cur_dn

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        if len(history) < self.slow + 2:
            return Signal(action="noop")
        f_series = ema(history["Close"], self.fast)
        m_series = ema(history["Close"], self.mid)
        s_series = ema(history["Close"], self.slow)
        f, m, s = float(f_series.iloc[-1]), float(m_series.iloc[-1]), float(s_series.iloc[-1])

        # Exit on stack break
        if broker.position is not None:
            if broker.position.side == "long" and not (f > m > s):
                return Signal(action="close", reason="stack broke")
            if broker.position.side == "short" and not (f < m < s):
                return Signal(action="close", reason="stack broke")
            return Signal(action="noop")

        # Entry on fresh stack
        f_p = float(f_series.iloc[-2])
        m_p = float(m_series.iloc[-2])
        s_p = float(s_series.iloc[-2])
        prev_up = f_p > m_p > s_p
        prev_dn = f_p < m_p < s_p
        cur_up = f > m > s
        cur_dn = f < m < s

        direction = None
        if cur_up and not prev_up:
            direction = "long"
        elif cur_dn and not prev_dn:
            direction = "short"
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
            reason=f"EMA stack {direction} ({self.fast}/{self.mid}/{self.slow})",
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """For ensemble polling — returns the current stack direction if
        a fresh transition just happened; else "none"."""
        if len(history) < self.slow + 2:
            return "none"
        f_series = ema(history["Close"], self.fast)
        m_series = ema(history["Close"], self.mid)
        s_series = ema(history["Close"], self.slow)
        f, m, s = float(f_series.iloc[-1]), float(m_series.iloc[-1]), float(s_series.iloc[-1])
        f_p = float(f_series.iloc[-2])
        m_p = float(m_series.iloc[-2])
        s_p = float(s_series.iloc[-2])
        prev_up = f_p > m_p > s_p
        prev_dn = f_p < m_p < s_p
        if f > m > s and not prev_up:
            return "long"
        if f < m < s and not prev_dn:
            return "short"
        return "none"
