"""
Heikin Ashi trend follower.

Heikin Ashi candles smooth raw OHLC into a clearer trend visualisation.
A run of N consecutive same-coloured HA bars indicates a sustained trend.

Rules:
  Long when last `consecutive_bars` HA candles are bullish (close > open).
  Short when last `consecutive_bars` HA candles are bearish.
  Exit when HA candle colour flips (one opposite-colour bar).

Filters out noise; bigger N = tighter trend filter = fewer trades.
"""
from __future__ import annotations
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import heikin_ashi, atr
from strategies._helpers import risk_based_stake


class HeikinAshiTrend:
    def __init__(self, consecutive_bars: int = 3,
                 atr_period: int = 14, atr_stop_mult: float = 2.0):
        self.consecutive_bars = int(consecutive_bars)
        self.atr_period = int(atr_period)
        self.atr_stop_mult = atr_stop_mult

    def _check_run(self, history: pd.DataFrame):
        if len(history) < self.consecutive_bars + 5:
            return None
        ha = heikin_ashi(history)
        last_n = ha.iloc[-self.consecutive_bars:]
        all_bull = (last_n["HA_Close"] > last_n["HA_Open"]).all()
        all_bear = (last_n["HA_Close"] < last_n["HA_Open"]).all()
        # Also require the candle just before the run to be the opposite (i.e.
        # a fresh flip), so we don't keep re-entering inside an existing run.
        if len(ha) < self.consecutive_bars + 1:
            return None
        prev_bar = ha.iloc[-(self.consecutive_bars + 1)]
        if all_bull and prev_bar["HA_Close"] <= prev_bar["HA_Open"]:
            return "long"
        if all_bear and prev_bar["HA_Close"] >= prev_bar["HA_Open"]:
            return "short"
        return None

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        # Exit if in a position and HA colour just flipped
        if broker.position is not None:
            ha = heikin_ashi(history)
            cur_bull = ha["HA_Close"].iloc[-1] > ha["HA_Open"].iloc[-1]
            if broker.position.side == "long" and not cur_bull:
                return Signal(action="close", reason="HA flipped bearish")
            if broker.position.side == "short" and cur_bull:
                return Signal(action="close", reason="HA flipped bullish")
            return Signal(action="noop")

        direction = self._check_run(history)
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
            reason=f"HA {direction} run of {self.consecutive_bars}",
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        return self._check_run(history) or "none"
