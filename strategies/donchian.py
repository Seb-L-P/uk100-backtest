"""
Donchian Channel Breakout (a.k.a. the original Turtle Trader system).

The classic Turtle entry rule from 1983:
  - BUY when price breaks above the highest high of the last N bars
  - SELL when price breaks below the lowest low of the last N bars
  - Stop: 2 × ATR away from entry
  - Exit: opposite signal OR ATR-trailing stop

WARNING about timeframe:
  This system was designed for daily/weekly futures and worked spectacularly
  in the 80s. On modern liquid index intraday (e.g. 15m FTSE 100) it will
  generate frequent breakouts that mostly fail because:
    - 15m noise is much greater than the trend signal
    - Spread + slippage eat small breakouts
    - Modern markets mean-revert intraday
  We're including it because (a) it's the canonical baseline strategy, (b) seeing
  it fail intraday is itself informative, and (c) it should run as a control
  group when comparing other strategies.

  For a fairer test, also run with `--interval 1d`.

Rules implemented:
  1. Channel = highest high / lowest low of last `channel_lookback` COMPLETED
     bars (excluding current bar — online-safe).
  2. Long entry: bar's CLOSE breaks above the upper channel.
  3. Short entry: bar's CLOSE breaks below the lower channel.
  4. Stop: entry ± `atr_stop_mult` × ATR(`atr_period`).
  5. No fixed target. Exit when price closes back inside the channel by
     `exit_lookback` bars (Turtle exit rule), or stop is hit.
  6. No session filter — Donchian was a 24/7 system originally.
"""
from __future__ import annotations

import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from strategies._helpers import risk_based_stake, atr, trailing_swing


class DonchianBreakout:
    def __init__(
        self,
        channel_lookback: int = 20,         # Turtle System 1 used 20
        exit_lookback: int = 10,            # Turtle System 1 exit was 10
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
    ):
        self.channel_lookback = channel_lookback
        self.exit_lookback = exit_lookback
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        warmup = max(self.channel_lookback, self.atr_period) + 2
        if i < warmup:
            return Signal(action="noop")

        bar_close = float(history.iloc[i]["Close"])
        upper = trailing_swing(history, i, self.channel_lookback, "high")
        lower = trailing_swing(history, i, self.channel_lookback, "low")
        upper_exit = trailing_swing(history, i, self.exit_lookback, "high")
        lower_exit = trailing_swing(history, i, self.exit_lookback, "low")

        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")
        stop_distance = self.atr_stop_mult * atr_now

        # ---- Exit logic (Turtle rule) -----------------------------------
        if broker.position is not None:
            if broker.position.side == "long" and bar_close < lower_exit:
                return Signal(action="close", reason="turtle_exit_long")
            if broker.position.side == "short" and bar_close > upper_exit:
                return Signal(action="close", reason="turtle_exit_short")
            return Signal(action="noop")

        # ---- Entry logic ------------------------------------------------
        if bar_close > upper:
            stop = bar_close - stop_distance
            stake = risk_based_stake(broker.balance, stop_distance, price=bar_close)
            return Signal(action="open_long", stake_per_point=stake,
                          stop_loss=stop,
                          reason=f"donch_up: close>{upper:.1f}")
        if bar_close < lower:
            stop = bar_close + stop_distance
            stake = risk_based_stake(broker.balance, stop_distance, price=bar_close)
            return Signal(action="open_short", stake_per_point=stake,
                          stop_loss=stop,
                          reason=f"donch_down: close<{lower:.1f}")

        return Signal(action="noop")
