"""
Exit / trailing-stop helpers.

Functions in here are FACTORIES — they take parameters and return a callable
that the broker invokes on each bar with `(position, bar_dict)` and expects a
new stop_loss value (or None to leave the stop unchanged).

The broker only RATCHETS stops — it never widens them. So a trailing function
that returns a wider stop than the current one is silently ignored. This means
every helper here is safe to use without worrying about loosening protection
by accident.

Usage in a strategy:
    from backtest.exits import atr_trailing, breakeven_after_R

    return Signal(
        action="open_long", stake_per_point=1.0, stop_loss=stop, take_profit=tp,
        trailing_stop_fn=atr_trailing(history, atr_period=14, mult=2.0),
    )

Note that history-dependent trailing functions (like atr_trailing) need access
to enough recent data to compute the indicator. They take a snapshot of the
history at signal time and update via the bars passed to them — accepting that
ATR will be slightly stale but cheap.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from backtest.indicators import atr as _atr_func


def atr_trailing(
    history: pd.DataFrame,
    atr_period: int = 14,
    mult: float = 2.0,
) -> Callable:
    """
    ATR trailing stop. On each bar, computes new stop = close ± (mult × ATR).
    Stop only ratchets in the favourable direction (broker enforces this).

    Captures a starting ATR from `history` at creation; subsequent updates use
    the current bar's close ± a moving ATR estimate that we update with each
    bar (using a simple EMA approximation since we don't get the full history
    in the callback).
    """
    # Initial ATR snapshot
    initial_atr = float(_atr_func(history, atr_period).iloc[-1]) if len(history) >= atr_period else 1.0
    if initial_atr <= 0 or pd.isna(initial_atr):
        initial_atr = 1.0

    # We'll smooth ATR with each bar using a simple recursive update
    state = {"atr": initial_atr, "prev_close": float(history["Close"].iloc[-1])}
    alpha = 1.0 / atr_period

    def _update(position, bar) -> float | None:
        # Update rolling ATR estimate
        tr = max(
            bar["High"] - bar["Low"],
            abs(bar["High"] - state["prev_close"]),
            abs(bar["Low"] - state["prev_close"]),
        )
        state["atr"] = (1 - alpha) * state["atr"] + alpha * tr
        state["prev_close"] = bar["Close"]

        # Compute candidate stop
        if position.side == "long":
            return bar["Close"] - mult * state["atr"]
        else:
            return bar["Close"] + mult * state["atr"]

    return _update


def chandelier_exit(
    atr_period: int = 22,
    mult: float = 3.0,
) -> Callable:
    """
    Chandelier exit (Le Beau): for longs, stop = highest_high(N) - mult × ATR.
    For shorts, lowest_low(N) + mult × ATR.

    We track the running high (or low) since entry rather than over a window —
    simpler to implement bar-by-bar without a history reference. This means it
    behaves like a "Chandelier from entry" — fine for most uses.
    """
    state = {
        "extreme_high": None,
        "extreme_low": None,
        "atr": None,
        "prev_close": None,
        "alpha": 1.0 / atr_period,
    }

    def _update(position, bar) -> float | None:
        # Track extremes since position opened
        h, l, c = bar["High"], bar["Low"], bar["Close"]
        state["extreme_high"] = h if state["extreme_high"] is None else max(state["extreme_high"], h)
        state["extreme_low"] = l if state["extreme_low"] is None else min(state["extreme_low"], l)

        # Update ATR
        if state["prev_close"] is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - state["prev_close"]), abs(l - state["prev_close"]))
        state["atr"] = tr if state["atr"] is None else ((1 - state["alpha"]) * state["atr"] + state["alpha"] * tr)
        state["prev_close"] = c

        if position.side == "long":
            return state["extreme_high"] - mult * state["atr"]
        else:
            return state["extreme_low"] + mult * state["atr"]

    return _update


def breakeven_after_R(
    risk_pts: float,
    move_to_R: float = 1.0,
    plus_pts: float = 0.0,
) -> Callable:
    """
    Move stop to break-even (entry price) once price moves `move_to_R × risk_pts`
    in your favour. Optional `plus_pts` adds a small buffer beyond entry so the
    stop is slightly profitable rather than scratch.

    Once moved, returns None on subsequent bars (stop stays at break-even forever).
    Combine with another trailing function if you want continued tightening.
    """
    state = {"moved": False}

    def _update(position, bar) -> float | None:
        if state["moved"]:
            return None
        target_move = move_to_R * risk_pts
        if position.side == "long":
            if bar["High"] - position.entry_price >= target_move:
                state["moved"] = True
                return position.entry_price + plus_pts
        else:
            if position.entry_price - bar["Low"] >= target_move:
                state["moved"] = True
                return position.entry_price - plus_pts
        return None

    return _update


def combine(*fns: Callable) -> Callable:
    """
    Combine multiple trailing functions. Each is called in order; the FIRST
    one that returns a non-None value wins. Useful for layering, e.g.
        combine(breakeven_after_R(20), atr_trailing(history))
    moves to break-even at 1R, then trails with ATR thereafter.
    """
    def _combined(position, bar) -> float | None:
        for f in fns:
            result = f(position, bar)
            if result is not None:
                return result
        return None
    return _combined
