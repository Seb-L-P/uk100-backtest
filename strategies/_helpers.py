"""
Shared helpers for strategies. Underscore prefix means "private to this package".

These exist so every strategy uses the SAME risk sizing, the SAME session-window
logic, and the SAME swing-point definitions. Inconsistencies between strategies
on these are a sneaky source of "this strategy looks better than that one"
results that are actually just different sizing rules.
"""
from __future__ import annotations

from datetime import time as _time
from typing import Literal

import pandas as pd

from config import ACCOUNT


def risk_based_stake(equity_gbp: float, risk_pts: float,
                     risk_pct: float | None = None,
                     min_stake: float = 0.10,
                     price: float | None = None,
                     existing_notional_gbp: float = 0.0) -> float:
    """
    Position size in £/point such that hitting the stop loses `risk_pct` of
    equity. `risk_pts` is the distance from entry to stop in index points.

    If `price` is provided, the stake is ALSO capped to fit within the
    account leverage cap (notional = stake × price). This matters on
    high-priced indices like FTSE 100 (~10,000), where 1% risk on a tight
    stop can produce a stake that breaches 20× leverage. Without this cap,
    the broker raises mid-backtest and the run dies.

    Example: £10k equity, 1% risk, 20-point stop → £100 risk / 20pt = £5/pt.
             At FTSE 10,200, notional = £5 × 10,200 = £51k. Within 20× cap (£200k). OK.

    Example with tight stop:
             £10k equity, 1% risk, 4-point stop → £100 / 4pt = £25/pt.
             At FTSE 10,200, notional = £25 × 10,200 = £255k. Breaches £200k cap.
             With `price=10200`, function caps to ~£19.6/pt = £200k notional.
             Effective risk drops to £19.6 × 4pt = £78 instead of £100.
             Better to take a smaller-than-planned trade than crash the run.
    """
    risk_pct = risk_pct if risk_pct is not None else ACCOUNT.risk_per_trade_pct
    if risk_pts <= 0:
        return 0.0
    risk_gbp = equity_gbp * risk_pct
    stake = max(min_stake, round(risk_gbp / risk_pts, 2))

    if price is not None and price > 0:
        # Cap by available leverage
        max_total_notional = equity_gbp * ACCOUNT.leverage_cap
        available_notional = max_total_notional - existing_notional_gbp
        if available_notional <= 0:
            return 0.0  # No leverage left, can't open
        # 0.99 safety margin to avoid edge-case floating-point breaches
        max_stake_by_leverage = (available_notional / price) * 0.99
        if stake * price > available_notional:
            stake = max(min_stake, round(max_stake_by_leverage, 2))

    return stake


def atr_threshold(history: pd.DataFrame, atr_mult: float,
                  atr_period: int = 14, fallback_pts: float = 1.0) -> float:
    """
    Convert an ATR-multiplier threshold to absolute points using the bar's
    current ATR. Used by strategies that want SCALE-INVARIANT thresholds
    (works the same on FTSE 10k, AAPL 200, BTC 60k, regardless of price).

    Example:
        min_gap = atr_threshold(history, atr_mult=0.5)  # half an ATR
    On FTSE with ATR=10pt → min_gap=5pt. On AAPL with ATR=2pt → min_gap=1pt.
    Same VOLATILITY-RELATIVE threshold; auto-adapts to instrument.

    Returns `fallback_pts` if ATR is unavailable (early in series).
    """
    from backtest.indicators import atr as _atr_fn
    if len(history) < atr_period + 2:
        return fallback_pts
    a = _atr_fn(history, atr_period).iloc[-1]
    if pd.isna(a) or a <= 0:
        return fallback_pts
    return atr_mult * float(a)


def in_session(now: _time, session_open: _time, session_close: _time) -> bool:
    """True if `now` falls in the [open, close] window inclusive."""
    return session_open <= now <= session_close


def is_first_bar_of_day(history: pd.DataFrame, i: int) -> bool:
    """True if bar i is the first bar of a new trading day (date changed)."""
    if i == 0:
        return True
    return history.index[i].date() != history.index[i - 1].date()


# ---- Swing point detection ---------------------------------------------
def trailing_swing(history: pd.DataFrame, i: int, lookback: int,
                   kind: Literal["high", "low"]) -> float:
    """
    Online-safe trailing swing high/low:
      The highest high (or lowest low) over the last `lookback` bars ENDING
      at bar i-1 (i.e. EXCLUDING the current bar). Used as the "level" that
      the current bar may sweep through.

    Returns NaN if not enough history.
    """
    if i < lookback:
        return float("nan")
    window = history.iloc[i - lookback:i]
    col = "High" if kind == "high" else "Low"
    return float(window[col].max() if kind == "high" else window[col].min())


# ---- ATR (also in indicators.py — re-export for convenience) -----------
def atr(history: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = history["High"], history["Low"], history["Close"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()
