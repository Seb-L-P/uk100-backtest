"""
Multi-timeframe (MTF) analysis API for strategies.

Goal: let any strategy "look up" at higher timeframes without manual
resampling on every bar. Pro traders constantly check the 1h/4h/daily for
trend and context; the backtester should make that one method call.

Public API
----------
    from backtest.mtf import current
    htf_trend = current().trend("1h")               # "up" / "down" / "neutral"
    htf_ema   = current().ema("4h", 50)             # latest 4h EMA-50 value
    htf_rsi   = current().rsi("1h", 14)
    htf_atr   = current().atr("1h", 14)
    swing_hi  = current().swing_high("1d", 20)      # last 20 daily highs
    htf_df    = current().htf("1h")                 # raw HTF DataFrame

How it works
------------
At backtest start, the engine creates an `MTFContext` bound to the FULL
data, then on every bar calls `mtf.set_now(timestamp)`. Strategies access
the active context via `current()`.

Each HTF lookup returns values ONLY from bars that closed strictly before
`now`. The look-ahead guard relies on `to_higher_timeframe(include_partial
=False)`, which drops any HTF bar still in-progress at the data's edge.

Performance
-----------
For each (interval, indicator, period) triple, we compute the indicator
once on the FULL precomputed HTF DataFrame and cache the result. Per-bar
lookups are then O(log n) bisect-slices on the cached series rather than
O(n) recomputation. So even on a 200k-bar run, calling
`mtf.trend("1h")` every bar adds negligible overhead.

Universal HTF filter
--------------------
`HtfFilterWrapper(strategy)` wraps any existing strategy with an HTF
trend filter — blocks signals/proposed directions that disagree with the
higher-timeframe trend. Zero strategy-code changes required.
"""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

from backtest.indicators import (
    to_higher_timeframe,
    _RESAMPLE_RULE,
    ema as _ema_fn,
    rsi as _rsi_fn,
    atr as _atr_fn,
)


# ---- Active context (set by engine) ----------------------------------
_active: "MTFContext | None" = None


def current() -> "MTFContext":
    """
    Return the active MTFContext, or a no-op stub if none is set.

    Strategies call this inside `on_bar` / `proposed_direction` to look up
    higher-timeframe state. When no context is active (e.g., unit tests
    calling the strategy directly), the stub returns "neutral" / NaN so
    strategies that DEPEND on MTF don't crash — they just degrade gracefully
    to "no HTF information available".
    """
    return _active if _active is not None else _NULL_MTF


def set_active(ctx: "MTFContext | None") -> None:
    """Engine-only: install the active MTF context for this backtest."""
    global _active
    _active = ctx


# ---- The real context -------------------------------------------------
TrendLabel = Literal["up", "down", "neutral"]


class MTFContext:
    """
    Per-backtest multi-timeframe lookup cache.

    Created once at backtest start with the FULL data. Each bar, the engine
    calls `set_now(t)` to update the as-of cursor. Strategies then call
    `trend(...)`, `ema(...)`, etc. — all lookups respect the cursor and
    return only data from CLOSED HTF bars strictly before `t`.
    """

    def __init__(self, full_data: pd.DataFrame):
        self._full = full_data
        self._now: Optional[pd.Timestamp] = None
        # Cached resampled DataFrames per interval (computed once per interval)
        self._htf_cache: dict[str, pd.DataFrame] = {}
        # Cached indicator Series keyed by (interval, name, *args)
        self._ind_cache: dict[tuple, pd.Series] = {}

    # ---- Engine interface --------------------------------------------
    def set_now(self, t: pd.Timestamp) -> None:
        """Engine calls this each bar to update the as-of cursor."""
        self._now = t

    # ---- Strategy-facing API -----------------------------------------
    def htf(self, interval: str) -> pd.DataFrame:
        """
        Higher-timeframe OHLCV containing only HTF bars that have FULLY
        CLOSED strictly before `now`.

        Look-ahead safety: an HTF bar with start time `t_htf` and length `P`
        covers `[t_htf, t_htf + P)`. It's only "closed" once we've observed a
        base bar at time >= `t_htf + P`. So when `now = t`, the latest closed
        HTF bar has start `t_htf <= t - P`. The bar containing `t` itself is
        still in-progress and excluded.

        Example: base 15m, HTF 1h, now=09:15. The 09:00 HTF bar (covering
        09:00–10:00) is still forming — only the 08:00 HTF bar and earlier
        are returned.
        """
        full = self._ensure_htf(interval)
        if self._now is None or full.empty:
            return full
        period = self._period(interval)
        cutoff = self._now - period
        # `<= cutoff` keeps a bar that started exactly one period before `now`
        # (and thus closed exactly at `now`).
        return full[full.index <= cutoff]

    def trend(self, interval: str, ema_period: int = 50) -> TrendLabel:
        """
        Classic SMC "is the higher TF trending?" check.

        Rules:
            up    — HTF close > EMA AND EMA sloping up (EMA[-1] > EMA[-3])
            down  — HTF close < EMA AND EMA sloping down
            neutral — anything else, including insufficient history

        Args:
            interval: e.g. "1h", "4h", "1d"
            ema_period: EMA length on the HTF close. Default 50.
        """
        ema_series = self._htf_indicator(interval, "ema", ema_period)
        if ema_series is None or len(ema_series) < 3:
            return "neutral"
        htf = self.htf(interval)
        if htf.empty:
            return "neutral"
        ema_now = float(ema_series.iloc[-1])
        ema_then = float(ema_series.iloc[-3])
        if pd.isna(ema_now) or pd.isna(ema_then):
            return "neutral"
        close = float(htf["Close"].iloc[-1])
        if close > ema_now and ema_now > ema_then:
            return "up"
        if close < ema_now and ema_now < ema_then:
            return "down"
        return "neutral"

    def ema(self, interval: str, period: int) -> float:
        """Latest closed-bar EMA on the higher timeframe."""
        s = self._htf_indicator(interval, "ema", period)
        return self._latest_or_nan(s)

    def rsi(self, interval: str, period: int = 14) -> float:
        """Latest closed-bar RSI on the higher timeframe."""
        s = self._htf_indicator(interval, "rsi", period)
        return self._latest_or_nan(s)

    def atr(self, interval: str, period: int = 14) -> float:
        """Latest closed-bar ATR on the higher timeframe (volatility regime)."""
        s = self._htf_indicator(interval, "atr", period)
        return self._latest_or_nan(s)

    def swing_high(self, interval: str, lookback: int) -> float:
        """Highest HIGH in the last `lookback` closed HTF bars. NaN if not enough data."""
        htf = self.htf(interval)
        if len(htf) < lookback:
            return float("nan")
        return float(htf["High"].iloc[-lookback:].max())

    def swing_low(self, interval: str, lookback: int) -> float:
        """Lowest LOW in the last `lookback` closed HTF bars."""
        htf = self.htf(interval)
        if len(htf) < lookback:
            return float("nan")
        return float(htf["Low"].iloc[-lookback:].min())

    def above_ema(self, interval: str, ema_period: int = 50) -> bool:
        """Convenience: True iff the latest HTF close is above its EMA."""
        e = self.ema(interval, ema_period)
        htf = self.htf(interval)
        if htf.empty or pd.isna(e):
            return False
        return float(htf["Close"].iloc[-1]) > e

    # ---- Internals ---------------------------------------------------
    def _ensure_htf(self, interval: str) -> pd.DataFrame:
        """Compute (once) and cache the FULL HTF DataFrame for `interval`."""
        if interval not in self._htf_cache:
            self._htf_cache[interval] = to_higher_timeframe(
                self._full, interval, include_partial=False
            )
        return self._htf_cache[interval]

    @staticmethod
    def _period(interval: str) -> pd.Timedelta:
        """Convert an interval string ('1h', '4h', '1d') to a Timedelta.

        Note: must NOT go via `pd.tseries.frequencies.to_offset(rule)` →
        `pd.Timedelta()` — newer pandas (≥2.2) refuses to convert a
        DateOffset like `<Day>` directly to Timedelta. Pass the rule
        STRING to Timedelta, which handles all our fixed-period rules
        ("1min", "15min", "1h", "1D", "1W") natively.
        """
        rule = _RESAMPLE_RULE.get(interval, interval)
        return pd.Timedelta(rule)

    def _htf_indicator(self, interval: str, name: str, *args) -> pd.Series | None:
        """
        Compute (once) and cache an indicator's full series on the HTF.
        Then return the SLICE up to `now`. Empty Series if HTF has no data.
        """
        key = (interval, name, args)
        if key not in self._ind_cache:
            full_htf = self._ensure_htf(interval)
            if full_htf.empty:
                self._ind_cache[key] = pd.Series(dtype=float)
            elif name == "ema":
                self._ind_cache[key] = _ema_fn(full_htf["Close"], args[0])
            elif name == "rsi":
                self._ind_cache[key] = _rsi_fn(full_htf["Close"], args[0])
            elif name == "atr":
                self._ind_cache[key] = _atr_fn(full_htf, args[0])
            else:
                raise ValueError(f"Unknown HTF indicator {name!r}")
        full = self._ind_cache[key]
        if self._now is None or full.empty:
            return full
        period = self._period(interval)
        cutoff = self._now - period
        return full[full.index <= cutoff]

    def _latest_or_nan(self, s: pd.Series | None) -> float:
        if s is None or s.empty:
            return float("nan")
        s = s.dropna()
        return float(s.iloc[-1]) if not s.empty else float("nan")


class _NullMTF:
    """
    Stub used when no MTFContext is active (e.g., unit tests that call a
    strategy's on_bar directly without going through the engine).

    All methods return "neutral"/NaN/empty so MTF-aware strategies still
    work — they just see "no HTF info available" and fall back to their
    base-TF behaviour.
    """
    def htf(self, interval: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def trend(self, interval: str, ema_period: int = 50) -> TrendLabel:
        return "neutral"

    def ema(self, *a, **k): return float("nan")
    def rsi(self, *a, **k): return float("nan")
    def atr(self, *a, **k): return float("nan")
    def swing_high(self, *a, **k): return float("nan")
    def swing_low(self, *a, **k): return float("nan")
    def above_ema(self, *a, **k): return False


_NULL_MTF = _NullMTF()


# Strategy-composition lives in backtest/graph.py — DecisionGraph +
# GraphOrchestrator. They use this MTFContext for look-ahead-safe HTF
# lookups but own the trigger/supporter/veto wiring themselves.
