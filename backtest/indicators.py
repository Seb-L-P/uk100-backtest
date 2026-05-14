"""
Indicators and pattern detectors.

Kept separate from strategies so they can be unit-tested and reused.
Every function here MUST be "online-safe" — it can only use data up to and
including the current bar. No `.shift(-1)` or anything that peeks forward.

Naming convention:
  - Functions that return a Series aligned with the input index (e.g. `rsi`,
    `vwap`, `atr`) — read the LAST value with `.iloc[-1]`.
  - Functions that return a tuple of Series (e.g. `bollinger`, `macd`) — same,
    but unpack first.
  - Pattern detectors like `detect_fvg` return a value or None for the bar
    being inspected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["bullish", "bearish"]


# ---- Moving averages ----------------------------------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. `adjust=False` matches the conventional
    recursive EMA formula used in trading platforms."""
    return series.ewm(span=period, adjust=False).mean()


# ---- ATR ----------------------------------------------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — Wilder's RMA approximation via SMA."""
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---- RSI (Wilder's) ----------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's Relative Strength Index. Uses smoothed (EMA-style with alpha=1/period)
    averages of gains and losses to avoid divide-by-zero on flat periods.

    Returns a Series of values in [0, 100]. <30 = oversold, >70 = overbought.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # When avg_loss is 0 (no down moves), the formula gives NaN. Distinguish:
    #   - if avg_gain > 0 → RSI = 100 (all-gains, max overbought)
    #   - if avg_gain == 0 → RSI = 50 (truly flat — neutral)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out.fillna(50.0)  # first bars (before .diff()) → neutral 50


# ---- Bollinger Bands ---------------------------------------------------
def bollinger(series: pd.Series, period: int = 20, mult: float = 2.0):
    """
    Bollinger Bands. Returns (middle, upper, lower) Series.

    middle = SMA(period); std = rolling std; bands = middle ± mult * std.
    """
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower


# ---- MACD --------------------------------------------------------------
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD = EMA(fast) - EMA(slow). Signal = EMA(MACD, signal). Histogram = MACD - Signal.
    Returns (macd_line, signal_line, histogram).
    """
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---- VWAP (session-anchored) -------------------------------------------
def vwap(df: pd.DataFrame, session_reset: bool = True) -> pd.Series:
    """
    Volume-Weighted Average Price.

    Uses the bar's typical price (H+L+C)/3 weighted by Volume.
    If `session_reset=True`, VWAP resets at the start of each trading day —
    this is what intraday traders mean by "VWAP" without further qualifier.
    If False, computes a cumulative VWAP over all data.

    Note: Yahoo's index volume is exchange volume for the underlying components,
    not the broker's flow on the CFD. It's a reasonable proxy but not perfect.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    volume = df["Volume"].replace(0, np.nan).fillna(1.0)  # avoid div-by-zero
    pv = typical_price * volume

    if not session_reset:
        return pv.cumsum() / volume.cumsum()

    # Reset each calendar day
    day_id = pd.Series(df.index.normalize(), index=df.index)
    cum_pv = pv.groupby(day_id).cumsum()
    cum_v = volume.groupby(day_id).cumsum()
    return cum_pv / cum_v


# ---- Volatility regime classifier --------------------------------------
def volatility_regime(df: pd.DataFrame, atr_period: int = 14,
                      lookback: int = 100) -> pd.Series:
    """
    Returns a Series of {-1, 0, 1} — -1 = low vol, 0 = normal, 1 = high vol.

    Compares current ATR to its rolling median over `lookback` bars.
    Useful as a filter: many mean-reversion strategies work in low/normal vol,
    many breakout strategies work in high vol.
    """
    a = atr(df, atr_period)
    rolling_median = a.rolling(lookback).median()
    ratio = a / rolling_median
    regime = pd.Series(0, index=df.index, dtype=int)
    regime[ratio < 0.75] = -1
    regime[ratio > 1.25] = 1
    return regime


# ---- Fair Value Gap -----------------------------------------------------
@dataclass
class FVG:
    """A 3-bar Fair Value Gap (imbalance).

    For a bullish FVG (created on bar i):
        bar[i-2].high < bar[i].low      # there's a "gap" between bars i-2 and i
        zone = (bar[i-2].high, bar[i].low)
        Price must trade DOWN into this zone to "fill" it.

    For a bearish FVG (created on bar i):
        bar[i-2].low > bar[i].high
        zone = (bar[i].high, bar[i-2].low)
        Price must trade UP into this zone to "fill" it.
    """
    direction: Direction
    created_at: pd.Timestamp
    creator_bar_index: int          # index of bar i (the third bar)
    zone_low: float
    zone_high: float
    creator_bar_low: float          # bar i values — for stop placement
    creator_bar_high: float
    size_points: float              # zone_high - zone_low

    def contains(self, price: float) -> bool:
        return self.zone_low <= price <= self.zone_high

    def is_filled_by(self, bar_low: float, bar_high: float) -> bool:
        """Has any subsequent bar's range completely crossed the zone?"""
        if self.direction == "bullish":
            # Filled when price trades all the way through the zone (bar_low <= zone_low)
            return bar_low <= self.zone_low
        else:
            return bar_high >= self.zone_high

    def is_touched_by(self, bar_low: float, bar_high: float) -> bool:
        """Has a subsequent bar at least touched the near edge of the zone?"""
        if self.direction == "bullish":
            # near edge for a bullish FVG is zone_high (price retracing down to it)
            return bar_low <= self.zone_high
        else:
            return bar_high >= self.zone_low

    @property
    def near_edge(self) -> float:
        """The edge price retraces TO (entry trigger price)."""
        return self.zone_high if self.direction == "bullish" else self.zone_low

    @property
    def far_edge(self) -> float:
        """The edge price has to break THROUGH to fully fill the FVG."""
        return self.zone_low if self.direction == "bullish" else self.zone_high


def detect_fvg(df: pd.DataFrame, i: int) -> FVG | None:
    """
    Check if bar at integer position `i` (the third bar in a 3-bar pattern)
    completes a Fair Value Gap. `df` is the full OHLC frame.

    Returns the FVG object or None.
    """
    if i < 2:
        return None
    b0 = df.iloc[i - 2]  # first bar
    b2 = df.iloc[i]      # third bar (current)
    # Bullish: b0.high < b2.low
    if b0["High"] < b2["Low"]:
        return FVG(
            direction="bullish",
            created_at=df.index[i],
            creator_bar_index=i,
            zone_low=float(b0["High"]),
            zone_high=float(b2["Low"]),
            creator_bar_low=float(b2["Low"]),
            creator_bar_high=float(b2["High"]),
            size_points=float(b2["Low"] - b0["High"]),
        )
    # Bearish: b0.low > b2.high
    if b0["Low"] > b2["High"]:
        return FVG(
            direction="bearish",
            created_at=df.index[i],
            creator_bar_index=i,
            zone_low=float(b2["High"]),
            zone_high=float(b0["Low"]),
            creator_bar_low=float(b2["Low"]),
            creator_bar_high=float(b2["High"]),
            size_points=float(b0["Low"] - b2["High"]),
        )
    return None
