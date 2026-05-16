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


# ---- Stochastic Oscillator ---------------------------------------------
def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
               smooth_k: int = 3):
    """
    Stochastic Oscillator: %K and %D.

    %K_raw = 100 × (close - lowest_low) / (highest_high - lowest_low)  over k_period
    %K = SMA(%K_raw, smooth_k)  — typically 3
    %D = SMA(%K, d_period)      — typically 3

    Values 0-100. Below 20 = oversold; above 80 = overbought.
    Returns (%K, %D).
    """
    lowest = df["Low"].rolling(k_period).min()
    highest = df["High"].rolling(k_period).max()
    raw_k = 100 * (df["Close"] - lowest) / (highest - lowest).replace(0, np.nan)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return k.fillna(50.0), d.fillna(50.0)


# ---- ADX + Directional Indicators --------------------------------------
def adx(df: pd.DataFrame, period: int = 14):
    """
    Average Directional Index. Returns (ADX, +DI, -DI).

    ADX measures TREND STRENGTH (0-100, higher = stronger trend) regardless
    of direction. +DI > -DI = upward momentum; -DI > +DI = downward.
    Common interpretation: ADX > 25 = trending; ADX < 20 = ranging.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # When both up- and down-moves on same bar, the smaller one is zeroed
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm.shift(0)] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Wilder's smoothing
    alpha = 1.0 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx_line.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


# ---- Williams %R --------------------------------------------------------
def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R — momentum oscillator, -100 to 0. Like RSI but on a different
    scale. Above -20 = overbought; below -80 = oversold.

    %R = -100 × (highest_high - close) / (highest_high - lowest_low) over period
    """
    highest = df["High"].rolling(period).max()
    lowest = df["Low"].rolling(period).min()
    wr = -100 * (highest - df["Close"]) / (highest - lowest).replace(0, np.nan)
    return wr.fillna(-50.0)


# ---- On-Balance Volume --------------------------------------------------
def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume — cumulative volume signed by close direction.

    OBV[t] = OBV[t-1] + Volume[t]  if Close[t] > Close[t-1]
             OBV[t-1] - Volume[t]  if Close[t] < Close[t-1]
             OBV[t-1]              if unchanged

    Trend confirmation: rising OBV during rising price = healthy trend.
    Divergence (price up, OBV flat/down) = potential reversal.
    """
    if "Volume" not in df.columns:
        return pd.Series(0.0, index=df.index)
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


# ---- Rate of Change (ROC) ----------------------------------------------
def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """
    Rate of Change as a percentage. ROC = (price - price[period ago]) / price[period ago] × 100.
    Pure momentum, positive = uptrending, negative = downtrending.
    """
    return (series / series.shift(period) - 1.0) * 100


# ---- Keltner Channels --------------------------------------------------
def keltner_channels(df: pd.DataFrame, ema_period: int = 20,
                     atr_period: int = 10, mult: float = 2.0):
    """
    Keltner Channels: like Bollinger Bands but the bands are centred on EMA
    and width is a multiple of ATR (not standard deviation).

    Returns (middle, upper, lower).
    Touches of the bands signal volatility extremes; less prone to "expanding
    BB during consolidation" issue because it doesn't use std.
    """
    middle = ema(df["Close"], ema_period)
    atr_val = atr(df, atr_period)
    upper = middle + mult * atr_val
    lower = middle - mult * atr_val
    return middle, upper, lower


# ---- Money Flow Index --------------------------------------------------
def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index — like RSI but volume-weighted. 0-100.
    Below 20 = oversold; above 80 = overbought.

    typical_price = (H+L+C)/3; money_flow = typical_price × volume
    Positive flow = days where typical_price rose; negative flow = fell.
    MFI = 100 - 100 / (1 + sum(positive_flow) / sum(negative_flow))
    """
    if "Volume" not in df.columns:
        return pd.Series(50.0, index=df.index)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"].fillna(0)
    tp_diff = tp.diff()
    pos_flow = mf.where(tp_diff > 0, 0.0)
    neg_flow = mf.where(tp_diff < 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mr = pos_sum / neg_sum.replace(0, np.nan)
    out = 100 - 100 / (1 + mr)
    # When neg_sum is 0:
    #   if pos_sum > 0  → MFI = 100 (all positive flow, max overbought)
    #   if pos_sum == 0 → MFI = 50  (truly flat)
    out = out.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
    out = out.mask((neg_sum == 0) & (pos_sum == 0), 50.0)
    return out.fillna(50.0)


# ---- Parabolic SAR -----------------------------------------------------
def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02,
                  af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """
    Parabolic SAR — trailing-stop / trend-reversal indicator.

    SAR sits below price in uptrends, above in downtrends, accelerating toward
    price each bar. When price crosses SAR, the trend "flips". Useful as a
    trailing stop level or trend filter.
    """
    high, low = df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(df)
    sar = np.zeros(n)
    if n < 2:
        return pd.Series(sar, index=df.index)

    # Initialise: assume uptrend at bar 0
    bull = True
    sar[0] = low[0]
    ep = high[0]   # extreme point
    af = af_start

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if bull:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[max(0, i - 2)])
            if low[i] < sar[i]:
                # Reversal: trend flips to bear
                bull = False
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af_max, af + af_step)
        else:
            sar[i] = prev_sar - af * (prev_sar - ep)
            sar[i] = max(sar[i], high[i - 1], high[max(0, i - 2)])
            if high[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af_max, af + af_step)
    return pd.Series(sar, index=df.index)


# ---- Heikin Ashi candles -----------------------------------------------
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLC to Heikin Ashi candles — smoothed view of trend.

    HA_Close = (O + H + L + C) / 4
    HA_Open  = (prev_HA_Open + prev_HA_Close) / 2
    HA_High  = max(H, HA_Open, HA_Close)
    HA_Low   = min(L, HA_Open, HA_Close)

    A run of green HA candles indicates a sustained uptrend (and vice versa).
    Useful as a trend filter (e.g. "only trade long when last 3 HA bars green").
    """
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0
    ha_open = pd.Series(index=df.index, dtype=float)
    ha_open.iloc[0] = (df["Open"].iloc[0] + df["Close"].iloc[0]) / 2.0
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["Low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame({
        "HA_Open": ha_open, "HA_High": ha_high,
        "HA_Low": ha_low, "HA_Close": ha_close,
    })


# ---- Classic Pivot Points (daily) --------------------------------------
def pivot_points_classic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic floor-trader Pivot Points — daily levels derived from yesterday's
    H/L/C. Returns a DataFrame with columns: P, R1, R2, R3, S1, S2, S3,
    indexed by date.

    Each level acts as potential support/resistance for the next session.
    """
    daily = df.resample("1D", label="left", closed="left").agg({
        "High": "max", "Low": "min", "Close": "last",
    }).dropna()
    # Prev session's HLC drives today's pivots
    prev = daily.shift(1)
    p = (prev["High"] + prev["Low"] + prev["Close"]) / 3.0
    r1 = 2 * p - prev["Low"]
    s1 = 2 * p - prev["High"]
    r2 = p + (prev["High"] - prev["Low"])
    s2 = p - (prev["High"] - prev["Low"])
    r3 = prev["High"] + 2 * (p - prev["Low"])
    s3 = prev["Low"] - 2 * (prev["High"] - p)
    return pd.DataFrame({"P": p, "R1": r1, "R2": r2, "R3": r3,
                         "S1": s1, "S2": s2, "S3": s3}).dropna()


# ---- Multi-timeframe resampling ---------------------------------------
# pandas frequency aliases that our intervals map to.
_RESAMPLE_RULE = {
    "1m": "1min", "2m": "2min", "5m": "5min", "10m": "10min",
    "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h",
    "1d": "1D", "1wk": "1W",
}


def to_higher_timeframe(
    history: pd.DataFrame,
    target_interval: str,
    include_partial: bool = False,
) -> pd.DataFrame:
    """
    Resample a base-timeframe OHLCV DataFrame to a higher timeframe.

    Use this inside a strategy's `on_bar` / `proposed_direction` to get HTF
    context for free, without needing a separate IG fetch:

        htf = to_higher_timeframe(history, "1h")
        htf_ema200 = ema(htf["Close"], 200).iloc[-1]
        if cur_close > htf_ema200:
            # in HTF uptrend...

    Args:
        history: base DataFrame with O/H/L/C (and optionally Volume, Spread).
        target_interval: one of our standard intervals — "1h", "4h", "1d", etc.
        include_partial: if True, includes the still-forming last HTF bar
            (its OHLC values reflect only the base bars seen so far).
            Default False — drops the partial last bar to prevent any chance
            of look-ahead. **Use True only when you genuinely want intra-bar
            HTF info (e.g., "is today's high above yesterday's close?").**

    Returns the resampled DataFrame, aligned to clock boundaries (e.g., 1h
    bars start on the hour). Standard pandas semantics — empty bars dropped.

    Look-ahead safety: when `include_partial=False`, we drop the last HTF
    bar if the latest base bar falls inside it (meaning the HTF bar isn't
    yet complete). This is the safe default.
    """
    if target_interval not in _RESAMPLE_RULE:
        raise ValueError(f"Unknown target_interval {target_interval!r}. "
                         f"Valid: {list(_RESAMPLE_RULE)}")
    rule = _RESAMPLE_RULE[target_interval]

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in history.columns:
        agg["Volume"] = "sum"
    if "Spread" in history.columns:
        agg["Spread"] = "mean"

    # label="left" / closed="left": bar timestamps are bar START times;
    # bars include their start moment, exclude the next.
    resampled = history.resample(rule, label="left", closed="left").agg(agg)
    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])

    if not include_partial and len(resampled) > 0:
        # Determine if the most recent HTF bar is "complete" — i.e., the base
        # data extends past the end of that HTF window. If not, drop it.
        last_htf_start = resampled.index[-1]
        htf_period = pd.tseries.frequencies.to_offset(rule)
        last_htf_end = last_htf_start + htf_period
        last_base_time = history.index[-1]
        # An HTF bar at 14:00 with 1h period ends at 15:00. We consider it
        # complete only if the latest base bar's timestamp is >= 15:00.
        if last_base_time < last_htf_end:
            resampled = resampled.iloc[:-1]

    return resampled


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
