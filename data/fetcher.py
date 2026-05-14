"""
Historical data fetcher for FTSE 100 (UK 100).

Source: yfinance — free, decent for daily back ~25 years, intraday is limited
(60d for 2m, 30d for 1m). For longer intraday history we'll need a paid source
later (Databento, IG's own API once you have credentials, etc.).

Caches everything to parquet so repeated runs are fast and offline-capable.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

from config import DATA_CACHE

Interval = Literal["1m", "2m", "5m", "15m", "30m", "1h", "1d", "1wk"]

# yfinance ticker for FTSE 100 cash index
FTSE_TICKER = "^FTSE"


def _cache_path(ticker: str, interval: str, start: str, end: str) -> Path:
    safe_ticker = ticker.replace("^", "")
    return DATA_CACHE / f"{safe_ticker}_{interval}_{start}_{end}.parquet"


def fetch(
    ticker: str = FTSE_TICKER,
    interval: Interval = "1d",
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
    source: str = "yfinance",
    ig_num_points: int = 5000,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for `ticker` at `interval` between `start` and `end`.

    Returns a DataFrame indexed by timezone-naive datetime with columns:
    Open, High, Low, Close, Volume.

    Args:
        source: "yfinance" (free, 60-day intraday limit) or "ig" (uses IG demo
                API, requires .env credentials, ~2 years intraday).
        ig_num_points: when source="ig", how many bars to fetch (working back
                       from now). IG has a weekly allowance — start small.
        start, end: only used by yfinance. IG fetches `ig_num_points` bars
                    working backward from "now".

    Caches results as parquet under data/cache/.
    """
    if source == "ig":
        from data.ig_fetcher import fetch_ig
        return fetch_ig(ticker=ticker, interval=interval,
                        num_points=ig_num_points, use_cache=use_cache)

    end = end or dt.date.today().isoformat()
    start = start or _default_start_for(interval)

    cache_file = _cache_path(ticker, interval, start, end)
    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return _validate(df)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} {interval} {start}->{end}. "
            f"Note: yfinance limits intraday history (1m: 7d trailing, 2m: 60d, "
            f"5m/15m/30m/1h: 60d for some intervals)."
        )

    # yfinance sometimes returns a MultiIndex on columns even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = _validate(df)

    df.to_parquet(cache_file)
    return df


def _default_start_for(interval: str) -> str:
    """Maximise free history per interval, respecting yfinance limits."""
    today = dt.date.today()
    if interval == "1m":
        return (today - dt.timedelta(days=7)).isoformat()
    if interval in ("2m", "5m", "15m", "30m", "90m"):
        return (today - dt.timedelta(days=59)).isoformat()
    if interval == "1h":
        return (today - dt.timedelta(days=729)).isoformat()
    # daily and above — go deep
    return "2000-01-01"


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity checks: no NaNs in OHLC, monotonic index, no duplicate timestamps."""
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    bad = df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    if bad.any():
        # Drop NaN bars rather than forward-fill — forward-fill creates fake bars
        df = df.loc[~bad]
    # Spot-check: high >= low, high >= max(open, close), low <= min(open, close)
    inconsistent = (
        (df["High"] < df["Low"])
        | (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    )
    if inconsistent.any():
        # Drop bad bars; log count
        n = int(inconsistent.sum())
        print(f"[fetcher] dropped {n} bars with inconsistent OHLC")
        df = df.loc[~inconsistent]
    return df


if __name__ == "__main__":
    daily = fetch(interval="1d", start="2000-01-01")
    print(f"Daily FTSE 100: {len(daily)} bars from {daily.index[0]} to {daily.index[-1]}")
    print(daily.tail())
