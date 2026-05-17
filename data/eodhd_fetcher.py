"""
Historical data fetcher for EODHD (eodhd.com).

Why use this:
  - 30+ years of intraday data on most assets including FTSE 100.
  - No per-week allowance like IG; the paid plans give 100k API calls/day,
    far more than we'll ever use.
  - Cleanly covers daily + 1m/5m/1h intraday via separate REST endpoints.

Caveats:
  - The FREE tier only includes a handful of test symbols (mostly AAPL.US)
    and is daily-only — sufficient to verify integration but not useful for
    real research. The EOD+Intraday All World Extended plan ($29.99/mo,
    sometimes discounted via SAVEONTRADING) unlocks FTSE 100.
  - EODHD's native intraday intervals are 1m, 5m, 1h. We resample 5m → 15m
    and 5m → 30m client-side if those are requested.
  - EODHD enforces a 600-day-per-request limit on intraday data. We chunk
    requests transparently — a 30-year intraday fetch becomes ~18 sequential
    API calls, each counting against your 100k/day allowance (so essentially
    unlimited for normal use).
  - No bid/ask spread data on indices — `Spread` column NOT emitted, so the
    backtester falls back to the active cost profile.

Credentials are loaded from .env. Required: `EODHD_API_KEY=...`.
Optional: `EODHD_FTSE_SYMBOL=FTSE.INDX` if the default doesn't work for you.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from config import DATA_CACHE


# ---- Credential loading ------------------------------------------------
def _load_api_key() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise RuntimeError(
            "python-dotenv not installed. Run: pip install -r requirements.txt"
        )
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    key = os.getenv("EODHD_API_KEY")
    if not key:
        raise RuntimeError(
            "EODHD_API_KEY missing from .env. Sign up at eodhd.com, paste your "
            "API key as EODHD_API_KEY in the .env file."
        )
    return key


# ---- Symbol + interval mapping ----------------------------------------
# Map our generic ticker shortcuts to EODHD symbols.
#
# IMPORTANT: EODHD's "EOD+Intraday All World Extended" plan ($29.99/mo) does
# NOT include direct cash-index access (UKX.INDX). Indices are a separate
# add-on. So we default to the ETF proxy ISF.LSE which:
#   - is available on the All World Extended plan
#   - tracks FTSE 100 cash to within ~0.05% on daily moves
#   - has real intraday bid/ask (the index itself doesn't anyway)
# Trade-off: IG's UK 100 spread bet is priced off the cash index, so using
# the ETF's prices introduces a small (~5-10 bps) tracking error. Usually
# negligible relative to spread costs.
#
# If you upgrade to a plan that includes indices, change the value to
# "UKX.INDX" — the cash index data IS more accurate to what IG quotes.
# Each mapping is (EODHD symbol, price_scale_factor). When the user asks for
# UK100 (which gets routed to the ISF.LSE proxy quoted in PENCE), the data
# is scaled up by 10× so it sits at the same price level as the real FTSE
# 100 cash index. That way the UK100 cost profile (1.5pt spread = ~2 bps
# at 8000) actually represents 2 bps of friction, not 19 bps.
#
# Without the scale factor, applying a fixed 1.5pt spread to an 800-priced
# ETF over-charges friction by 10× and structurally guarantees a losing
# backtest regardless of strategy quality. Confirmed bug — Apr 2026.
EPIC_MAP: dict[str, tuple[str, float]] = {
    "^FTSE":   ("ISF.LSE", 10.0),
    "FTSE":    ("ISF.LSE", 10.0),
    "UKX":     ("ISF.LSE", 10.0),  # official FTSE 100 ticker, but EODHD
                                    # routes it to the ETF proxy on this plan
    "UK100":   ("ISF.LSE", 10.0),
    "FTSE100": ("ISF.LSE", 10.0),
}

# Map our interval shortcuts to EODHD's native intraday intervals.
# EODHD supports 1m, 5m, 1h natively; we resample 5m → 15m and 5m → 30m.
Interval = Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk"]
EODHD_NATIVE_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "5m",   # we'll resample 5m → 15m client-side
    "30m": "5m",   # we'll resample 5m → 30m client-side
    "1h": "1h",
    "1d": None,    # daily uses the /eod endpoint instead of /intraday
    "1wk": None,
}


def _to_symbol(ticker: str) -> tuple[str, float]:
    """
    Map ticker shortcut to (EODHD symbol, price_scale_factor).
    Unknown tickers pass through verbatim with scale 1.0.
    """
    return EPIC_MAP.get(ticker.upper(), (ticker, 1.0))


# ---- Cache -------------------------------------------------------------
def _cache_path(symbol: str, interval: str, num_points: int) -> Path:
    # `v2` prefix invalidates pre-scale-aware caches (Apr 2026). Without
    # this, an ISF.LSE cache from before the proxy-rescale fix would be
    # loaded as-is and silently apply the 10x-too-tight friction bug.
    safe = symbol.replace(".", "_").replace("^", "")
    return DATA_CACHE / f"eodhd_v2_{safe}_{interval}_{num_points}pts.parquet"


# ---- Date helpers ------------------------------------------------------
def _intraday_range(interval: str, num_points: int) -> tuple[int, int]:
    """
    Compute (from_ts, to_ts) for an intraday fetch, given the desired number
    of bars at the resolution. Returns UNIX seconds.

    Notes:
      - Liquid intraday markets trade ~6.5-8h/day, ~252 days/year, so calendar
        time is ~5x longer than trading time. We use 5x as the safety factor.
      - The chunked fetcher walks back in 599-day windows and stops early when
        EODHD returns no data (i.e. we've gone past the asset's history start).
        So overshooting `from_ts` is harmless.
      - Capped at 30 years which is EODHD's maximum coverage for most assets.
    """
    to_ts = int(dt.datetime.now().timestamp())
    minutes_per_bar = {"1m": 1, "5m": 5, "15m": 5, "30m": 5, "1h": 60}.get(interval, 1)
    # Wall clock minutes needed = (bars × bar minutes) × calendar-to-trading ratio
    wall_clock_minutes = num_points * minutes_per_bar * 5
    # Floor: 7 days (so weekend runs always include at least one trading session)
    # Ceiling: 30 years (EODHD's max intraday coverage)
    THIRTY_YEARS_MIN = 30 * 365 * 24 * 60
    SEVEN_DAYS_MIN = 7 * 24 * 60
    wall_clock_minutes = max(SEVEN_DAYS_MIN, min(wall_clock_minutes, THIRTY_YEARS_MIN))
    from_ts = to_ts - wall_clock_minutes * 60
    return from_ts, to_ts


def _eod_range(num_points: int) -> tuple[str, str]:
    """For daily/weekly data, return (from_date, to_date) as ISO strings."""
    today = dt.date.today()
    days_back = max(num_points * 2, 30)  # conservative: 2x for weekends/holidays
    start = today - dt.timedelta(days=days_back)
    return start.isoformat(), today.isoformat()


# ---- Public API --------------------------------------------------------
def fetch_eodhd(
    ticker: str = "^FTSE",
    interval: str = "1d",
    num_points: int = 1000,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from EODHD.

    Returns a DataFrame indexed by timezone-naive datetime with columns:
    Open, High, Low, Close, Volume — same shape as the yfinance fetcher.

    No `Spread` column (EODHD doesn't expose bid/ask history on indices),
    so the backtester falls back to the flat config.spread_points default.

    Args:
        ticker: shortcut like "^FTSE" or "UK100", or a raw EODHD symbol like
                "FTSE.INDX" or "AAPL.US".
        interval: "1m", "5m", "15m", "30m", "1h", "1d", or "1wk". 15m and 30m
                  are resampled from 5m.
        num_points: number of bars to fetch (working backwards from now).
        use_cache: read from parquet cache if present.
    """
    symbol, price_scale = _to_symbol(ticker)
    # Cache key includes the scale so a switched mapping doesn't poison the
    # cached file. A separate cache file per (symbol, interval, num, scale).
    cache_file = _cache_path(symbol, interval, num_points)
    if use_cache and cache_file.exists():
        df = _validate(pd.read_parquet(cache_file))
        # Cache files store already-scaled OHLCV, so just return.
        return df

    api_key = _load_api_key()

    # Daily / weekly → /eod endpoint
    if interval in ("1d", "1wk"):
        df = _fetch_eod(symbol, num_points, api_key, period="d" if interval == "1d" else "w")
    else:
        # Intraday → /intraday endpoint at the closest native resolution
        native = EODHD_NATIVE_INTERVAL.get(interval)
        if native is None:
            raise ValueError(f"Unsupported interval {interval!r}")
        df = _fetch_intraday(symbol, native, num_points, api_key)
        # Resample if user asked for a non-native interval
        if interval != native:
            df = _resample(df, interval)

    # Apply the proxy → real-instrument scale factor (UK100 alias is at
    # ISF.LSE ÷10, so scale=10 lifts prices back into FTSE-index range).
    if price_scale != 1.0:
        for col in ("Open", "High", "Low", "Close"):
            if col in df.columns:
                df[col] = df[col] * price_scale

    df = _validate(df)
    df.to_parquet(cache_file)
    return df


def _fetch_eod(symbol: str, num_points: int, api_key: str, period: str = "d") -> pd.DataFrame:
    from_date, to_date = _eod_range(num_points)
    url = "https://eodhd.com/api/eod/" + symbol
    params = {
        "api_token": api_key,
        "fmt": "json",
        "from": from_date,
        "to": to_date,
        "period": period,
    }
    resp = _get(url, params)
    rows = resp.json()
    if not rows:
        raise RuntimeError(
            f"EODHD returned no daily data for {symbol} {from_date}..{to_date}. "
            f"Common causes: wrong symbol (try FTSE.INDX or AAPL.US), or your "
            f"plan doesn't include this asset (free tier is limited to a few "
            f"test symbols)."
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Use 'adjusted_close' for Close so dividend/split adjustments are baked in;
    # but for index data, adjusted_close == close so it doesn't matter.
    out = pd.DataFrame({
        "Open": df["open"].astype(float),
        "High": df["high"].astype(float),
        "Low": df["low"].astype(float),
        "Close": df["close"].astype(float),
        "Volume": df.get("volume", 0).astype(float),
    })
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    return out.tail(num_points)


def _fetch_intraday(symbol: str, native_interval: str, num_points: int,
                    api_key: str) -> pd.DataFrame:
    """
    Fetch intraday data, automatically chunking requests into ≤600-day
    windows (EODHD's per-request limit). Stitches results together.

    Strategy:
      - Compute the wall-clock window we need to cover (with weekend safety).
      - Walk backwards from now in 599-day chunks until we've collected
        enough bars or we've gone far enough back.
      - Stop early if a chunk returns no data (we've hit the asset's earliest
        available history — e.g. TSLA IPO'd 2010, BTC began 2014).
    """
    from_ts, to_ts = _intraday_range(native_interval, num_points)
    # EODHD's documented limit is 600 calendar days per request. We use 599
    # to be safe against off-by-one edge cases.
    MAX_DAYS_PER_REQUEST = 599
    MAX_SECONDS_PER_REQUEST = MAX_DAYS_PER_REQUEST * 24 * 3600

    chunks: list[pd.DataFrame] = []
    total_rows = 0
    chunk_to = to_ts
    earliest_reached = False
    chunk_count = 0
    MAX_CHUNKS = 50  # safety: cap at ~33 years even if user asked for more

    while chunk_to > from_ts and not earliest_reached and chunk_count < MAX_CHUNKS:
        chunk_from = max(from_ts, chunk_to - MAX_SECONDS_PER_REQUEST)
        url = "https://eodhd.com/api/intraday/" + symbol
        params = {
            "api_token": api_key,
            "fmt": "json",
            "interval": native_interval,
            "from": chunk_from,
            "to": chunk_to,
        }
        resp = _get(url, params)
        rows = resp.json()
        chunk_count += 1

        if not rows:
            # Empty chunk: usually means we've gone past the asset's history.
            # Stop walking back further.
            if not chunks:
                # No data at all on first try — surface a useful error
                from_iso = dt.datetime.fromtimestamp(chunk_from).isoformat(timespec="minutes")
                to_iso = dt.datetime.fromtimestamp(chunk_to).isoformat(timespec="minutes")
                raise RuntimeError(
                    f"EODHD returned no intraday data for {symbol} {native_interval} "
                    f"from {from_iso} to {to_iso}.\n"
                    f"  HTTP {resp.status_code}, body preview: {resp.text[:200]}\n"
                    f"  Common causes: symbol doesn't have intraday on EODHD, plan "
                    f"doesn't include intraday for this asset class, or the date "
                    f"range hit a no-trading window."
                )
            earliest_reached = True
            break

        df_chunk = pd.DataFrame(rows)
        df_chunk["datetime"] = pd.to_datetime(df_chunk["datetime"])
        chunks.append(df_chunk)
        total_rows += len(df_chunk)

        # Step the to-timestamp back by one full window
        chunk_to = chunk_from

    if not chunks:
        raise RuntimeError(f"EODHD: no data fetched for {symbol}")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"]).set_index("datetime").sort_index()
    out = pd.DataFrame({
        "Open": df["open"].astype(float),
        "High": df["high"].astype(float),
        "Low": df["low"].astype(float),
        "Close": df["close"].astype(float),
        "Volume": df.get("volume", 0).fillna(0).astype(float),
    })
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    return out.tail(num_points)


def _resample(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Resample base 5m bars up to 15m or 30m."""
    rule_map = {"15m": "15min", "30m": "30min"}
    rule = rule_map[target]
    out = df.resample(rule, label="left", closed="left").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    })
    return out.dropna()


def _get(url: str, params: dict) -> requests.Response:
    """GET wrapper with a clean error if the API rejects."""
    try:
        resp = requests.get(url, params=params, timeout=30)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"EODHD network error: {e}") from None
    if resp.status_code == 401:
        raise RuntimeError(
            "EODHD: invalid API key (HTTP 401). Check EODHD_API_KEY in .env."
        )
    if resp.status_code == 402:
        raise RuntimeError(
            "EODHD: this endpoint requires a paid plan (HTTP 402). The free "
            "tier doesn't include this symbol or interval. Upgrade at eodhd.com."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            f"EODHD: symbol not found (HTTP 404). Check the symbol format — "
            f"FTSE 100 cash index = FTSE.INDX, ETF = ISF.LSE."
        )
    if resp.status_code != 200:
        raise RuntimeError(f"EODHD HTTP {resp.status_code}: {resp.text[:200]}")
    return resp


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Same checks as the other fetchers: drop NaN OHLC, sort, dedupe."""
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    bad = df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    if bad.any():
        df = df.loc[~bad]
    inconsistent = (
        (df["High"] < df["Low"])
        | (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    )
    if inconsistent.any():
        n = int(inconsistent.sum())
        print(f"[eodhd_fetcher] dropped {n} bars with inconsistent OHLC")
        df = df.loc[~inconsistent]
    return df


def check_connection() -> dict:
    """
    Verify the API key works by fetching a tiny sample of AAPL.US (available
    on the free tier). Returns a dict with status info.
    """
    api_key = _load_api_key()
    df = _fetch_eod("AAPL.US", num_points=5, api_key=api_key)
    return {
        "ok": True,
        "test_symbol": "AAPL.US",
        "bars_returned": len(df),
        "latest_date": str(df.index[-1].date()) if not df.empty else None,
    }
