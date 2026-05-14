"""
Historical data fetcher for IG (spread bet, demo account).

Why use this instead of yfinance:
  - IG's data is exactly what your trades will see live (matches the broker's
    feed including their adjustments).
  - Up to ~2 years of intraday at any timeframe (yfinance caps at 60 days for
    intraday), depending on your weekly allowance.
  - Real bid/ask spread observable per bar — we use the mid for backtesting
    OHLC, but spread can be measured separately for cost calibration.

Caveats:
  - IG has a weekly historical-data allowance (10,000 points/week on demo by
    default). We aggressively cache to parquet to avoid burning through it.
  - Authentication uses session tokens that expire. We re-authenticate lazily
    on token expiry rather than per-call.
  - Demo and live API have different base URLs; controlled by IG_ENV in .env.

Credentials are loaded from a .env file at the project root. See README for
the expected variables. Nothing in this module ever prints your password.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from config import DATA_CACHE


# ---- Credential loading ------------------------------------------------
def _load_env() -> dict:
    """Load IG creds from .env file. Raises with a clear message if missing."""
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

    creds = {
        "username": os.getenv("IG_USERNAME"),
        "password": os.getenv("IG_PASSWORD"),
        "api_key": os.getenv("IG_API_KEY"),
        "account_id": os.getenv("IG_ACCOUNT_ID"),
        "env": (os.getenv("IG_ENV") or "demo").upper(),
    }
    missing = [k for k, v in creds.items() if not v and k != "account_id"]
    if missing:
        raise RuntimeError(
            f"IG credentials missing in .env: {missing}. "
            f"Expected file at {env_path} with IG_USERNAME, IG_PASSWORD, "
            f"IG_API_KEY, IG_ACCOUNT_ID, IG_ENV (=demo|live)."
        )
    return creds


# ---- Lazy session ------------------------------------------------------
_session_cache: dict = {"service": None, "logged_in_at": None}


def _get_service():
    """Return a logged-in IGService, reusing if recent (sessions last ~6h)."""
    try:
        from trading_ig import IGService
    except ImportError:
        raise RuntimeError(
            "trading-ig not installed. Run: pip install -r requirements.txt"
        )

    # Reuse existing session if it's < 4 hours old (IG sessions last 6h)
    if _session_cache["service"] is not None and _session_cache["logged_in_at"] is not None:
        age = dt.datetime.now() - _session_cache["logged_in_at"]
        if age < dt.timedelta(hours=4):
            return _session_cache["service"]

    creds = _load_env()
    service = IGService(
        username=creds["username"],
        password=creds["password"],
        api_key=creds["api_key"],
        acc_type=creds["env"],   # "DEMO" or "LIVE"
    )
    try:
        service.create_session()
    except Exception as e:
        # Don't echo credentials in the error
        msg = str(e)
        # Strip any password-like patterns from the message just in case
        raise RuntimeError(
            f"IG login failed: {msg}. Common causes: wrong username (it's the "
            f"login username NOT email), wrong password, wrong API key, or "
            f"IG_ENV mismatch (using DEMO key against live URL or vice versa). "
            f"Run scripts/ig_test.py for a clearer diagnostic."
        ) from None

    _session_cache["service"] = service
    _session_cache["logged_in_at"] = dt.datetime.now()
    return service


# ---- Symbol + resolution mapping --------------------------------------
# Map our generic ticker shortcuts to IG epic codes.
# IG epic format: IX.D.FTSE.DAILY.IP — index, daily-funded, IG product code.
# Common UK 100 epics:
#   IX.D.FTSE.DAILY.IP   — UK 100 daily-funded spread bet (USE THIS for spread bet)
#   IX.D.FTSE.CFD.IP     — UK 100 CFD product
#   IX.D.FTSE.MONTH1.IP  — UK 100 first-month future
EPIC_MAP = {
    "^FTSE": "IX.D.FTSE.DAILY.IP",
    "FTSE": "IX.D.FTSE.DAILY.IP",
    "UK100": "IX.D.FTSE.DAILY.IP",
}

Resolution = Literal["1m", "2m", "5m", "10m", "15m", "30m", "1h", "2h", "3h", "4h", "1d", "1wk"]

# Map our generic intervals to trading-ig's expected resolution strings.
# Note: trading-ig 0.0.20+ uses pandas-style frequency aliases internally,
# not the older IG REST API constants ("DAY", "HOUR", etc.).
RESOLUTION_MAP = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
    "1d": "1D",
    "1wk": "1W",
}


def _to_epic(ticker: str) -> str:
    """Convert our shortcuts to IG epic. Pass-through if it already looks like an epic."""
    if ticker.startswith("IX.") or ticker.startswith("CS.") or ticker.startswith("KA."):
        return ticker  # Already an epic
    return EPIC_MAP.get(ticker.upper(), ticker)


def _to_resolution(interval: str) -> str:
    if interval not in RESOLUTION_MAP:
        raise ValueError(f"Unknown interval {interval!r}. Valid: {list(RESOLUTION_MAP)}")
    return RESOLUTION_MAP[interval]


# ---- Cache ------------------------------------------------------------
def _cache_path(epic: str, resolution: str, num_points: int) -> Path:
    safe = epic.replace(".", "_")
    return DATA_CACHE / f"ig_{safe}_{resolution}_{num_points}pts.parquet"


# ---- Public API -------------------------------------------------------
def fetch_ig(
    ticker: str = "^FTSE",
    interval: str = "1d",
    num_points: int = 1000,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from IG.

    Returns a DataFrame indexed by timezone-naive datetime with columns:
    Open, High, Low, Close, Volume — same shape as the yfinance fetcher.

    OHLC values are the MID of bid/ask (since spread bet has both sides).
    Volume is the last-traded volume reported by IG; it's a useful proxy
    for activity but is the underlying market's volume, not your broker's flow.

    Args:
        ticker: shortcut like "^FTSE" or "UK100", or a raw IG epic.
        interval: same vocabulary as yfinance ("1m", "15m", "1h", "1d", ...).
        num_points: number of bars to fetch (working backwards from now).
                    IG has a weekly allowance — start small and increase only
                    if needed.
        use_cache: read from parquet cache if present.
    """
    epic = _to_epic(ticker)
    resolution = _to_resolution(interval)
    cache_file = _cache_path(epic, resolution, num_points)

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return _validate(df)

    service = _get_service()
    try:
        response = service.fetch_historical_prices_by_epic_and_num_points(
            epic=epic, resolution=resolution, numpoints=num_points,
        )
    except Exception as e:
        raise RuntimeError(
            f"IG historical-prices request failed for {epic}/{resolution}/{num_points}pts: {e}. "
            f"Common causes: wrong epic (try IX.D.FTSE.DAILY.IP for UK 100 spread bet), "
            f"weekly allowance exceeded (check via scripts/ig_test.py), "
            f"or session expired (try restarting Python)."
        ) from None

    raw_df = response["prices"]
    df = _ig_to_ohlcv(raw_df)
    df = _validate(df)
    df.to_parquet(cache_file)

    # Print weekly allowance status so user can manage their budget
    allowance = response.get("allowance", {})
    if allowance:
        rem = allowance.get("remainingAllowance", "?")
        total = allowance.get("totalAllowance", "?")
        reset = allowance.get("allowanceExpiry", "?")
        print(f"[ig] Weekly allowance: {rem}/{total} remaining "
              f"(resets in {reset}s).")
    return df


def _ig_to_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert IG's multi-level prices DataFrame to flat OHLCV.

    IG returns columns: ('bid', 'Open'), ('bid', 'High'), ('bid', 'Low'),
    ('bid', 'Close'), ('ask', 'Open'), ..., ('last', 'Open'), ...,
    plus 'Volume' as a top-level column.

    We use mid (= (bid + ask) / 2) for OHLC because that's the most honest
    "price" for backtesting purposes — actual spread is modelled separately
    in our cost model.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        out = pd.DataFrame(index=raw.index)
        out["Open"] = (raw[("bid", "Open")] + raw[("ask", "Open")]) / 2
        out["High"] = (raw[("bid", "High")] + raw[("ask", "High")]) / 2
        out["Low"] = (raw[("bid", "Low")] + raw[("ask", "Low")]) / 2
        out["Close"] = (raw[("bid", "Close")] + raw[("ask", "Close")]) / 2
        # Volume might be in the multi-index or as a top-level column
        if ("last", "Volume") in raw.columns:
            out["Volume"] = raw[("last", "Volume")]
        elif "Volume" in raw.columns:
            out["Volume"] = raw["Volume"]
        else:
            out["Volume"] = 0.0
    else:
        # Already flat — just ensure column casing
        out = raw.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume",
        })

    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Same checks as the yfinance fetcher: drop NaN OHLC, sort, dedupe."""
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
        print(f"[ig_fetcher] dropped {n} bars with inconsistent OHLC")
        df = df.loc[~inconsistent]
    return df


def check_connection() -> dict:
    """
    Verify creds + connection. Returns a dict of account info if OK.
    Raises RuntimeError with a clear message otherwise.
    """
    service = _get_service()
    accounts = service.fetch_accounts()
    return {
        "ok": True,
        "n_accounts": len(accounts),
        "account_ids": list(accounts["accountId"]) if "accountId" in accounts.columns else [],
        "account_types": list(accounts["accountType"]) if "accountType" in accounts.columns else [],
    }
