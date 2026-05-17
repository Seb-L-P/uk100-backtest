"""
Timezone normalisation for fetched data.

Every fetcher returns a DataFrame; we run it through `to_trading_tz` so
the index is in the user's trading timezone (config.TRADING_TZ, default
Europe/London), with the tz dropped at the end so the rest of the
backtester sees tz-naive timestamps consistently.

Why this exists:
    yfinance returns market-local time (e.g. America/New_York for US
    stocks). EODHD intraday returns UTC. IG returns UTC. Strategies
    compare bar.time() against session_open/close, which are wall-clock
    times in the trader's mind. Without normalisation, a UK trader
    backtesting TSLA on yfinance sees session_open=8:30 land at 03:30 GMT
    (which is wrong for them in every way).

Pattern:
    The fetcher knows the SOURCE timezone of its raw data:
      - yfinance:  market-local (each ticker's exchange tz)
      - EODHD:     UTC
      - IG:        UTC
    It localises to that source tz, converts to the user's trading tz,
    then strips tz info so downstream code stays simple.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def to_trading_tz(df: pd.DataFrame, source_tz: str = "UTC") -> pd.DataFrame:
    """
    Convert df.index to the active trading timezone (config.TRADING_TZ),
    then strip tz info so downstream code (which expects tz-naive
    timestamps) keeps working.

    Args:
        df:        DataFrame with a DatetimeIndex (any tz state).
        source_tz: The timezone the raw data is IN, if the index is tz-naive.
                   Ignored when the index is already tz-aware. For yfinance
                   this is typically the exchange's local tz; for EODHD /
                   IG it's "UTC".

    No-ops if df is empty.
    """
    if df.empty:
        return df

    import config as _cfg
    target = getattr(_cfg, "TRADING_TZ", "Europe/London")

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(source_tz, ambiguous="infer", nonexistent="shift_forward")
    idx = idx.tz_convert(target)
    # Strip tz — strategies compare `.time()` directly and expect naive ts.
    idx = idx.tz_localize(None)

    df = df.copy()
    df.index = idx
    return df
