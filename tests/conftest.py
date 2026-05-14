"""
Shared test fixtures.

The synthetic_ohlc fixture returns a deterministic small OHLC frame we can
reason about exactly. We use this rather than real market data wherever
possible because the assertions become "this number must equal that number"
rather than "the result is approximately reasonable".
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable for tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlc() -> pd.DataFrame:
    """
    20 bars of deterministic OHLC data, indexed daily.

    Designed so that:
      - Bar 5-7 form a bullish FVG (b5.high=104 < b7.low=106)
      - Bar 11-13 form a bearish FVG (b11.low=110 > b13.high=108)
      - General uptrend so directional strategies have something to trade
    """
    # bar:  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19
    opens = [100,101,102,103,103,104,105,107,107,108,109,110,111,109,108,109,110,111,112,113]
    highs = [101,102,103,104,104,104,107,108,109,110,111,112,111,110,109,110,111,112,113,114]
    lows  = [99, 100,101,102,102,103,104,106,106,107,108,110,108,107,107,108,109,110,111,112]
    closes= [101,102,103,103,104,104,107,107,108,109,110,111,109,108,108,109,110,111,112,113]
    vols  = [1000] * 20

    idx = pd.date_range("2024-01-02", periods=20, freq="D")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


@pytest.fixture
def trending_ohlc() -> pd.DataFrame:
    """100 bars of a clean uptrend — useful for testing trend strategies."""
    idx = pd.date_range("2024-01-02", periods=100, freq="D")
    closes = [100 + i * 0.5 for i in range(100)]
    opens = [c - 0.2 for c in closes]
    highs = [c + 0.4 for c in closes]
    lows = [c - 0.5 for c in closes]
    vols = [1000] * 100
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )
