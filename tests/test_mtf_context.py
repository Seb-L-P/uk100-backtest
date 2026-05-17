"""
Tests for the MTF context — the lookup primitive that lets strategies
peek at higher timeframes safely.

Critical guarantees:
  1. Look-ahead safety — HTF data for time `t` includes only HTF bars
     fully closed at or before `t`. The bar containing `t` is hidden.
  2. Indicator series cached on full HTF data, then sliced — yields the
     same numeric value as computing on the sliced HTF data directly.
  3. _NullMTF returns neutral defaults when no context is active.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.mtf import MTFContext, current, set_active


def _synthetic_data(n_days: int = 5, interval_min: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_price = 1000.0
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
         + pd.Timedelta(hours=h, minutes=m)
         for d in range(n_days) for h in range(9, 17)
         for m in range(0, 60, interval_min)]
    )
    rows = []
    for _ in idx:
        delta = rng.normal(0, 1.0)
        base_price += delta * 0.5
        rows.append({
            "Open": base_price - 0.5,
            "High": base_price + 1.5,
            "Low": base_price - 1.5,
            "Close": base_price + 0.5,
            "Volume": 1000,
        })
    return pd.DataFrame(rows, index=idx)


def test_htf_returns_only_closed_bars():
    """HTF bar containing `now` must NOT be returned (it's partial)."""
    data = _synthetic_data(n_days=3)
    mtf = MTFContext(data)
    t = pd.Timestamp("2024-01-01 10:15")
    mtf.set_now(t)
    htf = mtf.htf("1h")
    # The 10:00 HTF bar is partly formed at 10:15 → must be excluded.
    if not htf.empty:
        assert htf.index[-1] <= pd.Timestamp("2024-01-01 09:00")


def test_htf_indicator_matches_direct_computation():
    data = _synthetic_data(n_days=10)
    mtf = MTFContext(data)
    t = pd.Timestamp("2024-01-08 15:30")
    mtf.set_now(t)
    via_mtf = mtf.ema("1h", 10)
    from backtest.indicators import to_higher_timeframe, ema as ema_fn
    full_htf = to_higher_timeframe(data, "1h", include_partial=False)
    full_ema = ema_fn(full_htf["Close"], 10)
    cutoff = t - pd.Timedelta("1h")
    expected = float(full_ema[full_ema.index <= cutoff].dropna().iloc[-1])
    assert via_mtf == pytest.approx(expected, rel=1e-9)


def test_trend_neutral_when_insufficient_history():
    data = _synthetic_data(n_days=1)
    mtf = MTFContext(data)
    mtf.set_now(data.index[2])
    assert mtf.trend("1h", ema_period=50) == "neutral"


def test_null_mtf_returns_neutral_defaults():
    set_active(None)
    ctx = current()
    assert ctx.trend("1h") == "neutral"
    assert np.isnan(ctx.ema("1h", 50))
    assert np.isnan(ctx.rsi("1h", 14))
    assert ctx.above_ema("1h", 50) is False
