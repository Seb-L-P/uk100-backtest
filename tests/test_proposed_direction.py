"""
Tests for proposed_direction across all strategies.

Verifies that:
  1. Every registered strategy has a proposed_direction method
  2. It returns one of "long", "short", "none" — never anything else
  3. On synth data, at least one strategy produces a non-"none" vote somewhere
     (otherwise ensembles can never trigger)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies import registry as reg


@pytest.fixture
def synth_data():
    """500 bars of synthetic data with enough volatility for signals."""
    rng = np.random.default_rng(seed=42)
    n = 500
    idx = pd.date_range("2024-01-01 08:00", periods=n, freq="15min")
    base = 10000 + np.cumsum(rng.normal(0.1, 8.0, n))
    opens = base + rng.normal(0, 0.5, n)
    highs = opens + np.abs(rng.normal(3, 1.5, n))
    lows = opens - np.abs(rng.normal(3, 1.5, n))
    closes = opens + rng.normal(0, 2.0, n)
    highs = np.maximum.reduce([opens, closes, highs])
    lows = np.minimum.reduce([opens, closes, lows])
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes,
        "Volume": rng.integers(500, 2000, n),
        "Spread": rng.uniform(0.5, 3.0, n),
    }, index=idx)


@pytest.mark.parametrize("strategy_key", list(reg.STRATEGIES.keys()))
def test_proposed_direction_exists_and_returns_valid_value(strategy_key, synth_data):
    """Every strategy must implement proposed_direction with the right shape."""
    spec = reg.get(strategy_key)
    strategy = spec.build(**spec.defaults())
    assert hasattr(strategy, "proposed_direction"), \
        f"{strategy_key} missing proposed_direction"

    # Probe at several positions along the timeline
    for end in (100, 200, 300, 499):
        direction = strategy.proposed_direction(synth_data.iloc[:end])
        assert direction in ("long", "short", "none"), \
            f"{strategy_key} returned invalid direction {direction!r} at bar {end}"


def test_at_least_one_strategy_proposes_signal(synth_data):
    """
    Across the 500 bars, at least one non-ensemble strategy should propose
    at least one non-'none' direction. Otherwise ensembles can never trigger.
    """
    non_ensemble_keys = [
        k for k in reg.STRATEGIES.keys()
        if k not in ("vote_meanrev", "vote_trend", "filter_fvg_rsi")
    ]
    any_signal = False
    for k in non_ensemble_keys:
        spec = reg.get(k)
        strat = spec.build(**spec.defaults())
        # Walk through history bar-by-bar; bail when we find the first signal
        for end in range(50, len(synth_data) + 1, 10):
            if strat.proposed_direction(synth_data.iloc[:end]) != "none":
                any_signal = True
                break
        if any_signal:
            break
    assert any_signal, "No strategy proposed any direction on synth data"


def test_ensemble_no_longer_uses_mock_broker(synth_data):
    """
    Regression check: ensembles should NOT crash on FVG-based children
    after the proposed_direction refactor — the bug that prompted this work.
    """
    from backtest.engine import run_backtest

    for ensemble_key in ("vote_meanrev", "vote_trend", "filter_fvg_rsi"):
        spec = reg.get(ensemble_key)
        ensemble = spec.build(**spec.defaults())
        # Should run without AttributeError on pending_orders
        result = run_backtest(synth_data, ensemble, warmup_bars=spec.warmup_bars)
        assert result.bars_processed == len(synth_data)
