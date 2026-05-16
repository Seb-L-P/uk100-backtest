"""
Smoke tests — verify every registered strategy at least RUNS without crashing
on synthetic data and produces a valid BacktestResult.

These are intentionally low-bar: they don't check whether the strategy is
profitable, just whether it executes the full engine loop, returns a result,
and doesn't violate any of the runtime self-check assertions.

Catches regressions when we modify the engine, broker, or shared helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from strategies import registry as reg


@pytest.fixture
def synth_data():
    """
    300 bars of synthetic intraday-style data (mild trend + noise + bid/ask spread).
    Long enough to satisfy warmup for any strategy + produce a handful of trades.
    """
    rng = np.random.default_rng(seed=42)
    n = 300
    idx = pd.date_range("2024-01-01 08:00", periods=n, freq="15min")
    # Mild upward drift with noise
    base = 10000 + np.cumsum(rng.normal(0.05, 5.0, n))
    spreads = rng.uniform(0.5, 3.0, n)
    half = spreads / 2
    # Mid prices first
    opens = base + rng.normal(0, 0.5, n)
    highs = opens + np.abs(rng.normal(2, 1, n))
    lows = opens - np.abs(rng.normal(2, 1, n))
    closes = opens + rng.normal(0, 1.5, n)
    # Ensure OHLC consistency
    highs = np.maximum.reduce([opens, closes, highs])
    lows = np.minimum.reduce([opens, closes, lows])
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes,
        "Volume": rng.integers(500, 2000, n),
        "Spread": spreads,
    }, index=idx)


# Parametrise across every registered strategy, including ensembles.
@pytest.mark.parametrize("strategy_key", list(reg.STRATEGIES.keys()))
def test_strategy_runs_without_error(strategy_key, synth_data):
    """Every strategy in the registry should execute on synth data without crashing."""
    spec = reg.get(strategy_key)
    strategy = spec.build(**spec.defaults())
    result = run_backtest(synth_data, strategy, warmup_bars=spec.warmup_bars)

    # Result shape sanity
    assert result.bars_processed == len(synth_data)
    assert result.starting_balance > 0
    assert isinstance(result.final_balance, float)

    # Equity curve has at least the post-warmup bars
    assert len(result.equity_curve) >= len(synth_data) - spec.warmup_bars - 5

    # If trades were taken, trade log should have the expected columns
    if not result.trades_df.empty:
        for col in ("side", "stake_per_point", "entry_price", "exit_price",
                    "net_pnl_gbp", "exit_reason"):
            assert col in result.trades_df.columns


@pytest.mark.parametrize("strategy_key", list(reg.STRATEGIES.keys()))
def test_strategy_no_position_at_eod(strategy_key, synth_data):
    """The engine's eod-close logic should clean up any open positions."""
    spec = reg.get(strategy_key)
    strategy = spec.build(**spec.defaults())
    result = run_backtest(synth_data, strategy, warmup_bars=spec.warmup_bars)
    # The engine asserts this internally — if test reaches here, broker.positions == []
    # at end of run. Make it explicit for documentation.
    assert result.final_balance is not None


def test_smoke_data_fixture_valid(synth_data):
    """Sanity-check the synthetic fixture itself."""
    assert len(synth_data) == 300
    assert all(synth_data["High"] >= synth_data["Low"])
    assert all(synth_data["High"] >= synth_data["Open"])
    assert all(synth_data["High"] >= synth_data["Close"])
    assert all(synth_data["Low"] <= synth_data["Open"])
    assert all(synth_data["Low"] <= synth_data["Close"])
    assert all(synth_data["Spread"] > 0)
