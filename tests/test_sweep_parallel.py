"""
Tests for the parallelised discovery sweep (`backtest.sweep_runner`).

Two guarantees are tested:

  1. DETERMINISM — the same seed + same SearchSpace produces a bit-identical
     final leaderboard whether trials run serially (`n_jobs=1`) or across a
     process pool (`n_jobs=4`). Graph sampling stays serial; only the (pure)
     backtests fan out, so results must not depend on `n_jobs`.

  2. SPEEDUP — on a machine with enough cores, `n_jobs=4` finishes a
     50-trial sweep in well under half the wall-clock of `n_jobs=1`.

Both run on synthetic 5m OHLC data so the assertions don't depend on a
network fetch or cached parquet.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from backtest.sweep_runner import run_sweep
from backtest.sweep_space import SearchSpace
from config import PROFILES


# ---- Synthetic data + search space helpers ------------------------------
def _make_synth_5m(n_bars: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic 5m OHLCV — a mean-reverting sinusoid plus a random walk.

    The blend gives crossover/mean-reversion strategies real trades to make,
    so the leaderboard exercises finite Sharpe values and the ranking sort
    (not just a wall of disqualified -inf rows).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 08:00", periods=n_bars, freq="5min")
    t = np.arange(n_bars)
    base = 8000.0
    cycle = 60.0 * np.sin(t / 40.0)              # slow oscillation → reversion
    drift = np.cumsum(rng.normal(0, 4.0, n_bars))  # random walk → trends
    close = base + cycle + drift
    open_ = np.empty(n_bars)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0, 3.0, n_bars)) + 1.0
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.integers(800, 1200, n_bars).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


def _make_space() -> SearchSpace:
    """A small, crash-free space of fast strategies at 5m data TF."""
    triggers = ["sma", "rsi_revert", "bb_revert", "macd_cross", "donchian"]
    return SearchSpace(
        triggers_pool=list(triggers),
        supporters_pool=list(triggers),
        vetoes_pool=list(triggers),
        max_supporters=2,
        max_vetoes=1,
        data_tf="5m",
        trigger_tf_options=["5m", "15m", "30m"],
    )


def _fingerprint(result) -> dict:
    """Reduce a SweepResult to comparable, order-sensitive leaderboard data.

    -inf compares equal to itself in Python, so disqualified rows are fine.
    `compute_metrics` never emits NaN Sharpe (it uses -inf), so plain tuple
    equality is a safe bit-for-bit check.
    """
    def rows(trials):
        return [
            (t.description, t.metrics.sharpe, t.metrics.n_trades,
             round(t.metrics.return_pct, 9))
            for t in trials
        ]
    return {
        "is": rows(result.is_trials),
        "val": rows(result.val_top),
        "oos": rows(result.oos_top),
    }


# ---- 1. Determinism: serial vs parallel produce identical leaderboards ----
def test_serial_and_parallel_leaderboards_identical():
    data = _make_synth_5m(2400, seed=7)
    space = _make_space()
    costs = PROFILES["UK100"]

    common = dict(
        n_trials=24, top_k=8, top_m=4, min_trades=5,
        warmup_bars=30, costs=costs, seed=42,
    )

    serial = run_sweep(data, space, n_jobs=1, **common)
    parallel = run_sweep(data, space, n_jobs=4, **common)

    assert _fingerprint(serial) == _fingerprint(parallel), (
        "Parallel sweep produced a different leaderboard than the serial "
        "sweep for the same seed — determinism is broken."
    )
    # Sanity: the sweep actually did meaningful work, not an all-disqualified
    # wall (which would make the determinism check vacuous).
    assert any(np.isfinite(t.metrics.sharpe) for t in serial.is_trials)


def test_n_jobs_auto_matches_serial():
    """n_jobs=None (auto cpu_count-1) must also match the serial path."""
    data = _make_synth_5m(1800, seed=3)
    space = _make_space()
    common = dict(
        n_trials=16, top_k=6, top_m=3, min_trades=5,
        warmup_bars=30, costs=PROFILES["UK100"], seed=99,
    )
    serial = run_sweep(data, space, n_jobs=1, **common)
    auto = run_sweep(data, space, n_jobs=None, **common)
    assert _fingerprint(serial) == _fingerprint(auto)


# ---- 2. Speedup: parallel beats serial by more than 2x --------------------
@pytest.mark.skipif(
    (os.cpu_count() or 1) < 4,
    reason="Need >= 4 cores to demonstrate a 2x+ speedup",
)
def test_parallel_is_more_than_twice_as_fast():
    # Sized so each trial does enough work that fixed pool/pickle overhead is
    # amortised — otherwise the speedup ratio is dominated by startup cost.
    data = _make_synth_5m(6000, seed=11)
    space = _make_space()
    common = dict(
        n_trials=50, top_k=10, top_m=5, min_trades=5,
        warmup_bars=30, costs=PROFILES["UK100"], seed=42,
    )

    t0 = time.perf_counter()
    serial = run_sweep(data, space, n_jobs=1, **common)
    serial_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel = run_sweep(data, space, n_jobs=4, **common)
    parallel_time = time.perf_counter() - t0

    # Results must still match (parallelism didn't change the answer)...
    assert _fingerprint(serial) == _fingerprint(parallel)
    # ...and parallel must be more than 2x faster.
    assert parallel_time < 0.5 * serial_time, (
        f"Expected parallel < half serial: serial={serial_time:.2f}s, "
        f"parallel={parallel_time:.2f}s "
        f"(ratio {parallel_time / serial_time:.2f})"
    )
