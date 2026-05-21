"""
Quick wall-clock benchmark of the discovery sweep: serial vs parallel.

Runs the same N-trial sweep (same seed) single-threaded (`n_jobs=1`) and
parallel (`n_jobs=None` → cpu_count-1), reports wall-clock for each, and
asserts the two leaderboards are bit-identical.

    python scripts/benchmark_sweep.py            # 200 trials, 5m synthetic
    python scripts/benchmark_sweep.py 100        # custom trial count
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtest.sweep_runner import run_sweep
from backtest.sweep_space import SearchSpace
from config import PROFILES


def make_synth_5m(n_bars: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic UK100-scale 5m OHLCV: sinusoid + random walk."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 08:00", periods=n_bars, freq="5min")
    t = np.arange(n_bars)
    close = 8000.0 + 60.0 * np.sin(t / 40.0) + np.cumsum(rng.normal(0, 4.0, n_bars))
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


def fingerprint(result):
    return [
        (t.description, t.metrics.sharpe, t.metrics.n_trades)
        for t in result.oos_top
    ]


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    data = make_synth_5m(6000, seed=11)
    triggers = ["sma", "rsi_revert", "bb_revert", "macd_cross", "donchian"]
    space = SearchSpace(
        triggers_pool=list(triggers),
        supporters_pool=list(triggers),
        vetoes_pool=list(triggers),
        max_supporters=2,
        max_vetoes=1,
        data_tf="5m",
        trigger_tf_options=["5m", "15m", "30m"],
    )
    common = dict(
        n_trials=n_trials, top_k=20, top_m=10, min_trades=5,
        warmup_bars=30, costs=PROFILES["UK100"], seed=42,
    )

    n_auto = max(1, (os.cpu_count() or 2) - 1)
    print(f"Benchmark: {n_trials} trials, {len(data)} bars @ 5m, "
          f"cpu_count={os.cpu_count()} (auto n_jobs={n_auto})\n")

    t0 = time.perf_counter()
    serial = run_sweep(data, space, n_jobs=1, **common)
    serial_t = time.perf_counter() - t0
    print(f"  serial   (n_jobs=1)    : {serial_t:6.2f}s")

    t0 = time.perf_counter()
    parallel = run_sweep(data, space, n_jobs=None, **common)
    parallel_t = time.perf_counter() - t0
    print(f"  parallel (n_jobs={n_auto})    : {parallel_t:6.2f}s")

    print(f"\n  speedup: {serial_t / parallel_t:.2f}x")
    same = fingerprint(serial) == fingerprint(parallel)
    print(f"  leaderboards identical: {same}")
    if not same:
        sys.exit("DETERMINISM FAILURE: leaderboards differ")


if __name__ == "__main__":
    main()
