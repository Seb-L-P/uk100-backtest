"""
Multi-seed sweep aggregation.

A single random-search sweep only samples a tiny corner of the design
space. The combination it finds at the top might just be where seed 42
happened to land — not where the actual edge is.

Running the SAME SearchSpace with multiple seeds and aggregating the
top-OOS rows tells you which structures REPEATEDLY surface. That's the
real robustness signal:

  - "Inside-bar @ 15m" in top-5 across seeds 42, 99, 1337, 7 → strong
    signal there's an edge there worth investigating.
  - "Engulfing @ 5m, supporter MACD @ 1h, weight 1.62" once across 4 seeds
    → almost certainly a parameter-specific fluke.

We aggregate by TRIGGER STRATEGY + TRIGGER TF (the structural signature).
Parameter values within those structures still vary — that's where
refinement comes in (phase 4d).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import math
from typing import Callable

import numpy as np
import pandas as pd

from backtest.sweep_runner import run_sweep, SweepResult, TrialResult
from backtest.sweep_space import SearchSpace


@dataclass
class SeedRun:
    """One sweep with one seed — the underlying SweepResult plus its seed."""
    seed: int
    result: SweepResult


@dataclass
class StructureSummary:
    """
    Aggregated stats for one (trigger_strategy, trigger_tf) structure
    across multiple seed runs.
    """
    trigger_strategy: str
    trigger_tf: str
    n_seeds_appeared_in: int       # how many distinct seeds had this in top-OOS
    total_appearances: int          # raw count across all seeds (≥ above)
    best_oos_sharpe: float
    median_oos_sharpe: float
    worst_oos_sharpe: float
    seed_appearances: list[int]    # which seeds did this appear under
    # Pointers to the best representative trial (one with best OOS) — UI uses
    # this to render a card for the structure.
    best_trial: TrialResult | None = None
    best_trial_seed: int | None = None


@dataclass
class MultiSeedResult:
    seed_runs: list[SeedRun]
    structures: list[StructureSummary]   # sorted by (n_seeds_appeared, best_sharpe) desc
    n_seeds: int

    def to_dataframe(self) -> pd.DataFrame:
        """Leaderboard of structures — one row per (trigger, trigger TF)."""
        rows = []
        for s in self.structures:
            rows.append({
                "trigger": s.trigger_strategy,
                "trigger TF": s.trigger_tf,
                "seeds": s.n_seeds_appeared_in,
                "appearances": s.total_appearances,
                "best OOS": s.best_oos_sharpe,
                "median OOS": s.median_oos_sharpe,
                "worst OOS": s.worst_oos_sharpe,
                "seed list": ",".join(str(x) for x in s.seed_appearances),
            })
        return pd.DataFrame(rows)


def _is_qualified(t: TrialResult) -> bool:
    s = t.metrics.sharpe
    return (t.metrics.disqualified_reason is None
            and not math.isinf(s) and not math.isnan(s))


def run_multi_seed(
    data: pd.DataFrame,
    space: SearchSpace,
    seeds: list[int],
    *,
    n_trials: int = 200,
    is_ratio: float = 0.6,
    val_ratio: float = 0.2,
    top_k: int = 20,
    top_m: int = 10,
    min_trades: int = 20,
    warmup_bars: int = 50,
    costs=None,
    progress_callback: Callable[[float, str], None] | None = None,
    n_jobs: int | None = None,
) -> MultiSeedResult:
    """
    Run `run_sweep` once per seed in `seeds`, then aggregate results by
    structural signature (trigger strategy + trigger TF).

    Progress callback receives normalised progress across ALL seeds —
    so a 4-seed run feeds the bar 0%, 25%, 50%, 75%, 100% boundaries.

    `n_jobs` is forwarded to each per-seed `run_sweep` (None = auto-detect
    `os.cpu_count() - 1` workers), so multi-seed runs parallelise the
    trials within each seed. Seeds themselves run sequentially.
    """
    if not seeds:
        raise ValueError("Need at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique. Got duplicates in {seeds}")

    seed_runs: list[SeedRun] = []
    n_seeds = len(seeds)

    for seed_index, seed in enumerate(seeds):
        # Per-seed progress weight: 1/N of total budget.
        if progress_callback:
            def _inner_cb(frac, msg, _i=seed_index):
                overall = (_i + frac) / n_seeds
                progress_callback(overall, f"Seed {seed} ({_i + 1}/{n_seeds}): {msg}")
        else:
            _inner_cb = None

        result = run_sweep(
            data, space,
            n_trials=n_trials, is_ratio=is_ratio, val_ratio=val_ratio,
            top_k=top_k, top_m=top_m, min_trades=min_trades,
            warmup_bars=warmup_bars, costs=costs, seed=seed,
            progress_callback=_inner_cb, n_jobs=n_jobs,
        )
        seed_runs.append(SeedRun(seed=seed, result=result))

    # Aggregate: group qualified OOS trials by (trigger_strategy, trigger_tf)
    bucket: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"trials": [], "sharpes": [], "seeds": set()}
    )
    for sr in seed_runs:
        for t in sr.result.oos_top:
            if not _is_qualified(t):
                continue
            key = (t.graph.trigger.strategy_key, t.graph.trigger.timeframe)
            bucket[key]["trials"].append((sr.seed, t))
            bucket[key]["sharpes"].append(t.metrics.sharpe)
            bucket[key]["seeds"].add(sr.seed)

    structures: list[StructureSummary] = []
    for (trig, tf), info in bucket.items():
        sharpes = info["sharpes"]
        seeds_set = sorted(info["seeds"])
        # Best trial = highest OOS sharpe, recorded with the seed that produced it
        best_trial_seed, best_trial = max(
            info["trials"], key=lambda t: t[1].metrics.sharpe
        )
        structures.append(StructureSummary(
            trigger_strategy=trig,
            trigger_tf=tf,
            n_seeds_appeared_in=len(seeds_set),
            total_appearances=len(info["trials"]),
            best_oos_sharpe=max(sharpes),
            median_oos_sharpe=float(np.median(sharpes)),
            worst_oos_sharpe=min(sharpes),
            seed_appearances=seeds_set,
            best_trial=best_trial,
            best_trial_seed=best_trial_seed,
        ))

    # Sort: most seed appearances first, then highest best OOS Sharpe
    structures.sort(
        key=lambda s: (-s.n_seeds_appeared_in, -s.best_oos_sharpe)
    )

    return MultiSeedResult(
        seed_runs=seed_runs,
        structures=structures,
        n_seeds=n_seeds,
    )
