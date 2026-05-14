"""
Parameter sweep with built-in IS/OOS discipline.

The reason this lives in its own module rather than being a flag on
run_backtest: sweeping is structurally different. You're testing many models
at once, which exposes you to the multiple-testing problem (run enough
combinations and one will look great by chance). The defence is procedural:

  1. `grid_sweep()` runs every combination on **in-sample data only**. Out-
     of-sample is reserved and never touched during the sweep.
  2. You inspect the sweep results, pick **one** parameter set you like.
  3. You call `evaluate_oos()` ONCE on that single chosen set.
  4. The OOS number is the honest verdict. You report it as-is even if bad.
     Re-picking after seeing OOS contaminates the test.

This module enforces the architecture; the discipline at step 4 is on you.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import pandas as pd

from backtest.engine import Strategy, run_backtest, BacktestResult
from backtest.metrics import Metrics, compute_metrics
from backtest.validation import (
    holdout_split, walk_forward, WalkForwardResult,
)


@dataclass
class SweepRun:
    """Result for one parameter combination, evaluated on IS data only."""
    params: dict
    is_metrics: Metrics
    wf_consistency: float        # fraction of walk-forward folds with PF > 1
    wf_results: WalkForwardResult


@dataclass
class SweepResult:
    runs: list[SweepRun]
    in_sample_period: tuple[pd.Timestamp, pd.Timestamp]
    oos_period: tuple[pd.Timestamp, pd.Timestamp]
    n_combos_attempted: int
    n_failures: int = 0

    def summary_df(self) -> pd.DataFrame:
        """Flatten runs into a DataFrame for inspection / sorting / saving."""
        rows = []
        for r in self.runs:
            row = dict(r.params)  # one column per param
            row.update({
                "trades": r.is_metrics.num_trades,
                "sharpe": round(r.is_metrics.sharpe, 3),
                "profit_factor": round(r.is_metrics.profit_factor, 3),
                "return_%": round(r.is_metrics.total_return_pct, 2),
                "max_dd_%": round(r.is_metrics.max_drawdown_pct, 2),
                "win_rate_%": round(r.is_metrics.win_rate_pct, 1),
                "wf_consistency": round(r.wf_consistency, 2),
            })
            rows.append(row)
        return pd.DataFrame(rows)

    def filter_min_trades(self, min_trades: int = 30) -> "SweepResult":
        """Keep only runs with at least `min_trades` trades — others are noise."""
        return SweepResult(
            runs=[r for r in self.runs if r.is_metrics.num_trades >= min_trades],
            in_sample_period=self.in_sample_period,
            oos_period=self.oos_period,
            n_combos_attempted=self.n_combos_attempted,
            n_failures=self.n_failures,
        )

    def top_by(self, metric: str = "sharpe", n: int = 10) -> list[SweepRun]:
        """Return the top-N runs by a metric on IS data."""
        # Map metric name to attribute
        getter = lambda r: getattr(r.is_metrics, metric, None) or 0.0
        return sorted(self.runs, key=getter, reverse=True)[:n]


def grid_sweep(
    data: pd.DataFrame,
    factory_for_params: Callable[[dict], Callable[[], Strategy]],
    param_grid: dict[str, list],
    oos_fraction: float = 0.2,
    n_folds: int = 4,
    warmup_bars: int = 50,
    bars_per_year: int = 252,
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> SweepResult:
    """
    Grid-search across `param_grid` (cartesian product) on IS data only.

    `factory_for_params` takes a params dict and returns a no-arg strategy
    factory (a callable that creates a fresh strategy instance). This
    indirection lets walk-forward fold create independent strategies for
    each combination.

    OOS data is split off and untouched. To evaluate a single chosen set
    on OOS, use `evaluate_oos()`.
    """
    in_sample, oos = holdout_split(data, oos_fraction)

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    combos = list(product(*value_lists))

    runs: list[SweepRun] = []
    failures = 0
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        try:
            factory = factory_for_params(params)
            result = run_backtest(in_sample, factory(), warmup_bars=warmup_bars)
            metrics = compute_metrics(result, bars_per_year=bars_per_year)
            wf = walk_forward(
                in_sample, factory, n_folds=n_folds,
                warmup_bars=warmup_bars, bars_per_year=bars_per_year,
            )
            wf_wins = sum(1 for m in wf.fold_metrics if m.profit_factor > 1.0)
            wf_consistency = wf_wins / len(wf.fold_metrics) if wf.fold_metrics else 0.0
            runs.append(SweepRun(
                params=params, is_metrics=metrics,
                wf_consistency=wf_consistency, wf_results=wf,
            ))
        except Exception as e:
            failures += 1
            if progress_callback:
                progress_callback(idx + 1, len(combos), {**params, "error": str(e)})
            continue
        if progress_callback:
            progress_callback(idx + 1, len(combos), params)

    return SweepResult(
        runs=runs,
        in_sample_period=(in_sample.index[0], in_sample.index[-1]),
        oos_period=(oos.index[0], oos.index[-1]),
        n_combos_attempted=len(combos),
        n_failures=failures,
    )


def evaluate_oos(
    data: pd.DataFrame,
    factory: Callable[[], Strategy],
    oos_fraction: float = 0.2,
    warmup_bars: int = 50,
    bars_per_year: int = 252,
) -> tuple[BacktestResult, Metrics]:
    """
    Evaluate ONE chosen strategy/factory on the held-out OOS slice.

    DISCIPLINE NOTE: only call this AFTER:
      - You've run grid_sweep on IS only.
      - You've inspected results and picked ONE param set you'd commit to.

    The metric you get back is your honest verdict. Don't re-pick after
    seeing it; that turns OOS into IS and undoes the entire point.
    """
    _, oos = holdout_split(data, oos_fraction)
    result = run_backtest(oos, factory(), warmup_bars=warmup_bars)
    metrics = compute_metrics(result, bars_per_year=bars_per_year)
    return result, metrics
