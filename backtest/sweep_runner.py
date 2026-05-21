"""
3-stage random sweep over DecisionGraphs.

Stages:
  Stage 1 — IS sweep:    sample N random graphs, backtest on IS, score.
  Stage 2 — Val rerank:  take top-K by IS Sharpe, re-evaluate on Val.
  Stage 3 — OOS report:  take top-M by Val Sharpe, run on OOS (looked at LAST).

The OOS Sharpe is the number you should trust. IS and Val Sharpe inflate
because of selection bias — you literally PICKED these graphs because they
scored well on IS. Sharpe should DEGRADE from IS → Val → OOS on a healthy
sweep; if OOS Sharpe is similar to IS Sharpe, either you got lucky or your
search space is small enough that overfitting wasn't possible. If OOS
Sharpe is much LOWER than IS Sharpe, that's evidence of overfitting; the
"best" IS graph is likely a coincidence and you shouldn't trust it for live.

Each trial is independent — the runner crashes gracefully on a single
trial failure (assertion error, ZeroDivision, etc.) and continues with the
next. Failed trials are recorded with sharpe=-inf and the exception
message in `disqualified_reason`, so the leaderboard still tells you what
went wrong.

Reproducibility: pass `seed` to get bit-identical results across runs.

Parallelism
-----------
Each stage is embarrassingly parallel: a trial is a pure function of
(graph, split, costs). The runner can fan trials out across a
`ProcessPoolExecutor` (`n_jobs` > 1) without affecting results.

The one thing that MUST stay serial is graph *sampling* — `sample_random_graph`
consumes the shared RNG in a strict order, so identical seeds only give
identical graphs if drawn one after another on a single thread. We
therefore split the two concerns: all N graphs are sampled serially up
front on the main process, then the (pure) backtests are dispatched to
the pool. Results are reassembled in trial order before sorting, so the
final leaderboard is bit-identical to the serial path for the same seed —
regardless of `n_jobs` or the order workers happen to finish in.

The read-only `data` splits and `costs` are shipped to each worker ONCE
via the pool `initializer` (stored in a per-process global), not re-pickled
per task. Per-trial crashes are caught inside the worker exactly as in the
serial path (`_run_one_trial` never raises), so one bad graph can't take
down a worker or the sweep.
"""
from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest.data_split import three_way_split, DataSplit
from backtest.engine import run_backtest
from backtest.graph import DecisionGraph, GraphOrchestrator
from backtest.sweep_objective import compute_metrics, TrialMetrics
from backtest.sweep_space import SearchSpace, sample_random_graph, describe_graph


_log = logging.getLogger(__name__)


# ---- Result types -----------------------------------------------------
@dataclass
class TrialResult:
    """Outcome of one (graph, split) backtest, including all reported metrics."""
    trial_index: int
    graph: DecisionGraph
    metrics: TrialMetrics
    error: Optional[str] = None  # set if backtest crashed

    @property
    def sharpe(self) -> float:
        return self.metrics.sharpe

    @property
    def description(self) -> str:
        return describe_graph(self.graph)


@dataclass
class SweepResult:
    """Full sweep output. The leaderboards are pre-sorted by their stage's Sharpe."""
    is_trials: list[TrialResult]              # all N stage-1 trials, sorted by IS Sharpe desc
    val_top: list[TrialResult]                # top-K, each re-eval'd on Val, sorted by Val Sharpe desc
    oos_top: list[TrialResult]                # top-M, each re-eval'd on OOS, sorted by OOS Sharpe desc
    split: DataSplit
    seed: int
    n_trials: int
    top_k: int
    top_m: int
    min_trades: int                            # the user-set floor
    is_min_trades: int = 0                     # adaptive per-split values
    val_min_trades: int = 0
    oos_min_trades: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """
        Build a leaderboard joining IS / Val / OOS Sharpe for the final
        top-M trials. The user reads this top-down — the winner is the
        TOP-RANKED OOS Sharpe row.

        Includes ALL three Sharpe values so you can spot degradation:
        a healthy strategy keeps decent Sharpe across all three splits;
        an overfit one drops sharply IS → OOS.
        """
        # Index IS results by graph id for fast joining
        is_by_id = {id(t.graph): t for t in self.is_trials}
        val_by_id = {id(t.graph): t for t in self.val_top}
        rows = []
        for t in self.oos_top:
            is_t = is_by_id.get(id(t.graph))
            val_t = val_by_id.get(id(t.graph))
            rows.append({
                "description": t.description,
                "OOS Sharpe": t.metrics.sharpe,
                "Val Sharpe": val_t.metrics.sharpe if val_t else float("nan"),
                "IS Sharpe": is_t.metrics.sharpe if is_t else float("nan"),
                "OOS trades": t.metrics.n_trades,
                "OOS return %": t.metrics.return_pct,
                "OOS max DD %": t.metrics.max_drawdown_pct,
                "OOS hit rate": t.metrics.hit_rate,
                "OOS profit factor": t.metrics.profit_factor,
                "disqualified": t.metrics.disqualified_reason or "",
            })
        return pd.DataFrame(rows)


# ---- One-trial wrapper --------------------------------------------------
def _effective_min_trades(user_floor: int, n_bars: int) -> int:
    """
    Adapt the user's min-trades floor to the actual split size.

    Rule: aim for roughly one trade per 80 bars. On a 758-bar IS that's ~9
    trades expected; the user's 20 caps it higher only if there's room.

    Returns the LOWER of:
      - the user's floor
      - max(3, n_bars // 80)

    Worked example: user_floor=20, IS=758 → effective = min(20, max(3, 9)) = 9.
    Same user_floor on OOS=254 → effective = min(20, max(3, 3)) = 3.

    This stops the 3-way split from over-disqualifying because the smaller
    Val/OOS windows can't physically support as many trades as IS — without
    this fix, perfectly fine strategies got Sharpe=-inf on smaller splits
    purely because the absolute floor was unattainable there.
    """
    if n_bars <= 0:
        return user_floor
    scaled = max(3, n_bars // 80)
    return min(user_floor, scaled)


def _run_one_trial(
    graph: DecisionGraph,
    data: pd.DataFrame,
    costs,
    starting_balance: float,
    min_trades: int,
    warmup_bars: int,
) -> TrialMetrics:
    """Run a single backtest, returning metrics. Never raises."""
    try:
        orchestrator = GraphOrchestrator(graph)
        result = run_backtest(
            data, orchestrator,
            warmup_bars=warmup_bars,
            costs=costs,
        )
        return compute_metrics(
            trades_df=result.trades_df,
            equity_curve=result.equity_curve,
            starting_balance=result.starting_balance,
            final_balance=result.final_balance,
            min_trades=min_trades,
        )
    except Exception as e:
        # Crashes are unfortunately common when randomly composing strategies
        # (param combos that produce zero ATR, etc.). Don't let one failure
        # halt the sweep — record it and continue.
        _log.warning("Trial crashed: %s\n%s", e, traceback.format_exc())
        return TrialMetrics(
            sharpe=float("-inf"),
            n_trades=0,
            final_balance=starting_balance,
            starting_balance=starting_balance,
            max_drawdown_pct=0.0,
            hit_rate=0.0,
            profit_factor=0.0,
            disqualified_reason=f"crashed: {type(e).__name__}: {e}",
        )


# ---- Parallel execution plumbing ---------------------------------------
# A pool worker keeps the read-only splits + costs in this per-process
# global, populated once by `_pool_init` when the worker starts. We never
# mutate it, so there's no cross-task contamination.
_WORKER_STATE: dict = {}


def _pool_init(splits, costs, starting_balance, warmup_bars) -> None:
    """ProcessPoolExecutor initializer: stash the read-only run context.

    Called once per worker process at pool startup. The (potentially large)
    split DataFrames and the cost model are pickled to each worker exactly
    once here, instead of being re-shipped with every task.
    """
    _WORKER_STATE["splits"] = splits
    _WORKER_STATE["costs"] = costs
    _WORKER_STATE["starting_balance"] = starting_balance
    _WORKER_STATE["warmup_bars"] = warmup_bars


def _pool_worker(key, graph: DecisionGraph, split_name: str, min_trades: int):
    """Run one trial inside a pool worker, reading context from the global.

    Returns `(key, TrialMetrics)`. `_run_one_trial` swallows trial-level
    exceptions, so this never raises for a bad graph — the metrics carry the
    `disqualified_reason` instead.
    """
    s = _WORKER_STATE
    metrics = _run_one_trial(
        graph, s["splits"][split_name], s["costs"],
        starting_balance=s["starting_balance"],
        min_trades=min_trades,
        warmup_bars=s["warmup_bars"],
    )
    return key, metrics


def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """Resolve the worker count. None → cpu_count()-1 (leave a core for the OS)."""
    if n_jobs is None:
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(n_jobs))


def _execute_stage(
    tasks: list[tuple],
    splits: dict,
    costs,
    starting_balance: float,
    warmup_bars: int,
    executor: Optional[ProcessPoolExecutor],
    on_done: Callable[[int], None],
) -> dict:
    """Run a stage's trials, serially or on the pool, returning {key: metrics}.

    `tasks` is a list of `(key, graph, split_name, min_trades)`. `on_done(n)`
    fires once per COMPLETED trial with the running count within this stage —
    when parallel, that's in completion order (via `as_completed`), not
    submission order, so the progress bar advances as work actually finishes.
    Results are keyed so the caller can reassemble in deterministic order.
    """
    results: dict = {}
    done = 0
    if executor is None:
        for key, graph, split_name, min_trades in tasks:
            results[key] = _run_one_trial(
                graph, splits[split_name], costs,
                starting_balance=starting_balance,
                min_trades=min_trades,
                warmup_bars=warmup_bars,
            )
            done += 1
            on_done(done)
        return results

    fut_to_key = {
        executor.submit(_pool_worker, key, graph, split_name, min_trades): key
        for key, graph, split_name, min_trades in tasks
    }
    for fut in as_completed(fut_to_key):
        key, metrics = fut.result()
        results[key] = metrics
        done += 1
        on_done(done)
    return results


# ---- The runner ---------------------------------------------------------
def run_sweep(
    data: pd.DataFrame,
    space: SearchSpace,
    n_trials: int = 200,
    is_ratio: float = 0.6,
    val_ratio: float = 0.2,
    top_k: int = 20,
    top_m: int = 10,
    min_trades: int = 20,
    warmup_bars: int = 50,
    costs=None,
    seed: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    n_jobs: Optional[int] = None,
) -> SweepResult:
    """
    Run the full 3-stage sweep.

    `progress_callback(fraction, message)`: called periodically with
    progress in [0, 1] and a short status message. Use this to drive a
    Streamlit progress bar. When running in parallel the callback fires as
    trials COMPLETE (not as they're submitted), so the bar reflects real
    work done.

    `n_jobs` controls trial parallelism:
      - None (default): auto-detect → `os.cpu_count() - 1` workers, leaving
        one core for the OS.
      - 1: run serially in-process (the reference path; no pool overhead).
      - N > 1: fan trials out across a `ProcessPoolExecutor` of N workers.
    Results are bit-identical across `n_jobs` values for the same seed.

    Stage 1 takes ~half the time (N trials × IS), stage 2 a small slice
    (top_k × Val), stage 3 a smaller slice (top_m × OOS). The callback
    receives progress on a normalised total budget so the bar grows
    monotonically.
    """
    if top_k > n_trials:
        top_k = n_trials
    if top_m > top_k:
        top_m = top_k

    n_jobs = _resolve_n_jobs(n_jobs)

    split = three_way_split(data, is_ratio, val_ratio)
    rng = np.random.default_rng(seed)

    # Adaptive min-trades per split — Val and OOS are smaller than IS so the
    # absolute floor would silently disqualify perfectly fine strategies there.
    is_min_trades = _effective_min_trades(min_trades, len(split.is_df))
    val_min_trades = _effective_min_trades(min_trades, len(split.val_df))
    oos_min_trades = _effective_min_trades(min_trades, len(split.oos_df))
    _log.info(
        "Effective min_trades per split: IS=%d, Val=%d, OOS=%d (user floor=%d)",
        is_min_trades, val_min_trades, oos_min_trades, min_trades,
    )

    # Total notional work = n_trials (IS) + top_k (Val) + top_m (OOS).
    # We allocate progress weights so the bar moves linearly with work done.
    total_work = n_trials + top_k + top_m
    starting_balance = 10_000.0  # display only; broker uses its own

    # Per-worker context (read-only). Keyed by split name so a single pool
    # can serve all three stages without re-shipping the data.
    splits = {"is": split.is_df, "val": split.val_df, "oos": split.oos_df}

    def _stage_reporter(stage_name: str, stage_total: int, base_done: int):
        """Build an `on_done(n)` callback that maps stage progress onto the
        global [0, 1] budget. `base_done` is the work already finished in
        earlier stages so the bar never goes backwards."""
        def _on_done(done_in_stage: int):
            if progress_callback:
                frac = (base_done + done_in_stage) / total_work
                progress_callback(frac, f"{stage_name} {done_in_stage}/{stage_total}")
        return _on_done

    # ---- Pre-generate ALL N graphs serially (determinism) ----
    # Sampling consumes the RNG in a strict order; do it on the main thread
    # so identical seeds give identical graphs regardless of n_jobs.
    graphs = [sample_random_graph(space, rng) for _ in range(n_trials)]

    executor = None
    if n_jobs > 1:
        _log.info("Sweep running on %d worker processes", n_jobs)
        executor = ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_pool_init,
            initargs=(splits, costs, starting_balance, warmup_bars),
        )
    else:
        _log.info("Sweep running serially (n_jobs=1)")

    try:
        # ---- Stage 1: IS sweep ----
        _log.info("Stage 1: IS sweep, %d trials", n_trials)
        is_tasks = [
            (i, graphs[i], "is", is_min_trades) for i in range(n_trials)
        ]
        is_metrics = _execute_stage(
            is_tasks, splits, costs, starting_balance, warmup_bars, executor,
            _stage_reporter("IS trial", n_trials, 0),
        )
        is_trials = [
            TrialResult(trial_index=i, graph=graphs[i], metrics=is_metrics[i])
            for i in range(n_trials)
        ]
        is_trials.sort(key=lambda t: t.metrics.sharpe, reverse=True)

        # ---- Stage 2: Val rerank of top-K ----
        _log.info("Stage 2: Val rerank, top %d", top_k)
        val_sources = is_trials[:top_k]
        val_tasks = [
            (j, val_sources[j].graph, "val", val_min_trades)
            for j in range(len(val_sources))
        ]
        val_metrics = _execute_stage(
            val_tasks, splits, costs, starting_balance, warmup_bars, executor,
            _stage_reporter("Val trial", top_k, n_trials),
        )
        val_top = [
            TrialResult(
                trial_index=val_sources[j].trial_index,
                graph=val_sources[j].graph,
                metrics=val_metrics[j],
            )
            for j in range(len(val_sources))
        ]
        val_top.sort(key=lambda t: t.metrics.sharpe, reverse=True)

        # ---- Stage 3: OOS report on top-M ----
        _log.info("Stage 3: OOS report, top %d", top_m)
        oos_sources = val_top[:top_m]
        oos_tasks = [
            (k, oos_sources[k].graph, "oos", oos_min_trades)
            for k in range(len(oos_sources))
        ]
        oos_metrics = _execute_stage(
            oos_tasks, splits, costs, starting_balance, warmup_bars, executor,
            _stage_reporter("OOS trial", top_m, n_trials + top_k),
        )
        oos_top = [
            TrialResult(
                trial_index=oos_sources[k].trial_index,
                graph=oos_sources[k].graph,
                metrics=oos_metrics[k],
            )
            for k in range(len(oos_sources))
        ]
        oos_top.sort(key=lambda t: t.metrics.sharpe, reverse=True)
    finally:
        if executor is not None:
            executor.shutdown()

    return SweepResult(
        is_trials=is_trials,
        val_top=val_top,
        oos_top=oos_top,
        split=split,
        seed=seed,
        n_trials=n_trials,
        top_k=top_k,
        top_m=top_m,
        min_trades=min_trades,
        is_min_trades=is_min_trades,
        val_min_trades=val_min_trades,
        oos_min_trades=oos_min_trades,
    )
