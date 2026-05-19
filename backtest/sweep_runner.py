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
"""
from __future__ import annotations

import logging
import traceback
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
) -> SweepResult:
    """
    Run the full 3-stage sweep.

    `progress_callback(fraction, message)`: called periodically with
    progress in [0, 1] and a short status message. Use this to drive a
    Streamlit progress bar.

    Stage 1 takes ~half the time (N trials × IS), stage 2 a small slice
    (top_k × Val), stage 3 a smaller slice (top_m × OOS). The callback
    receives progress on a normalised total budget so the bar grows
    monotonically.
    """
    if top_k > n_trials:
        top_k = n_trials
    if top_m > top_k:
        top_m = top_k

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

    def _report(work_done: int, message: str):
        if progress_callback:
            progress_callback(work_done / total_work, message)

    # ---- Stage 1: IS sweep ----
    _log.info("Stage 1: IS sweep, %d trials", n_trials)
    is_trials: list[TrialResult] = []
    for i in range(n_trials):
        g = sample_random_graph(space, rng)
        m = _run_one_trial(
            g, split.is_df, costs,
            starting_balance=10_000.0,  # display only; broker uses its own
            min_trades=is_min_trades,
            warmup_bars=warmup_bars,
        )
        is_trials.append(TrialResult(trial_index=i, graph=g, metrics=m))
        _report(i + 1, f"IS trial {i + 1}/{n_trials}")
    is_trials.sort(key=lambda t: t.metrics.sharpe, reverse=True)

    # ---- Stage 2: Val rerank of top-K ----
    _log.info("Stage 2: Val rerank, top %d", top_k)
    val_top: list[TrialResult] = []
    for j, src in enumerate(is_trials[:top_k]):
        m = _run_one_trial(
            src.graph, split.val_df, costs,
            starting_balance=10_000.0,
            min_trades=val_min_trades,
            warmup_bars=warmup_bars,
        )
        val_top.append(TrialResult(trial_index=src.trial_index, graph=src.graph, metrics=m))
        _report(n_trials + j + 1, f"Val trial {j + 1}/{top_k}")
    val_top.sort(key=lambda t: t.metrics.sharpe, reverse=True)

    # ---- Stage 3: OOS report on top-M ----
    _log.info("Stage 3: OOS report, top %d", top_m)
    oos_top: list[TrialResult] = []
    for k, src in enumerate(val_top[:top_m]):
        m = _run_one_trial(
            src.graph, split.oos_df, costs,
            starting_balance=10_000.0,
            min_trades=oos_min_trades,
            warmup_bars=warmup_bars,
        )
        oos_top.append(TrialResult(trial_index=src.trial_index, graph=src.graph, metrics=m))
        _report(n_trials + top_k + k + 1, f"OOS trial {k + 1}/{top_m}")
    oos_top.sort(key=lambda t: t.metrics.sharpe, reverse=True)

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
