"""
Refinement pass — Optuna-driven focused optimisation around a candidate.

Random sweep casts a wide net. Refinement narrows in: given a candidate
DecisionGraph from a sweep, build a tight parameter range around each
of its current parameters (default ±25% around the value, clipped to the
strategy's ParamSpec bounds) and let Optuna's TPE explore that local
region.

What we tune in refinement:
  - Trigger strategy params (always)
  - Supporter params (always)
  - Veto params (always)
  - Supporter weights (if `tune_weights=True`)

What we DON'T tune (intentionally fixed at candidate values):
  - Trigger strategy choice — refinement is about THIS structure
  - Trigger TF, supporter TFs, veto TFs — same reason
  - Structure (which supporters/vetoes are present) — same reason
  - Graph-level knobs (min_score, risk_floor/ceiling) — kept constant to
    isolate the param effect. Sweep these separately in the main pass.

Why narrower ranges:
  Random search at sweep stage explores [param_min, param_max] uniformly.
  Refinement assumes the candidate is near a local optimum; we explore
  a tight neighbourhood. Optuna's TPE adapts FAST in narrow ranges.

Output: the best refined graph + a comparison vs the original candidate
on the same data splits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backtest.data_split import three_way_split, DataSplit
from backtest.engine import run_backtest
from backtest.graph import (
    DecisionGraph, TriggerNode, SupporterNode, VetoNode,
)
from backtest.sweep_objective import compute_metrics, TrialMetrics
from backtest.sweep_runner import GraphOrchestrator
from strategies.registry import STRATEGIES, ParamSpec


# ---- Result types ------------------------------------------------------
@dataclass
class RefinementResult:
    """Output of a refinement pass."""
    original_graph: DecisionGraph
    original_metrics: TrialMetrics       # cross-split: IS/Val/OOS sharpe etc.
    refined_graph: DecisionGraph
    refined_is_sharpe: float
    refined_val_sharpe: float
    refined_oos_sharpe: float
    refined_oos_metrics: TrialMetrics
    n_trials: int
    n_completed: int
    n_pruned: int
    split: DataSplit


# ---- Range derivation --------------------------------------------------
def _narrow_range(spec: ParamSpec, current_value, narrow_factor: float = 0.25
                  ) -> tuple:
    """
    Return a (lo, hi) tuple for refinement around `current_value`.

    Width = narrow_factor × original range. Clipped to ParamSpec [min, max].
    """
    spec_min = spec.min if spec.min is not None else 0
    spec_max = spec.max if spec.max is not None else spec_min + 1
    width = (spec_max - spec_min) * narrow_factor
    lo = max(spec_min, current_value - width / 2)
    hi = min(spec_max, current_value + width / 2)
    if hi <= lo:
        # Edge case: narrow range collapsed (current value at boundary).
        # Expand a bit.
        lo, hi = max(spec_min, current_value - width), min(spec_max, current_value + width)
        if hi <= lo:
            lo, hi = spec_min, spec_max
    return lo, hi


def _build_refinement_params(strategy_key: str, current_params: dict,
                              trial, narrow_factor: float, key_prefix: str
                              ) -> dict:
    """
    Build a dict of Optuna-sampled parameters for the given strategy,
    using its current_params as anchors.
    """
    spec = STRATEGIES[strategy_key]
    refined = {}
    for p in (spec.params or []):
        cur = current_params.get(p.name, p.default)
        full_key = f"{key_prefix}_{p.name}"
        if p.type == "int":
            lo, hi = _narrow_range(p, cur, narrow_factor)
            lo_i, hi_i = int(round(lo)), int(round(hi))
            if hi_i <= lo_i:
                hi_i = lo_i + 1
            refined[p.name] = trial.suggest_int(full_key, lo_i, hi_i)
        elif p.type == "float":
            lo, hi = _narrow_range(p, cur, narrow_factor)
            if hi <= lo:
                hi = lo + 1e-6
            step = float(p.step) if p.step else None
            if step is not None and step > 0:
                refined[p.name] = trial.suggest_float(full_key, lo, hi, step=step)
            else:
                refined[p.name] = trial.suggest_float(full_key, lo, hi)
        elif p.type == "bool":
            refined[p.name] = trial.suggest_categorical(full_key, [True, False])
        else:
            refined[p.name] = cur  # unknown type — leave alone
    return refined


# ---- Optuna objective --------------------------------------------------
def _make_objective(
    candidate: DecisionGraph,
    split: DataSplit,
    costs,
    *,
    warmup_bars: int,
    is_min_trades: int,
    narrow_factor: float,
    tune_weights: bool,
    weight_range: tuple[float, float],
):
    """Build the Optuna objective closure."""
    def objective(trial):
        # ---- Sample trigger params ----
        trig_params = _build_refinement_params(
            candidate.trigger.strategy_key,
            candidate.trigger.params,
            trial, narrow_factor, "trig",
        )
        new_trigger = TriggerNode(
            strategy_key=candidate.trigger.strategy_key,
            params=trig_params,
            timeframe=candidate.trigger.timeframe,
        )

        # ---- Supporters ----
        new_supporters = []
        for i, s in enumerate(candidate.supporters):
            s_params = _build_refinement_params(
                s.strategy_key, s.params, trial, narrow_factor, f"sup{i}",
            )
            if tune_weights:
                w = trial.suggest_float(f"sup{i}_weight",
                                        weight_range[0], weight_range[1])
            else:
                w = s.weight
            new_supporters.append(SupporterNode(
                strategy_key=s.strategy_key,
                params=s_params,
                timeframe=s.timeframe,
                weight=w,
            ))

        # ---- Vetoes ----
        new_vetoes = []
        for i, v in enumerate(candidate.vetoes):
            v_params = _build_refinement_params(
                v.strategy_key, v.params, trial, narrow_factor, f"veto{i}",
            )
            new_vetoes.append(VetoNode(
                strategy_key=v.strategy_key,
                params=v_params,
                timeframe=v.timeframe,
            ))

        graph = DecisionGraph(
            trigger=new_trigger,
            supporters=new_supporters,
            vetoes=new_vetoes,
            min_score=candidate.min_score,
            risk_floor=candidate.risk_floor,
            risk_ceiling=candidate.risk_ceiling,
            risk_curve=candidate.risk_curve,
            session_open_override=candidate.session_open_override,
            session_close_override=candidate.session_close_override,
            flat_by_override=candidate.flat_by_override,
            allow_overnight=candidate.allow_overnight,
        )
        try:
            orch = GraphOrchestrator(graph)
            result = run_backtest(split.is_df, orch,
                                  warmup_bars=warmup_bars, costs=costs)
            m = compute_metrics(
                trades_df=result.trades_df,
                equity_curve=result.equity_curve,
                starting_balance=result.starting_balance,
                final_balance=result.final_balance,
                min_trades=is_min_trades,
            )
            # Store the graph on the trial so we can rebuild the best one
            trial.set_user_attr("graph_obj", graph)
            if m.sharpe == float("-inf"):
                return -1e9   # finite sentinel for Optuna
            return m.sharpe
        except Exception:
            return -1e9

    return objective


# ---- Public API --------------------------------------------------------
def refine_candidate(
    data: pd.DataFrame,
    candidate: DecisionGraph,
    *,
    n_trials: int = 50,
    is_ratio: float = 0.6,
    val_ratio: float = 0.2,
    narrow_factor: float = 0.25,
    tune_weights: bool = True,
    weight_range: tuple[float, float] = (0.3, 2.0),
    min_trades: int = 20,
    warmup_bars: int = 50,
    costs=None,
    seed: int = 42,
    progress_callback: Callable[[float, str], None] | None = None,
) -> RefinementResult:
    """
    Run an Optuna refinement pass on `candidate`.

    Splits `data` 3 ways (same convention as the main sweep). Optuna
    tunes parameters on IS, then we evaluate the best trial on Val + OOS
    so the user can see whether refinement improved (or just overfit) IS.
    """
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError(
            "Optuna not installed. Run: pip install optuna"
        ) from e

    split = three_way_split(data, is_ratio, val_ratio)

    # Same adaptive floor logic as run_sweep
    from backtest.sweep_runner import _effective_min_trades
    is_floor = _effective_min_trades(min_trades, len(split.is_df))
    val_floor = _effective_min_trades(min_trades, len(split.val_df))
    oos_floor = _effective_min_trades(min_trades, len(split.oos_df))

    # ---- Original-candidate metrics for comparison ----
    orig_oos_orch = GraphOrchestrator(candidate)
    orig_result = run_backtest(split.oos_df, orig_oos_orch,
                               warmup_bars=warmup_bars, costs=costs)
    orig_metrics = compute_metrics(
        trades_df=orig_result.trades_df,
        equity_curve=orig_result.equity_curve,
        starting_balance=orig_result.starting_balance,
        final_balance=orig_result.final_balance,
        min_trades=oos_floor,
    )

    # ---- Optuna study ----
    objective = _make_objective(
        candidate, split, costs,
        warmup_bars=warmup_bars,
        is_min_trades=is_floor,
        narrow_factor=narrow_factor,
        tune_weights=tune_weights,
        weight_range=weight_range,
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if progress_callback:
        progress_callback(0.0, "Starting refinement...")

    # Wrap objective to push progress
    completed = {"n": 0}
    def _wrapped(trial):
        v = objective(trial)
        completed["n"] += 1
        if progress_callback:
            progress_callback(
                completed["n"] / n_trials,
                f"Refinement trial {completed['n']}/{n_trials}",
            )
        return v

    study.optimize(_wrapped, n_trials=n_trials, show_progress_bar=False)

    # Get best graph (stored as user attr)
    best_trial = study.best_trial
    refined_graph = best_trial.user_attrs.get("graph_obj", candidate)

    # ---- Re-evaluate on Val + OOS ----
    def _eval_on(df, floor):
        orch = GraphOrchestrator(refined_graph)
        r = run_backtest(df, orch, warmup_bars=warmup_bars, costs=costs)
        return compute_metrics(
            trades_df=r.trades_df,
            equity_curve=r.equity_curve,
            starting_balance=r.starting_balance,
            final_balance=r.final_balance,
            min_trades=floor,
        )

    refined_is_metrics = _eval_on(split.is_df, is_floor)
    refined_val_metrics = _eval_on(split.val_df, val_floor)
    refined_oos_metrics = _eval_on(split.oos_df, oos_floor)

    return RefinementResult(
        original_graph=candidate,
        original_metrics=orig_metrics,
        refined_graph=refined_graph,
        refined_is_sharpe=refined_is_metrics.sharpe,
        refined_val_sharpe=refined_val_metrics.sharpe,
        refined_oos_sharpe=refined_oos_metrics.sharpe,
        refined_oos_metrics=refined_oos_metrics,
        n_trials=len(study.trials),
        n_completed=sum(1 for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE),
        n_pruned=sum(1 for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED),
        split=split,
    )
