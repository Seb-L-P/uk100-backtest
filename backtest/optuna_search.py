"""
Bayesian parameter optimisation using Optuna (TPE sampler).

Why this in addition to the grid sweep:
  - Grid scales poorly: 5 params × 5 values each = 3125 backtests.
  - Optuna's Tree-Structured Parzen Estimator (TPE) probes the space adaptively,
    focusing trials on promising regions. Typically finds near-optimal configs
    in 50-200 trials where grid would need thousands.
  - Same downstream defences apply: walk-forward consistency, IS/OOS split,
    Deflated Sharpe correction. The optimiser is just smarter than grid search.

Objective design:
  We optimise a COMPOSITE score, not raw Sharpe, to avoid Optuna gaming
  thin-trade flukes:

    score = walk_forward_consistency × min(1.0, num_trades / target_trades)

  - Walk-forward consistency = fraction of folds with PF > 1.0. Rewards
    strategies that work across regimes.
  - Trade-count penalty: scales linearly until target_trades, then 1.0.
    Without this Optuna finds (e.g.) tight thresholds producing 3 lucky
    winning trades with WF consistency = 100% — meaningless overfitting.

Returns an OptunaResult with: best_params, all_trials, study object (for
visualisation), and the full sweep DataFrame for compatibility with the
existing pick-best-then-evaluate-OOS workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import pandas as pd

from backtest.engine import Strategy, run_backtest
from backtest.metrics import Metrics, compute_metrics
from backtest.validation import holdout_split, walk_forward, WalkForwardResult
from backtest.sweep import SweepRun


@dataclass
class OptunaTrial:
    """One Optuna trial — params + IS metrics + walk-forward consistency."""
    params: dict
    is_metrics: Metrics
    wf_consistency: float
    wf_results: WalkForwardResult
    score: float


@dataclass
class OptunaResult:
    trials: list[OptunaTrial]
    best_params: dict
    best_score: float
    in_sample_period: tuple
    oos_period: tuple
    study: Any   # the raw optuna.Study, for visualisation
    optimization_metric: str
    target_trades: int

    def summary_df(self) -> pd.DataFrame:
        """Flat DataFrame compatible with the grid sweep result format."""
        rows = []
        for t in self.trials:
            row = dict(t.params)
            row.update({
                "trades": t.is_metrics.num_trades,
                "sharpe": round(t.is_metrics.sharpe, 3),
                "profit_factor": round(t.is_metrics.profit_factor, 3),
                "return_%": round(t.is_metrics.total_return_pct, 2),
                "max_dd_%": round(t.is_metrics.max_drawdown_pct, 2),
                "win_rate_%": round(t.is_metrics.win_rate_pct, 1),
                "wf_consistency": round(t.wf_consistency, 2),
                "score": round(t.score, 3),
            })
            rows.append(row)
        return pd.DataFrame(rows)


def _suggest_params(trial, spec) -> dict:
    """Use Optuna's `trial.suggest_*` API to sample params from the spec ranges."""
    params = {}
    for p in (spec.params or []):
        if p.type == "int":
            step = int(p.step) if p.step else 1
            params[p.name] = trial.suggest_int(p.name, int(p.min), int(p.max), step=step)
        elif p.type == "float":
            step = float(p.step) if p.step else 0.05
            params[p.name] = trial.suggest_float(p.name, float(p.min), float(p.max), step=step)
        elif p.type == "bool":
            params[p.name] = trial.suggest_categorical(p.name, [True, False])
    return params


def run_optuna_study(
    data: pd.DataFrame,
    spec,
    factory_for_params: Callable[[dict], Callable[[], Strategy]],
    n_trials: int = 100,
    timeout: float | None = None,
    oos_fraction: float = 0.2,
    n_folds: int = 4,
    warmup_bars: int = 50,
    bars_per_year: int = 252,
    target_trades: int = 50,
    optimization_metric: str = "wf_consistency",  # "wf_consistency" | "sharpe"
    seed: int = 42,
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> OptunaResult:
    """
    Run Optuna's TPE sampler for `n_trials` trials. Returns an OptunaResult.

    Holds the OOS slice untouched throughout — same discipline as grid_sweep.
    Each trial runs ONE backtest on in-sample + one walk-forward per fold.
    """
    try:
        import optuna
    except ImportError:
        raise RuntimeError("optuna not installed. Run: pip install -r requirements.txt")

    # Quiet Optuna's per-trial logging — we have our own progress callback
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    in_sample, oos = holdout_split(data, oos_fraction=oos_fraction)
    trial_records: list[OptunaTrial] = []

    def objective(trial):
        params = _suggest_params(trial, spec)
        try:
            factory = factory_for_params(params)
            is_result = run_backtest(in_sample, factory(), warmup_bars=warmup_bars)
            is_metrics = compute_metrics(is_result, bars_per_year=bars_per_year)

            # Skip if no trades at all — Optuna gets NaN/0, treats as bad
            if is_metrics.num_trades == 0:
                return 0.0

            wf = walk_forward(
                in_sample, factory, n_folds=n_folds,
                warmup_bars=warmup_bars, bars_per_year=bars_per_year,
            )
            wf_wins = sum(1 for m in wf.fold_metrics if m.profit_factor > 1.0)
            wf_consistency = wf_wins / len(wf.fold_metrics) if wf.fold_metrics else 0.0

            # Composite score with trade-count penalty
            trade_penalty = min(1.0, is_metrics.num_trades / target_trades)
            if optimization_metric == "wf_consistency":
                base = wf_consistency
            elif optimization_metric == "sharpe":
                # Normalise Sharpe to (0,1] via sigmoid-ish; cap at 3.0
                base = max(0.0, min(1.0, is_metrics.sharpe / 3.0))
            elif optimization_metric == "profit_factor":
                base = max(0.0, min(1.0, (is_metrics.profit_factor - 1.0) / 2.0))
            else:
                base = wf_consistency
            score = base * trade_penalty

            trial_records.append(OptunaTrial(
                params=params, is_metrics=is_metrics,
                wf_consistency=wf_consistency, wf_results=wf, score=score,
            ))
            if progress_callback:
                progress_callback(len(trial_records), n_trials, params)
            return score
        except Exception as e:
            # Failed trial — return 0 so Optuna avoids the region
            if progress_callback:
                progress_callback(len(trial_records) + 1, n_trials,
                                  {**params, "error": str(e)})
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=False)

    # Best params: from the trial with highest score
    if trial_records:
        best = max(trial_records, key=lambda t: t.score)
        best_params = best.params
        best_score = best.score
    else:
        best_params, best_score = {}, 0.0

    return OptunaResult(
        trials=trial_records,
        best_params=best_params,
        best_score=best_score,
        in_sample_period=(in_sample.index[0], in_sample.index[-1]),
        oos_period=(oos.index[0], oos.index[-1]),
        study=study,
        optimization_metric=optimization_metric,
        target_trades=target_trades,
    )
