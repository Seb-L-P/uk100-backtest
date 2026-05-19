"""
Validation tools for a SINGLE candidate DecisionGraph.

The discovery sweep produces candidates ranked by OOS Sharpe. That's the
honest single-number proxy for "is this any good?" — but it's still ONE
number from ONE window. Before paper-trading, you want to break that
single number down into several different angles:

  1. Walk-forward consistency — does the strategy work across multiple
     non-overlapping windows of the same asset, with fixed parameters?
     A genuine edge survives. A regime-specific fluke doesn't.

  2. Multi-asset cross-check — does the strategy work on similar assets
     (different equities, different indices) without re-tuning? If the
     edge is real ("close above mean reverts the next bar") it should
     transfer; if it's a coincidence on one asset's price action it won't.

  3. Monte Carlo trade-shuffle — confidence interval on the OOS Sharpe.
     Shuffles trade order; tells you how lucky the SEQUENCE of trades
     was. Wide CIs = the headline number is fragile.

This module reuses existing infrastructure in `backtest.validation`
(walk_forward, monte_carlo_trade_shuffle, bootstrap_metrics) — wrappers
here adapt them to take a DecisionGraph as the candidate instead of a
single Strategy class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest.engine import run_backtest, BacktestResult
from backtest.graph import DecisionGraph, GraphOrchestrator
from backtest.validation import (
    walk_forward as _walk_forward_base,
    monte_carlo_trade_shuffle,
    bootstrap_metrics,
    WalkForwardResult,
    MonteCarloResult,
    BootstrapResult,
)
from backtest.metrics import compute_metrics
from backtest.sweep_objective import compute_metrics as compute_sweep_metrics


# ====================================================================
#                       WALK-FORWARD ON A CANDIDATE
# ====================================================================
@dataclass
class CandidateWalkForwardResult:
    """Per-fold metrics for a candidate graph."""
    n_folds: int
    fold_boundaries: list[tuple]
    fold_sharpes: list[float]
    fold_trades: list[int]
    fold_returns: list[float]
    fold_max_dds: list[float]

    @property
    def positive_fold_fraction(self) -> float:
        """Fraction of folds with Sharpe > 0. >=0.6 = decent stability."""
        if not self.fold_sharpes:
            return 0.0
        return sum(1 for s in self.fold_sharpes if s > 0) / len(self.fold_sharpes)

    @property
    def median_sharpe(self) -> float:
        return float(np.median(self.fold_sharpes)) if self.fold_sharpes else 0.0


def walk_forward_candidate(
    data: pd.DataFrame,
    graph: DecisionGraph,
    n_folds: int = 5,
    warmup_bars: int = 50,
    costs=None,
    min_trades_per_fold: int = 5,
) -> CandidateWalkForwardResult:
    """
    Run the candidate graph on K consecutive folds.

    Each fold gets a fresh GraphOrchestrator (no state leakage). Returns
    per-fold Sharpe + trade count so the UI can render a 'consistency
    profile' bar chart.
    """
    if n_folds < 2:
        raise ValueError("Need at least 2 folds")
    n = len(data)
    fold_size = n // n_folds
    if fold_size < warmup_bars * 2:
        raise ValueError(
            f"Fold size ({fold_size} bars) too small for warmup ({warmup_bars}). "
            f"Reduce n_folds or use longer data."
        )

    sharpes, trades, returns, max_dds, boundaries = [], [], [], [], []
    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n
        fold_data = data.iloc[start:end]
        orch = GraphOrchestrator(graph)
        try:
            result = run_backtest(fold_data, orch, warmup_bars=warmup_bars,
                                  costs=costs)
            sm = compute_sweep_metrics(
                trades_df=result.trades_df,
                equity_curve=result.equity_curve,
                starting_balance=result.starting_balance,
                final_balance=result.final_balance,
                min_trades=min_trades_per_fold,
            )
            sharpe = sm.sharpe if not (sm.sharpe == float("-inf")) else float("nan")
        except Exception:
            sharpe = float("nan")
            result = None
            sm = None

        sharpes.append(sharpe)
        trades.append(sm.n_trades if sm else 0)
        returns.append(sm.return_pct if sm else 0.0)
        max_dds.append(sm.max_drawdown_pct if sm else 0.0)
        boundaries.append((fold_data.index[0], fold_data.index[-1]))

    return CandidateWalkForwardResult(
        n_folds=n_folds,
        fold_boundaries=boundaries,
        fold_sharpes=sharpes,
        fold_trades=trades,
        fold_returns=returns,
        fold_max_dds=max_dds,
    )


# ====================================================================
#                       MULTI-ASSET CROSS-CHECK
# ====================================================================
@dataclass
class AssetResult:
    """One asset's outcome under the candidate."""
    asset_label: str
    ticker: str
    sharpe: float
    n_trades: int
    return_pct: float
    max_dd_pct: float
    hit_rate: float
    error: Optional[str] = None   # set if fetch / backtest failed


@dataclass
class MultiAssetResult:
    candidate_label: str
    assets: list[AssetResult]

    @property
    def n_passed(self) -> int:
        """How many assets have a positive Sharpe."""
        return sum(1 for a in self.assets
                   if a.error is None and not np.isnan(a.sharpe) and a.sharpe > 0)

    @property
    def pass_fraction(self) -> float:
        if not self.assets:
            return 0.0
        return self.n_passed / len(self.assets)


def multi_asset_check(
    graph: DecisionGraph,
    asset_specs: list[dict],
    data_loader: Callable[[str, str], pd.DataFrame],
    *,
    warmup_bars: int = 50,
    min_trades: int = 10,
    cost_profile_lookup: Callable[[str], object] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> MultiAssetResult:
    """
    Run the candidate graph on a set of other assets to test transfer.

    asset_specs: list of dicts {"label": "...", "ticker": "...", "data_tf": "..."}
    data_loader: callable(ticker, data_tf) -> OHLCV DataFrame. Caller
                 provides this so we don't hard-couple to a fetcher.
    cost_profile_lookup: callable(ticker) -> CostModel. Optional; falls
                        back to the default profile.
    """
    out = []
    n = len(asset_specs)
    for i, spec in enumerate(asset_specs):
        label = spec["label"]
        ticker = spec["ticker"]
        data_tf = spec.get("data_tf", graph.trigger.timeframe)

        if progress_callback:
            progress_callback(i / n, f"Loading {label} ({ticker})...")

        try:
            data = data_loader(ticker, data_tf)
        except Exception as e:
            out.append(AssetResult(
                asset_label=label, ticker=ticker,
                sharpe=float("nan"), n_trades=0, return_pct=0.0,
                max_dd_pct=0.0, hit_rate=0.0,
                error=f"Data fetch failed: {e}",
            ))
            continue

        if len(data) < 200:
            out.append(AssetResult(
                asset_label=label, ticker=ticker,
                sharpe=float("nan"), n_trades=0, return_pct=0.0,
                max_dd_pct=0.0, hit_rate=0.0,
                error=f"Only {len(data)} bars — too short.",
            ))
            continue

        costs = cost_profile_lookup(ticker) if cost_profile_lookup else None

        if progress_callback:
            progress_callback(
                (i + 0.5) / n, f"Backtesting {label}...",
            )

        try:
            orch = GraphOrchestrator(graph)
            result = run_backtest(data, orch, warmup_bars=warmup_bars,
                                  costs=costs)
            sm = compute_sweep_metrics(
                trades_df=result.trades_df,
                equity_curve=result.equity_curve,
                starting_balance=result.starting_balance,
                final_balance=result.final_balance,
                min_trades=min_trades,
            )
            sharpe = (sm.sharpe if sm.sharpe != float("-inf")
                      else float("nan"))
            out.append(AssetResult(
                asset_label=label, ticker=ticker,
                sharpe=sharpe, n_trades=sm.n_trades,
                return_pct=sm.return_pct,
                max_dd_pct=sm.max_drawdown_pct,
                hit_rate=sm.hit_rate,
                error=sm.disqualified_reason,
            ))
        except Exception as e:
            out.append(AssetResult(
                asset_label=label, ticker=ticker,
                sharpe=float("nan"), n_trades=0, return_pct=0.0,
                max_dd_pct=0.0, hit_rate=0.0,
                error=f"Backtest crashed: {e}",
            ))

    if progress_callback:
        progress_callback(1.0, "Done")

    return MultiAssetResult(
        candidate_label=str(graph.trigger.strategy_key),
        assets=out,
    )


# ====================================================================
#                  MONTE CARLO + BOOTSTRAP
# ====================================================================
def monte_carlo_candidate(
    data: pd.DataFrame,
    graph: DecisionGraph,
    n_simulations: int = 1000,
    warmup_bars: int = 50,
    costs=None,
    seed: int = 42,
) -> MonteCarloResult:
    """Run the candidate, then shuffle its trade order N times."""
    orch = GraphOrchestrator(graph)
    result = run_backtest(data, orch, warmup_bars=warmup_bars, costs=costs)
    return monte_carlo_trade_shuffle(
        result, n_simulations=n_simulations, seed=seed,
    )


def bootstrap_candidate(
    data: pd.DataFrame,
    graph: DecisionGraph,
    n_simulations: int = 1000,
    warmup_bars: int = 50,
    costs=None,
    seed: int = 42,
    bars_per_year: int = 252,
) -> BootstrapResult:
    """
    Bootstrap CI on the candidate's Sharpe + profit factor.
    Resamples trade P&Ls with replacement.
    """
    orch = GraphOrchestrator(graph)
    result = run_backtest(data, orch, warmup_bars=warmup_bars, costs=costs)
    return bootstrap_metrics(
        result, n_simulations=n_simulations, seed=seed,
        bars_per_year=bars_per_year,
    )
