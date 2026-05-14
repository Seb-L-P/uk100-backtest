"""
Validation tools — the part that keeps us honest.

Six complementary techniques (the first three are basic, the last three are
statistical-rigour upgrades for when you want to actually trust a result):

1. **Holdout split**: chop off the most recent X% of data and DON'T touch it
   during development. Every strategy tweak happens on the in-sample portion.

2. **Walk-forward**: split the in-sample period into K consecutive windows
   and run the strategy on each. Tests regime stability with FIXED params.

3. **Monte Carlo trade shuffling**: shuffle trade order N times. Tells you
   about path-dependence (max drawdown sensitivity to sequencing).

4. **Bootstrap confidence intervals**: resample trade P&Ls with replacement
   N times, recompute Sharpe / profit factor each time. Tells you how stable
   those headline numbers are. Wide CIs = small sample, headline is fragile.

5. **Probabilistic / Deflated Sharpe Ratio** (Bailey & López de Prado):
   PSR returns the probability your TRUE Sharpe exceeds a benchmark, given
   sample size, skew, kurtosis. DSR adjusts that for multiple testing —
   essential when you've swept many param combinations.

6. **Adaptive walk-forward optimisation**: re-fit params in each in-sample
   fold, evaluate on the next out-of-sample fold. The "real" walk-forward
   that hedge funds use. Tests whether your strategy can adapt to changing
   regimes by re-tuning, or only works with a single fixed set of params.

None of these prove a strategy works. They just make it harder for a bad
strategy to look good.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest.engine import BacktestResult, Strategy, run_backtest
from backtest.metrics import Metrics, compute_metrics


# ---- Holdout split ------------------------------------------------------
def holdout_split(data: pd.DataFrame, oos_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reserve the most recent `oos_fraction` of `data` as out-of-sample.
    Returns (in_sample, out_of_sample).

    Convention: in-sample is for ALL development work. Out-of-sample is
    touched ONCE per strategy version, at the end, to see whether the
    in-sample result generalised.
    """
    if not 0 < oos_fraction < 1:
        raise ValueError(f"oos_fraction must be in (0, 1), got {oos_fraction}")
    n = len(data)
    split = int(n * (1 - oos_fraction))
    return data.iloc[:split].copy(), data.iloc[split:].copy()


# ---- Walk-forward -------------------------------------------------------
@dataclass
class WalkForwardResult:
    fold_results: list[BacktestResult]
    fold_metrics: list[Metrics]
    fold_boundaries: list[tuple[pd.Timestamp, pd.Timestamp]]

    def summary_table(self) -> pd.DataFrame:
        """One row per fold with key metrics — quick visual stability check."""
        rows = []
        for (start, end), m in zip(self.fold_boundaries, self.fold_metrics):
            rows.append({
                "start": start,
                "end": end,
                "trades": m.num_trades,
                "return_%": round(m.total_return_pct, 2),
                "sharpe": round(m.sharpe, 2),
                "profit_factor": round(m.profit_factor, 2),
                "win_rate_%": round(m.win_rate_pct, 1),
                "max_dd_%": round(m.max_drawdown_pct, 2),
            })
        return pd.DataFrame(rows)


def walk_forward(
    data: pd.DataFrame,
    strategy_factory: Callable[[], Strategy],
    n_folds: int = 5,
    warmup_bars: int = 50,
    bars_per_year: int = 252,
) -> WalkForwardResult:
    """
    Split `data` into `n_folds` consecutive non-overlapping windows. Run a
    FRESH strategy on each (important — we want each fold to be independent;
    re-using strategy state between folds would leak information).

    Read: did the strategy perform consistently across windows, or only in
    one? Consistent = real edge (or consistent failure). Inconsistent =
    regime-dependent or overfit.
    """
    if n_folds < 2:
        raise ValueError("Need at least 2 folds")
    n = len(data)
    fold_size = n // n_folds
    if fold_size < warmup_bars * 2:
        raise ValueError(
            f"Fold size ({fold_size} bars) is too small for warmup ({warmup_bars}). "
            f"Reduce n_folds or warmup_bars."
        )

    results, metrics_list, boundaries = [], [], []
    for k in range(n_folds):
        start_i = k * fold_size
        end_i = (k + 1) * fold_size if k < n_folds - 1 else n
        fold_data = data.iloc[start_i:end_i]
        strategy = strategy_factory()  # fresh instance per fold
        result = run_backtest(fold_data, strategy, warmup_bars=warmup_bars)
        m = compute_metrics(result, bars_per_year=bars_per_year)
        results.append(result)
        metrics_list.append(m)
        boundaries.append((fold_data.index[0], fold_data.index[-1]))

    return WalkForwardResult(
        fold_results=results,
        fold_metrics=metrics_list,
        fold_boundaries=boundaries,
    )


# ---- Monte Carlo trade shuffling ----------------------------------------
@dataclass
class MonteCarloResult:
    final_balance_p5: float
    final_balance_p50: float
    final_balance_p95: float
    max_dd_p5: float           # most extreme (worst-case)
    max_dd_p50: float
    max_dd_p95: float
    win_rate_constant: float
    actual_final_balance: float
    actual_max_dd: float
    n_simulations: int

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "actual_final_balance": round(self.actual_final_balance, 2),
            "final_balance_p5": round(self.final_balance_p5, 2),
            "final_balance_p50": round(self.final_balance_p50, 2),
            "final_balance_p95": round(self.final_balance_p95, 2),
            "actual_max_dd_%": round(self.actual_max_dd, 2),
            "max_dd_p5_%": round(self.max_dd_p5, 2),
            "max_dd_p50_%": round(self.max_dd_p50, 2),
            "max_dd_p95_%": round(self.max_dd_p95, 2),
        }


def monte_carlo_trade_shuffle(
    result: BacktestResult,
    n_simulations: int = 1000,
    seed: int = 42,
) -> MonteCarloResult:
    """
    Reshuffle the order of realised trade P&Ls N times and recompute the
    equity curve + max drawdown for each shuffle.

    Why this is useful: a strategy that gets all its wins clustered early
    looks great on the equity curve but is no different from one where the
    same wins are clustered late — except the late-clustered version has a
    much worse max drawdown along the way. This sim shows the range of
    plausible outcomes given the SAME trades in a different order.
    """
    trades = result.trades_df
    if len(trades) == 0:
        return MonteCarloResult(
            final_balance_p5=result.starting_balance,
            final_balance_p50=result.starting_balance,
            final_balance_p95=result.starting_balance,
            max_dd_p5=0.0, max_dd_p50=0.0, max_dd_p95=0.0,
            win_rate_constant=0.0,
            actual_final_balance=result.final_balance,
            actual_max_dd=0.0,
            n_simulations=n_simulations,
        )

    pnls = trades["net_pnl_gbp"].values
    start = result.starting_balance
    rng = np.random.default_rng(seed)

    finals = np.empty(n_simulations)
    max_dds_pct = np.empty(n_simulations)

    for s in range(n_simulations):
        shuffled = rng.permutation(pnls)
        equity = start + np.cumsum(shuffled)
        finals[s] = equity[-1]
        running_max = np.maximum.accumulate(np.concatenate([[start], equity]))
        dd_pct = (np.concatenate([[start], equity]) / running_max - 1) * 100
        max_dds_pct[s] = float(dd_pct.min())

    # Actual max DD from the real run
    actual_eq = result.equity_curve.values
    actual_running_max = np.maximum.accumulate(actual_eq)
    actual_dd_pct = (actual_eq / actual_running_max - 1) * 100
    actual_max_dd = float(actual_dd_pct.min()) if len(actual_dd_pct) else 0.0

    win_rate = (pnls > 0).mean() * 100.0

    return MonteCarloResult(
        final_balance_p5=float(np.percentile(finals, 5)),
        final_balance_p50=float(np.percentile(finals, 50)),
        final_balance_p95=float(np.percentile(finals, 95)),
        max_dd_p5=float(np.percentile(max_dds_pct, 5)),
        max_dd_p50=float(np.percentile(max_dds_pct, 50)),
        max_dd_p95=float(np.percentile(max_dds_pct, 95)),
        win_rate_constant=float(win_rate),
        actual_final_balance=float(result.final_balance),
        actual_max_dd=actual_max_dd,
        n_simulations=n_simulations,
    )


# ---- Bootstrap confidence intervals ------------------------------------
@dataclass
class BootstrapResult:
    """
    Confidence intervals for headline metrics by resampling trade P&Ls.

    Standard bootstrap = sample with replacement (assumes trades are i.i.d.).
    Block bootstrap = sample blocks of consecutive trades (preserves serial
    autocorrelation, which can be present if trades cluster in regimes).
    """
    sharpe_p5: float
    sharpe_p50: float
    sharpe_p95: float
    profit_factor_p5: float
    profit_factor_p50: float
    profit_factor_p95: float
    n_simulations: int
    block_size: int | None     # None = standard bootstrap

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "block_size": self.block_size,
            "sharpe_p5": round(self.sharpe_p5, 3),
            "sharpe_p50": round(self.sharpe_p50, 3),
            "sharpe_p95": round(self.sharpe_p95, 3),
            "profit_factor_p5": round(self.profit_factor_p5, 3),
            "profit_factor_p50": round(self.profit_factor_p50, 3),
            "profit_factor_p95": round(self.profit_factor_p95, 3),
        }


def bootstrap_metrics(
    trades_df: pd.DataFrame,
    n_simulations: int = 1000,
    block_size: int | None = None,
    seed: int = 42,
) -> BootstrapResult:
    """
    Bootstrap CIs for trade-level Sharpe and profit factor.

    `block_size`:
      - None: standard bootstrap (sample individual trades with replacement)
      - int K: block bootstrap (sample contiguous blocks of K trades)

    Sharpe is per-trade (mean / std), not annualised — useful for relative
    comparison and CI width, but don't compare to typical "annualised Sharpe"
    benchmarks directly.
    """
    if len(trades_df) < 10:
        nan = float("nan")
        return BootstrapResult(nan, nan, nan, nan, nan, nan, n_simulations, block_size)

    pnls = trades_df["net_pnl_gbp"].to_numpy()
    n = len(pnls)
    rng = np.random.default_rng(seed)

    sharpes = np.empty(n_simulations)
    pfs = np.empty(n_simulations)

    for s in range(n_simulations):
        if block_size is None or block_size <= 1:
            sample = rng.choice(pnls, size=n, replace=True)
        else:
            n_blocks = max(1, n // block_size)
            blocks = []
            for _ in range(n_blocks):
                start = int(rng.integers(0, n - block_size + 1))
                blocks.append(pnls[start:start + block_size])
            sample = np.concatenate(blocks)[:n]

        wins = sample[sample > 0]
        losses = sample[sample < 0]
        gross_wins = wins.sum() if len(wins) else 0.0
        gross_losses = abs(losses.sum()) if len(losses) else 1e-9
        pfs[s] = gross_wins / gross_losses

        std = sample.std()
        sharpes[s] = (sample.mean() / std) if std > 0 else 0.0

    return BootstrapResult(
        sharpe_p5=float(np.percentile(sharpes, 5)),
        sharpe_p50=float(np.percentile(sharpes, 50)),
        sharpe_p95=float(np.percentile(sharpes, 95)),
        profit_factor_p5=float(np.percentile(pfs, 5)),
        profit_factor_p50=float(np.percentile(pfs, 50)),
        profit_factor_p95=float(np.percentile(pfs, 95)),
        n_simulations=n_simulations,
        block_size=block_size,
    )


# ---- Probabilistic / Deflated Sharpe Ratio ------------------------------
def probabilistic_sharpe_ratio(
    trades_df: pd.DataFrame,
    benchmark_sr: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    Returns the probability that the strategy's TRUE Sharpe ratio exceeds
    `benchmark_sr`, given the observed Sharpe, sample size, and the higher
    moments (skewness, kurtosis) of the trade returns.

    Interpretation:
      PSR > 0.95 → 95% confident strategy beats benchmark Sharpe (strong)
      PSR ≈ 0.5 → no real evidence either way (noise)
      PSR < 0.05 → 95% confident strategy is WORSE than benchmark

    Returns NaN if too few trades (<30) for a reliable estimate.
    """
    if len(trades_df) < 30:
        return float("nan")

    pnls = trades_df["net_pnl_gbp"].to_numpy()
    if pnls.std() == 0:
        return float("nan")

    sr = pnls.mean() / pnls.std()
    n = len(pnls)
    skew = float(pd.Series(pnls).skew())
    # pandas .kurtosis() returns excess kurtosis; raw kurtosis = excess + 3
    kurt = float(pd.Series(pnls).kurtosis()) + 3.0

    numerator = (sr - benchmark_sr) * math.sqrt(n - 1)
    denominator_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denominator_sq <= 0:
        return float("nan")
    z = numerator / math.sqrt(denominator_sq)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    trades_df: pd.DataFrame,
    n_trials: int,
    sharpe_variance_across_trials: float,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts PSR for multiple-testing bias. When you've swept N parameter
    combinations and found the one with the highest Sharpe, you should
    NOT compare it to a benchmark of 0 — by chance alone, the best of N
    will be elevated. DSR computes the appropriate elevated benchmark.

    Args:
      trades_df: trades from the BEST parameter combination
      n_trials: number of parameter combinations you tested
      sharpe_variance_across_trials: variance of Sharpe ratios across trials

    Returns probability the BEST is genuinely better than what you'd get
    from running N random strategies. > 0.95 = strong evidence of real edge.
    """
    if len(trades_df) < 30 or n_trials < 2:
        return float("nan")

    # Euler-Mascheroni constant
    gamma = 0.5772156649

    # Adjusted benchmark: E[max of N standard normals approximation]
    # SR_0 = sqrt(V) * ((1-gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N*e)))
    sd_sr = math.sqrt(sharpe_variance_across_trials)
    inv1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    inv2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    benchmark = sd_sr * ((1 - gamma) * inv1 + gamma * inv2)

    return probabilistic_sharpe_ratio(trades_df, benchmark_sr=benchmark)


# ---- Adaptive walk-forward (true walk-forward optimisation) ------------
@dataclass
class AdaptiveFold:
    """One fold of an adaptive walk-forward run."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen_params: dict
    train_metric_value: float       # the metric value on train (in-sample to that fold)
    test_metrics: Metrics
    test_result: BacktestResult


@dataclass
class AdaptiveWalkForwardResult:
    folds: list[AdaptiveFold]
    optimization_metric: str

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            row = {
                "train_period": f"{f.train_start.date()} → {f.train_end.date()}",
                "test_period": f"{f.test_start.date()} → {f.test_end.date()}",
                f"train_{self.optimization_metric}": round(f.train_metric_value, 3),
                "test_trades": f.test_metrics.num_trades,
                "test_sharpe": round(f.test_metrics.sharpe, 3),
                "test_pf": round(f.test_metrics.profit_factor, 3),
                "test_return_%": round(f.test_metrics.total_return_pct, 2),
            }
            # Add chosen params as columns
            for k, v in f.chosen_params.items():
                row[f"param_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def aggregate_test_metrics(self) -> dict:
        """Pool all test-fold trades and report combined metrics."""
        if not self.folds:
            return {}
        total_return = sum(f.test_metrics.total_return_pct for f in self.folds)
        avg_sharpe = np.mean([f.test_metrics.sharpe for f in self.folds])
        avg_pf = np.mean([f.test_metrics.profit_factor for f in self.folds
                          if not np.isinf(f.test_metrics.profit_factor)])
        consistency = sum(1 for f in self.folds if f.test_metrics.profit_factor > 1.0)
        consistency_pct = consistency / len(self.folds)
        return {
            "n_folds": len(self.folds),
            "sum_test_returns_%": round(total_return, 2),
            "avg_test_sharpe": round(float(avg_sharpe), 3),
            "avg_test_profit_factor": round(float(avg_pf), 3),
            "test_consistency": round(consistency_pct, 2),
        }


def adaptive_walk_forward(
    data: pd.DataFrame,
    factory_for_params: Callable[[dict], Callable[[], Strategy]],
    param_grid: dict[str, list],
    n_folds: int = 5,
    optimization_metric: str = "sharpe",
    warmup_bars: int = 50,
    bars_per_year: int = 252,
    min_train_folds: int = 1,
) -> AdaptiveWalkForwardResult:
    """
    True walk-forward optimisation:

      For each test fold k (starting from `min_train_folds`):
        1. Use folds [0..k-1] as TRAINING data.
        2. Sweep `param_grid` on that training data, pick the params that
           maximise `optimization_metric`.
        3. Run that single param set on fold k as TEST.
        4. Record the test result.
      Aggregate.

    What this tests: does the strategy require params that change over time
    (and can it find them in the past), and does that re-tuning generalise
    to the immediate future? A strategy that needs the SAME magic params
    every fold is brittle. A strategy where re-tuning each fold produces
    consistent test profitability is robust to regime shifts.

    Note this is significantly more expensive than fixed walk-forward —
    ~n_folds × len(param_grid) backtests instead of n_folds.
    """
    if n_folds < min_train_folds + 1:
        raise ValueError(f"Need at least {min_train_folds + 1} folds for adaptive WF")

    n = len(data)
    fold_size = n // n_folds
    if fold_size < warmup_bars * 2:
        raise ValueError(f"Fold size ({fold_size}) too small for warmup ({warmup_bars})")

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    combos = list(product(*value_lists))

    folds_out: list[AdaptiveFold] = []

    for k in range(min_train_folds, n_folds):
        train_end_i = k * fold_size
        test_start_i = train_end_i
        test_end_i = (k + 1) * fold_size if k < n_folds - 1 else n
        train_data = data.iloc[:train_end_i]
        test_data = data.iloc[test_start_i:test_end_i]

        # Sweep on train
        best_score = float("-inf")
        best_params: dict | None = None
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                factory = factory_for_params(params)
                tr_result = run_backtest(train_data, factory(), warmup_bars=warmup_bars)
                tr_metrics = compute_metrics(tr_result, bars_per_year=bars_per_year)
                # Need at least some trades to consider this a valid candidate
                if tr_metrics.num_trades < 5:
                    continue
                score = float(getattr(tr_metrics, optimization_metric, 0.0) or 0.0)
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue

        if best_params is None:
            continue

        # Test on next fold with the chosen params
        factory = factory_for_params(best_params)
        test_result = run_backtest(test_data, factory(), warmup_bars=warmup_bars)
        test_metrics = compute_metrics(test_result, bars_per_year=bars_per_year)

        folds_out.append(AdaptiveFold(
            train_start=train_data.index[0],
            train_end=train_data.index[-1],
            test_start=test_data.index[0],
            test_end=test_data.index[-1],
            chosen_params=best_params,
            train_metric_value=best_score,
            test_metrics=test_metrics,
            test_result=test_result,
        ))

    return AdaptiveWalkForwardResult(folds=folds_out, optimization_metric=optimization_metric)


# ---- Combined validation report -----------------------------------------
@dataclass
class ValidationReport:
    in_sample_metrics: Metrics
    out_of_sample_metrics: Metrics
    walk_forward: WalkForwardResult
    monte_carlo: MonteCarloResult
    bootstrap: BootstrapResult | None = None
    psr: float | None = None    # Probabilistic Sharpe Ratio of in-sample run vs zero

    def consistency_score(self) -> float:
        """
        Crude robustness check: across walk-forward folds, what fraction had
        profit factor > 1.0? 1.0 = every fold profitable, 0.0 = every fold lost.
        Anything < 0.5 means the strategy is regime-dependent at best.
        """
        if not self.walk_forward.fold_metrics:
            return 0.0
        wins = sum(1 for m in self.walk_forward.fold_metrics if m.profit_factor > 1.0)
        return wins / len(self.walk_forward.fold_metrics)

    def is_to_oos_drift(self) -> float | None:
        """
        Drift between in-sample and out-of-sample Sharpe. Negative drift means
        the strategy looked better in-sample than out-of-sample — classic
        overfit signature.
        """
        if self.in_sample_metrics is None or self.out_of_sample_metrics is None:
            return None
        return self.out_of_sample_metrics.sharpe - self.in_sample_metrics.sharpe


def run_full_validation(
    data: pd.DataFrame,
    strategy_factory: Callable[[], Strategy],
    oos_fraction: float = 0.2,
    n_folds: int = 4,
    n_mc_simulations: int = 1000,
    warmup_bars: int = 50,
    bars_per_year: int = 252,
) -> tuple[BacktestResult, BacktestResult, ValidationReport]:
    """
    Run all three validation techniques in one go.

    Returns (in_sample_result, out_of_sample_result, ValidationReport).
    The two BacktestResult objects let the caller write per-period reports.
    """
    in_sample, oos = holdout_split(data, oos_fraction=oos_fraction)

    # In-sample: walk-forward + Monte Carlo + final IS metrics + bootstrap + PSR
    is_result = run_backtest(in_sample, strategy_factory(), warmup_bars=warmup_bars)
    is_metrics = compute_metrics(is_result, bars_per_year=bars_per_year)
    wf = walk_forward(in_sample, strategy_factory, n_folds=n_folds,
                      warmup_bars=warmup_bars, bars_per_year=bars_per_year)
    mc = monte_carlo_trade_shuffle(is_result, n_simulations=n_mc_simulations)
    bs = bootstrap_metrics(is_result.trades_df, n_simulations=1000) if not is_result.trades_df.empty else None
    psr_value = probabilistic_sharpe_ratio(is_result.trades_df) if not is_result.trades_df.empty else None

    # Out-of-sample: single run, no parameter peeking
    oos_result = run_backtest(oos, strategy_factory(), warmup_bars=warmup_bars)
    oos_metrics = compute_metrics(oos_result, bars_per_year=bars_per_year)

    report = ValidationReport(
        in_sample_metrics=is_metrics,
        out_of_sample_metrics=oos_metrics,
        walk_forward=wf,
        monte_carlo=mc,
        bootstrap=bs,
        psr=psr_value,
    )
    return is_result, oos_result, report


# ---- Reporting ----------------------------------------------------------
def write_validation_report(
    in_sample_result: BacktestResult,
    oos_result: BacktestResult,
    report: ValidationReport,
    output_dir,
    run_name: str,
) -> "Path":
    """
    Write a single markdown report that shows in-sample vs out-of-sample,
    walk-forward fold table, and Monte Carlo percentiles side-by-side.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_m = report.in_sample_metrics
    oos_m = report.out_of_sample_metrics
    mc = report.monte_carlo
    wf_table = report.walk_forward.summary_table()
    drift = report.is_to_oos_drift()
    consistency = report.consistency_score()

    lines = [
        f"# Validation report: {run_name}",
        f"",
        f"## In-sample vs out-of-sample",
        f"",
        f"| Metric | In-sample | Out-of-sample | Drift |",
        f"|---|---|---|---|",
        f"| Period | {in_sample_result.equity_curve.index[0].date()} → {in_sample_result.equity_curve.index[-1].date()} "
        f"| {oos_result.equity_curve.index[0].date()} → {oos_result.equity_curve.index[-1].date()} | |",
        f"| Trades | {is_m.num_trades} | {oos_m.num_trades} | |",
        f"| Total return | {is_m.total_return_pct:+.2f}% | {oos_m.total_return_pct:+.2f}% | {oos_m.total_return_pct - is_m.total_return_pct:+.2f}pp |",
        f"| Sharpe | {is_m.sharpe:.2f} | {oos_m.sharpe:.2f} | {oos_m.sharpe - is_m.sharpe:+.2f} |",
        f"| Profit factor | {is_m.profit_factor:.2f} | {oos_m.profit_factor:.2f} | {oos_m.profit_factor - is_m.profit_factor:+.2f} |",
        f"| Win rate | {is_m.win_rate_pct:.1f}% | {oos_m.win_rate_pct:.1f}% | {oos_m.win_rate_pct - is_m.win_rate_pct:+.1f}pp |",
        f"| Max DD | {is_m.max_drawdown_pct:.2f}% | {oos_m.max_drawdown_pct:.2f}% | {oos_m.max_drawdown_pct - is_m.max_drawdown_pct:+.2f}pp |",
        f"",
        f"**Sharpe drift (OOS - IS):** {drift:+.2f}" if drift is not None else "",
        f"Large negative drift suggests in-sample overfit. Near zero is good. Positive (rare) suggests in-sample was unlucky.",
        f"",
        f"## Walk-forward folds (in-sample only)",
        f"",
        f"Each fold is an independent run of the strategy on a consecutive slice of in-sample data.",
        f"Consistency = fraction of folds with profit factor > 1.0.",
        f"",
        f"**Consistency score: {consistency:.0%}**",
        f"",
        wf_table.to_markdown(index=False) if not wf_table.empty else "(no folds)",
        f"",
        f"## Monte Carlo trade-order simulation (in-sample, {mc.n_simulations} shuffles)",
        f"",
        f"How sensitive is the *path* to the *order* trades occurred in?",
        f"Final balance is order-invariant (it's just sum of trade P&Ls), but max",
        f"drawdown depends on sequencing. Wide DD range = your real-world experience",
        f"could have been much worse (or better) than what you saw, by luck of order.",
        f"",
        f"Final balance (same for all shuffles, mathematical certainty): **£{mc.actual_final_balance:,.2f}**",
        f"",
        f"| Percentile | Max drawdown |",
        f"|---|---|",
        f"| p5 (worst-case shuffle) | {mc.max_dd_p5:.2f}% |",
        f"| p50 (median shuffle) | {mc.max_dd_p50:.2f}% |",
        f"| p95 (best-case shuffle) | {mc.max_dd_p95:.2f}% |",
        f"| **Actual sequence** | **{mc.actual_max_dd:.2f}%** |",
        f"",
        *_bootstrap_section(report.bootstrap),
        *_psr_section(report.psr, report.in_sample_metrics.num_trades),
        f"## Verdict",
        f"",
        *_verdict_lines(report),
    ]
    report_path = output_dir / f"{run_name}_validation.md"
    report_path.write_text("\n".join(l for l in lines if l is not None))
    return report_path


def _bootstrap_section(bs: BootstrapResult | None) -> list[str]:
    if bs is None or math.isnan(bs.sharpe_p50):
        return []
    return [
        f"## Bootstrap confidence intervals (in-sample, {bs.n_simulations} resamples)",
        f"",
        f"How stable are the headline numbers if we resample the trades? Wide bands = small sample, fragile result.",
        f"",
        f"| Metric | p5 (worst case) | p50 (median) | p95 (best case) |",
        f"|---|---|---|---|",
        f"| Per-trade Sharpe | {bs.sharpe_p5:.3f} | {bs.sharpe_p50:.3f} | {bs.sharpe_p95:.3f} |",
        f"| Profit factor | {bs.profit_factor_p5:.3f} | {bs.profit_factor_p50:.3f} | {bs.profit_factor_p95:.3f} |",
        f"",
    ]


def _psr_section(psr: float | None, n_trades: int) -> list[str]:
    if psr is None or math.isnan(psr):
        if n_trades < 30:
            return [
                f"## Probabilistic Sharpe Ratio",
                f"",
                f"_Not computed — only {n_trades} trades; need ≥30 for a reliable estimate._",
                f"",
            ]
        return []
    interp = (
        "**strong evidence** of positive edge" if psr > 0.95 else
        "**moderate evidence** of positive edge" if psr > 0.75 else
        "weak / inconclusive — likely noise" if psr > 0.5 else
        "evidence the strategy is **NOT profitable**"
    )
    return [
        f"## Probabilistic Sharpe Ratio (in-sample)",
        f"",
        f"Probability that the TRUE Sharpe exceeds 0, accounting for sample size, skew, and kurtosis:",
        f"",
        f"**PSR = {psr:.1%}** — {interp}",
        f"",
        f"_For context: PSR > 0.95 = 95% confident the strategy beats random; PSR ≈ 0.5 = noise._",
        f"",
    ]


def _verdict_lines(report: ValidationReport) -> list[str]:
    """Opinionated interpretation. Use as a starting point, not gospel."""
    lines = []
    is_m = report.in_sample_metrics
    oos_m = report.out_of_sample_metrics
    drift = report.is_to_oos_drift() or 0.0
    consistency = report.consistency_score()

    # PSR-based first verdict (most rigorous single number)
    if report.psr is not None and not math.isnan(report.psr):
        if report.psr > 0.95:
            lines.append(f"- ✅ PSR {report.psr:.0%} — strong statistical evidence of positive edge.")
        elif report.psr > 0.75:
            lines.append(f"- 🟡 PSR {report.psr:.0%} — moderate evidence; not conclusive.")
        elif report.psr > 0.5:
            lines.append(f"- ⚠️ PSR {report.psr:.0%} — weak / borderline; treat as inconclusive noise.")
        else:
            lines.append(f"- ❌ PSR {report.psr:.0%} — evidence strategy is NOT profitable.")

    # Bootstrap CI commentary
    if report.bootstrap is not None and not math.isnan(report.bootstrap.sharpe_p50):
        bs = report.bootstrap
        if bs.profit_factor_p5 > 1.0:
            lines.append(f"- ✅ Bootstrap p5 profit factor = {bs.profit_factor_p5:.2f} > 1.0 — even worst-case resample is profitable.")
        elif bs.profit_factor_p95 < 1.0:
            lines.append(f"- ❌ Bootstrap p95 profit factor = {bs.profit_factor_p95:.2f} < 1.0 — even best-case resample loses money.")
        elif bs.sharpe_p95 - bs.sharpe_p5 > 0.5:
            lines.append(f"- ⚠️ Bootstrap Sharpe spans {bs.sharpe_p5:.2f} to {bs.sharpe_p95:.2f} — wide CI, headline number is fragile.")

    # Cases
    if is_m.num_trades < 30 or oos_m.num_trades < 10:
        lines.append("- ⚠️ **Sample size too small** for reliable conclusions. Need more data or higher-frequency.")
    if is_m.profit_factor < 1.0 and oos_m.profit_factor < 1.0:
        lines.append("- ❌ Strategy lost on BOTH in-sample and out-of-sample. No edge — abandon or rework.")
    elif is_m.profit_factor > 1.3 and oos_m.profit_factor > 1.3 and consistency >= 0.6:
        lines.append("- ✅ Profitable on both samples AND consistent across walk-forward folds. **Worth iterating on.**")
    elif is_m.profit_factor > 1.3 and oos_m.profit_factor < 1.0:
        lines.append("- 🚨 **Looks overfit** — works in-sample, fails out-of-sample. Drift is the classic overfit signature.")
    elif drift < -0.5:
        lines.append(f"- 🚨 Sharpe degraded by {abs(drift):.2f} from in-sample to out-of-sample. Possible overfit.")
    elif consistency < 0.5:
        lines.append(f"- ⚠️ Inconsistent across walk-forward folds ({consistency:.0%}). Regime-dependent — may not work in current market.")

    # Monte Carlo — compare max DD, since final balance is order-invariant
    mc = report.monte_carlo
    if mc.actual_max_dd < mc.max_dd_p5:  # actual worse than p5 worst-case
        lines.append(f"- ℹ️ Actual max drawdown ({mc.actual_max_dd:.1f}%) was worse than 95% of shuffled sequences. You got unlucky with timing.")
    elif mc.actual_max_dd > mc.max_dd_p95:  # actual better than p95 best-case
        lines.append(f"- ℹ️ Actual max drawdown ({mc.actual_max_dd:.1f}%) was better than 95% of shuffled sequences. You got lucky with timing — expect deeper drawdowns live.")
    elif mc.actual_max_dd < mc.max_dd_p50:
        lines.append(f"- ℹ️ Actual max drawdown ({mc.actual_max_dd:.1f}%) was worse than the median shuffle ({mc.max_dd_p50:.1f}%). Slightly unlucky sequencing.")

    if not lines:
        lines.append("- Result is borderline / inconclusive. More data or further iteration needed.")
    return lines
