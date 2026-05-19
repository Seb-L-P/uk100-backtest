"""
Objective function for the sweep: Sharpe ratio with a min-trades floor.

Why Sharpe?
  Penalises both LOW returns and HIGH volatility. A strategy with £500
  expectancy and a £200 drawdown standard-deviation beats one with £500
  expectancy and a £2000 drawdown stdev — Sharpe ratio captures that.

Why a min-trades floor?
  A graph that takes one lucky trade can post insane Sharpe by luck.
  Setting `min_trades=20` (or whatever) forces strategies to demonstrate
  they fire often enough to be statistically meaningful before they can
  win the rank.

Computation details:
  - Sharpe is computed from DAILY equity returns (resample equity_curve to
    1D, take pct_change). This makes Sharpe comparable across strategies
    that trade at vastly different frequencies (a 1m scalper vs a daily
    swing trader).
  - Annualisation factor `sqrt(252)`. Convention is `sqrt(N_periods/year)`
    — for daily returns that's 252 trading days.
  - Failing graphs (NaN std, empty trades, crashes) return -inf so they
    rank last without contaminating the leaderboard with garbage numbers.

Other metrics reported alongside Sharpe:
  - n_trades
  - final_balance
  - max_drawdown_pct
  - hit_rate
  - profit_factor
These don't drive the rank — they're for human inspection of the top trials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrialMetrics:
    """All the numbers we report per trial. Sharpe drives the rank."""
    sharpe: float                # -inf if disqualified
    n_trades: int
    final_balance: float
    starting_balance: float
    max_drawdown_pct: float      # peak-to-trough on the equity curve
    hit_rate: float              # wins / trades
    profit_factor: float         # gross wins / abs(gross losses); inf if no losses
    disqualified_reason: Optional[str] = None

    @property
    def return_pct(self) -> float:
        if self.starting_balance <= 0:
            return 0.0
        return (self.final_balance / self.starting_balance - 1.0) * 100.0


def _max_drawdown_pct(equity: pd.Series) -> float:
    """Peak-to-trough drawdown as a positive percentage."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    dd = (equity - running_peak) / running_peak.replace(0, np.nan)
    return float(abs(dd.min()) * 100.0) if not dd.dropna().empty else 0.0


def compute_metrics(
    trades_df: Optional[pd.DataFrame],
    equity_curve: pd.Series,
    starting_balance: float,
    final_balance: float,
    min_trades: int = 20,
    periods_per_year: int = 252,
) -> TrialMetrics:
    """
    Compute all reported metrics for a single backtest result.

    `min_trades` is the disqualification floor — graphs that trade fewer
    than this get Sharpe = -inf so they fall to the bottom of the
    leaderboard. Reported metrics are still populated so the user can
    inspect WHY a graph was disqualified.
    """
    n_trades = 0 if trades_df is None else int(len(trades_df))

    # Hit rate / profit factor (defined even for disqualified runs so we
    # can show them in the leaderboard).
    if n_trades > 0 and trades_df is not None and "net_pnl_gbp" in trades_df.columns:
        wins = trades_df[trades_df["net_pnl_gbp"] > 0]["net_pnl_gbp"]
        losses = trades_df[trades_df["net_pnl_gbp"] <= 0]["net_pnl_gbp"]
        hit_rate = float(len(wins) / n_trades)
        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
    else:
        hit_rate = 0.0
        profit_factor = 0.0

    max_dd = _max_drawdown_pct(equity_curve)

    # Disqualifications
    if n_trades < min_trades:
        return TrialMetrics(
            sharpe=float("-inf"),
            n_trades=n_trades,
            final_balance=float(final_balance),
            starting_balance=float(starting_balance),
            max_drawdown_pct=max_dd,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            disqualified_reason=f"only {n_trades} trades (< {min_trades})",
        )

    # Sharpe from daily equity returns. Resampling to 1D and taking the
    # last value of each day, then pct_change, gives one return per day.
    daily = equity_curve.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0 or np.isnan(rets.std()):
        return TrialMetrics(
            sharpe=float("-inf"),
            n_trades=n_trades,
            final_balance=float(final_balance),
            starting_balance=float(starting_balance),
            max_drawdown_pct=max_dd,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            disqualified_reason="zero return-variance (probably no trades net)",
        )

    sharpe = float((rets.mean() / rets.std()) * np.sqrt(periods_per_year))

    return TrialMetrics(
        sharpe=sharpe,
        n_trades=n_trades,
        final_balance=float(final_balance),
        starting_balance=float(starting_balance),
        max_drawdown_pct=max_dd,
        hit_rate=hit_rate,
        profit_factor=profit_factor,
        disqualified_reason=None,
    )
