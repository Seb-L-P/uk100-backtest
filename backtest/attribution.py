"""
Performance attribution — break down trade results by time slices.

When a strategy "works", it usually works during specific conditions: certain
hours of the day, certain market regimes, certain volatility levels. This
module slices a trade log by various dimensions and reports per-slice metrics.

Useful for:
  - "When in the day does this strategy actually win?" (often the answer is
    not what you think — e.g. you find all losses are concentrated near close)
  - "Did the equity curve come from one good month or many?"
  - "Does it work better in trending or ranging weeks?"
  - "Is loss clustered around news bars or random throughout?"

The functions all take a trades_df (as produced by run_backtest) and return
a DataFrame summarising per-slice metrics. Designed to feed directly into UI
display (st.dataframe, plotly heatmaps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _slice_metrics(trades: pd.DataFrame) -> dict:
    """Compute headline metrics for a slice of trades."""
    n = len(trades)
    if n == 0:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate_%": 0.0,
            "total_pnl_gbp": 0.0,
            "avg_pnl_gbp": 0.0,
            "expectancy_R": 0.0,
            "profit_factor": 0.0,
        }
    pnls = trades["net_pnl_gbp"]
    wins = trades[pnls > 0]
    losses = trades[pnls < 0]
    gross_wins = float(wins["net_pnl_gbp"].sum()) if not wins.empty else 0.0
    gross_losses = abs(float(losses["net_pnl_gbp"].sum())) if not losses.empty else 0.0
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    avg_loss = abs(float(losses["net_pnl_gbp"].mean())) if not losses.empty else 0.0
    expectancy_r = float(pnls.mean()) / avg_loss if avg_loss > 0 else 0.0
    return {
        "trades": n,
        "wins": len(wins),
        "win_rate_%": round(len(wins) / n * 100, 1),
        "total_pnl_gbp": round(float(pnls.sum()), 2),
        "avg_pnl_gbp": round(float(pnls.mean()), 2),
        "expectancy_R": round(expectancy_r, 3),
        "profit_factor": round(pf, 3) if not np.isinf(pf) else float("inf"),
    }


def _ensure_entry_time(trades_df: pd.DataFrame) -> pd.Series:
    """
    Get entry_time as a pandas Series (not Index) so `.dt` accessors work.
    Handles both cases: entry_time as a column or as the DataFrame's index.
    """
    if "entry_time" in trades_df.columns:
        return pd.to_datetime(trades_df["entry_time"]).reset_index(drop=True)
    # entry_time is the index — convert Index to a Series so .dt works
    return pd.Series(pd.to_datetime(trades_df.index)).reset_index(drop=True)


def by_hour_of_day(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per hour of entry."""
    if trades_df.empty:
        return pd.DataFrame(columns=["hour"] + list(_slice_metrics(trades_df).keys()))
    entry_times = _ensure_entry_time(trades_df)
    hours = entry_times.dt.hour
    rows = []
    for h in sorted(hours.unique()):
        slice_ = trades_df[hours.values == h]
        m = _slice_metrics(slice_)
        rows.append({"hour": int(h), **m})
    return pd.DataFrame(rows)


def by_day_of_week(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per day of week. Mon=0 through Sun=6."""
    if trades_df.empty:
        return pd.DataFrame(columns=["day"] + list(_slice_metrics(trades_df).keys()))
    entry_times = _ensure_entry_time(trades_df)
    dows = entry_times.dt.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rows = []
    for d in sorted(dows.unique()):
        slice_ = trades_df[dows.values == d]
        m = _slice_metrics(slice_)
        rows.append({"day": day_names[int(d)], "dow": int(d), **m})
    return pd.DataFrame(rows).sort_values("dow").drop(columns=["dow"]).reset_index(drop=True)


def by_month(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per calendar month (YYYY-MM)."""
    if trades_df.empty:
        return pd.DataFrame(columns=["month"] + list(_slice_metrics(trades_df).keys()))
    entry_times = _ensure_entry_time(trades_df)
    months = entry_times.dt.strftime("%Y-%m")
    rows = []
    for m_str in sorted(months.unique()):
        slice_ = trades_df[months.values == m_str]
        metrics = _slice_metrics(slice_)
        rows.append({"month": m_str, **metrics})
    return pd.DataFrame(rows)


def by_session_phase(
    trades_df: pd.DataFrame,
    open_hour: int = 8,
    close_hour: int = 16,
    phase_duration_hours: float = 1.0,
) -> pd.DataFrame:
    """
    Bucket trades into 'first hour' / 'middle' / 'last hour' of the trading session.

    Often reveals that a strategy is profitable in liquid mid-session but
    loses badly in the chaotic first/last hour.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["phase"] + list(_slice_metrics(trades_df).keys()))
    entry_times = _ensure_entry_time(trades_df)
    hour_decimal = entry_times.dt.hour + entry_times.dt.minute / 60.0
    phase = pd.Series("middle", index=trades_df.index)
    phase[hour_decimal.values < (open_hour + phase_duration_hours)] = "first_hour"
    phase[hour_decimal.values >= (close_hour - phase_duration_hours)] = "last_hour"

    rows = []
    for label in ["first_hour", "middle", "last_hour"]:
        slice_ = trades_df[phase.values == label]
        metrics = _slice_metrics(slice_)
        rows.append({"phase": label, **metrics})
    return pd.DataFrame(rows)


def by_side(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Long vs short breakdown — does the strategy work both ways?"""
    if trades_df.empty:
        return pd.DataFrame(columns=["side"] + list(_slice_metrics(trades_df).keys()))
    rows = []
    for side in ["long", "short"]:
        slice_ = trades_df[trades_df["side"] == side]
        metrics = _slice_metrics(slice_)
        rows.append({"side": side, **metrics})
    return pd.DataFrame(rows)


def by_exit_reason(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    How did each trade end (stop / target / session_end / reverse)?
    Reveals whether wins come from targets vs hitting time-out, etc.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["exit_reason"] + list(_slice_metrics(trades_df).keys()))
    rows = []
    for reason in sorted(trades_df["exit_reason"].unique()):
        slice_ = trades_df[trades_df["exit_reason"] == reason]
        metrics = _slice_metrics(slice_)
        rows.append({"exit_reason": reason, **metrics})
    return pd.DataFrame(rows)


def equity_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Underwater equity — drawdown from running peak, as a percentage.
    Use for plotting an underwater curve.
    """
    if equity_curve.empty:
        return equity_curve
    running_max = equity_curve.cummax()
    return (equity_curve / running_max - 1.0) * 100.0


def monthly_return_heatmap_data(equity_curve: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by year with columns for each month (1-12),
    suitable for heatmap rendering. Cell value = % return in that month.
    """
    if equity_curve.empty:
        return pd.DataFrame()
    monthly_ending = equity_curve.resample("ME").last()
    monthly_returns = monthly_ending.pct_change() * 100
    df = pd.DataFrame({
        "year": monthly_returns.index.year,
        "month": monthly_returns.index.month,
        "return_%": monthly_returns.values,
    })
    pivot = df.pivot(index="year", columns="month", values="return_%")
    return pivot.round(2)
