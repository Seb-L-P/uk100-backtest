"""
3-way time-based data splitter for sweep/discovery.

The cardinal sin of strategy search is to optimise on data and then report
numbers from the same data — you'll always find "the strategy that worked
best ON THE WINDOW YOU WERE LOOKING AT". 3-way split is the defence:

    IS  (in-sample, oldest 60%)  — where the sweep optimises.
    Val (validation, middle 20%) — where we rank the top-K from IS.
                                   This is what kills "overfit to one
                                   window" strategies.
    OOS (out-of-sample, newest 20%) — UNTOUCHED until final report.
                                      Look at this LAST. The OOS Sharpe
                                      is the only one you should believe.

Each fold is a contiguous time window — we DON'T interleave bars across
folds because intraday strategies have persistent state (open positions,
inside-bar tracking, FVG zones) that would leak signal if you randomly
sampled bars.

The implementation is deliberately a pure function — no caching, no
side-effects — so the sweep runner can repeatedly invoke it with the same
inputs and get the same outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd


class DataSplit(NamedTuple):
    """Three time-contiguous DataFrames with no overlap."""
    is_df: pd.DataFrame      # in-sample, used for the sweep
    val_df: pd.DataFrame     # validation, used to rank top-K
    oos_df: pd.DataFrame     # out-of-sample, looked at LAST

    @property
    def ratios(self) -> tuple[float, float, float]:
        total = len(self.is_df) + len(self.val_df) + len(self.oos_df)
        if total == 0:
            return 0.0, 0.0, 0.0
        return (len(self.is_df) / total,
                len(self.val_df) / total,
                len(self.oos_df) / total)

    def describe(self) -> str:
        def _span(df):
            if len(df) == 0:
                return "(empty)"
            return f"{df.index[0]} → {df.index[-1]} ({len(df)} bars)"
        return (f"IS:  {_span(self.is_df)}\n"
                f"Val: {_span(self.val_df)}\n"
                f"OOS: {_span(self.oos_df)}")


def three_way_split(
    data: pd.DataFrame,
    is_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> DataSplit:
    """
    Split `data` into (IS, Val, OOS) by row position, ordered by time.

    Inputs:
      - `data`: time-indexed OHLCV DataFrame, sorted ascending.
      - `is_ratio`, `val_ratio`: between 0 and 1; OOS gets the remainder.

    Raises:
      - ValueError if ratios are nonsensical (negative, >1, etc.).
      - ValueError if the data is too short to produce non-empty splits.
    """
    if not (0 < is_ratio < 1):
        raise ValueError(f"is_ratio must be in (0, 1), got {is_ratio}")
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if is_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"is_ratio + val_ratio must be < 1 to leave room for OOS "
            f"(got {is_ratio} + {val_ratio} = {is_ratio + val_ratio})"
        )
    n = len(data)
    if n < 100:
        raise ValueError(
            f"Need at least 100 bars to split, got {n}. "
            f"Use a longer date range or a finer timeframe."
        )

    is_end = int(n * is_ratio)
    val_end = is_end + int(n * val_ratio)
    # Guarantee each split is non-empty
    if is_end == 0 or val_end == is_end or val_end == n:
        raise ValueError(
            f"Ratios produced an empty split (IS={is_end}, "
            f"Val={val_end - is_end}, OOS={n - val_end}). "
            f"Try larger ratios or more data."
        )

    is_df = data.iloc[:is_end].copy()
    val_df = data.iloc[is_end:val_end].copy()
    oos_df = data.iloc[val_end:].copy()
    return DataSplit(is_df=is_df, val_df=val_df, oos_df=oos_df)
