"""
Fair Value Gap (FVG) retest strategy — day-trade only.

Rules:
  1. On each bar, detect any new 3-bar FVG. Track unfilled ones in a rolling
     list, capped by `max_age_bars` (older FVGs are stale).
  2. Filter out FVGs that are too small (< min_gap_points) or too large
     (> max_gap_points) — small gaps are noise, large gaps blow up risk.
  3. When price retraces and TOUCHES an unfilled FVG's near edge, enter:
        Bullish FVG → go LONG at near edge (zone_high)
        Bearish FVG → go SHORT at near edge (zone_low)
  4. Stop: just past the far edge of the FVG (+ a buffer of `stop_buffer_pts`)
  5. Target: 2R from entry (configurable via `r_target`)
  6. Session filter: only enter during UK cash hours (09:00–15:00 by default,
     avoid the first 30m and last 90m). All positions closed by `session_close`.
  7. One position at a time.

Why these choices:
  - Day-trade only: avoids the financing-cost killer we saw in the SMA test.
  - 2R fixed target: simple, removes "when to exit" discretion that's easy to
    overfit. If the FVG concept has edge, 2R should work; if not, fancy exits
    won't save it.
  - Near-edge entry vs. far-edge: near-edge gives better R:R but lower fill
    probability. We're optimising for cleaner signal, not trade count.
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import detect_fvg, FVG
from config import ACCOUNT, COSTS


class FvgRetest:
    def __init__(
        self,
        min_gap_points: float = 5.0,
        max_gap_points: float = 50.0,
        max_age_bars: int = 30,
        stop_buffer_pts: float = 2.0,
        r_target: float = 2.0,
        session_open: time = time(9, 0),
        session_close: time = time(15, 0),
        flat_by: time = time(15, 30),
    ):
        self.min_gap_points = min_gap_points
        self.max_gap_points = max_gap_points
        self.max_age_bars = max_age_bars
        self.stop_buffer_pts = stop_buffer_pts
        self.r_target = r_target
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by

        # Rolling list of unfilled FVGs
        self._open_fvgs: list[FVG] = []
        # Cache of the last bar we processed, to avoid re-detecting same FVG
        self._last_processed_index: int = -1

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        bar = history.iloc[i]
        now = history.index[i].time()
        bar_low = float(bar["Low"])
        bar_high = float(bar["High"])
        bar_close = float(bar["Close"])

        # ---- 1. Update FVG list with any new FVG completed on this bar ----
        if i != self._last_processed_index:
            self._last_processed_index = i
            new_fvg = detect_fvg(history, i)
            if new_fvg is not None and self.min_gap_points <= new_fvg.size_points <= self.max_gap_points:
                self._open_fvgs.append(new_fvg)

            # Drop stale FVGs (older than max_age_bars) and filled ones
            self._open_fvgs = [
                f for f in self._open_fvgs
                if (i - f.creator_bar_index) <= self.max_age_bars
                and not f.is_filled_by(bar_low, bar_high)
            ]

        # ---- 2. Force-flat at session close ------------------------------
        if broker.position is not None and now >= self.flat_by:
            return Signal(action="close", reason="session_end")

        # ---- 3. Session filter for new entries ---------------------------
        if not (self.session_open <= now <= self.session_close):
            return Signal(action="noop")

        # ---- 4. Don't enter if already in a position ---------------------
        if broker.position is not None:
            return Signal(action="noop")

        # ---- 5. Look for an entry: any FVG touched by THIS bar -----------
        # We only consider FVGs that were created BEFORE this bar
        candidates = [
            f for f in self._open_fvgs
            if f.creator_bar_index < i and f.is_touched_by(bar_low, bar_high)
        ]
        if not candidates:
            return Signal(action="noop")

        # Pick the most recently created candidate (freshest signal)
        fvg = max(candidates, key=lambda f: f.creator_bar_index)

        # Entry at near edge; stop past far edge + buffer; target = 2R
        entry = fvg.near_edge
        if fvg.direction == "bullish":
            stop = fvg.far_edge - self.stop_buffer_pts
            risk_pts = entry - stop
            target = entry + self.r_target * risk_pts
            action = "open_long"
        else:
            stop = fvg.far_edge + self.stop_buffer_pts
            risk_pts = stop - entry
            target = entry - self.r_target * risk_pts
            action = "open_short"

        if risk_pts <= 0:
            return Signal(action="noop")

        # Position sizing: risk 1% of equity on stop distance, capped by leverage
        # to avoid over-sizing on high-priced indices like FTSE 100.
        from strategies._helpers import risk_based_stake
        stake_per_point = risk_based_stake(broker.balance, risk_pts, price=entry)

        # Remove this FVG from the open list (one shot)
        try:
            self._open_fvgs.remove(fvg)
        except ValueError:
            pass

        return Signal(
            action=action,
            stake_per_point=stake_per_point,
            stop_loss=stop,
            take_profit=target,
            reason=f"{fvg.direction}_fvg_size={fvg.size_points:.1f}pts",
        )
