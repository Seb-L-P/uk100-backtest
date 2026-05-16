"""
Balanced Price Range (BPR) — SMC concept built on top of FVG.

What it is:
  A BPR is the zone where a bullish FVG and a bearish FVG OVERLAP. The setup
  typically forms when:
    1. Price moves UP, creating a bullish FVG.
    2. Price reverses and moves DOWN through that bullish FVG area.
    3. During the down-move, a bearish FVG forms whose zone overlaps with
       the original bullish FVG.
  The intersection of the two zones is the BPR. The theory: this zone holds
  unfilled liquidity from BOTH directions, making it a particularly strong
  reversal area when price retests.

Rules:
  1. Track all unfilled bullish FVGs and unfilled bearish FVGs.
  2. On each new FVG, check whether it overlaps with any existing FVG of the
     OPPOSITE direction. If so, the overlap is a BPR. Store it.
  3. When price enters an existing BPR zone, enter against the approach
     direction:
       - Approach from above (last N bars net-down) → expect bounce → LONG
       - Approach from below (last N bars net-up) → expect rejection → SHORT
  4. Stop: just past the outer edge of the BPR + buffer.
  5. Target: 2R.
  6. UK session, day-trade only, flat by close.

Why it might work better than plain FVG:
  Filters down to only the highest-quality FVG setups (the ones that became
  BPRs). Should produce fewer but cleaner trades.

Why it might not:
  Strict overlap requirement means very few signals. With our 60-day 15m
  sample, may produce essentially no trades.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import detect_fvg, FVG
from strategies._helpers import risk_based_stake, in_session


@dataclass
class BPR:
    """The intersection zone of two opposing FVGs."""
    created_at: pd.Timestamp
    creator_bar_index: int
    zone_low: float
    zone_high: float
    bullish_fvg: FVG
    bearish_fvg: FVG

    @property
    def size_points(self) -> float:
        return self.zone_high - self.zone_low

    def is_touched_by(self, bar_low: float, bar_high: float) -> bool:
        return not (bar_high < self.zone_low or bar_low > self.zone_high)


def _find_overlap(a: FVG, b: FVG) -> tuple[float, float] | None:
    """Returns (low, high) of overlap zone if `a` and `b` overlap, else None."""
    low = max(a.zone_low, b.zone_low)
    high = min(a.zone_high, b.zone_high)
    return (low, high) if high > low else None


class BalancedPriceRange:
    def __init__(
        self,
        min_fvg_size: float = 3.0,
        max_fvg_age: int = 50,
        min_bpr_size: float = 2.0,
        max_bpr_age: int = 50,
        stop_buffer_pts: float = 2.0,
        r_target: float = 2.0,
        approach_lookback: int = 5,
        session_open: time = time(9, 0),
        session_close: time = time(15, 0),
        flat_by: time = time(15, 30),
    ):
        self.min_fvg_size = min_fvg_size
        # Cast int-typed params defensively — pandas/numpy round-trips can
        # convert ints to floats, which breaks iloc indexing below.
        self.max_fvg_age = int(max_fvg_age)
        self.min_bpr_size = min_bpr_size
        self.max_bpr_age = int(max_bpr_age)
        self.stop_buffer_pts = stop_buffer_pts
        self.r_target = r_target
        self.approach_lookback = int(approach_lookback)
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by

        self._fvgs: list[FVG] = []
        self._bprs: list[BPR] = []
        self._last_processed_index = -1

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        bar = history.iloc[i]
        bar_low = float(bar["Low"])
        bar_high = float(bar["High"])
        bar_close = float(bar["Close"])
        now = history.index[i].time()

        # Maintain FVG / BPR state (once per bar)
        if i != self._last_processed_index:
            self._last_processed_index = i
            new_fvg = detect_fvg(history, i)
            if new_fvg is not None and new_fvg.size_points >= self.min_fvg_size:
                # Check for BPR overlap with any opposite-direction existing FVG
                opposite_dir = "bearish" if new_fvg.direction == "bullish" else "bullish"
                for other in self._fvgs:
                    if other.direction != opposite_dir:
                        continue
                    overlap = _find_overlap(new_fvg, other)
                    if overlap is None:
                        continue
                    low, high = overlap
                    if high - low >= self.min_bpr_size:
                        bull = new_fvg if new_fvg.direction == "bullish" else other
                        bear = new_fvg if new_fvg.direction == "bearish" else other
                        self._bprs.append(BPR(
                            created_at=history.index[i],
                            creator_bar_index=i,
                            zone_low=low,
                            zone_high=high,
                            bullish_fvg=bull,
                            bearish_fvg=bear,
                        ))
                self._fvgs.append(new_fvg)

            # Prune stale and filled
            self._fvgs = [f for f in self._fvgs
                          if (i - f.creator_bar_index) <= self.max_fvg_age
                          and not f.is_filled_by(bar_low, bar_high)]
            self._bprs = [b for b in self._bprs
                          if (i - b.creator_bar_index) <= self.max_bpr_age]

        # Force-flat
        if broker.position is not None and now >= self.flat_by:
            return Signal(action="close", reason="session_end")
        if not in_session(now, self.session_open, self.session_close):
            return Signal(action="noop")
        if broker.position is not None:
            return Signal(action="noop")

        # Find a BPR being retested by this bar (must be created BEFORE this bar)
        candidates = [b for b in self._bprs
                      if b.creator_bar_index < i and b.is_touched_by(bar_low, bar_high)]
        if not candidates:
            return Signal(action="noop")
        bpr = max(candidates, key=lambda b: b.creator_bar_index)

        # Determine approach direction from last N bars
        if i >= self.approach_lookback:
            recent_close = history["Close"].iloc[i - self.approach_lookback]
            net_move = bar_close - recent_close
        else:
            net_move = 0

        if net_move < 0:
            # Approach from above (selling into the BPR) → expect bounce → LONG
            entry = bpr.zone_high  # near edge from above
            stop = bpr.zone_low - self.stop_buffer_pts
            risk = entry - stop
            target = entry + self.r_target * risk
            action = "open_long"
        elif net_move > 0:
            # Approach from below (rallying into BPR) → expect rejection → SHORT
            entry = bpr.zone_low
            stop = bpr.zone_high + self.stop_buffer_pts
            risk = stop - entry
            target = entry - self.r_target * risk
            action = "open_short"
        else:
            return Signal(action="noop")

        if risk <= 0:
            return Signal(action="noop")

        stake = risk_based_stake(broker.balance, risk, price=entry)

        # Remove this BPR (one-shot)
        try:
            self._bprs.remove(bpr)
        except ValueError:
            pass

        return Signal(action=action, stake_per_point=stake,
                      stop_loss=stop, take_profit=target,
                      reason=f"BPR size={bpr.size_points:.1f}, approach={'down' if net_move<0 else 'up'}")

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """
        For ensemble polling. Stateless: scans recent FVGs to find any BPR
        (overlapping opposite-direction FVGs), checks if current bar is in
        a BPR zone, returns direction based on approach.
        """
        i = len(history) - 1
        if i < 2:
            return "none"
        bar = history.iloc[i]
        bar_low, bar_high, bar_close = float(bar["Low"]), float(bar["High"]), float(bar["Close"])

        # Collect recent valid FVGs (not yet fully filled)
        bull_fvgs: list[FVG] = []
        bear_fvgs: list[FVG] = []
        for j in range(max(2, i - self.max_fvg_age), i):
            fvg = detect_fvg(history, j)
            if fvg is None or fvg.size_points < self.min_fvg_size:
                continue
            after = history.iloc[fvg.creator_bar_index + 1: i + 1]
            if fvg.direction == "bullish":
                if not after.empty and bool((after["Low"] <= fvg.zone_low).any()):
                    continue
                bull_fvgs.append(fvg)
            else:
                if not after.empty and bool((after["High"] >= fvg.zone_high).any()):
                    continue
                bear_fvgs.append(fvg)

        # Find BPRs = overlaps between bullish and bearish FVGs
        for bull in bull_fvgs:
            for bear in bear_fvgs:
                low = max(bull.zone_low, bear.zone_low)
                high = min(bull.zone_high, bear.zone_high)
                if high - low < self.min_bpr_size:
                    continue
                # Is current bar in the BPR?
                if bar_high < low or bar_low > high:
                    continue
                # Direction based on approach
                if i < self.approach_lookback:
                    return "none"
                recent_close = float(history["Close"].iloc[i - self.approach_lookback])
                net_move = bar_close - recent_close
                if net_move < 0:
                    return "long"
                if net_move > 0:
                    return "short"
        return "none"
