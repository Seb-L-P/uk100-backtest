"""
FVG retest with scale-out + trailing — demonstrates the new engine features.

Same entry trigger as FvgRetest (3-bar imbalance, retest at near edge), but
the exit logic is more sophisticated:

  1. Enter at FVG near edge with stop past the far edge (same as before).
  2. When price moves 1R in your favour, SCALE OUT 50% of the position and
     move the stop to break-even on the remaining 50% (the runner).
  3. The runner trails with ATR; only ratchets favourably.

This is a real-world "let winners run, lock in gains" pattern. Worth comparing
against the basic FvgRetest 2R fixed-target version to see whether scale-out
actually improves outcomes or just adds complexity.

Note: because we have a runner that may be open across many bars, this
strategy may produce notably more financing cost than the fixed-target
version in swing time-frames. On true intraday (flat by close) it should be
similar.
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker, OpenPosition
from backtest.engine import Strategy, Signal
from backtest.indicators import detect_fvg, FVG
from backtest.exits import atr_trailing, breakeven_after_R, combine
from strategies._helpers import risk_based_stake, in_session
from config import ACCOUNT


class FvgScaleOut:
    def __init__(
        self,
        min_gap_points: float = 5.0,
        max_gap_points: float = 50.0,
        max_age_bars: int = 30,
        stop_buffer_pts: float = 2.0,
        scale_at_R: float = 1.0,
        scale_fraction: float = 0.5,
        trail_atr_period: int = 14,
        trail_atr_mult: float = 2.0,
        session_open: time = time(9, 0),
        session_close: time = time(15, 0),
        flat_by: time = time(15, 30),
    ):
        self.min_gap_points = min_gap_points
        self.max_gap_points = max_gap_points
        self.max_age_bars = max_age_bars
        self.stop_buffer_pts = stop_buffer_pts
        self.scale_at_R = scale_at_R
        self.scale_fraction = scale_fraction
        self.trail_atr_period = trail_atr_period
        self.trail_atr_mult = trail_atr_mult
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by

        self._open_fvgs: list[FVG] = []
        self._last_processed_index = -1
        # Track which positions we've scaled out of, to avoid double-scaling
        self._scaled_out: set[str] = set()
        # Track entry risk per position so we know what 1R is for each
        self._entry_risk: dict[str, float] = {}

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        bar = history.iloc[i]
        bar_low, bar_high, bar_close = float(bar["Low"]), float(bar["High"]), float(bar["Close"])
        now = history.index[i].time()

        # ---- Bind entry risk to any newly-opened position ----
        # When we sent an open signal last bar, broker has now created the
        # position with an id. Record our planned risk for that id so the
        # scale-out check below knows what 1R means for it.
        pending = getattr(self, "_pending_entry_risk", None)
        if pending is not None:
            for pos in broker.positions:
                if pos.id not in self._entry_risk:
                    self._entry_risk[pos.id] = pending
                    break
            self._pending_entry_risk = None

        # Garbage-collect entries for positions that have closed
        active_ids = {p.id for p in broker.positions}
        self._entry_risk = {k: v for k, v in self._entry_risk.items() if k in active_ids}
        self._scaled_out = {pid for pid in self._scaled_out if pid in active_ids}

        # ---- Update FVG list ----
        if i != self._last_processed_index:
            self._last_processed_index = i
            new_fvg = detect_fvg(history, i)
            if new_fvg is not None and self.min_gap_points <= new_fvg.size_points <= self.max_gap_points:
                self._open_fvgs.append(new_fvg)
            self._open_fvgs = [
                f for f in self._open_fvgs
                if (i - f.creator_bar_index) <= self.max_age_bars
                and not f.is_filled_by(bar_low, bar_high)
            ]

        # ---- Force-flat at session end ----
        if broker.positions and now >= self.flat_by:
            return Signal(action="close_all", reason="session_end")

        # ---- Scale-out logic ----
        # If we have an open position that has hit 1R and we haven't scaled yet, scale 50%
        for pos in broker.positions:
            if pos.id in self._scaled_out:
                continue
            entry_risk = self._entry_risk.get(pos.id)
            if entry_risk is None:
                continue
            target_pts = self.scale_at_R * entry_risk
            if pos.side == "long":
                hit_target = bar_high - pos.entry_price >= target_pts
            else:
                hit_target = pos.entry_price - bar_low >= target_pts
            if hit_target:
                self._scaled_out.add(pos.id)
                # Scale out + leave runner with ATR trailing (set on entry already)
                # Note: returned scale_out will fill at NEXT bar's open, not this bar.
                return Signal(
                    action="scale_out",
                    position_id=pos.id,
                    scale_fraction=self.scale_fraction,
                    reason=f"scale_at_{self.scale_at_R}R",
                )

        # ---- Don't open new entries outside session or while in a position ----
        if not in_session(now, self.session_open, self.session_close):
            return Signal(action="noop")
        if broker.positions:  # already in a trade — don't stack FVGs
            return Signal(action="noop")

        # ---- Entry: any unfilled FVG retested by this bar ----
        candidates = [
            f for f in self._open_fvgs
            if f.creator_bar_index < i and f.is_touched_by(bar_low, bar_high)
        ]
        if not candidates:
            return Signal(action="noop")
        fvg = max(candidates, key=lambda f: f.creator_bar_index)

        if fvg.direction == "bullish":
            entry = fvg.near_edge
            stop = fvg.far_edge - self.stop_buffer_pts
            risk = entry - stop
            action = "open_long"
        else:
            entry = fvg.near_edge
            stop = fvg.far_edge + self.stop_buffer_pts
            risk = stop - entry
            action = "open_short"

        if risk <= 0:
            return Signal(action="noop")

        stake = risk_based_stake(broker.balance, risk, price=entry)

        # Compose trailing: breakeven after 1R, then ATR trail on the runner.
        # Note: scale-out itself triggers via the strategy's on_bar logic above,
        # but the breakeven in the trailing fn is a redundant safety net.
        trailing = combine(
            breakeven_after_R(risk_pts=risk, move_to_R=1.0, plus_pts=0.5),
            atr_trailing(history, atr_period=self.trail_atr_period,
                         mult=self.trail_atr_mult),
        )

        try:
            self._open_fvgs.remove(fvg)
        except ValueError:
            pass

        # Record entry risk on the next bar — we don't have the position id
        # yet (broker assigns it). We use a side-channel by tagging via the
        # strategy: the engine creates the position with this stake/risk, and
        # we look up risk by position id on subsequent bars. We capture risk
        # via a closure on the next-bar signal callback... actually simpler:
        # store the planned risk by entry price + time, then look up after.
        self._pending_entry_risk = risk

        return Signal(
            action=action,
            stake_per_point=stake,
            stop_loss=stop,
            take_profit=None,  # no fixed target — exits via scale + trail + stop
            reason=f"{fvg.direction}_fvg_size={fvg.size_points:.1f}pts",
            trailing_stop_fn=trailing,
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """For ensemble polling — same FVG-retouch detection as FvgRetest."""
        i = len(history) - 1
        if i < 2:
            return "none"
        bar = history.iloc[i]
        bar_low, bar_high = float(bar["Low"]), float(bar["High"])
        best: FVG | None = None
        for j in range(max(2, i - self.max_age_bars), i + 1):
            fvg = detect_fvg(history, j)
            if fvg is None:
                continue
            if not (self.min_gap_points <= fvg.size_points <= self.max_gap_points):
                continue
            after = history.iloc[fvg.creator_bar_index + 1: i + 1]
            if fvg.direction == "bullish":
                if not after.empty and bool((after["Low"] <= fvg.zone_low).any()):
                    continue
            else:
                if not after.empty and bool((after["High"] >= fvg.zone_high).any()):
                    continue
            if not fvg.is_touched_by(bar_low, bar_high):
                continue
            if best is None or fvg.creator_bar_index > best.creator_bar_index:
                best = fvg
        if best is None:
            return "none"
        return "long" if best.direction == "bullish" else "short"

    def post_open(self, position: OpenPosition) -> None:
        """
        Called by external code when a new position has been opened.
        Records the entry risk so we know what 1R is for the scale-out check.

        (The engine doesn't currently call this — strategies that need it
        track entry risk themselves. We approximate here by recording risk
        via _pending_entry_risk just before the signal goes out, then
        binding it to the new position on the next on_bar call.)
        """
        risk = getattr(self, "_pending_entry_risk", None)
        if risk is not None:
            self._entry_risk[position.id] = risk
            self._pending_entry_risk = None
