"""
Multi-timeframe FVG: take FVG entries on the BASE timeframe (typically 15m),
but only in the direction of the HIGHER-timeframe trend (typically 1h or 4h).

The classic SMC pattern: "trade with the higher timeframe trend." If 1h is
in an uptrend, only long FVGs are taken. If 1h is in a downtrend, only shorts.

How HTF trend is defined here:
  - Compute EMA(htf_ema_period) on the higher timeframe.
  - Trend is "up" if HTF close > EMA AND EMA is sloping up.
  - Trend is "down" if HTF close < EMA AND EMA is sloping down.
  - Else "neutral" — no trades.

Why this might add value:
  - Filters out counter-trend FVGs that often fail.
  - The 1h trend reflects regime; trading with it concentrates trades during
    aligned conditions.

Why it might not:
  - The filter is heuristic — many real edges work counter-trend.
  - Reducing trade count amplifies the statistical-sample problem on a fixed
    data window.
"""
from __future__ import annotations

from datetime import time
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import detect_fvg, FVG, ema, to_higher_timeframe
from strategies._helpers import risk_based_stake


class MtfTrendFvg:
    def __init__(
        self,
        # FVG entry params (same as FvgRetest)
        min_gap_atr_mult: float = 0.5,
        max_gap_atr_mult: float = 5.0,
        max_age_bars: int = 30,
        stop_buffer_atr_mult: float = 0.2,
        atr_period: int = 14,
        r_target: float = 2.0,
        # MTF trend filter params
        htf_interval: str = "1h",
        htf_ema_period: int = 50,
        # Session
        session_open: time = time(9, 0),
        session_close: time = time(15, 0),
        flat_by: time = time(15, 30),
    ):
        self.min_gap_atr_mult = min_gap_atr_mult
        self.max_gap_atr_mult = max_gap_atr_mult
        self.max_age_bars = int(max_age_bars)
        self.stop_buffer_atr_mult = stop_buffer_atr_mult
        self.atr_period = int(atr_period)
        self.r_target = r_target
        self.htf_interval = htf_interval
        self.htf_ema_period = int(htf_ema_period)
        self.session_open = session_open
        self.session_close = session_close
        self.flat_by = flat_by

        self._open_fvgs: list[FVG] = []
        self._fvg_to_order_id: dict[int, str] = {}
        self._last_processed_index: int = -1

    # ---- HTF trend logic -----------------------------------------------
    def _htf_trend(self, history: pd.DataFrame) -> str:
        """Compute the HTF trend direction. Returns 'up', 'down', or 'neutral'."""
        try:
            htf = to_higher_timeframe(history, self.htf_interval,
                                       include_partial=False)
        except ValueError:
            return "neutral"
        if len(htf) < self.htf_ema_period + 3:
            return "neutral"
        ema_series = ema(htf["Close"], self.htf_ema_period)
        if pd.isna(ema_series.iloc[-1]) or pd.isna(ema_series.iloc[-3]):
            return "neutral"
        ema_now = float(ema_series.iloc[-1])
        ema_then = float(ema_series.iloc[-3])
        htf_close = float(htf["Close"].iloc[-1])
        ema_slope_up = ema_now > ema_then
        if htf_close > ema_now and ema_slope_up:
            return "up"
        if htf_close < ema_now and not ema_slope_up:
            return "down"
        return "neutral"

    # ---- Standalone execution (limit orders) ---------------------------
    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        i = len(history) - 1
        bar = history.iloc[i]
        bar_low, bar_high = float(bar["Low"]), float(bar["High"])
        now = history.index[i].time()

        # Sync order tracking
        live_order_ids = {o.id for o in broker.pending_orders}
        self._fvg_to_order_id = {
            k: v for k, v in self._fvg_to_order_id.items() if v in live_order_ids
        }

        # First-fill-wins: cancel other pending orders once we have a position
        if broker.position is not None and self._fvg_to_order_id:
            for order_id in list(self._fvg_to_order_id.values()):
                broker.cancel_pending_order(order_id)
            self._fvg_to_order_id.clear()

        # Update FVG list + place limit if a new FVG matches HTF trend
        if i != self._last_processed_index:
            self._last_processed_index = i

            # Drop stale / filled FVGs and their orders
            kept: list[FVG] = []
            for f in self._open_fvgs:
                age = i - f.creator_bar_index
                if age > self.max_age_bars or f.is_filled_by(bar_low, bar_high):
                    order_id = self._fvg_to_order_id.pop(f.creator_bar_index, None)
                    if order_id is not None:
                        broker.cancel_pending_order(order_id)
                else:
                    kept.append(f)
            self._open_fvgs = kept

            in_session = self.session_open <= now <= self.session_close
            is_flat = broker.position is None and not self._fvg_to_order_id
            new_fvg = detect_fvg(history, i)
            from strategies._helpers import atr_threshold
            min_gap = atr_threshold(history, self.min_gap_atr_mult, self.atr_period)
            max_gap = atr_threshold(history, self.max_gap_atr_mult, self.atr_period,
                                     fallback_pts=1e9)
            if (new_fvg is not None
                    and min_gap <= new_fvg.size_points <= max_gap
                    and in_session and is_flat):
                # The MTF FILTER: only place if HTF trend agrees with FVG direction
                htf_trend = self._htf_trend(history)
                fvg_dir = new_fvg.direction
                if (htf_trend == "up" and fvg_dir == "bullish") or \
                   (htf_trend == "down" and fvg_dir == "bearish"):
                    self._open_fvgs.append(new_fvg)
                    self._place_limit_for_fvg(new_fvg, broker, history, history.index[i])

        # Force-flat at session close
        if now >= self.flat_by:
            for order_id in list(self._fvg_to_order_id.values()):
                broker.cancel_pending_order(order_id)
            self._fvg_to_order_id.clear()
            if broker.position is not None:
                return Signal(action="close", reason="session_end")

        return Signal(action="noop")

    def _place_limit_for_fvg(self, fvg: FVG, broker: Broker,
                             history: pd.DataFrame, time) -> None:
        from strategies._helpers import atr_threshold
        stop_buf = atr_threshold(history, self.stop_buffer_atr_mult, self.atr_period)
        if fvg.direction == "bullish":
            entry = fvg.near_edge
            stop = fvg.far_edge - stop_buf
            risk_pts = entry - stop
            target = entry + self.r_target * risk_pts
            side = "long"
        else:
            entry = fvg.near_edge
            stop = fvg.far_edge + stop_buf
            risk_pts = stop - entry
            target = entry - self.r_target * risk_pts
            side = "short"

        if risk_pts <= 0:
            return
        stake = risk_based_stake(broker.balance, risk_pts, price=entry)
        if stake <= 0:
            return

        order = broker.place_pending_order(
            side=side, order_type="limit",
            trigger_price=entry, stake_per_point=stake, time=time,
            stop_loss=stop, take_profit=target,
            expires_after_bars=self.max_age_bars,
        )
        self._fvg_to_order_id[fvg.creator_bar_index] = order.id

    # ---- Ensemble polling ----------------------------------------------
    def proposed_direction(self, history: pd.DataFrame) -> str:
        """
        For ensemble polling. Returns the direction of the most recent
        FVG retest that ALSO aligns with HTF trend. Otherwise 'none'.
        """
        i = len(history) - 1
        if i < 2:
            return "none"

        htf_trend = self._htf_trend(history)
        if htf_trend == "neutral":
            return "none"

        bar = history.iloc[i]
        bar_low, bar_high = float(bar["Low"]), float(bar["High"])

        best: FVG | None = None
        for j in range(max(2, i - self.max_age_bars), i + 1):
            fvg = detect_fvg(history, j)
            if fvg is None:
                continue
            from strategies._helpers import atr_threshold
            _min_gap = atr_threshold(history, self.min_gap_atr_mult, self.atr_period)
            _max_gap = atr_threshold(history, self.max_gap_atr_mult, self.atr_period,
                                       fallback_pts=1e9)
            if not (_min_gap <= fvg.size_points <= _max_gap):
                continue
            # Filter by trend alignment
            if htf_trend == "up" and fvg.direction != "bullish":
                continue
            if htf_trend == "down" and fvg.direction != "bearish":
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
