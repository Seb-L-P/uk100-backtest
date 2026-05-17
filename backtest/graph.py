"""
Decision-graph synthesis framework.

A `DecisionGraph` is the unified replacement for "single strategy",
"custom ensemble", and "HTF filter". It declares:

    Trigger    — exactly one node, at the base timeframe. Owns the full
                 trade lifecycle (entry order type, stop, target, partial
                 exits, trailing). Behaviour is whatever the chosen
                 strategy does today on its own.

    Supporters — any number, any timeframe ≥ trigger TF. Each one outputs
                 a direction opinion and contributes a confidence score in
                 [0, 1] toward an aggregate. Higher aggregate score → take
                 trade with slightly larger position size.

    Vetoes     — any number, any timeframe ≥ trigger TF. Hard-block the
                 trigger's direction if they disagree. No grading: one
                 firing veto kills the trade.

Same strategy class can appear at multiple TFs simultaneously with
independent state because each node builds a fresh instance.

How scores combine
------------------
For each entry attempt the orchestrator:

  1. Asks every veto for `proposed_direction(node_history)`. If ANY
     veto returns the OPPOSITE direction of the trigger's intent, the
     trade is blocked. Equal direction or "none" is fine.

  2. Asks every supporter for `proposed_direction(node_history)`.
       - matches trigger's side    → score 1.0
       - opposite                   → score 0.0
       - "none" (no opinion)        → EXCLUDED from the aggregate
                                      (a fence-sitter doesn't drag the
                                      score down — see graph.md design notes)

  3. Aggregate = Σ(score_i · weight_i) / Σ(weight_i) over the supporters
     that had an opinion. With zero opinions, aggregate = 1.0 (no signal
     to veto, no signal to confirm — passes through unchanged).

  4. Default supporter weight follows the TF-distance heuristic:
        weight ∝ (trigger_minutes / supporter_minutes) ** α   (α = 0.5)
     So an HTF 4× the trigger's gets √0.25 = 0.5 weight; a 64× HTF gets
     1/8. The user can override each weight in the UI — those overrides
     are stored on `SupporterNode` instances. They are deliberately NOT
     surfaced as `ParamSpec`s, so they're INVISIBLE to Optuna / grid
     sweep: the optimiser tunes strategy params only, never the weights
     that grade their voices.

  5. If aggregate < `min_score`, block. Else scale risk:
        multiplier = floor + (ceiling - floor) · shape(score)
     with `floor=0.7, ceiling=1.0` by default. The 0.7 floor exists
     because fixed per-trade costs (spread + slippage) eat tiny stakes —
     0–100% scaling kills any real edge by sending borderline trades in
     at unprofitable size.

Look-ahead safety
-----------------
Centralised in `MTFContext`: HTF data returned for time `t` includes
only bars that have FULLY CLOSED at or before `t`. The orchestrator
asks the MTF context for each supporter/veto's view; it never reaches
into raw data itself. Tests in `test_decision_graph.py` prove this.

Recording on trades
-------------------
At every entry attempt that passes, the orchestrator attaches to the
Signal's `metadata`:

    {
        "confluence_score": <float>,
        "risk_multiplier": <float>,
        "supporters": [{strategy, tf, direction, score, weight}, ...]
    }

This propagates through Signal → PendingOrder → OpenPosition → Trade,
ending up in `trades_df["entry_metadata"]` plus first-class columns
`confluence_score` and `risk_multiplier`. That's how you check after
the run whether realised P&L actually rises with confidence — if it
doesn't, the score is fake.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from backtest.mtf import current as _mtf_current


# ---- Timeframe helpers --------------------------------------------------
_TF_MINUTES: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240,
    "1d": 24 * 60, "1wk": 7 * 24 * 60,
}

# Public list of TFs the UI can offer. Ordered by length, ascending.
ALL_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1wk"]


def tf_minutes(tf: str) -> int:
    if tf not in _TF_MINUTES:
        raise ValueError(f"Unknown timeframe {tf!r}; known: {list(_TF_MINUTES)}")
    return _TF_MINUTES[tf]


def _parse_hhmm(s: str | None):
    """Parse 'HH:MM' to datetime.time, or None for None/empty input."""
    if not s:
        return None
    from datetime import time as _time
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
        return _time(h, m)
    except ValueError:
        return None


def is_higher_or_equal_tf(supporter_tf: str, trigger_tf: str) -> bool:
    return tf_minutes(supporter_tf) >= tf_minutes(trigger_tf)


# ---- Nodes --------------------------------------------------------------
@dataclass(frozen=True)
class TriggerNode:
    """The base-TF strategy that owns trade lifecycle decisions."""
    strategy_key: str
    params: dict
    timeframe: str

    def build(self):
        from strategies.registry import get
        return get(self.strategy_key).build(**self.params)


@dataclass(frozen=True)
class SupporterNode:
    """A supporting opinion. Weight is user-tunable but NOT sweep-exposed."""
    strategy_key: str
    params: dict
    timeframe: str
    weight: float = 1.0  # user override; final weight = this * TF-distance term

    def build(self):
        from strategies.registry import get
        return get(self.strategy_key).build(**self.params)


@dataclass(frozen=True)
class VetoNode:
    """Disagreement here kills the trade. No grading, no scaling."""
    strategy_key: str
    params: dict
    timeframe: str

    def build(self):
        from strategies.registry import get
        return get(self.strategy_key).build(**self.params)


# ---- Graph + risk scaling ----------------------------------------------
RiskCurve = Literal["linear", "sqrt", "step"]


@dataclass
class DecisionGraph:
    """The full decision graph for one strategy 'design'."""
    trigger: TriggerNode
    supporters: list[SupporterNode] = field(default_factory=list)
    vetoes: list[VetoNode] = field(default_factory=list)

    # Score → action knobs (user-tunable, NOT sweep-exposed):
    min_score: float = 0.5
    risk_floor: float = 0.70    # min stake multiplier (never below)
    risk_ceiling: float = 1.00  # max stake multiplier (default = base risk)
    risk_curve: RiskCurve = "linear"

    # TF-distance falloff for the DEFAULT supporter weight. Each supporter's
    # `weight` field is multiplied by `(t_trigger/t_sup) ** alpha`. α=0.5 by
    # default → halving every doubling of the TF.
    tf_alpha: float = 0.5

    # ---- Session timing knobs ----------------------------------------
    # Late-entry filter: refuse new entries when fewer than this many bars
    # remain until the trigger's `flat_by`. Stops the engine from opening
    # trades that get force-closed before they can develop. Default 0 = off.
    min_bars_before_flat_by: int = 0

    # Disable the trigger's session-end force-close entirely. Positions
    # carry overnight and accrue real IG financing (already modelled in
    # the cost profile). Use this for swing/positional setups where holding
    # past 15:30 is part of the design — not for true day-trade strategies.
    allow_overnight: bool = False

    # ---- Per-graph session-time overrides ----
    # When set (HH:MM strings), these override the trigger strategy's own
    # defaults. Useful for trading a US asset in UK time, where the FVG
    # strategy's UK-100-default `flat_by=15:30` would force-close at AAPL's
    # 30-minutes-after-open. Leave None to use the strategy's hardcoded
    # defaults. Stored as strings so they JSON-serialise cleanly into presets.
    session_open_override: str | None = None
    session_close_override: str | None = None
    flat_by_override: str | None = None

    # Optional preset name (for run-history linkage). Set by load/save.
    preset_name: str | None = None

    def default_tf_weight(self, supporter_tf: str) -> float:
        t = tf_minutes(self.trigger.timeframe)
        s = tf_minutes(supporter_tf)
        if s <= 0:
            return 0.0
        return (t / s) ** self.tf_alpha

    def risk_multiplier(self, score: float) -> float:
        if self.risk_curve == "linear":
            shape = score
        elif self.risk_curve == "sqrt":
            shape = score ** 0.5
        elif self.risk_curve == "step":
            shape = 1.0 if score >= 0.75 else 0.5
        else:
            shape = score
        return self.risk_floor + (self.risk_ceiling - self.risk_floor) * shape


# ---- Orchestrator ------------------------------------------------------
class GraphOrchestrator:
    """
    Runs a `DecisionGraph` over base-TF data. Acts as the engine-facing
    strategy: implements `on_bar(history, broker)` and
    `proposed_direction(history)` so it slots into `run_backtest` exactly
    like any single strategy.

    Wraps the broker with a proxy so trigger strategies that use
    `broker.place_pending_order` ALSO get filtered by supporters/vetoes
    (vital for FVG retest, BPR, inside-bar, etc.).
    """

    def __init__(self, graph: DecisionGraph):
        self.graph = graph
        self._trigger = graph.trigger.build()

        # ---- Session-time ownership ----
        # Strategies have internal `flat_by` / `session_open` / `session_close`
        # checks that compare `history.index[-1].time()` to the configured
        # times. When trigger TF > base TF, `history.index[-1]` is the
        # RESAMPLED bar's start time (label="left"), NOT the real wall-clock
        # time — so those checks fire one trigger period LATE.
        #
        # Fix: capture the trigger's session times into orchestrator state,
        # then NEUTRALISE the checks inside the trigger by pushing the
        # attributes to "always-in-session" sentinels. The orchestrator runs
        # on every BASE bar and enforces real-time session gating itself.
        from datetime import time as _time

        # Resolve from graph override → trigger attribute → None
        def _resolve_time(graph_override, trigger_attr):
            if graph_override is not None:
                return _parse_hhmm(graph_override)
            return getattr(self._trigger, trigger_attr, None)

        self._session_open = _resolve_time(
            graph.session_open_override, "session_open"
        )
        self._session_close = _resolve_time(
            graph.session_close_override, "session_close"
        )
        self._session_flat_by = _resolve_time(
            graph.flat_by_override, "flat_by"
        )
        if graph.allow_overnight:
            self._session_flat_by = None  # never force-close

        # Neutralise the trigger's internal session-time checks so they
        # don't fire at the (lagged) trigger-TF cadence. The orchestrator
        # owns timing now.
        if hasattr(self._trigger, "session_open"):
            self._trigger.session_open = _time(0, 0, 0)
        if hasattr(self._trigger, "session_close"):
            self._trigger.session_close = _time(23, 59, 59)
        if hasattr(self._trigger, "flat_by"):
            self._trigger.flat_by = _time(23, 59, 59)

        # Build supporter/veto instances. Independent state per node, so the
        # same strategy class can appear multiple times with no aliasing.
        self._supporters: list[tuple[float, object, str]] = []
        for s in graph.supporters:
            inst = s.build()
            effective_weight = s.weight * graph.default_tf_weight(s.timeframe)
            self._supporters.append((effective_weight, inst, s.timeframe))
        self._vetoes: list[tuple[object, str]] = [
            (v.build(), v.timeframe) for v in graph.vetoes
        ]

        # Stats surfaced to the UI / saved with run history
        self.attempted = 0
        self.blocked_veto = 0
        self.blocked_score = 0
        self.blocked_late_entry = 0
        self.scores: list[float] = []
        self.last_supporters: list[dict] = []

        # Base TF (in minutes) detected from the FIRST history we see. The
        # engine feeds us base-TF bars even if the trigger runs on a higher
        # TF — decoupling lets stops/targets/pending-orders fill at base
        # resolution while the trigger decides at its own cadence.
        self._base_tf_min: int | None = None

    # ---- Engine interface ----
    def on_bar(self, history, broker):
        # Detect the base TF on the first bar. We trust the cadence of the
        # data we're being fed — that's what the engine iterates at.
        if self._base_tf_min is None and len(history) >= 2:
            d = history.index[-1] - history.index[-2]
            self._base_tf_min = max(1, int(round(d.total_seconds() / 60)))

        trigger_tf_min = tf_minutes(self.graph.trigger.timeframe)
        base_tf_min = self._base_tf_min or trigger_tf_min

        # If trigger TF < base TF, that's a config error — supporters can
        # be at base TF or higher, the trigger MUST run at base TF or higher.
        if trigger_tf_min < base_tf_min:
            return self._noop_signal(
                f"trigger TF ({self.graph.trigger.timeframe}) < base TF "
                f"({base_tf_min}m). Pick trigger TF >= data interval."
            )

        # ---- Session-end enforcement (BASE-TF precision) ----
        # Runs every base bar. Catches flat_by at 1-minute precision even
        # when the trigger is decoupled to 15m, fixing the "trade closes
        # 15-30 min late" bug.
        real_time = history.index[-1].time()
        if (self._session_flat_by is not None
                and real_time >= self._session_flat_by):
            # Cancel any pending orders (so they can't fill after hours)
            if broker.pending_orders:
                for o in list(broker.pending_orders):
                    broker.cancel_pending_order(o.id)
            # Close any open position
            if broker.positions:
                return self._close_all_signal("session_end")
            return self._noop_signal("past session_end")

        # ---- Out-of-session: no new entries ----
        # Before session_open or after session_close (but before flat_by):
        # let existing positions run, but don't even consult the trigger
        # for new entries.
        if (self._session_open is not None
                and self._session_close is not None
                and not (self._session_open <= real_time <= self._session_close)):
            return self._noop_signal("outside entry window")

        # Decide whether THIS base bar coincides with a trigger-TF close.
        # On a 1m base / 15m trigger: only base bars at HH:00, HH:15, HH:30,
        # HH:45 will trigger. On equal TFs, every bar triggers (current
        # behaviour).
        if not self._is_trigger_close(history.index[-1], base_tf_min, trigger_tf_min):
            return self._noop_signal("not a trigger-TF close")

        # Build the trigger's history at ITS own TF — resampled from the
        # base bars we've seen so far. include_partial=False guarantees we
        # never feed it a still-forming bar (look-ahead safety).
        if trigger_tf_min == base_tf_min:
            trigger_history = history
        else:
            from backtest.indicators import to_higher_timeframe
            trigger_history = to_higher_timeframe(
                history, self.graph.trigger.timeframe, include_partial=False
            )
            if len(trigger_history) < 2:
                return self._noop_signal("not enough trigger-TF history yet")

        # IMPORTANT: hand the trigger a proxy whose `history` reference is
        # ALSO at the trigger's TF, so supporter/veto checks done inside
        # pending-order placements see consistent data.
        proxy = _GraphBrokerProxy(broker, self, trigger_history)
        signal = self._trigger.on_bar(trigger_history, proxy)

        if signal.action in ("open_long", "open_short"):
            side = "long" if signal.action == "open_long" else "short"
            self.attempted += 1

            # ---- Late-entry filter ----
            # If the trigger has a flat_by AND the graph wants a minimum
            # number of bars before it, refuse to open a trade that won't
            # have room to develop. Prevents the "entered at 14:45, forced
            # out at 15:30" pattern that destroys 2R-target strategies.
            if self.graph.min_bars_before_flat_by > 0:
                remaining = self._bars_until_flat_by(history.index[-1])
                if remaining is not None and remaining < self.graph.min_bars_before_flat_by:
                    self.blocked_late_entry += 1
                    signal.action = "noop"
                    signal.reason = (f"late_entry: only {remaining} bars "
                                     f"to flat_by, need "
                                     f"{self.graph.min_bars_before_flat_by}")
                    return signal

            veto = self._find_firing_veto(side, history)
            if veto is not None:
                self.blocked_veto += 1
                signal.action = "noop"
                signal.reason = f"veto by {veto}"
                return signal

            score, sup_breakdown = self._aggregate_score(side, history)
            self.scores.append(score)
            self.last_supporters = sup_breakdown

            if score < self.graph.min_score:
                self.blocked_score += 1
                signal.action = "noop"
                signal.reason = (f"confluence={score:.2f} < min="
                                 f"{self.graph.min_score:.2f}")
                return signal

            mult = self.graph.risk_multiplier(score)
            if signal.stake_per_point is not None:
                signal.stake_per_point = signal.stake_per_point * mult
            signal.metadata = {
                "confluence_score": score,
                "risk_multiplier": mult,
                "supporters": sup_breakdown,
            }
            base_reason = signal.reason or ""
            signal.reason = (f"{base_reason} [conf={score:.2f} ×{mult:.2f}]"
                             ).strip()
        return signal

    def proposed_direction(self, history) -> str:
        if not hasattr(self._trigger, "proposed_direction"):
            return "none"
        d = self._trigger.proposed_direction(history)
        if d == "none":
            return "none"
        if self._find_firing_veto(d, history) is not None:
            return "none"
        score, _ = self._aggregate_score(d, history)
        return d if score >= self.graph.min_score else "none"

    # Forward other attributes (post_open, custom state, etc.)
    def __getattr__(self, name):
        return getattr(self._trigger, name)

    # ---- Internals ----
    @staticmethod
    def _noop_signal(reason: str):
        """Build a noop Signal — broken out for use in early-returns of on_bar."""
        from backtest.engine import Signal
        return Signal(action="noop", reason=reason)

    @staticmethod
    def _close_all_signal(reason: str):
        from backtest.engine import Signal
        return Signal(action="close_all", reason=reason)

    @staticmethod
    def _is_trigger_close(now: pd.Timestamp,
                          base_tf_min: int, trigger_tf_min: int) -> bool:
        """
        True iff this base-TF bar coincides with the close of a trigger-TF
        bar — i.e. the NEXT base bar would start a new trigger-TF window.

        Conventions:
          - bar timestamps in our data are bar START times (label="left").
          - a 15m trigger window at 09:00 covers [09:00, 09:15).
          - it closes at 09:15. The last 1m base bar inside the window is
            the one starting at 09:14 (covering 09:14-09:15). After mark
            and stop checks on that bar, we ARE at the 15m close moment.

        So the test is: (now + base_tf_min) is a multiple of trigger_tf_min
        from session start (using minute-of-day so it works across DST).
        """
        if base_tf_min >= trigger_tf_min:
            return True  # equal TFs always trigger; sanity-checked by caller
        end_minute = (now.hour * 60 + now.minute + base_tf_min)
        # For daily / weekly triggers, the boundary alignment is per-day, so
        # match if end_minute is divisible by trigger_tf_min OR end_minute is
        # at the configured day boundary (00:00). The simple modulo handles
        # all the intra-day cases we care about (15m/30m/1h/4h triggers).
        if trigger_tf_min < 24 * 60:
            return end_minute % trigger_tf_min == 0
        # Daily+ trigger on intraday base: fire only at end-of-day.
        return end_minute % (24 * 60) == 0

    def _bars_until_flat_by(self, now: pd.Timestamp) -> int | None:
        """
        Number of BASE-TF bars between `now` and the resolved flat_by today.
        Uses the orchestrator's captured `_session_flat_by` (NOT the trigger's
        own attribute, which we've neutralised). Returns None if no flat_by
        is configured (allow_overnight, or strategy without session timing).
        """
        flat_by = self._session_flat_by
        if flat_by is None:
            return None
        if now.time() >= flat_by:
            return 0
        flat_dt = pd.Timestamp.combine(now.date(), flat_by)
        if hasattr(now, "tz") and now.tz is not None:
            flat_dt = flat_dt.tz_localize(now.tz)
        delta_min = (flat_dt - now).total_seconds() / 60.0
        bar_min = self._base_tf_min or tf_minutes(self.graph.trigger.timeframe)
        return int(delta_min // bar_min)

    def _history_at_tf(self, base_history: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        Return look-ahead-safe history at `tf`.

        - tf == trigger TF: same as base_history
        - tf > trigger TF: MTFContext.htf(tf) — only fully-closed HTF bars
        """
        if tf == self.graph.trigger.timeframe:
            return base_history
        return _mtf_current().htf(tf)

    def _find_firing_veto(self, side: str, base_history: pd.DataFrame):
        """Return the name of the first veto disagreeing, or None."""
        opposite = "short" if side == "long" else "long"
        for strat, tf in self._vetoes:
            if not hasattr(strat, "proposed_direction"):
                continue
            hist = self._history_at_tf(base_history, tf)
            if hist.empty:
                continue
            try:
                d = strat.proposed_direction(hist)
            except Exception:
                d = "none"
            if d == opposite:
                return f"{type(strat).__name__}@{tf}"
        return None

    def _aggregate_score(self, side: str,
                         base_history: pd.DataFrame) -> tuple[float, list[dict]]:
        """
        Average of (score · weight) over OPINIONATED supporters.
        Fence-sitters ("none") are excluded from numerator AND denominator.

        Returns (aggregate, breakdown) where breakdown is a list of dicts
        suitable for storing on the trade.
        """
        breakdown: list[dict] = []
        total = 0.0
        total_w = 0.0
        for effective_weight, strat, tf in self._supporters:
            entry = {
                "strategy": type(strat).__name__,
                "tf": tf,
                "weight": effective_weight,
                "direction": "none",
                "score": None,
                "counted": False,
            }
            if not hasattr(strat, "proposed_direction"):
                breakdown.append(entry)
                continue
            hist = self._history_at_tf(base_history, tf)
            if hist.empty:
                breakdown.append(entry)
                continue
            try:
                d = strat.proposed_direction(hist)
            except Exception:
                d = "none"
            entry["direction"] = d
            if d == "none":
                # Fence-sitter excluded by design.
                breakdown.append(entry)
                continue
            score = 1.0 if d == side else 0.0
            entry["score"] = score
            entry["counted"] = True
            breakdown.append(entry)
            total += score * effective_weight
            total_w += effective_weight
        if total_w <= 0:
            # No supporters had an opinion. Don't suppress the trade —
            # treat as full pass-through (the trigger alone is enough).
            return 1.0, breakdown
        return total / total_w, breakdown


class _GraphBrokerProxy:
    """
    Wraps a broker so the orchestrator can intercept pending-order
    placements. Limit/stop strategies (FVG, BPR, inside-bar) go through
    this transparently.
    """

    def __init__(self, real_broker, orchestrator, history):
        self.__dict__["_real"] = real_broker
        self.__dict__["_orch"] = orchestrator
        self.__dict__["_history"] = history

    def place_pending_order(self, side, order_type, trigger_price,
                            stake_per_point, time, **kwargs):
        self._orch.attempted += 1

        # ---- Late-entry filter (also applies to pending orders) ----
        # The trigger may have placed a limit / stop well in advance, but if
        # it would only have a handful of bars to develop before flat_by we
        # cancel it just like a fresh market entry. Returns a ghost order
        # that's never registered with the broker, so it can't fill.
        if self._orch.graph.min_bars_before_flat_by > 0:
            remaining = self._orch._bars_until_flat_by(self._history.index[-1])
            if (remaining is not None
                    and remaining < self._orch.graph.min_bars_before_flat_by):
                self._orch.blocked_late_entry += 1
                return _ghost_order(side, order_type, trigger_price, time, kwargs)

        veto = self._orch._find_firing_veto(side, self._history)
        if veto is not None:
            self._orch.blocked_veto += 1
            return _ghost_order(side, order_type, trigger_price, time, kwargs)
        score, sup = self._orch._aggregate_score(side, self._history)
        self._orch.scores.append(score)
        self._orch.last_supporters = sup
        if score < self._orch.graph.min_score:
            self._orch.blocked_score += 1
            return _ghost_order(side, order_type, trigger_price, time, kwargs)
        mult = self._orch.graph.risk_multiplier(score)
        return self._real.place_pending_order(
            side=side, order_type=order_type, trigger_price=trigger_price,
            stake_per_point=stake_per_point * mult,
            time=time,
            entry_metadata={
                "confluence_score": score,
                "risk_multiplier": mult,
                "supporters": sup,
            },
            **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def _ghost_order(side, order_type, trigger_price, time, kwargs):
    """A PendingOrder that's never added to the broker — never fills."""
    from backtest.broker import PendingOrder
    safe_kwargs = {k: v for k, v in kwargs.items()
                   if k in ("stop_loss", "take_profit",
                            "trailing_stop_fn", "expires_after_bars")}
    return PendingOrder(
        id=f"_ghost_graph_{id(time)}_{trigger_price:.4f}",
        side=side, order_type=order_type,
        trigger_price=float(trigger_price),
        stake_per_point=0.0,
        placed_time=time,
        **safe_kwargs,
    )
