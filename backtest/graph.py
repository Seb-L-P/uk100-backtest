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
        self.scores: list[float] = []
        self.last_supporters: list[dict] = []

    # ---- Engine interface ----
    def on_bar(self, history, broker):
        proxy = _GraphBrokerProxy(broker, self, history)
        signal = self._trigger.on_bar(history, proxy)

        if signal.action in ("open_long", "open_short"):
            side = "long" if signal.action == "open_long" else "short"
            self.attempted += 1
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
