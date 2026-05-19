"""
Search space description + random DecisionGraph sampler.

The sweep optimises along every dimension a human would have to tune by
hand: STRUCTURE (which strategies act as trigger / supporter / veto),
TIMEFRAMES (trigger TF + per-supporter TF, decoupled from the data TF),
PARAMETERS (per-strategy tunables), and optionally WEIGHTS and graph-level
KNOBS.

Data vs trigger TF
------------------
The data TF is the bar resolution we feed into the engine (1m, 5m, etc.).
The trigger TF is the cadence at which the trigger strategy makes
decisions. They can be decoupled — data at 1m + trigger at 15m means:
fills happen at 1-minute resolution (better SL/TP / pending-order
accuracy) while the trigger decides every 15 minutes (its native cadence).
The orchestrator already supports this; the sweep now exercises it by
sampling the trigger TF from `trigger_tf_options` each trial.

Search-space dimensions
-----------------------
  1. Trigger
       - strategy_key from `triggers_pool`
       - timeframe from `trigger_tf_options` (filtered to >= data_tf)
       - parameters from each ParamSpec

  2. Supporters (0..max_supporters)
       - distinct strategy keys from `supporters_pool`
       - per-supporter TF from {tf >= trigger_tf}
       - per-supporter weight from `weight_range` if `sweep_weights` is on

  3. Vetoes (0..max_vetoes)
       - distinct strategy keys from `vetoes_pool`
       - per-veto TF from {tf >= trigger_tf}
       - vetoes don't carry weights (binary block / no-block)

  4. Graph-level knobs (optional, gated on `sweep_graph_knobs`)
       - min_score from `min_score_range`
       - risk_floor / risk_ceiling from `risk_range`
       - risk_curve uniformly from {linear, sqrt, step}

No-overlap rule: trigger strategy can't also appear as supporter or veto
in the same graph. Aliased state would silently break decision logic.

Determinism: every sample uses a numpy `Generator`. Seed once at the
runner level, pass down. Identical seed + identical SearchSpace produce
identical graphs in identical order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtest.graph import (
    DecisionGraph, TriggerNode, SupporterNode, VetoNode,
    ALL_TIMEFRAMES, tf_minutes,
)
from strategies.registry import STRATEGIES, ParamSpec


# Strategy keys that don't compose well as supporters/vetoes because their
# `proposed_direction` is too noisy at HTF (they only signal on intraday
# events that don't survive resampling). We allow them as TRIGGERS but
# exclude from supporters/vetoes by default. Tune via SearchSpace if needed.
_NOISY_HTF_SUPPORTERS = {
    "orb",                 # session-bound; HTF resample garbles the open
    "overnight_range",     # same reason
    "pivot_reversal",      # daily pivots — meaningless on intra-day TF check
    "vwap_revert",         # session-VWAP doesn't survive HTF resampling
}


@dataclass
class SearchSpace:
    """Declarative description of what the sweep is allowed to try."""

    # ---- Strategy pools ------------------------------------------------
    triggers_pool: list[str] = field(default_factory=lambda: list(STRATEGIES))
    supporters_pool: list[str] = field(default_factory=list)
    vetoes_pool: list[str] = field(default_factory=list)

    # ---- Structural bounds (inclusive) ---------------------------------
    max_supporters: int = 2
    max_vetoes: int = 1

    # ---- Timeframes ----------------------------------------------------
    # The interval the engine sees raw bars at. Stops/targets/pending
    # orders fill at this granularity.
    data_tf: str = "1m"

    # Set of TFs the sweep is allowed to use as the trigger TF.
    # Each trial picks one uniformly. Must all be >= data_tf.
    trigger_tf_options: list[str] = field(
        default_factory=lambda: ["5m", "15m", "30m", "1h"]
    )

    # Cap on supporter/veto TF as a multiple of trigger TF. Default 32 means
    # a 15m trigger gets supporters up to 8h, which is sane on a 60-day
    # backtest (8h has ~150 bars over 60 days).
    max_supporter_tf_multiple: int = 32

    # ---- Weight sweep --------------------------------------------------
    # If True, sample a per-supporter weight from `weight_range` instead of
    # using the default 1.0. The original graph design said weights weren't
    # sweepable; this overrides that for users who want maximal coverage.
    sweep_weights: bool = False
    weight_range: tuple[float, float] = (0.3, 2.0)

    # ---- Graph-level knob sweep ----------------------------------------
    # When True, also sample min_score, risk_floor, risk_ceiling, risk_curve.
    # Off by default — adds a lot of dimensions and benefits diminish past
    # the structural sweep.
    sweep_graph_knobs: bool = False
    min_score_range: tuple[float, float] = (0.0, 0.7)
    risk_floor_range: tuple[float, float] = (0.4, 0.9)
    risk_ceiling_range: tuple[float, float] = (1.0, 1.5)
    risk_curve_options: list[str] = field(
        default_factory=lambda: ["linear", "sqrt", "step"]
    )

    # ---- Backward-compat alias --------------------------------------
    # Older callers used `base_tf`; we now express it as `data_tf`. Accept
    # `base_tf` via __post_init__ if it's been set externally.
    base_tf: str | None = None

    def __post_init__(self):
        # Back-compat: accept either base_tf or data_tf
        if self.base_tf is not None and self.base_tf != self.data_tf:
            self.data_tf = self.base_tf

        if self.data_tf not in ALL_TIMEFRAMES:
            raise ValueError(
                f"data_tf {self.data_tf!r} not in {ALL_TIMEFRAMES}"
            )

        # Filter trigger TF options to those >= data_tf. We DON'T raise on
        # invalid TFs — the field has a generic default and users shouldn't
        # have to re-tune it every time they change data_tf.
        d = tf_minutes(self.data_tf)
        valid = [tf for tf in self.trigger_tf_options
                 if tf in ALL_TIMEFRAMES and tf_minutes(tf) >= d]
        # Always include data_tf itself if nothing else qualifies, so the
        # sampler always has at least one option (trigger TF == data TF).
        if not valid:
            valid = [self.data_tf]
        self.trigger_tf_options = valid

        # Auto-populate supporter/veto pools from triggers if caller hasn't.
        if not self.supporters_pool:
            self.supporters_pool = [
                k for k in self.triggers_pool
                if k not in _NOISY_HTF_SUPPORTERS
            ]
        if not self.vetoes_pool:
            self.vetoes_pool = list(self.supporters_pool)

    def supporter_tf_options_for(self, trigger_tf: str) -> list[str]:
        """
        Timeframes a supporter/veto may pick from given the trigger TF.

        Rule: supporter TF must be >= trigger TF (so HTF context confirms
        a trigger-TF signal, not the other way around) and <= max multiple
        of trigger TF.
        """
        t = tf_minutes(trigger_tf)
        hi = t * max(1, self.max_supporter_tf_multiple)
        out = [tf for tf in ALL_TIMEFRAMES if t <= tf_minutes(tf) <= hi]
        if trigger_tf not in out:
            out.insert(0, trigger_tf)
        return out


# ---- Sampling ----------------------------------------------------------
def _sample_param(p: ParamSpec, rng: np.random.Generator):
    """Sample a single parameter value from its ParamSpec."""
    if p.type == "int":
        lo, hi = int(p.min), int(p.max)
        return int(rng.integers(lo, hi + 1))
    if p.type == "float":
        lo, hi = float(p.min), float(p.max)
        step = float(p.step) if p.step else 1e-6
        val = float(rng.uniform(lo, hi))
        if step > 0:
            val = round(val / step) * step
            val = max(lo, min(hi, val))
        return float(val)
    if p.type == "bool":
        return bool(rng.integers(0, 2))
    raise ValueError(f"Unknown ParamSpec type: {p.type}")


def sample_strategy_params(strategy_key: str,
                           rng: np.random.Generator) -> dict:
    """Sample a random parameter dict for the given strategy."""
    spec = STRATEGIES[strategy_key]
    return {p.name: _sample_param(p, rng) for p in (spec.params or [])}


def _sample_uniform(rng, lo: float, hi: float, ndigits: int = 2) -> float:
    """Random float in [lo, hi], rounded for UI readability."""
    return round(float(rng.uniform(lo, hi)), ndigits)


def sample_random_graph(space: SearchSpace,
                        rng: np.random.Generator) -> DecisionGraph:
    """
    Sample a single random DecisionGraph from the search space.

    Determinism: identical seeds + identical SearchSpace produce identical
    graphs in identical order.
    """
    if not space.triggers_pool:
        raise ValueError("SearchSpace.triggers_pool is empty")

    # ---- Trigger TF then strategy ----
    trigger_tf = str(rng.choice(space.trigger_tf_options))
    trig_key = str(rng.choice(space.triggers_pool))
    trig_params = sample_strategy_params(trig_key, rng)
    trigger = TriggerNode(
        strategy_key=trig_key, params=trig_params, timeframe=trigger_tf,
    )

    # ---- Supporters ----
    n_sup = int(rng.integers(0, space.max_supporters + 1))
    available_sup = [k for k in space.supporters_pool if k != trig_key]
    sup_keys: list[str] = []
    if n_sup > 0 and available_sup:
        size = min(n_sup, len(available_sup))
        sup_keys = list(rng.choice(available_sup, size=size, replace=False))
    sup_tf_options = space.supporter_tf_options_for(trigger_tf)
    supporters = []
    for k in sup_keys:
        if space.sweep_weights:
            w = _sample_uniform(rng, *space.weight_range, ndigits=2)
        else:
            w = 1.0
        supporters.append(SupporterNode(
            strategy_key=k,
            params=sample_strategy_params(k, rng),
            timeframe=str(rng.choice(sup_tf_options)),
            weight=w,
        ))

    # ---- Vetoes ----
    n_veto = int(rng.integers(0, space.max_vetoes + 1))
    used = {trig_key, *sup_keys}
    available_veto = [k for k in space.vetoes_pool if k not in used]
    veto_keys: list[str] = []
    if n_veto > 0 and available_veto:
        size = min(n_veto, len(available_veto))
        veto_keys = list(rng.choice(available_veto, size=size, replace=False))
    vetoes = [
        VetoNode(
            strategy_key=k,
            params=sample_strategy_params(k, rng),
            timeframe=str(rng.choice(sup_tf_options)),
        )
        for k in veto_keys
    ]

    # ---- Graph-level knobs ----
    graph_kwargs = {}
    if space.sweep_graph_knobs:
        graph_kwargs["min_score"] = _sample_uniform(
            rng, *space.min_score_range, ndigits=2
        )
        graph_kwargs["risk_floor"] = _sample_uniform(
            rng, *space.risk_floor_range, ndigits=2
        )
        graph_kwargs["risk_ceiling"] = max(
            graph_kwargs["risk_floor"] + 0.05,
            _sample_uniform(rng, *space.risk_ceiling_range, ndigits=2),
        )
        graph_kwargs["risk_curve"] = str(rng.choice(space.risk_curve_options))

    return DecisionGraph(
        trigger=trigger,
        supporters=supporters,
        vetoes=vetoes,
        **graph_kwargs,
    )


def describe_graph(g: DecisionGraph) -> str:
    """Compact one-liner for leaderboards. Useful for debug + UI tables."""
    sup_parts = []
    for s in g.supporters:
        tag = f"{s.strategy_key}@{s.timeframe}"
        if s.weight != 1.0:
            tag += f"×{s.weight:.2g}"
        sup_parts.append(tag)
    sup_part = ", ".join(sup_parts) or "—"
    veto_part = ", ".join(
        f"{v.strategy_key}@{v.timeframe}" for v in g.vetoes
    ) or "—"
    return (f"TRIG {g.trigger.strategy_key}@{g.trigger.timeframe} | "
            f"SUP [{sup_part}] | VETO [{veto_part}]")
