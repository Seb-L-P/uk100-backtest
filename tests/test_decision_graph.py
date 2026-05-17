"""
Tests for the decision-graph framework (backtest/graph.py).

Coverage:
  - TF-distance weighting follows the documented formula.
  - Veto blocks the trade when its direction disagrees with the trigger.
  - Aggregate score excludes fence-sitters (proposed_direction == "none").
  - Risk-multiplier shape respects floor/ceiling bounds.
  - Multi-instance state isolation: same strategy class used at multiple
    nodes maintains independent state.
  - End-to-end: a graph run preserves the engine's accounting identity.
  - Look-ahead safety inside a real backtest: a 1h supporter never sees a
    bar that hasn't closed yet at the trigger's current time.
  - Trade metadata (confluence_score, risk_multiplier) ends up in
    trades_df after a real run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.graph import (
    DecisionGraph, TriggerNode, SupporterNode, VetoNode,
    GraphOrchestrator, tf_minutes,
)
from backtest.engine import Signal, run_backtest
from backtest.mtf import set_active, MTFContext


# ---- Helpers ------------------------------------------------------------
def _synth_data(n_days: int = 5, interval_min: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_price = 1000.0
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
            + pd.Timedelta(hours=h, minutes=m)
        for d in range(n_days)
        for h in range(9, 17)
        for m in range(0, 60, interval_min)
    ])
    rows = []
    for _ in idx:
        base_price += rng.normal(0, 1.0) * 0.5
        rows.append({"Open": base_price - 0.5, "High": base_price + 1.5,
                     "Low": base_price - 1.5, "Close": base_price + 0.5,
                     "Volume": 1000})
    return pd.DataFrame(rows, index=idx)


# ---- TF distance weighting ---------------------------------------------
def test_tf_weight_equal_returns_one():
    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        tf_alpha=0.5,
    )
    assert g.default_tf_weight("15m") == pytest.approx(1.0)


def test_tf_weight_halves_when_tf_quadruples():
    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        tf_alpha=0.5,
    )
    # (15/60) ** 0.5 = 0.5
    assert g.default_tf_weight("1h") == pytest.approx(0.5)


def test_tf_weight_alpha_zero_means_flat_weights():
    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        tf_alpha=0.0,
    )
    assert g.default_tf_weight("1h") == 1.0
    assert g.default_tf_weight("4h") == 1.0


# ---- Risk multiplier shape ---------------------------------------------
def test_risk_multiplier_respects_floor_and_ceiling():
    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        risk_floor=0.7, risk_ceiling=1.0, risk_curve="linear",
    )
    assert g.risk_multiplier(0.0) == pytest.approx(0.7)
    assert g.risk_multiplier(1.0) == pytest.approx(1.0)
    assert g.risk_multiplier(0.5) == pytest.approx(0.85)


def test_risk_multiplier_sqrt_is_above_linear():
    g_lin = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        risk_floor=0.0, risk_ceiling=1.0, risk_curve="linear",
    )
    g_sqrt = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        risk_floor=0.0, risk_ceiling=1.0, risk_curve="sqrt",
    )
    # At score 0.25: linear=0.25, sqrt=0.5 — sqrt is gentler (higher)
    assert g_sqrt.risk_multiplier(0.25) > g_lin.risk_multiplier(0.25)


# ---- Aggregate score: fence-sitter exclusion ---------------------------
def test_aggregate_excludes_none_supporters():
    """Supporter with 'none' opinion must not drag the aggregate down."""
    class _Yes:
        def proposed_direction(self, hist): return "long"
    class _None:
        def proposed_direction(self, hist): return "none"

    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        supporters=[
            SupporterNode("sma", {}, "15m", weight=1.0),
            SupporterNode("sma", {}, "15m", weight=1.0),
        ],
        tf_alpha=0.0,
    )
    orch = GraphOrchestrator(g)
    # Force-replace built supporters with our fakes (independent instances)
    orch._supporters = [(1.0, _Yes(), "15m"), (1.0, _None(), "15m")]
    hist = pd.DataFrame({"Close": [100.0]},
                       index=[pd.Timestamp("2024-01-01 10:00")])
    score, breakdown = orch._aggregate_score("long", hist)
    # Only the _Yes supporter counts; _None is excluded.
    # Score should be 1.0 (full agreement among opinionated supporters).
    assert score == pytest.approx(1.0)
    counted = [b for b in breakdown if b.get("counted")]
    assert len(counted) == 1


def test_aggregate_all_fence_sitters_returns_one():
    """No opinions = no signal to contradict the trigger = passes through."""
    class _None:
        def proposed_direction(self, hist): return "none"
    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        supporters=[SupporterNode("sma", {}, "15m")],
        tf_alpha=0.0,
    )
    orch = GraphOrchestrator(g)
    orch._supporters = [(1.0, _None(), "15m")]
    hist = pd.DataFrame({"Close": [100.0]},
                       index=[pd.Timestamp("2024-01-01 10:00")])
    score, _ = orch._aggregate_score("long", hist)
    assert score == 1.0


# ---- Veto behaviour ----------------------------------------------------
def test_veto_blocks_when_opposite():
    class _Disagree:
        def proposed_direction(self, hist): return "short"
    class _Trigger:
        def on_bar(self, hist, broker):
            return Signal(action="open_long", stake_per_point=1.0)

    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        vetoes=[VetoNode("sma", {}, "15m")],
    )
    orch = GraphOrchestrator(g)
    orch._trigger = _Trigger()
    orch._vetoes = [(_Disagree(), "15m")]
    hist = pd.DataFrame({"Close": [100.0]},
                       index=[pd.Timestamp("2024-01-01 10:00")])
    sig = orch.on_bar(hist, None)
    assert sig.action == "noop"
    assert "veto" in sig.reason
    assert orch.blocked_veto == 1


def test_veto_does_not_block_when_agreeing():
    class _Agree:
        def proposed_direction(self, hist): return "long"
    class _Trigger:
        def on_bar(self, hist, broker):
            return Signal(action="open_long", stake_per_point=1.0)
    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        vetoes=[VetoNode("sma", {}, "15m")],
        min_score=0.0,
    )
    orch = GraphOrchestrator(g)
    orch._trigger = _Trigger()
    orch._vetoes = [(_Agree(), "15m")]
    hist = pd.DataFrame({"Close": [100.0]},
                       index=[pd.Timestamp("2024-01-01 10:00")])
    sig = orch.on_bar(hist, None)
    assert sig.action == "open_long"


# ---- Multi-instance state isolation ------------------------------------
def test_same_strategy_class_independent_instances():
    """
    Building a graph with the same strategy at multiple nodes must give
    each node its own Python instance — state never leaks between them.
    """
    g = DecisionGraph(
        trigger=TriggerNode("fvg", {"min_gap_atr_mult": 0.3}, "15m"),
        supporters=[
            SupporterNode("fvg", {"min_gap_atr_mult": 0.5}, "1h"),
            SupporterNode("fvg", {"min_gap_atr_mult": 0.7}, "4h"),
        ],
    )
    orch = GraphOrchestrator(g)
    inst_t = orch._trigger
    inst_s1 = orch._supporters[0][1]
    inst_s2 = orch._supporters[1][1]
    # All three are distinct Python objects
    assert inst_t is not inst_s1
    assert inst_s1 is not inst_s2
    assert inst_t is not inst_s2
    # And carry independent params
    assert inst_t.min_gap_atr_mult == 0.3
    assert inst_s1.min_gap_atr_mult == 0.5
    assert inst_s2.min_gap_atr_mult == 0.7


# ---- End-to-end run: accounting identity + metadata ---------------------
def test_graph_e2e_preserves_accounting_identity_and_records_metadata():
    """
    Full backtest with a graph. Engine's internal accounting assertion
    must hold (otherwise the run crashes). Trade metadata must appear.
    """
    data = _synth_data(n_days=10, interval_min=15)
    g = DecisionGraph(
        trigger=TriggerNode("sma", {"fast": 5, "slow": 20}, "15m"),
        supporters=[],
        vetoes=[],
        min_score=0.0,
        risk_floor=0.7, risk_ceiling=1.0,
    )
    orch = GraphOrchestrator(g)
    result = run_backtest(data, orch, warmup_bars=25)
    # Accounting identity is asserted inside run_backtest. If we reached
    # here, it held. Sanity-check the result shape:
    assert result.bars_processed == len(data)
    if not result.trades_df.empty:
        # Metadata columns exist (may be None for trigger-only graph)
        assert "confluence_score" in result.trades_df.columns
        assert "risk_multiplier" in result.trades_df.columns
        assert "entry_metadata" in result.trades_df.columns


def test_supporter_drives_risk_scaling_in_real_run():
    """
    With a supporter that always agrees, risk multiplier should be at the
    ceiling. Verifies the metadata round-trip from graph → broker → trade.
    """
    data = _synth_data(n_days=10, interval_min=15)
    g = DecisionGraph(
        trigger=TriggerNode("sma", {"fast": 5, "slow": 20}, "15m"),
        supporters=[SupporterNode("sma", {"fast": 5, "slow": 20}, "1h")],
        min_score=0.0,
        risk_floor=0.7, risk_ceiling=1.0,
    )
    orch = GraphOrchestrator(g)
    result = run_backtest(data, orch, warmup_bars=50)
    # If any trades fired, their risk_multiplier must be in [0.7, 1.0]
    if not result.trades_df.empty:
        muls = result.trades_df["risk_multiplier"].dropna()
        if len(muls):
            assert (muls >= 0.7 - 1e-9).all()
            assert (muls <= 1.0 + 1e-9).all()


# ---- Look-ahead safety inside a real run ------------------------------
def test_supporter_at_higher_tf_never_sees_open_bar():
    """
    Custom supporter that records, at every call, the latest HTF index
    it was shown. Compared against the base-TF cursor: must always be
    strictly less than the bar containing the cursor.
    """
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    class _SnoopSupporter:
        def proposed_direction(self, history):
            # We're called with the HTF slice. Record its last index +
            # the current MTF cursor's now (read via the global ctx).
            from backtest.mtf import current as _c
            mtf = _c()
            cursor = mtf._now    # private but fine in tests
            if len(history):
                seen.append((cursor, history.index[-1]))
            return "long"

    class _PingTrigger:
        """Fires open_long every bar so the supporter is queried often."""
        def __init__(self): pass
        def on_bar(self, history, broker):
            return Signal(action="open_long", stake_per_point=0.01)
        def proposed_direction(self, h): return "long"

    data = _synth_data(n_days=5, interval_min=15)
    g = DecisionGraph(
        trigger=TriggerNode("sma", {}, "15m"),
        supporters=[SupporterNode("sma", {}, "1h")],
        min_score=0.0,
        risk_floor=1.0, risk_ceiling=1.0,
    )
    orch = GraphOrchestrator(g)
    orch._trigger = _PingTrigger()
    orch._supporters = [(1.0, _SnoopSupporter(), "1h")]

    run_backtest(data, orch, warmup_bars=10)
    # For every (cursor, latest_htf_index) seen, the HTF bar must have
    # closed strictly before the cursor. With 1h HTF, that means
    # latest_htf_index + 1h <= cursor.
    period = pd.Timedelta("1h")
    for cursor, latest_htf in seen:
        assert latest_htf + period <= cursor, (
            f"Look-ahead: at cursor {cursor} the HTF supporter saw "
            f"bar {latest_htf} (close at {latest_htf + period})"
        )
    # Sanity: we did observe at least a few calls
    assert len(seen) > 5
