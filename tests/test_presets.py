"""
Tests for presets: save / load / round-trip of decision graphs.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backtest.graph import (
    DecisionGraph, TriggerNode, SupporterNode, VetoNode,
)
from backtest import presets as P


@pytest.fixture
def isolated_presets_dir(monkeypatch, tmp_path):
    """Redirect the presets directory to a temp location for each test."""
    monkeypatch.setattr(P, "PRESETS_DIR", tmp_path)
    return tmp_path


def _example_graph() -> DecisionGraph:
    return DecisionGraph(
        trigger=TriggerNode("fvg", {"min_gap_atr_mult": 0.5}, "15m"),
        supporters=[
            SupporterNode("sma", {"fast": 10, "slow": 30}, "1h", weight=1.5),
            SupporterNode("rsi_revert", {}, "4h", weight=0.5),
        ],
        vetoes=[VetoNode("donchian", {"channel_lookback": 20}, "1d")],
        min_score=0.55,
        risk_floor=0.65, risk_ceiling=1.10,
        risk_curve="sqrt",
        tf_alpha=0.4,
    )


def test_save_and_load_roundtrip(isolated_presets_dir):
    g_in = _example_graph()
    P.save_preset("my_test_preset", g_in)
    g_out = P.load_preset("my_test_preset")

    assert g_out.trigger.strategy_key == g_in.trigger.strategy_key
    assert g_out.trigger.timeframe == g_in.trigger.timeframe
    assert g_out.trigger.params == g_in.trigger.params

    assert len(g_out.supporters) == len(g_in.supporters)
    for a, b in zip(g_out.supporters, g_in.supporters):
        assert a.strategy_key == b.strategy_key
        assert a.timeframe == b.timeframe
        assert a.weight == pytest.approx(b.weight)

    assert len(g_out.vetoes) == 1
    assert g_out.vetoes[0].strategy_key == "donchian"

    assert g_out.min_score == pytest.approx(g_in.min_score)
    assert g_out.risk_floor == pytest.approx(g_in.risk_floor)
    assert g_out.risk_ceiling == pytest.approx(g_in.risk_ceiling)
    assert g_out.risk_curve == g_in.risk_curve
    assert g_out.tf_alpha == pytest.approx(g_in.tf_alpha)
    assert g_out.preset_name == "my_test_preset"


def test_list_presets_shows_saved(isolated_presets_dir):
    P.save_preset("alpha", _example_graph())
    P.save_preset("beta",  _example_graph())
    names = P.list_presets()
    assert "alpha" in names
    assert "beta" in names


def test_delete_preset(isolated_presets_dir):
    P.save_preset("to_delete", _example_graph())
    assert "to_delete" in P.list_presets()
    assert P.delete_preset("to_delete") is True
    assert "to_delete" not in P.list_presets()


def test_safe_filename_strips_bad_chars(isolated_presets_dir):
    """Slashes and other unsafe chars are stripped from the filename."""
    P.save_preset("hello / world", _example_graph())
    assert (isolated_presets_dir / "hello__world.json").exists()
