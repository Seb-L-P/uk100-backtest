"""
Decision-graph presets — save / load named graphs as JSON.

Graphs live in `presets/` at the project root, one JSON file per preset.
The schema is straightforward: keys mirror the dataclass fields.

    {
      "name": "FVG retest with HTF confluence",
      "trigger":    {"strategy_key": "fvg",  "params": {...}, "timeframe": "15m"},
      "supporters": [ {"strategy_key": "sma", "params": {...},
                        "timeframe": "1h", "weight": 1.0}, ... ],
      "vetoes":     [ ... ],
      "min_score": 0.5,
      "risk_floor": 0.7,
      "risk_ceiling": 1.0,
      "risk_curve": "linear",
      "tf_alpha": 0.5
    }

The run history database stores `preset_name` alongside each row so you
can later filter every backtest tied to a given preset.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from backtest.graph import (
    DecisionGraph, TriggerNode, SupporterNode, VetoNode,
)


PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_\- ]")


def _safe_filename(name: str) -> str:
    """Strip filesystem-unsafe characters from a preset name."""
    safe = _SAFE_NAME.sub("", name).strip()
    return safe.replace(" ", "_") or "preset"


def _path_for(name: str) -> Path:
    return PRESETS_DIR / f"{_safe_filename(name)}.json"


def graph_to_dict(graph: DecisionGraph, name: str) -> dict:
    """Serialise a DecisionGraph (plus its preset name) to a JSON-ready dict."""
    return {
        "name": name,
        "trigger": asdict(graph.trigger),
        "supporters": [asdict(s) for s in graph.supporters],
        "vetoes": [asdict(v) for v in graph.vetoes],
        "min_score": graph.min_score,
        "risk_floor": graph.risk_floor,
        "risk_ceiling": graph.risk_ceiling,
        "risk_curve": graph.risk_curve,
        "tf_alpha": graph.tf_alpha,
        "min_bars_before_flat_by": graph.min_bars_before_flat_by,
        "allow_overnight": graph.allow_overnight,
        "session_open_override": graph.session_open_override,
        "session_close_override": graph.session_close_override,
        "flat_by_override": graph.flat_by_override,
    }


def dict_to_graph(d: dict) -> DecisionGraph:
    """Inverse of graph_to_dict (the `name` is stashed on `preset_name`)."""
    trig = TriggerNode(**d["trigger"])
    sups = [SupporterNode(**s) for s in d.get("supporters", [])]
    vets = [VetoNode(**v) for v in d.get("vetoes", [])]
    g = DecisionGraph(
        trigger=trig, supporters=sups, vetoes=vets,
        min_score=d.get("min_score", 0.5),
        risk_floor=d.get("risk_floor", 0.7),
        risk_ceiling=d.get("risk_ceiling", 1.0),
        risk_curve=d.get("risk_curve", "linear"),
        tf_alpha=d.get("tf_alpha", 0.5),
        min_bars_before_flat_by=int(d.get("min_bars_before_flat_by", 0)),
        allow_overnight=bool(d.get("allow_overnight", False)),
        session_open_override=d.get("session_open_override"),
        session_close_override=d.get("session_close_override"),
        flat_by_override=d.get("flat_by_override"),
        preset_name=d.get("name"),
    )
    return g


def save_preset(name: str, graph: DecisionGraph) -> Path:
    """Write the graph to `presets/<safe_name>.json`. Returns the path."""
    path = _path_for(name)
    payload = graph_to_dict(graph, name=name)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_preset(name: str) -> DecisionGraph:
    path = _path_for(name)
    if not path.exists():
        raise FileNotFoundError(f"No preset named {name!r} at {path}")
    return dict_to_graph(json.loads(path.read_text()))


def list_presets() -> list[str]:
    """List all saved preset names (file stem)."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def delete_preset(name: str) -> bool:
    path = _path_for(name)
    if path.exists():
        path.unlink()
        return True
    return False
