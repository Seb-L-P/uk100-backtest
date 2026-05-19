"""
Save & load sweep results.

Each saved sweep file is a JSON document containing:
  - metadata (asset, data TF, date range, search-space summary, seeds used)
  - top-N qualified OOS candidates: each one is the FULL DecisionGraph
    description (trigger + supporters + vetoes + graph knobs) plus the
    cross-split metrics (IS / Val / OOS Sharpe + trade stats)

We deliberately DON'T save the full IS trial set — that's hundreds-to-
thousands of trials each with a redundant DataFrame's worth of metadata.
The top-N is the bit you'll want to validate, refine, and paper-trade.

Filename convention:
    saved_sweeps/{timestamp}_{asset}_{tf}_{n_trials}t.json
e.g. saved_sweeps/2026-05-19_1430_UK100_15m_500t.json

The format is human-readable JSON so you can paste a single candidate's
JSON into the main backtester for a fresh manual run.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional

from backtest.graph import (
    DecisionGraph, TriggerNode, SupporterNode, VetoNode,
)
from backtest.sweep_runner import SweepResult, TrialResult
from config import SAVED_SWEEPS_DIR


# Schema version — bump when the on-disk JSON format changes incompatibly.
SCHEMA_VERSION = 1


# ---- Serialisation helpers ---------------------------------------------
def _graph_to_dict(g: DecisionGraph) -> dict:
    """Serialise a DecisionGraph to a plain-old-Python dict."""
    return {
        "trigger": {
            "strategy_key": g.trigger.strategy_key,
            "params": dict(g.trigger.params),
            "timeframe": g.trigger.timeframe,
        },
        "supporters": [{
            "strategy_key": s.strategy_key,
            "params": dict(s.params),
            "timeframe": s.timeframe,
            "weight": s.weight,
        } for s in g.supporters],
        "vetoes": [{
            "strategy_key": v.strategy_key,
            "params": dict(v.params),
            "timeframe": v.timeframe,
        } for v in g.vetoes],
        "min_score": g.min_score,
        "risk_floor": g.risk_floor,
        "risk_ceiling": g.risk_ceiling,
        "risk_curve": g.risk_curve,
        "session_open_override": g.session_open_override,
        "session_close_override": g.session_close_override,
        "flat_by_override": g.flat_by_override,
        "allow_overnight": g.allow_overnight,
    }


def _graph_from_dict(d: dict) -> DecisionGraph:
    """Recreate a DecisionGraph from a serialised dict."""
    trig = d["trigger"]
    trigger = TriggerNode(
        strategy_key=trig["strategy_key"],
        params=dict(trig["params"]),
        timeframe=trig["timeframe"],
    )
    supporters = [
        SupporterNode(
            strategy_key=s["strategy_key"],
            params=dict(s["params"]),
            timeframe=s["timeframe"],
            weight=float(s.get("weight", 1.0)),
        )
        for s in d.get("supporters", [])
    ]
    vetoes = [
        VetoNode(
            strategy_key=v["strategy_key"],
            params=dict(v["params"]),
            timeframe=v["timeframe"],
        )
        for v in d.get("vetoes", [])
    ]
    return DecisionGraph(
        trigger=trigger,
        supporters=supporters,
        vetoes=vetoes,
        min_score=float(d.get("min_score", 0.5)),
        risk_floor=float(d.get("risk_floor", 0.70)),
        risk_ceiling=float(d.get("risk_ceiling", 1.00)),
        risk_curve=d.get("risk_curve", "linear"),
        session_open_override=d.get("session_open_override"),
        session_close_override=d.get("session_close_override"),
        flat_by_override=d.get("flat_by_override"),
        allow_overnight=bool(d.get("allow_overnight", False)),
    )


def _safe_float(x) -> float | None:
    """JSON-safe float. Returns None for inf/-inf/nan (JSON-illegal)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isinf(f) or math.isnan(f):
        return None
    return f


def _trial_to_dict(t: TrialResult) -> dict:
    """Serialise a TrialResult to JSON-friendly form."""
    m = t.metrics
    return {
        "trial_index": t.trial_index,
        "graph": _graph_to_dict(t.graph),
        "metrics": {
            "sharpe": _safe_float(m.sharpe),
            "n_trades": m.n_trades,
            "final_balance": _safe_float(m.final_balance),
            "starting_balance": _safe_float(m.starting_balance),
            "return_pct": _safe_float(m.return_pct),
            "max_drawdown_pct": _safe_float(m.max_drawdown_pct),
            "hit_rate": _safe_float(m.hit_rate),
            "profit_factor": _safe_float(m.profit_factor),
            "disqualified_reason": m.disqualified_reason,
        },
    }


# ---- Public API --------------------------------------------------------
def save_sweep(
    result: SweepResult,
    *,
    asset: str,
    data_tf: str,
    date_range_label: str,
    n_bars_loaded: int,
    cost_profile: str,
    search_space_summary: dict,
    out_dir: Path | None = None,
    note: str | None = None,
) -> Path:
    """
    Save the top-N qualified candidates from a sweep to disk.

    Returns the path to the written JSON file.
    """
    out_dir = out_dir or SAVED_SWEEPS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same disqualification check as the UI uses
    def _is_dq(t):
        s = t.metrics.sharpe
        return (t.metrics.disqualified_reason is not None
                or math.isinf(s) or math.isnan(s))

    qualified = [t for t in result.oos_top if not _is_dq(t)]
    disqualified = [t for t in result.oos_top if _is_dq(t)]

    # Pair each candidate with its IS / Val metrics (the cards use these too)
    is_by_id = {id(t.graph): t for t in result.is_trials}
    val_by_id = {id(t.graph): t for t in result.val_top}

    def _enrich(t: TrialResult) -> dict:
        d = _trial_to_dict(t)
        is_t = is_by_id.get(id(t.graph))
        val_t = val_by_id.get(id(t.graph))
        d["cross_split_metrics"] = {
            "is_sharpe": _safe_float(is_t.metrics.sharpe) if is_t else None,
            "val_sharpe": _safe_float(val_t.metrics.sharpe) if val_t else None,
            "oos_sharpe": _safe_float(t.metrics.sharpe),
            "is_trades": is_t.metrics.n_trades if is_t else None,
            "val_trades": val_t.metrics.n_trades if val_t else None,
            "oos_trades": t.metrics.n_trades,
        }
        return d

    payload = {
        "schema_version": SCHEMA_VERSION,
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "asset": asset,
        "data_tf": data_tf,
        "date_range_label": date_range_label,
        "n_bars_loaded": n_bars_loaded,
        "cost_profile": cost_profile,
        "search_space": search_space_summary,
        "sweep_config": {
            "n_trials": result.n_trials,
            "top_k": result.top_k,
            "top_m": result.top_m,
            "min_trades_cap": result.min_trades,
            "is_min_trades": result.is_min_trades,
            "val_min_trades": result.val_min_trades,
            "oos_min_trades": result.oos_min_trades,
            "seed": result.seed,
        },
        "split_info": {
            "is_bars": len(result.split.is_df),
            "val_bars": len(result.split.val_df),
            "oos_bars": len(result.split.oos_df),
            "is_start": str(result.split.is_df.index[0]) if len(result.split.is_df) else None,
            "is_end": str(result.split.is_df.index[-1]) if len(result.split.is_df) else None,
            "val_start": str(result.split.val_df.index[0]) if len(result.split.val_df) else None,
            "val_end": str(result.split.val_df.index[-1]) if len(result.split.val_df) else None,
            "oos_start": str(result.split.oos_df.index[0]) if len(result.split.oos_df) else None,
            "oos_end": str(result.split.oos_df.index[-1]) if len(result.split.oos_df) else None,
        },
        "qualified_candidates": [_enrich(t) for t in qualified],
        "disqualified_candidates": [_enrich(t) for t in disqualified],
        "note": note,
    }

    # Filename: timestamp + asset + data_tf + n_trials
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    safe_asset = asset.replace("/", "_").replace(" ", "_")
    fname = f"{ts}_{safe_asset}_{data_tf}_{result.n_trials}t.json"
    path = out_dir / fname
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def list_saved_sweeps(out_dir: Path | None = None) -> list[dict]:
    """
    Return a list of metadata dicts describing each saved sweep.
    Doesn't load the full candidate lists (those can be big) — just the
    headline info for picking which sweep to open.
    """
    out_dir = out_dir or SAVED_SWEEPS_DIR
    if not out_dir.exists():
        return []
    summaries = []
    for path in sorted(out_dir.glob("*.json"), reverse=True):
        try:
            with path.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append({
            "path": str(path),
            "filename": path.name,
            "saved_at": data.get("saved_at"),
            "asset": data.get("asset"),
            "data_tf": data.get("data_tf"),
            "date_range": data.get("date_range_label"),
            "n_trials": data.get("sweep_config", {}).get("n_trials"),
            "seed": data.get("sweep_config", {}).get("seed"),
            "n_qualified": len(data.get("qualified_candidates", [])),
            "n_disqualified": len(data.get("disqualified_candidates", [])),
            "top_oos_sharpe": (
                data["qualified_candidates"][0]["metrics"]["sharpe"]
                if data.get("qualified_candidates") else None
            ),
            "note": data.get("note"),
        })
    return summaries


def load_sweep(path: str | Path) -> dict:
    """
    Load a saved sweep from disk. Returns the raw payload dict with
    DecisionGraph objects rehydrated for `qualified_candidates` and
    `disqualified_candidates` (the rest of the payload stays as dicts).

    The rehydrated graph lives at `candidate["graph_obj"]` next to the
    serialised `candidate["graph"]` dict — handy for downstream tools
    (validation, refinement) that want a DecisionGraph object.
    """
    path = Path(path)
    with path.open() as f:
        payload = json.load(f)

    if payload.get("schema_version") != SCHEMA_VERSION:
        # Soft-warn but don't refuse — future schema bumps should stay
        # backward compatible at the dict level.
        payload["_schema_mismatch"] = (
            f"file schema_version={payload.get('schema_version')!r}, "
            f"runtime SCHEMA_VERSION={SCHEMA_VERSION}"
        )

    for bucket_key in ("qualified_candidates", "disqualified_candidates"):
        for cand in payload.get(bucket_key, []):
            cand["graph_obj"] = _graph_from_dict(cand["graph"])

    return payload


def candidate_to_graph(candidate: dict) -> DecisionGraph:
    """
    Convenience: rehydrate a single candidate dict back into a DecisionGraph
    (whether or not it has `graph_obj` already).
    """
    if "graph_obj" in candidate and isinstance(candidate["graph_obj"], DecisionGraph):
        return candidate["graph_obj"]
    return _graph_from_dict(candidate["graph"])
