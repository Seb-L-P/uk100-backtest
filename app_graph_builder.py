"""
Decision-graph builder UI (Streamlit).

Renders the graph picker that replaces the old "Pick from registry" /
"Build custom ensemble" / "Multi-timeframe confluences" blocks. Returns
the fully-built DecisionGraph plus a few helper bits the rest of app.py
needs (display label, warmup, factory functions for sweeps).

Why this lives outside app.py
-----------------------------
The graph-builder block was getting too large to keep in-line with the
rest of the run-time UI. Putting it here lets app.py read top-to-bottom
as a state-machine over the run mode while the builder owns its own
locality.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from backtest.graph import (
    DecisionGraph, TriggerNode, SupporterNode, VetoNode,
    ALL_TIMEFRAMES, tf_minutes,
)
from backtest.presets import (
    list_presets, load_preset, save_preset, delete_preset,
    graph_to_dict, dict_to_graph,
)
from strategies import registry as reg


# Strategies we DON'T offer as graph nodes — the old ensembles are
# subsumed by the graph itself.
_HIDDEN_STRATEGIES = {"vote_meanrev", "vote_trend", "filter_fvg_rsi"}


def _selectable_keys() -> list[str]:
    return [k for k in reg.STRATEGIES.keys() if k not in _HIDDEN_STRATEGIES]


def _params_widget(spec, params: dict, key_prefix: str,
                   is_grid: bool = False, is_optuna: bool = False) -> dict:
    """
    Render param widgets for a single strategy spec into a dict.

    `is_grid`: render text inputs taking comma-separated values (sweep mode).
    `is_optuna`: render nothing (Optuna uses spec.params directly).

    Returns the param dict (or, in grid mode, a {param: [values]} grid).
    """
    if is_optuna:
        st.caption(
            f"Optuna will sweep {len(spec.params or [])} params from "
            f"the registry."
        )
        return {}

    out: dict = {}
    for p in (spec.params or []):
        existing = params.get(p.name, p.default)
        if is_grid:
            txt = st.text_input(
                p.label, value=str(existing),
                help=f"Type: {p.type}. Default: {p.default}. "
                     f"Comma-separated for multiple values. {p.help or ''}",
                key=f"{key_prefix}_{p.name}",
            )
            try:
                items = [x.strip() for x in txt.split(",") if x.strip()]
                if p.type == "int":
                    out[p.name] = [int(x) for x in items] or [p.default]
                elif p.type == "float":
                    out[p.name] = [float(x) for x in items] or [p.default]
                elif p.type == "bool":
                    out[p.name] = [
                        x.lower() in ("1", "true", "yes", "y") for x in items
                    ] or [p.default]
            except ValueError:
                out[p.name] = [p.default]
        else:
            if p.type == "int":
                out[p.name] = st.slider(
                    p.label, int(p.min), int(p.max), int(existing),
                    step=int(p.step) if p.step else 1,
                    help=p.help, key=f"{key_prefix}_{p.name}",
                )
            elif p.type == "float":
                out[p.name] = st.slider(
                    p.label, float(p.min), float(p.max), float(existing),
                    step=float(p.step) if p.step else 0.1,
                    help=p.help, key=f"{key_prefix}_{p.name}",
                )
            elif p.type == "bool":
                out[p.name] = st.checkbox(
                    p.label, value=bool(existing), help=p.help,
                    key=f"{key_prefix}_{p.name}",
                )
    return out


def _default_graph_spec(interval: str) -> dict:
    keys = _selectable_keys()
    return {
        "name": "",
        "trigger": {"strategy_key": keys[0] if keys else "sma",
                    "params": {}, "timeframe": interval},
        "supporters": [],
        "vetoes": [],
        "min_score": 0.5,
        "risk_floor": 0.7,
        "risk_ceiling": 1.0,
        "risk_curve": "linear",
        "tf_alpha": 0.5,
    }


def graph_builder_ui(interval: str, is_grid: bool = False,
                     is_optuna: bool = False) -> tuple[DecisionGraph, dict]:
    """
    Render the graph builder. Returns (DecisionGraph, ui_state) where
    ui_state has: display_label, display_desc, warmup_bars, param_grid
    (only in grid mode), trigger_spec.
    """
    keys = _selectable_keys()
    labels = [reg.STRATEGIES[k].label for k in keys]

    # ---- Preset row ---------------------------------------------------
    preset_names = list_presets()
    with st.expander("Presets (load / save / delete)",
                     expanded=False):
        if preset_names:
            chosen = st.selectbox("Load preset",
                                  ["(no preset selected)"] + preset_names,
                                  index=0, key="preset_load_pick")
            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("Load", disabled=(chosen.startswith("(no")),
                             key="preset_load_btn", width="stretch"):
                    g = load_preset(chosen)
                    st.session_state["graph_spec"] = graph_to_dict(g, chosen)
                    st.rerun()
            with cols[1]:
                if st.button("Delete", disabled=(chosen.startswith("(no")),
                             key="preset_del_btn", width="stretch"):
                    delete_preset(chosen)
                    st.rerun()
        else:
            st.caption("No saved presets yet. Build a graph below and save it.")

        save_name = st.text_input(
            "Save current graph as",
            value=st.session_state.get("graph_spec", {}).get("name", ""),
            placeholder="e.g. 'fvg-15m-with-1h-trend'",
            key="preset_save_name",
        )
        if st.button("Save preset", disabled=(not save_name.strip()),
                     key="preset_save_btn", width="stretch"):
            gs_to_save = st.session_state.get("graph_spec",
                                                _default_graph_spec(interval))
            gs_to_save["name"] = save_name.strip()
            try:
                graph_to_save = dict_to_graph(gs_to_save)
                save_preset(save_name.strip(), graph_to_save)
                st.success(f"Saved preset '{save_name.strip()}'")
            except Exception as e:
                st.error(f"Could not save preset: {e}")

    # ---- Graph spec lives in session_state so it survives reruns ------
    if "graph_spec" not in st.session_state:
        st.session_state["graph_spec"] = _default_graph_spec(interval)
    gs: dict[str, Any] = st.session_state["graph_spec"]

    # ---- Trigger node -------------------------------------------------
    st.markdown("**Trigger** — owns entry + full trade lifecycle.")
    trig_key_idx = (keys.index(gs["trigger"]["strategy_key"])
                    if gs["trigger"]["strategy_key"] in keys else 0)
    trig_label = st.selectbox(
        "Trigger strategy", labels, index=trig_key_idx,
        key="trigger_pick",
        help="Owns the trade lifecycle: entry order type, stop, target, "
             "partial exits, trailing. Supporters and vetoes only modulate "
             "WHETHER and HOW BIG.",
    )
    trig_key = keys[labels.index(trig_label)]
    trig_spec = reg.get(trig_key)
    gs["trigger"]["strategy_key"] = trig_key

    # Trigger TF is LOCKED to the data interval. Mixing them (e.g. 5m data
    # with a 15m trigger label) silently fed the trigger the wrong-resolution
    # bars. If you want a different trigger TF, change the data interval.
    trig_tf = interval
    gs["trigger"]["timeframe"] = trig_tf
    st.caption(
        f"Trigger runs at **{trig_tf}** — locked to the data interval above. "
        f"To trigger at a different TF, change the data interval. "
        f"Supporters/vetoes can be at this TF or HIGHER."
    )

    st.caption(trig_spec.description)
    param_grid: dict[str, list] = {}
    with st.expander("Trigger params", expanded=not is_grid):
        if is_grid:
            param_grid = _params_widget(
                trig_spec, gs["trigger"].get("params") or {},
                "trig_grid", is_grid=True,
            )
        else:
            gs["trigger"]["params"] = _params_widget(
                trig_spec, gs["trigger"].get("params") or {},
                "trig_p", is_optuna=is_optuna,
            )

    # ---- Supporters ----------------------------------------------------
    st.markdown("**Supporters** — graded opinions (excluded if 'none').")
    n_sup = st.number_input(
        "Supporter count", 0, 6, value=len(gs["supporters"]),
        key="n_supporters",
        help="Each supporter contributes a [0,1] score; their TF-weighted "
             "average becomes the confluence score.",
    )
    # Rebuild list to match count
    while len(gs["supporters"]) < n_sup:
        gs["supporters"].append({
            "strategy_key": keys[0] if keys else "sma",
            "params": {},
            "timeframe": "1h" if tf_minutes(trig_tf) < 60 else "4h",
            "weight": 1.0,
        })
    gs["supporters"] = gs["supporters"][:n_sup]

    for i, sup in enumerate(gs["supporters"]):
        with st.expander(
            f"Supporter #{i+1}: "
            f"{reg.STRATEGIES.get(sup['strategy_key'], reg.STRATEGIES[keys[0]]).label} "
            f"@ {sup['timeframe']}",
            expanded=False,
        ):
            sup_key_idx = (keys.index(sup["strategy_key"])
                           if sup["strategy_key"] in keys else 0)
            new_label = st.selectbox(
                "Strategy", labels, index=sup_key_idx, key=f"sup_{i}_pick",
            )
            new_key = keys[labels.index(new_label)]
            sup["strategy_key"] = new_key
            # TF: must be >= trigger TF
            valid_tfs = [tf for tf in ALL_TIMEFRAMES
                         if tf_minutes(tf) >= tf_minutes(trig_tf)]
            cur_tf_idx = (valid_tfs.index(sup["timeframe"])
                          if sup["timeframe"] in valid_tfs else 0)
            sup["timeframe"] = st.selectbox(
                "Timeframe", valid_tfs, index=cur_tf_idx,
                key=f"sup_{i}_tf",
                help="Must be ≥ trigger TF.",
            )
            sup["weight"] = st.number_input(
                "Weight (UI-only, never swept)", 0.0, 10.0,
                float(sup.get("weight", 1.0)), 0.5,
                key=f"sup_{i}_w",
                help="Multiplied by the TF-distance term. Tunable by you "
                     "but deliberately invisible to Optuna/grid sweeps.",
            )
            sup_spec = reg.get(new_key)
            sup["params"] = _params_widget(
                sup_spec, sup.get("params") or {},
                f"sup_{i}_p",
            )

    # ---- Vetoes --------------------------------------------------------
    st.markdown("**Vetoes** — opposite direction kills the trade.")
    n_veto = st.number_input(
        "Veto count", 0, 4, value=len(gs["vetoes"]),
        key="n_vetoes",
        help="Any one veto disagreeing with the trigger's direction blocks the trade.",
    )
    while len(gs["vetoes"]) < n_veto:
        gs["vetoes"].append({
            "strategy_key": keys[0] if keys else "sma",
            "params": {},
            "timeframe": "4h" if tf_minutes(trig_tf) < 240 else "1d",
        })
    gs["vetoes"] = gs["vetoes"][:n_veto]
    for i, vt in enumerate(gs["vetoes"]):
        with st.expander(
            f"Veto #{i+1}: "
            f"{reg.STRATEGIES.get(vt['strategy_key'], reg.STRATEGIES[keys[0]]).label} "
            f"@ {vt['timeframe']}",
            expanded=False,
        ):
            v_key_idx = (keys.index(vt["strategy_key"])
                         if vt["strategy_key"] in keys else 0)
            new_label = st.selectbox(
                "Strategy", labels, index=v_key_idx, key=f"vt_{i}_pick",
            )
            new_key = keys[labels.index(new_label)]
            vt["strategy_key"] = new_key
            valid_tfs = [tf for tf in ALL_TIMEFRAMES
                         if tf_minutes(tf) >= tf_minutes(trig_tf)]
            cur_tf_idx = (valid_tfs.index(vt["timeframe"])
                          if vt["timeframe"] in valid_tfs else 0)
            vt["timeframe"] = st.selectbox(
                "Timeframe", valid_tfs, index=cur_tf_idx,
                key=f"vt_{i}_tf",
            )
            vt_spec = reg.get(new_key)
            vt["params"] = _params_widget(
                vt_spec, vt.get("params") or {}, f"vt_{i}_p",
            )

    # ---- Score → action knobs -----------------------------------------
    st.markdown("**Confluence aggregation & risk scaling**")
    gs["min_score"] = st.slider(
        "Min score to take trade", 0.0, 1.0, float(gs["min_score"]), 0.05,
        help="Block trades whose aggregate score is below this.",
    )
    c_a, c_b = st.columns(2)
    with c_a:
        gs["risk_floor"] = st.slider(
            "Risk multiplier floor", 0.5, 1.0, float(gs["risk_floor"]), 0.05,
            help="Never risk LESS than this fraction of base. Don't go to 0 "
                 "— fixed costs eat tiny stakes.",
        )
    with c_b:
        gs["risk_ceiling"] = st.slider(
            "Risk multiplier ceiling", float(gs["risk_floor"]), 2.0,
            float(gs["risk_ceiling"]), 0.05,
            help="Max stake multiplier on max-confidence trades.",
        )
    gs["risk_curve"] = st.selectbox(
        "Risk-multiplier shape",
        ["linear", "sqrt", "step"],
        index=["linear", "sqrt", "step"].index(gs["risk_curve"]),
        help="linear: smooth. sqrt: gentler scaling. step: 0.5×/1×.",
    )
    gs["tf_alpha"] = st.slider(
        "TF-distance falloff α", 0.0, 1.5, float(gs.get("tf_alpha", 0.5)), 0.05,
        help="0 = all supporters weighted equally. 0.5 = closer TFs get √-weighted. "
             "1.0 = strict inverse-ratio. Higher = bigger penalty for distant TFs.",
    )

    # ---- Build the actual graph ---------------------------------------
    graph = dict_to_graph(gs)

    # ---- Summary label ------------------------------------------------
    parts = [trig_spec.label]
    if graph.supporters:
        parts.append(f"+{len(graph.supporters)} sup")
    if graph.vetoes:
        parts.append(f"+{len(graph.vetoes)} veto")
    display_label = " ".join(parts)

    warmup = max(
        [reg.get(graph.trigger.strategy_key).warmup_bars]
        + [reg.get(s.strategy_key).warmup_bars for s in graph.supporters]
        + [reg.get(v.strategy_key).warmup_bars for v in graph.vetoes]
        + [50]
    )

    state = {
        "display_label": display_label,
        "display_desc": trig_spec.description,
        "warmup_bars": warmup,
        "trigger_spec": trig_spec,
        "param_grid": param_grid,
    }
    return graph, state
