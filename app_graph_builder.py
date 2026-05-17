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
    """
    Fresh graph spec. Session-time overrides default to whatever the
    ACTIVE cost profile's RTH window is — that's the most likely correct
    setting for the asset, so users don't fall through to the strategy's
    UK-100-default 9:00-15:30 when trading AAPL.
    """
    keys = _selectable_keys()
    import config as _cfg
    profile = getattr(_cfg, "COSTS", None)
    profile_name = profile.instrument if profile is not None else "DEFAULT"
    sess = _cfg.session_defaults_for(profile_name, mode="rth")
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
        "min_bars_before_flat_by": 0,
        "allow_overnight": sess is None,  # crypto/FX → carry overnight
        "session_open_override": sess[0] if sess else None,
        "session_close_override": sess[1] if sess else None,
        "flat_by_override": sess[2] if sess else None,
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

    # Trigger TF can now be DECOUPLED from the data interval. The orchestrator
    # only fires the trigger on bars that close a window of the chosen TF, so:
    #   - data 1m + trigger 15m → trigger decides every 15m, fills happen at
    #     1m granularity (better intrabar SL/TP resolution, more accurate gaps)
    #   - data == trigger → trigger fires every bar (no decoupling overhead)
    # Trigger TF must be ≥ data interval (we can't downsample to a finer TF).
    valid_trig_tfs = [tf for tf in ALL_TIMEFRAMES
                      if tf_minutes(tf) >= tf_minutes(interval)]
    default_trig_tf = (interval if interval in valid_trig_tfs
                       else valid_trig_tfs[0])
    cur_trig_tf = gs["trigger"].get("timeframe", default_trig_tf)
    if cur_trig_tf not in valid_trig_tfs:
        cur_trig_tf = default_trig_tf
    trig_tf = st.selectbox(
        "Trigger timeframe", valid_trig_tfs,
        index=valid_trig_tfs.index(cur_trig_tf),
        key="trigger_tf_pick",
        help=(
            "TF at which the trigger STRATEGY makes decisions. Defaults to "
            "the data interval. Pick higher to decouple decisions from fills: "
            "data 1m + trigger 15m means the strategy decides every 15m, "
            "but stops/targets and pending limit orders fill at 1m "
            "granularity — better intrabar accuracy on gappy bars."
        ),
    )
    gs["trigger"]["timeframe"] = trig_tf
    if trig_tf != interval:
        st.caption(
            f"⚙️ Decoupled: data at **{interval}**, trigger decides on "
            f"**{trig_tf}** closes. Fills (SL/TP/pending) happen at {interval} "
            f"resolution. Slower to backtest but more accurate."
        )
    else:
        st.caption(
            f"Trigger runs at **{trig_tf}** — same as data interval. "
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

    # ---- Session timing -----------------------------------------------
    st.markdown("**Session timing**")
    gs["allow_overnight"] = st.checkbox(
        "Carry positions overnight (disable session force-flat)",
        value=bool(gs.get("allow_overnight", False)),
        help=("ON: positions stay open past session-end; real IG financing "
              "is charged daily (~7.75% annual on long notional). Use for "
              "swing/positional setups. "
              "OFF (default): true day-trade — force-flat at the trigger "
              "strategy's flat_by time (typically 15:30 UK)."),
    )
    gs["min_bars_before_flat_by"] = st.slider(
        "Min bars before flat_by to allow new entry", 0, 32,
        int(gs.get("min_bars_before_flat_by", 0)), 1,
        help=("Blocks new entries that wouldn't have room to develop "
              "before the session close. 0 = off. "
              "Applies to both market signals AND pending limit/stop orders "
              "placed by the trigger (limit orders that would fill too late "
              "get cancelled). Ignored when 'Carry overnight' is on."),
        disabled=bool(gs.get("allow_overnight", False)),
    )

    # ---- Session-time overrides --------------------------------------
    # The trigger strategy has hardcoded defaults for session_open /
    # session_close / flat_by — usually for the UK 100. Trading another
    # asset (AAPL in UK time, BTC, etc.) means those defaults are wrong.
    # Overrides here propagate to the orchestrator which enforces them at
    # base-TF precision. Format: HH:MM in the trading timezone you picked.
    with st.expander("Session times (overrides — leave blank to use strategy defaults)",
                     expanded=False):
        st.caption(
            "If blank, the trigger's own session_open/close/flat_by are used "
            "(UK 100 defaults: 8:30 / 15:00 / 15:30). For other assets, set "
            "these so the strategy doesn't trade outside the asset's real "
            "market hours."
        )
        # Quick preset buttons for common cases.
        # NOTE: when a preset is clicked, we MUST update both the graph_spec
        # dict AND the text_input widget keys in st.session_state — Streamlit
        # gives widget-state priority over the `value=` argument on rerun,
        # so updating only the dict was a silent no-op. Confirmed bug from
        # AAPL pre-market trades in May 2026.
        def _apply_preset(open_v, close_v, flat_v, overnight=False):
            gs["session_open_override"] = open_v
            gs["session_close_override"] = close_v
            gs["flat_by_override"] = flat_v
            if overnight:
                gs["allow_overnight"] = True
            # CRITICAL: also update the widget-state keys, otherwise the
            # text_inputs keep their stale (empty) values on rerun.
            st.session_state["sess_open_o"] = open_v or ""
            st.session_state["sess_close_o"] = close_v or ""
            st.session_state["sess_flat_o"] = flat_v or ""

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("UK 100\n08:30 / 15:00 / 15:30", key="sess_uk100"):
                _apply_preset("08:30", "15:00", "15:30")
                st.rerun()
        with b2:
            if st.button("US stocks (UK time)\n14:30 / 20:00 / 20:30",
                         key="sess_us"):
                _apply_preset("14:30", "20:00", "20:30")
                st.rerun()
        with b3:
            if st.button("Crypto 24/7\n00:00 / 23:59 / 23:59", key="sess_btc"):
                _apply_preset("00:00", "23:59", "23:59", overnight=True)
                st.rerun()
        with b4:
            if st.button("Clear overrides\n(use strategy defaults)",
                         key="sess_clear"):
                _apply_preset("", "", "")
                # Also clear the underlying values so they really go to None
                gs["session_open_override"] = None
                gs["session_close_override"] = None
                gs["flat_by_override"] = None
                st.rerun()
        # ---- Text inputs ----
        # Streamlit rule: a widget can use EITHER `value=` OR `key=` with
        # session_state, never both. The preset buttons write to
        # st.session_state[<key>], so we must NOT also pass `value=`. We
        # seed st.session_state from gs once (if the key isn't already set),
        # then let the widget read/write its own session state directly.
        for _k, _gk in (("sess_open_o", "session_open_override"),
                        ("sess_close_o", "session_close_override"),
                        ("sess_flat_o", "flat_by_override")):
            if _k not in st.session_state:
                st.session_state[_k] = gs.get(_gk) or ""

        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input("session_open", placeholder="e.g. 14:30",
                          key="sess_open_o")
        with c2:
            st.text_input("session_close", placeholder="e.g. 20:00",
                          key="sess_close_o")
        with c3:
            st.text_input("flat_by", placeholder="e.g. 20:30",
                          key="sess_flat_o")

        # Read back into graph spec — "" becomes None so the orchestrator
        # falls through to strategy defaults.
        gs["session_open_override"] = st.session_state["sess_open_o"] or None
        gs["session_close_override"] = st.session_state["sess_close_o"] or None
        gs["flat_by_override"] = st.session_state["sess_flat_o"] or None
        # Inline warning when STOCK profile selected but no override set —
        # the most common footgun (default UK times applied to US stock).
        try:
            from config import COSTS as _cur_cost
            if (_cur_cost.instrument in ("STOCK", "ETF")
                    and not gs.get("session_open_override")
                    and not gs.get("session_close_override")):
                st.warning(
                    "⚠️ Cost profile is **STOCK/ETF** but session overrides "
                    "are blank — the strategy will trade using UK 100 hours "
                    "(8:30–15:30), which is PRE-MARKET for US stocks. "
                    "Click '**US stocks (UK time)**' above to fix."
                )
        except Exception:
            pass
    # Translate bar-count → human-readable lead time so the user can pick
    # without doing the arithmetic. Driven by the TRIGGER's TF (which is
    # locked to the data interval).
    _bar_min = tf_minutes(trig_tf)
    _total_min = int(gs["min_bars_before_flat_by"]) * _bar_min
    if _total_min == 0:
        _lead = "no filter (any entry accepted up to flat_by)"
    elif _total_min < 60:
        _lead = f"≈ {_total_min} min of room before session close"
    else:
        _h, _m = divmod(_total_min, 60)
        _lead = (f"≈ {_h}h {_m:02d}m of room before session close"
                 if _m else f"≈ {_h}h of room before session close")
    st.caption(_lead)

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
