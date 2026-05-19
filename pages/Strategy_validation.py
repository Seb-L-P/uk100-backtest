"""
Strategy Validation — pressure-test a saved candidate before paper trading.

Workflow:
  1. Pick a saved sweep file (or paste a candidate JSON).
  2. Pick which candidate from that sweep to validate.
  3. Choose which tests to run:
       - Walk-forward (K folds across the full date range)
       - Multi-asset (run the SAME graph on other assets you select)
       - Monte Carlo trade shuffle (sequence-luck CI on OOS Sharpe)
       - Refinement (Optuna pass narrowing in on this graph's params)
  4. Read the results — each test answers a different angle of
     "should I actually trade this?".

A real edge survives walk-forward consistency AND transfers across at
least SOME related assets AND has a Monte Carlo P5 that's positive. A
strategy that only "wins" on its discovery window is a fluke.
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import config
from config import PROFILES, profile_for, trading_window_for
from data.fetcher import fetch
from backtest.graph import _parse_hhmm
from backtest.sweep_persistence import (
    list_saved_sweeps, load_sweep, candidate_to_graph, _graph_from_dict,
)
from backtest.candidate_validation import (
    walk_forward_candidate,
    multi_asset_check,
    monte_carlo_candidate,
)
from strategies import registry as reg


st.set_page_config(
    page_title="Strategy validation",
    layout="wide",
    initial_sidebar_state="expanded",
)


VALIDATION_ASSETS = [
    {"label": "UK 100 (FTSE)", "ticker": "UK100"},
    {"label": "AAPL",          "ticker": "AAPL"},
    {"label": "TSLA",          "ticker": "TSLA"},
    {"label": "MSFT",          "ticker": "MSFT"},
    {"label": "BTC-USD",       "ticker": "BTC-USD"},
]


# ====================================================================
#                              SIDEBAR
# ====================================================================
with st.sidebar:
    st.header("Validation setup")
    st.caption(
        "Load a candidate from a saved sweep, then run battery of "
        "out-of-window tests on it."
    )

    # ---- Pick a saved sweep ----
    sweeps = list_saved_sweeps()
    if not sweeps:
        st.info(
            "No saved sweeps yet. Run a sweep on the **Strategy discovery** "
            "page and click **Save sweep**."
        )
        st.stop()

    sweep_labels = []
    for s in sweeps:
        sharpe_str = (f" · OOS={s['top_oos_sharpe']:.2f}"
                      if s["top_oos_sharpe"] is not None else "")
        n_q = s["n_qualified"]
        sweep_labels.append(
            f"{s['asset']} @ {s['data_tf']} · {s['n_trials']}t · "
            f"{n_q}Q{sharpe_str} · {s['saved_at']}"
        )
    sweep_idx = st.selectbox(
        "Saved sweep",
        range(len(sweeps)),
        format_func=lambda i: sweep_labels[i],
    )
    chosen_sweep_meta = sweeps[sweep_idx]
    sweep_path = chosen_sweep_meta["path"]

    sweep_data = load_sweep(sweep_path)

    # ---- Pick a candidate within the sweep ----
    all_candidates = (sweep_data.get("qualified_candidates", [])
                      + sweep_data.get("disqualified_candidates", []))
    if not all_candidates:
        st.error("This saved sweep contains no candidates.")
        st.stop()

    cand_labels = []
    for i, c in enumerate(all_candidates):
        g = c["graph"]
        trig_label = reg.STRATEGIES.get(g["trigger"]["strategy_key"],
                                         type("X", (), {"label": g["trigger"]["strategy_key"]}))().__class__.__name__ if False else g["trigger"]["strategy_key"]
        # Use the registry label for the trigger
        try:
            trig_label = reg.STRATEGIES[g["trigger"]["strategy_key"]].label
        except KeyError:
            trig_label = g["trigger"]["strategy_key"]
        sharpe = c["metrics"]["sharpe"]
        sharpe_str = f"{sharpe:+.2f}" if sharpe is not None else "DQ"
        n_sup = len(g.get("supporters", []))
        n_veto = len(g.get("vetoes", []))
        cand_labels.append(
            f"#{i + 1}  {trig_label} @ {g['trigger']['timeframe']}  ·  "
            f"OOS={sharpe_str}  ·  {n_sup}sup/{n_veto}veto"
        )

    cand_idx = st.selectbox(
        "Candidate", range(len(all_candidates)),
        format_func=lambda i: cand_labels[i],
    )
    chosen_candidate = all_candidates[cand_idx]

    # ---- Tests to run ----
    st.divider()
    st.subheader("Tests")

    run_walkforward = st.checkbox(
        "Walk-forward", value=True,
        help="Split the full date range into K consecutive folds, "
             "run the candidate on each. Consistent positive sharpe across "
             "folds = real edge. Wildly varying = regime-specific.",
    )
    wf_folds = st.slider(
        "WF folds", min_value=3, max_value=10, value=5, step=1,
    ) if run_walkforward else 5

    run_multi_asset = st.checkbox(
        "Multi-asset cross-check", value=True,
        help="Run the SAME graph on different assets to see if the edge "
             "transfers. A real edge generally does; an asset-specific "
             "fluke doesn't.",
    )
    if run_multi_asset:
        ma_asset_labels = [a["label"] for a in VALIDATION_ASSETS]
        # Default = all assets except the one the sweep was on
        sweep_asset = sweep_data.get("asset", "")
        ma_default = [lbl for lbl in ma_asset_labels if lbl != sweep_asset][:4]
        ma_selected = st.multiselect(
            "Assets to test", ma_asset_labels,
            default=ma_default,
        )
        ma_data_tf = st.selectbox(
            "Multi-asset data TF",
            ["5m", "15m", "30m", "1h"],
            index=1,
        )
        ma_days_back = st.slider(
            "Multi-asset days", min_value=30, max_value=365, value=60, step=10,
        )
    else:
        ma_selected = []
        ma_data_tf = "15m"
        ma_days_back = 60

    run_monte_carlo = st.checkbox(
        "Monte Carlo (trade shuffle)", value=True,
        help="Reshuffle realised trades N times. P5/P50/P95 final balance "
             "and max DD show how lucky the trade SEQUENCE was. Tight "
             "interval = robust; wide = the headline number depends "
             "heavily on order.",
    )
    mc_sims = st.slider(
        "MC simulations", min_value=100, max_value=5000, value=1000, step=100,
    ) if run_monte_carlo else 1000

    run_refinement = st.checkbox(
        "Refinement (Optuna)", value=False,
        help="Bayesian-driven focused search around this candidate's "
             "parameters. Slow — disabled by default. Use after the other "
             "tests show the structure is promising.",
    )
    if run_refinement:
        refine_n_trials = st.slider(
            "Refinement trials", min_value=20, max_value=300, value=50, step=10,
        )
        refine_tune_weights = st.checkbox(
            "Tune supporter weights too", value=True,
        )
        refine_narrow = st.slider(
            "Narrowness around current values",
            min_value=0.1, max_value=0.6, value=0.25, step=0.05,
            help="0.1 = very tight range; 0.5 = half of each ParamSpec range.",
        )
    else:
        refine_n_trials = 50
        refine_tune_weights = True
        refine_narrow = 0.25

    st.divider()
    run_validation = st.button("Run validation", type="primary",
                                width="stretch")


# ====================================================================
#                              MAIN
# ====================================================================
st.title("Strategy Validation")
st.caption(
    "Pressure-test a saved candidate. Walk-forward proves regime stability; "
    "multi-asset proves the edge isn't asset-specific; Monte Carlo gives a "
    "confidence interval on OOS performance; refinement narrows in on the "
    "best parameters."
)

# ---- Always render the candidate summary ----
g = candidate_to_graph(chosen_candidate)
try:
    trig_label = reg.STRATEGIES[g.trigger.strategy_key].label
except KeyError:
    trig_label = g.trigger.strategy_key

st.markdown(f"## Candidate: **{trig_label}** @ `{g.trigger.timeframe}`")
cross = chosen_candidate.get("cross_split_metrics", {})
col1, col2, col3 = st.columns(3)
col1.metric("IS Sharpe",
            f"{cross.get('is_sharpe', 0):+.2f}" if cross.get('is_sharpe') is not None else "—")
col2.metric("Val Sharpe",
            f"{cross.get('val_sharpe', 0):+.2f}" if cross.get('val_sharpe') is not None else "—")
col3.metric("OOS Sharpe",
            f"{cross.get('oos_sharpe', 0):+.2f}" if cross.get('oos_sharpe') is not None else "—")

# Supporters & vetoes
col_sup, col_veto = st.columns(2)
with col_sup:
    if g.supporters:
        st.markdown(f"**Supporters ({len(g.supporters)})**")
        for s in g.supporters:
            lbl = reg.STRATEGIES.get(s.strategy_key,
                  type("X", (), {"label": s.strategy_key})).label
            st.markdown(f"- {lbl} @ `{s.timeframe}` (w={s.weight:.2f})")
    else:
        st.markdown("**Supporters** — none")
with col_veto:
    if g.vetoes:
        st.markdown(f"**Vetoes ({len(g.vetoes)})**")
        for v in g.vetoes:
            lbl = reg.STRATEGIES.get(v.strategy_key,
                  type("X", (), {"label": v.strategy_key})).label
            st.markdown(f"- {lbl} @ `{v.timeframe}`")
    else:
        st.markdown("**Vetoes** — none")

with st.expander("Full parameters", expanded=False):
    st.json(chosen_candidate["graph"])

if not run_validation:
    st.info("Configure the tests on the left and click **Run validation**.")
    st.stop()


# ====================================================================
#                  LOAD THE SWEEP'S ORIGINAL DATA
# ====================================================================
sweep_asset = sweep_data.get("asset", "UK 100 (FTSE)")
sweep_data_tf = sweep_data.get("data_tf", "15m")
n_bars = sweep_data.get("n_bars_loaded", 5000)
cost_profile_name = sweep_data.get("cost_profile", "UK100")

# Reverse-lookup the ticker from the asset preset
preset_to_ticker = {
    "UK 100 (FTSE)": "UK100",
    "AAPL": "AAPL", "TSLA": "TSLA", "MSFT": "MSFT",
    "BTC-USD": "BTC-USD",
}
sweep_ticker = preset_to_ticker.get(sweep_asset, sweep_asset)

with st.spinner(f"Fetching {sweep_ticker} {sweep_data_tf} for walk-forward..."):
    try:
        sweep_data_df = fetch(
            ticker=sweep_ticker, interval=sweep_data_tf,
            source="eodhd", ig_num_points=n_bars,
        )
    except Exception as e:
        st.error(f"Couldn't refetch the sweep's source data: {e}")
        st.stop()

# Apply RTH filter same as the discovery page
rth_window = trading_window_for(cost_profile_name, "rth")
if rth_window is not None:
    open_t = _parse_hhmm(rth_window[0])
    close_t = _parse_hhmm(rth_window[1])
    if open_t and close_t:
        tt = sweep_data_df.index.time
        sweep_data_df = sweep_data_df.loc[(tt >= open_t) & (tt <= close_t)]


# ====================================================================
#                          WALK-FORWARD
# ====================================================================
if run_walkforward:
    st.markdown("---")
    st.markdown("### Walk-forward")
    st.caption(
        f"Splitting {len(sweep_data_df):,} bars into {wf_folds} consecutive "
        "folds, running the candidate FRESH on each. Identical params "
        "across folds — this tests REGIME STABILITY, not adaptive tuning."
    )
    with st.spinner("Running walk-forward..."):
        try:
            wf_result = walk_forward_candidate(
                sweep_data_df, g, n_folds=wf_folds,
                costs=PROFILES.get(cost_profile_name),
                min_trades_per_fold=5,
            )
        except Exception as e:
            st.error(f"Walk-forward failed: {e}")
            wf_result = None

    if wf_result:
        # Headline
        h1, h2, h3 = st.columns(3)
        h1.metric("Positive folds",
                  f"{wf_result.positive_fold_fraction:.0%}",
                  help="Fraction of folds with Sharpe > 0. ≥60% = decent stability.")
        h2.metric("Median fold Sharpe", f"{wf_result.median_sharpe:+.2f}")
        h3.metric("Total folds run", wf_result.n_folds)

        # Per-fold table
        wf_df = pd.DataFrame({
            "Fold": range(1, wf_result.n_folds + 1),
            "Start": [b[0] for b in wf_result.fold_boundaries],
            "End": [b[1] for b in wf_result.fold_boundaries],
            "Sharpe": wf_result.fold_sharpes,
            "Trades": wf_result.fold_trades,
            "Return %": wf_result.fold_returns,
            "Max DD %": wf_result.fold_max_dds,
        })
        st.dataframe(
            wf_df.style.format({
                "Sharpe": "{:+.2f}",
                "Return %": "{:+.2f}",
                "Max DD %": "{:.2f}",
            }, na_rep="—"),
            width="stretch", hide_index=True,
        )

        # Quick chart of fold Sharpes
        import altair as alt
        chart_data = wf_df.dropna(subset=["Sharpe"])
        if not chart_data.empty:
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("Fold:O"),
                y=alt.Y("Sharpe:Q"),
                color=alt.condition(
                    alt.datum.Sharpe > 0,
                    alt.value("#4ade80"), alt.value("#f87171"),
                ),
            )
            st.altair_chart(chart, use_container_width=True)


# ====================================================================
#                      MULTI-ASSET CROSS-CHECK
# ====================================================================
if run_multi_asset and ma_selected:
    st.markdown("---")
    st.markdown("### Multi-asset cross-check")
    st.caption(
        f"Same graph, same parameters — run on **{len(ma_selected)}** "
        f"different assets ({ma_days_back} days each at {ma_data_tf}). "
        "An edge that transfers across at least some related assets is "
        "more credible than one that only works on the sweep asset."
    )

    minutes_per_bar = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}[ma_data_tf]
    ma_bars_per_day = max(1, (6.5 * 60) // minutes_per_bar)
    ma_bar_count = int(ma_days_back * ma_bars_per_day * 1.2)

    asset_specs = [
        {"label": a["label"], "ticker": a["ticker"], "data_tf": ma_data_tf}
        for a in VALIDATION_ASSETS if a["label"] in ma_selected
    ]

    def _ma_loader(ticker, data_tf):
        df = fetch(ticker=ticker, interval=data_tf,
                   source="eodhd", ig_num_points=ma_bar_count)
        # Apply RTH filter for this asset's cost profile
        profile = profile_for(ticker)
        window = trading_window_for(profile.instrument, "rth")
        if window is not None:
            open_t = _parse_hhmm(window[0])
            close_t = _parse_hhmm(window[1])
            if open_t and close_t:
                tt = df.index.time
                df = df.loc[(tt >= open_t) & (tt <= close_t)]
        return df

    ma_progress = st.progress(0.0, text="Starting multi-asset check...")
    def _ma_cb(frac, msg):
        ma_progress.progress(min(1.0, frac), text=msg)

    with st.spinner("Running multi-asset cross-check..."):
        try:
            ma_result = multi_asset_check(
                g, asset_specs, _ma_loader,
                cost_profile_lookup=lambda t: profile_for(t),
                progress_callback=_ma_cb,
            )
            ma_progress.progress(1.0, text="Multi-asset done.")
        except Exception as e:
            st.error(f"Multi-asset check failed: {e}")
            ma_result = None

    if ma_result:
        hm1, hm2 = st.columns(2)
        hm1.metric("Assets with positive Sharpe",
                   f"{ma_result.n_passed}/{len(ma_result.assets)}")
        hm2.metric("Pass fraction", f"{ma_result.pass_fraction:.0%}")
        ma_df = pd.DataFrame([{
            "Asset": a.asset_label,
            "Ticker": a.ticker,
            "Sharpe": a.sharpe if not (a.sharpe != a.sharpe) else None,
            "Trades": a.n_trades,
            "Return %": a.return_pct,
            "Max DD %": a.max_dd_pct,
            "Hit rate": a.hit_rate,
            "Error": a.error or "",
        } for a in ma_result.assets])
        st.dataframe(
            ma_df.style.format({
                "Sharpe": "{:+.2f}",
                "Return %": "{:+.2f}",
                "Max DD %": "{:.2f}",
                "Hit rate": "{:.1%}",
            }, na_rep="—"),
            width="stretch", hide_index=True,
        )


# ====================================================================
#                          MONTE CARLO
# ====================================================================
if run_monte_carlo:
    st.markdown("---")
    st.markdown("### Monte Carlo trade shuffle")
    st.caption(
        f"Run the candidate once on the sweep's full window, then reshuffle "
        f"the trade order **{mc_sims:,}** times. The percentile spread tells "
        "you how lucky the realised sequence was."
    )
    with st.spinner("Running Monte Carlo..."):
        try:
            mc_result = monte_carlo_candidate(
                sweep_data_df, g,
                n_simulations=mc_sims,
                costs=PROFILES.get(cost_profile_name),
            )
        except Exception as e:
            st.error(f"Monte Carlo failed: {e}")
            mc_result = None

    if mc_result:
        st.markdown("**Final balance distribution**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Actual", f"£{mc_result.actual_final_balance:,.2f}")
        mc2.metric("P5 (worst 5%)", f"£{mc_result.final_balance_p5:,.2f}")
        mc3.metric("P50 (median)", f"£{mc_result.final_balance_p50:,.2f}")
        mc4.metric("P95 (best 5%)", f"£{mc_result.final_balance_p95:,.2f}")

        st.markdown("**Max drawdown distribution**")
        dd1, dd2, dd3, dd4 = st.columns(4)
        dd1.metric("Actual", f"{mc_result.actual_max_dd:.2f}%")
        dd2.metric("P5 (best 5%)", f"{mc_result.max_dd_p5:.2f}%")
        dd3.metric("P50 (median)", f"{mc_result.max_dd_p50:.2f}%")
        dd4.metric("P95 (worst 5%)", f"{mc_result.max_dd_p95:.2f}%")

        # Reading hint
        if mc_result.final_balance_p5 < mc_result.actual_final_balance * 0.5:
            st.warning(
                "⚠️ The P5 final balance is much lower than the actual — "
                "the realised trade SEQUENCE got lucky. A worst-case "
                "shuffle would have hurt a lot more."
            )


# ====================================================================
#                              REFINEMENT
# ====================================================================
if run_refinement:
    st.markdown("---")
    st.markdown("### Refinement pass (Optuna)")
    st.caption(
        f"Bayesian-driven focused search around this candidate's parameters. "
        f"Narrowness={refine_narrow:.2f} (=ratio of original range explored). "
        f"Reports whether refinement IMPROVES the OOS Sharpe — if not, the "
        f"candidate was already near a local optimum, or the structure has "
        "no room to improve."
    )
    from backtest.refinement import refine_candidate

    refine_progress = st.progress(0.0, text="Starting refinement...")
    def _r_cb(frac, msg):
        refine_progress.progress(min(1.0, frac), text=msg)

    with st.spinner("Refining..."):
        try:
            refined = refine_candidate(
                sweep_data_df, g,
                n_trials=refine_n_trials,
                narrow_factor=refine_narrow,
                tune_weights=refine_tune_weights,
                costs=PROFILES.get(cost_profile_name),
                progress_callback=_r_cb,
            )
            refine_progress.progress(1.0, text="Refinement done.")
        except Exception as e:
            st.error(f"Refinement failed: {e}")
            refined = None

    if refined:
        # Headline comparison
        st.markdown("**Original vs Refined (on the SAME 3-way split)**")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Original OOS Sharpe",
                   f"{refined.original_metrics.sharpe:+.2f}"
                   if not math.isinf(refined.original_metrics.sharpe) else "DQ")
        rc2.metric("Refined OOS Sharpe",
                   f"{refined.refined_oos_sharpe:+.2f}"
                   if not math.isinf(refined.refined_oos_sharpe) else "DQ",
                   delta=(f"{refined.refined_oos_sharpe - refined.original_metrics.sharpe:+.2f}"
                          if (not math.isinf(refined.refined_oos_sharpe)
                              and not math.isinf(refined.original_metrics.sharpe))
                          else None))
        rc3.metric("Refined Val Sharpe",
                   f"{refined.refined_val_sharpe:+.2f}"
                   if not math.isinf(refined.refined_val_sharpe) else "DQ")
        rc4.metric("Refined IS Sharpe",
                   f"{refined.refined_is_sharpe:+.2f}"
                   if not math.isinf(refined.refined_is_sharpe) else "DQ")
        st.caption(
            f"Optuna ran {refined.n_completed}/{refined.n_trials} trials "
            f"successfully ({refined.n_pruned} pruned)."
        )

        with st.expander("Refined graph (full JSON)", expanded=False):
            from backtest.sweep_persistence import _graph_to_dict
            st.json(_graph_to_dict(refined.refined_graph))


# ====================================================================
#                       OVERALL READING
# ====================================================================
st.markdown("---")
with st.expander("How to read all these results together", expanded=False):
    st.markdown("""
The point of running ALL these tests is to triangulate. Any single test
can mislead; the conjunction is harder to fool:

- **Walk-forward says positive across most folds + Multi-asset says
  positive across most assets + Monte Carlo P5 is positive** → strong
  candidate for paper trading.

- **Walk-forward inconsistent (some folds great, some negative)** →
  regime-dependent. Either accept the strategy WON'T trade in
  unfavourable regimes (gate it on regime indicators) or move on.

- **Multi-asset fails completely** → asset-specific edge. Could still
  be real — some edges are asset-specific (e.g. AAPL has lots of open
  gap behaviour MSFT doesn't share). Investigate WHY before trusting.

- **Monte Carlo P5 deeply negative** → sequence-luck heavy. The strategy
  needs a lot of trades to escape sequencing risk; don't size up.

- **Refinement substantially improves OOS Sharpe** → the original
  candidate wasn't at a local optimum. Worth using the refined version,
  but ALSO worth re-running walk-forward + multi-asset on the refined
  graph — it might be more fragile than the original.
""")
