"""
Strategy Discovery — random sweep over the full DecisionGraph design space.

What gets swept:
  - Trigger strategy choice + parameters
  - Trigger TIMEFRAME (sampled from user-selected pool — decoupled from data TF)
  - 0..N supporters: strategy choice + TF + parameters + (optional) weight
  - 0..M vetoes: strategy choice + TF + parameters
  - (Optional) Graph-level knobs: min_score, risk_floor/ceiling, risk_curve

Data is fetched at the finest TF the user picks (default 1m). The trigger
runs at its own TF (15m, 1h, etc.) — fills happen at data-TF granularity,
giving accurate SL/TP and pending-order behaviour while still letting the
strategy reason at its preferred cadence.

The sweep evaluates every random graph on the IS window, ranks them, sends
the top-K to Val for re-ranking, and the top-M of THOSE to OOS for final
report. OOS is the only number you should trust; the other two carry
selection bias.
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
from config import PROFILES, profile_for, session_defaults_for, trading_window_for
from data.fetcher import fetch
from backtest.graph import _parse_hhmm, ALL_TIMEFRAMES, tf_minutes
from backtest.sweep_space import SearchSpace, describe_graph
from backtest.sweep_runner import run_sweep
from backtest.sweep_persistence import save_sweep, list_saved_sweeps
from backtest.multi_seed import run_multi_seed
from strategies import registry as reg


st.set_page_config(
    page_title="Strategy discovery",
    layout="wide",
    initial_sidebar_state="expanded",
)


ASSET_PRESETS: dict[str, dict] = {
    "UK 100 (FTSE)":   {"ticker": "UK100",    "data_tf": "1m"},
    "AAPL":            {"ticker": "AAPL",     "data_tf": "1m"},
    "TSLA":            {"ticker": "TSLA",     "data_tf": "1m"},
    "MSFT":            {"ticker": "MSFT",     "data_tf": "1m"},
    "BTC-USD":         {"ticker": "BTC-USD",  "data_tf": "5m"},
    "Custom...":       {"ticker": "",         "data_tf": "5m"},
}

TIME_RANGES: dict[str, int] = {
    "Last 60 days":   60,
    "Last 120 days":  120,
    "Last 1 year":    365,
    "Last 2 years":   730,
}

# All TFs the user can pick as data interval (must support sampling)
DATA_TF_CHOICES = ["1m", "5m", "15m", "30m", "1h"]

# All TFs the trigger can be sampled at (must be >= data_tf)
TRIGGER_TF_CHOICES = ["5m", "15m", "30m", "1h", "4h"]


# ====================================================================
#                              SIDEBAR
# ====================================================================
with st.sidebar:
    st.header("Discovery setup")
    st.caption(
        "Random sweep over decision graphs — trigger, supporters, vetoes, "
        "timeframes, parameters, and (optionally) weights and graph knobs. "
        "3-way IS / Val / OOS split."
    )

    # ---- Asset + data ----
    asset_preset = st.selectbox("Asset", list(ASSET_PRESETS.keys()), index=0)
    preset_cfg = ASSET_PRESETS[asset_preset]
    ticker = preset_cfg["ticker"]
    default_data_tf = preset_cfg["data_tf"]
    if asset_preset == "Custom...":
        ticker = st.text_input("Ticker", value="UK100",
                                help="EODHD symbol (e.g. AAPL, TSLA, BTC-USD, "
                                     "UK100, GDAXI.INDX).")

    data_tf = st.selectbox(
        "Data TF (bar resolution)",
        DATA_TF_CHOICES,
        index=DATA_TF_CHOICES.index(default_data_tf),
        help="Finest interval the engine sees. Fills, SL/TP, and "
             "pending-order triggers happen at this granularity. "
             "1m gives the best intrabar accuracy at the cost of "
             "more bars per backtest (slower sweep). 5m is a fair "
             "compromise.",
    )

    range_label = st.selectbox("Date range", list(TIME_RANGES.keys()), index=0)
    days_back = TIME_RANGES[range_label]
    minutes_per_bar = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}[data_tf]
    bars_per_day = max(1, (6.5 * 60) // minutes_per_bar)
    bar_count = int(days_back * bars_per_day * 1.2)

    auto_profile = profile_for(ticker)
    profile_names = list(PROFILES.keys())
    try:
        idx_default = profile_names.index(auto_profile.instrument)
    except ValueError:
        idx_default = profile_names.index("DEFAULT")
    chosen_profile = st.selectbox("Cost profile", profile_names,
                                   index=idx_default)
    config.COSTS = PROFILES[chosen_profile]

    data_hours = st.radio(
        "Data hours",
        ["RTH (regular hours)", "Extended hours", "All available"],
        index=0 if PROFILES[chosen_profile].instrument not in
              ("BTC", "ETH", "EURUSD", "GBPUSD", "USDJPY") else 2,
    )
    hours_mode = {"RTH (regular hours)": "rth",
                  "Extended hours": "eth",
                  "All available": None}[data_hours]

    # Performance warning for 1m sweeps
    if data_tf == "1m" and days_back > 120:
        st.warning(
            f"⚠️ 1m × {days_back} days produces ~{bar_count:,} bars per backtest. "
            "A 200-trial sweep could take 30+ minutes. Consider 5m for "
            "longer windows, or 60–120 days at 1m."
        )

    # ---- Sweep budget ----
    st.divider()
    st.subheader("Sweep budget")

    n_trials = st.slider(
        "Total IS trials", min_value=20, max_value=2000, value=200, step=20,
        help="More trials = more thorough but slower. Each trial is one "
             "full backtest on the IS window.",
    )
    top_k = st.slider(
        "Top-K to Val", min_value=5, max_value=200, value=20, step=5,
        help="Best K from IS get re-evaluated on the Val window.",
    )
    top_m = st.slider(
        "Top-M to OOS", min_value=3, max_value=50, value=10, step=1,
        help="Best M from Val get final OOS evaluation.",
    )
    min_trades = st.slider(
        "Min trades floor (cap)", min_value=3, max_value=100, value=20, step=1,
        help="Maximum min-trades floor. Auto-scaled DOWN per split based "
             "on each split's bar count.",
    )

    # ---- Data split ----
    st.divider()
    st.subheader("Data split")
    is_ratio = st.slider(
        "IS ratio", min_value=0.4, max_value=0.8, value=0.6, step=0.05,
    )
    val_ratio = st.slider(
        "Val ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05,
    )

    # ---- Search space ----
    st.divider()
    st.subheader("Search space")

    all_keys = list(reg.STRATEGIES.keys())
    all_labels = {k: reg.STRATEGIES[k].label for k in all_keys}

    trigger_choices = st.multiselect(
        "Triggers allowed", all_keys,
        default=all_keys,
        format_func=lambda k: all_labels[k],
        help="Strategies the sweep is allowed to use as the trigger.",
    )

    # Trigger TF sampling pool (filtered to >= data_tf)
    valid_trigger_tfs = [tf for tf in TRIGGER_TF_CHOICES
                         if tf_minutes(tf) >= tf_minutes(data_tf)]
    if data_tf in ALL_TIMEFRAMES and data_tf not in valid_trigger_tfs:
        valid_trigger_tfs.insert(0, data_tf)
    # Sensible default: cover 15m / 30m / 1h
    default_trigger_tfs = [tf for tf in ["15m", "30m", "1h"]
                           if tf in valid_trigger_tfs] or valid_trigger_tfs[:1]
    trigger_tf_options = st.multiselect(
        "Trigger TFs (sampled per trial)",
        valid_trigger_tfs,
        default=default_trigger_tfs,
        help="Each trial picks one of these as the trigger TF. Data is "
             "fetched at data_tf; the trigger strategy decides at its "
             "own TF. Supporters/vetoes are picked at TFs >= trigger TF.",
    )

    max_supporters = st.slider(
        "Max supporters per graph", min_value=0, max_value=4, value=2, step=1,
    )
    max_vetoes = st.slider(
        "Max vetoes per graph", min_value=0, max_value=3, value=1, step=1,
    )

    # ---- Extra sweep dimensions ----
    st.divider()
    st.subheader("Extra dimensions")

    sweep_weights = st.checkbox(
        "Sweep supporter weights",
        value=True,
        help="Each supporter gets a random weight from the range below. "
             "When off, all supporters use weight 1.0 (TF-distance fallback "
             "still applies). Original graph design left these off the sweep; "
             "with this on, the sampler explores combinations that need a "
             "particular supporter to be louder/quieter.",
    )
    if sweep_weights:
        weight_lo, weight_hi = st.slider(
            "Weight range",
            min_value=0.1, max_value=3.0,
            value=(0.3, 2.0), step=0.1,
        )
    else:
        weight_lo, weight_hi = 1.0, 1.0

    sweep_graph_knobs = st.checkbox(
        "Sweep graph-level knobs",
        value=False,
        help="Sample min_score (confidence threshold), risk_floor, "
             "risk_ceiling, and risk_curve per trial. Adds a lot of "
             "dimensions — leave off until the structural sweep is "
             "doing well.",
    )

    # ---- Multi-seed ----
    st.divider()
    st.subheader("Multi-seed (robustness)")

    multi_seed_mode = st.checkbox(
        "Run with multiple seeds",
        value=False,
        help="Run the SAME SearchSpace across N seeds. The results aggregator "
             "shows which (trigger, trigger TF) STRUCTURES repeatedly surface "
             "across seeds — far better evidence of robustness than a single "
             "seed's top result. Each seed runs a full sweep, so total time "
             "scales linearly with N.",
    )
    if multi_seed_mode:
        seeds_text = st.text_input(
            "Seeds (comma-separated)",
            value="42, 99, 1337, 7",
            help="Each becomes a full independent sweep. Total time = "
                 "this many × single-sweep time.",
        )
        try:
            seeds_list = [int(s.strip()) for s in seeds_text.split(",")
                          if s.strip()]
        except ValueError:
            st.error(f"Couldn't parse seeds — use integers comma-separated.")
            seeds_list = [42]
        seed = seeds_list[0] if seeds_list else 42
    else:
        seed = st.number_input(
            "Random seed", min_value=0, max_value=10_000, value=42, step=1,
            help="Same seed → identical sweep results across runs.",
        )
        seeds_list = [int(seed)]

    run_clicked = st.button("Run sweep", type="primary", width="stretch")


# ====================================================================
#                          MAIN AREA — INTRO
# ====================================================================
st.title("Strategy Discovery")
st.caption(
    "Random sweep over decision graphs. Data is pulled at fine resolution "
    "(1m / 5m); trigger decisions happen at higher TFs you select. "
    "Trust the **OOS Sharpe** — the other Sharpe columns inflate from "
    "selection bias."
)

with st.expander("How to read these results", expanded=False):
    st.markdown("""
**The three Sharpe values per graph:**

- **OOS Sharpe** — annualised Sharpe on the held-out window. **The number
  to trust.** No selection bias touched this column.
- **Val Sharpe** — Sharpe on the validation window. Mild bias (we picked
  top-K from IS).
- **IS Sharpe** — Sharpe on the optimisation window. Heavy bias (we
  literally chose the highest of N trials). Always too good.

**Red flags:**
1. IS Sharpe is high, OOS Sharpe is negative → overfit. Don't trade this.
2. OOS trades count much lower than IS/Val → strategy fired rarely on
   OOS, the Sharpe is statistical noise.
3. OOS Sharpe drops sharply from Val → the Val period was selection-fit.

**Good signs:**
- IS ≥ Val ≥ OOS, with OOS still positive and trade count adequate.
- Multiple distinct triggers in the top-5 with similar OOS Sharpes →
  evidence the edge is real, not parameter-specific.
""")

if not run_clicked:
    st.info("Configure on the left, then click **Run sweep**.")
    st.stop()

if not trigger_choices:
    st.error("Pick at least one trigger strategy.")
    st.stop()

if not trigger_tf_options:
    st.error("Pick at least one trigger timeframe.")
    st.stop()

if is_ratio + val_ratio >= 1.0:
    st.error(f"IS ratio + Val ratio = {is_ratio + val_ratio:.2f}. "
             f"Must be < 1.0 to leave room for OOS.")
    st.stop()


# ====================================================================
#                            FETCH DATA
# ====================================================================
with st.spinner(f"Fetching {ticker} {data_tf} ({bar_count:,} bars)..."):
    try:
        data = fetch(ticker=ticker, interval=data_tf,
                     source="eodhd", ig_num_points=bar_count)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

if hours_mode is not None:
    w = trading_window_for(chosen_profile, hours_mode)
    if w is not None:
        open_t = _parse_hhmm(w[0])
        close_t = _parse_hhmm(w[1])
        if open_t and close_t:
            tt = data.index.time
            data = data.loc[(tt >= open_t) & (tt <= close_t)]

if len(data) < 500:
    st.error(f"Only {len(data)} bars after filtering — too short for a "
             f"meaningful 3-way sweep. Increase the date range or pick a "
             f"finer TF.")
    st.stop()

# Data-quality warning
expected_min_bars = int(days_back * bars_per_day * 0.5)
if len(data) < expected_min_bars:
    actual_days = (data.index[-1] - data.index[0]).days
    st.warning(
        f"⚠️ Only **{len(data):,} bars** (≈{actual_days} days) loaded, "
        f"you asked for ~{days_back} days. The data source may not have "
        f"intraday history that far back, or your plan caps the lookback. "
        f"Try a higher data TF (15m, 1h) for longer date ranges."
    )

st.caption(f"Loaded **{len(data):,}** bars from "
           f"`{data.index[0]}` to `{data.index[-1]}` "
           f"(≈{(data.index[-1] - data.index[0]).days} days).")


# ====================================================================
#                          BUILD SEARCH SPACE
# ====================================================================
space = SearchSpace(
    triggers_pool=trigger_choices,
    max_supporters=max_supporters,
    max_vetoes=max_vetoes,
    data_tf=data_tf,
    trigger_tf_options=trigger_tf_options,
    sweep_weights=sweep_weights,
    weight_range=(float(weight_lo), float(weight_hi)),
    sweep_graph_knobs=sweep_graph_knobs,
)


# ====================================================================
#                              RUN SWEEP
# ====================================================================
progress_bar = st.progress(0.0, text="Starting sweep...")

def _cb(frac: float, msg: str):
    progress_bar.progress(min(1.0, frac), text=msg)

multi_result = None  # only set in multi-seed mode

with st.spinner("Running sweep..."):
    try:
        if multi_seed_mode and len(seeds_list) > 1:
            multi_result = run_multi_seed(
                data, space, seeds=seeds_list,
                n_trials=int(n_trials),
                is_ratio=float(is_ratio),
                val_ratio=float(val_ratio),
                top_k=int(top_k),
                top_m=int(top_m),
                min_trades=int(min_trades),
                costs=PROFILES[chosen_profile],
                progress_callback=_cb,
            )
            # Show the FIRST seed's sweep as the "result" for the card view
            # below — the multi-seed structure leaderboard is shown separately.
            result = multi_result.seed_runs[0].result
        else:
            result = run_sweep(
                data, space,
                n_trials=int(n_trials),
                is_ratio=float(is_ratio),
                val_ratio=float(val_ratio),
                top_k=int(top_k),
                top_m=int(top_m),
                min_trades=int(min_trades),
                costs=PROFILES[chosen_profile],
                seed=int(seed),
                progress_callback=_cb,
            )
    except Exception as e:
        st.error(f"Sweep crashed: {type(e).__name__}: {e}")
        st.exception(e)
        st.stop()

progress_bar.progress(1.0, text="Sweep complete.")


# ====================================================================
#                              RESULTS
# ====================================================================
st.markdown("## Results")

# ---- Multi-seed structure leaderboard (when relevant) ----
if multi_result is not None:
    st.markdown("### Cross-seed structure leaderboard")
    st.caption(
        f"Same SearchSpace × **{multi_result.n_seeds}** seeds. Structures "
        "are grouped by (trigger strategy, trigger TF). A structure that "
        "appears in top-OOS across multiple seeds is much more credible "
        "than a single-seed peak."
    )
    msdf = multi_result.to_dataframe()
    if not msdf.empty:
        msdf_display = msdf.copy()
        msdf_display["trigger"] = msdf_display["trigger"].map(
            lambda k: reg.STRATEGIES[k].label if k in reg.STRATEGIES else k
        )
        st.dataframe(
            msdf_display.style.format({
                "best OOS": "{:+.2f}",
                "median OOS": "{:+.2f}",
                "worst OOS": "{:+.2f}",
            }),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "**'seeds'** column: how many distinct seeds produced this structure "
            "in their top-OOS. Higher = more robust. **'appearances'** can "
            "exceed 'seeds' when a structure shows up multiple times in one "
            "seed's top-OOS (different parameter combos)."
        )
    else:
        st.info("No qualified structures surfaced across seeds. "
                "Try lower min-trades floor, longer data, or fewer max-supporters.")
    st.markdown("---")
    st.markdown(f"### First seed's cards (seed={multi_result.seed_runs[0].seed})")
    st.caption("Per-card view below is for the FIRST seed only. "
               "Use the table above to compare structures across all seeds.")

# ---- Split + sweep stats summary ----
sp1, sp2, sp3 = st.columns(3)
sp1.metric("IS bars", f"{len(result.split.is_df):,}",
           help=f"{result.split.is_df.index[0]} → {result.split.is_df.index[-1]}")
sp2.metric("Val bars", f"{len(result.split.val_df):,}",
           help=f"{result.split.val_df.index[0]} → {result.split.val_df.index[-1]}")
sp3.metric("OOS bars", f"{len(result.split.oos_df):,}",
           help=f"{result.split.oos_df.index[0]} → {result.split.oos_df.index[-1]}")

disqualified_is = sum(1 for t in result.is_trials if t.metrics.disqualified_reason)
crashed_is = sum(1 for t in result.is_trials
                 if t.metrics.disqualified_reason
                 and t.metrics.disqualified_reason.startswith("crashed"))
qualified_is = result.n_trials - disqualified_is

cm1, cm2, cm3 = st.columns(3)
cm1.metric("Trials run", result.n_trials)
cm2.metric("IS qualified", f"{qualified_is}/{result.n_trials}",
           help="Met the min-trades floor AND produced non-zero return variance.")
cm3.metric("Crashed", crashed_is,
           help="Engine exceptions (caught, doesn't halt the sweep).")

st.caption(
    f"**Adaptive min-trades floor**: IS={result.is_min_trades}, "
    f"Val={result.val_min_trades}, OOS={result.oos_min_trades} "
    f"(your cap was {result.min_trades}). Scaled to each split's bar count."
)

if disqualified_is > result.n_trials * 0.6:
    st.warning(
        f"⚠️ **{disqualified_is}/{result.n_trials} trials disqualified.** "
        "Common fixes: lower the min-trades floor, use a longer date range, "
        "pick a finer trigger TF (5m / 15m fire more often than 1h), "
        "or restrict the trigger pool to known frequent-firing strategies."
    )


# ---- Result bucketing ----
def _is_dq(t):
    s = t.metrics.sharpe
    return (t.metrics.disqualified_reason is not None
            or s == float("-inf") or math.isinf(s))

qualified_oos = [t for t in result.oos_top if not _is_dq(t)]
disqualified_oos = [t for t in result.oos_top if _is_dq(t)]


# ====================================================================
#                       STRATEGY CARDS (TOP-N)
# ====================================================================
def _sharpe_str(s) -> str:
    if s is None or math.isinf(s) or math.isnan(s):
        return "DQ"
    sign = "+" if s >= 0 else ""
    return f"{sign}{s:.2f}"


def _sharpe_emoji(s) -> str:
    if s is None or math.isinf(s) or math.isnan(s):
        return "🚫"
    if s >= 1.5:
        return "🟢"
    if s >= 0.5:
        return "🟡"
    if s >= 0:
        return "🟠"
    return "🔴"


def _looks_up(graph) -> str:
    """Build a compact 'where to inspect' summary string."""
    return describe_graph(graph)


def _render_card(rank: int, trial, is_t, val_t):
    """Render one strategy as a rich card."""
    g = trial.graph
    trig_spec = reg.STRATEGIES[g.trigger.strategy_key]

    # Determine card emphasis: green border for healthy OOS, neutral otherwise
    oos_sharpe = trial.metrics.sharpe
    emoji = _sharpe_emoji(oos_sharpe)

    with st.container(border=True):
        # ---- Header row ----
        hdr_cols = st.columns([1, 4, 2])
        with hdr_cols[0]:
            st.markdown(f"### #{rank}  {emoji}")
        with hdr_cols[1]:
            st.markdown(f"### **{trig_spec.label}** @ `{g.trigger.timeframe}`")
            st.caption(trig_spec.description or "")
        with hdr_cols[2]:
            st.metric(
                "OOS Sharpe",
                _sharpe_str(oos_sharpe),
                help="Untouched during selection — trust this.",
            )

        # ---- Sharpe profile bar ----
        sp1, sp2, sp3 = st.columns(3)
        with sp1:
            st.metric("IS Sharpe",
                      _sharpe_str(is_t.metrics.sharpe) if is_t else "—",
                      help="Heavy selection bias. Always too good.")
        with sp2:
            st.metric("Val Sharpe",
                      _sharpe_str(val_t.metrics.sharpe) if val_t else "—",
                      help="Mild selection bias.")
        with sp3:
            # Cross-split degradation: red if OOS dropped sharply from IS
            if is_t and not math.isinf(is_t.metrics.sharpe):
                delta = oos_sharpe - is_t.metrics.sharpe
                st.metric("IS→OOS drop", f"{delta:+.2f}",
                          delta=f"{delta:+.2f}",
                          delta_color="inverse",
                          help="Negative = OOS worse than IS (overfit signal). "
                               "Near zero = robust.")
            else:
                st.metric("IS→OOS drop", "—")

        # ---- Trade stats row ----
        ts1, ts2, ts3, ts4 = st.columns(4)
        ts1.metric("OOS trades", trial.metrics.n_trades)
        ts2.metric("Hit rate", f"{trial.metrics.hit_rate:.0%}")
        ts3.metric("Return", f"{trial.metrics.return_pct:+.2f}%")
        ts4.metric("Max DD", f"{trial.metrics.max_drawdown_pct:.2f}%")

        # ---- Supporters & Vetoes side by side ----
        col_sup, col_veto = st.columns(2)
        with col_sup:
            if g.supporters:
                st.markdown(f"**✓ Supporters ({len(g.supporters)})**")
                for s in g.supporters:
                    s_label = reg.STRATEGIES[s.strategy_key].label
                    weight_tag = f"  ·  weight **{s.weight:.2f}**" if s.weight != 1.0 else ""
                    st.markdown(f"- {s_label}  ·  `{s.timeframe}`{weight_tag}")
            else:
                st.markdown("**✓ Supporters**  — *none*")

        with col_veto:
            if g.vetoes:
                st.markdown(f"**✗ Vetoes ({len(g.vetoes)})**")
                for v in g.vetoes:
                    v_label = reg.STRATEGIES[v.strategy_key].label
                    st.markdown(f"- {v_label}  ·  `{v.timeframe}`")
            else:
                st.markdown("**✗ Vetoes**  — *none*")

        # ---- Graph knobs (only show if non-default) ----
        non_default = []
        if g.min_score != 0.5:
            non_default.append(f"min_score={g.min_score:.2f}")
        if g.risk_floor != 0.70:
            non_default.append(f"risk_floor={g.risk_floor:.2f}")
        if g.risk_ceiling != 1.00:
            non_default.append(f"risk_ceiling={g.risk_ceiling:.2f}")
        if g.risk_curve != "linear":
            non_default.append(f"risk_curve={g.risk_curve}")
        if non_default:
            st.caption(f"⚙️ Graph knobs: {', '.join(non_default)}")

        # ---- Parameters & reproduce JSON (collapsed) ----
        with st.expander("Full parameters + reproduce JSON", expanded=False):
            st.markdown("**Trigger params**")
            st.json(g.trigger.params)

            if g.supporters:
                st.markdown("**Supporters**")
                st.json([
                    {"strategy": s.strategy_key, "tf": s.timeframe,
                     "weight": s.weight, "params": s.params}
                    for s in g.supporters
                ])
            if g.vetoes:
                st.markdown("**Vetoes**")
                st.json([
                    {"strategy": v.strategy_key, "tf": v.timeframe,
                     "params": v.params}
                    for v in g.vetoes
                ])

            st.markdown("**Reproduce in main backtester**")
            st.code(json.dumps({
                "trigger": {
                    "strategy_key": g.trigger.strategy_key,
                    "params": g.trigger.params,
                    "timeframe": g.trigger.timeframe,
                },
                "supporters": [{
                    "strategy_key": s.strategy_key,
                    "params": s.params,
                    "timeframe": s.timeframe,
                    "weight": s.weight,
                } for s in g.supporters],
                "vetoes": [{
                    "strategy_key": v.strategy_key,
                    "params": v.params,
                    "timeframe": v.timeframe,
                } for v in g.vetoes],
                "min_score": g.min_score,
                "risk_floor": g.risk_floor,
                "risk_ceiling": g.risk_ceiling,
                "risk_curve": g.risk_curve,
                "seed": result.seed,
            }, indent=2), language="json")


# ---- Render top-N cards ----
st.markdown("---")

if not qualified_oos and not disqualified_oos:
    st.warning("No OOS results produced. Try a longer date range or more trials.")
elif not qualified_oos:
    st.error(
        f"**No qualified OOS results.** All {len(disqualified_oos)} graphs "
        "made it through IS + Val selection but failed the OOS bar — "
        "this is the classic 'overfit to IS+Val' signal. Try a wider "
        "search space, longer data, or a different asset."
    )
else:
    st.markdown(f"### Top {len(qualified_oos)} qualified strategies — sorted by OOS Sharpe")
    st.caption(
        "Each card is one strategy the sweep found. The trigger is the "
        "core decision-maker; supporters scale confidence; vetoes block trades."
    )

    # Build lookups for cross-split metrics
    is_by_id = {id(t.graph): t for t in result.is_trials}
    val_by_id = {id(t.graph): t for t in result.val_top}

    for rank, trial in enumerate(qualified_oos, start=1):
        _render_card(
            rank=rank,
            trial=trial,
            is_t=is_by_id.get(id(trial.graph)),
            val_t=val_by_id.get(id(trial.graph)),
        )


# ====================================================================
#                       SAVE CANDIDATES TO DISK
# ====================================================================
st.markdown("---")
st.markdown("### Save these candidates")
st.caption(
    "Save the top-OOS results (qualified AND disqualified) as a JSON file. "
    "The **Strategy validation** page can load them and run walk-forward, "
    "multi-asset, and Monte Carlo tests on the picks you most trust."
)
save_cols = st.columns([3, 1])
with save_cols[0]:
    save_note = st.text_input(
        "Note for this sweep (optional)",
        value="",
        placeholder=f"e.g. UK100 first sweep, weights on, {n_trials} trials",
        key="sweep_save_note",
    )
with save_cols[1]:
    save_clicked = st.button("💾 Save sweep", width="stretch")

if save_clicked:
    try:
        path = save_sweep(
            result,
            asset=asset_preset,
            data_tf=data_tf,
            date_range_label=range_label,
            n_bars_loaded=len(data),
            cost_profile=chosen_profile,
            search_space_summary={
                "triggers_pool_size": len(space.triggers_pool),
                "trigger_tfs": space.trigger_tf_options,
                "max_supporters": space.max_supporters,
                "max_vetoes": space.max_vetoes,
                "sweep_weights": space.sweep_weights,
                "weight_range": list(space.weight_range) if space.sweep_weights else None,
                "sweep_graph_knobs": space.sweep_graph_knobs,
                "multi_seed": multi_seed_mode,
                "seeds": seeds_list,
            },
            note=save_note or None,
        )
        st.success(f"✅ Saved to `{path.name}`. Open the **Strategy validation** "
                   f"page to run deeper tests.")
    except Exception as e:
        st.error(f"Save failed: {e}")


# ====================================================================
#                  COMPACT TABLE (full leaderboard)
# ====================================================================
def _to_lb_df(trials):
    is_by_id = {id(t.graph): t for t in result.is_trials}
    val_by_id = {id(t.graph): t for t in result.val_top}
    rows = []
    for t in trials:
        is_t = is_by_id.get(id(t.graph))
        val_t = val_by_id.get(id(t.graph))
        def _safe(x):
            return x if (x is not None and not math.isinf(x) and not math.isnan(x)) else float("nan")
        rows.append({
            "trigger": f"{reg.STRATEGIES[t.graph.trigger.strategy_key].label} @ {t.graph.trigger.timeframe}",
            "supporters": ", ".join(
                f"{reg.STRATEGIES[s.strategy_key].label}@{s.timeframe}"
                + (f"×{s.weight:.2g}" if s.weight != 1.0 else "")
                for s in t.graph.supporters
            ) or "—",
            "vetoes": ", ".join(
                f"{reg.STRATEGIES[v.strategy_key].label}@{v.timeframe}"
                for v in t.graph.vetoes
            ) or "—",
            "OOS Sharpe": _safe(t.metrics.sharpe),
            "Val Sharpe": _safe(val_t.metrics.sharpe if val_t else None),
            "IS Sharpe":  _safe(is_t.metrics.sharpe if is_t else None),
            "trades": t.metrics.n_trades,
            "return %": t.metrics.return_pct,
            "max DD %": t.metrics.max_drawdown_pct,
            "hit rate": t.metrics.hit_rate,
            "DQ reason": t.metrics.disqualified_reason or "",
        })
    return pd.DataFrame(rows)


if qualified_oos:
    with st.expander("Compact leaderboard (all qualified rows)", expanded=False):
        st.dataframe(
            _to_lb_df(qualified_oos).style.format({
                "OOS Sharpe": "{:.2f}",
                "Val Sharpe": "{:.2f}",
                "IS Sharpe":  "{:.2f}",
                "return %": "{:+.2f}",
                "max DD %": "{:.2f}",
                "hit rate": "{:.1%}",
            }, na_rep="—"),
            width="stretch",
            hide_index=True,
        )

if disqualified_oos:
    with st.expander(f"Disqualified OOS rows ({len(disqualified_oos)}) — "
                     f"why each one was disqualified", expanded=False):
        st.dataframe(
            _to_lb_df(disqualified_oos).style.format({
                "OOS Sharpe": "{:.2f}",
                "Val Sharpe": "{:.2f}",
                "IS Sharpe":  "{:.2f}",
                "return %": "{:+.2f}",
                "max DD %": "{:.2f}",
                "hit rate": "{:.1%}",
            }, na_rep="DQ"),
            width="stretch",
            hide_index=True,
        )


# ====================================================================
#                       NEXT-STEP GUIDANCE
# ====================================================================
with st.expander("What to do with these results", expanded=False):
    st.markdown("""
1. **Top OOS Sharpe row** is your best candidate. Note its trigger
   strategy + supporters + vetoes.
2. **Copy the JSON** from the expander on the card, paste into the main
   backtester (`app.py`), and run a FRESH full backtest. Confirm the
   numbers reproduce — they should.
3. If they do, **try a DIFFERENT asset or date range**. A real edge
   survives changes of context; a fit-to-this-window fluke doesn't.
4. **Run walk-forward** on the candidate to test regime stability.
5. **Only then** consider paper trading.

**Common traps:**

- Top OOS Sharpe is barely positive AND dropped a lot from IS → noise.
  Don't paper-trade it.
- Only ONE trigger strategy dominates the top-10 → may be asset-specific.
- Same strategy across the leaderboard with slightly different params →
  your parameter range is too wide, or you need more validation evidence.
""")
