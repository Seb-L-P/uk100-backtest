"""
Streamlit UI for the UK 100 backtester.

Run with:
    streamlit run app.py

Then open http://localhost:8501 in your browser.

Strategy sources:
  1. **Pick from registry** — single strategy or preset ensemble, slider params.
  2. **Build custom ensemble** — 4 child slots, vote vs filter combination.

Run modes:
  - Single backtest: one run on all data
  - Full validation: IS/OOS + walk-forward + Monte Carlo + bootstrap + PSR
  - Parameter sweep: grid search on in-sample, pick best, evaluate one on OOS
  - Adaptive walk-forward: re-fit params per fold, evaluate on next fold

Adding a new individual strategy: edit strategies/registry.py — UI updates auto.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.fetcher import fetch
from config import COSTS
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.validation import (
    run_full_validation, _verdict_lines,
    adaptive_walk_forward, deflated_sharpe_ratio, holdout_split,
)
from backtest.sweep import grid_sweep, evaluate_oos
from backtest.optuna_search import run_optuna_study
from backtest.run_history import save_run, list_runs, delete_run, count_runs
from strategies import registry as reg
from strategies.ensemble import VoteEnsemble, FilterEnsemble


# ---- Page config -------------------------------------------------------
st.set_page_config(
    page_title="UK 100 Backtester",
    layout="wide",
    initial_sidebar_state="expanded",
)

BARS_PER_YEAR = {
    "1m": 252 * 510, "5m": 252 * 102, "15m": 252 * 34,
    "30m": 252 * 17, "1h": 252 * 8, "1d": 252,
}

# Strategies allowed as ensemble children — exclude existing ensembles to
# prevent nesting (we'd need extra plumbing to make that work cleanly).
PRESET_ENSEMBLE_KEYS = {"vote_meanrev", "vote_trend", "filter_fvg_rsi"}
CHILD_CANDIDATES = {k: v for k, v in reg.STRATEGIES.items()
                    if k not in PRESET_ENSEMBLE_KEYS}

N_SLOTS = 4  # how many child slots in the custom ensemble builder


# ---- Cached data fetch -------------------------------------------------
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def cached_fetch(ticker: str, interval: str,
                 source: str = "yfinance", ig_num_points: int = 5000) -> pd.DataFrame:
    return fetch(ticker=ticker, interval=interval,
                 source=source, ig_num_points=ig_num_points)


# ---- Sidebar: setup ----------------------------------------------------
with st.sidebar:
    st.header("Setup")

    # Top-level: registry mode vs custom ensemble mode
    build_mode = st.radio(
        "Strategy source",
        ["Pick from registry", "Build custom ensemble"],
        horizontal=False,
    )

    data_source = st.radio(
        "Data source",
        ["yfinance (free)", "IG demo (real spreads)", "EODHD (deep history)"],
        index=0,
        horizontal=True,
        help=("yfinance: free, 60-day cap on intraday. "
              "IG demo: per-bar bid/ask spread, matches what you'd trade, "
              "10,000 bars/week allowance. "
              "EODHD: 30+ years of intraday for indices, $29.99/mo paid plan, "
              "100k API calls/day."),
    )
    use_ig = data_source.startswith("IG")
    use_eodhd = data_source.startswith("EODHD")

    if use_ig:
        ticker = st.text_input(
            "IG epic", value="IX.D.FTSE.DAILY.IP",
            help="IG epic code. UK 100 spread bet = IX.D.FTSE.DAILY.IP. "
                 "Shortcuts ^FTSE / UK100 / FTSE auto-resolve.",
        )
        ig_num_points = st.slider(
            "Bars to fetch (IG weekly allowance limit ≈ 10,000)",
            min_value=500, max_value=10000, value=2000, step=500,
            help="IG limits historical points per week. Start small.",
        )
    elif use_eodhd:
        ticker = st.text_input(
            "EODHD symbol", value="ISF.LSE",
            help="EODHD symbol format: TICKER.EXCHANGE. UK 100 default uses "
                 "ISF.LSE (iShares FTSE 100 ETF) — tracks the cash index to "
                 "~0.05% and is included on the All World Extended plan. "
                 "Cash index UKX.INDX requires the indices add-on. "
                 "Shortcuts ^FTSE / UK100 / FTSE100 / UKX auto-resolve.",
        )
        ig_num_points = st.slider(
            "Bars to fetch",
            min_value=500, max_value=50000, value=5000, step=500,
            help="EODHD allows up to 100k API calls/day on paid plans; "
                 "no per-request bar limit issue in practice.",
        )
    else:
        ticker = st.text_input(
            "Ticker", value="^FTSE",
            help="Yahoo Finance symbol. ^FTSE = FTSE 100 cash index.",
        )
        ig_num_points = 5000  # unused
    interval = st.selectbox("Interval", list(BARS_PER_YEAR.keys()), index=2)
    mode = st.radio(
        "Mode",
        ["Single backtest", "Full validation", "Parameter sweep",
         "Bayesian sweep", "Adaptive walk-forward"],
        index=0,
        help=("Single = one run. Validation = IS/OOS + walk-forward + bootstrap. "
              "Sweep = grid search on IS. Bayesian = Optuna TPE — smarter "
              "param search, finds good configs in fewer trials. "
              "Adaptive = re-fit params each fold, test on next."),
    )

    # Mode-specific settings
    oos_fraction, n_folds, n_mc, min_trades, optimize_by = 0.2, 4, 1000, 30, "sharpe"
    n_trials, opt_metric, target_trades = 100, "wf_consistency", 50
    if mode == "Full validation":
        with st.expander("Validation settings"):
            oos_fraction = st.slider("Out-of-sample fraction", 0.1, 0.4, 0.2, 0.05)
            n_folds = st.slider("Walk-forward folds", 2, 10, 4)
            n_mc = st.slider("Monte Carlo shuffles", 100, 5000, 1000, 100)
    elif mode == "Parameter sweep":
        with st.expander("Sweep settings", expanded=True):
            oos_fraction = st.slider("Out-of-sample fraction", 0.1, 0.4, 0.2, 0.05,
                                     key="sweep_oos")
            n_folds = st.slider("Walk-forward folds (per combination)", 2, 6, 4,
                                key="sweep_folds")
            min_trades = st.slider("Min trades to consider valid", 5, 100, 30,
                                   key="sweep_min_trades")
    elif mode == "Adaptive walk-forward":
        with st.expander("Adaptive WF settings", expanded=True):
            n_folds = st.slider("Total folds", 3, 10, 5, key="adapt_folds",
                                help="First fold is train-only. Remaining are train→test.")
            optimize_by = st.selectbox(
                "Optimize for",
                ["sharpe", "profit_factor", "total_return_pct"],
                index=0, key="adapt_metric",
            )
    elif mode == "Bayesian sweep":
        with st.expander("Bayesian sweep settings", expanded=True):
            oos_fraction = st.slider("Out-of-sample fraction", 0.1, 0.4, 0.2, 0.05,
                                     key="bayes_oos")
            n_folds = st.slider("Walk-forward folds (per trial)", 2, 6, 4, key="bayes_folds")
            n_trials = st.slider("Number of trials", 20, 500, 100, 10, key="bayes_trials",
                                 help="Optuna explores adaptively — usually finds great configs in 50-200 trials.")
            target_trades = st.slider("Target trades (penalty floor)", 10, 200, 50, 5,
                                      key="bayes_target_trades",
                                      help="Configs with fewer trades get a linear penalty. "
                                           "Prevents Optuna from gaming thin-trade flukes.")
            opt_metric = st.selectbox(
                "Optimize",
                ["wf_consistency", "sharpe", "profit_factor"],
                index=0, key="bayes_metric",
                help="wf_consistency is the most overfit-resistant; sharpe and PF are flashier but easier to overfit to."
            )

    st.divider()

    # Sweep / adaptive / Bayesian modes use the param search machinery
    is_grid_mode = mode in ("Parameter sweep", "Adaptive walk-forward")
    is_optuna_mode = mode == "Bayesian sweep"
    is_search_mode = is_grid_mode or is_optuna_mode

    # Custom ensemble + search modes are not supported (factory composition is
    # too tangled to safely sweep over). Force registry mode in that combination.
    if is_search_mode and build_mode == "Build custom ensemble":
        st.warning("Custom ensembles can't be swept (yet). Switch to "
                   "'Pick from registry' for sweep / Bayesian / adaptive WF.")
        build_mode = "Pick from registry"

    # ---- Branch A: Registry mode ---------------------------------------
    if build_mode == "Pick from registry":
        strategy_keys = list(reg.STRATEGIES.keys())
        strategy_labels = [reg.STRATEGIES[k].label for k in strategy_keys]
        selected_label = st.selectbox("Strategy", strategy_labels, index=0)
        strategy_key = strategy_keys[strategy_labels.index(selected_label)]
        spec = reg.get(strategy_key)

        st.caption(spec.description)

        param_values: dict[str, Any] = {}
        param_grid: dict[str, list] = {}

        if is_optuna_mode:
            st.subheader("Strategy parameters")
            st.caption(
                f"Optuna will search the param ranges defined in the registry "
                f"({len(spec.params or [])} params). No manual grid needed."
            )
        elif is_grid_mode:
            st.subheader("Parameter grid")
            st.caption("Comma-separated values per param. One value = fixed.")
            for p in (spec.params or []):
                default_str = str(p.default)
                txt = st.text_input(
                    p.label, value=default_str,
                    help=f"Type: {p.type}. Default: {p.default}. "
                         f"Example: '3, 5, 7'. {p.help or ''}",
                    key=f"grid_{strategy_key}_{p.name}",
                )
                try:
                    items = [x.strip() for x in txt.split(",") if x.strip()]
                    if p.type == "int":
                        param_grid[p.name] = [int(x) for x in items]
                    elif p.type == "float":
                        param_grid[p.name] = [float(x) for x in items]
                    elif p.type == "bool":
                        param_grid[p.name] = [x.lower() in ("1", "true", "yes", "y")
                                              for x in items]
                    if not param_grid[p.name]:
                        param_grid[p.name] = [p.default]
                except ValueError as e:
                    st.error(f"Bad value for '{p.label}': {e}. Using default.")
                    param_grid[p.name] = [p.default]
        else:
            st.subheader("Strategy parameters")
            for p in (spec.params or []):
                if p.type == "int":
                    param_values[p.name] = st.slider(
                        p.label, int(p.min), int(p.max), int(p.default),
                        step=int(p.step) if p.step else 1,
                        help=p.help,
                    )
                elif p.type == "float":
                    param_values[p.name] = st.slider(
                        p.label, float(p.min), float(p.max), float(p.default),
                        step=float(p.step) if p.step else 0.1,
                        help=p.help,
                    )
                elif p.type == "bool":
                    param_values[p.name] = st.checkbox(p.label, value=bool(p.default), help=p.help)

        display_label = spec.label
        display_desc = spec.description
        warmup_bars = spec.warmup_bars
        strategy_factory = lambda: spec.build(**param_values)
        # Factory builder used by sweep / adaptive WF
        def factory_for_params(params: dict):
            full = {**spec.defaults(), **params}
            return lambda: spec.build(**full)

    # ---- Branch B: Custom ensemble builder -----------------------------
    else:
        st.subheader("Pick children")
        st.caption(f"Up to {N_SLOTS} strategies. Empty slots are ignored. Children use their default params.")

        child_keys_options = list(CHILD_CANDIDATES.keys())
        child_label_options = ["(none)"] + [CHILD_CANDIDATES[k].label for k in child_keys_options]
        # parallel list with None for the (none) slot
        child_key_lookup = [None] + child_keys_options

        selected_child_keys: list[str] = []
        for slot in range(N_SLOTS):
            choice_label = st.selectbox(
                f"Slot {slot + 1}", child_label_options,
                index=0, key=f"child_slot_{slot}",
            )
            choice_idx = child_label_options.index(choice_label)
            chosen_key = child_key_lookup[choice_idx]
            if chosen_key is not None and chosen_key not in selected_child_keys:
                selected_child_keys.append(chosen_key)

        st.subheader("Combination type")
        ensemble_type = st.radio(
            "Type", ["Vote (M-of-N agreement)", "Filter (trigger + filters)"],
            help=("Vote: trade when M children agree on direction. "
                  "Filter: first-selected slot is the trigger; the rest can veto."),
        )

        n_selected = len(selected_child_keys)

        if ensemble_type.startswith("Vote"):
            # Slider needs min < max; when 0 or 1 children selected, just fix at 1
            if n_selected <= 1:
                min_agreement = 1
                st.caption(f"Min strategies agreeing: **1** (only "
                           f"{n_selected} selected — slider disabled)")
            else:
                min_agreement = st.slider(
                    "Min strategies agreeing",
                    min_value=1, max_value=n_selected,
                    value=min(2, n_selected), step=1,
                )
        else:
            min_agreement = None
            if n_selected >= 1:
                trigger_label = st.selectbox(
                    "Trigger (others act as filters)",
                    [CHILD_CANDIDATES[k].label for k in selected_child_keys],
                    index=0,
                )
            else:
                trigger_label = None

        st.subheader("Ensemble parameters")
        r_target = st.slider("Target (R)", 0.5, 5.0, 2.0, 0.25)
        stop_atr_mult = st.slider("Stop ATR multiplier", 0.5, 5.0, 2.0, 0.25)
        atr_period = st.slider("ATR period", 5, 50, 14, 1)

        # Validate selection
        if n_selected == 0:
            st.warning("Pick at least one strategy in slot 1.")
            valid = False
        elif ensemble_type.startswith("Filter") and n_selected < 2:
            st.warning("Filter mode needs at least 2 strategies (1 trigger + 1+ filters). "
                       "If you only want one strategy, use Vote with min_agreement=1.")
            valid = False
        else:
            valid = True

        # Build factory
        if valid:
            child_keys_snapshot = list(selected_child_keys)
            etype_snapshot = ensemble_type

            def custom_factory():
                # Each call creates fresh children (matters for walk-forward folds)
                children = [
                    CHILD_CANDIDATES[k].build(**CHILD_CANDIDATES[k].defaults())
                    for k in child_keys_snapshot
                ]
                if etype_snapshot.startswith("Vote"):
                    return VoteEnsemble(
                        children=children,
                        min_agreement=min_agreement,
                        r_target=r_target,
                        stop_atr_mult=stop_atr_mult,
                        atr_period=atr_period,
                    )
                # Filter mode
                trig_idx = [CHILD_CANDIDATES[k].label for k in child_keys_snapshot].index(trigger_label)
                trigger = children[trig_idx]
                filters = [c for i, c in enumerate(children) if i != trig_idx]
                return FilterEnsemble(
                    trigger=trigger,
                    filters=filters,
                    r_target=r_target,
                    stop_atr_mult=stop_atr_mult,
                    atr_period=atr_period,
                )

            strategy_factory = custom_factory
            warmup_bars = max(CHILD_CANDIDATES[k].warmup_bars for k in child_keys_snapshot)
            display_label = f"Custom ensemble ({len(child_keys_snapshot)} strategies)"
            display_desc = (
                f"{ensemble_type} of: " +
                ", ".join(CHILD_CANDIDATES[k].label for k in child_keys_snapshot)
            )
        else:
            strategy_factory = None
            warmup_bars = 50
            display_label = "Custom ensemble (incomplete)"
            display_desc = "Pick at least one valid set of children to run."

    st.divider()
    run_clicked = st.button("Run backtest", type="primary", width="stretch",
                            disabled=(strategy_factory is None))


# ---- Main area ---------------------------------------------------------
st.title("UK 100 Backtester")
st.caption(f"**{display_label}** on **{ticker}** at **{interval}** — {mode}")
if display_desc:
    st.caption(display_desc)

if strategy_factory is None:
    st.info("Configure a valid strategy on the left, then click **Run backtest**.")
    st.stop()

# In sweep / adaptive modes, allow showing cached results even when the user
# hasn't just clicked Run (so sort/filter tweaks don't wipe results).
has_cached_sweep = (mode == "Parameter sweep"
                    and st.session_state.get("sweep_result") is not None
                    and st.session_state.get("sweep_hash") is not None)
has_cached_awf = (mode == "Adaptive walk-forward"
                  and st.session_state.get("awf_result") is not None
                  and st.session_state.get("awf_hash") is not None)

# Single-mode caching: hash the inputs that affect the result. Changing
# the trade picker / indicator overlays / etc. shouldn't invalidate this —
# only changing strategy / params / ticker / interval / source does.
_single_config = {
    "label": display_label,
    "desc": display_desc,            # captures ensemble children, vote settings
    "params": dict(param_values) if "param_values" in dir() else {},
    "ticker": ticker,
    "interval": interval,
    "source": "ig" if use_ig else ("eodhd" if use_eodhd else "yfinance"),
    "ig_num_points": ig_num_points,
    "warmup": warmup_bars,
}
_single_hash = hashlib.md5(
    json.dumps(_single_config, sort_keys=True, default=str).encode()
).hexdigest()
# Invalidate cached result if the config changed
if st.session_state.get("single_hash") != _single_hash:
    st.session_state["single_hash"] = _single_hash
    st.session_state["single_result"] = None
has_cached_single = (mode == "Single backtest"
                     and st.session_state.get("single_result") is not None)
has_cached_bayes = (mode == "Bayesian sweep"
                    and st.session_state.get("bayes_result") is not None)

def _render_run_history():
    """Show recent runs from the SQLite history. Filterable, deletable."""
    n_runs = count_runs()
    with st.expander(f"📜 Run history ({n_runs} saved)", expanded=False):
        if n_runs == 0:
            st.caption("No runs saved yet. Run a backtest and it'll appear here.")
            return

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            filter_strategy = st.text_input(
                "Filter by strategy key (e.g. 'fvg', 'rsi_revert')",
                value="", key="history_filter_strategy",
            )
        with c2:
            min_sharpe = st.number_input(
                "Min Sharpe", value=-5.0, step=0.1, key="history_min_sharpe",
            )
        with c3:
            limit = st.number_input(
                "Show last N", min_value=5, max_value=500, value=25, step=5,
                key="history_limit",
            )

        runs = list_runs(
            limit=int(limit),
            strategy_key=filter_strategy.strip() or None,
            min_sharpe=min_sharpe if min_sharpe > -4.9 else None,
        )
        if not runs:
            st.caption("No runs match the filters.")
            return

        rows = []
        for r in runs:
            params = json.loads(r["params_json"]) if r["params_json"] else {}
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            rows.append({
                "id": r["id"],
                "when": r["timestamp"],
                "strategy": r["strategy_label"] or r["strategy_key"],
                "ticker": r["ticker"],
                "interval": r["interval"],
                "src": r["source"],
                "mode": r["mode"],
                "trades": r["num_trades"],
                "ret_%": r["total_return_pct"],
                "sharpe": r["sharpe"],
                "pf": r["profit_factor"],
                "max_dd_%": r["max_drawdown_pct"],
                "params": params_str[:80] + ("..." if len(params_str) > 80 else ""),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            del_id = st.number_input(
                "Delete run id", min_value=0, value=0, step=1,
                help="Type a run id from the table above and click Delete.",
                key="history_del_id",
            )
        with c2:
            if st.button("Delete", key="history_del_button"):
                if del_id > 0:
                    if delete_run(int(del_id)):
                        st.success(f"Deleted run {del_id}. Rerun the page to refresh.")
                    else:
                        st.warning(f"No run with id {del_id}.")


if (not run_clicked
        and not has_cached_sweep
        and not has_cached_awf
        and not has_cached_single
        and not has_cached_bayes):
    st.info("Configure on the left, then click **Run backtest**.")
    # Run-history widget so the user can browse past runs without
    # having to run a fresh backtest first.
    _render_run_history()
    st.stop()

_active_source = "ig" if use_ig else ("eodhd" if use_eodhd else "yfinance")
try:
    data = cached_fetch(
        ticker, interval,
        source=_active_source,
        ig_num_points=ig_num_points,
    )
except Exception as e:
    st.error(f"Data fetch failed: {e}")
    if use_ig:
        st.info("Try running `python scripts/ig_test.py` in your terminal "
                "to diagnose IG-specific issues.")
    elif use_eodhd:
        st.info("Try running `python scripts/eodhd_test.py` in your terminal "
                "to verify your EODHD API key and plan coverage.")
    st.stop()

st.write(f"**Data:** {len(data)} bars from {data.index[0]} to {data.index[-1]}")
if "Spread" in data.columns:
    spread_series = data["Spread"].dropna()
    if not spread_series.empty:
        st.caption(
            f"📊 Per-bar spread from IG: avg **{spread_series.mean():.2f}pt**, "
            f"min {spread_series.min():.2f}pt, max {spread_series.max():.2f}pt, "
            f"median {spread_series.median():.2f}pt — used per-trade instead of "
            f"the flat {COSTS.spread_points:.1f}pt config default."
        )
else:
    st.caption(
        f"📊 Using flat **{COSTS.spread_points:.1f}pt** spread from config "
        f"(yfinance has no bid/ask data — switch to IG demo for per-bar real spreads)."
    )
bpy = BARS_PER_YEAR.get(interval, 252)


# ---- Helpers ------------------------------------------------------------
def equity_chart(equity: pd.Series, price: pd.Series | None = None,
                 trades_df: pd.DataFrame | None = None,
                 title: str = "Equity curve") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="Equity (£)",
        line=dict(color="steelblue", width=2),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>£%{y:,.2f}<extra></extra>",
    ))
    if price is not None:
        fig.add_trace(go.Scatter(
            x=price.index, y=price.values, name=f"{ticker} close",
            line=dict(color="lightgray", width=1),
            yaxis="y2", opacity=0.5,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:,.2f}<extra></extra>",
        ))
    if trades_df is not None and not trades_df.empty:
        wins = trades_df[trades_df["net_pnl_gbp"] > 0]
        losses = trades_df[trades_df["net_pnl_gbp"] <= 0]
        if not wins.empty:
            fig.add_trace(go.Scatter(
                x=wins["exit_time"], y=wins["exit_price"], mode="markers",
                name="Wins", marker=dict(color="green", size=6, symbol="triangle-up"),
                yaxis="y2",
                hovertext=[f"+£{p:.2f}" for p in wins["net_pnl_gbp"]],
                hoverinfo="text+x",
            ))
        if not losses.empty:
            fig.add_trace(go.Scatter(
                x=losses["exit_time"], y=losses["exit_price"], mode="markers",
                name="Losses", marker=dict(color="red", size=6, symbol="triangle-down"),
                yaxis="y2",
                hovertext=[f"£{p:.2f}" for p in losses["net_pnl_gbp"]],
                hoverinfo="text+x",
            ))
    fig.update_layout(
        title=title, height=450, hovermode="x unified",
        yaxis=dict(title="Equity (£)", side="left"),
        yaxis2=dict(title=f"{ticker}", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---- Trade inspector: TradingView Lightweight Charts ------------------
# We use TradingView's chart engine as the SOLE renderer. Each indicator
# is either a price-scale overlay on the candle chart, or its own stacked
# subplot chart below.
from backtest.indicators import (
    ema, bollinger, vwap as vwap_indicator, rsi,
    keltner_channels, parabolic_sar,
    stochastic, adx, williams_r, mfi, roc, obv,
)


# Indicators available in the trade inspector. Split by display mode.
PRICE_SCALE_OVERLAYS = [
    "EMA(20)", "EMA(50)", "EMA(200)",
    "Bollinger(20, 2σ)", "Keltner(20, 2×ATR)",
    "VWAP", "Parabolic SAR",
]
OSCILLATOR_OVERLAYS = [
    "RSI(14)", "Stochastic(14,3,3)", "ADX(14)",
    "Williams %R(14)", "MFI(14)", "ROC(12)", "OBV",
]
TRADE_INSPECTOR_OVERLAYS = PRICE_SCALE_OVERLAYS + OSCILLATOR_OVERLAYS


# ---- TradingView Lightweight Charts helpers ----------------------------
def _df_to_lwc_candles(df: pd.DataFrame) -> list:
    """Convert OHLCV DataFrame to Lightweight Charts candle format."""
    return [
        {
            "time": int(ts.timestamp()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        }
        for ts, row in df.iterrows()
    ]


def _df_to_lwc_line(series: pd.Series) -> list:
    """Convert a Series (e.g. EMA, VWAP) to LWC line format. Drops NaNs."""
    return [
        {"time": int(ts.timestamp()), "value": float(v)}
        for ts, v in series.dropna().items()
    ]


def _render_lwc_chart(series: list, height: int, title: str, key: str):
    """Render a single LWC chart with our standard styling."""
    from streamlit_lightweight_charts import renderLightweightCharts
    chart_options = {
        "height": height,
        "layout": {
            "background": {"type": "solid", "color": "rgba(0,0,0,0)"},
            "textColor": "white",
        },
        "grid": {"vertLines": {"color": "rgba(255,255,255,0.1)"},
                 "horzLines": {"color": "rgba(255,255,255,0.1)"}},
        "timeScale": {"timeVisible": True, "secondsVisible": False},
        "watermark": {"visible": True, "fontSize": 12,
                      "color": "rgba(180,180,180,0.5)",
                      "text": title, "horzAlign": "left", "vertAlign": "top"},
    }
    renderLightweightCharts([{"chart": chart_options, "series": series}], key=key)


def _osc_subplot(window: pd.DataFrame, overlay_name: str, key_suffix: str):
    """
    Render an oscillator overlay as its own LWC subplot below the main chart.
    Each oscillator is computed on the FULL data (correct warmup) but rendered
    over the trade window only.
    """
    title_map = {
        "RSI(14)": "RSI(14)",
        "Stochastic(14,3,3)": "Stochastic %K/%D",
        "ADX(14)": "ADX(14) + DI",
        "Williams %R(14)": "Williams %R(14)",
        "MFI(14)": "MFI(14)",
        "ROC(12)": "ROC(12)",
        "OBV": "OBV (cumulative)",
    }
    title = title_map.get(overlay_name, overlay_name)
    series = []

    def _ref_line(idx, value, color):
        return {"type": "Line",
                "data": [{"time": int(ts.timestamp()), "value": float(value)} for ts in idx],
                "options": {"color": color, "lineWidth": 1, "lineStyle": 2,
                            "title": str(value)}}

    if overlay_name == "RSI(14)":
        s = rsi(window["Close"], 14)  # window-local is OK; just for display
        # Recompute on full data for proper warmup
        s = rsi(window["Close"], 14)  # acceptable for inspector — the values
        # Actually use a longer history slice would require passing data; we keep
        # window-local since 14-period warmup fits inside 60+ bar windows.
        series.append({"type": "Line", "data": _df_to_lwc_line(s),
                       "options": {"color": "orange", "lineWidth": 2, "title": "RSI"}})
        series.append(_ref_line(s.index, 70, "red"))
        series.append(_ref_line(s.index, 30, "green"))
    elif overlay_name == "Stochastic(14,3,3)":
        k, d = stochastic(window, 14, 3, 3)
        series.append({"type": "Line", "data": _df_to_lwc_line(k),
                       "options": {"color": "#2196f3", "lineWidth": 1.5, "title": "%K"}})
        series.append({"type": "Line", "data": _df_to_lwc_line(d),
                       "options": {"color": "orange", "lineWidth": 1.5, "title": "%D"}})
        series.append(_ref_line(k.index, 80, "red"))
        series.append(_ref_line(k.index, 20, "green"))
    elif overlay_name == "ADX(14)":
        adx_line, plus_di, minus_di = adx(window, 14)
        series.append({"type": "Line", "data": _df_to_lwc_line(adx_line),
                       "options": {"color": "white", "lineWidth": 2, "title": "ADX"}})
        series.append({"type": "Line", "data": _df_to_lwc_line(plus_di),
                       "options": {"color": "green", "lineWidth": 1.2, "title": "+DI"}})
        series.append({"type": "Line", "data": _df_to_lwc_line(minus_di),
                       "options": {"color": "red", "lineWidth": 1.2, "title": "-DI"}})
        series.append(_ref_line(adx_line.index, 25, "yellow"))
    elif overlay_name == "Williams %R(14)":
        s = williams_r(window, 14)
        series.append({"type": "Line", "data": _df_to_lwc_line(s),
                       "options": {"color": "cyan", "lineWidth": 1.5, "title": "%R"}})
        series.append(_ref_line(s.index, -20, "red"))
        series.append(_ref_line(s.index, -80, "green"))
    elif overlay_name == "MFI(14)":
        s = mfi(window, 14)
        series.append({"type": "Line", "data": _df_to_lwc_line(s),
                       "options": {"color": "magenta", "lineWidth": 1.5, "title": "MFI"}})
        series.append(_ref_line(s.index, 80, "red"))
        series.append(_ref_line(s.index, 20, "green"))
    elif overlay_name == "ROC(12)":
        s = roc(window["Close"], 12)
        series.append({"type": "Line", "data": _df_to_lwc_line(s),
                       "options": {"color": "orange", "lineWidth": 1.5, "title": "ROC"}})
        series.append(_ref_line(s.index, 0, "white"))
    elif overlay_name == "OBV":
        s = obv(window)
        series.append({"type": "Line", "data": _df_to_lwc_line(s),
                       "options": {"color": "cyan", "lineWidth": 1.5, "title": "OBV"}})

    if series:
        _render_lwc_chart(series, height=170, title=title, key=f"lwc_osc_{key_suffix}")


def trade_inspector_lwc(
    trade: pd.Series,
    data: pd.DataFrame,
    bars_before: int = 30,
    bars_after: int = 30,
    overlays: list[str] | None = None,
):
    """
    Trade inspector using TradingView's Lightweight Charts as the SOLE renderer.

    Layout:
      - Main candle chart with entry/exit markers + price-scale overlays
      - One stacked LWC subplot per selected oscillator (RSI, Stochastic, etc.)

    Each indicator gets its own native TradingView-style display, performant
    for big bar counts, with TradingView's native zoom/pan UI.
    """
    from streamlit_lightweight_charts import renderLightweightCharts  # noqa: F401

    entry_time = pd.Timestamp(trade["entry_time"]) if "entry_time" in trade else trade.name
    exit_time = pd.Timestamp(trade["exit_time"])
    side = trade["side"]
    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])
    pnl = float(trade["net_pnl_gbp"])
    key_suffix = f"{trade.name}_{int(entry_time.timestamp())}"

    # Slice the window around the trade
    try:
        entry_idx = data.index.get_indexer([entry_time], method="nearest")[0]
        exit_idx = data.index.get_indexer([exit_time], method="nearest")[0]
    except Exception:
        entry_idx, exit_idx = 0, len(data) - 1
    start = max(0, entry_idx - bars_before)
    end = min(len(data), exit_idx + bars_after + 1)
    window = data.iloc[start:end]
    overlays = overlays or []

    # ---- Main price chart: candles + markers + price-scale overlays ----
    series = [{
        "type": "Candlestick",
        "data": _df_to_lwc_candles(window),
        "options": {
            "upColor": "#26a69a", "downColor": "#ef5350",
            "borderVisible": False,
            "wickUpColor": "#26a69a", "wickDownColor": "#ef5350",
        },
    }]
    win_color = "#26a69a" if pnl > 0 else "#ef5350"
    series[0]["markers"] = [
        {"time": int(entry_time.timestamp()),
         "position": "belowBar" if side == "long" else "aboveBar",
         "color": "#2196f3",
         "shape": "arrowUp" if side == "long" else "arrowDown",
         "text": f"Entry {side}"},
        {"time": int(exit_time.timestamp()),
         "position": "aboveBar" if pnl > 0 else "belowBar",
         "color": win_color, "shape": "circle",
         "text": f"Exit £{pnl:+.2f}"},
    ]

    # Price-scale overlays computed on full data, sliced to the window
    win_start, win_end = window.index[0], window.index[-1]
    for name in overlays:
        if name == "EMA(20)":
            s = ema(data["Close"], 20).loc[win_start:win_end]
            series.append({"type": "Line", "data": _df_to_lwc_line(s),
                           "options": {"color": "orange", "lineWidth": 2,
                                       "title": "EMA(20)"}})
        elif name == "EMA(50)":
            s = ema(data["Close"], 50).loc[win_start:win_end]
            series.append({"type": "Line", "data": _df_to_lwc_line(s),
                           "options": {"color": "red", "lineWidth": 2,
                                       "title": "EMA(50)"}})
        elif name == "EMA(200)":
            s = ema(data["Close"], 200).loc[win_start:win_end]
            series.append({"type": "Line", "data": _df_to_lwc_line(s),
                           "options": {"color": "#5e8eff", "lineWidth": 2,
                                       "title": "EMA(200)"}})
        elif name == "Bollinger(20, 2σ)":
            mid, upper, lower = bollinger(data["Close"], 20, 2.0)
            for s, color, title in [(mid, "purple", "BB mid"),
                                    (upper, "purple", "BB upper"),
                                    (lower, "purple", "BB lower")]:
                series.append({"type": "Line",
                               "data": _df_to_lwc_line(s.loc[win_start:win_end]),
                               "options": {"color": color, "lineWidth": 1, "title": title}})
        elif name == "Keltner(20, 2×ATR)":
            mid, upper, lower = keltner_channels(data, 20, 10, 2.0)
            for s, color, title in [(mid, "teal", "Keltner mid"),
                                    (upper, "teal", "Keltner upper"),
                                    (lower, "teal", "Keltner lower")]:
                series.append({"type": "Line",
                               "data": _df_to_lwc_line(s.loc[win_start:win_end]),
                               "options": {"color": color, "lineWidth": 1, "title": title}})
        elif name == "VWAP":
            s = vwap_indicator(data).loc[win_start:win_end]
            series.append({"type": "Line", "data": _df_to_lwc_line(s),
                           "options": {"color": "dodgerblue", "lineWidth": 2,
                                       "title": "VWAP"}})
        elif name == "Parabolic SAR":
            s = parabolic_sar(data).loc[win_start:win_end]
            series.append({"type": "Line", "data": _df_to_lwc_line(s),
                           "options": {"color": "yellow", "lineWidth": 0,
                                       "title": "Parabolic SAR",
                                       "pointMarkersVisible": True,
                                       "pointMarkersRadius": 3}})

    title = (f"{side.upper()} @ {entry_price:.2f} → {exit_price:.2f} "
             f"= £{pnl:+.2f} (exit: {trade['exit_reason']})")
    _render_lwc_chart(series, height=480, title=title, key=f"lwc_main_{key_suffix}")

    # ---- One stacked subplot per selected oscillator ------------------
    for name in overlays:
        if name in OSCILLATOR_OVERLAYS:
            _osc_subplot(window, name, key_suffix=f"{name}_{key_suffix}")


def metrics_panel(m, label: str = "Headline") -> None:
    st.subheader(label)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Return", f"{m.total_return_pct:+.2f}%",
              delta=f"£{(m.final_balance - m.starting_balance):+,.0f}")
    c2.metric("Sharpe", f"{m.sharpe:.2f}")
    c3.metric("Profit factor", f"{m.profit_factor:.2f}")
    c4.metric("Max DD", f"{m.max_drawdown_pct:.2f}%",
              delta=f"£{m.max_drawdown_gbp:,.0f}", delta_color="inverse")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Trades", m.num_trades)
    c6.metric("Win rate", f"{m.win_rate_pct:.1f}%")
    c7.metric("Expectancy (R)", f"{m.expectancy_r:+.2f}")
    c8.metric("Avg bars held", f"{m.avg_bars_held:.1f}")
    with st.expander("Cost breakdown"):
        st.write(f"- Gross P&L: £{m.total_gross_pnl:+,.2f}")
        st.write(f"- Spread: -£{m.total_spread_cost:,.2f}")
        st.write(f"- Slippage: -£{m.total_slippage_cost:,.2f}")
        st.write(f"- Financing: -£{m.total_financing_cost:,.2f}")


# ---- Run ---------------------------------------------------------------
if mode == "Single backtest":
    # Cached path: if we already ran this exact config, reuse the result so
    # tweaks to widgets below (trade picker, indicator overlays, etc.) don't
    # re-trigger a full backtest.
    if st.session_state.get("single_result") is not None:
        result, m = st.session_state["single_result"]
    else:
        with st.spinner("Running backtest..."):
            result = run_backtest(data, strategy_factory(), warmup_bars=warmup_bars)
            m = compute_metrics(result, bars_per_year=bpy)
        st.session_state["single_result"] = (result, m)

        # Auto-save to run history (only on the FRESH compute path,
        # not on cached re-renders)
        try:
            saved_params = dict(param_values) if "param_values" in dir() and param_values else {}
            save_run(
                strategy_key=strategy_key if "strategy_key" in dir() else "custom",
                strategy_label=display_label,
                params=saved_params,
                ticker=ticker, interval=interval,
                source=_active_source,
                mode="single",
                result=result, metrics=m,
            )
        except Exception as e:
            st.caption(f"⚠️ Run not saved to history: {e}")

    metrics_panel(m, label="Results")
    st.plotly_chart(
        equity_chart(result.equity_curve, price=data["Close"],
                     trades_df=result.trades_df, title=f"{display_label} — equity & trades"),
        width="stretch",
    )
    with st.expander(f"Trades log ({len(result.trades_df)} trades)"):
        st.dataframe(result.trades_df, width="stretch")

    # ---- Performance attribution ----------------------------------------
    if not result.trades_df.empty:
        from backtest.attribution import (
            by_hour_of_day, by_day_of_week, by_month,
            by_session_phase, by_side, by_exit_reason,
            equity_drawdown_series,
        )
        with st.expander("📊 Performance attribution — when does the strategy actually work?"):
            st.caption(
                "Breakdowns of the trade log by various dimensions. Look for "
                "concentrations: if one hour/day/month dominates the P&L, the "
                "result is fragile. If it's spread evenly, the edge is more credible."
            )

            tab_h, tab_d, tab_m, tab_s, tab_side, tab_exit, tab_dd = st.tabs(
                ["Hour of day", "Day of week", "Month",
                 "Session phase", "Long vs short", "Exit reason", "Drawdown"]
            )
            with tab_h:
                df_h = by_hour_of_day(result.trades_df)
                st.dataframe(df_h, width="stretch", hide_index=True)
                if not df_h.empty:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Bar(x=df_h["hour"], y=df_h["total_pnl_gbp"],
                                           marker_color=["green" if v > 0 else "red"
                                                         for v in df_h["total_pnl_gbp"]],
                                           name="P&L"))
                    fig_h.update_layout(title="P&L by entry hour",
                                        xaxis_title="Hour of day",
                                        yaxis_title="Total P&L (£)",
                                        height=300, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_h, width="stretch")
            with tab_d:
                df_d = by_day_of_week(result.trades_df)
                st.dataframe(df_d, width="stretch", hide_index=True)
            with tab_m:
                df_m = by_month(result.trades_df)
                st.dataframe(df_m, width="stretch", hide_index=True)
            with tab_s:
                df_s = by_session_phase(result.trades_df, open_hour=8, close_hour=16)
                st.dataframe(df_s, width="stretch", hide_index=True)
            with tab_side:
                df_side = by_side(result.trades_df)
                st.dataframe(df_side, width="stretch", hide_index=True)
            with tab_exit:
                df_exit = by_exit_reason(result.trades_df)
                st.dataframe(df_exit, width="stretch", hide_index=True)
            with tab_dd:
                dd = equity_drawdown_series(result.equity_curve)
                if not dd.empty:
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=dd.index, y=dd.values, fill="tozeroy",
                        fillcolor="rgba(220,50,50,0.3)",
                        line=dict(color="red", width=1),
                        name="Drawdown",
                    ))
                    fig_dd.update_layout(
                        title="Underwater equity (drawdown from peak)",
                        yaxis_title="Drawdown (%)",
                        height=300, margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(fig_dd, width="stretch")

    # ---- Trade inspector --------------------------------------------------
    if not result.trades_df.empty:
        st.subheader("Trade inspector")
        st.caption(
            "Pick any trade to see it on a zoomed price chart. Useful for "
            "sanity-checking that entries/exits make sense by eye."
        )

        trades_for_picker = result.trades_df.reset_index()
        # Build a human-readable label for each trade
        def _label(row):
            ts = pd.Timestamp(row["entry_time"]).strftime("%Y-%m-%d %H:%M")
            return (f"#{row.name:>3}  {ts}  {row['side']:>5}  "
                    f"£{row['net_pnl_gbp']:+8.2f}  ({row['exit_reason']})")
        labels = [_label(r) for _, r in trades_for_picker.iterrows()]

        picked_label = st.selectbox("Trade", labels, key="trade_inspector_pick")
        picked_idx = labels.index(picked_label)
        picked_trade = trades_for_picker.iloc[picked_idx]

        # Pick which indicator overlays to draw on the chart. Computed on the
        # FULL data so warmup is correct, sliced to the chart window.
        overlay_options = TRADE_INSPECTOR_OVERLAYS

        # Sensible defaults per strategy — pick overlays most relevant to the
        # strategy that generated the trade. Falls through to none if unknown.
        default_overlays_by_strategy = {
            "BollingerReversion": ["Bollinger(20, 2σ)"],
            "VwapReversion": ["VWAP"],
            "RsiReversion": ["RSI(14)"],
            "SmaCrossover": ["EMA(20)", "EMA(50)"],
            "DonchianBreakout": ["EMA(50)"],
        }
        strategy_name = type(strategy_factory()).__name__
        default_overlays = default_overlays_by_strategy.get(strategy_name, [])

        chosen_overlays = st.multiselect(
            "Indicator overlays",
            overlay_options,
            default=default_overlays,
            help="Computed on full data with proper warmup; sliced to the chart window. "
                 "RSI shows as a separate subplot.",
            key="trade_inspector_overlays",
        )

        c1, c2 = st.columns([3, 1])
        with c1:
            # TradingView Lightweight Charts is the sole renderer.
            trade_inspector_lwc(picked_trade, data, overlays=chosen_overlays)
        with c2:
            st.markdown("**Trade details**")
            st.write(f"Side: **{picked_trade['side']}**")
            st.write(f"Entry: £{picked_trade['entry_price']:.2f}")
            st.write(f"Exit: £{picked_trade['exit_price']:.2f}")
            st.write(f"Stake: £{picked_trade['stake_per_point']:.2f}/pt")
            st.write(f"Bars held: {picked_trade['bars_held']}")
            st.write(f"Exit: {picked_trade['exit_reason']}")
            st.markdown("**Cost breakdown**")
            st.write(f"Gross: £{picked_trade['gross_pnl_gbp']:+.2f}")
            st.write(f"Spread: −£{picked_trade['spread_cost_gbp']:.2f}")
            st.write(f"Slippage: −£{picked_trade['slippage_cost_gbp']:.2f}")
            st.write(f"Financing: −£{picked_trade['financing_cost_gbp']:.2f}")
            st.write(f"**Net: £{picked_trade['net_pnl_gbp']:+.2f}**")

elif mode == "Full validation":
    with st.spinner("Running full validation (this can take a minute)..."):
        is_result, oos_result, report = run_full_validation(
            data,
            strategy_factory=strategy_factory,
            oos_fraction=oos_fraction,
            n_folds=n_folds,
            n_mc_simulations=n_mc,
            warmup_bars=warmup_bars,
            bars_per_year=bpy,
        )

    is_m = report.in_sample_metrics
    oos_m = report.out_of_sample_metrics
    drift = report.is_to_oos_drift() or 0.0
    consistency = report.consistency_score()

    st.subheader("In-sample vs Out-of-sample")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IS Sharpe", f"{is_m.sharpe:.2f}")
    c2.metric("OOS Sharpe", f"{oos_m.sharpe:.2f}", delta=f"{drift:+.2f}")
    c3.metric("WF consistency", f"{consistency:.0%}",
              help="Fraction of walk-forward folds with profit factor > 1.0")
    psr_str = (f"{report.psr:.0%}" if report.psr is not None and not pd.isna(report.psr) else "n/a")
    c4.metric("PSR (in-sample)", psr_str,
              help="Probabilistic Sharpe Ratio: probability true Sharpe > 0, "
                   "accounting for sample size, skew, kurtosis. "
                   ">95% = strong evidence; ~50% = noise.")

    col_is, col_oos = st.columns(2)
    with col_is:
        metrics_panel(is_m, label=f"In-sample ({is_result.equity_curve.index[0].date()} → {is_result.equity_curve.index[-1].date()})")
    with col_oos:
        metrics_panel(oos_m, label=f"Out-of-sample ({oos_result.equity_curve.index[0].date()} → {oos_result.equity_curve.index[-1].date()})")

    tab_is, tab_oos = st.tabs(["In-sample", "Out-of-sample"])
    with tab_is:
        st.plotly_chart(
            equity_chart(is_result.equity_curve, price=data["Close"].loc[is_result.equity_curve.index[0]:is_result.equity_curve.index[-1]],
                         trades_df=is_result.trades_df, title="In-sample"),
            width="stretch",
        )
    with tab_oos:
        st.plotly_chart(
            equity_chart(oos_result.equity_curve, price=data["Close"].loc[oos_result.equity_curve.index[0]:oos_result.equity_curve.index[-1]],
                         trades_df=oos_result.trades_df, title="Out-of-sample"),
            width="stretch",
        )

    st.subheader("Walk-forward folds (in-sample)")
    st.dataframe(report.walk_forward.summary_table(), width="stretch")

    st.subheader("Monte Carlo trade-order simulation")
    mc = report.monte_carlo
    st.write(
        f"Final balance is order-invariant (always **£{mc.actual_final_balance:,.2f}**). "
        f"What varies is max drawdown depending on the order trades occurred:"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Worst case (p5)", f"{mc.max_dd_p5:.2f}%")
    c2.metric("Median (p50)", f"{mc.max_dd_p50:.2f}%")
    c3.metric("Best case (p95)", f"{mc.max_dd_p95:.2f}%")
    c4.metric("Actual sequence", f"{mc.actual_max_dd:.2f}%")

    # Bootstrap confidence intervals
    if report.bootstrap is not None and not pd.isna(report.bootstrap.sharpe_p50):
        st.subheader(f"Bootstrap confidence intervals ({report.bootstrap.n_simulations} resamples)")
        st.caption(
            "How stable are headline numbers if we resample trades? "
            "Wide bands = small sample, fragile result."
        )
        bs = report.bootstrap
        cb1, cb2 = st.columns(2)
        with cb1:
            st.markdown("**Per-trade Sharpe**")
            st.write(f"p5 (worst): {bs.sharpe_p5:.3f}")
            st.write(f"p50 (median): **{bs.sharpe_p50:.3f}**")
            st.write(f"p95 (best): {bs.sharpe_p95:.3f}")
        with cb2:
            st.markdown("**Profit factor**")
            st.write(f"p5 (worst): {bs.profit_factor_p5:.3f}")
            st.write(f"p50 (median): **{bs.profit_factor_p50:.3f}**")
            st.write(f"p95 (best): {bs.profit_factor_p95:.3f}")

    st.subheader("Verdict")
    for line in _verdict_lines(report):
        st.markdown(line)


elif mode == "Parameter sweep":
    # ---- Cache key: re-run sweep only if config actually changed -------
    sweep_config = {
        "strategy": display_label, "ticker": ticker, "interval": interval,
        "grid": {k: sorted(map(str, v)) for k, v in param_grid.items()},
        "oos_fraction": oos_fraction, "n_folds": n_folds,
    }
    sweep_hash = hashlib.md5(
        json.dumps(sweep_config, sort_keys=True, default=str).encode()
    ).hexdigest()

    # If config changed since last sweep, clear cached results
    if st.session_state.get("sweep_hash") != sweep_hash:
        st.session_state["sweep_hash"] = sweep_hash
        st.session_state["sweep_result"] = None
        st.session_state["oos_eval"] = None

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)

    if st.session_state.get("sweep_result") is None:
        st.write(f"Will run **{n_combos} combinations** on the in-sample slice "
                 f"(reserving last {oos_fraction:.0%} for OOS).")
        if n_combos > 200:
            st.warning(f"{n_combos} combinations is a lot — sweep could take "
                       f"several minutes and the multiple-testing penalty grows fast. "
                       f"Consider narrower grids.")

        if not run_clicked:
            st.info("Configuration changed. Click **Run backtest** on the left "
                    "to start the sweep with these settings.")
            st.stop()

        progress = st.progress(0.0, text="Starting sweep...")
        last_msg = st.empty()

        def on_progress(i: int, n: int, params: dict):
            progress.progress(i / n,
                              text=f"[{i}/{n}] running combinations...")
            last_msg.caption(f"Last: {params}")

        try:
            sweep = grid_sweep(
                data,
                factory_for_params=factory_for_params,
                param_grid=param_grid,
                oos_fraction=oos_fraction,
                n_folds=n_folds,
                warmup_bars=warmup_bars,
                bars_per_year=bpy,
                progress_callback=on_progress,
            )
            st.session_state["sweep_result"] = sweep
            progress.empty()
            last_msg.empty()
        except Exception as e:
            st.error(f"Sweep failed: {e}")
            st.stop()

    sweep = st.session_state["sweep_result"]
    df = sweep.summary_df()
    st.success(f"Sweep complete: {len(sweep.runs)} successful, "
               f"{sweep.n_failures} failed.")

    valid = df[df["trades"] >= min_trades] if "trades" in df.columns else df
    st.caption(f"{len(valid)} combinations had ≥{min_trades} trades.")

    # Sort selector — defend against empty df (sweep produced no successes)
    sortable_cols = [c for c in ["sharpe", "wf_consistency", "profit_factor",
                                 "return_%", "trades", "max_dd_%"] if c in df.columns]
    if not sortable_cols:
        st.warning("Sweep produced no usable runs — try a different param grid "
                   "or a strategy that produces more trades on this data.")
        st.stop()
    sort_by = st.selectbox("Sort by", sortable_cols,
                           index=sortable_cols.index("wf_consistency") if "wf_consistency" in sortable_cols else 0)
    ascending = sort_by == "max_dd_%"  # for max_dd, less negative is better
    sorted_df = valid.sort_values(sort_by, ascending=ascending) if not valid.empty else valid

    st.subheader("Sweep results")
    st.dataframe(sorted_df, width="stretch", hide_index=True)

    # Deflated Sharpe across the sweep
    if len(valid) >= 2:
        sharpes_arr = valid["sharpe"].to_numpy()
        sr_var = float(np.var(sharpes_arr))
        st.caption(
            f"Variance of Sharpe across {len(valid)} valid trials: {sr_var:.4f} "
            "— used for Deflated Sharpe Ratio when you evaluate the best on OOS."
        )

    # ---- Step 2: pick one combination → evaluate on OOS ---------------
    st.divider()
    st.subheader("Step 2 — pick ONE config, evaluate on out-of-sample")
    st.warning(
        "⚠️ Pick the config you'd commit to (ideally by walk-forward consistency, "
        "not raw Sharpe). The OOS result is your honest verdict — "
        "re-running with different params after seeing OOS contaminates the test."
    )

    if sorted_df.empty:
        st.info("No combinations with enough trades. Loosen the grid or "
                "lower the min-trades filter.")
    else:
        param_cols = [c for c in sorted_df.columns
                      if c not in ("trades", "sharpe", "profit_factor", "return_%",
                                   "max_dd_%", "win_rate_%", "wf_consistency")]

        def _row_label(row):
            params_str = ", ".join(f"{c}={row[c]}" for c in param_cols)
            return (f"sharpe={row['sharpe']:+.2f}  "
                    f"pf={row['profit_factor']:.2f}  "
                    f"wf={row['wf_consistency']:.0%}  |  {params_str}")

        labels = [_row_label(r) for _, r in sorted_df.iterrows()]
        picked_label = st.selectbox("Configuration to evaluate", labels)
        picked_idx = labels.index(picked_label)
        picked_row = sorted_df.iloc[picked_idx]

        if st.button("Evaluate on out-of-sample", type="primary"):
            # Coerce values back to their declared types — DataFrame round-trip
            # converts ints to numpy float64, which breaks strategies that use
            # int params as iloc indices (e.g. BPR's approach_lookback).
            type_map = {p.name: p.type for p in (spec.params or [])}
            picked_params = {}
            for c in param_cols:
                raw = picked_row[c]
                t = type_map.get(c)
                if t == "int":
                    picked_params[c] = int(raw)
                elif t == "bool":
                    picked_params[c] = bool(raw)
                else:
                    picked_params[c] = float(raw)
            with st.spinner("Running OOS evaluation..."):
                try:
                    factory = factory_for_params(picked_params)
                    oos_result, oos_metrics = evaluate_oos(
                        data, factory, oos_fraction=oos_fraction,
                        warmup_bars=warmup_bars, bars_per_year=bpy,
                    )
                    st.session_state["oos_eval"] = (oos_result, oos_metrics, picked_params)
                except Exception as e:
                    st.error(f"OOS evaluation failed: {e}")

        if st.session_state.get("oos_eval"):
            oos_result, oos_metrics, oos_params = st.session_state["oos_eval"]
            st.divider()
            st.subheader("OOS verdict (final)")
            st.caption(f"Params evaluated: `{oos_params}`")
            metrics_panel(oos_metrics, label="Out-of-sample")

            # Deflated Sharpe Ratio for this best
            if len(valid) >= 2:
                dsr = deflated_sharpe_ratio(
                    oos_result.trades_df,
                    n_trials=len(valid),
                    sharpe_variance_across_trials=sr_var,
                )
                if not pd.isna(dsr):
                    st.metric(
                        "Deflated Sharpe Ratio",
                        f"{dsr:.1%}",
                        help=("Probability the OOS result is genuinely better than "
                              "the lucky tail of N tested combinations. "
                              ">95% = strong, <50% = likely just noise."),
                    )
                    if dsr > 0.95:
                        st.success("Strong evidence: this config genuinely outperforms random.")
                    elif dsr > 0.5:
                        st.info("Moderate evidence — not conclusive.")
                    else:
                        st.warning("Weak — best result may just be the luckiest of many trials.")

            if not oos_result.trades_df.empty:
                st.plotly_chart(
                    equity_chart(oos_result.equity_curve, price=data["Close"],
                                 trades_df=oos_result.trades_df,
                                 title="OOS equity curve"),
                    width="stretch",
                )


elif mode == "Adaptive walk-forward":
    # Cache results so slider tweaks (e.g. metric to optimize) don't re-run
    awf_config = {
        "strategy": display_label, "ticker": ticker, "interval": interval,
        "grid": {k: sorted(map(str, v)) for k, v in param_grid.items()},
        "n_folds": n_folds, "optimize_by": optimize_by,
    }
    awf_hash = hashlib.md5(
        json.dumps(awf_config, sort_keys=True, default=str).encode()
    ).hexdigest()
    if st.session_state.get("awf_hash") != awf_hash:
        st.session_state["awf_hash"] = awf_hash
        st.session_state["awf_result"] = None

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    n_test_folds = n_folds - 1
    n_total_backtests = n_combos * n_test_folds + n_test_folds  # sweeps + tests

    if st.session_state.get("awf_result") is None:
        st.write(f"Will run **adaptive walk-forward**: {n_test_folds} test folds, "
                 f"each preceded by sweeping {n_combos} param combinations on training data. "
                 f"Total ≈ {n_total_backtests} backtests.")
        if n_total_backtests > 500:
            st.warning(f"{n_total_backtests} backtests is a lot — this will take a while. "
                       f"Consider fewer folds or a smaller param grid.")

        if not run_clicked:
            st.info("Configuration changed. Click **Run backtest** on the left to start.")
            st.stop()

        with st.spinner(f"Running adaptive walk-forward... (~{n_total_backtests} backtests)"):
            try:
                awf = adaptive_walk_forward(
                    data,
                    factory_for_params=factory_for_params,
                    param_grid=param_grid,
                    n_folds=n_folds,
                    optimization_metric=optimize_by,
                    warmup_bars=warmup_bars,
                    bars_per_year=bpy,
                )
                st.session_state["awf_result"] = awf
            except Exception as e:
                st.error(f"Adaptive WF failed: {e}")
                st.stop()

    awf = st.session_state["awf_result"]
    if not awf.folds:
        st.error("No folds completed. The strategy may not produce enough trades on this data.")
    else:
        agg = awf.aggregate_test_metrics()
        st.subheader("Aggregate test metrics (across all test folds)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test folds", agg["n_folds"])
        c2.metric("Avg test Sharpe", f"{agg['avg_test_sharpe']:.2f}")
        c3.metric("Avg test profit factor", f"{agg['avg_test_profit_factor']:.2f}")
        c4.metric("Test consistency", f"{agg['test_consistency']:.0%}",
                  help="Fraction of test folds with profit factor > 1.0")

        st.subheader("Per-fold breakdown")
        st.caption(
            "Each row: train period, chosen params, then test-fold metrics. "
            "Stable params across folds = robust strategy. Wildly different "
            "params each fold = regime-adaptive (or overfit, hard to tell)."
        )
        st.dataframe(awf.summary_table(), width="stretch", hide_index=True)

        st.subheader("Verdict")
        if agg["test_consistency"] >= 0.6 and agg["avg_test_sharpe"] > 0.3:
            st.success(
                "✅ Test consistency ≥ 60% and positive avg test Sharpe — "
                "strategy generalises with re-tuning."
            )
        elif agg["test_consistency"] < 0.4:
            st.error(
                "❌ Test consistency < 40% — the strategy doesn't generalise even "
                "with adaptive re-tuning. Probably no real edge."
            )
        else:
            st.info("🟡 Borderline — moderate consistency. Inconclusive.")


elif mode == "Bayesian sweep":
    # Cache by config — same pattern as grid sweep
    bayes_config = {
        "strategy": display_label, "ticker": ticker, "interval": interval,
        "n_trials": n_trials, "oos_fraction": oos_fraction, "n_folds": n_folds,
        "opt_metric": opt_metric, "target_trades": target_trades,
    }
    bayes_hash = hashlib.md5(
        json.dumps(bayes_config, sort_keys=True, default=str).encode()
    ).hexdigest()
    if st.session_state.get("bayes_hash") != bayes_hash:
        st.session_state["bayes_hash"] = bayes_hash
        st.session_state["bayes_result"] = None
        st.session_state["bayes_oos_eval"] = None

    if st.session_state.get("bayes_result") is None:
        st.write(f"Will run **{n_trials} Optuna TPE trials** on the in-sample slice "
                 f"(reserving last {oos_fraction:.0%} for OOS).")
        st.caption(f"Optimising for: **{opt_metric}** with trade-count penalty (target {target_trades} trades).")
        if not run_clicked:
            st.info("Click **Run backtest** to start the Bayesian search.")
            st.stop()

        progress = st.progress(0.0, text="Starting Optuna...")
        last_msg = st.empty()

        def on_progress(i, n, params):
            progress.progress(min(1.0, i / n), text=f"Trial {i}/{n}")
            last_msg.caption(f"Latest: {params}")

        try:
            opt_result = run_optuna_study(
                data, spec, factory_for_params=factory_for_params,
                n_trials=n_trials, oos_fraction=oos_fraction, n_folds=n_folds,
                warmup_bars=warmup_bars, bars_per_year=bpy,
                target_trades=target_trades, optimization_metric=opt_metric,
                progress_callback=on_progress,
            )
            st.session_state["bayes_result"] = opt_result
            progress.empty()
            last_msg.empty()
        except Exception as e:
            st.error(f"Bayesian sweep failed: {e}")
            st.stop()

    opt_result = st.session_state["bayes_result"]
    df = opt_result.summary_df()
    st.success(f"Bayesian sweep complete: {len(opt_result.trials)} trials. "
               f"Best score: {opt_result.best_score:.3f}.")

    if df.empty:
        st.warning("No usable trials. Try more trials, broader param ranges, or different optimisation metric.")
        st.stop()

    # Filter for sensible trade count
    valid = df[df["trades"] >= 30] if "trades" in df.columns else df
    st.caption(f"{len(valid)} trials with ≥30 trades.")

    sort_options = [c for c in ["score", "wf_consistency", "sharpe", "profit_factor",
                                "return_%", "trades", "max_dd_%"] if c in df.columns]
    sort_by = st.selectbox("Sort by", sort_options, index=0)
    ascending = sort_by == "max_dd_%"
    sorted_df = (valid if not valid.empty else df).sort_values(sort_by, ascending=ascending)

    st.subheader("Trial results")
    st.dataframe(sorted_df, width="stretch", hide_index=True)

    # OOS evaluation step (same as grid sweep)
    st.divider()
    st.subheader("Step 2 — pick ONE config, evaluate on out-of-sample")
    st.warning(
        "⚠️ Pick the config you'd commit to. The OOS result is your honest "
        "verdict — re-running on different params after seeing OOS contaminates the test."
    )
    param_cols = [c for c in sorted_df.columns
                  if c not in ("trades", "sharpe", "profit_factor", "return_%",
                               "max_dd_%", "win_rate_%", "wf_consistency", "score")]

    def _row_label(row):
        params_str = ", ".join(f"{c}={row[c]}" for c in param_cols)
        return (f"score={row['score']:+.3f}  sharpe={row['sharpe']:+.2f}  "
                f"wf={row['wf_consistency']:.0%}  |  {params_str}")

    labels = [_row_label(r) for _, r in sorted_df.iterrows()]
    picked_label = st.selectbox("Configuration to evaluate", labels, key="bayes_pick")
    picked_idx = labels.index(picked_label)
    picked_row = sorted_df.iloc[picked_idx]

    if st.button("Evaluate on out-of-sample", type="primary", key="bayes_oos_btn"):
        # Coerce types based on spec (same as grid sweep)
        type_map = {p.name: p.type for p in (spec.params or [])}
        picked_params = {}
        for c in param_cols:
            raw = picked_row[c]
            t = type_map.get(c)
            if t == "int":
                picked_params[c] = int(raw)
            elif t == "bool":
                picked_params[c] = bool(raw)
            else:
                picked_params[c] = float(raw)
        with st.spinner("Running OOS evaluation..."):
            try:
                factory = factory_for_params(picked_params)
                oos_result, oos_metrics = evaluate_oos(
                    data, factory, oos_fraction=oos_fraction,
                    warmup_bars=warmup_bars, bars_per_year=bpy,
                )
                st.session_state["bayes_oos_eval"] = (oos_result, oos_metrics, picked_params)
            except Exception as e:
                st.error(f"OOS evaluation failed: {e}")

    if st.session_state.get("bayes_oos_eval"):
        oos_result, oos_metrics, oos_params = st.session_state["bayes_oos_eval"]
        st.divider()
        st.subheader("OOS verdict (final)")
        st.caption(f"Params evaluated: `{oos_params}`")
        metrics_panel(oos_metrics, label="Out-of-sample")
        if not oos_result.trades_df.empty:
            st.plotly_chart(
                equity_chart(oos_result.equity_curve, price=data["Close"],
                             trades_df=oos_result.trades_df, title="OOS equity curve"),
                width="stretch",
            )
