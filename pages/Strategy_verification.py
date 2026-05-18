"""
Strategy verification page.

Sanity-check every strategy in the registry by running it on a few small
samples across different assets and date ranges, then SHOW the trades it
took — chart, entry/exit levels, plain-English explanation of WHY the
strategy fired.

The point is to catch "this strategy is silently broken" or "this strategy
is detecting setups that don't actually exist on the chart" before you
trust its backtest numbers. If you eyeball 5 trades and they all look
sensible, you can trust the aggregate stats. If 2 of 5 look wrong, the
strategy's logic has a bug regardless of what the equity curve says.

This page is intentionally minimal. It uses the same engine + broker
the main backtester does, so a trade rendered here is the EXACT same
trade you'd get in a regular backtest.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when Streamlit runs this page directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import config
from config import PROFILES, profile_for, session_defaults_for, trading_window_for
from data.fetcher import fetch
from backtest.engine import run_backtest
from backtest.graph import (
    DecisionGraph, TriggerNode, GraphOrchestrator, _parse_hhmm,
)
from strategies import registry as reg


st.set_page_config(
    page_title="Strategy verification",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Preset asset configurations ---------------------------------------
# Quick "test on these typical instruments" presets. The user can also
# enter a custom ticker. Each preset declares the EODHD symbol, a sensible
# data interval, and (where relevant) a hint that this is an unusual
# trading-hours instrument.
ASSET_PRESETS: dict[str, dict] = {
    "UK 100 (FTSE)":   {"ticker": "UK100",    "interval": "15m"},
    "AAPL":            {"ticker": "AAPL",     "interval": "15m"},
    "TSLA":            {"ticker": "TSLA",     "interval": "15m"},
    "MSFT":            {"ticker": "MSFT",     "interval": "15m"},
    "BTC-USD":         {"ticker": "BTC-USD",  "interval": "1h"},
    "Custom...":       {"ticker": "",         "interval": "15m"},
}

TIME_RANGES: dict[str, int] = {
    "Last 30 days":   30,
    "Last 60 days":   60,
    "Last 120 days":  120,
    "Last 1 year":    365,
}


# ---- Sidebar ----------------------------------------------------------
with st.sidebar:
    st.header("Verification setup")
    st.caption(
        "Pick a strategy, an asset, and a time window. We'll run a small "
        "backtest with the strategy's default params and show example "
        "trades so you can sanity-check the logic visually."
    )

    # Strategy picker
    strategy_keys = list(reg.STRATEGIES.keys())
    strategy_labels = [reg.STRATEGIES[k].label for k in strategy_keys]
    sel_label = st.selectbox("Strategy", strategy_labels, index=0)
    strategy_key = strategy_keys[strategy_labels.index(sel_label)]
    spec = reg.get(strategy_key)

    # Asset preset
    asset_preset = st.selectbox("Asset", list(ASSET_PRESETS.keys()), index=0)
    preset_cfg = ASSET_PRESETS[asset_preset]
    ticker = preset_cfg["ticker"]
    default_interval = preset_cfg["interval"]
    if asset_preset == "Custom...":
        ticker = st.text_input("Ticker", value="UK100",
                                help="EODHD symbol (e.g. AAPL, TSLA, MSFT, "
                                     "BTC-USD, UK100, GDAXI.INDX).")

    # Interval
    interval = st.selectbox(
        "Interval",
        ["1m", "5m", "15m", "30m", "1h"],
        index=["1m", "5m", "15m", "30m", "1h"].index(default_interval),
    )

    # Time range
    range_label = st.selectbox("Date range", list(TIME_RANGES.keys()), index=1)
    days_back = TIME_RANGES[range_label]
    # Convert to a bar count: avg trading-time-per-day × bars/day × N days
    minutes_per_bar = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}[interval]
    bars_per_day = max(1, (6.5 * 60) // minutes_per_bar)  # rough RTH coverage
    bar_count = int(days_back * bars_per_day * 1.2)  # 20% headroom

    # Trade count
    n_trades_show = st.slider(
        "Trades to display", min_value=3, max_value=20, value=8, step=1,
        help="How many trades to render below. We pick a representative "
             "mix (winners + losers, long + short).",
    )

    # Cost profile (auto-detected; user can override)
    auto_profile = profile_for(ticker)
    profile_names = list(PROFILES.keys())
    try:
        idx = profile_names.index(auto_profile.instrument)
    except ValueError:
        idx = profile_names.index("DEFAULT")
    chosen_profile = st.selectbox("Cost profile", profile_names, index=idx)
    config.COSTS = PROFILES[chosen_profile]

    # Data hours filter
    data_hours = st.radio(
        "Data hours",
        ["RTH (regular hours)", "Extended hours", "All available"],
        index=0 if PROFILES[chosen_profile].instrument not in
              ("BTC", "ETH", "EURUSD", "GBPUSD", "USDJPY") else 2,
    )
    hours_mode = {"RTH (regular hours)": "rth",
                  "Extended hours": "eth",
                  "All available": None}[data_hours]

    run_clicked = st.button("Verify", type="primary", width="stretch")


# ---- Main area --------------------------------------------------------
st.title("Strategy verification")
st.caption(
    "Click **Verify** in the sidebar after picking a strategy + asset. "
    "Trades shown are real backtests — same engine, broker, cost model "
    "as the main backtester."
)

# ---- Strategy summary card (always visible) -------------------------
st.markdown(f"## {spec.label}")
if spec.description:
    st.markdown(f"*{spec.description}*")

with st.expander("Defaults that will be used", expanded=False):
    cols = st.columns(3)
    for i, p in enumerate(spec.params or []):
        with cols[i % 3]:
            st.metric(p.label, str(p.default))

if not run_clicked:
    st.info("Configure on the left, then click **Verify**.")
    st.stop()


# ---- Fetch + filter --------------------------------------------------
with st.spinner(f"Fetching {ticker} {interval} ({bar_count:,} bars)..."):
    try:
        data = fetch(ticker=ticker, interval=interval,
                     source="eodhd", ig_num_points=bar_count)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

# Apply data-hours filter same as main page
if hours_mode is not None:
    w = trading_window_for(chosen_profile, hours_mode)
    if w is not None:
        open_t = _parse_hhmm(w[0])
        close_t = _parse_hhmm(w[1])
        if open_t and close_t:
            tt = data.index.time
            data = data.loc[(tt >= open_t) & (tt <= close_t)]

if len(data) < 200:
    st.warning(f"Only {len(data)} bars after filtering — strategy may not "
                f"have enough warmup. Increase the date range.")

st.caption(f"Loaded **{len(data):,}** bars from "
           f"`{data.index[0]}` to `{data.index[-1]}`.")


# ---- Build minimal graph (trigger-only, no supporters/vetoes) -------
# Session times pulled from the cost profile so we don't accidentally
# trade pre-market on AAPL etc.
sess = session_defaults_for(chosen_profile, mode=(hours_mode or "rth"))
graph = DecisionGraph(
    trigger=TriggerNode(strategy_key=strategy_key,
                        params=spec.defaults(),
                        timeframe=interval),
    supporters=[], vetoes=[],
    min_score=0.0, risk_floor=1.0, risk_ceiling=1.0,
    risk_curve="linear",
    allow_overnight=(sess is None),
    session_open_override=sess[0] if sess else None,
    session_close_override=sess[1] if sess else None,
    flat_by_override=sess[2] if sess else None,
)

# Run
with st.spinner("Running backtest..."):
    orch = GraphOrchestrator(graph)
    result = run_backtest(data, orch, warmup_bars=spec.warmup_bars)

trades = result.trades_df
if trades.empty:
    st.warning(
        "**No trades** fired in this window. Either the strategy didn't "
        "find a setup in this data range, or the data is too short for "
        "warmup. Try a longer date range or different asset."
    )
    st.stop()


# ---- Aggregate stats --------------------------------------------------
st.markdown("### Backtest stats")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", len(trades))
win_rate = (trades["net_pnl_gbp"] > 0).mean() * 100 if len(trades) else 0
c2.metric("Win rate", f"{win_rate:.0f}%")
total_pnl = trades["net_pnl_gbp"].sum()
c3.metric("Total P&L", f"£{total_pnl:+,.2f}")
avg_bars = trades["bars_held"].mean() if "bars_held" in trades else 0
c4.metric("Avg bars held", f"{avg_bars:.0f}")


# ---- Pick a representative sample of trades --------------------------
# We want a MIX so the user can see different behaviours: winners +
# losers, long + short, different exit reasons. Without this we might
# show 5 stop-outs in a row and they'd never see a target hit.
def _pick_representative(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    buckets = []
    for side in ("long", "short"):
        for outcome in ("win", "loss"):
            mask = df["side"].eq(side)
            if outcome == "win":
                mask &= df["net_pnl_gbp"].gt(0)
            else:
                mask &= df["net_pnl_gbp"].le(0)
            sub = df[mask]
            if len(sub):
                # Take evenly-spaced indices so we see trades from
                # different parts of the date range, not all clustered
                step = max(1, len(sub) // max(1, n // 4))
                buckets.append(sub.iloc[::step])
    if not buckets:
        return df.head(n)
    combined = pd.concat(buckets)
    # Deduplicate by INDEX (entry_time) rather than `drop_duplicates()`,
    # which would try to hash the trade row including its `entry_metadata`
    # dict column and explode with `unhashable type: 'dict'`.
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined.head(n)

sample = _pick_representative(trades, n_trades_show)

st.markdown("---")
st.markdown(f"### {len(sample)} example trades — sanity-check them by eye")
st.caption(
    "Each card shows the chart window around the trade with entry, exit, "
    "stop, and target marked. The 'reason' field is the EXACT explanation "
    "the strategy emitted at signal time — if it doesn't match the chart, "
    "the strategy has a bug."
)


# ---- Render trade gallery --------------------------------------------
# We import the chart helpers from app.py. They're at module level so
# this works because the page runs inside the same Python process as
# the main app.
def _render_trade_card(trade, data: pd.DataFrame, idx: int) -> None:
    """One mini trade panel. Headline + reason + chart."""
    side = trade["side"]
    entry = float(trade["entry_price"])
    exitp = float(trade["exit_price"])
    pnl = float(trade["net_pnl_gbp"])
    bars = int(trade.get("bars_held", 0))
    exit_reason = trade.get("exit_reason", "?")
    reason = trade.get("reason", "(no reason text emitted by strategy)")
    entry_time = pd.Timestamp(trade.name if "entry_time" not in trade
                              else trade["entry_time"])

    icon = "🟢" if pnl > 0 else "🔴"
    direction = "📈" if side == "long" else "📉"
    st.markdown(
        f"#### {icon} Trade {idx + 1}: {direction} **{side.upper()}** "
        f"@ {entry:.2f} → {exitp:.2f} ({exit_reason}) · "
        f"**£{pnl:+,.2f}** · {bars} bars · {entry_time.strftime('%Y-%m-%d %H:%M')}"
    )

    # Plain-English reasoning derived from the trade record
    expl = _explain_trade(strategy_key, trade)
    if expl:
        st.markdown(f"**Why this fired:** {expl}")
    if reason and reason != expl:
        st.code(f"strategy.reason = {reason!r}", language="python")

    # Compact chart
    _mini_chart(trade, data, key_suffix=str(idx))
    st.markdown("---")


def _explain_trade(strategy_key: str, trade) -> str:
    """
    Plain-English rationale derived from trade fields + strategy type.
    Augments the strategy's own `reason` string with structural context
    (R distance, stop placement, target).
    """
    side = trade["side"]
    entry = float(trade["entry_price"])
    sl = trade.get("planned_stop_loss")
    tp = trade.get("planned_take_profit")
    pieces = []
    if sl is not None and not pd.isna(sl):
        risk = abs(entry - float(sl))
        pieces.append(f"risk = {risk:.2f}pt ({'below' if side == 'long' else 'above'} entry)")
    if tp is not None and not pd.isna(tp):
        rew = abs(float(tp) - entry)
        if sl is not None and not pd.isna(sl):
            risk = abs(entry - float(sl))
            r_mult = rew / risk if risk > 0 else 0
            pieces.append(f"target = {rew:.2f}pt ({r_mult:.1f}R)")
        else:
            pieces.append(f"target = {rew:.2f}pt")
    if not pieces:
        return ""
    return "; ".join(pieces)


def _mini_chart(trade, data: pd.DataFrame, key_suffix: str) -> None:
    """A compact TradingView LWC chart with markers + SL/TP zones."""
    from streamlit_lightweight_charts import renderLightweightCharts

    entry_time = pd.Timestamp(trade.name if "entry_time" not in trade
                              else trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    entry_price = float(trade["entry_price"])
    side = trade["side"]
    pnl = float(trade["net_pnl_gbp"])
    sl = trade.get("planned_stop_loss")
    tp = trade.get("planned_take_profit")

    BARS_BEFORE = 20
    BARS_AFTER = 30
    try:
        e_idx = data.index.get_indexer([entry_time], method="nearest")[0]
        x_idx = data.index.get_indexer([exit_time], method="nearest")[0]
    except Exception:
        return
    start = max(0, e_idx - BARS_BEFORE)
    end = min(len(data), x_idx + BARS_AFTER + 1)
    window = data.iloc[start:end]

    def _candles(df):
        return [{
            "time": int(t.timestamp()),
            "open": float(r["Open"]), "high": float(r["High"]),
            "low": float(r["Low"]),  "close": float(r["Close"]),
        } for t, r in df.iterrows()]

    series = []
    e_ts = int(entry_time.timestamp())
    x_ts = int(exit_time.timestamp())
    if x_ts <= e_ts:
        x_ts = e_ts + 1

    def _zone(level, top_color, bot_color, line_color, title):
        return {
            "type": "Baseline",
            "data": [{"time": e_ts, "value": float(level)},
                     {"time": x_ts, "value": float(level)}],
            "options": {
                "baseValue": {"type": "price", "price": float(entry_price)},
                "topFillColor1": top_color, "topFillColor2": top_color.replace("0.28", "0.04"),
                "bottomFillColor1": bot_color,
                "bottomFillColor2": bot_color.replace("0.28", "0.04"),
                "topLineColor": line_color, "bottomLineColor": line_color,
                "lineWidth": 1, "lineStyle": 2,
                "lastValueVisible": False, "priceLineVisible": False,
                "title": title,
            },
        }
    if tp is not None and not pd.isna(tp):
        series.append(_zone(float(tp),
                            "rgba(38, 166, 154, 0.28)",
                            "rgba(38, 166, 154, 0.28)",
                            "rgba(38, 166, 154, 0.9)",
                            f"Target {float(tp):.2f}"))
    if sl is not None and not pd.isna(sl):
        series.append(_zone(float(sl),
                            "rgba(239, 83, 80, 0.28)",
                            "rgba(239, 83, 80, 0.28)",
                            "rgba(239, 83, 80, 0.9)",
                            f"Stop {float(sl):.2f}"))

    win_color = "#26a69a" if pnl > 0 else "#ef5350"
    series.append({
        "type": "Candlestick",
        "data": _candles(window),
        "options": {"upColor": "#26a69a", "downColor": "#ef5350",
                     "borderVisible": False,
                     "wickUpColor": "#26a69a", "wickDownColor": "#ef5350"},
        "markers": [
            {"time": e_ts,
             "position": "belowBar" if side == "long" else "aboveBar",
             "color": "#2196f3",
             "shape": "arrowUp" if side == "long" else "arrowDown",
             "text": f"Entry {side}"},
            {"time": x_ts,
             "position": "aboveBar" if pnl > 0 else "belowBar",
             "color": win_color, "shape": "circle",
             "text": f"Exit £{pnl:+.2f}"},
        ],
    })

    chart_opts = {
        "height": 280,
        "layout": {"background": {"type": "solid", "color": "transparent"},
                   "textColor": "#cfcfcf"},
        "grid": {"vertLines": {"color": "rgba(255,255,255,0.05)"},
                  "horzLines": {"color": "rgba(255,255,255,0.05)"}},
        "timeScale": {"timeVisible": True, "secondsVisible": False},
    }
    renderLightweightCharts(
        [{"chart": chart_opts, "series": series}],
        f"verify_chart_{key_suffix}",
    )


for i, (_, row) in enumerate(sample.iterrows()):
    _render_trade_card(row, data, i)


# ---- Footer guidance --------------------------------------------------
with st.expander("What to look for when sanity-checking"):
    st.markdown(
        "- **Entry alignment**: does the entry arrow sit where the "
        "strategy says it should? (e.g. FVG retest near the FVG zone "
        "edge, ORB breakout exactly at the OR boundary)\n"
        "- **Stop / target geometry**: red zone on the loss side, green "
        "on the profit side, NEVER both on the same side.\n"
        "- **Bars held vs exit reason**: a `target` exit after 1 bar in a "
        "tight-range market is suspicious; a `session_end` exit after 30 "
        "bars is the strategy genuinely holding through the session.\n"
        "- **Win/loss MIX**: if all 8 trades hit target, the sampling is "
        "biased; if all are stops, the strategy may be entering against "
        "obvious momentum.\n"
        "- **Reason text vs visual**: the strategy's emitted `reason` "
        "field should be consistent with what you see on the chart. "
        "If reason says 'bullish FVG at 7892' but the chart shows no FVG, "
        "the strategy's detection is buggy."
    )
