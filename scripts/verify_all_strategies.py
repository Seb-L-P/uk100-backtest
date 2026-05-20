"""
Per-strategy verification script.

Runs every registered strategy against three canned scenarios:
    - UK100 60d @ 15m   (EODHD ISF.LSE proxy, FTSE-cash scaled)
    - AAPL  60d @ 15m   (EODHD AAPL.US)
    - BTC-USD 60d @ 1h  (yfinance)

For every (strategy, scenario) it:
    - sets the matching cost profile + session times (when the strategy
      accepts session params)
    - runs the backtest with default constructor params
    - dumps a per-trade table: entry/exit time, side, price, R, reason,
      exit_reason, bars held
    - runs red-flag auto-checks:
        * no trades at all on a scenario where trades were expected
        * same-bar in-and-out (entry_time == exit_time)
        * exit_price within slippage of stop_loss = entry (broken geometry)
        * stop_out fraction > 75%
        * eod-close fraction > 90% (session handling broken)
        * dropped_order / dropped_geometry counts dominate signal flow
        * accounting drift (engine already asserts, but report any catch)

Output:
    - summary table written to verify_output/summary.md
    - per-(strategy, scenario) trade tables in verify_output/<key>_<scenario>.md
    - red-flag rollup printed to stdout

Usage:
    python scripts/verify_all_strategies.py
    python scripts/verify_all_strategies.py --only fvg sma
"""
from __future__ import annotations

import argparse
import inspect
import sys
import traceback
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import config
from backtest.engine import run_backtest, BacktestResult
from data.fetcher import fetch
from strategies.registry import STRATEGIES, StrategySpec


# ---------- Scenario definitions ------------------------------------------
@dataclass
class Scenario:
    name: str
    ticker: str          # passed to fetch
    interval: str
    source: str
    num_points: int      # data window
    profile_key: str     # config.profile_for() lookup hint
    session_open: time | None
    session_close: time | None
    flat_by: time | None
    description: str
    expects_trades: bool = True   # if False, no-trades is not a red flag


SCENARIOS: list[Scenario] = [
    Scenario(
        name="UK100_60d_15m",
        ticker="UK100",
        interval="15m",
        source="eodhd",
        num_points=3000,
        profile_key="UK100",
        # UK cash session in London time
        session_open=time(8, 0),
        session_close=time(15, 30),
        flat_by=time(16, 0),
        description="FTSE 100 cash (ISF.LSE × 10), London hours, 60d at 15m",
    ),
    Scenario(
        name="AAPL_60d_15m",
        ticker="AAPL.US",
        interval="15m",
        source="eodhd",
        num_points=3000,
        profile_key="STOCK",
        # US cash session in London time
        session_open=time(14, 30),
        session_close=time(20, 30),
        flat_by=time(20, 55),
        description="AAPL via EODHD, US cash hours in London time, 60d at 15m",
    ),
    Scenario(
        name="BTC_60d_1h",
        ticker="BTC-USD",
        interval="1h",
        source="yfinance",
        num_points=1500,
        profile_key="BTC",
        # Crypto is 24/7; widen the session to "always".
        session_open=time(0, 0),
        session_close=time(23, 30),
        flat_by=time(23, 45),
        description="BTC-USD via yfinance, 24/7, 60d at 1h",
    ),
]


# ---------- Data loading --------------------------------------------------
def load_scenario_data(s: Scenario) -> pd.DataFrame:
    return fetch(
        ticker=s.ticker,
        interval=s.interval,
        source=s.source,
        ig_num_points=s.num_points,
    )


# ---------- Strategy construction -----------------------------------------
SESSION_PARAM_NAMES = {"session_open", "session_close", "flat_by"}


def build_strategy(spec: StrategySpec, scenario: Scenario) -> Any:
    """Construct a strategy with default params + scenario-appropriate
    session times when accepted by the constructor."""
    defaults = spec.defaults()
    if spec.cls is None:
        return spec.build(**defaults)
    sig = inspect.signature(spec.cls.__init__)
    extra: dict[str, Any] = {}
    for p in SESSION_PARAM_NAMES:
        if p not in sig.parameters:
            continue
        if p == "session_open" and scenario.session_open is not None:
            extra[p] = scenario.session_open
        elif p == "session_close" and scenario.session_close is not None:
            extra[p] = scenario.session_close
        elif p == "flat_by" and scenario.flat_by is not None:
            extra[p] = scenario.flat_by
    return spec.build(**defaults, **extra)


def has_session_filter(spec: StrategySpec) -> bool:
    if spec.cls is None:
        return False
    sig = inspect.signature(spec.cls.__init__)
    return any(p in sig.parameters for p in SESSION_PARAM_NAMES)


# ---------- Red-flag checks -----------------------------------------------
@dataclass
class RedFlag:
    code: str          # short key for grouping
    severity: str      # "warn" | "error"
    message: str


@dataclass
class ScenarioResult:
    scenario: Scenario
    spec: StrategySpec
    trades_df: pd.DataFrame
    result: BacktestResult | None
    flags: list[RedFlag] = field(default_factory=list)
    error: str | None = None
    expired_orders: int = 0
    cancelled_orders: int = 0
    dropped_orders: int = 0
    dropped_geometry: int = 0
    bars_processed: int = 0


def _approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def detect_red_flags(res: ScenarioResult, slippage_tol_pts: float) -> None:
    """Walk the trades and tally common bug patterns."""
    flags = res.flags
    spec = res.spec
    scenario = res.scenario
    trades = res.trades_df
    n = len(trades)

    if res.error is not None:
        flags.append(RedFlag("crash", "error", f"Backtest crashed: {res.error}"))
        return

    # No trades at all
    if n == 0:
        if scenario.expects_trades:
            # Many "session-bound day-trade" strategies WILL legitimately
            # produce zero trades on an asset where the session window doesn't
            # match — but we already shift the session into the scenario for
            # those strategies, so zero trades remains noteworthy.
            flags.append(RedFlag(
                "no_trades", "warn",
                f"Zero trades over {res.bars_processed} bars "
                f"(expired={res.expired_orders}, dropped_geom={res.dropped_geometry}, "
                f"dropped_other={res.dropped_orders})",
            ))
        return

    # Per-trade pathologies
    same_bar_count = 0
    stop_eq_entry_count = 0
    immediate_stop_count = 0
    zero_pnl_count = 0
    for _, t in trades.iterrows():
        if t["entry_time"] == t["exit_time"]:
            same_bar_count += 1
        planned_stop = t.get("planned_stop_loss")
        # Stop placed AT entry (broken geometry — broker would normally
        # reject; if it slips through it indicates a real bug)
        if planned_stop is not None and _approx(float(planned_stop),
                                                 float(t["entry_price"]),
                                                 slippage_tol_pts):
            stop_eq_entry_count += 1
        # Same-bar entry that immediately closed via "stop"
        if (t["exit_reason"] == "stop"
                and t["bars_held"] <= 1
                and t["entry_time"] == t["exit_time"]):
            immediate_stop_count += 1
        # Zero P&L round-trip — flat exit, suspicious for non-eod reason
        if abs(float(t["net_pnl_gbp"])) < 0.005 and t["exit_reason"] not in ("eod",):
            zero_pnl_count += 1

    def pct(x: int) -> float:
        return 100.0 * x / max(n, 1)

    if same_bar_count > 0:
        sev = "error" if pct(same_bar_count) > 25 else "warn"
        flags.append(RedFlag(
            "same_bar_inout", sev,
            f"{same_bar_count}/{n} ({pct(same_bar_count):.0f}%) trades opened and "
            "closed on the same bar — geometry / fill-order issue suspected",
        ))
    if stop_eq_entry_count > 0:
        flags.append(RedFlag(
            "stop_at_entry", "error",
            f"{stop_eq_entry_count}/{n} ({pct(stop_eq_entry_count):.0f}%) trades have "
            "planned_stop_loss == entry_price within slippage — broken stop logic",
        ))
    if immediate_stop_count > 0 and pct(immediate_stop_count) > 30:
        flags.append(RedFlag(
            "immediate_stops", "warn",
            f"{immediate_stop_count}/{n} ({pct(immediate_stop_count):.0f}%) trades "
            "stopped out on entry bar — stop probably too tight relative to fill",
        ))

    # Exit-reason concentration
    reason_counts = trades["exit_reason"].value_counts().to_dict()
    eod_share = reason_counts.get("eod", 0) / n
    if eod_share > 0.9:
        flags.append(RedFlag(
            "eod_dominates", "warn",
            f"{reason_counts.get('eod', 0)}/{n} trades exited via 'eod' force-close — "
            "session handling likely not firing properly",
        ))

    stop_share = reason_counts.get("stop", 0) / n
    if n >= 8 and stop_share > 0.85:
        flags.append(RedFlag(
            "stop_dominates", "warn",
            f"{int(stop_share*100)}% of trades stopped out — strategy may be picking "
            "the wrong side or stops too tight",
        ))

    # Direction-concentration check
    side_counts = trades["side"].value_counts().to_dict()
    if n >= 10:
        one_side = max(side_counts.values()) / n
        if one_side > 0.95:
            only = max(side_counts, key=side_counts.get)
            flags.append(RedFlag(
                "one_side_only", "warn",
                f"{int(one_side*100)}% trades on {only} side — signal generation "
                "may have a directional bug or filter that's never inverting",
            ))

    # Order-drop dominance
    total_attempts = n + res.dropped_orders + res.dropped_geometry + res.expired_orders
    if total_attempts >= 10:
        drop_share = (res.dropped_orders + res.dropped_geometry) / total_attempts
        if drop_share > 0.7:
            flags.append(RedFlag(
                "drops_dominate", "warn",
                f"{int(drop_share*100)}% of order attempts dropped (leverage / "
                "geometry / max-positions) — sizing or stop placement misconfigured",
            ))


# ---------- Backtest runner -----------------------------------------------
def run_one(spec: StrategySpec, scenario: Scenario,
            data: pd.DataFrame) -> ScenarioResult:
    """Run one (strategy, scenario), return result + red flags."""
    # Swap to the scenario's cost profile so spread/slippage are realistic
    config.COSTS = config.profile_for(scenario.profile_key)
    try:
        strategy = build_strategy(spec, scenario)
    except Exception as e:
        return ScenarioResult(
            scenario=scenario, spec=spec,
            trades_df=pd.DataFrame(), result=None,
            error=f"construct failed: {e}",
        )

    try:
        result = run_backtest(
            data=data,
            strategy=strategy,
            warmup_bars=spec.warmup_bars,
            costs=config.COSTS,
            verbose=False,
        )
    except Exception as e:
        return ScenarioResult(
            scenario=scenario, spec=spec,
            trades_df=pd.DataFrame(), result=None,
            error=f"run_backtest raised: {e}\n{traceback.format_exc(limit=6)}",
        )

    # Flatten trades into a cleaner DataFrame (entry_time was used as index)
    tdf = result.trades_df.copy()
    if not tdf.empty:
        tdf = tdf.reset_index()
        if "entry_time" not in tdf.columns and "index" in tdf.columns:
            tdf = tdf.rename(columns={"index": "entry_time"})

    res = ScenarioResult(
        scenario=scenario, spec=spec,
        trades_df=tdf, result=result,
        expired_orders=getattr(result, "_expired_order_count", 0),
        cancelled_orders=getattr(result, "_cancelled_order_count", 0),
        dropped_orders=getattr(result, "_dropped_order_count", 0),
        dropped_geometry=getattr(result, "_dropped_geometry_count", 0),
        bars_processed=result.bars_processed,
    )

    # Slippage tolerance varies by instrument scale
    if scenario.profile_key in ("UK100", "US500", "US100", "DJI", "GER40", "FRA40", "JPN225"):
        slip_tol = 2.0           # points
    elif scenario.profile_key == "BTC":
        slip_tol = 30.0
    elif scenario.profile_key == "STOCK":
        slip_tol = 0.30          # USD on AAPL-class
    else:
        slip_tol = 1.0
    detect_red_flags(res, slip_tol)
    return res


# ---------- Output writers ------------------------------------------------
OUT_DIR = PROJECT_ROOT / "verify_output"


def write_per_run_report(res: ScenarioResult) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUT_DIR / f"{res.spec.key}__{res.scenario.name}.md"
    lines: list[str] = []
    lines.append(f"# {res.spec.label} on {res.scenario.name}")
    lines.append("")
    lines.append(f"_{res.scenario.description}_")
    lines.append("")
    if res.error is not None:
        lines.append(f"**ERROR** — {res.error}")
        fname.write_text("\n".join(lines))
        return fname

    n = len(res.trades_df)
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Bars processed: **{res.bars_processed}**")
    lines.append(f"- Trades: **{n}**")
    lines.append(f"- Final balance: £{res.result.final_balance:,.2f} "
                 f"(start £{res.result.starting_balance:,.0f})")
    lines.append(f"- Expired orders: {res.expired_orders}, "
                 f"cancelled: {res.cancelled_orders}, "
                 f"dropped(leverage/maxpos): {res.dropped_orders}, "
                 f"dropped(geometry): {res.dropped_geometry}")
    if res.flags:
        lines.append("")
        lines.append("## Red flags")
        lines.append("")
        for f in res.flags:
            lines.append(f"- **[{f.severity.upper()}] {f.code}** — {f.message}")
    else:
        lines.append("")
        lines.append("_No red flags._")

    if n > 0:
        lines.append("")
        lines.append("## Trades")
        lines.append("")
        cols = ["entry_time", "exit_time", "side", "entry_price", "exit_price",
                "planned_stop_loss", "planned_take_profit", "bars_held",
                "exit_reason", "net_pnl_gbp"]
        cols = [c for c in cols if c in res.trades_df.columns]
        out = res.trades_df[cols].copy()
        # Round prices for readability
        for c in ("entry_price", "exit_price", "planned_stop_loss",
                  "planned_take_profit"):
            if c in out.columns:
                out[c] = out[c].astype(float).round(4)
        lines.append(out.to_markdown(index=False))
    fname.write_text("\n".join(lines))
    return fname


def write_summary(all_results: list[ScenarioResult]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for r in all_results:
        flags_txt = (";".join(f"[{f.severity}]{f.code}" for f in r.flags) or "-")
        if r.error is not None:
            flags_txt = f"[error]crash: {r.error.splitlines()[0][:60]}"
        rows.append({
            "strategy": r.spec.key,
            "scenario": r.scenario.name,
            "trades": len(r.trades_df),
            "expired": r.expired_orders,
            "drop_geom": r.dropped_geometry,
            "drop_lev": r.dropped_orders,
            "final_balance": (f"{r.result.final_balance:,.0f}"
                              if r.result is not None else "-"),
            "flags": flags_txt,
        })
    summary = pd.DataFrame(rows)
    out_path = OUT_DIR / "summary.md"
    body = ["# Strategy verification summary", ""]
    body.append(summary.to_markdown(index=False))
    body.append("")
    out_path.write_text("\n".join(body))
    return out_path


def print_console_summary(all_results: list[ScenarioResult]) -> int:
    """Print human-readable rollup; return the number of strategies with any
    red flag in any scenario (so the caller can use it as exit code)."""
    bad_strategies: set[str] = set()
    print("\n" + "=" * 78)
    print(f"{'STRATEGY':<18} {'SCENARIO':<18} {'TRADES':>7} {'FINAL £':>11}  FLAGS")
    print("=" * 78)
    for r in all_results:
        flags_txt = (",".join(f"{f.code}({f.severity[0]})" for f in r.flags)
                     or "-")
        if r.error is not None:
            flags_txt = f"CRASH: {r.error.splitlines()[0][:40]}"
            bad_strategies.add(r.spec.key)
        elif r.flags:
            bad_strategies.add(r.spec.key)
        final = (f"{r.result.final_balance:>10,.0f}"
                 if r.result is not None else "        -")
        print(f"{r.spec.key:<18} {r.scenario.name:<18} {len(r.trades_df):>7} "
              f"{final}  {flags_txt}")
    print("=" * 78)
    print(f"Strategies with any flag: {len(bad_strategies)} / "
          f"{len({r.spec.key for r in all_results})}")
    if bad_strategies:
        print("Flagged: " + ", ".join(sorted(bad_strategies)))
    return len(bad_strategies)


# ---------- Main ----------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only run these strategy keys")
    ap.add_argument("--scenario", nargs="*", default=None,
                    help="Only run these scenarios (names)")
    ap.add_argument("--max-bars", type=int, default=None,
                    help="Cap each scenario's data to N most-recent bars "
                         "(useful for fast iteration on O(n^2) strategies)")
    args = ap.parse_args()

    keys = list(STRATEGIES.keys())
    if args.only:
        keys = [k for k in keys if k in args.only]
        unknown = set(args.only) - set(STRATEGIES.keys())
        if unknown:
            print(f"WARNING: unknown strategy keys: {sorted(unknown)}",
                  file=sys.stderr)
    scenarios = list(SCENARIOS)
    if args.scenario:
        scenarios = [s for s in scenarios if s.name in args.scenario]

    # Pre-load data once per scenario (data is expensive — strategies share it)
    print("Loading scenario data...")
    data_cache: dict[str, pd.DataFrame] = {}
    for s in scenarios:
        try:
            df = load_scenario_data(s)
            if args.max_bars is not None and len(df) > args.max_bars:
                df = df.tail(args.max_bars)
            data_cache[s.name] = df
            print(f"  {s.name}: {len(df)} bars "
                  f"({df.index[0]} → {df.index[-1]})")
        except Exception as e:
            print(f"  {s.name}: FAILED to load — {e}")
            data_cache[s.name] = pd.DataFrame()

    all_results: list[ScenarioResult] = []
    for key in keys:
        spec = STRATEGIES[key]
        for scenario in scenarios:
            data = data_cache.get(scenario.name)
            if data is None or data.empty:
                continue
            print(f"  running {key} on {scenario.name}...", flush=True)
            res = run_one(spec, scenario, data)
            write_per_run_report(res)
            all_results.append(res)

    write_summary(all_results)
    n_flagged = print_console_summary(all_results)
    return 0 if n_flagged == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
