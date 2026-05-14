"""
Parameter sweep CLI.

Two-step workflow:

  STEP 1 — sweep on in-sample (lots of combinations OK):
      python scripts/sweep.py --strategy fvg --interval 1h \\
          --params "min_gap_points=3,5,7,10;r_target=1.5,2.0,2.5,3.0"

  STEP 2 — pick ONE param set from the table you got back, then evaluate
  on out-of-sample (single combination, ONCE):
      python scripts/sweep.py --strategy fvg --interval 1h \\
          --params "min_gap_points=5;r_target=2.0" --evaluate-oos

The --evaluate-oos flag enforces a single value per param so you can't
accidentally turn the OOS check into another sweep.

The OOS verdict is final. If it's bad, the right move is "this strategy
doesn't work" — not "let me try different params." That second move turns
OOS into IS and the whole exercise becomes meaningless.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fetcher import fetch
from backtest.sweep import grid_sweep, evaluate_oos
from backtest.metrics import print_summary
from backtest.validation import adaptive_walk_forward, deflated_sharpe_ratio
from strategies import registry as reg
from config import REPORTS_DIR
import numpy as np


BARS_PER_YEAR = {
    "1m": 252 * 510, "5m": 252 * 102, "15m": 252 * 34,
    "30m": 252 * 17, "1h": 252 * 8, "1d": 252,
}


def parse_param_grid(s: str, spec: reg.StrategySpec) -> dict[str, list]:
    """
    Parse 'key=v1,v2,v3;key2=v4,v5' into {key: [v1,v2,v3], key2: [v4,v5]}.
    Coerces values to the type declared in the strategy's param spec.
    """
    type_map = {p.name: p.type for p in (spec.params or [])}
    grid = {}
    for clause in s.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        if "=" not in clause:
            raise ValueError(f"Bad --params clause: {clause!r} (expected key=v1,v2,...)")
        k, v = clause.split("=", 1)
        k = k.strip()
        if k not in type_map:
            raise ValueError(
                f"Unknown param '{k}' for strategy '{spec.label}'. "
                f"Valid: {list(type_map)}"
            )
        values = [x.strip() for x in v.split(",") if x.strip()]
        t = type_map[k]
        if t == "int":
            values = [int(x) for x in values]
        elif t == "float":
            values = [float(x) for x in values]
        elif t == "bool":
            values = [x.lower() in ("1", "true", "yes", "y") for x in values]
        grid[k] = values
    return grid


def main():
    p = argparse.ArgumentParser(
        description="Parameter sweep with built-in IS/OOS discipline.",
    )
    p.add_argument("--strategy", choices=list(reg.STRATEGIES.keys()), required=True)
    p.add_argument("--interval", default="1h",
                   choices=["1m", "5m", "15m", "30m", "1h", "1d"])
    p.add_argument("--ticker", default="^FTSE")
    p.add_argument("--params", required=True,
                   help='Param grid in "key=v1,v2,v3;key2=v4,v5" format.')
    p.add_argument("--oos-fraction", type=float, default=0.2)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-trades", type=int, default=30,
                   help="Filter out runs with fewer trades — too few = noise.")
    p.add_argument("--evaluate-oos", action="store_true",
                   help="Skip sweep. Evaluate ONE param set on OOS only. "
                        "Each --params clause must have exactly one value.")
    p.add_argument("--adaptive", action="store_true",
                   help="Run adaptive walk-forward instead of basic sweep. "
                        "Re-fits params each fold, evaluates on the next fold.")
    p.add_argument("--optimize-by", default="sharpe",
                   choices=["sharpe", "profit_factor", "total_return_pct"],
                   help="Which IS metric to optimize during adaptive WF.")
    args = p.parse_args()

    spec = reg.get(args.strategy)
    grid = parse_param_grid(args.params, spec)
    bpy = BARS_PER_YEAR.get(args.interval, 252)

    print(f"Fetching {args.ticker} @ {args.interval}...")
    data = fetch(ticker=args.ticker, interval=args.interval)
    print(f"  -> {len(data)} bars")

    def factory_for(params: dict):
        full = {**spec.defaults(), **params}
        return lambda: spec.build(**full)

    # ---- Adaptive walk-forward mode -------------------------------------
    if args.adaptive:
        print(f"\n=== ADAPTIVE WALK-FORWARD ({args.n_folds} folds) ===")
        print(f"Strategy: {spec.label}, optimizing for {args.optimize_by}")
        print(f"Param grid: {grid}")
        print()
        result = adaptive_walk_forward(
            data,
            factory_for_params=factory_for,
            param_grid=grid,
            n_folds=args.n_folds,
            optimization_metric=args.optimize_by,
            warmup_bars=spec.warmup_bars,
            bars_per_year=bpy,
        )
        if not result.folds:
            print("No folds completed — strategy may not have produced enough trades.")
            return
        df = result.summary_table()
        print("Per-fold results (params chosen on train, evaluated on test):")
        print(df.to_string(index=False))
        agg = result.aggregate_test_metrics()
        print(f"\nAggregate across {agg['n_folds']} test folds:")
        for k, v in agg.items():
            print(f"  {k}: {v}")

        csv_path = REPORTS_DIR / f"adaptive_wf_{args.strategy}_{args.interval}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nFull results saved to {csv_path}")

        print()
        print("=" * 60)
        print("If test_consistency >= 0.6 and avg_test_sharpe > 0.5: real edge.")
        print("If params change wildly between folds but test still works: regime-adaptive.")
        print("If test fails despite train succeeding: the strategy doesn't generalize.")
        print("=" * 60)
        return

    # ---- Step 2: OOS evaluation of a single chosen set ------------------
    if args.evaluate_oos:
        if any(len(v) > 1 for v in grid.values()):
            print("ERROR: --evaluate-oos requires a SINGLE value per --params clause.")
            print(f"  You provided: {grid}")
            sys.exit(1)
        params = {k: v[0] for k, v in grid.items()}
        print(f"\n=== OUT-OF-SAMPLE EVALUATION ===")
        print(f"Strategy: {spec.label}")
        print(f"Params: {params}")
        print(f"OOS slice: last {args.oos_fraction:.0%} of data")
        print()
        factory = factory_for(params)
        result, metrics = evaluate_oos(
            data, factory, oos_fraction=args.oos_fraction,
            warmup_bars=spec.warmup_bars, bars_per_year=bpy,
        )
        print_summary(metrics)
        print()
        print("=" * 60)
        print("⚠️  This is the honest verdict for these specific params.")
        print("    Re-running with different params now would invalidate it.")
        print("=" * 60)
        return

    # ---- Step 1: in-sample sweep ----------------------------------------
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f"\nSweeping {n_combos} combinations on IS data only "
          f"(reserving last {args.oos_fraction:.0%} for OOS)...")

    def progress(i: int, n: int, params: dict):
        # Inline progress with carriage return
        msg = f"  [{i}/{n}] {params}"
        # Truncate if too long
        if len(msg) > 100:
            msg = msg[:97] + "..."
        print(msg.ljust(100), end="\r")

    sweep = grid_sweep(
        data,
        factory_for_params=factory_for,
        param_grid=grid,
        oos_fraction=args.oos_fraction,
        n_folds=args.n_folds,
        warmup_bars=spec.warmup_bars,
        bars_per_year=bpy,
        progress_callback=progress,
    )
    print()  # newline after progress
    print(f"  -> {len(sweep.runs)} successful, {sweep.n_failures} failed")

    df = sweep.summary_df()
    csv_path = REPORTS_DIR / f"sweep_{args.strategy}_{args.interval}.csv"
    df.to_csv(csv_path, index=False)

    # Filter for stat-significant trade counts
    valid = df[df["trades"] >= args.min_trades]
    if len(valid) == 0:
        print(f"\n⚠️  No combinations produced ≥{args.min_trades} trades. "
              f"Try a strategy/interval that trades more frequently.")
        return

    print(f"\n=== TOP {args.top_n} BY SHARPE (in-sample, ≥{args.min_trades} trades) ===")
    print(valid.sort_values("sharpe", ascending=False).head(args.top_n).to_string(index=False))

    print(f"\n=== TOP {args.top_n} BY WALK-FORWARD CONSISTENCY ===")
    print(valid.sort_values("wf_consistency", ascending=False).head(args.top_n).to_string(index=False))

    print(f"\n=== TOP {args.top_n} BY PROFIT FACTOR ===")
    print(valid.sort_values("profit_factor", ascending=False).head(args.top_n).to_string(index=False))

    # ---- Deflated Sharpe Ratio for the BEST run -------------------------
    # Adjusts for the fact that you tested many combinations
    sharpes = [r.is_metrics.sharpe for r in sweep.runs
               if r.is_metrics.num_trades >= args.min_trades]
    if len(sharpes) >= 2:
        sr_variance = float(np.var(sharpes))
        best_run = max(sweep.runs,
                       key=lambda r: r.is_metrics.sharpe if r.is_metrics.num_trades >= args.min_trades else -1)
        # Need the actual trades from the best run — re-run it on IS data
        from backtest.engine import run_backtest as _rb
        from backtest.validation import holdout_split as _hs
        in_sample, _ = _hs(data, args.oos_fraction)
        best_factory = factory_for(best_run.params)
        best_result = _rb(in_sample, best_factory(), warmup_bars=spec.warmup_bars)
        dsr = deflated_sharpe_ratio(
            best_result.trades_df,
            n_trials=len(sharpes),
            sharpe_variance_across_trials=sr_variance,
        )
        if not np.isnan(dsr):
            print(f"\n=== DEFLATED SHARPE RATIO (multiple-testing corrected) ===")
            print(f"Best params: {best_run.params}")
            print(f"  Raw Sharpe: {best_run.is_metrics.sharpe:.3f}")
            print(f"  Tested {len(sharpes)} combinations; Sharpe variance: {sr_variance:.4f}")
            print(f"  DSR: {dsr:.1%}")
            if dsr > 0.95:
                print(f"  ✅ Strong evidence the best is genuinely better than random.")
            elif dsr > 0.5:
                print(f"  🟡 Moderate evidence; not conclusive.")
            else:
                print(f"  ⚠️  Weak — best result may just be the lucky tail of {len(sharpes)} trials.")

    print(f"\nFull results saved to {csv_path}")
    print()
    print("=" * 60)
    print("NEXT STEP — pick ONE param set, then run OOS evaluation:")
    print(f"  python scripts/sweep.py --strategy {args.strategy} --interval {args.interval} \\")
    print(f"      --params 'key1=val;key2=val' --evaluate-oos")
    print()
    print("Discipline: pick by walk-forward consistency, not raw Sharpe.")
    print("A param set with Sharpe 2.0 in 1 fold and -1.0 in 3 others is overfit.")
    print("=" * 60)


if __name__ == "__main__":
    main()
