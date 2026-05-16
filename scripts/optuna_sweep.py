"""
Bayesian parameter sweep via Optuna's TPE sampler.

Way more efficient than grid sweep for big param spaces — typically finds
near-optimal configs in 50-200 trials where grid would need thousands.

Usage:
    python scripts/optuna_sweep.py --strategy fvg --interval 1h --n-trials 100

Then pick ONE config from the printed top-10 and evaluate on OOS:
    python scripts/sweep.py --strategy fvg --interval 1h \\
        --params 'min_gap_points=5;r_target=2.0' --evaluate-oos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fetcher import fetch
from backtest.optuna_search import run_optuna_study
from strategies import registry as reg
from config import REPORTS_DIR


BARS_PER_YEAR = {
    "1m": 252 * 510, "5m": 252 * 102, "15m": 252 * 34,
    "30m": 252 * 17, "1h": 252 * 8, "1d": 252,
}


def main():
    p = argparse.ArgumentParser(
        description="Bayesian parameter sweep via Optuna's TPE sampler.",
    )
    p.add_argument("--strategy", choices=list(reg.STRATEGIES.keys()), required=True)
    p.add_argument("--interval", default="1h",
                   choices=["1m", "5m", "15m", "30m", "1h", "1d"])
    p.add_argument("--ticker", default="^FTSE")
    p.add_argument("--source", default="yfinance", choices=["yfinance", "ig"])
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--timeout", type=float, default=None,
                   help="Max seconds to run (overrides --n-trials if hit).")
    p.add_argument("--oos-fraction", type=float, default=0.2)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--target-trades", type=int, default=50,
                   help="Score-penalty floor: trials with fewer trades get scaled down.")
    p.add_argument("--metric", default="wf_consistency",
                   choices=["wf_consistency", "sharpe", "profit_factor"],
                   help="Which dimension to optimise (composite score uses + trade penalty).")
    p.add_argument("--top-n", type=int, default=10)

    args = p.parse_args()
    spec = reg.get(args.strategy)

    print(f"Fetching {args.ticker} @ {args.interval} from {args.source}...")
    data = fetch(ticker=args.ticker, interval=args.interval, source=args.source)
    print(f"  -> {len(data)} bars")
    bpy = BARS_PER_YEAR.get(args.interval, 252)

    def factory_for(params: dict):
        full = {**spec.defaults(), **params}
        return lambda: spec.build(**full)

    def progress(i: int, n: int, params: dict):
        msg = f"  [{i}/{n}] {params}"
        if len(msg) > 100:
            msg = msg[:97] + "..."
        print(msg.ljust(100), end="\r")

    print(f"\nRunning {args.n_trials} Optuna trials, optimising '{args.metric}' "
          f"(target ≥{args.target_trades} trades per config)...")
    result = run_optuna_study(
        data, spec,
        factory_for_params=factory_for,
        n_trials=args.n_trials,
        timeout=args.timeout,
        oos_fraction=args.oos_fraction,
        n_folds=args.n_folds,
        warmup_bars=spec.warmup_bars,
        bars_per_year=bpy,
        target_trades=args.target_trades,
        optimization_metric=args.metric,
        progress_callback=progress,
    )
    print()
    print(f"  -> {len(result.trials)} successful trials")
    print(f"  -> Best score: {result.best_score:.3f}")
    print(f"  -> Best params: {result.best_params}")

    df = result.summary_df()
    csv_path = REPORTS_DIR / f"optuna_{args.strategy}_{args.interval}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAll trials saved to {csv_path}")

    print(f"\n=== TOP {args.top_n} BY COMPOSITE SCORE ===")
    print(df.sort_values("score", ascending=False).head(args.top_n).to_string(index=False))

    print(f"\n=== TOP {args.top_n} BY RAW SHARPE (≥{args.target_trades} trades) ===")
    qualifying = df[df["trades"] >= args.target_trades]
    if not qualifying.empty:
        print(qualifying.sort_values("sharpe", ascending=False).head(args.top_n).to_string(index=False))
    else:
        print(f"(no trials with ≥{args.target_trades} trades)")

    print()
    print("=" * 60)
    print("NEXT STEP — pick ONE config and evaluate on out-of-sample:")
    print(f"  python scripts/sweep.py --strategy {args.strategy} --interval {args.interval} \\")
    print(f"      --params 'key=val;key=val' --evaluate-oos")
    print("=" * 60)


if __name__ == "__main__":
    main()
