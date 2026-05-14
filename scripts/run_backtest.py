"""
Run a backtest end-to-end. CLI entry point.

Two modes:
  --mode single    one backtest on the full data range, no validation
  --mode validate  in-sample/out-of-sample split + walk-forward + Monte Carlo

Usage:
    # Simple smoke test (SMA on daily):
    python scripts/run_backtest.py --strategy sma --interval 1d

    # FVG with full validation:
    python scripts/run_backtest.py --strategy fvg --interval 15m --mode validate

    # Custom dates + tweaked params:
    python scripts/run_backtest.py --strategy fvg --interval 1h \\
        --start 2024-06-01 --fvg-r-target 1.5 --mode validate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fetcher import fetch
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, write_report, print_summary
from backtest.validation import run_full_validation, write_validation_report
from strategies import registry as reg
from config import REPORTS_DIR


BARS_PER_YEAR = {
    "1m": 252 * 510, "5m": 252 * 102, "15m": 252 * 34,
    "30m": 252 * 17, "1h": 252 * 8, "1d": 252, "1wk": 52,
}


def parse_param_overrides(override_str: str | None, spec: reg.StrategySpec) -> dict:
    """Parse 'key=value,key=value' into a dict, with type coercion from the spec."""
    overrides = {}
    if not override_str:
        return overrides
    type_map = {p.name: p.type for p in spec.params}
    for kv in override_str.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"Bad --params item: {kv!r}. Expected key=value.")
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        t = type_map.get(k)
        if t is None:
            raise ValueError(f"Unknown param '{k}' for {spec.label}. Valid: {list(type_map)}")
        if t == "int":
            overrides[k] = int(v)
        elif t == "float":
            overrides[k] = float(v)
        elif t == "bool":
            overrides[k] = v.lower() in ("1", "true", "yes", "y")
        else:
            overrides[k] = v
    return overrides


def main():
    strategy_keys = list(reg.STRATEGIES.keys())

    p = argparse.ArgumentParser(
        description="Run a backtest. Strategies + parameters are defined in strategies/registry.py.",
        epilog=f"Available strategies: {', '.join(strategy_keys)}",
    )
    p.add_argument("--strategy", choices=strategy_keys, default="fvg")
    p.add_argument("--mode", choices=["single", "validate"], default="single",
                   help="single = one backtest; validate = IS/OOS + walk-forward + Monte Carlo")
    p.add_argument("--ticker", default="^FTSE")
    p.add_argument("--interval", default="15m",
                   choices=["1m", "5m", "15m", "30m", "1h", "1d"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--params", default=None,
                   help='Override strategy params, format "key=value,key=value". '
                        'See registry.py for valid keys per strategy.')

    # Validation params
    p.add_argument("--oos-fraction", type=float, default=0.2)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--n-mc", type=int, default=1000)

    args = p.parse_args()
    spec = reg.get(args.strategy)
    overrides = parse_param_overrides(args.params, spec)
    final_params = {**spec.defaults(), **overrides}
    print(f"Strategy: {spec.label}  params: {final_params}")

    print(f"Fetching {args.ticker} @ {args.interval} ...")
    data = fetch(ticker=args.ticker, interval=args.interval,
                 start=args.start, end=args.end)
    print(f"  -> {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    factory = lambda: spec.build(**final_params)
    warmup = spec.warmup_bars
    bpy = BARS_PER_YEAR.get(args.interval, 252)
    run_name = f"{args.strategy}_{args.interval}"

    if args.mode == "single":
        print(f"Running {args.strategy} on {args.interval} (single mode) ...")
        result = run_backtest(data, factory(), warmup_bars=warmup)
        metrics = compute_metrics(result, bars_per_year=bpy)
        print_summary(metrics)
        report = write_report(result, metrics, REPORTS_DIR, run_name=run_name)
        print(f"\nReport: {report}")
        return

    # ---- Validation mode ----
    print(f"Running {args.strategy} on {args.interval} (validation mode) ...")
    print(f"  IS/OOS split: {(1-args.oos_fraction):.0%} / {args.oos_fraction:.0%}")
    print(f"  Walk-forward folds: {args.n_folds}")
    print(f"  Monte Carlo simulations: {args.n_mc}")

    is_result, oos_result, report = run_full_validation(
        data,
        strategy_factory=factory,
        oos_fraction=args.oos_fraction,
        n_folds=args.n_folds,
        n_mc_simulations=args.n_mc,
        warmup_bars=warmup,
        bars_per_year=bpy,
    )

    is_m, oos_m = report.in_sample_metrics, report.out_of_sample_metrics
    drift = report.is_to_oos_drift() or 0.0
    consistency = report.consistency_score()

    print("\n" + "=" * 60)
    print(f"  {'In-sample':>20} {'Out-of-sample':>16}")
    print(f"  Trades:        {is_m.num_trades:>10}        {oos_m.num_trades:>10}")
    print(f"  Return:        {is_m.total_return_pct:>+9.2f}%       {oos_m.total_return_pct:>+9.2f}%")
    print(f"  Sharpe:        {is_m.sharpe:>10.2f}        {oos_m.sharpe:>10.2f}")
    print(f"  Profit factor: {is_m.profit_factor:>10.2f}        {oos_m.profit_factor:>10.2f}")
    print(f"  Win rate:      {is_m.win_rate_pct:>9.1f}%       {oos_m.win_rate_pct:>9.1f}%")
    print(f"  Max DD:        {is_m.max_drawdown_pct:>9.2f}%       {oos_m.max_drawdown_pct:>9.2f}%")
    print("=" * 60)
    print(f"  Sharpe drift (OOS - IS):  {drift:+.2f}")
    print(f"  Walk-forward consistency: {consistency:.0%}")
    print(f"  MC final balance p5/p50/p95: "
          f"£{report.monte_carlo.final_balance_p5:,.0f} / "
          f"£{report.monte_carlo.final_balance_p50:,.0f} / "
          f"£{report.monte_carlo.final_balance_p95:,.0f}")
    print("=" * 60)

    write_report(is_result, is_m, REPORTS_DIR, run_name=f"{run_name}_is")
    write_report(oos_result, oos_m, REPORTS_DIR, run_name=f"{run_name}_oos")
    report_path = write_validation_report(
        is_result, oos_result, report, REPORTS_DIR, run_name=run_name
    )
    print(f"\nValidation report: {report_path}")


if __name__ == "__main__":
    main()
