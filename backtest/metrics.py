"""
Performance metrics + report writer.

All metrics are computed honestly:
  - Sharpe/Sortino use bar-frequency returns scaled to annual
  - Max drawdown is peak-to-trough on the equity curve
  - Profit factor = gross wins / gross losses
  - Costs are reported separately so you can see how much edge they ate
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest.engine import BacktestResult


@dataclass
class Metrics:
    starting_balance: float
    final_balance: float
    total_return_pct: float
    cagr_pct: float | None
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    max_drawdown_gbp: float
    profit_factor: float
    win_rate_pct: float
    expectancy_r: float
    num_trades: int
    avg_bars_held: float
    total_gross_pnl: float
    total_spread_cost: float
    total_slippage_cost: float
    total_financing_cost: float


def compute_metrics(result: BacktestResult, bars_per_year: int = 252) -> Metrics:
    eq = result.equity_curve
    trades = result.trades_df

    starting = result.starting_balance
    final = result.final_balance
    total_return_pct = (final / starting - 1.0) * 100.0

    # CAGR
    cagr_pct = None
    if len(eq) > 1:
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        if years > 0 and starting > 0:
            try:
                cagr_pct = ((final / starting) ** (1 / years) - 1) * 100.0
            except (ValueError, ZeroDivisionError):
                cagr_pct = None

    # Sharpe / Sortino on equity returns
    returns = eq.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(bars_per_year)
    else:
        sortino = 0.0

    # Drawdown
    running_max = eq.cummax()
    dd = eq - running_max
    dd_pct = (eq / running_max - 1.0) * 100.0
    max_dd_gbp = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = float(dd_pct.min()) if len(dd_pct) else 0.0

    # Trade stats
    if len(trades) > 0:
        wins = trades[trades["net_pnl_gbp"] > 0]["net_pnl_gbp"]
        losses = trades[trades["net_pnl_gbp"] < 0]["net_pnl_gbp"]
        gross_wins = wins.sum()
        gross_losses = abs(losses.sum())
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")
        win_rate_pct = (len(wins) / len(trades)) * 100.0
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = abs(losses.mean()) if len(losses) else 0.0
        # Expectancy in R: avg P&L per trade / avg loss
        expectancy_r = trades["net_pnl_gbp"].mean() / avg_loss if avg_loss > 0 else 0.0
        total_gross = trades["gross_pnl_gbp"].sum()
        total_spread = trades["spread_cost_gbp"].sum()
        total_slip = trades["slippage_cost_gbp"].sum()
        total_fin = trades["financing_cost_gbp"].sum()
        avg_bars = trades["bars_held"].mean()
    else:
        profit_factor = 0.0
        win_rate_pct = 0.0
        expectancy_r = 0.0
        total_gross = total_spread = total_slip = total_fin = 0.0
        avg_bars = 0.0

    return Metrics(
        starting_balance=starting,
        final_balance=final,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown_pct=max_dd_pct,
        max_drawdown_gbp=max_dd_gbp,
        profit_factor=float(profit_factor),
        win_rate_pct=win_rate_pct,
        expectancy_r=float(expectancy_r),
        num_trades=int(len(trades)),
        avg_bars_held=float(avg_bars),
        total_gross_pnl=float(total_gross),
        total_spread_cost=float(total_spread),
        total_slippage_cost=float(total_slip),
        total_financing_cost=float(total_fin),
    )


def write_report(result: BacktestResult, metrics: Metrics, output_dir: Path,
                 run_name: str | None = None) -> Path:
    """Write a markdown report + plots + CSV of trades. Returns report path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or f"{result.strategy_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    # Save trades CSV
    trades_path = output_dir / f"{run_name}_trades.csv"
    result.trades_df.to_csv(trades_path)

    # Plot equity curve + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(result.equity_curve.index, result.equity_curve.values, color="steelblue")
    ax1.axhline(result.starting_balance, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("Equity (£)")
    ax1.set_title(f"{result.strategy_name} — equity curve")
    ax1.grid(alpha=0.3)

    running_max = result.equity_curve.cummax()
    dd_pct = (result.equity_curve / running_max - 1.0) * 100
    ax2.fill_between(dd_pct.index, dd_pct.values, 0, color="indianred", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / f"{run_name}_equity.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)

    # Markdown report
    report_path = output_dir / f"{run_name}.md"
    m = metrics
    cagr_str = f"{m.cagr_pct:.2f}%" if m.cagr_pct is not None else "n/a"
    lines = [
        f"# Backtest report: {result.strategy_name}",
        f"",
        f"**Run:** `{run_name}`  ",
        f"**Period:** {result.equity_curve.index[0]} → {result.equity_curve.index[-1]}  ",
        f"**Bars processed:** {result.bars_processed}",
        f"",
        f"## Headline numbers",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Starting balance | £{m.starting_balance:,.2f} |",
        f"| Final balance | £{m.final_balance:,.2f} |",
        f"| Total return | {m.total_return_pct:+.2f}% |",
        f"| CAGR | {cagr_str} |",
        f"| Sharpe | {m.sharpe:.2f} |",
        f"| Sortino | {m.sortino:.2f} |",
        f"| Max drawdown | {m.max_drawdown_pct:.2f}% (£{m.max_drawdown_gbp:,.2f}) |",
        f"| Profit factor | {m.profit_factor:.2f} |",
        f"| Win rate | {m.win_rate_pct:.1f}% |",
        f"| Expectancy (R) | {m.expectancy_r:+.2f} |",
        f"| Number of trades | {m.num_trades} |",
        f"| Avg bars held | {m.avg_bars_held:.1f} |",
        f"",
        f"## Cost breakdown",
        f"",
        f"| Component | £ |",
        f"|---|---|",
        f"| Gross P&L (before costs) | £{m.total_gross_pnl:+,.2f} |",
        f"| Spread cost | -£{m.total_spread_cost:,.2f} |",
        f"| Slippage cost | -£{m.total_slippage_cost:,.2f} |",
        f"| Financing cost | -£{m.total_financing_cost:,.2f} |",
        f"| **Net P&L** | **£{(m.final_balance - m.starting_balance):+,.2f}** |",
        f"",
        f"## Honest reading",
        f"",
        *_honest_reading(m),
        f"",
        f"![equity curve]({plot_path.name})",
        f"",
        f"Trades log: `{trades_path.name}`",
    ]
    report_path.write_text("\n".join(lines))
    return report_path


def _pct(num: float, denom: float) -> float:
    return (num / denom * 100) if denom else 0.0


def _honest_reading(m: "Metrics") -> list[str]:
    """Conditional warnings — only fires the ones that actually apply to this run."""
    lines: list[str] = []
    cost_total = m.total_spread_cost + m.total_slippage_cost + m.total_financing_cost
    if abs(m.total_gross_pnl) > 0:
        lines.append(f"- Costs ate **{_pct(cost_total, abs(m.total_gross_pnl)):.1f}%** of |gross P&L|.")
    # Which cost dominated?
    if cost_total > 0:
        biggest = max(
            ("spread", m.total_spread_cost),
            ("slippage", m.total_slippage_cost),
            ("financing", m.total_financing_cost),
            key=lambda x: x[1],
        )
        lines.append(f"- Dominant cost component: **{biggest[0]}** (£{biggest[1]:,.2f}).")
    if m.sharpe < 1.0:
        lines.append(f"- Sharpe of {m.sharpe:.2f} is below 1.0 — generally not worth trading after costs.")
    if m.max_drawdown_pct < -30.0:
        lines.append(f"- Max drawdown of {m.max_drawdown_pct:.1f}% is usually psychologically intolerable in live trading.")
    if m.num_trades < 30:
        lines.append(f"- Only {m.num_trades} trades — too few for metrics to be statistically reliable.")
    if m.profit_factor < 1.0 and m.num_trades > 0:
        lines.append(f"- Profit factor {m.profit_factor:.2f} < 1.0 — losers outweigh winners.")
    if m.win_rate_pct < 30 and m.profit_factor > 1.5:
        lines.append(f"- Low win rate ({m.win_rate_pct:.1f}%) but high profit factor — typical trend-follower; needs psychological tolerance for losing streaks.")
    return lines


def print_summary(metrics: Metrics) -> None:
    m = metrics
    cagr_str = f"{m.cagr_pct:.2f}%" if m.cagr_pct is not None else "n/a"
    print("="*60)
    print(f"  Starting balance:    £{m.starting_balance:>12,.2f}")
    print(f"  Final balance:       £{m.final_balance:>12,.2f}")
    print(f"  Total return:        {m.total_return_pct:>+12.2f}%")
    print(f"  CAGR:                {cagr_str:>13}")
    print(f"  Sharpe:              {m.sharpe:>12.2f}")
    print(f"  Sortino:             {m.sortino:>12.2f}")
    print(f"  Max drawdown:        {m.max_drawdown_pct:>12.2f}%")
    print(f"  Profit factor:       {m.profit_factor:>12.2f}")
    print(f"  Win rate:            {m.win_rate_pct:>12.1f}%")
    print(f"  Expectancy (R):      {m.expectancy_r:>+12.2f}")
    print(f"  Trades:              {m.num_trades:>12}")
    print(f"  Avg bars held:       {m.avg_bars_held:>12.1f}")
    print(f"  ---- costs ----")
    print(f"  Gross P&L:           £{m.total_gross_pnl:>+12,.2f}")
    print(f"  Spread:             -£{m.total_spread_cost:>12,.2f}")
    print(f"  Slippage:           -£{m.total_slippage_cost:>12,.2f}")
    print(f"  Financing:          -£{m.total_financing_cost:>12,.2f}")
    print("="*60)
