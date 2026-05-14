# UK 100 backtester

A small, honest event-driven backtester for trading the UK 100 (FTSE 100)
via IG spread bet. Built to validate strategies before risking real money.

## What this is — and isn't

**Is:** a transparent ~600-line Python project you can read top-to-bottom.
It pulls FTSE 100 history, simulates IG spread bet costs (spread + slippage +
SONIA-based overnight financing), runs strategies bar-by-bar with no
look-ahead, and reports honest performance metrics.

**Isn't:** a guaranteed profitable system, an HFT framework, or a substitute
for understanding what you're trading. Most strategies you test in here will
not survive realistic costs. That is the point.

## Quick start

```bash
cd ~/Documents/Finance/uk100-backtest
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt

# Run the smoke-test backtest (SMA crossover on FTSE 100 daily, ~20 years)
python scripts/run_backtest.py
```

The first run downloads data and caches it under `data/cache/` as parquet,
so subsequent runs are fast and work offline.

## Project layout

```
uk100-backtest/
├── config.py                  # cost model + account defaults — audit these first
├── data/
│   ├── fetcher.py             # yfinance pulls + parquet cache
│   └── cache/                 # gitignored
├── backtest/
│   ├── broker.py              # IG spread bet execution + cost application
│   ├── engine.py              # event-driven bar loop
│   └── metrics.py             # Sharpe, drawdown, profit factor, report writer
├── strategies/
│   └── sma_crossover.py       # smoke-test strategy — NOT meant to be profitable
├── scripts/
│   └── run_backtest.py        # CLI entry point
└── reports/                   # markdown + PNG + CSV per run
```

## Reading order (~30 mins)

1. **`config.py`** — the cost assumptions live here. Everything downstream
   depends on these being approximately right. If IG's actual financing is
   different from `sonia_annual + admin_annual`, this is where to fix it.

2. **`backtest/broker.py`** — how the spread bet is executed. Pay attention
   to how spread, slippage, and overnight financing are applied.

3. **`backtest/engine.py`** — the bar loop. The execution model (decision on
   bar `i`, fill at bar `i+1`'s open) is the most important detail to
   understand.

4. **`backtest/metrics.py`** — the metrics. Read the docstrings; the
   formulas are standard but worth knowing.

5. **`strategies/sma_crossover.py`** — example strategy. Use this as the
   template for our real strategies (FVG, liquidity sweep, etc.).

## Honest expectations

- A Sharpe ratio under 1.0 is generally not worth trading after costs.
- A profit factor under 1.3 in backtest usually doesn't survive forward.
- A max drawdown over 30% is psychologically intolerable for most retail traders.
- Number of trades under 30 means metrics are statistically meaningless.

## Things to add later

- Walk-forward validation (train on year 1, test on year 2, slide, repeat)
- Monte Carlo trade-order shuffling to estimate metric confidence intervals
- Multi-timeframe support (e.g. HTF context + LTF entry)
- IG REST API integration for paper trading (once demo account is set up)
- Better intraday data source (Databento, Polygon, or IG's own historical)

## License / disclaimer

Personal use. Spread betting is a high-risk leveraged product; the FCA reports
that ~70–80% of retail CFD/spread bet accounts lose money. Past backtest
performance is not indicative of future results — and overfit backtests are
the rule, not the exception.
