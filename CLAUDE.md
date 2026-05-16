# UK 100 Backtester — Project Context

You are continuing a multi-month project to build an honest backtester for
the UK 100 (FTSE 100) via IG spread bet. The user's eventual goal is to
find a strategy worth paper trading, then live trading.

**Read this whole file before doing anything else.** It captures the design
decisions, conventions, gotchas, and current state of the system. Most
"why is it like this?" questions have answers in here.

---

## Project goal

Build a backtester that is honest enough to detect when a strategy has real
edge vs. when it just looks good due to data-mining, optimistic cost
modelling, or look-ahead bias. The user is a UK retail trader; the product
they'd eventually trade is IG spread bet on UK 100 (epic `IX.D.FTSE.DAILY.IP`).

**Strategy R&D itself is a separate concern, often happening in a separate
conversation in the same project.** This file is about the infrastructure.

---

## Quick start

```bash
cd ~/Developer/UK-100-Backtest/uk100-backtest
source .venv/bin/activate
pip install -r requirements.txt        # syncs any missing deps
pytest tests/ -v                        # ~125 tests; all should pass
streamlit run app.py                    # opens at http://localhost:8501
```

For IG data (real bid/ask spread, ~2 years of intraday vs yfinance's 60 days):

```bash
python scripts/ig_test.py               # verify .env credentials work
```

The `.env` file (gitignored) contains:

```
IG_USERNAME=...   # the demo LOGIN username, not email
IG_PASSWORD=...
IG_API_KEY=...
IG_ACCOUNT_ID=Z6BB5G  # spread bet account (CFD account is Z6BB5E)
IG_ENV=demo
```

---

## Architecture map

```
uk100-backtest/
├── config.py                    # cost model + account defaults — audit first
├── app.py                       # Streamlit UI (~1300 lines, single file)
├── data/
│   ├── fetcher.py               # yfinance pull + dispatch to IG when needed
│   └── ig_fetcher.py            # IG REST API, parquet cache, Spread column
├── backtest/
│   ├── engine.py                # event-driven bar loop, no look-ahead
│   ├── broker.py                # multi-position, partial exits, pending orders
│   ├── indicators.py            # ATR, RSI, BB, MACD, EMA, VWAP, FVG detection
│   ├── exits.py                 # trailing stop factories (ATR, chandelier, breakeven)
│   ├── metrics.py               # Sharpe, Sortino, drawdown, profit factor, plot writer
│   ├── validation.py            # holdout, walk-forward, MC, bootstrap, PSR, DSR, adaptive WF
│   ├── attribution.py           # slice trades by hour/day/month/side/exit_reason
│   ├── sweep.py                 # grid search with IS/OOS discipline
│   ├── optuna_search.py         # TPE Bayesian search (replaces grid for big spaces)
│   └── run_history.py           # SQLite DB of every backtest's config + outcome
├── strategies/
│   ├── registry.py              # central StrategySpec registry — add new strategies here
│   ├── _helpers.py              # risk_based_stake, in_session, trailing_swing, etc.
│   ├── ensemble.py              # VoteEnsemble + FilterEnsemble (poll proposed_direction)
│   ├── sma_crossover.py         # smoke-test strategy
│   ├── fvg_retest.py            # FVG limit-order entry — most-developed strategy
│   ├── fvg_scale_out.py         # FVG with partial exits + ATR trailing
│   ├── bpr.py                   # Balanced Price Range (overlapping FVGs)
│   ├── orb.py                   # Opening Range Breakout
│   ├── liquidity_sweep.py       # SMC liquidity sweep
│   ├── donchian.py              # Turtle-style channel breakout
│   ├── bb_reversion.py          # Bollinger Band mean reversion
│   ├── vwap_reversion.py        # day-trade VWAP mean reversion
│   └── rsi_reversion.py         # RSI hook from oversold/overbought
├── scripts/
│   ├── run_backtest.py          # CLI: single + validation modes
│   ├── sweep.py                 # CLI: grid sweep + OOS evaluation
│   ├── optuna_sweep.py          # CLI: Bayesian sweep
│   └── ig_test.py               # verify IG demo credentials + fetch sample
├── tests/                       # ~125 tests; run with `pytest tests/ -v`
├── reports/                     # markdown + PNG + CSV per backtest run
├── ASSUMPTIONS.md               # WHAT the backtester does/doesn't model — read this
├── README.md                    # setup + reading order
├── requirements.txt
├── run_history.db               # SQLite, auto-created
└── .env                         # IG credentials, gitignored
```

---

## What we've built (chronological summary)

1. **Phase 1: Foundation** — data fetcher, event-driven engine, IG cost model
   (1.5pt spread default, SONIA+admin financing), basic metrics, SMA smoke test.
2. **Phase 2: FVG** — `detect_fvg` indicator, FvgRetest strategy (originally
   market-order, later rewritten to use limit orders).
3. **Phase 3: Validation** — holdout, walk-forward, Monte Carlo shuffle.
4. **Phase 4: UI** — Streamlit app with single backtest, full validation,
   parameter sweep, adaptive walk-forward, trade inspector (with indicator
   overlays), custom ensemble builder.
5. **Phase 5: More strategies** — ORB, Liquidity Sweep, Donchian, BPR, BB,
   VWAP, RSI reversion, plus FvgScaleOut as a demo of the new engine features.
6. **Phase 6: Multi-position + partial exits + trailing stops** — broker
   refactored, OpenPosition has remaining_stake, scale_out method, trailing
   stop callback on each position.
7. **Phase 7: IG integration** — `data/ig_fetcher.py` uses `trading-ig`
   library, parquet cache, emits per-bar Spread column.
8. **Phase 8: Variable cost model** — spread is per-bar from IG data (with
   yfinance fallback to flat config). Slippage scales with spread.
9. **Phase 9: Pending orders** — broker has `place_pending_order`, the engine
   checks triggers each bar. FvgRetest rewritten to use limits.
10. **Phase 10: Statistical rigour** — bootstrap CIs, Probabilistic Sharpe,
    Deflated Sharpe (multiple-testing correction), Adaptive Walk-Forward.
11. **Phase 11: Optuna** — TPE Bayesian search, composite objective with
    trade-count penalty to prevent gaming. CLI + UI mode.
12. **Phase 12: Attribution + history** — per-hour/day/month/side/exit-reason
    breakdowns; SQLite run history database.
13. **Phase 13: `proposed_direction` refactor** — every strategy has a pure
    stateless method that ensembles poll. Cleaner architecture; MockBroker deleted.

---

## Conventions

- **Tests first when refactoring.** Engine and broker have ~50 tests between
  them. Run `pytest tests/ -v` before claiming work is done.
- **Strategies follow the Strategy protocol:** `on_bar(history, broker) -> Signal`
  AND `proposed_direction(history) -> "long"|"short"|"none"`. The first is for
  standalone execution, the second is for being polled by ensembles.
- **All cost / risk constants live in `config.py`.** Never hardcode them.
- **Use the registry pattern.** Adding a new strategy = file in `strategies/`
  + entry in `strategies/registry.py`. UI + CLI pick it up automatically.
- **No look-ahead.** Strategies must only use `history.iloc[:i+1]`. The engine
  enforces this by passing only the visible slice.
- **Position sizing uses `risk_based_stake` from `_helpers.py`** — leverage-aware,
  honours `ACCOUNT.risk_per_trade_pct` and `ACCOUNT.leverage_cap`. Never inline
  the calculation.
- **Self-check assertions stay.** The engine asserts the accounting identity
  at end of every run; if it ever fires, that's a real bug.
- **Markdown reports go to `reports/`** with the strategy name + timestamp.
  These are gitignored.

---

## Known gotchas

- **`FvgRetest` places limit orders directly via the broker, not via Signal.**
  This is why it has both `on_bar` (places limits, returns mostly `noop`) AND
  `proposed_direction` (for ensemble polling). When an ensemble uses FvgRetest,
  the ensemble takes a market entry at next-bar-open based on the polled
  direction — fills are different (worse) than the standalone limit-order
  version. Same trade-off applies to FvgScaleOut and BPR.
- **Param types degrade through DataFrames.** The sweep results table stores
  ints as numpy float64. The UI now coerces back via the spec's ParamSpec
  type before evaluating on OOS. If you see `iloc[<float>]` errors, this is
  why — fix by coercing or by `int()`-casting inside the strategy.
- **`broker.close()` has two calling conventions** for backward compat:
  `close(time, price, reason=)` (legacy, single-position only) and
  `close(position_or_id, time, price, reason=)` (multi-position). Auto-detected.
- **Volume is 0 for FTSE 100 from IG** (indices don't report volume). VWAP-based
  strategies behave slightly differently on IG vs yfinance data (yfinance has
  synthetic volume).
- **`max_concurrent_positions=1` by default** in `config.py`. Most strategies
  assume single-position semantics. Increase only if you've designed for it.
- **Streamlit's `width="stretch"`** replaces the deprecated `use_container_width=True`.
  Don't reintroduce the old kwarg.
- **IG weekly historical data allowance is 10,000 bars** on demo. Cache keyed by
  `(epic, resolution, num_points)` — match all three for cache hits. Re-fetching
  with different `num_points` burns the allowance.
- **RSI on a pure monotonic series saturates to 100 or 0** — the indicator has a
  special case for `avg_loss == 0` (and a test for it). Don't "fix" the special case.

---

## What's NOT modelled (read ASSUMPTIONS.md for full list)

- Requoting / dealer rejection
- Order book depth / queue position
- News-bar microstructure (within-bar spread spikes)
- Latency between signal and fill
- Your own psychology
- Correlated drawdowns across multiple live strategies

These are inherent gaps in any retail backtester, not fixable without tick
data + actual broker API live tests.

---

## Open items / what's next

Roughly prioritised:

- **Use the system to actually evaluate strategies.** The honest next move
  isn't more backtester work — it's paper trading a candidate strategy on IG
  demo. The bridge for that doesn't exist yet (Phase 14 if we go).
- **News calendar integration** — fetch FOMC/NFP/BoE/CPI times, let strategies
  avoid trading 30 min before/after. Would meaningfully improve intraday cost
  realism.
- **Empirical slippage calibration** — once paper trading runs, compare actual
  fills to backtest predictions and update `slip_spread_multiplier`.
- **Parallel sweeps** — requires strategies to be picklable. Use Optuna as a
  workaround for big searches.
- **PDF export of validation reports** — nice-to-have, not impactful.

---

## Honest status

The backtester is now better than typical retail tools at:

- Catching look-ahead and accounting bugs (runtime assertions)
- Modelling per-bar spread from real broker data
- Multiple-testing correction (Deflated Sharpe Ratio after sweeps)
- IS/OOS discipline enforced by the workflow

The gaps that remain are mostly inherent to bar-resolution data + retail
data feeds. No more backtester polish will close them without tick data.

For any strategy that survives our validation gauntlet (PSR > 0.95, walk-
forward consistency > 60%, OOS Sharpe similar to IS), the next step is paper
trading on the IG demo account, NOT more backtest iteration.
