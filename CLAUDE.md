# Backtester — Project Context

You are continuing a multi-month project building an honest backtester. The
user is a UK retail trader; the eventual product they'll trade is IG spread
bet — but the system is now **multi-asset**: UK 100, US stocks (TSLA, AAPL),
crypto (BTC), and forex all work via cost profiles.

**Read this whole file before doing anything else.** It captures the
design decisions, conventions, gotchas, and current state of the system.
Most "why is it like this?" questions have answers in here.

---

## Project goal

Build a backtester that is honest enough to detect when a strategy has
real edge vs. when it just looks good due to data-mining, optimistic
cost modelling, look-ahead bias, or any of the dozen subtle execution
bugs that plague typical retail backtesters.

**Strategy R&D itself is a separate concern**, often in a separate
conversation. This file is about the infrastructure.

---

## Quick start

```bash
cd ~/Developer/UK-100-Backtest/uk100-backtest
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v                # ~188 tests; all should pass
streamlit run app.py            # http://localhost:8501
```

`.env` (gitignored) holds IG and EODHD credentials. Sample test scripts:
`scripts/ig_test.py`, `scripts/eodhd_test.py`.

---

## Architecture map

```
uk100-backtest/
├── config.py                    # cost PROFILES dict, TRADING_TZ, ACCOUNT, trading_window_for()
├── app.py                       # Streamlit UI shell (~1500 lines)
├── app_graph_builder.py         # Sidebar graph-builder UI (trigger / supporters / vetoes)
├── data/
│   ├── fetcher.py               # yfinance dispatcher (+ IG / EODHD routing)
│   ├── ig_fetcher.py            # IG REST API, parquet cache, Spread column
│   ├── eodhd_fetcher.py         # EODHD API, chunked intraday fetcher
│   └── _tz.py                   # to_trading_tz: convert fetched data → user tz
├── backtest/
│   ├── engine.py                # event-driven bar loop, gap-aware order ops
│   ├── broker.py                # multi-position, partial exits, pending orders,
│   │                            #   gap-aware fills, same-bar SL/TP, geometry guard
│   ├── indicators.py            # ATR, RSI, BB, MACD, EMA, VWAP, FVG detection,
│   │                            #   to_higher_timeframe (look-ahead-safe resample)
│   ├── exits.py                 # trailing stop factories (ATR, chandelier, breakeven)
│   ├── metrics.py               # Sharpe, Sortino, drawdown, PF, plot writer
│   ├── validation.py            # holdout, walk-forward, MC, bootstrap, PSR, DSR, adaptive WF
│   ├── attribution.py           # slice trades by hour / day / month / side / exit_reason
│   ├── sweep.py                 # grid search with IS/OOS discipline
│   ├── optuna_search.py         # TPE Bayesian search
│   ├── run_history.py           # SQLite DB; preset_name + graph_json columns
│   ├── mtf.py                   # MTFContext: cached look-ahead-safe HTF lookups
│   ├── graph.py                 # DecisionGraph + GraphOrchestrator (THE composition layer)
│   ├── presets.py               # save/load DecisionGraph as JSON in presets/
│   └── confluence.py            # REMOVED — stub left for backwards-compat error
├── strategies/
│   ├── registry.py              # central StrategySpec registry — all 17 atoms
│   ├── _helpers.py              # risk_based_stake, atr_threshold, in_session, trailing_swing
│   ├── ensemble.py              # REMOVED — replaced by DecisionGraph
│   ├── sma_crossover.py         # smoke test
│   ├── fvg_retest.py            # 3-bar imbalance limit-order retest
│   ├── fvg_scale_out.py         # FVG with 1R scale-out + ATR trail
│   ├── bpr.py                   # Balanced Price Range (overlapping FVGs, limit orders)
│   ├── orb.py                   # Opening Range Breakout (twin stop orders)
│   ├── liquidity_sweep.py       # SMC sweep-and-reverse (limit orders)
│   ├── donchian.py              # Turtle channel breakout
│   ├── bb_reversion.py          # Bollinger band mean reversion
│   ├── vwap_reversion.py        # Day-trade VWAP fade
│   ├── rsi_reversion.py         # RSI hook from oversold/overbought
│   ├── macd_crossover.py        # MACD signal-line cross
│   ├── stoch_crossover.py       # Stochastic %K x %D in extremes
│   ├── engulfing.py             # Engulfing reversal pattern
│   ├── inside_bar.py            # Inside-bar breakout (twin stop orders)
│   ├── heikin_ashi_trend.py     # N consecutive HA bars same colour
│   ├── triple_ema.py            # 3-EMA stack
│   └── mtf_trend_fvg.py         # FVG entry, HTF EMA-trend filter
├── presets/                     # *.json files, one per saved DecisionGraph
├── reports/                     # markdown + PNG + CSV per run (gitignored)
├── scripts/                     # CLI entry points
├── tests/                       # 188 tests; pytest tests/ -v
├── ASSUMPTIONS.md               # WHAT the backtester does/doesn't model
├── README.md                    # setup
├── requirements.txt
├── run_history.db               # SQLite, auto-created
└── .env                         # IG + EODHD credentials, gitignored
```

---

## The decision-graph framework (replaces old ensembles)

`backtest/graph.py` is the single composition layer. Every backtest is
defined by a `DecisionGraph`:

- **Trigger** (exactly one, base TF): a strategy that owns the full trade
  lifecycle (entry, SL, TP, partial exits, trailing). Whatever the strategy
  does standalone.
- **Supporters** (any number, TF ≥ trigger): grade each potential entry on
  a 0–1 confidence score via their `proposed_direction()`. Weighted by
  user weight × TF-distance term. "none" (no opinion) is excluded from the
  aggregate.
- **Vetoes** (any number, TF ≥ trigger): any opposite-direction veto kills
  the trade.

The `GraphOrchestrator`:
1. Owns session timing at base-TF precision (captures the trigger's
   `flat_by` / `session_open` / `session_close` at init, neutralises the
   trigger's own checks, enforces them itself on every base bar).
2. Decouples trigger TF from data TF — strategy decides on (e.g.) 15m
   closes while engine ticks at 1m for SL/TP fill precision.
3. Wraps broker with `_GraphBrokerProxy` so pending orders also get
   scored against confluences.
4. Scales risk: `multiplier = floor + (ceiling − floor) × shape(score)`.
   Default floor 0.7, ceiling 1.0 (mild scaling — fixed costs eat tiny
   stakes if you go 0–100%).
5. Records `confluence_score`, `risk_multiplier`, full supporter
   breakdown on every trade's `entry_metadata`.

**Supporter / veto / score-threshold weights are NEVER exposed to
sweep/Optuna search.** They're instance fields on `SupporterNode`, not
`ParamSpec` entries in the registry. The optimiser tunes trigger params
only, never the scoring layer.

Presets save the whole graph as JSON in `presets/`. Each run links to a
preset (or "ad-hoc") via `run_history.runs.preset_name`.

---

## Cost model (config.py)

Per-instrument `CostModel` profiles in `PROFILES`. Each has either
`spread_points` (fixed pt for indices) or `spread_bps` (bps × price for
stocks / unknowns). `profile_for(ticker)` auto-detects.

Profiles: UK100, US500, US100, DJI, GER40, JPN225, FRA40, EURUSD, GBPUSD,
USDJPY, BTC, ETH, STOCK, ETF, DEFAULT.

`trading_window_for(profile, mode)` returns (open, close) in user tz for
RTH/ETH/None data filtering. `session_defaults_for(profile, mode)` returns
(session_open, session_close, flat_by) HH:MM strings for graph defaults.

`TRADING_TZ` (default Europe/London) is set globally; all fetchers
convert data to this tz on load.

`COSTS` is the active profile. Broker reads `config.COSTS` dynamically so
the UI can swap profiles without re-plumbing.

---

## Execution semantics (the parts that took the most iteration)

### Order of operations per base bar (engine.py)
1. Update MTF cursor (look-ahead-safe HTF lookups)
2. **`broker.check_stops()`** — existing positions' SL/TP fire FIRST
   (gap-aware fills: target/stop at `max/min(trigger, bar.Open)`)
3. Apply pending strategy signal at bar OPEN (market entries; gap-invalid
   trades silently skipped via `_dropped_geometry_count`)
4. `broker.check_pending_orders()` — limit/stop fills
5. Same-bar SL/TP check on positions just opened mid-bar (level-based,
   not gap-aware — entry happened intrabar)
6. `broker.mark()` — equity update, financing accrual
7. Orchestrator (graph) decides if this base bar is a trigger-TF close
8. Trigger.on_bar gets a TF-correct history slice; supporters/vetoes
   only queried at entry attempts

### Pending-order fill semantics (post-Apr-2026)
- **Limit BUY at T**:
  - `bar.Open <= T` → fill at `bar.Open` (favourable; broker gives you
    the better price, not the limit)
  - `bar.Open > T` AND `bar.Low <= T` → fill at T
  - else: no fill
- **Limit SELL at T**: mirror.
- **Stop BUY at T**:
  - `bar.Open >= T` → fill at `bar.Open + slippage` (gap-up, worse)
  - `bar.Open < T` AND `bar.High >= T` → fill at `T + slippage`
- **Stop SELL at T**: mirror.

### Broker geometry guard
`broker.open()` raises `ValueError` if `stop_loss` ends up on the wrong
side of `entry_price` (or `take_profit` on the loss side). Engine catches
this for MARKET fills (increments `_dropped_geometry_count`); for pending
fills the exception bubbles, increments `_dropped_order_count`.

---

## Data hours filter

Three modes selectable per run: RTH / Extended / All. Bars outside the
window get dropped on fetch using `trading_window_for(profile, mode)`.
Eliminates pre-market spike bars on US stocks polluting FVG detection
and chart inspector.

---

## Conventions

- **Tests first when refactoring.** `pytest tests/ -v` should be 188/188.
- **Every strategy implements** `on_bar(history, broker) → Signal` AND
  `proposed_direction(history) → "long"|"short"|"none"`. Standalone exec
  uses on_bar; ensembles / graph orchestrator polls proposed_direction.
- **All cost/risk constants live in `config.py`.** Never hardcode.
- **Position sizing via `risk_based_stake`** — leverage-aware, honours
  `ACCOUNT.risk_per_trade_pct` and `leverage_cap`. Never inline.
- **No look-ahead.** Engine passes `history.iloc[:i+1]`. MTF resamples
  drop the in-progress HTF bar via `include_partial=False`.
- **Accounting identity** asserted at every run end:
  `starting_balance + Σ(net_pnl) == final_balance`. Never suppress.
- **ATR-relative thresholds.** Strategy params expressed in ATR multipliers
  (e.g. `min_gap_atr_mult`), not absolute points. Lets the same params
  work across instruments at different price levels.

---

## Known gotchas

- **Trigger TF locked to data TF in the orchestrator unless explicitly
  decoupled.** When decoupled (data 1m + trigger 15m), the trigger only
  fires on 15m boundary closes; engine ticks at 1m for fill precision.
- **EODHD intraday API limits per resolution:** 1m → 119 days/request,
  5m → 599 days, 1h → 7199 days. Chunked fetcher handles this.
- **EODHD UK100 alias** maps to `ISF.LSE` (the iShares ETF, in pence)
  with a `price_scale=10` factor applied so prices match the FTSE 100
  cash index scale (~8000). Without this, the UK100 cost profile's
  1.5pt spread becomes ~17 bps on the unscaled 800-price ETF — death.
- **Volume is 0 for indices from IG** (cash index has no exchange
  volume). VWAP-based strategies behave slightly different on IG vs
  yfinance data (yfinance has synthetic volume).
- **`max_concurrent_positions=1`** by default. Two stop orders firing
  on the same bar (ORB / inside_bar): first fills, second is dropped.
  Both strategies now handle this cleanly via "both dead" branch.
- **Session times come from the GRAPH, not the strategy** (post-Apr-2026
  refactor). Strategy's `session_open / session_close / flat_by` attrs
  are neutralised by the orchestrator at init.

---

## What's NOT modelled

- Order-book depth / queue position
- Requoting / dealer rejection
- News-bar microstructure (within-bar spread spikes beyond bar Spread)
- Latency between signal and broker fill
- Correlated drawdowns across multiple live strategies
- Trader psychology

What IS modelled (post-iteration):
- Variable spread (per-bar from IG bid/ask; bps × price fallback)
- Slippage scaling with spread on stops + market orders
- Overnight financing (annualised, charged daily)
- Gap-aware SL/TP fills (worse than trigger if bar opens past)
- Same-bar SL/TP after limit fills (entry mid-bar can hit stop same bar)
- Leverage cap (FCA 20× retail)
- Min stake (IG £0.50/pt floor)
- DST + per-asset trading hours (US stocks 14:30–21:00 UK, etc.)
- Look-ahead safety across multiple timeframes

---

## Open items / what's next

- **Use the system** to evaluate strategies. Most "strategies" don't
  work; the backtester now tells you so honestly.
- **Paper-trade candidate strategies on IG demo** before risking real
  money. PSR > 0.95 + walk-forward consistency > 60% gates a candidate.
- **News-calendar avoidance** — fetch FOMC / NFP / BoE / CPI times and
  skip entries within ±30 min. Would improve cost realism further.
- **Empirical slippage calibration** — once paper trading runs, compare
  live fills to backtest predictions, update `slip_spread_multiplier`.
- **Parallel sweeps** — requires picklable strategies. Optuna serial mode
  works fine for most needs.

---

## Status

The backtester is honest about:
- Look-ahead and accounting bugs (runtime assertions on every run)
- Per-bar spread from real broker data when available
- Multiple-testing correction (DSR after sweeps)
- IS/OOS discipline enforced by workflow
- Gap risk on stops and targets
- Pending-order fill semantics matching real-broker behaviour
- Session-time precision regardless of trigger TF

Remaining gaps are mostly inherent to bar-resolution data + retail data
feeds. No more backtester polish will close them without tick-level data.
