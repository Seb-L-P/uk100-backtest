# Backtester — Project Context

You are continuing a multi-month project building an honest backtester. The
user is a UK retail trader; the eventual product they'll trade is IG spread
bet — but the system is now **multi-asset**: UK 100, US stocks (TSLA, AAPL,
MSFT), crypto (BTC), forex, all working via cost profiles.

**Read this whole file before doing anything else.** It captures the
design decisions, conventions, gotchas, and current state. Most "why is
it like this?" questions have answers in here.

---

## Project goal

Build a backtester that is honest enough to detect when a strategy has
real edge vs. when it just looks good due to data-mining, optimistic
cost modelling, look-ahead bias, or any of the dozen subtle execution
bugs that plague typical retail backtesters.

There are now THREE Streamlit pages working in concert:
- **`app.py`** — manual backtester: pick a single graph, run it, inspect.
- **`pages/Strategy_verification.py`** — eyeball a strategy on a small
  sample (specific asset / date range) and verify trades look sensible.
- **`pages/Strategy_discovery.py`** — random sweep over the full design
  space (trigger + supporters + vetoes + parameters + TFs + weights +
  optional graph knobs) with a 3-way IS/Val/OOS split.
- **`pages/Strategy_validation.py`** — pressure-test a saved candidate
  with walk-forward, multi-asset cross-check, Monte Carlo, and Optuna
  refinement.

---

## Quick start

```bash
cd ~/Developer/UK-100-Backtest/uk100-backtest
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py            # http://localhost:8501
```

`.env` (gitignored) holds IG and EODHD credentials. Sample test scripts:
`scripts/ig_test.py`, `scripts/eodhd_test.py`.

---

## Architecture map

```
uk100-backtest/
├── config.py                    # cost PROFILES, TRADING_TZ, ACCOUNT,
│                                #   trading_window_for(), SAVED_SWEEPS_DIR
├── app.py                       # Streamlit UI shell (manual backtester)
├── app_graph_builder.py         # Sidebar graph-builder widget (used by app.py)
├── pages/
│   ├── Strategy_verification.py # eyeball trades on a single strategy / asset
│   ├── Strategy_discovery.py    # random sweep, 3-way split, card-style results
│   └── Strategy_validation.py   # walk-forward / multi-asset / MC / refinement
├── data/
│   ├── fetcher.py               # source dispatcher (IG / EODHD / yfinance)
│   ├── ig_fetcher.py            # IG REST API, parquet cache, real Spread column
│   ├── eodhd_fetcher.py         # EODHD API, chunked intraday fetcher
│   │                            #   (v4 cache — uses OUTPUT TF for window calc)
│   └── _tz.py                   # to_trading_tz: convert fetched data → user tz
├── backtest/
│   ├── engine.py                # event-driven bar loop, gap-aware order ops
│   ├── broker.py                # multi-position, partial exits, pending orders,
│   │                            #   gap-aware fills, same-bar SL/TP, geometry guard
│   ├── indicators.py            # ATR, RSI, BB, MACD, EMA, VWAP, FVG, ADX,
│   │                            #   Keltner, MFI, PSAR, pivots,
│   │                            #   to_higher_timeframe (look-ahead-safe resample)
│   ├── exits.py                 # trailing stop factories (ATR, chandelier, BE)
│   ├── metrics.py               # Sharpe, Sortino, drawdown, PF, plot writer
│   ├── validation.py            # holdout, walk-forward, MC, bootstrap, PSR, DSR
│   ├── attribution.py           # slice trades by hour / day / month / side / exit
│   ├── sweep.py                 # LEGACY grid search (single strategy only)
│   ├── optuna_search.py         # LEGACY TPE Bayesian search (single strategy)
│   ├── run_history.py           # SQLite DB; preset_name + graph_json columns
│   ├── mtf.py                   # MTFContext: cached look-ahead-safe HTF lookups
│   ├── graph.py                 # DecisionGraph + GraphOrchestrator
│   ├── presets.py               # save/load DecisionGraph as JSON in presets/
│   │
│   │   --- Discovery sweep (Phase 3) ---
│   ├── data_split.py            # 3-way IS/Val/OOS time split
│   ├── sweep_space.py           # SearchSpace + sample_random_graph()
│   ├── sweep_objective.py       # Sharpe with adaptive min-trades floor
│   ├── sweep_runner.py          # 3-stage sweep orchestrator
│   │                            #   (parallel via ProcessPoolExecutor, n_jobs)
│   │
│   │   --- Validation / refinement (Phase 4) ---
│   ├── sweep_persistence.py     # save/load top candidates as JSON
│   ├── multi_seed.py            # aggregate sweeps across multiple seeds
│   ├── candidate_validation.py  # walk-forward / multi-asset / MC wrappers
│   │                            #   that take a DecisionGraph
│   └── refinement.py            # Optuna focused search around a candidate
├── strategies/
│   ├── registry.py              # central StrategySpec registry — 23 strategies
│   ├── _helpers.py              # risk_based_stake, atr_threshold, in_session
│   │
│   │   --- Original 17 ---
│   ├── sma_crossover.py         # smoke test
│   ├── fvg_retest.py            # 3-bar imbalance limit-order retest
│   ├── fvg_scale_out.py         # FVG with 1R scale-out + ATR trail
│   ├── bpr.py                   # Balanced Price Range (overlapping FVGs)
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
│   ├── mtf_trend_fvg.py         # FVG entry, HTF EMA-trend filter
│   │
│   │   --- Phase 2 additions ---
│   ├── pivot_reversal.py        # Day-trade fade at S1/R1 classic pivots
│   ├── adx_trend.py             # SMA crossover gated by ADX > threshold
│   ├── psar_flip.py             # Parabolic SAR stop-and-reverse
│   ├── keltner_breakout.py      # Close outside Keltner upper/lower
│   ├── overnight_range.py       # Stop orders at prior-session high/low
│   └── mfi_extremes.py          # Volume-weighted RSI extreme reversal
├── presets/                     # *.json files, one per saved DecisionGraph
├── saved_sweeps/                # JSON dumps of top candidates from sweeps
├── reports/                     # markdown + PNG + CSV per run (gitignored)
├── scripts/                     # CLI entry points
├── tests/                       # pytest tests/ -v
├── ASSUMPTIONS.md               # WHAT the backtester does/doesn't model
├── README.md                    # setup
├── requirements.txt
├── run_history.db               # SQLite, auto-created
└── .env                         # IG + EODHD credentials, gitignored
```

---

## The decision-graph framework

`backtest/graph.py` is the single composition layer. Every backtest is
defined by a `DecisionGraph`:

- **Trigger** (exactly one): a strategy that owns the full trade lifecycle
  (entry, SL, TP, partial exits, trailing). Whatever the strategy does
  standalone. Has its OWN `timeframe` field (can be ≥ data TF).
- **Supporters** (any number, TF ≥ trigger): grade each potential entry on
  a 0–1 confidence score via their `proposed_direction()`. Weighted by
  user weight × TF-distance term. "none" (no opinion) is excluded from
  the aggregate.
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

**Status update on weight tuning:** The original "weights NEVER exposed to
sweep" rule has been relaxed — `SearchSpace.sweep_weights=True` enables
per-supporter weight sampling in the discovery sweep (range default
0.3–2.0). The manual UI in `app_graph_builder.py` still lets the user
override weights by hand; the sweep can now explore that dimension too.
Graph-level knobs (`min_score`, `risk_floor`, `risk_ceiling`,
`risk_curve`) are tunable via `SearchSpace.sweep_graph_knobs=True`,
still off by default to keep search space manageable.

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
3. Apply pending strategy signal at bar OPEN. Wrapped in try/except for
   `(ValueError, RuntimeError)` — covers two cases:
     - **Leverage cap exceeded at fill time** (drawdown between signal &
       fill made the sized stake un-fitable). Increments
       `broker._dropped_order_count`.
     - **Stop/target geometry invalidated by gap.** Engine catches via
       `_gap_invalidates` pre-check; increments `_dropped_geometry_count`.
   For `close` / `close_position` / `scale_out` signals, we also check
   `_position_open(broker, position_id)` and silently skip if the
   position has already been closed by stops/targets on the same bar.
4. `broker.check_pending_orders()` — limit/stop fills (also try/except'd)
5. Same-bar SL/TP check on positions just opened mid-bar (level-based,
   not gap-aware — entry happened intrabar)
6. `broker.mark()` — equity update, financing accrual
7. Orchestrator (graph) decides if this base bar is a trigger-TF close
8. Trigger.on_bar gets a TF-correct history slice; supporters/vetoes
   only queried at entry attempts

### Pending-order fill semantics
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

## Discovery sweep architecture (Phase 3 + 3.5)

The discovery page builds a `SearchSpace` and feeds it to `run_sweep()`
which executes the 3-stage IS→Val→OOS flow:

- **`backtest/data_split.py`** — `three_way_split(data, is_ratio,
  val_ratio)` returns a `DataSplit` named-tuple with non-overlapping IS /
  Val / OOS DataFrames. Validates input sizing.
- **`backtest/sweep_space.py`** — `SearchSpace` dataclass declares what's
  sweepable: triggers pool, supporters pool, vetoes pool, max counts,
  data TF, trigger TF options (the sample pool), supporter TF cap,
  optional weight sweep, optional graph-knob sweep. `sample_random_graph`
  samples one full `DecisionGraph` per call.
- **`backtest/sweep_objective.py`** — `compute_metrics(...)` returns a
  `TrialMetrics` with Sharpe + supporting numbers, disqualifying graphs
  below the min-trades floor (`-inf` Sharpe so they sink in rank).
- **`backtest/sweep_runner.py`** — `run_sweep(data, space, ...)` runs N
  IS trials, takes top-K to Val, top-M to OOS. **Per-split adaptive
  min-trades floor**: each split's effective floor is `min(user_cap,
  max(3, n_bars // 80))`, so smaller Val/OOS windows don't
  over-disqualify. Returns a `SweepResult` with all three tiers + the
  adaptive floors actually used.
  **Parallel execution** (`n_jobs` param): each stage is embarrassingly
  parallel — a trial is a pure function of (graph, split, costs). All N
  graphs are sampled SERIALLY up front (graph sampling consumes the RNG in
  strict order — that's the only way identical seeds give identical graphs),
  then the backtests fan out over a `ProcessPoolExecutor`. `n_jobs=None`
  (default) → `os.cpu_count()-1` workers; `n_jobs=1` → in-process serial
  (reference path, no pool overhead); `n_jobs>1` → that many workers. The
  read-only split DataFrames + cost model are shipped to each worker ONCE
  via the pool `initializer` (per-process global), not re-pickled per task.
  Results are reassembled in trial order before sorting, so the final
  leaderboard is **bit-identical across `n_jobs` for the same seed**.
  `progress_callback` fires as trials COMPLETE (`as_completed`), not as
  they're submitted. Per-trial crashes are still caught inside the worker
  (`_run_one_trial` never raises). `run_multi_seed` forwards `n_jobs`, so
  multi-seed runs parallelise within each seed (seeds run sequentially).

### Backward-compat alias

`SearchSpace` accepts both `data_tf` (new) and `base_tf` (legacy) — the
old API still works but maps internally to the new field.

### Strategies excluded from default supporter/veto pools

The sampler's `_NOISY_HTF_SUPPORTERS` set excludes
`{orb, overnight_range, pivot_reversal, vwap_revert}` from
supporters/vetoes by default — their `proposed_direction()` is
session-bound and doesn't survive HTF resampling. They're still valid
as triggers.

---

## Validation / refinement architecture (Phase 4)

- **`backtest/sweep_persistence.py`** — `save_sweep(...)` writes the
  top-N qualified + disqualified candidates to JSON in `SAVED_SWEEPS_DIR`
  (filename `{ts}_{asset}_{tf}_{n_trials}t.json`). `load_sweep(path)`
  reads and rehydrates DecisionGraph objects at `candidate["graph_obj"]`.
  Schema-versioned (`SCHEMA_VERSION = 1`).
- **`backtest/multi_seed.py`** — `run_multi_seed(data, space, seeds=[...])`
  runs the SAME SearchSpace under each seed and aggregates by structural
  signature `(trigger_strategy, trigger_tf)`. Returns a list of
  `StructureSummary` objects sorted by (seeds appeared, best Sharpe).
  Strong robustness signal: same trigger in top-3 across 3+ seeds.
- **`backtest/candidate_validation.py`** — three wrappers taking a
  `DecisionGraph`:
    - `walk_forward_candidate(data, graph, n_folds)` — K consecutive
      folds, fresh orchestrator each. Reports per-fold Sharpe.
    - `multi_asset_check(graph, asset_specs, data_loader)` — same graph
      across N other assets. Caller injects `data_loader` to keep the
      module decoupled from the fetcher.
    - `monte_carlo_candidate(data, graph, n_simulations)` — runs the
      candidate then shuffles trade order N times (delegates to existing
      `backtest.validation.monte_carlo_trade_shuffle`).
- **`backtest/refinement.py`** — `refine_candidate(data, graph,
  n_trials)`. Optuna TPE focused search anchored on the candidate's
  current parameters, narrowed to `narrow_factor × original_range` per
  ParamSpec (default 0.25). Tunes params + optional weights. Holds
  TFs, structure, and graph knobs fixed. Returns original-vs-refined
  cross-split Sharpe.

---

## Data hours filter

Three modes selectable per run: RTH / Extended / All. Bars outside the
window get dropped on fetch using `trading_window_for(profile, mode)`.
Eliminates pre-market spike bars on US stocks polluting FVG detection
and chart inspector.

---

## Conventions

- **Tests first when refactoring.** `pytest tests/ -v`.
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

- **EODHD intraday API limits per resolution:** 1m → 119 days/request,
  5m → 599 days, 1h → 7199 days. Chunked fetcher handles this.
- **EODHD intraday `num_points` semantics (post-v4 fix):** the wall-clock
  range is computed from the OUTPUT interval's minutes, not the native
  fetch interval. Cache prefix bumped to `eodhd_v4_` to invalidate old
  short-fetched caches. If you find a `eodhd_v3_*.parquet` in the cache,
  it was fetched under the old bug and is shorter than it claims.
- **EODHD UK100 alias** maps to `ISF.LSE` (the iShares ETF, in pence)
  with a `price_scale=10` factor applied so prices match the FTSE 100
  cash index scale (~8000). Without this, the UK100 cost profile's
  1.5pt spread becomes ~17 bps on the unscaled 800-price ETF — death.
- **Volume is 0 for indices from IG** (cash index has no exchange
  volume). VWAP-based and MFI-based strategies behave slightly different
  on IG vs yfinance/EODHD data (the latter have synthetic / actual volume).
- **`max_concurrent_positions=1`** by default. Two stop orders firing
  on the same bar (ORB / inside_bar / overnight_range): first fills,
  second is dropped. All three strategies handle this via "both dead"
  branch.
- **Session times come from the GRAPH, not the strategy** (post-Apr-2026
  refactor). Strategy's `session_open / session_close / flat_by` attrs
  are neutralised by the orchestrator at init.
- **`MTFContext._period` uses `pd.Timedelta(rule_string)`** directly,
  NOT `pd.Timedelta(pd.tseries.frequencies.to_offset(rule))` — newer
  pandas (≥2.2) refuses to convert a `Day` offset directly to a
  Timedelta. This was a real crash on any sample with a `1d` supporter.

---

## What's NOT modelled

- Order-book depth / queue position
- Requoting / dealer rejection
- News-bar microstructure (within-bar spread spikes beyond bar Spread)
- Latency between signal and broker fill
- Correlated drawdowns across multiple live strategies
- Trader psychology

What IS modelled:
- Variable spread (per-bar from IG bid/ask; bps × price fallback)
- Slippage scaling with spread on stops + market orders
- Overnight financing (annualised, charged daily)
- Gap-aware SL/TP fills (worse than trigger if bar opens past)
- Same-bar SL/TP after limit fills (entry mid-bar can hit stop same bar)
- Leverage cap (FCA 20× retail) — re-checked at FILL time, gracefully
  drops the order if the account drew down between signal and fill
- Min stake (IG £0.50/pt floor)
- DST + per-asset trading hours (US stocks 14:30–21:00 UK, etc.)
- Look-ahead safety across multiple timeframes

---

## Open items / what's next

- **Verify all 23 strategies behave as their docstrings claim.** Some
  edge cases (volume-zero bars on indices, session-boundary behaviour,
  HTF supporter semantics on resampled bars) deserve a per-strategy
  shakedown. The validation page makes this easier.
- **Use the discovery sweep + validation pipeline** to find real
  candidates. The 3-way IS/Val/OOS split + multi-asset cross-check +
  walk-forward consistency together gate paper-trade candidates.
- **Paper-trade survivors on IG demo** before risking real money.
  PSR > 0.95, walk-forward positive-fold-fraction > 60%, and multi-asset
  cross-check pass-fraction > 50% is the rough quality bar.
- **News-calendar avoidance** — fetch FOMC / NFP / BoE / CPI times and
  skip entries within ±30 min. Would improve cost realism further.
- **Empirical slippage calibration** — once paper trading runs, compare
  live fills to backtest predictions, update `slip_spread_multiplier`.
- **Parallel sweeps** — ✅ DONE. `run_sweep(..., n_jobs=...)` fans trials
  out over a `ProcessPoolExecutor` (default `os.cpu_count()-1` workers).
  The picklable unit is the `DecisionGraph` (a plain dataclass), not the
  orchestrator — workers build their own `GraphOrchestrator` from it.
  Determinism preserved by sampling all graphs serially up front. See the
  sweep_runner bullet above.

---

## Status

The backtester is honest about:
- Look-ahead and accounting bugs (runtime assertions on every run)
- Per-bar spread from real broker data when available
- Multiple-testing correction (DSR after sweeps)
- IS/Val/OOS discipline enforced by the discovery sweep
- Gap risk on stops and targets
- Pending-order fill semantics matching real-broker behaviour
- Session-time precision regardless of trigger TF
- Leverage-cap rejection at FILL time (not just signal time)
- Stale signal handling (close/scale_out on already-closed positions)
- Multi-seed robustness (run discovery sweep under N seeds, aggregate)
- Parallel sweeps (ProcessPoolExecutor, deterministic across `n_jobs`)

Remaining gaps are mostly inherent to bar-resolution data + retail data
feeds. No more backtester polish will close them without tick-level data.
