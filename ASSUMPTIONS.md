# Backtester assumptions and limitations

What the backtester models honestly, what it doesn't model, and where to
mistrust it. Read this once. Re-read it whenever a backtest result looks
too good — it'll usually point at where the gap between simulation and
reality lives.

## What we model honestly

- **Spread**: fixed at 1.5pt per round trip (configurable in `config.py`).
- **Slippage**: applied on stop fills and market orders only, 0.5pt per side.
- **Overnight financing**: daily charge using `(SONIA + 2.5%) / 365 × notional`
  for longs, symmetric for shorts. Updated by editing `config.py`.
- **Leverage cap**: 20:1 on indices (matches FCA retail limit).
- **Position sizing**: risk-based, configurable per-trade risk percentage.
- **No look-ahead**: strategies see history up to and including the current
  bar only. Signals fill at the NEXT bar's open price, never the same-bar
  close. The engine enforces this in `run_backtest()`.
- **Stop-and-target priority**: when a single bar's range contains both
  your stop and your target, we conservatively assume the stop hit first.

## What we DON'T model — these could matter

### Cost realism

- **Variable spread**. Real IG spread varies — ~1pt during liquid UK hours,
  wider overnight, much wider on news. We use 1.5pt flat. For 15-minute
  intraday strategies that trade in the first/last 30 minutes, real costs
  may be up to 2x ours.
- **No requoting / dealer rejection**. Real CFD and spread-bet platforms
  may reject your fill request and requote at a worse price. We assume you
  always get filled at the stated price.
- **No slippage on signal exits**. Stops and market orders incur our 0.5pt
  slippage; signal-based exits (e.g. "close at this bar's close") incur
  nothing. Real signal exits may slip in fast markets.
- **No size-based slippage**. We assume your stake size has no effect on
  the price you get. Realistic for retail-sized trades on FTSE 100; would
  matter for larger sizes or thinner markets.

### Execution model

- **Gap-aware SL/TP fills**. When a bar's OPEN has already crossed the
  stop or target level, we fill at `bar.Open` (worse than the level for
  stops, better than the level for targets) plus slippage where
  applicable. This is closer to real broker behaviour than the older
  "always fill at the level" model. Slippage on stops still scales with
  bar spread; the floor is `min_slippage_points`.
- **Same-bar SL/TP after limit fills**. When a pending limit/stop opens a
  position mid-bar, the bar's continuing range is checked against the
  new position's SL/TP immediately. Without this, the next bar's gap-fill
  logic would fire and inflate losses.
- **Limit fills use favourable price**. If a long limit at 100 sees the
  bar open at 98, we fill at 98 (a real broker gives you the better
  price). Stop orders mirror this with slippage on the unfavourable side.
- **Bar-by-bar resolution**. Intrabar dynamics are invisible. A 15-min
  bar that swung high→low→high→low looks like just OHLC to us. We
  conservatively assume STOP fires first when both SL and TP fall inside
  one bar's range — worst case for the trader. This is why we recommend
  trigger-TF decoupling (data at 1m, trigger at 15m) for strategies
  where intrabar ordering matters: fills resolve at 1m precision while
  strategy decisions stay on 15m closes.

### Data quality

- **Yahoo Finance is free data**. Mid-price or last-trade history, not bid/
  ask. We can't directly measure historical spreads. Occasional missing bars
  (we drop them silently). 60-day limit on 15m intraday history — a binding
  constraint on day-trade backtests.
- **Index volume is exchange volume**. It's volume for the FTSE 100
  *underlying components*, not your broker's CFD/spread-bet flow. VWAP
  computed from it is a reasonable proxy, not the institutional VWAP your
  algo competitors are tracking.
- **No corporate actions**. Doesn't matter for index (FTSE 100 rebalances
  automatically); would matter for single stocks.

### Walk-forward and Monte Carlo

- **Walk-forward uses the SAME parameters across folds**. Real walk-forward
  optimisation re-optimises parameters in each in-sample window. We test
  parameter STABILITY across regimes; we don't simulate adaptive optimisation.
  Adding adaptive optimisation is a meaningful future improvement.
- **Monte Carlo shuffles trade order, not trade outcomes**. It tells you
  about path-dependence (max drawdown sensitivity to sequencing). It does
  NOT tell you whether the trades themselves were robust — for that use
  walk-forward and out-of-sample.

### Account model

- **Single position only**. One trade at a time across the entire account.
  Ensembles work around this by combining signals at decision time and
  taking one consolidated position; they don't run multiple positions.
- **No real-broker quirks**. No margin top-up logic, no per-position
  stops vs account-level stops, no overnight position caps, no per-instrument
  exposure limits. IG's real account has these constraints; ours doesn't.

## Cost-model parameters to keep current

These live in `config.py` and degrade in accuracy as time passes:

- `sonia_annual` (currently `0.0525`). Update if the Bank of England base
  rate changes materially. SONIA is roughly base rate minus a few basis
  points.
- `admin_annual` (currently `0.025`). IG's published admin component on
  overnight financing. Verify from IG's charges page once a year.
- `spread_points` (currently `1.5`). Our flat assumption. If you ever get
  IG's API or proper bid/ask data, replace with the empirical average for
  the hours you actually trade.

## Self-checks the engine runs automatically

After every backtest, the engine asserts these and crashes if any fail:

1. `final_balance - starting_balance == sum(trade net P&Ls)` to within 1p
2. `net_pnl == gross_pnl - spread - slippage - financing` exactly, per trade
3. No open position at end of backtest
4. Equity timestamps monotonically increasing

If any of these ever fire in normal use, paste the assertion error into a
session — that's a real bug, not user error.

## Things you can do to stress-test before trusting a result

- **Inspect 5 trades by hand.** Pick wins and losses, verify entry/exit
  prices match what the bars showed, verify costs match the cost-model
  formulae in `config.py`. The UI trade inspector helps here.
- **Run with `spread_points=3.0` and `slippage_points=2.0`** in `config.py`
  to simulate "really bad day" cost conditions. If your edge survives
  this, that's encouraging. If it disappears, the edge was the cost gap.
- **Bump `sonia_annual` to 0.07** and re-run swing strategies. Tests
  sensitivity to rate-regime changes.
- **Run the strategy on a totally different period** (e.g. 2018-2019 daily)
  to check it's not regime-dependent.
- **Run walk-forward with 10 folds instead of 4.** If consistency collapses,
  the strategy is over-fit to specific time windows.
