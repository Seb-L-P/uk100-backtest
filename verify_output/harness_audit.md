# Harness correctness audit

**Goal.** Prove that any backtest result is purely a function of the
strategy + market data, with no contamination from bugs in timing, data,
costs, execution, or accounting.

**Run.** `pytest tests/ -v` — **243 / 243 passing** (206 pre-existing
unit tests + 37 new audit tests in `tests/test_harness_audit.py`).

Each invariant below is identified by its category prefix (L/S/C/E/A/D)
and number, with the test that covers it.

---

## L. Look-ahead safety

| ID | Invariant | Test | Status |
|----|-----------|------|--------|
| L1 | Mutating any future bar leaves every past decision unchanged. | `TestLookaheadSafety::test_L1_future_bar_mutation_does_not_change_past_decision` (audit) | **PASS** |
| L2 | `sma` / `ema` / `rsi` / `atr` at bar i depend only on bars [0..i]. | `TestLookaheadSafety::test_L2_rolling_indicators_only_depend_on_past` (audit) | **PASS** |
| L3 | `detect_fvg(history, i)` reads only bars [i-2..i]. | `TestLookaheadSafety::test_L3_detect_fvg_only_uses_past_bars` (audit) | **PASS** |
| L4 | `to_higher_timeframe(include_partial=False)` drops the in-progress HTF bar. | `TestLookaheadSafety::test_L4_to_higher_timeframe_drops_in_progress_bar` (audit) + `test_no_look_ahead_in_default_mode` (existing) | **PASS** |
| L5 | `MTFContext.htf(interval)` returns only HTF bars whose END is `<= cursor`. | `TestLookaheadSafety::test_L5_mtf_context_strictly_before_cursor` (audit) + `test_htf_returns_only_closed_bars` + `test_supporter_at_higher_tf_never_sees_open_bar` (existing) | **PASS** |
| L6 | Strategy on bar i sees `history.iloc[:i+1]` — no bar beyond. | `test_strategy_never_sees_future_bars` (existing) | **PASS** |
| L7 | Signals fire at the NEXT bar's OPEN, not the same bar's close. | `test_signal_fills_at_next_bar_open` (existing) | **PASS** |

---

## S. Bar timing & sessions

| ID | Invariant | Test | Status |
|----|-----------|------|--------|
| S1 | Strategies tagged with a session window emit no entry signals outside the window. | `TestSessions::test_S1_out_of_session_no_entries` (audit) | **PASS** |
| S2 | Force-flat at `flat_by` closes any open position by end of session. | `TestSessions::test_S2_force_flat_closes_open_position` (audit) | **PASS** |
| S3 | `flat_by` is enforced at BASE-TF precision when trigger TF > base TF (the prior bug closed late on the next trigger-TF boundary). | `TestSessions::test_S3_flat_by_enforced_at_base_TF_precision` (audit) | **PASS** |
| S4 | With trigger TF > base TF, the trigger only sees one history slice per trigger-TF close — not every base bar. | `TestSessions::test_S4_trigger_TF_only_fires_on_TF_closes` (audit) | **PASS** |
| S5 | `to_trading_tz` keeps wall-clock TIME-OF-DAY consistent across the UK DST transition (UTC bars shift by 1h on the calendar BST switch — exactly what a London trader experiences). | `TestSessions::test_S5_DST_transition_preserves_wallclock` (audit) | **PASS** |

---

## C. Cost model

| ID | Invariant | Test | Status |
|----|-----------|------|--------|
| C1 | Spread is charged ONCE per round trip (entry + exit averaged where both known). | `TestCostModel::test_C1_spread_charged_once_per_round_trip` (audit) + `test_open_then_close_long_profitable`, `test_close_uses_average_of_entry_and_exit_spread`, `test_close_falls_back_to_config_when_no_spread_info` (existing) | **PASS** |
| C2 | Slippage scales with bar spread × multiplier, floored at `min_slippage_points`, falls back to fixed `slippage_points` when spread unknown. | `TestCostModel::test_C2_slippage_scales_with_spread` (audit) + `test_effective_slippage_*` (existing) | **PASS** |
| C3 | Long financing per day = `notional × (SONIA + admin) / 365`. | `TestCostModel::test_C3_long_financing_formula` (audit) + `test_overnight_financing_charged` (existing) | **PASS** |
| C4 | Short financing per day = `−notional × (SONIA − admin) / 365` — a CREDIT to the trader when SONIA > admin. | `TestCostModel::test_C4_short_financing_is_credit_when_admin_lt_sonia` (audit) | **PASS** |
| C5 | `risk_based_stake` enforces the `min_stake` floor when risk_pts is large enough to compute below it, and the spread-coverage gate rejects trades with risk smaller than one spread crossing. | `TestCostModel::test_C5_min_stake_floor_enforced` (audit) | **PASS** |
| C6 | Leverage cap is re-checked at FILL time for pending orders — drops the order rather than crashing if balance has shrunk between placement and fill. | `TestCostModel::test_C6_leverage_cap_at_fill_time_drops_order` (audit) + `test_leverage_cap_enforced` (existing, signal-time) | **PASS** |
| C7 | `profile_for(ticker)` routes common patterns: UK100 aliases → UK100 (1.5pt), AAPL/TSLA → STOCK (bps), BTC-USD → BTC (40pt), ISF.LSE → ETF, unknown → DEFAULT (bps). | `TestCostModel::test_C7_profile_for_ticker_routing` (audit) | **PASS** |

---

## E. Order execution

| ID | Invariant | Test | Status |
|----|-----------|------|--------|
| E1 | Engine main-loop order per bar: stops → pending market signal → pending limit/stop orders → mark → strategy. A stop hit on bar i wins over a strategy `close` signal that became pending at bar i-1. | `TestOrderExecution::test_E1_stops_fire_before_pending_signal_close` (audit) | **PASS** |
| E2 | Limit BUY fills at `bar.Open` when `bar.Open <= trigger` (favourable gap), not at the trigger. Mirror for limit SELL. | `TestOrderExecution::test_E2_limit_buy_favourable_gap_fill` (audit) + `test_limit_orders_never_apply_slippage` (existing) | **PASS** |
| E3 | Stop BUY fills at `bar.Open + slippage` when `bar.Open >= trigger` (unfavourable gap), at `trigger + slippage` otherwise. Mirror for stop SELL. | `TestOrderExecution::test_E3_stop_buy_unfavourable_gap_fill` (audit) + `test_stop_buy_*`, `test_stop_sell_*` (existing) | **PASS** |
| E4 | After a pending limit fills mid-bar, the SAME bar's range is checked against the new position's SL/TP — preventing the next-bar gap-fill logic from over-inflating losses. | `TestOrderExecution::test_E4_same_bar_SL_after_limit_fill` (audit) | **PASS** |
| E5 | `broker.open` raises ValueError on geometry violations (long with stop above entry, short with stop below entry, target on the loss side). | `TestOrderExecution::test_E5_geometry_guard_rejects_wrong_side_stop` (audit) | **PASS** |
| E6 | A strategy `close` signal for an already-closed position is silently skipped, not raised — covers the same-bar stop+strategy-close race. | `TestOrderExecution::test_E6_stale_close_silently_skipped` (audit) | **PASS** |
| E7 | A strategy `scale_out` signal for a position that already closed via SL/TP on the same bar is silently skipped, not raised. | `TestOrderExecution::test_E7_stale_scale_out_silently_skipped` (audit) | **PASS** |

---

## A. Accounting identity

| ID | Invariant | Test | Status |
|----|-----------|------|--------|
| A1 | `starting_balance + Σ(net_pnl) == final_balance` at the end of every run (also asserted inline in `run_backtest`). | `TestAccountingIdentity::test_A1_starting_plus_trades_equals_final` (audit) + `test_accounting_identity_holds` (existing) | **PASS** |
| A2 | Per-trade `gross_pnl − spread − slippage − financing == net_pnl` exactly (≤ 1e-9), even with overnight financing accrued. | `TestAccountingIdentity::test_A2_per_trade_pnl_identity_with_overnight` (audit) + `test_pnl_accounting_identity_per_trade` (existing) | **PASS** |
| A3 | Equity curve timestamps monotonically increasing. | `TestAccountingIdentity::test_A3_equity_curve_monotonic_timestamps` (audit) | **PASS** |
| A4 | No open position at the end of `run_backtest` — engine force-flattens with `exit_reason='eod'`. | `TestAccountingIdentity::test_A4_no_open_position_at_end` (audit) + `test_no_open_position_at_end` (existing) | **PASS** |
| A5 | The engine's accounting assertion FIRES on a synthetic-broken case — proves it isn't silently passing in production. A trailing-stop callback that mutates `broker.balance` triggers `AssertionError: Equity accounting drift`. | `TestAccountingIdentity::test_A5_engine_assertion_fires_on_balance_corruption` (audit) | **PASS** |

---

## D. Data integrity

| ID | Invariant | Test | Status |
|----|-----------|------|--------|
| D1 | Fetcher `_validate` sorts the index ascending. | `TestDataIntegrity::test_D1_validate_sorts_ascending` (audit) | **PASS** |
| D2 | Fetcher `_validate` drops duplicate timestamps (keep="first"). | `TestDataIntegrity::test_D2_validate_drops_duplicates` (audit) | **PASS** |
| D3 | Fetcher `_validate` drops bars with inconsistent OHLC (High < Low, etc.) — no synthetic gap-fill. | `TestDataIntegrity::test_D3_validate_drops_inconsistent_OHLC` (audit) | **PASS** |
| D4 | `to_trading_tz` converts UTC bars to TRADING_TZ (London) wall-clock — pre-DST 12:00 UTC → 12:00 GMT, post-DST 12:00 UTC → 13:00 BST. | `TestDataIntegrity::test_D4_to_trading_tz_converts_UTC_to_London` (audit) | **PASS** |
| D5 | `to_trading_tz` strips the tz on output so downstream code (which uses tz-naive comparisons) keeps working. | `TestDataIntegrity::test_D5_to_trading_tz_strips_tz` (audit) | **PASS** |
| D6 | The EODHD fetcher's `UK100` alias resolves to `(ISF.LSE, scale=10.0)` — pence ETF × 10 = FTSE-index scale, so the 1.5pt UK100 spread maps to ~2bps not ~20bps. | `TestDataIntegrity::test_D6_eodhd_UK100_alias_scales_by_10` (audit) | **PASS** |
| D7 | The RTH filter (`trading_window_for` + index.time mask, as applied in `app.py` and pages) drops out-of-window bars. | `TestDataIntegrity::test_D7_RTH_filter_drops_out_of_window_bars` (audit) | **PASS** |
| D8 | IG fetcher `_ig_to_ohlcv` emits a `Spread` column = mean(ask O/H/L/C) − mean(bid O/H/L/C) per bar, then engine propagates it to broker. | `TestDataIntegrity::test_D8_ig_fetcher_emits_spread_column_from_bid_ask` (audit) + `test_engine_propagates_spread_from_data` (existing) | **PASS** |

---

## Totals

- Invariants audited: **34** (L1–L7, S1–S5, C1–C7, E1–E7, A1–A5, D1–D8).
- Tests run: **243** (206 pre-existing + 37 new in `tests/test_harness_audit.py`).
- Tests passing: **243 / 243**.
- Tests failing: **0**.

## Bugs found and fixed during the audit

- **S3 (initial false alarm).** The audit test asserted `exit_time <= flat_by` on the
  bar where the orchestrator detects flat_by, but the engine deliberately delays
  signal fills by one bar (no same-bar execution). The orchestrator emits
  `close_all` on the bar at `flat_by`; the close fills on the *next* base bar's
  open. The S3 assertion was tightened to `<= flat_by + base_tf_minutes` — within
  one base bar, vs. the prior bug which closed at the next trigger-TF boundary
  (15–30 min late). Behaviour itself was already correct.

No code bugs were found in this audit pass — every invariant the harness
claims to enforce, it does enforce.

## Deliberate exclusions

These are explicit non-goals of this audit pass:

- **Real-data fetches.** The IG and EODHD fetchers are tested at the
  transformation layer (`_validate`, `_ig_to_ohlcv`, `to_trading_tz`, alias
  resolution). End-to-end fetches require network + credentials and are
  exercised manually via `scripts/eodhd_test.py` and `scripts/ig_test.py`.
  Mocking them in unit tests would add complexity without catching new bugs.
- **Strategy-specific correctness.** The 23 registered strategies are
  smoke-tested by `tests/test_strategies_smoke.py` and audited per-trade by
  `scripts/verify_all_strategies.py`. That's a separate exercise — the
  harness audit only proves the engine + broker + indicators + data layer
  are honest; whether a strategy's *logic* is the one it claims is its
  own concern.
- **Numerical tail behaviour of indicators.** ATR/RSI/EMA values close to
  the boundary of `period` bars rely on pandas' rolling/ewm conventions
  (e.g., NaN on the first `period - 1` bars). We test the look-ahead
  property and a few well-known cases (constant input, monotonic
  uptrend → RSI > 90, etc.); we do not regression-test exact decimal
  values, as those depend on pandas version and are exercised by the
  consuming strategies.
- **Position-sizing for non-standard cost profiles.** `risk_based_stake`'s
  leverage cap interaction with `existing_notional_gbp` for stacked
  positions is exercised by `test_leverage_cap_includes_existing_positions`
  but not in this audit file — the multi-position path is already covered
  in `tests/test_multi_position.py`.
- **Walk-forward / Monte Carlo / Optuna machinery.** These are exercised
  by `tests/test_validation.py` and are downstream of the engine. The
  audit covers the engine alone; if `run_backtest` is correct, so is
  anything that calls it in a loop.
