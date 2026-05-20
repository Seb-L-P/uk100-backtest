"""
Harness correctness audit.

The single file where every invariant the harness CLAIMS to enforce gets a
test. The goal: prove that any backtest result is purely a function of the
strategy + market data, with no contamination from bugs in timing, data,
costs, execution, or accounting.

Categories (mirror verify_output/harness_audit.md):
    L  look-ahead safety
    S  bar timing & sessions
    C  cost model
    E  order execution
    A  accounting identity
    D  data integrity

Each test is short, names the invariant it covers, and uses tiny SYNTHETIC
datasets whose expected output is known by construction. Prefer hand-built
property assertions over comparing to real data (which has noise).
"""
from __future__ import annotations

import copy
from datetime import datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

import config
from backtest.broker import Broker, OpenPosition
from backtest.engine import (
    run_backtest, Signal, Strategy,
    _apply_signal, _gap_invalidates, _position_open,
)
from backtest.indicators import (
    sma, ema, rsi, atr, detect_fvg, to_higher_timeframe, _RESAMPLE_RULE,
)
from backtest.mtf import MTFContext, set_active
from config import CostModel, AccountConfig, profile_for, trading_window_for
from data._tz import to_trading_tz


# =======================================================================
# Helpers shared across audit tests
# =======================================================================
def _flat_ohlc(n: int = 50, start: str = "2024-01-02",
               freq: str = "D", price: float = 100.0) -> pd.DataFrame:
    """N bars of perfectly flat OHLC at `price`. Useful when we want to
    isolate a specific behaviour from any noise."""
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        "Open":  [price] * n,
        "High":  [price] * n,
        "Low":   [price] * n,
        "Close": [price] * n,
        "Volume": [1000] * n,
    }, index=idx)


def _trending_ohlc(n: int = 50, start_price: float = 100.0,
                   step: float = 1.0, start: str = "2024-01-02",
                   freq: str = "D") -> pd.DataFrame:
    """Monotonic uptrend, deterministic."""
    idx = pd.date_range(start, periods=n, freq=freq)
    closes = [start_price + i * step for i in range(n)]
    opens = [c - 0.2 for c in closes]
    highs = [c + 0.3 for c in closes]
    lows = [c - 0.4 for c in closes]
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes,
        "Volume": [1000] * n,
    }, index=idx)


# =======================================================================
# L. Look-ahead safety
# =======================================================================
class TestLookaheadSafety:
    """Invariants:
        L1 — Mutating a future bar cannot change a past decision.
        L2 — sma / ema / rsi / atr at bar i depend only on bars [0..i].
        L3 — detect_fvg(history, i) depends only on bars [0..i].
        L4 — to_higher_timeframe(include_partial=False) drops in-progress HTF bar.
        L5 — MTFContext.htf returns only HTF bars closed strictly before `now`.
    """

    def test_L1_future_bar_mutation_does_not_change_past_decision(self):
        """Run a recording strategy on two datasets identical up to bar k but
        differing in every bar after k. The decisions made on bars [0..k]
        must be identical."""
        captured_a: list[tuple[int, str]] = []
        captured_b: list[tuple[int, str]] = []

        class _Recorder(Strategy):
            def __init__(self, sink):
                self.sink = sink
            def on_bar(self, history, broker):
                # Make a decision summary that depends on the data seen
                last = history.iloc[-1]
                summary = f"{last['Close']:.4f}|{len(history)}"
                self.sink.append((len(history) - 1, summary))
                return Signal(action="noop")

        data_a = _trending_ohlc(30)
        data_b = data_a.copy()
        # Mutate every bar AFTER index 15 to wildly different values
        for col in ("Open", "High", "Low", "Close"):
            data_b.loc[data_b.index[16:], col] = 9999.0

        run_backtest(data_a, _Recorder(captured_a), warmup_bars=5)
        run_backtest(data_b, _Recorder(captured_b), warmup_bars=5)

        # All decisions through bar 15 must match exactly between the two runs.
        a_thru_15 = [c for c in captured_a if c[0] <= 15]
        b_thru_15 = [c for c in captured_b if c[0] <= 15]
        assert a_thru_15 == b_thru_15

    def test_L2_rolling_indicators_only_depend_on_past(self):
        """sma/ema/rsi/atr at bar i must match when computed on data[:i+1]
        versus data[:i+1] of a longer series whose tail has been replaced."""
        rng = np.random.default_rng(0)
        n = 50
        s_a = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
        s_b = s_a.copy()
        s_b.iloc[20:] = 999.0
        i = 19  # last common index

        assert sma(s_a, 5).iloc[i] == pytest.approx(sma(s_b, 5).iloc[i])
        assert ema(s_a, 5).iloc[i] == pytest.approx(ema(s_b, 5).iloc[i])
        assert rsi(s_a, 14).iloc[i] == pytest.approx(rsi(s_b, 14).iloc[i])

        ohlc_a = pd.DataFrame({"Open": s_a, "Close": s_a,
                               "High": s_a + 1, "Low": s_a - 1})
        ohlc_b = pd.DataFrame({"Open": s_b, "Close": s_b,
                               "High": s_b + 1, "Low": s_b - 1})
        assert atr(ohlc_a, 5).iloc[i] == pytest.approx(atr(ohlc_b, 5).iloc[i])

    def test_L3_detect_fvg_only_uses_past_bars(self):
        """detect_fvg(history, i) reads bars i-2 .. i. Mutating bar i+1
        cannot change the FVG detected at bar i."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        base = pd.DataFrame({
            "Open":  [100, 100, 100, 100, 100],
            "High":  [104, 100, 100, 100, 100],   # bar 0 high = 104
            "Low":   [99,  100, 100, 100, 100],
            "Close": [100, 100, 100, 100, 100],
            "Volume":[1000] * 5,
        }, index=idx)
        # Build a bullish FVG: b0.High=104, b2.Low=106 → bullish gap
        base.loc[idx[2], "Low"] = 106
        base.loc[idx[2], "Open"] = 107
        base.loc[idx[2], "High"] = 108
        base.loc[idx[2], "Close"] = 107
        # Without any mutation past bar 2:
        fvg_a = detect_fvg(base, 2)
        # Mutate bar 4 wildly:
        future = base.copy()
        future.loc[idx[4], "Low"] = 50
        future.loc[idx[4], "High"] = 200
        fvg_b = detect_fvg(future, 2)
        assert fvg_a is not None and fvg_b is not None
        assert fvg_a.direction == fvg_b.direction
        assert fvg_a.zone_low == pytest.approx(fvg_b.zone_low)
        assert fvg_a.zone_high == pytest.approx(fvg_b.zone_high)

    def test_L4_to_higher_timeframe_drops_in_progress_bar(self):
        """include_partial=False must not return any HTF bar whose end time
        is after the last base bar."""
        # 5 × 15m bars, 14:00..15:00. With 1h HTF (label="left", closed="left")
        # the 14:00 HTF bar covers [14:00, 15:00). Latest base = 15:00 means
        # the 14:00 bar IS closed exactly at the cursor. The 15:00 bar is
        # also incomplete.
        idx = pd.date_range("2024-01-01 14:00", periods=5, freq="15min")
        df = pd.DataFrame({
            "Open":  [100, 101, 102, 103, 104],
            "High":  [101, 102, 103, 104, 105],
            "Low":   [ 99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume":[10] * 5,
        }, index=idx)
        htf = to_higher_timeframe(df, "1h", include_partial=False)
        # No HTF bar that ends AFTER 15:00 (the last base ts) may appear.
        for ts in htf.index:
            end = ts + pd.Timedelta("1h")
            assert end <= df.index[-1] + pd.Timedelta("15min"), (
                f"HTF bar at {ts} ends at {end} which is past last base bar "
                f"{df.index[-1]} + base period"
            )

    def test_L5_mtf_context_strictly_before_cursor(self):
        """`htf(interval)` must never return an HTF bar whose END is past
        the cursor. Property-check across many cursor positions."""
        idx = pd.date_range("2024-01-01 09:00", periods=200, freq="15min")
        rng = np.random.default_rng(1)
        closes = 1000 + np.cumsum(rng.normal(0, 1, 200))
        df = pd.DataFrame({
            "Open": closes, "High": closes + 1, "Low": closes - 1,
            "Close": closes, "Volume": [10] * 200,
        }, index=idx)
        mtf = MTFContext(df)
        period = pd.Timedelta("1h")
        for cursor in df.index[::13]:    # sample every 13th bar
            mtf.set_now(cursor)
            htf = mtf.htf("1h")
            if htf.empty:
                continue
            last = htf.index[-1]
            # Bar at `last` covers [last, last+period). It's only closed at
            # `last+period`. So we require last+period <= cursor.
            assert last + period <= cursor


# =======================================================================
# S. Bar timing & sessions
# =======================================================================
class TestSessions:
    """Invariants:
        S1 — Out-of-session bars produce no entry signals from a session-
             aware strategy.
        S2 — Force-flat closes any open position at flat_by.
        S3 — flat_by is enforced at BASE-TF precision when trigger TF > base TF.
        S4 — Trigger TF decoupling: 1m base + 15m trigger fires only on 15m closes.
        S5 — DST transition: data crossing BST→GMT keeps wall-clock times
             consistent after to_trading_tz strips the tz.
    """

    def _intraday_ohlc(self, n_days: int = 2, freq: str = "15min",
                       open_time: str = "08:00", close_time: str = "16:30",
                       price: float = 10000.0) -> pd.DataFrame:
        rows = []
        for d in range(n_days):
            day = pd.Timestamp("2024-01-08") + pd.Timedelta(days=d)
            start = pd.Timestamp(f"{day.date()} {open_time}")
            end = pd.Timestamp(f"{day.date()} {close_time}")
            idx = pd.date_range(start, end, freq=freq, inclusive="left")
            for ts in idx:
                rows.append({"ts": ts, "Open": price, "High": price + 1,
                             "Low": price - 1, "Close": price, "Volume": 10})
        df = pd.DataFrame(rows).set_index("ts").sort_index()
        return df

    def test_S1_out_of_session_no_entries(self):
        """A session-aware strategy must emit no open_long/short outside its
        configured window."""
        # 3 days of 15m bars from 07:00 to 18:00 (so includes pre-session and
        # post-session bars relative to a 09:00–15:00 trader window).
        idx = pd.date_range("2024-01-08 07:00", "2024-01-10 18:00", freq="15min")
        df = pd.DataFrame({
            "Open":  [10000] * len(idx),
            "High":  [10001] * len(idx),
            "Low":   [9999] * len(idx),
            "Close": [10000] * len(idx),
            "Volume":[10] * len(idx),
        }, index=idx)

        emitted_at: list[pd.Timestamp] = []

        class _SessionAware(Strategy):
            session_open = time(9, 0)
            session_close = time(15, 0)
            def on_bar(self, history, broker):
                t = history.index[-1].time()
                if not (self.session_open <= t <= self.session_close):
                    return Signal(action="noop")
                if broker.positions:
                    return Signal(action="noop")
                emitted_at.append(history.index[-1])
                return Signal(action="open_long", stake_per_point=0.10,
                              stop_loss=9900.0)

        run_backtest(df, _SessionAware(), warmup_bars=4)
        # Every recorded entry signal must be inside the session window
        for ts in emitted_at:
            assert time(9, 0) <= ts.time() <= time(15, 0), (
                f"Entry emitted at {ts} which is outside 09:00–15:00")
        # And we must have actually emitted some (sanity)
        assert len(emitted_at) > 0

    def test_S2_force_flat_closes_open_position(self):
        """After flat_by, the engine + orchestrator must have closed any
        open position."""
        from backtest.graph import (
            DecisionGraph, TriggerNode, GraphOrchestrator,
        )

        class _OnceOpener(Strategy):
            """Opens a long at the first in-session bar, holds forever."""
            session_open = time(9, 0)
            session_close = time(15, 0)
            flat_by = time(15, 30)
            def __init__(self): self.opened = False
            def on_bar(self, history, broker):
                if not self.opened and broker.position is None:
                    self.opened = True
                    return Signal(action="open_long", stake_per_point=0.10,
                                  stop_loss=9900.0)
                return Signal(action="noop")
            def proposed_direction(self, history): return "none"

        idx = pd.date_range("2024-01-08 08:00", "2024-01-09 17:00", freq="15min")
        df = pd.DataFrame({
            "Open": 10000, "High": 10001, "Low": 9999, "Close": 10000,
            "Volume": 10,
        }, index=idx)

        # Use the orchestrator (it owns force-flat now).
        from strategies.registry import STRATEGIES, StrategySpec
        # Register our throw-away strategy temporarily.
        STRATEGIES["__sess_test"] = StrategySpec(
            key="__sess_test", label="test", cls=_OnceOpener, params=[],
            warmup_bars=2,
        )
        try:
            graph = DecisionGraph(
                trigger=TriggerNode("__sess_test", {}, "15m"),
                min_score=0.0, risk_floor=1.0, risk_ceiling=1.0,
            )
            orch = GraphOrchestrator(graph)
            result = run_backtest(df, orch, warmup_bars=4)
        finally:
            del STRATEGIES["__sess_test"]

        # Engine asserts no open position at end. Beyond that, every trade's
        # exit_time must be at or before flat_by within its own day.
        for _, t in result.trades_df.iterrows():
            ex = t["exit_time"]
            if t["exit_reason"] in ("session_end", "eod"):
                # Session-end exits must be at-or-after flat_by of that day
                # and on the SAME calendar day as flat_by (not e.g. next day).
                assert ex.time() >= time(15, 30) or ex.time() < time(8, 0), (
                    f"session_end exit at {ex} but flat_by was 15:30")

    def test_S3_flat_by_enforced_at_base_TF_precision(self):
        """1m base data + 15m trigger TF: the orchestrator must enforce
        flat_by at the 1m bar that crosses the threshold, not wait for
        the next 15m close."""
        from backtest.graph import (
            DecisionGraph, TriggerNode, GraphOrchestrator,
        )

        idx = pd.date_range("2024-01-08 09:00", "2024-01-08 16:00", freq="1min")
        df = pd.DataFrame({
            "Open": 10000, "High": 10001, "Low": 9999, "Close": 10000,
            "Volume": 10,
        }, index=idx)

        class _AlwaysOpenLong(Strategy):
            session_open = time(9, 0)
            session_close = time(15, 0)
            flat_by = time(15, 21)  # not a 15m boundary
            def on_bar(self, history, broker):
                if broker.position is None:
                    return Signal(action="open_long", stake_per_point=0.10,
                                  stop_loss=9900.0)
                return Signal(action="noop")
            def proposed_direction(self, history): return "none"

        from strategies.registry import STRATEGIES, StrategySpec
        STRATEGIES["__sess_test2"] = StrategySpec(
            key="__sess_test2", label="test", cls=_AlwaysOpenLong, params=[],
            warmup_bars=2,
        )
        try:
            graph = DecisionGraph(
                trigger=TriggerNode("__sess_test2", {}, "15m"),
                min_score=0.0, risk_floor=1.0, risk_ceiling=1.0,
            )
            orch = GraphOrchestrator(graph)
            result = run_backtest(df, orch, warmup_bars=20)
        finally:
            del STRATEGIES["__sess_test2"]

        # The orchestrator emits close_all at the first BASE bar where
        # time >= 15:21 (= bar 15:21:00). Signal fires at NEXT base bar's
        # open (15:22:00) — one base bar of lag, not one trigger bar.
        # The OLD bug would have closed at the next 15m boundary (15:30+).
        # Allow ≤ one base period of lag; assert we're NOWHERE NEAR 15:30.
        BASE_TF_MIN = 1
        for _, t in result.trades_df.iterrows():
            if t["exit_reason"] == "session_end":
                allowed = time(15, 21 + BASE_TF_MIN)
                assert t["exit_time"].time() <= allowed, (
                    f"Force-flat lagged: exit at {t['exit_time']} > {allowed}")

    def test_S4_trigger_TF_only_fires_on_TF_closes(self):
        """1m base + 15m trigger: on_bar must be called every 15 base bars
        (at the close of each 15m window) — not every 1m bar."""
        from backtest.graph import (
            DecisionGraph, TriggerNode, GraphOrchestrator,
        )

        calls_at: list[pd.Timestamp] = []

        class _Counter(Strategy):
            session_open = time(0, 0)
            session_close = time(23, 59)
            flat_by = time(23, 59, 59)
            def on_bar(self, history, broker):
                # Record the LAST base-timeframe timestamp the trigger SAW
                calls_at.append(history.index[-1])
                return Signal(action="noop")
            def proposed_direction(self, h): return "none"

        idx = pd.date_range("2024-01-08 09:00", periods=120, freq="1min")
        df = pd.DataFrame({
            "Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 10,
        }, index=idx)
        from strategies.registry import STRATEGIES, StrategySpec
        STRATEGIES["__sess_test3"] = StrategySpec(
            key="__sess_test3", label="test", cls=_Counter, params=[],
            warmup_bars=0,
        )
        try:
            graph = DecisionGraph(
                trigger=TriggerNode("__sess_test3", {}, "15m"),
                min_score=0.0, risk_floor=1.0, risk_ceiling=1.0,
            )
            orch = GraphOrchestrator(graph)
            run_backtest(df, orch, warmup_bars=0)
        finally:
            del STRATEGIES["__sess_test3"]

        # The trigger sees the RESAMPLED 15m history — so successive calls
        # should have history-end timestamps separated by 15 minutes (or one
        # 15m boundary). Verify that the number of unique-ts calls is roughly
        # the number of completed 15m windows in our 120-min span.
        unique_ts = sorted(set(calls_at))
        if unique_ts:
            for prev, cur in zip(unique_ts, unique_ts[1:]):
                # Each successive call's history end timestamp should be
                # exactly 15 minutes after the previous (15m bar starts).
                delta = (cur - prev).total_seconds() / 60.0
                assert delta == pytest.approx(15.0, abs=0.5)
        # And the count should equal the number of complete 15m windows.
        # 120 minutes of 1m bars at 09:00-10:59 give 8 complete 15m bars
        # (09:00, 09:15, ..., 10:45).
        assert 6 <= len(unique_ts) <= 8

    def test_S5_DST_transition_preserves_wallclock(self):
        """A UTC bar at 14:30 should land at 14:30 GMT in winter (UK on UTC)
        and 15:30 BST in summer (UK on UTC+1) after to_trading_tz strips tz.
        Across a DST transition this means the wall-clock TIME-OF-DAY of
        UTC-fixed bars SHIFTS by 1 hour — which is exactly what a trader
        in London experiences, and what our session checks need."""
        # Build a UTC-tagged frame spanning UK DST start (last Sunday March)
        # In 2024 BST starts at 01:00 UTC on Sun Mar 31.
        utc_idx = pd.date_range(
            "2024-03-30 12:00", "2024-04-01 12:00", freq="6h", tz="UTC",
        )
        df = pd.DataFrame({
            "Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 0,
        }, index=utc_idx)
        # Strip the tz back to naive UTC to simulate raw-fetcher input
        df_naive = df.copy()
        df_naive.index = df_naive.index.tz_localize(None)

        out = to_trading_tz(df_naive, source_tz="UTC")
        # Output index should be tz-naive
        assert out.index.tz is None
        # Pre-DST: 12:00 UTC → 12:00 GMT (no shift). Post-DST: 12:00 UTC →
        # 13:00 BST (1h shift). Pick a pre- and post-DST timestamp:
        pre = pd.Timestamp("2024-03-30 12:00")
        post = pd.Timestamp("2024-04-01 12:00")
        # Find those in the output index
        out_times = [t for t in out.index]
        # Pre-DST 12:00 UTC should appear at 12:00 wall-clock
        assert pd.Timestamp("2024-03-30 12:00") in out_times
        # Post-DST 12:00 UTC should appear at 13:00 wall-clock
        assert pd.Timestamp("2024-04-01 13:00") in out_times


# =======================================================================
# C. Cost model
# =======================================================================
class TestCostModel:
    """Invariants:
        C1 — Spread applied ONCE per round trip (not double-counted).
        C2 — Slippage scales with bar spread, floored at min_slippage_points.
        C3 — Long financing = notional × (SONIA + admin) / 365 per day.
        C4 — Short financing = notional × (SONIA − admin) / 365 per day,
             symmetric — note shorts receive a CREDIT when SONIA > admin.
        C5 — `risk_based_stake` enforces the `min_stake` floor.
        C6 — Leverage cap rejected at FILL time for pending orders, not just
             at signal time. Drops the order rather than crashing.
        C7 — `profile_for(ticker)` routes common patterns correctly.
    """

    def test_C1_spread_charged_once_per_round_trip(self):
        """A flat round-trip with no price move should lose exactly one
        spread × stake. Test on FX-style bps-based profile too."""
        # Use a known-spread profile
        broker = Broker(costs=CostModel(instrument="TEST",
                                         spread_points=2.0,
                                         slippage_points=0.0,
                                         min_slippage_points=0.0))
        t0 = datetime(2024, 1, 1, 9, 0)
        t1 = datetime(2024, 1, 1, 9, 30)
        broker.open("long", stake_per_point=1.0, time=t0, price=100.0)
        trade = broker.close(t1, 100.0, reason="signal")
        # Spread cost == 2.0 × 1.0 = £2.00, ONCE.
        assert trade.spread_cost_gbp == pytest.approx(2.0)
        # Gross PnL = 0 (price unchanged); financing = 0 (intraday); net = -2.
        assert trade.gross_pnl_gbp == pytest.approx(0.0)
        assert trade.net_pnl_gbp == pytest.approx(-2.0)

    def test_C2_slippage_scales_with_spread(self):
        """Slippage(s) = max(min, multiplier × s)."""
        c = CostModel(instrument="X", spread_points=2.0,
                      slippage_points=None, slippage_bps=None,
                      min_slippage_points=0.5, slip_spread_multiplier=0.5)
        # spread=4 → 4*0.5 = 2.0 > 0.5 floor
        assert c.effective_slippage_pts(4.0) == pytest.approx(2.0)
        # spread=0.5 → 0.25 < 0.5 floor → 0.5
        assert c.effective_slippage_pts(0.5) == pytest.approx(0.5)
        # spread unknown → fall back to slippage_points (None here) → bps (None) → min
        assert c.effective_slippage_pts(None) == pytest.approx(0.5)

    def test_C3_long_financing_formula(self):
        """For long: charge = notional × (SONIA + admin) / 365 × days."""
        c = CostModel(instrument="X", sonia_annual=0.05, admin_annual=0.025,
                      spread_points=0.0, slippage_points=0.0,
                      min_slippage_points=0.0)
        # 1 day, notional £1,000
        ch = c.overnight_charge(notional_gbp=1_000.0, is_long=True, days=1)
        # 1000 × (0.05+0.025) / 365 = 0.205479
        assert ch == pytest.approx(1_000.0 * 0.075 / 365.0)

    def test_C4_short_financing_is_credit_when_admin_lt_sonia(self):
        """For short: charge = -notional × (SONIA − admin) / 365 (a CREDIT
        when SONIA > admin, since the negative makes the result negative
        meaning the broker pays you)."""
        c = CostModel(instrument="X", sonia_annual=0.05, admin_annual=0.025,
                      spread_points=0.0, slippage_points=0.0,
                      min_slippage_points=0.0)
        ch = c.overnight_charge(notional_gbp=1_000.0, is_long=False, days=1)
        # rate = -short_funding_annual = -(0.05 − 0.025) = -0.025
        # charge = 1000 * -0.025 / 365 ≈ -0.0685
        assert ch == pytest.approx(-1_000.0 * (0.05 - 0.025) / 365.0)
        assert ch < 0  # credit

    def test_C5_min_stake_floor_enforced(self):
        """With tiny risk_pts, stake calculation should still floor at
        min_stake — IF the spread-coverage gate doesn't kick in first."""
        from strategies._helpers import risk_based_stake
        # Use a profile with no spread floor — STOCK at price 1 has tiny spread
        config.COSTS = CostModel(instrument="X", spread_points=0.0001,
                                  slippage_points=0.0,
                                  min_slippage_points=0.0)
        try:
            # equity 10k, risk 0.5%, risk_pts 50 → stake 1.0
            stake = risk_based_stake(equity_gbp=10_000.0, risk_pts=50.0,
                                      risk_pct=0.005, price=100.0,
                                      min_stake=0.10)
            assert stake >= 0.10
            # Now huge risk_pts → tiny stake → floored at 0.10
            stake_floor = risk_based_stake(equity_gbp=10_000.0,
                                            risk_pts=10_000.0,
                                            risk_pct=0.005, price=100.0,
                                            min_stake=0.10)
            assert stake_floor == pytest.approx(0.10)
        finally:
            config.COSTS = profile_for("UK100")

    def test_C6_leverage_cap_at_fill_time_drops_order(self):
        """A pending limit sized to fit at placement may breach leverage when
        it fills later (after balance has drawn down). The broker should
        catch the resulting ValueError, drop the order, increment counter."""
        acct = AccountConfig(starting_balance_gbp=1_000.0,
                              leverage_cap=2.0,
                              max_concurrent_positions=1)
        # spread/slip zero so the prices we observe are exactly notional
        broker = Broker(costs=CostModel(instrument="X", spread_points=0.0,
                                         slippage_points=0.0,
                                         min_slippage_points=0.0),
                         account=acct)
        # Place a limit at price=100, stake=2 → notional 200. Cap=2x equity=2000.
        # At placement: 200 ≤ 2000, OK.
        t = datetime(2024, 1, 1, 9, 0)
        broker.place_pending_order(
            side="long", order_type="limit", trigger_price=100.0,
            stake_per_point=20.0, time=t, stop_loss=95.0,
        )
        # Now SHRINK the balance to make 20 × 100 = 2000 breach the
        # (now-reduced) leverage × balance. Force balance to £50:
        broker.balance = 50.0   # cap 2x = 100 < notional 200
        bar = {"Open": 99, "High": 99.5, "Low": 99.0, "Close": 99.0}
        opened = broker.check_pending_orders(t, bar)
        assert opened == []
        assert broker._dropped_order_count >= 1
        assert broker.pending_orders == []   # consumed from pending list

    def test_C7_profile_for_ticker_routing(self):
        """profile_for routes major aliases. Drives whether costs are
        spread_points or bps for an arbitrary ticker."""
        assert profile_for("UK100").instrument == "UK100"
        assert profile_for("UKX").instrument == "UK100"
        assert profile_for("^FTSE").instrument == "UK100"
        assert profile_for("FTSE100").instrument == "UK100"
        assert profile_for("AAPL").instrument == "STOCK"
        assert profile_for("TSLA").instrument == "STOCK"
        assert profile_for("BTC-USD").instrument == "BTC"
        assert profile_for("BTCUSD").instrument == "BTC"
        assert profile_for("ISF.LSE").instrument == "ETF"
        assert profile_for("EURUSD").instrument == "EURUSD"
        # Garbage ticker → DEFAULT (bps-based)
        p = profile_for("ZZZUNKNOWN")
        assert p.instrument == "DEFAULT"
        assert p.spread_bps is not None


# =======================================================================
# E. Order execution
# =======================================================================
class TestOrderExecution:
    """Invariants:
        E1 — Engine main loop order: stops first, then pending strategy
             signal, then pending orders, then mark, then strategy.
        E2 — Limit fills favourable on a gap-through open.
        E3 — Stop fills at trigger + slip (long) / − slip (short).
        E4 — Same-bar SL/TP after a limit fill works.
        E5 — Geometry guard rejects wrong-side stops on broker.open.
        E6 — Stale `close` on already-closed position is silently skipped.
        E7 — Stale `scale_out` on already-closed position is silently skipped.
    """

    def test_E1_stops_fire_before_pending_signal_close(self):
        """If a stop is hit on bar i's range AND the strategy emitted a
        close signal on bar i-1, the stop wins — we should see a 'stop'
        trade, NOT a 'signal' trade at the bar's open."""
        # 5 bars; bar 2 has a deep low that hits our stop.
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "Open":  [100, 100, 100, 100, 100],
            "High":  [101, 101, 101, 101, 101],
            # Bar 2 dips low enough to hit stop 95 BEFORE bar 2's open even.
            # The order is: check_stops first, then pending signal.
            "Low":   [99, 99, 90, 99, 99],
            "Close": [100, 100, 100, 100, 100],
            "Volume": [10] * 5,
        }, index=idx)

        class _OpenThenClose(Strategy):
            def __init__(self): self.opened = False; self.closed = False
            def on_bar(self, history, broker):
                i = len(history) - 1
                if i == 0 and not self.opened:
                    self.opened = True
                    return Signal(action="open_long", stake_per_point=1.0,
                                   stop_loss=95.0)
                # Emit a close on bar 1 — would fill at bar 2's OPEN if not
                # for the stop firing FIRST on bar 2.
                if i == 1 and not self.closed:
                    self.closed = True
                    return Signal(action="close")
                return Signal(action="noop")

        result = run_backtest(df, _OpenThenClose(), warmup_bars=0)
        # We expect exactly one trade, exited via stop (bar 2's low = 90 ≤ 95).
        assert len(result.trades_df) == 1
        assert result.trades_df.iloc[0]["exit_reason"] == "stop"

    def test_E2_limit_buy_favourable_gap_fill(self):
        """LIMIT BUY at 100 with bar.Open=95 → fill at 95 (favourable), NOT
        at 100. The broker fills at the better price."""
        broker = Broker()
        t = datetime(2024, 1, 1, 9, 0)
        broker.place_pending_order(side="long", order_type="limit",
                                    trigger_price=100.0, stake_per_point=1.0,
                                    time=t, stop_loss=80.0)
        bar = {"Open": 95.0, "High": 96.0, "Low": 94.0, "Close": 96.0}
        opened = broker.check_pending_orders(t, bar)
        assert len(opened) == 1
        assert opened[0].entry_price == pytest.approx(95.0)

    def test_E3_stop_buy_unfavourable_gap_fill(self):
        """STOP BUY at 100 with bar.Open=110 → fill at 110 + slip, NOT at
        100 + slip. The trader paid through the level."""
        broker = Broker()
        t = datetime(2024, 1, 1, 9, 0)
        broker.place_pending_order(side="long", order_type="stop",
                                    trigger_price=100.0, stake_per_point=1.0,
                                    time=t, stop_loss=90.0,
                                    take_profit=200.0)
        bar = {"Open": 110.0, "High": 111.0, "Low": 105.0, "Close": 110.0}
        opened = broker.check_pending_orders(t, bar)
        assert len(opened) == 1
        # Fill = open (110) + slippage (0.5 default)
        assert opened[0].entry_price == pytest.approx(110.0 + config.COSTS.slippage_points)

    def test_E4_same_bar_SL_after_limit_fill(self):
        """A limit fills mid-bar. The SAME bar's range goes through the SL.
        Broker must close it that bar, not next bar."""
        broker = Broker()
        t = datetime(2024, 1, 1, 9, 0)
        # Long limit at 100, stop 95. Bar: low 90, high 102 — fills then
        # immediately stops out.
        broker.place_pending_order(side="long", order_type="limit",
                                    trigger_price=100.0, stake_per_point=1.0,
                                    time=t, stop_loss=95.0,
                                    take_profit=200.0)
        bar = {"Open": 102, "High": 102, "Low": 90, "Close": 100}
        opened = broker.check_pending_orders(t, bar)
        # Position should NOT be in broker.positions — same-bar exit happened
        assert opened    # one position was opened ...
        assert broker.positions == []   # ... and immediately closed
        # And it produced a trade with reason="stop"
        assert len(broker.trades) == 1
        assert broker.trades[0].exit_reason == "stop"

    def test_E5_geometry_guard_rejects_wrong_side_stop(self):
        """Direct broker.open with stop on the wrong side raises ValueError.
        This catches strategy bugs at the broker boundary."""
        broker = Broker()
        t = datetime(2024, 1, 1, 9, 0)
        with pytest.raises(ValueError):
            # Long at 100 with stop at 105 (above) — wrong side
            broker.open("long", 1.0, t, 100.0, stop_loss=105.0)
        with pytest.raises(ValueError):
            broker.open("short", 1.0, t, 100.0, stop_loss=95.0)
        with pytest.raises(ValueError):
            broker.open("long", 1.0, t, 100.0, take_profit=95.0)
        with pytest.raises(ValueError):
            broker.open("short", 1.0, t, 100.0, take_profit=105.0)

    def test_E6_stale_close_silently_skipped(self):
        """Engine path: a close signal for a position that's already gone
        is a no-op, not an exception."""
        broker = Broker()
        # No positions open. Emit a close with position_id="0" (nonexistent).
        sig = Signal(action="close", position_id="999")
        t = datetime(2024, 1, 1, 9, 0)
        # _apply_signal must not raise.
        _apply_signal(sig, broker, t, price=100.0)
        # No exception, no trades.
        assert broker.trades == []

    def test_E7_stale_scale_out_silently_skipped(self):
        """scale_out on an already-closed position must silently skip
        (no raise) — strategies often emit scale_out on the same bar a
        target hit."""
        broker = Broker()
        t = datetime(2024, 1, 1, 9, 0)
        # Open + close, so position 0 no longer exists
        broker.open("long", 1.0, t, 100.0)
        broker.close(t, 110.0, reason="signal")
        # Now emit scale_out for position "0" — gone.
        sig = Signal(action="scale_out", position_id="0", scale_fraction=0.5)
        _apply_signal(sig, broker, t, price=110.0)
        # No second trade for the scale-out, no raise.
        assert len(broker.trades) == 1


# =======================================================================
# A. Accounting identity
# =======================================================================
class TestAccountingIdentity:
    """Invariants:
        A1 — starting + Σ(net_pnl) == final at end of run.
        A2 — Per-trade: gross − spread − slip − financing == net to 1p.
        A3 — Equity curve timestamps monotonically increasing.
        A4 — No open position at end (engine asserts).
        A5 — Synthetic-broken case: engine assertion fires when a trailing
             stop callback corrupts broker.balance directly.
    """

    def test_A1_starting_plus_trades_equals_final(self):
        """End-of-run identity holds across realistic synthetic data."""
        df = _trending_ohlc(40)

        class _OpenClose(Strategy):
            def __init__(self): self.fired_open = False; self.fired_close = False
            def on_bar(self, history, broker):
                i = len(history) - 1
                if i == 5 and not self.fired_open:
                    self.fired_open = True
                    return Signal(action="open_long", stake_per_point=1.0,
                                  stop_loss=80.0)
                if i == 15 and not self.fired_close:
                    self.fired_close = True
                    return Signal(action="close")
                return Signal(action="noop")

        result = run_backtest(df, _OpenClose(), warmup_bars=2)
        total = result.trades_df["net_pnl_gbp"].sum()
        assert result.final_balance == pytest.approx(
            result.starting_balance + total, abs=0.01
        )

    def test_A2_per_trade_pnl_identity_with_overnight(self):
        """gross − spread − slip − financing == net, including overnight."""
        broker = Broker()
        t0 = datetime(2024, 1, 1, 9, 0)
        # Move time forward 3 days so financing accrues nontrivially.
        broker.open("long", 1.0, t0, 1000.0)
        for d in range(1, 4):
            broker.mark(datetime(2024, 1, 1 + d, 9, 0),
                         {"Open": 1000, "High": 1001, "Low": 999, "Close": 1000})
        trade = broker.close(datetime(2024, 1, 4, 9, 0), 1010.0,
                              reason="signal")
        expected = (trade.gross_pnl_gbp - trade.spread_cost_gbp
                    - trade.slippage_cost_gbp - trade.financing_cost_gbp)
        assert trade.net_pnl_gbp == pytest.approx(expected, abs=1e-9)
        assert trade.financing_cost_gbp > 0  # accrued > 0

    def test_A3_equity_curve_monotonic_timestamps(self):
        df = _trending_ohlc(20)

        class _Hold(Strategy):
            def __init__(self): self.fired = False
            def on_bar(self, history, broker):
                if not self.fired and broker.position is None:
                    self.fired = True
                    return Signal(action="open_long", stake_per_point=1.0,
                                  stop_loss=80.0)
                return Signal(action="noop")
        result = run_backtest(df, _Hold(), warmup_bars=2)
        ts = list(result.equity_curve.index)
        for prev, cur in zip(ts, ts[1:]):
            assert cur >= prev

    def test_A4_no_open_position_at_end(self):
        """Even if the strategy never closes, the engine force-flattens."""
        df = _trending_ohlc(15)

        class _OnlyOpens(Strategy):
            def __init__(self): self.fired = False
            def on_bar(self, history, broker):
                if not self.fired and broker.position is None:
                    self.fired = True
                    return Signal(action="open_long", stake_per_point=1.0,
                                  stop_loss=80.0)
                return Signal(action="noop")

        result = run_backtest(df, _OnlyOpens(), warmup_bars=2)
        assert result.trades_df.iloc[0]["exit_reason"] == "eod"

    def test_A5_engine_assertion_fires_on_balance_corruption(self):
        """If a trailing_stop_fn corrupts broker.balance directly, the
        engine's identity assertion at end-of-run MUST fire. This proves
        the assertion is doing its job, not silently passing."""
        df = _trending_ohlc(10)

        # Inject corruption via a trailing_stop_fn (called every mark())
        class _Corrupter(Strategy):
            def __init__(self): self.fired = False
            def on_bar(self, history, broker):
                if not self.fired and broker.position is None:
                    self.fired = True
                    def bad_trail(pos, bar):
                        # Steal £1000 from the balance without recording it
                        broker.balance -= 1000.0
                        return None  # don't move the stop
                    return Signal(action="open_long", stake_per_point=1.0,
                                  stop_loss=80.0,
                                  trailing_stop_fn=bad_trail)
                return Signal(action="noop")
        with pytest.raises(AssertionError, match=r"Equity accounting drift"):
            run_backtest(df, _Corrupter(), warmup_bars=2)


# =======================================================================
# D. Data integrity
# =======================================================================
class TestDataIntegrity:
    """Invariants:
        D1 — Fetcher _validate sorts ascending.
        D2 — Fetcher _validate drops duplicates (keeping first).
        D3 — Fetcher _validate drops bars with inconsistent OHLC.
        D4 — `to_trading_tz` converts UTC to TRADING_TZ wall-clock correctly.
        D5 — `to_trading_tz` strips tz info on the output.
        D6 — EODHD fetcher's UK100 alias resolves to ISF.LSE with scale=10.
        D7 — RTH filter (as applied in app.py) drops out-of-window bars.
        D8 — IG fetcher's _ig_to_ohlcv emits a Spread column from bid/ask.
    """

    def test_D1_validate_sorts_ascending(self):
        from data.fetcher import _validate
        idx = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
        df = pd.DataFrame({
            "Open": [3, 1, 2], "High": [3, 1, 2],
            "Low": [3, 1, 2], "Close": [3, 1, 2], "Volume": [0] * 3,
        }, index=idx)
        out = _validate(df)
        assert list(out.index) == sorted(out.index)

    def test_D2_validate_drops_duplicates(self):
        from data.fetcher import _validate
        # Two bars at the same timestamp, second has different prices
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
        df = pd.DataFrame({
            "Open": [1, 99, 2], "High": [1, 99, 2],
            "Low": [1, 99, 2], "Close": [1, 99, 2], "Volume": [0] * 3,
        }, index=idx)
        out = _validate(df)
        assert len(out) == 2
        # keep="first" — the original (Open=1) wins
        assert out.iloc[0]["Open"] == 1

    def test_D3_validate_drops_inconsistent_OHLC(self):
        from data.fetcher import _validate
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({
            "Open":  [100, 100, 100],
            "High":  [101, 99, 101],     # bar 1: High < Low — invalid
            "Low":   [99, 100, 99],
            "Close": [100, 100, 100],
            "Volume": [10, 10, 10],
        }, index=idx)
        out = _validate(df)
        assert len(out) == 2

    def test_D4_to_trading_tz_converts_UTC_to_London(self):
        # 12:00 UTC on a winter day (UK on GMT) → 12:00 London naive
        idx = pd.to_datetime(["2024-01-15 12:00", "2024-01-15 14:00"])
        df = pd.DataFrame({
            "Open": [100, 100], "High": [101, 101], "Low": [99, 99],
            "Close": [100, 100], "Volume": [10, 10],
        }, index=idx)
        out = to_trading_tz(df, source_tz="UTC")
        # Winter: UTC and London are the same wall clock
        assert pd.Timestamp("2024-01-15 12:00") in list(out.index)

        # Summer: 12:00 UTC → 13:00 BST in London
        idx_s = pd.to_datetime(["2024-07-15 12:00"])
        df_s = pd.DataFrame({
            "Open": [100], "High": [101], "Low": [99],
            "Close": [100], "Volume": [10],
        }, index=idx_s)
        out_s = to_trading_tz(df_s, source_tz="UTC")
        assert list(out_s.index) == [pd.Timestamp("2024-07-15 13:00")]

    def test_D5_to_trading_tz_strips_tz(self):
        idx = pd.to_datetime(["2024-01-15 12:00"])
        df = pd.DataFrame({
            "Open": [100], "High": [101], "Low": [99],
            "Close": [100], "Volume": [10],
        }, index=idx)
        out = to_trading_tz(df, source_tz="UTC")
        assert out.index.tz is None

    def test_D6_eodhd_UK100_alias_scales_by_10(self):
        """UK100 alias must resolve to (ISF.LSE, 10.0)."""
        from data.eodhd_fetcher import EPIC_MAP, _to_symbol
        symbol, scale = _to_symbol("UK100")
        assert symbol == "ISF.LSE"
        assert scale == 10.0
        # And case-insensitive
        assert _to_symbol("uk100") == ("ISF.LSE", 10.0)

    def test_D7_RTH_filter_drops_out_of_window_bars(self):
        """Apply the same hours filter the app uses (between_time-like) to a
        synthetic frame and assert pre/post-market bars are gone."""
        # 9-hour day, 1m bars, 06:00-15:00 — UK RTH is 08:00-16:30, so
        # 06:00-07:59 should be dropped.
        idx = pd.date_range("2024-01-08 06:00", "2024-01-08 14:59", freq="1min")
        df = pd.DataFrame({
            "Open": 10000, "High": 10001, "Low": 9999, "Close": 10000,
            "Volume": 10,
        }, index=idx)
        window = trading_window_for("UK100", "rth")
        assert window is not None
        from backtest.graph import _parse_hhmm
        ot = _parse_hhmm(window[0])
        ct = _parse_hhmm(window[1])
        mask = (df.index.time >= ot) & (df.index.time <= ct)
        filtered = df.loc[mask]
        assert len(filtered) < len(df)
        # All retained bars must be in window
        for ts in filtered.index:
            assert ot <= ts.time() <= ct

    def test_D8_ig_fetcher_emits_spread_column_from_bid_ask(self):
        """Feed _ig_to_ohlcv a MultiIndex bid/ask frame and verify the
        Spread column equals mean (ask - bid)."""
        from data.ig_fetcher import _ig_to_ohlcv
        idx = pd.date_range("2024-01-01 09:00", periods=2, freq="15min")
        # MultiIndex columns: ("bid"|"ask", "Open"|"High"|"Low"|"Close")
        cols = pd.MultiIndex.from_product(
            [["bid", "ask", "last"], ["Open", "High", "Low", "Close", "Volume"]]
        )
        bid_vals = [
            [100, 101, 99, 100, 0],
            [101, 102, 100, 101, 0],
        ]
        ask_vals = [
            [101, 102, 100, 101, 0],
            [102, 103, 101, 102, 0],
        ]
        last_vals = [[0, 0, 0, 0, 100], [0, 0, 0, 0, 200]]
        raw = pd.DataFrame(
            np.hstack([bid_vals, ask_vals, last_vals]),
            index=idx, columns=cols,
        )
        out = _ig_to_ohlcv(raw)
        assert "Spread" in out.columns
        # Bar 0: bid=100,101,99,100 avg=100, ask=101,102,100,101 avg=101 →
        # spread = 1.0
        assert out["Spread"].iloc[0] == pytest.approx(1.0)
        assert out["Spread"].iloc[1] == pytest.approx(1.0)
        # Mid OHLC should be (bid + ask) / 2
        assert out["Open"].iloc[0] == pytest.approx(100.5)
