"""
Tests for the validation utilities — making sure holdout, walk-forward, and
Monte Carlo behave as advertised.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.validation import (
    holdout_split, walk_forward, monte_carlo_trade_shuffle,
)
from backtest.engine import run_backtest, Signal, Strategy
from backtest.metrics import compute_metrics


class _SimpleStrategy(Strategy):
    """Opens long on bar 5, closes on bar 10. Predictable for tests."""
    def __init__(self):
        self.opened_at = -1

    def on_bar(self, history, broker):
        i = len(history) - 1
        if i == 5 and self.opened_at == -1 and broker.position is None:
            self.opened_at = i
            return Signal(action="open_long", stake_per_point=1.0)
        if i == 10 and self.opened_at >= 0 and broker.position is not None:
            return Signal(action="close")
        return Signal(action="noop")


# ---- Holdout split -----------------------------------------------------
def test_holdout_split_lengths(synthetic_ohlc):
    """Default 20% OOS on 20 bars → 16 IS, 4 OOS."""
    is_data, oos_data = holdout_split(synthetic_ohlc, oos_fraction=0.2)
    assert len(is_data) == 16
    assert len(oos_data) == 4


def test_holdout_split_no_overlap(synthetic_ohlc):
    """IS and OOS must not share any rows."""
    is_data, oos_data = holdout_split(synthetic_ohlc, oos_fraction=0.25)
    assert is_data.index.intersection(oos_data.index).empty


def test_holdout_split_invalid_fraction(synthetic_ohlc):
    """Fractions outside (0, 1) should raise."""
    with pytest.raises(ValueError):
        holdout_split(synthetic_ohlc, oos_fraction=0.0)
    with pytest.raises(ValueError):
        holdout_split(synthetic_ohlc, oos_fraction=1.0)


# ---- Walk-forward ------------------------------------------------------
def test_walk_forward_creates_n_folds(trending_ohlc):
    """walk_forward returns exactly n_folds results."""
    wf = walk_forward(
        trending_ohlc,
        strategy_factory=lambda: _SimpleStrategy(),
        n_folds=4,
        warmup_bars=2,
    )
    assert len(wf.fold_results) == 4
    assert len(wf.fold_metrics) == 4


def test_walk_forward_fresh_state_per_fold(trending_ohlc):
    """
    Each fold gets a fresh strategy. If state leaked, the strategy's
    `opened_at` attribute (which only sets once) would prevent re-opens
    across folds.
    """
    wf = walk_forward(
        trending_ohlc,
        strategy_factory=lambda: _SimpleStrategy(),
        n_folds=4,
        warmup_bars=2,
    )
    # Most folds should have a trade — would not happen if strategy state
    # leaked between folds.
    n_with_trade = sum(1 for r in wf.fold_results if len(r.trades_df) >= 1)
    assert n_with_trade >= 3


# ---- Monte Carlo trade shuffle -----------------------------------------
def test_monte_carlo_final_balance_invariant(synthetic_ohlc):
    """
    Trade-order shuffling MUST leave final balance unchanged — final balance
    is just starting + sum(pnls), which is order-independent.
    """
    result = run_backtest(synthetic_ohlc, _SimpleStrategy(), warmup_bars=2)
    if len(result.trades_df) == 0:
        pytest.skip("No trades to shuffle")
    mc = monte_carlo_trade_shuffle(result, n_simulations=100)
    # p5 == p50 == p95 to within rounding
    assert mc.final_balance_p5 == pytest.approx(mc.final_balance_p50, abs=1e-6)
    assert mc.final_balance_p50 == pytest.approx(mc.final_balance_p95, abs=1e-6)


def test_monte_carlo_max_dd_varies():
    """
    With multiple trades, shuffling SHOULD produce a range of max drawdowns
    (assuming at least one losing trade exists).
    """
    # Build a fake result with both winners and losers
    import pandas as pd
    from backtest.engine import BacktestResult

    trades_df = pd.DataFrame({
        "net_pnl_gbp": [10, -5, 8, -3, 12, -7, 6, -4],
        "exit_time": pd.date_range("2024-01-01", periods=8, freq="D"),
        "exit_price": [100] * 8,
    })
    equity = pd.Series([10000] + list(10000 + pd.Series([10, -5, 8, -3, 12, -7, 6, -4]).cumsum()),
                       index=pd.date_range("2024-01-01", periods=9, freq="D"))
    result = BacktestResult(
        trades_df=trades_df, equity_curve=equity,
        final_balance=float(equity.iloc[-1]),
        starting_balance=10000.0,
        bars_processed=9, strategy_name="dummy",
    )
    mc = monte_carlo_trade_shuffle(result, n_simulations=200, seed=42)
    # p5 should differ from p95 (different shuffle orderings give different DDs)
    assert mc.max_dd_p5 < mc.max_dd_p95
