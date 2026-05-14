"""
Event-driven backtest engine.

The engine loops over historical bars one at a time and calls the strategy's
`on_bar` method with ONLY the data visible up to and including that bar.
No look-ahead. The strategy returns signals, the broker executes them at the
*next* bar's open (realistic — you can't fill at the close of the bar you
make your decision on).

Why event-driven rather than vectorised:
  - SMC/ICT-style logic (FVG, liquidity sweeps, order blocks) needs to know
    what was visible at the moment of decision. Vectorised approaches are
    fast but easy to leak future data into.
  - Realistic order execution (next-bar fills, stop/target checks within bar
    range, financing accruing daily) is naturally expressed as a loop.
  - It's slow for huge parameter sweeps — that's a deliberate trade-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from backtest.broker import Broker, Side


# ---- Strategy interface --------------------------------------------------
class Signal:
    """A trading signal from a strategy.

    Action vocabulary:
      - "noop": do nothing
      - "open_long" / "open_short": open a new position
      - "close": close the only open position (legacy single-position behaviour
                 — closes ALL positions if multiple are open and replace_all=True,
                 else needs `position_id`)
      - "close_position": close one specific position (requires `position_id`)
      - "close_all": close every open position
      - "scale_out": close a fraction of one position (requires `position_id` + `scale_fraction`)

    For new opens:
      - `replace_all` (default True for backward compat): if True, close all
        existing positions before opening. If False (multi-position style),
        opens additionally without disturbing existing positions.
      - `trailing_stop_fn` (optional): callable (position, bar) -> new_stop or None.
        Called every bar by broker.mark() to update the position's stop_loss.
        See backtest/exits.py for ready-made trailing functions.
    """

    def __init__(
        self,
        action: str,
        stake_per_point: float = 0.0,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        reason: str = "",
        position_id: str | None = None,
        scale_fraction: float | None = None,
        replace_all: bool = True,
        trailing_stop_fn=None,
    ):
        self.action = action
        self.stake_per_point = stake_per_point
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reason = reason
        self.position_id = position_id
        self.scale_fraction = scale_fraction
        self.replace_all = replace_all
        self.trailing_stop_fn = trailing_stop_fn


class Strategy(Protocol):
    """Strategies implement this protocol. See strategies/sma_crossover.py."""

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        ...


# ---- Engine --------------------------------------------------------------
@dataclass
class BacktestResult:
    trades_df: pd.DataFrame
    equity_curve: pd.Series
    final_balance: float
    starting_balance: float
    bars_processed: int
    strategy_name: str


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    warmup_bars: int = 50,
    verbose: bool = False,
) -> BacktestResult:
    """
    Run a backtest of `strategy` on `data`.

    `data` must be a DataFrame with columns Open/High/Low/Close/Volume, indexed
    by datetime, sorted ascending. `warmup_bars` is how many bars to give the
    strategy before allowing signals (so e.g. SMAs are populated).

    Execution model:
      1. At bar i, broker.mark() updates equity using bar[i]'s close.
      2. broker.check_stops() may close the position if stops/targets are hit.
      3. Strategy sees history[0..i] and may return a signal.
      4. Signal acts on bar[i+1]'s OPEN (no same-bar execution).
    """
    broker = Broker()
    n = len(data)

    # Track pending signals to execute at next bar's open
    pending: Signal | None = None

    for i in range(n):
        time = data.index[i]
        bar = data.iloc[i]
        bar_dict = {
            "Open": float(bar["Open"]),
            "High": float(bar["High"]),
            "Low": float(bar["Low"]),
            "Close": float(bar["Close"]),
        }

        # Execute any pending signal at this bar's open
        if pending is not None:
            _apply_signal(pending, broker, time, bar_dict["Open"])
            pending = None

        # Mark equity + accrue financing on any open position
        broker.mark(time, bar_dict)

        # Stop/target hit during this bar?
        broker.check_stops(time, bar_dict)

        # Strategy decides (only after warmup)
        if i >= warmup_bars:
            history = data.iloc[: i + 1]
            signal = strategy.on_bar(history, broker)
            if signal.action != "noop":
                pending = signal
                if verbose:
                    print(f"{time} signal: {signal.action} reason={signal.reason}")

    # Close ALL final open positions at the last close
    if broker.positions:
        last_time = data.index[-1]
        last_close = float(data.iloc[-1]["Close"])
        broker.close_all(last_time, last_close, reason="eod")

    # Build outputs
    trades_df = pd.DataFrame([_trade_to_dict(t) for t in broker.trades])
    if not trades_df.empty:
        trades_df = trades_df.set_index("entry_time")

    eq = pd.DataFrame(broker.equity_curve, columns=["time", "equity"]).set_index("time")["equity"]

    # ---- Self-check assertions ---------------------------------------
    # The fundamental accounting identity: starting balance + sum of trade
    # net P&L should equal final balance. If this drifts, we have a bug.
    starting = broker.account.starting_balance_gbp
    sum_trade_pnl = sum(t.net_pnl_gbp for t in broker.trades)
    expected_final = starting + sum_trade_pnl
    drift = abs(broker.balance - expected_final)
    assert drift < 0.01, (
        f"Equity accounting drift £{drift:.4f}: starting £{starting} + trades "
        f"£{sum_trade_pnl:.2f} = £{expected_final:.2f}, but broker balance £{broker.balance:.2f}. "
        "This means the engine and broker disagree about money — bug somewhere."
    )

    # No position should be open at end (we forced close above)
    assert not broker.positions, (
        f"Backtest ended with {len(broker.positions)} open position(s) — eod close logic failed."
    )

    # Equity curve should never go before the first bar
    if len(eq) > 0:
        assert eq.index.is_monotonic_increasing, (
            "Equity curve timestamps not monotonically increasing — bug in mark()."
        )

    return BacktestResult(
        trades_df=trades_df,
        equity_curve=eq,
        final_balance=broker.balance,
        starting_balance=starting,
        bars_processed=n,
        strategy_name=type(strategy).__name__,
    )


def _apply_signal(signal: Signal, broker: Broker, time, price: float) -> None:
    """
    Apply a strategy signal at the next bar's OPEN.

    Multi-position aware. For backward compat with single-position strategies,
    open_long/open_short with replace_all=True (the default) closes any
    existing positions before opening — preserves "reverse on signal" behaviour.
    """
    if signal.action in ("open_long", "open_short"):
        side = "long" if signal.action == "open_long" else "short"
        if signal.replace_all and broker.positions:
            broker.close_all(time, price, reason="reverse")
        broker.open(
            side, signal.stake_per_point, time, price,
            stop_loss=signal.stop_loss, take_profit=signal.take_profit,
            trailing_stop_fn=signal.trailing_stop_fn,
        )

    elif signal.action == "close":
        # Legacy: close THE position. With multi-position, close all unless ID given.
        if signal.position_id is not None:
            broker.close(signal.position_id, time, price,
                         reason=signal.reason or "signal")
        elif broker.positions:
            # Close all (matches legacy single-position semantics when only one open)
            broker.close_all(time, price, reason=signal.reason or "signal")

    elif signal.action == "close_position":
        if signal.position_id is None:
            raise ValueError("close_position requires position_id")
        broker.close(signal.position_id, time, price,
                     reason=signal.reason or "signal")

    elif signal.action == "close_all":
        if broker.positions:
            broker.close_all(time, price, reason=signal.reason or "close_all")

    elif signal.action == "scale_out":
        if signal.position_id is None or signal.scale_fraction is None:
            raise ValueError("scale_out requires position_id and scale_fraction")
        broker.scale_out(signal.position_id, time, price,
                         fraction=signal.scale_fraction,
                         reason=signal.reason or "scale_out")


def _trade_to_dict(t) -> dict:
    return {
        "position_id": t.position_id,
        "side": t.side,
        "stake_per_point": t.stake_per_point,
        "entry_time": t.entry_time,
        "entry_price": t.entry_price,
        "exit_time": t.exit_time,
        "exit_price": t.exit_price,
        "gross_pnl_gbp": round(t.gross_pnl_gbp, 2),
        "spread_cost_gbp": round(t.spread_cost_gbp, 2),
        "slippage_cost_gbp": round(t.slippage_cost_gbp, 2),
        "financing_cost_gbp": round(t.financing_cost_gbp, 2),
        "net_pnl_gbp": round(t.net_pnl_gbp, 2),
        "bars_held": t.bars_held,
        "exit_reason": t.exit_reason,
    }
