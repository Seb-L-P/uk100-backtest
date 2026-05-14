"""
Ensemble / combination strategies.

An ensemble is a "meta-strategy" — it doesn't generate signals from scratch,
it asks several child strategies for their opinion on each bar and combines
those opinions into a single trade decision.

The position-handling model:
  - The ensemble OWNS the broker's single position. Children are polled in
    "advisory mode" via a MockBroker that always reports zero position.
  - This means child internal state (e.g. FvgRetest's tracked FVGs) still
    updates correctly, but their "I'm already in a position" logic doesn't
    interfere with voting.
  - The ensemble uses its OWN R-target + ATR stop logic; child stop/target
    suggestions are ignored.

Two ensemble flavours implemented here:
  1. VoteEnsemble — N children vote; M must agree on direction to trade.
  2. FilterEnsemble — one "trigger" strategy must signal; one or more "filter"
     strategies must either agree or stay neutral (no veto).

Future extensions (not built yet):
  - RegimeRouter — picks which child to use based on volatility/trend regime.
  - PortfolioEnsemble — multi-position; each child runs independently with
    its share of capital. Requires multi-position engine support.

When evaluating an ensemble, remember the multiple-testing risk: combining N
strategies and tuning combination parameters means many models tested. Always
walk-forward + reserve OOS data. Validation handles this if you use it.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import atr
from strategies._helpers import risk_based_stake


# ---- Mock broker -------------------------------------------------------
class _MockBroker:
    """
    A broker stand-in used to poll child strategies without leaking the
    ensemble's actual position state to them.

    Children can read `balance` (so position-sizing still works) but always
    see `position is None` — so they only emit entry signals, never exits.
    """
    def __init__(self, real: Broker):
        self.balance = real.balance
        self.account = real.account
        self.position = None
        self.costs = real.costs


# ---- Vote ensemble -----------------------------------------------------
class VoteEnsemble:
    """
    Take a vote across `children`. If `min_agreement` or more children agree
    on a direction (and strictly more than the opposing direction), open a
    position in that direction. Else, do nothing.

    Exits use the ensemble's own R-target + ATR-based stop set on entry.
    """
    def __init__(
        self,
        children: Sequence[Strategy],
        min_agreement: int = 2,
        r_target: float = 2.0,
        stop_atr_mult: float = 2.0,
        atr_period: int = 14,
    ):
        if not children:
            raise ValueError("VoteEnsemble needs at least one child strategy")
        if min_agreement < 1 or min_agreement > len(children):
            raise ValueError(f"min_agreement must be in [1, {len(children)}]")
        self.children = list(children)
        self.min_agreement = min_agreement
        self.r_target = r_target
        self.stop_atr_mult = stop_atr_mult
        self.atr_period = atr_period

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        # If we already have a position, the broker's stop_loss/take_profit
        # handle the exit. Don't re-enter or override.
        if broker.position is not None:
            return Signal(action="noop")

        i = len(history) - 1
        if i < self.atr_period + 2:
            return Signal(action="noop")

        mock = _MockBroker(broker)
        long_votes = 0
        short_votes = 0
        reasons = []
        for child in self.children:
            sig = child.on_bar(history, mock)
            if sig.action == "open_long":
                long_votes += 1
                reasons.append(f"+{type(child).__name__}")
            elif sig.action == "open_short":
                short_votes += 1
                reasons.append(f"-{type(child).__name__}")
            # close/noop don't count as votes

        # Need min_agreement votes for a side AND strictly more than the other
        net = long_votes - short_votes
        if long_votes >= self.min_agreement and net > 0:
            direction = "open_long"
        elif short_votes >= self.min_agreement and net < 0:
            direction = "open_short"
        else:
            return Signal(action="noop")

        # Ensemble's own stop + target
        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")
        cur_close = float(history["Close"].iloc[-1])
        risk_pts = self.stop_atr_mult * atr_now
        if direction == "open_long":
            stop = cur_close - risk_pts
            target = cur_close + self.r_target * risk_pts
        else:
            stop = cur_close + risk_pts
            target = cur_close - self.r_target * risk_pts
        stake = risk_based_stake(broker.balance, risk_pts, price=cur_close)

        return Signal(
            action=direction,
            stake_per_point=stake,
            stop_loss=stop,
            take_profit=target,
            reason=f"vote {long_votes}L/{short_votes}S [{','.join(reasons)}]",
        )


# ---- Filter ensemble ---------------------------------------------------
class FilterEnsemble:
    """
    One `trigger` strategy generates the entry signal. The trade is taken
    only if every `filter` strategy either AGREES (same direction) or stays
    neutral (noop). Any disagreement vetoes the trade.

    Useful pattern: take FVG entries only when RSI isn't extremely against
    the trade direction — keeps the core idea but adds a quality filter.
    """
    def __init__(
        self,
        trigger: Strategy,
        filters: Sequence[Strategy],
        r_target: float = 2.0,
        stop_atr_mult: float = 2.0,
        atr_period: int = 14,
    ):
        self.trigger = trigger
        self.filters = list(filters)
        self.r_target = r_target
        self.stop_atr_mult = stop_atr_mult
        self.atr_period = atr_period

    def on_bar(self, history: pd.DataFrame, broker: Broker) -> Signal:
        if broker.position is not None:
            return Signal(action="noop")

        i = len(history) - 1
        if i < self.atr_period + 2:
            return Signal(action="noop")

        mock = _MockBroker(broker)
        trigger_signal = self.trigger.on_bar(history, mock)
        if trigger_signal.action not in ("open_long", "open_short"):
            return Signal(action="noop")

        # All filters must agree or be neutral
        opposite = "open_short" if trigger_signal.action == "open_long" else "open_long"
        for f in self.filters:
            fsig = f.on_bar(history, mock)
            if fsig.action == opposite:
                return Signal(action="noop")

        # Build the entry using ensemble's own risk model
        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")
        cur_close = float(history["Close"].iloc[-1])
        risk_pts = self.stop_atr_mult * atr_now
        if trigger_signal.action == "open_long":
            stop = cur_close - risk_pts
            target = cur_close + self.r_target * risk_pts
        else:
            stop = cur_close + risk_pts
            target = cur_close - self.r_target * risk_pts
        stake = risk_based_stake(broker.balance, risk_pts, price=cur_close)

        filter_names = ",".join(type(f).__name__ for f in self.filters)
        return Signal(
            action=trigger_signal.action,
            stake_per_point=stake,
            stop_loss=stop,
            take_profit=target,
            reason=f"trigger={type(self.trigger).__name__} filters_ok=[{filter_names}]",
        )
