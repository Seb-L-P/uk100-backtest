"""
Ensemble / combination strategies.

An ensemble is a "meta-strategy" — it doesn't generate signals from scratch,
it asks each child strategy "what direction would you take right now?" and
combines those intents into a single trade decision.

The polling interface — `proposed_direction(history) -> "long" | "short" | "none"`
— is implemented by every strategy in the registry. It's a pure function of
the visible history: no side effects, no broker mutations, no signals. This
keeps strategies' standalone execution (limit orders, signal-based entries,
session filters, etc.) cleanly separate from how ensembles consume them.

Two ensemble flavours:
  1. VoteEnsemble — N children vote; M must agree on direction to trade.
  2. FilterEnsemble — one "trigger" strategy must propose a direction; all
     "filter" strategies must either agree or stay neutral (no veto).

The ensemble owns the final entry: it computes its OWN stop / target / stake
based on current bar's price and the ensemble-level ATR risk model. Child
strategies' specific entry prices (limit levels, etc.) are not used — the
ensemble takes a market entry at the next bar's open.

Multiple-testing risk note: combining N strategies and tuning combination
parameters means many models tested. Walk-forward + held-out OOS data are
the defences. The validation framework handles this if you use it.
"""
from __future__ import annotations

from typing import Sequence
import pandas as pd

from backtest.broker import Broker
from backtest.engine import Strategy, Signal
from backtest.indicators import atr
from strategies._helpers import risk_based_stake


# ---- Vote ensemble -----------------------------------------------------
class VoteEnsemble:
    """
    Poll each child for `proposed_direction`. If `min_agreement` or more
    agree on a direction (and strictly more than the opposite side), open
    a position. Else, do nothing.

    Exits use the ensemble's own ATR-based stop + R-target set on entry.
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
        # Don't stack on top of an existing position; broker stops/targets
        # handle the exit.
        if broker.position is not None:
            return Signal(action="noop")

        i = len(history) - 1
        if i < self.atr_period + 2:
            return Signal(action="noop")

        long_votes = 0
        short_votes = 0
        contributors: list[str] = []
        for child in self.children:
            try:
                direction = child.proposed_direction(history)
            except Exception:
                # If a child crashes, skip its vote rather than die.
                direction = "none"
            if direction == "long":
                long_votes += 1
                contributors.append(f"+{type(child).__name__}")
            elif direction == "short":
                short_votes += 1
                contributors.append(f"-{type(child).__name__}")

        net = long_votes - short_votes
        if long_votes >= self.min_agreement and net > 0:
            side = "open_long"
        elif short_votes >= self.min_agreement and net < 0:
            side = "open_short"
        else:
            return Signal(action="noop")

        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")
        cur_close = float(history["Close"].iloc[-1])
        risk_pts = self.stop_atr_mult * atr_now
        if side == "open_long":
            stop = cur_close - risk_pts
            target = cur_close + self.r_target * risk_pts
        else:
            stop = cur_close + risk_pts
            target = cur_close - self.r_target * risk_pts
        stake = risk_based_stake(broker.balance, risk_pts, price=cur_close)

        return Signal(
            action=side,
            stake_per_point=stake,
            stop_loss=stop,
            take_profit=target,
            reason=f"vote {long_votes}L/{short_votes}S [{','.join(contributors)}]",
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """Ensembles can themselves be polled — returns the vote outcome."""
        long_votes = sum(1 for c in self.children if _safe_dir(c, history) == "long")
        short_votes = sum(1 for c in self.children if _safe_dir(c, history) == "short")
        net = long_votes - short_votes
        if long_votes >= self.min_agreement and net > 0:
            return "long"
        if short_votes >= self.min_agreement and net < 0:
            return "short"
        return "none"


# ---- Filter ensemble ---------------------------------------------------
class FilterEnsemble:
    """
    One `trigger` strategy proposes a direction. The trade is taken only if
    every `filter` strategy either AGREES (same direction) or stays neutral.
    Any disagreement vetoes the trade.

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

        trigger_dir = _safe_dir(self.trigger, history)
        if trigger_dir == "none":
            return Signal(action="noop")

        opposite = "short" if trigger_dir == "long" else "long"
        for f in self.filters:
            if _safe_dir(f, history) == opposite:
                return Signal(action="noop")

        atr_now = float(atr(history, self.atr_period).iloc[-1])
        if pd.isna(atr_now) or atr_now <= 0:
            return Signal(action="noop")
        cur_close = float(history["Close"].iloc[-1])
        risk_pts = self.stop_atr_mult * atr_now
        if trigger_dir == "long":
            stop = cur_close - risk_pts
            target = cur_close + self.r_target * risk_pts
            action = "open_long"
        else:
            stop = cur_close + risk_pts
            target = cur_close - self.r_target * risk_pts
            action = "open_short"
        stake = risk_based_stake(broker.balance, risk_pts, price=cur_close)

        filter_names = ",".join(type(f).__name__ for f in self.filters)
        return Signal(
            action=action,
            stake_per_point=stake,
            stop_loss=stop,
            take_profit=target,
            reason=f"trigger={type(self.trigger).__name__} filters_ok=[{filter_names}]",
        )

    def proposed_direction(self, history: pd.DataFrame) -> str:
        """Ensembles can be polled too — same trigger+filter logic."""
        trigger_dir = _safe_dir(self.trigger, history)
        if trigger_dir == "none":
            return "none"
        opposite = "short" if trigger_dir == "long" else "long"
        for f in self.filters:
            if _safe_dir(f, history) == opposite:
                return "none"
        return trigger_dir


def _safe_dir(child: Strategy, history: pd.DataFrame) -> str:
    """Call proposed_direction safely; return 'none' on any failure."""
    try:
        d = child.proposed_direction(history)
        return d if d in ("long", "short", "none") else "none"
    except Exception:
        return "none"
