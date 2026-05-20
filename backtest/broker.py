"""
Broker simulator for IG spread bet on UK 100.

Multi-position capable. Existing single-position strategies still work via the
`broker.position` backward-compat property — it returns the only open position
or None when there's exactly zero or one. Strategies that want to manage
multiple positions concurrently use `broker.positions` (a list).

Responsibilities:
  - Apply spread on entry and exit (cost = effective_spread_pts(price) * stake)
  - Apply slippage on stop fills and market orders
  - Charge/credit overnight financing on positions held past the daily roll
  - Validate stake against account leverage cap (sum of notional across positions)
  - Support partial exits via scale_out()
  - Optional per-position trailing stop callback that runs each bar in mark()

Spread bet semantics: stake is in £/point. P&L = (exit - entry) * stake * direction.
Notional exposure for financing = remaining_stake * index_level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

from config import COSTS, ACCOUNT, CostModel, AccountConfig

Side = Literal["long", "short"]
OrderType = Literal["market", "limit", "stop"]


@dataclass
class PendingOrder:
    """
    A pending order sitting in the order book, waiting to be triggered.

    Limit orders fill when price REACHES the trigger from the favourable side
    (limit buy: price drops to trigger or below; limit sell: price rises to
    trigger or above). Fill happens at the trigger price (we got our price).

    Stop orders fill when price CROSSES the trigger (stop buy: price rises
    through trigger; stop sell: price falls through). Fill happens at the
    trigger price + slippage (we crossed the level, expect worse fill).

    `expires_after_bars`: order is cancelled after this many bars alive.
    None means "good till cancelled" (lives forever until filled or cancelled).
    """
    id: str
    side: Side
    order_type: OrderType                       # "limit" or "stop"; not "market"
    trigger_price: float
    stake_per_point: float
    placed_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_fn: Callable | None = None
    expires_after_bars: int | None = None
    bars_alive: int = 0
    # Free-form metadata to carry through the fill (confluence score, etc.)
    entry_metadata: dict = field(default_factory=dict)


@dataclass
class Trade:
    """A round-trip trade record for the trade log.

    For partial exits, each scale-out creates its own Trade with the
    portion of stake that was closed. The remainder generates another
    Trade when it eventually exits.
    """
    side: Side
    stake_per_point: float          # the stake CLOSED in this trade record
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    gross_pnl_gbp: float
    spread_cost_gbp: float
    slippage_cost_gbp: float
    financing_cost_gbp: float
    net_pnl_gbp: float
    bars_held: int
    exit_reason: str
    position_id: str = ""           # links partial exits back to their original position
    # Original SL/TP set at entry. Preserved across trailing-stop adjustments
    # so the trade inspector can visualise what the strategy WAS AIMING FOR.
    # None if the strategy didn't specify one.
    planned_stop_loss: float | None = None
    planned_take_profit: float | None = None
    # Free-form metadata captured at entry (e.g. confluence_score,
    # supporters breakdown from the decision-graph orchestrator).
    entry_metadata: dict = field(default_factory=dict)


@dataclass
class OpenPosition:
    """An open position. `remaining_stake_per_point` decreases on partial exits."""
    id: str
    side: Side
    stake_per_point: float                  # original stake at open (immutable)
    remaining_stake_per_point: float        # decreases on scale_out; 0 → fully closed
    entry_time: datetime
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    # Snapshots of SL/TP at entry. Preserved even when `stop_loss` is moved
    # by trailing logic — used by the inspector to draw "what the strategy
    # was originally aiming for" boxes.
    initial_stop_loss: float | None = None
    initial_take_profit: float | None = None
    accumulated_financing: float = 0.0      # only counts on remaining stake (proportionally)
    bars_held: int = 0
    last_funding_apply: datetime | None = None
    # Optional: function called each bar to update stop_loss (trailing stop, etc.)
    # Signature: (position, bar_dict) -> new_stop_loss or None (no change)
    trailing_stop_fn: Callable | None = None
    # The bid/ask spread observed on the bar this position was opened on.
    # When set, used (combined with the exit bar's spread) to compute the
    # round-trip spread cost. None → falls back to the active cost profile.
    entry_spread_pts: float | None = None
    # Free-form metadata captured at entry time. The decision-graph
    # orchestrator stuffs the trade's confluence score + per-supporter
    # breakdown here so it can be analysed post-run.
    entry_metadata: dict = field(default_factory=dict)


class Broker:
    """
    Simulates IG spread bet execution + financing for one or more positions.

    Single-position usage (backward-compatible — most existing strategies):
        broker.open(side="long", stake_per_point=1.0, time=t, price=p, stop_loss=...)
        # ... bars pass, broker.mark() and check_stops() called by engine ...
        broker.close(time=t, price=p, reason="signal")

    Multi-position usage:
        pos1 = broker.open("long", 1.0, t, p1, stop_loss=...)
        pos2 = broker.open("long", 0.5, t, p2, stop_loss=..., take_profit=...)
        broker.scale_out(pos1, time=t, price=p, fraction=0.5, reason="1R")
        broker.close(pos1, time=t, price=p, reason="trail_hit")

    Costs:
      Entry: implicit (spread cost recorded on close)
      Round-trip spread: effective_spread_pts(price) * stake (or proportional fraction on partial)
      Stop fills: + slippage_points * stake
      Overnight: COSTS.overnight_charge(notional, is_long) per day held
    """

    def __init__(self, costs: CostModel | None = None,
                 account: AccountConfig = ACCOUNT):
        # Look up COSTS dynamically (not at function-def time) so callers can
        # set the cost profile globally via `config.COSTS = profile_for(...)`
        # without having to plumb it through every wrapper.
        if costs is None:
            import config as _config
            costs = _config.COSTS
        self.costs = costs
        self.account = account
        self.balance = account.starting_balance_gbp
        self.equity_curve: list[tuple[datetime, float]] = []
        self.trades: list[Trade] = []
        self.positions: list[OpenPosition] = []
        self.pending_orders: list[PendingOrder] = []
        self._next_id = 0
        self._next_order_id = 0
        # Number of pending orders rejected at FILL TIME — leverage cap
        # exceeded, max positions hit, or invalid stop/target geometry.
        self._dropped_order_count = 0
        # Number of market-order entries skipped because the next bar's open
        # invalidated the strategy's stop/target geometry. Surfaced too so
        # the user can tell if a strategy is bleeding signals to gaps.
        self._dropped_geometry_count = 0
        # Number of pending orders that aged out without ever filling
        # (price never came back to the trigger within expires_after_bars).
        # For FVG-style strategies this is typically the BIGGEST cause of
        # "attempts > trades" — most FVGs don't get retested in time.
        self._expired_order_count = 0
        # Number of pending orders explicitly cancelled (by strategy logic,
        # session-end cleanup, etc.) before they could fill.
        self._cancelled_order_count = 0

    # ---- Backward-compat: single-position view ------------------------
    @property
    def position(self) -> OpenPosition | None:
        """
        Returns the single open position, or None if zero or multiple.
        Existing single-position strategies use this; they break gracefully
        if multi-position is in use (returning None instead of giving them
        the "wrong" one).
        """
        return self.positions[0] if len(self.positions) == 1 else None

    @position.setter
    def position(self, value):
        # Allow tests / legacy code to assign None to clear
        if value is None:
            self.positions = []
        else:
            raise NotImplementedError(
                "Direct assignment to broker.position is not supported in multi-position mode. "
                "Use broker.open() and broker.close()."
            )

    # ---- Position management ------------------------------------------
    def open(
        self,
        side: Side,
        stake_per_point: float,
        time: datetime,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_fn: Callable | None = None,
        entry_spread_pts: float | None = None,
        entry_metadata: dict | None = None,
    ) -> OpenPosition:
        # ---- Stop / target geometry guard --------------------------------
        # Refuses to open a position with an invalid stop/target layout —
        # i.e. a stop on the PROFIT side, or a target on the LOSS side.
        # See docs on _dropped_geometry_count for the two causes:
        #   1. Strategy-bug class (BPR pre-fix): stop locked to a static
        #      level chosen many bars ago; market fill lands past it.
        #   2. Gap class (BB/VWAP reversion): stop is ATR-relative but the
        #      next bar gaps past it.
        # For (1) the fix is to use a limit/stop pending order.
        # For (2) the right move is to SKIP the trade — the setup is gone.
        # Skipping vs raising is the caller's choice: the engine's
        # `_apply_signal` pre-checks and skips silently for market orders.
        # Pending fills still hit this and raise, because their fill price
        # equals the trigger price — a violation there means the strategy
        # set up the order wrong (a real bug we want to surface).
        if stop_loss is not None:
            if side == "long" and stop_loss >= price:
                raise ValueError(
                    f"Invalid stop geometry: LONG entry at {price:.4f} with "
                    f"stop_loss={stop_loss:.4f} (must be below entry). "
                    f"For pending limit/stop fills this is a real strategy "
                    f"bug. For market-order fills, this can happen on a "
                    f"gap — use Engine._apply_signal's pre-check to skip."
                )
            if side == "short" and stop_loss <= price:
                raise ValueError(
                    f"Invalid stop geometry: SHORT entry at {price:.4f} with "
                    f"stop_loss={stop_loss:.4f} (must be above entry)."
                )
        if take_profit is not None:
            if side == "long" and take_profit <= price:
                raise ValueError(
                    f"Invalid target geometry: LONG entry at {price:.4f} with "
                    f"take_profit={take_profit:.4f} (must be above entry)."
                )
            if side == "short" and take_profit >= price:
                raise ValueError(
                    f"Invalid target geometry: SHORT entry at {price:.4f} with "
                    f"take_profit={take_profit:.4f} (must be below entry)."
                )
        # Stop-too-close check. Catches the case where a limit fills at a
        # favourable price (bar.Open well past trigger) that compresses
        # planned risk down to less than one spread crossing. Example:
        # FvgScaleOut placed a long limit at 10160 with stop at 10148.66
        # (planned risk 11.3pt); bar.Open gapped down to 10149.96, filling
        # the limit but leaving only 1.3pt of risk above the stop — guaranteed
        # near-1R loser on intra-bar noise. Engine's _apply_signal already
        # catches the equivalent on market fills; this is the pending-order
        # path.
        if stop_loss is not None:
            risk_pts = abs(price - stop_loss)
            try:
                min_viable = self.costs.effective_spread_pts(price=price)
            except Exception:
                min_viable = 0.0
            if risk_pts < min_viable:
                raise ValueError(
                    f"Stop too close to entry: {side.upper()} at {price:.4f} "
                    f"with stop_loss={stop_loss:.4f} (risk {risk_pts:.4f}pt < "
                    f"one-spread {min_viable:.4f}pt). Pending limit likely "
                    f"filled at a favourable gap that crushed planned risk."
                )

        # Validate leverage against TOTAL notional (existing + new)
        new_notional = stake_per_point * price
        existing_notional = sum(
            p.remaining_stake_per_point * p.entry_price for p in self.positions
        )
        total = new_notional + existing_notional
        cap = self.balance * self.account.leverage_cap
        if total > cap:
            raise ValueError(
                f"Total notional £{total:.0f} (existing £{existing_notional:.0f} "
                f"+ new £{new_notional:.0f}) exceeds {self.account.leverage_cap}x "
                f"leverage cap on £{self.balance:.0f} (cap £{cap:.0f})"
            )

        # Enforce account-level concurrent-position cap
        if len(self.positions) >= self.account.max_concurrent_positions:
            raise RuntimeError(
                f"Already at max_concurrent_positions={self.account.max_concurrent_positions}. "
                f"Increase ACCOUNT.max_concurrent_positions in config.py to allow more."
            )

        pos = OpenPosition(
            id=str(self._next_id),
            side=side,
            stake_per_point=stake_per_point,
            remaining_stake_per_point=stake_per_point,
            entry_time=time,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            initial_stop_loss=stop_loss,        # immutable snapshot
            initial_take_profit=take_profit,
            last_funding_apply=time,
            trailing_stop_fn=trailing_stop_fn,
            entry_spread_pts=entry_spread_pts,
            entry_metadata=entry_metadata or {},
        )
        self._next_id += 1
        self.positions.append(pos)
        return pos

    def _resolve(self, position_or_id) -> OpenPosition:
        """Accept an OpenPosition or its id, return the OpenPosition."""
        if isinstance(position_or_id, OpenPosition):
            if position_or_id not in self.positions:
                raise ValueError(f"Position {position_or_id.id} is not open")
            return position_or_id
        if isinstance(position_or_id, str):
            for p in self.positions:
                if p.id == position_or_id:
                    return p
            raise ValueError(f"No open position with id {position_or_id!r}")
        raise TypeError(f"Expected OpenPosition or id string, got {type(position_or_id)}")

    def close(
        self,
        position_or_time=None,
        time_or_price=None,
        price_or_reason=None,
        reason: str = "signal",
        exit_spread_pts: float | None = None,
    ) -> Trade:
        """
        Close a position. Supports two calling conventions for backward compat:

        Multi-position:
            broker.close(position_or_id, time, price, reason="signal", exit_spread_pts=...)

        Single-position (legacy, when only one position is open):
            broker.close(time, price, reason="signal", exit_spread_pts=...)
        """
        if isinstance(position_or_time, (OpenPosition, str)) or (
            position_or_time is not None and isinstance(position_or_time, str)
        ):
            position = self._resolve(position_or_time)
            time = time_or_price
            price = price_or_reason
        else:
            if len(self.positions) != 1:
                raise RuntimeError(
                    f"Legacy close(time, price) requires exactly one open position; "
                    f"have {len(self.positions)}. Use close(position, time, price, reason=)."
                )
            position = self.positions[0]
            time = position_or_time
            price = time_or_price
            if isinstance(price_or_reason, str):
                reason = price_or_reason

        return self._close_full(position, time, price, reason, exit_spread_pts)

    def _close_full(
        self,
        position: OpenPosition,
        time: datetime,
        price: float,
        reason: str,
        exit_spread_pts: float | None = None,
    ) -> Trade:
        """Close 100% of the remaining stake on `position`."""
        return self._close_portion(
            position, time, price,
            portion_stake=position.remaining_stake_per_point,
            reason=reason,
            exit_spread_pts=exit_spread_pts,
        )

    def scale_out(
        self,
        position_or_id,
        time: datetime,
        price: float,
        fraction: float,
        reason: str = "scale_out",
        exit_spread_pts: float | None = None,
    ) -> Trade:
        """Close `fraction` (0 < x ≤ 1) of `position`'s remaining stake.

        Records a Trade for the closed portion. If `fraction == 1.0`, fully
        closes the position (equivalent to `close()`).
        """
        if not 0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        position = self._resolve(position_or_id)
        portion = position.remaining_stake_per_point * fraction
        return self._close_portion(position, time, price, portion, reason, exit_spread_pts)

    def _effective_spread_pts(
        self,
        position: OpenPosition,
        exit_spread_pts: float | None,
        exit_price: float | None = None,
    ) -> float:
        """
        Determine the spread (in points) to charge for this round-trip.

        - If both entry and exit spreads are known (IG data), use the average
        - If only one is known, use that one
        - Else fall back to the cost profile's effective_spread_pts(price),
          which is points-based for major indices and bps × price for stocks/
          ETFs / unknown instruments.
        """
        if position.entry_spread_pts is not None and exit_spread_pts is not None:
            return (position.entry_spread_pts + exit_spread_pts) / 2.0
        if position.entry_spread_pts is not None:
            return position.entry_spread_pts
        if exit_spread_pts is not None:
            return exit_spread_pts
        # Profile fallback — uses exit_price for bps-based instruments
        return self.costs.effective_spread_pts(price=exit_price)

    def _close_portion(
        self,
        position: OpenPosition,
        time: datetime,
        price: float,
        portion_stake: float,
        reason: str,
        exit_spread_pts: float | None = None,
    ) -> Trade:
        """Close `portion_stake` of `position`. Removes position if fully closed."""
        if portion_stake <= 0:
            raise ValueError(f"portion_stake must be positive, got {portion_stake}")
        if portion_stake > position.remaining_stake_per_point + 1e-9:
            raise ValueError(
                f"portion_stake {portion_stake} exceeds remaining "
                f"{position.remaining_stake_per_point} on position {position.id}"
            )

        direction = 1 if position.side == "long" else -1
        gross_pnl = (price - position.entry_price) * direction * portion_stake

        # Per-trade spread — prefer real bar bid/ask, else profile (points or
        # bps × price depending on instrument).
        spread_pts = self._effective_spread_pts(position, exit_spread_pts,
                                                 exit_price=price)
        spread_cost = spread_pts * portion_stake
        # Variable slippage: scales with the current bar's spread (when known),
        # so stops slip more during volatile/news bars and less in calm conditions.
        if reason in ("stop", "market"):
            slip_pts = self.costs.effective_slippage_pts(exit_spread_pts,
                                                          price=price)
            slip_cost = slip_pts * portion_stake
        else:
            slip_cost = 0.0

        # Financing on this portion (proportional to fraction of full position)
        full_stake = position.stake_per_point
        portion_fraction_of_original = portion_stake / full_stake if full_stake > 0 else 0.0
        financing_cost = position.accumulated_financing * portion_fraction_of_original

        net_pnl = gross_pnl - spread_cost - slip_cost - financing_cost

        # Self-check on accounting
        expected = gross_pnl - spread_cost - slip_cost - financing_cost
        assert abs(net_pnl - expected) < 1e-9, (
            f"net_pnl accounting drift on partial close: {net_pnl} != {expected}"
        )

        trade = Trade(
            side=position.side,
            stake_per_point=portion_stake,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=time,
            exit_price=price,
            gross_pnl_gbp=gross_pnl,
            spread_cost_gbp=spread_cost,
            slippage_cost_gbp=slip_cost,
            financing_cost_gbp=financing_cost,
            net_pnl_gbp=net_pnl,
            bars_held=position.bars_held,
            exit_reason=reason,
            position_id=position.id,
            planned_stop_loss=position.initial_stop_loss,
            planned_take_profit=position.initial_take_profit,
            entry_metadata=dict(position.entry_metadata),
        )
        self.trades.append(trade)
        self.balance += net_pnl

        # Update position
        position.remaining_stake_per_point -= portion_stake
        position.accumulated_financing -= financing_cost  # remove the portion we just realized
        if position.remaining_stake_per_point < 1e-9:
            self.positions.remove(position)

        return trade

    def close_all(self, time: datetime, price: float, reason: str = "close_all",
                  exit_spread_pts: float | None = None) -> list[Trade]:
        """Close every open position at the given price. Returns list of trades."""
        return [self._close_full(p, time, price, reason, exit_spread_pts)
                for p in list(self.positions)]

    # ---- Pending orders (limit / stop) -------------------------------
    def place_pending_order(
        self,
        side: Side,
        order_type: OrderType,
        trigger_price: float,
        stake_per_point: float,
        time: datetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop_fn: Callable | None = None,
        expires_after_bars: int | None = None,
        entry_metadata: dict | None = None,
    ) -> PendingOrder:
        """Place a limit or stop order in the order book."""
        if order_type not in ("limit", "stop"):
            raise ValueError(f"order_type must be 'limit' or 'stop', got {order_type!r}")
        order = PendingOrder(
            id=f"o{self._next_order_id}",
            side=side,
            order_type=order_type,
            trigger_price=trigger_price,
            stake_per_point=stake_per_point,
            placed_time=time,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_fn=trailing_stop_fn,
            expires_after_bars=expires_after_bars,
            entry_metadata=entry_metadata or {},
        )
        self._next_order_id += 1
        self.pending_orders.append(order)
        return order

    def cancel_pending_order(self, order_id: str) -> bool:
        """Cancel a pending order by id. Returns True if found and cancelled."""
        for o in self.pending_orders:
            if o.id == order_id:
                self.pending_orders.remove(o)
                self._cancelled_order_count += 1
                return True
        return False

    def cancel_all_pending(self) -> int:
        """Cancel every pending order. Returns the number cancelled."""
        n = len(self.pending_orders)
        self.pending_orders.clear()
        self._cancelled_order_count += n
        return n

    def check_pending_orders(self, time: datetime, bar: dict,
                             bar_spread: float | None = None) -> list[OpenPosition]:
        """
        Check each pending order against this bar's range. Fill any that triggered.

        Fill price model:
          - Limit: filled at the trigger price exactly (you got your price).
          - Stop: filled at the trigger price + slippage (you crossed and slipped).

        Order processing order on a bar is determined by `pending_orders` list
        order (insertion order); this matters if two orders interact (e.g. one
        that breaches leverage after the first fills).

        If a triggered order can't fill (leverage cap, max positions), it's
        silently removed and a warning printed — same as a failed direct open.
        """
        newly_opened: list[OpenPosition] = []
        for order in list(self.pending_orders):
            order.bars_alive += 1
            if (order.expires_after_bars is not None
                    and order.bars_alive > order.expires_after_bars):
                self.pending_orders.remove(order)
                self._expired_order_count += 1
                continue

            triggered, fill_price = self._check_order_trigger(order, bar, bar_spread)
            if not triggered:
                continue

            try:
                pos = self.open(
                    side=order.side,
                    stake_per_point=order.stake_per_point,
                    time=time,
                    price=fill_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    trailing_stop_fn=order.trailing_stop_fn,
                    entry_spread_pts=bar_spread,
                    entry_metadata=dict(order.entry_metadata),
                )
                newly_opened.append(pos)
                # ---- Same-bar SL/TP check ----
                # The position just opened MID-BAR via a limit/stop fill.
                # The rest of this bar's range can still hit our SL/TP —
                # in real life the broker's stop sits live with the fill and
                # fires the moment price crosses. Without this check we'd
                # wait until next bar's open and (worse) apply the gap-fill
                # logic, which uses next bar's open as the reference — a
                # bizarre price for an exit that should have happened in
                # the same minute as the entry.
                # Confirmed bug: AAPL trade #23, May 2026, lost 5R instead
                # of planned 1R because the same-bar stop wasn't checked.
                self._check_same_bar_exit(pos, time, bar, bar_spread)
            except (ValueError, RuntimeError) as e:
                # Common (and usually expected): leverage cap or max-positions
                # hit. Count silently — the run-end summary surfaces the count
                # without spamming the console. Strategies are expected to
                # manage their own order density.
                self._dropped_order_count += 1
            self.pending_orders.remove(order)
        return newly_opened

    def _check_same_bar_exit(self, pos: OpenPosition, time: datetime,
                             bar: dict, bar_spread: float | None) -> None:
        """
        After a position opens mid-bar (limit/stop fill), check whether the
        SAME bar's range would also have crossed its SL/TP. Uses level-based
        fills (not gap-aware) because the position was created INSIDE this
        bar — the "gap" to the bar's open is irrelevant.

        Tie-break when both could fire: stop wins (worst-case for trader).
        """
        bar_low = bar["Low"]
        bar_high = bar["High"]
        if pos.side == "long":
            if pos.stop_loss is not None and bar_low <= pos.stop_loss:
                self._close_full(pos, time, pos.stop_loss, "stop", bar_spread)
                return
            if pos.take_profit is not None and bar_high >= pos.take_profit:
                self._close_full(pos, time, pos.take_profit, "target", bar_spread)
        else:  # short
            if pos.stop_loss is not None and bar_high >= pos.stop_loss:
                self._close_full(pos, time, pos.stop_loss, "stop", bar_spread)
                return
            if pos.take_profit is not None and bar_low <= pos.take_profit:
                self._close_full(pos, time, pos.take_profit, "target", bar_spread)

    def _check_order_trigger(self, order: PendingOrder, bar: dict,
                             bar_spread: float | None = None) -> tuple[bool, float]:
        """
        Return (triggered, fill_price) given the bar's range.

        Realistic fill model — limit and stop orders behave differently
        depending on whether the bar's OPEN already crossed the trigger:

        LIMIT BUY at T:
          - bar.Open <= T   →  market opened past limit (FAVOURABLE).
                                Real broker fills you at bar.Open (better
                                than T). Skipping this case is the classic
                                backtester bug — you'd never pay T when the
                                market opened at a better price.
          - bar.Open > T but bar.Low <= T → bar wandered down to T → fill at T
          - else → no fill

        LIMIT SELL at T:
          - bar.Open >= T   →  fill at bar.Open (favourable for seller)
          - bar.Open < T but bar.High >= T → fill at T
          - else → no fill

        STOP BUY at T:
          - bar.Open >= T   →  gap up past trigger (UNFAVOURABLE).
                                Fill at bar.Open + slippage (worse than T).
          - bar.Open < T but bar.High >= T → fill at T + slippage
          - else → no fill

        STOP SELL at T:
          - bar.Open <= T   →  gap down past trigger → fill at bar.Open − slip
          - bar.Open > T but bar.Low <= T → fill at T − slip
          - else → no fill

        Bug history: prior version always filled at trigger price, which
        let limit orders open positions at unrealistic prices when bar.Open
        was already past the trigger. The strategy's stop/target geometry
        (computed pre-fill) was then often invalid relative to the actual
        market level — and the orchestrator's same-bar SL/TP check would
        immediately stop the trade out at -1R. Net effect: phantom trades.
        Confirmed in AAPL backtest, May 2026.
        """
        open_, low, high = bar["Open"], bar["Low"], bar["High"]

        if order.order_type == "limit":
            T = order.trigger_price
            if order.side == "long":
                if open_ <= T:
                    return True, open_           # already favourable
                if low <= T:
                    return True, T               # came down to trigger
            else:  # short
                if open_ >= T:
                    return True, open_           # already favourable
                if high >= T:
                    return True, T               # came up to trigger

        elif order.order_type == "stop":
            T = order.trigger_price
            slip = self.costs.effective_slippage_pts(bar_spread, price=T)
            if order.side == "long":
                if open_ >= T:
                    return True, open_ + slip    # gap-up past stop (worse)
                if high >= T:
                    return True, T + slip
            else:  # short
                if open_ <= T:
                    return True, open_ - slip    # gap-down past stop (worse)
                if low <= T:
                    return True, T - slip
        return False, 0.0

    # ---- Per-bar updates ----------------------------------------------
    def mark(self, time: datetime, bar: dict) -> None:
        """
        Update equity curve, accrue financing, and run any trailing-stop callbacks.

        `bar` is a dict-like with 'Open', 'High', 'Low', 'Close'.
        """
        unrealised = 0.0
        for pos in self.positions:
            direction = 1 if pos.side == "long" else -1
            unrealised += (bar["Close"] - pos.entry_price) * direction * pos.remaining_stake_per_point
            # Subtract expected close-out spread cost from unrealised equity,
            # so the equity curve reflects what we'd actually walk away with.
            spread_pts = self.costs.effective_spread_pts(price=bar["Close"])
            unrealised -= spread_pts * pos.remaining_stake_per_point
            unrealised -= pos.accumulated_financing
            pos.bars_held += 1

            # Daily financing on remaining stake
            if pos.last_funding_apply is not None:
                days_elapsed = (time.date() - pos.last_funding_apply.date()).days
                if days_elapsed >= 1:
                    notional = pos.remaining_stake_per_point * bar["Close"]
                    charge = self.costs.overnight_charge(
                        notional, is_long=(pos.side == "long"), days=days_elapsed
                    )
                    pos.accumulated_financing += charge
                    pos.last_funding_apply = time

            # Trailing stop callback
            if pos.trailing_stop_fn is not None:
                try:
                    new_stop = pos.trailing_stop_fn(pos, bar)
                    if new_stop is not None:
                        # Only ratchet — never widen the stop
                        if pos.side == "long":
                            if pos.stop_loss is None or new_stop > pos.stop_loss:
                                pos.stop_loss = new_stop
                        else:
                            if pos.stop_loss is None or new_stop < pos.stop_loss:
                                pos.stop_loss = new_stop
                except Exception as e:
                    # Don't let a bad trailing fn crash the backtest
                    print(f"[broker] trailing_stop_fn error on position {pos.id}: {e}")

        self.equity_curve.append((time, self.balance + unrealised))

    # ---- Stop/target check --------------------------------------------
    def check_stops(self, time: datetime, bar: dict) -> list[Trade]:
        """Check ALL open positions for stop/target hits. Return list of trades closed."""
        closed: list[Trade] = []
        bar_spread = bar.get("Spread")  # None if not present (yfinance data)
        for pos in list(self.positions):  # iterate over copy — close may mutate
            trade = self._check_one_position_stops(pos, time, bar, bar_spread)
            if trade is not None:
                closed.append(trade)
        return closed

    def _check_one_position_stops(self, pos: OpenPosition, time: datetime, bar: dict,
                                  bar_spread: float | None = None) -> Trade | None:
        """
        Check this position for stop / target hits during the bar.

        Realism note — gap fills:
          Real-life limit/stop fills happen at the FIRST traded price that
          crosses the trigger level. On a normal bar that price IS the
          trigger level (price walks through it intrabar). On a GAP bar
          (open is already past the trigger) you fill at the OPEN — better
          than your target on a favourable gap, WORSE than your stop on
          an adverse gap. Modelling this is essential for honest gap-risk
          reporting; otherwise stops appear safer than reality.

        Conservative tie-break: if both stop and target are inside the bar
        range, we assume STOP fires first (worst-case for the trader).
        """
        bar_open = bar["Open"]
        if pos.side == "long":
            # Long stop: bar's low touched the stop.
            if pos.stop_loss is not None and bar["Low"] <= pos.stop_loss:
                # Gap-down past the stop? Fill at the gap open (worse for us),
                # else fill at the stop level itself.
                fill = min(pos.stop_loss, bar_open)
                return self._close_full(pos, time, fill, "stop", bar_spread)
            # Long target: bar's high touched the target.
            if pos.take_profit is not None and bar["High"] >= pos.take_profit:
                # Gap-up past the target? Fill at the gap open (better for us),
                # else fill at the target level.
                fill = max(pos.take_profit, bar_open)
                return self._close_full(pos, time, fill, "target", bar_spread)
        else:
            # Short stop: bar's high touched the stop.
            if pos.stop_loss is not None and bar["High"] >= pos.stop_loss:
                # Gap-up past the stop? Fill at the gap open (worse for us).
                fill = max(pos.stop_loss, bar_open)
                return self._close_full(pos, time, fill, "stop", bar_spread)
            # Short target: bar's low touched the target.
            if pos.take_profit is not None and bar["Low"] <= pos.take_profit:
                # Gap-down past the target? Fill at the gap open (better for us).
                fill = min(pos.take_profit, bar_open)
                return self._close_full(pos, time, fill, "target", bar_spread)
        return None
