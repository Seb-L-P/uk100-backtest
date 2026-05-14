"""
Strategy registry — central place to declare strategies + their tunable parameters.

Adding a new strategy is a 3-step process:
  1. Implement the strategy class in strategies/your_strategy.py (must follow
     the Strategy protocol from backtest/engine.py — i.e. have an `on_bar` method).
  2. Add it to STRATEGIES below with a label, the class, and a list of ParamSpec
     describing each constructor parameter.
  3. Done. The CLI and the Streamlit UI will pick it up automatically.

Why a registry: lets the UI auto-generate parameter widgets, lets us iterate on
the strategy library without touching UI code, and forces strategies to declare
their param ranges (so we can do parameter sweeps later without manual coding).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from strategies.sma_crossover import SmaCrossover
from strategies.fvg_retest import FvgRetest
from strategies.orb import OpeningRangeBreakout
from strategies.liquidity_sweep import LiquiditySweep
from strategies.donchian import DonchianBreakout
from strategies.bpr import BalancedPriceRange
from strategies.bb_reversion import BollingerReversion
from strategies.vwap_reversion import VwapReversion
from strategies.rsi_reversion import RsiReversion
from strategies.fvg_scale_out import FvgScaleOut
from strategies.ensemble import VoteEnsemble, FilterEnsemble


ParamType = Literal["int", "float", "bool"]


@dataclass(frozen=True)
class ParamSpec:
    name: str               # constructor kwarg name
    label: str              # human-friendly label for the UI
    type: ParamType
    default: Any
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    help: str | None = None


@dataclass(frozen=True)
class StrategySpec:
    key: str                # short identifier ("fvg", "sma")
    label: str              # human-friendly name ("FVG retest")
    cls: type | None = None
    factory: Callable[..., Any] | None = None  # custom builder; used instead of cls if set
    params: list[ParamSpec] = None
    warmup_bars: int = 50
    description: str = ""

    def build(self, **kwargs) -> Any:
        """
        Instantiate the strategy. If `factory` is set, use it (this lets
        ensembles construct their child strategies). Otherwise call `cls`.
        """
        if self.factory is not None:
            return self.factory(**kwargs)
        if self.cls is None:
            raise ValueError(f"StrategySpec {self.key!r} has neither cls nor factory")
        return self.cls(**kwargs)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in (self.params or [])}


# ---- Registry ----------------------------------------------------------
STRATEGIES: dict[str, StrategySpec] = {
    "fvg": StrategySpec(
        key="fvg",
        label="FVG retest",
        cls=FvgRetest,
        warmup_bars=20,
        description=(
            "Day-trade Fair Value Gap (3-bar imbalance) retests. Detects bullish/bearish "
            "FVGs, waits for price to retrace into the gap, enters at near edge, stops past "
            "far edge, targets 2R. Flat by session close. UK session only."
        ),
        params=[
            ParamSpec("min_gap_points", "Min gap (pts)", "float",
                      default=5.0, min=1.0, max=50.0, step=0.5,
                      help="Reject FVGs smaller than this — too noisy."),
            ParamSpec("max_gap_points", "Max gap (pts)", "float",
                      default=50.0, min=10.0, max=200.0, step=5.0,
                      help="Reject FVGs larger than this — risk size becomes tiny."),
            ParamSpec("max_age_bars", "Max FVG age (bars)", "int",
                      default=30, min=5, max=200, step=5,
                      help="Drop FVGs older than this many bars — stale signals."),
            ParamSpec("stop_buffer_pts", "Stop buffer (pts)", "float",
                      default=2.0, min=0.0, max=10.0, step=0.5,
                      help="Extra cushion past the FVG far edge for stop placement."),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25,
                      help="Profit target as a multiple of risk."),
        ],
    ),
    "sma": StrategySpec(
        key="sma",
        label="SMA crossover",
        cls=SmaCrossover,
        warmup_bars=50,
        description=(
            "Classic moving-average crossover. Long when fast SMA crosses above slow, "
            "reverse to short on cross-down. ATR-based stops, 1% risk per trade. Included "
            "as a pipeline smoke-test, not as a real strategy."
        ),
        params=[
            ParamSpec("fast", "Fast SMA period", "int",
                      default=20, min=5, max=100, step=1),
            ParamSpec("slow", "Slow SMA period", "int",
                      default=50, min=20, max=300, step=5),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "orb": StrategySpec(
        key="orb",
        label="Opening Range Breakout",
        cls=OpeningRangeBreakout,
        warmup_bars=5,
        description=(
            "Day-trade. Define the high/low of the first N bars after open as the "
            "opening range. Enter long on close above the range, short on close below. "
            "Stop on the opposite side, target = R-multiple. One trade per day max."
        ),
        params=[
            ParamSpec("opening_range_bars", "Opening range bars", "int",
                      default=2, min=1, max=8, step=1,
                      help="Number of bars defining the OR. 2 on 15m = first 30 min."),
            ParamSpec("r_target", "Target (R)", "float",
                      default=1.5, min=0.5, max=4.0, step=0.25),
            ParamSpec("stop_buffer_pts", "Stop buffer (pts)", "float",
                      default=2.0, min=0.0, max=10.0, step=0.5),
        ],
    ),
    "liqsweep": StrategySpec(
        key="liqsweep",
        label="Liquidity sweep",
        cls=LiquiditySweep,
        warmup_bars=25,
        description=(
            "Day-trade. Detects when price pierces a recent swing high/low by a small "
            "amount but closes back inside the range — a 'stop hunt' rejection. Enter "
            "AGAINST the direction of the sweep."
        ),
        params=[
            ParamSpec("swing_lookback", "Swing lookback (bars)", "int",
                      default=20, min=5, max=100, step=5,
                      help="How many bars to look back for the swing high/low."),
            ParamSpec("sweep_min_pts", "Min sweep distance (pts)", "float",
                      default=3.0, min=0.5, max=20.0, step=0.5,
                      help="Minimum amount price must pierce the level by."),
            ParamSpec("stop_buffer_pts", "Stop buffer (pts)", "float",
                      default=2.0, min=0.0, max=10.0, step=0.5),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "donchian": StrategySpec(
        key="donchian",
        label="Donchian breakout (Turtle)",
        cls=DonchianBreakout,
        warmup_bars=25,
        description=(
            "Original Turtle Trader system: buy N-bar high, sell N-bar low, ATR stops, "
            "exit when price closes back inside opposite-direction channel. Designed for "
            "trend-following on daily; expect frequent fakeouts on intraday."
        ),
        params=[
            ParamSpec("channel_lookback", "Entry channel lookback", "int",
                      default=20, min=5, max=100, step=5,
                      help="N for the entry breakout. Turtles used 20 (System 1) or 55 (System 2)."),
            ParamSpec("exit_lookback", "Exit channel lookback", "int",
                      default=10, min=3, max=50, step=1,
                      help="N for the opposite-side exit channel. Should be < entry lookback."),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "bpr": StrategySpec(
        key="bpr",
        label="Balanced Price Range (BPR)",
        cls=BalancedPriceRange,
        warmup_bars=20,
        description=(
            "Day-trade. SMC concept: detects zones where a bullish FVG and a bearish FVG "
            "overlap. Trades reversal at the BPR retest, direction determined by approach. "
            "Strict filter — produces fewer but theoretically higher-quality signals."
        ),
        params=[
            ParamSpec("min_fvg_size", "Min FVG size (pts)", "float",
                      default=3.0, min=1.0, max=20.0, step=0.5),
            ParamSpec("max_fvg_age", "Max FVG age (bars)", "int",
                      default=50, min=10, max=200, step=10),
            ParamSpec("min_bpr_size", "Min BPR overlap (pts)", "float",
                      default=2.0, min=0.5, max=15.0, step=0.5),
            ParamSpec("max_bpr_age", "Max BPR age (bars)", "int",
                      default=50, min=10, max=200, step=10),
            ParamSpec("stop_buffer_pts", "Stop buffer (pts)", "float",
                      default=2.0, min=0.0, max=10.0, step=0.5),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("approach_lookback", "Approach lookback (bars)", "int",
                      default=5, min=2, max=20, step=1,
                      help="Bars back to measure approach direction into the BPR."),
        ],
    ),
    "bb_revert": StrategySpec(
        key="bb_revert",
        label="Bollinger Band reversion",
        cls=BollingerReversion,
        warmup_bars=25,
        description=(
            "Mean reversion: when close breaks outside the upper/lower band, fade back "
            "to the middle SMA. ATR-based stops. Optional regime filter to skip in high "
            "volatility periods where the strategy turns into a trend-chaser by mistake."
        ),
        params=[
            ParamSpec("bb_period", "BB period", "int",
                      default=20, min=5, max=100, step=1),
            ParamSpec("bb_mult", "BB std multiplier", "float",
                      default=2.0, min=1.0, max=4.0, step=0.25),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=1.5, min=0.5, max=5.0, step=0.25),
            ParamSpec("require_low_vol_regime", "Skip in high vol regime", "bool",
                      default=False),
        ],
    ),
    "vwap_revert": StrategySpec(
        key="vwap_revert",
        label="VWAP mean reversion",
        cls=VwapReversion,
        warmup_bars=20,
        description=(
            "Day-trade. When price stretches a fixed number of points from session "
            "VWAP, fade back to VWAP. Resets each day. One trade per day max."
        ),
        params=[
            ParamSpec("entry_stretch_pts", "Min stretch from VWAP (pts)", "float",
                      default=15.0, min=2.0, max=100.0, step=1.0),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=1.5, min=0.5, max=5.0, step=0.25),
            ParamSpec("max_trades_per_day", "Max trades per day", "int",
                      default=1, min=1, max=5, step=1),
        ],
    ),
    "fvg_scale_out": StrategySpec(
        key="fvg_scale_out",
        label="FVG retest with scale-out + ATR trail",
        cls=FvgScaleOut,
        warmup_bars=20,
        description=(
            "Same FVG entry as 'fvg' but exit logic uses partial profit-taking: "
            "scale out half at 1R, move the runner to break-even, then trail "
            "with ATR. Demonstrates the new multi-exit / trailing-stop engine."
        ),
        params=[
            ParamSpec("min_gap_points", "Min gap (pts)", "float",
                      default=5.0, min=1.0, max=50.0, step=0.5),
            ParamSpec("max_gap_points", "Max gap (pts)", "float",
                      default=50.0, min=10.0, max=200.0, step=5.0),
            ParamSpec("max_age_bars", "Max FVG age (bars)", "int",
                      default=30, min=5, max=200, step=5),
            ParamSpec("stop_buffer_pts", "Stop buffer (pts)", "float",
                      default=2.0, min=0.0, max=10.0, step=0.5),
            ParamSpec("scale_at_R", "Scale-out at (R)", "float",
                      default=1.0, min=0.5, max=3.0, step=0.25,
                      help="When price travels this many R in your favour, scale out."),
            ParamSpec("scale_fraction", "Scale-out fraction", "float",
                      default=0.5, min=0.1, max=0.9, step=0.1,
                      help="Fraction of position to close at the scale-out trigger."),
            ParamSpec("trail_atr_period", "Trail ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("trail_atr_mult", "Trail ATR multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "rsi_revert": StrategySpec(
        key="rsi_revert",
        label="RSI mean reversion",
        cls=RsiReversion,
        warmup_bars=20,
        description=(
            "Long when RSI hooks up from oversold; short when RSI hooks down from "
            "overbought. Exit when RSI crosses back through 50. ATR-based stops."
        ),
        params=[
            ParamSpec("rsi_period", "RSI period", "int",
                      default=14, min=2, max=50, step=1),
            ParamSpec("oversold", "Oversold threshold", "float",
                      default=30.0, min=10.0, max=45.0, step=1.0),
            ParamSpec("overbought", "Overbought threshold", "float",
                      default=70.0, min=55.0, max=90.0, step=1.0),
            ParamSpec("exit_level", "Exit RSI level", "float",
                      default=50.0, min=40.0, max=60.0, step=1.0),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),

    # ---- ENSEMBLES ------------------------------------------------------
    # Combination strategies. Each preset has fixed children; only the
    # ensemble-level params (vote threshold, R-target, ATR-stop) are tunable.
    # Once you've validated that the basic ensembles work, custom combinations
    # can be added by mirroring the factory pattern below.
    "vote_meanrev": StrategySpec(
        key="vote_meanrev",
        label="Vote: mean-revert (BB+RSI+VWAP)",
        factory=lambda **p: VoteEnsemble(
            children=[BollingerReversion(), RsiReversion(), VwapReversion()],
            **p,
        ),
        warmup_bars=25,
        description=(
            "Three mean-reversion children vote. By default trades when at least 2 "
            "of the 3 (BB / RSI / VWAP) agree on direction. Ensemble's own ATR stop + R-target."
        ),
        params=[
            ParamSpec("min_agreement", "Min strategies agreeing", "int",
                      default=2, min=1, max=3, step=1,
                      help="2 = majority; 3 = unanimous."),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("stop_atr_mult", "Stop ATR multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
        ],
    ),
    "vote_trend": StrategySpec(
        key="vote_trend",
        label="Vote: trend (Donchian+SMA)",
        factory=lambda **p: VoteEnsemble(
            children=[DonchianBreakout(), SmaCrossover()],
            **p,
        ),
        warmup_bars=55,
        description=(
            "Two trend-following children vote. Both must agree by default. "
            "Designed as a counterpoint to vote_meanrev — completely different regime."
        ),
        params=[
            ParamSpec("min_agreement", "Min strategies agreeing", "int",
                      default=2, min=1, max=2, step=1),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.5, min=0.5, max=5.0, step=0.25),
            ParamSpec("stop_atr_mult", "Stop ATR multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
        ],
    ),
    "filter_fvg_rsi": StrategySpec(
        key="filter_fvg_rsi",
        label="FVG filtered by RSI",
        factory=lambda **p: FilterEnsemble(
            trigger=FvgRetest(),
            filters=[RsiReversion()],
            **p,
        ),
        warmup_bars=25,
        description=(
            "FVG retest is the trigger; only takes the trade if RSI strategy doesn't "
            "veto (i.e. RSI isn't screaming the opposite direction). Example of "
            "trigger + filter pattern: keep the core idea, add a quality screen."
        ),
        params=[
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("stop_atr_mult", "Stop ATR multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
        ],
    ),
}


def get(key: str) -> StrategySpec:
    if key not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {key}. Known: {list(STRATEGIES)}")
    return STRATEGIES[key]


def labels() -> dict[str, str]:
    """Map of key -> label, for UI dropdowns."""
    return {spec.key: spec.label for spec in STRATEGIES.values()}
