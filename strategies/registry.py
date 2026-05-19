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
from strategies.mtf_trend_fvg import MtfTrendFvg
from strategies.macd_crossover import MacdCrossover
from strategies.stoch_crossover import StochasticCrossover
from strategies.engulfing import EngulfingReversal
from strategies.inside_bar import InsideBarBreakout
from strategies.heikin_ashi_trend import HeikinAshiTrend
from strategies.triple_ema import TripleEma
from strategies.pivot_reversal import PivotReversal
from strategies.adx_trend import AdxTrend
from strategies.psar_flip import ParabolicSarFlip
from strategies.keltner_breakout import KeltnerBreakout
from strategies.overnight_range import OvernightRangeBreakout
from strategies.mfi_extremes import MfiExtremes
# NOTE: strategy composition (the old vote/filter ensembles) now lives in
# the decision-graph framework — backtest/graph.py. The registry only holds
# atomic strategies.


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
            ParamSpec("min_gap_atr_mult", "Min gap (× ATR)", "float",
                      default=0.5, min=0.1, max=3.0, step=0.1,
                      help="Reject FVGs smaller than this many ATRs — too noisy. "
                           "Scale-invariant across instruments."),
            ParamSpec("max_gap_atr_mult", "Max gap (× ATR)", "float",
                      default=5.0, min=1.0, max=20.0, step=0.5,
                      help="Reject FVGs larger than this many ATRs — risk size becomes tiny."),
            ParamSpec("max_age_bars", "Max FVG age (bars)", "int",
                      default=30, min=5, max=200, step=5,
                      help="Drop FVGs older than this many bars — stale signals."),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.2, min=0.0, max=2.0, step=0.05,
                      help="Extra cushion past the FVG far edge for stop placement, in ATRs."),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
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
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.2, min=0.0, max=2.0, step=0.05),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
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
            ParamSpec("sweep_min_atr_mult", "Min sweep distance (× ATR)", "float",
                      default=0.3, min=0.05, max=2.0, step=0.05,
                      help="Minimum amount price must pierce the level by, in ATRs."),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.2, min=0.0, max=2.0, step=0.05),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
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
            ParamSpec("min_fvg_atr_mult", "Min FVG size (× ATR)", "float",
                      default=0.3, min=0.05, max=2.0, step=0.05),
            ParamSpec("max_fvg_age", "Max FVG age (bars)", "int",
                      default=50, min=10, max=200, step=10),
            ParamSpec("min_bpr_atr_mult", "Min BPR overlap (× ATR)", "float",
                      default=0.2, min=0.05, max=1.5, step=0.05),
            ParamSpec("max_bpr_age", "Max BPR age (bars)", "int",
                      default=50, min=10, max=200, step=10),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.2, min=0.0, max=2.0, step=0.05),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
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
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.1, min=0.0, max=2.0, step=0.05),
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
            ParamSpec("entry_stretch_atr_mult", "Min stretch from VWAP (× ATR)", "float",
                      default=1.5, min=0.3, max=6.0, step=0.1,
                      help="How far in ATRs price must stretch from VWAP before fading."),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=1.5, min=0.5, max=5.0, step=0.25),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.1, min=0.0, max=2.0, step=0.05),
            ParamSpec("max_trades_per_day", "Max trades per day", "int",
                      default=1, min=1, max=5, step=1),
        ],
    ),
    "mtf_trend_fvg": StrategySpec(
        key="mtf_trend_fvg",
        label="MTF FVG (HTF trend filter)",
        cls=MtfTrendFvg,
        warmup_bars=20,
        description=(
            "Day-trade. Same limit-order FVG entry as the base FVG strategy, "
            "but FILTERED by higher-timeframe trend (default: 1h EMA50 + slope). "
            "Only takes longs in HTF uptrends, shorts in HTF downtrends. "
            "Trades less but theoretically with the trend on its side."
        ),
        params=[
            ParamSpec("min_gap_atr_mult", "Min gap (× ATR)", "float",
                      default=0.5, min=0.1, max=3.0, step=0.1),
            ParamSpec("max_gap_atr_mult", "Max gap (× ATR)", "float",
                      default=5.0, min=1.0, max=20.0, step=0.5),
            ParamSpec("max_age_bars", "Max FVG age (bars)", "int",
                      default=30, min=5, max=200, step=5),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.2, min=0.0, max=2.0, step=0.05),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("htf_ema_period", "HTF EMA period", "int",
                      default=50, min=10, max=200, step=10,
                      help="EMA on the higher timeframe; trend slope and "
                           "close-vs-EMA decide bias."),
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
            ParamSpec("min_gap_atr_mult", "Min gap (× ATR)", "float",
                      default=0.5, min=0.1, max=3.0, step=0.1),
            ParamSpec("max_gap_atr_mult", "Max gap (× ATR)", "float",
                      default=5.0, min=1.0, max=20.0, step=0.5),
            ParamSpec("max_age_bars", "Max FVG age (bars)", "int",
                      default=30, min=5, max=200, step=5),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.2, min=0.0, max=2.0, step=0.05),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
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
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.1, min=0.0, max=2.0, step=0.05),
        ],
    ),

    # ---- New strategies (Phase 14) -------------------------------------
    "macd_cross": StrategySpec(
        key="macd_cross",
        label="MACD signal-line crossover",
        cls=MacdCrossover,
        warmup_bars=30,
        description=(
            "Classic momentum: long when MACD line crosses above signal "
            "line, short on cross-down. ATR-based stops, reversal on opposite signal."
        ),
        params=[
            ParamSpec("fast", "MACD fast EMA", "int", default=12, min=3, max=30, step=1),
            ParamSpec("slow", "MACD slow EMA", "int", default=26, min=10, max=60, step=1),
            ParamSpec("signal", "MACD signal EMA", "int", default=9, min=3, max=20, step=1),
            ParamSpec("atr_period", "ATR period", "int", default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "stoch_cross": StrategySpec(
        key="stoch_cross",
        label="Stochastic %K x %D",
        cls=StochasticCrossover,
        warmup_bars=25,
        description=(
            "Oscillator strategy: long when %K crosses %D inside the oversold "
            "zone, short when crossing in overbought zone. ATR-based stops."
        ),
        params=[
            ParamSpec("k_period", "%K period", "int", default=14, min=5, max=30, step=1),
            ParamSpec("d_period", "%D period", "int", default=3, min=1, max=10, step=1),
            ParamSpec("smooth_k", "Smooth %K", "int", default=3, min=1, max=10, step=1),
            ParamSpec("oversold", "Oversold threshold", "float",
                      default=30.0, min=10.0, max=40.0, step=1.0),
            ParamSpec("overbought", "Overbought threshold", "float",
                      default=70.0, min=60.0, max=90.0, step=1.0),
            ParamSpec("atr_period", "ATR period", "int", default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "engulfing": StrategySpec(
        key="engulfing",
        label="Engulfing reversal pattern",
        cls=EngulfingReversal,
        warmup_bars=25,
        description=(
            "Take entries on bullish/bearish engulfing candles. Optional "
            "trend-context filter ensures we're trading reversals (against "
            "a recent EMA-slope) not continuations."
        ),
        params=[
            ParamSpec("trend_ema_period", "Trend EMA period", "int",
                      default=20, min=5, max=100, step=1),
            ParamSpec("min_body_atr_mult", "Min engulfing body (× ATR)", "float",
                      default=0.5, min=0.1, max=3.0, step=0.1),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.1, min=0.0, max=2.0, step=0.05),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("require_trend_context", "Require opposite-trend context", "bool",
                      default=True),
        ],
    ),
    "inside_bar": StrategySpec(
        key="inside_bar",
        label="Inside-bar breakout",
        cls=InsideBarBreakout,
        warmup_bars=20,
        description=(
            "Place pending STOP orders at the high and low of any inside bar. "
            "Whichever fills first wins; the other is cancelled. ATR-based "
            "stop + 2R target. Captures range-break momentum."
        ),
        params=[
            ParamSpec("atr_period", "ATR period", "int", default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=1.5, min=0.5, max=5.0, step=0.25),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("min_inside_range_atr_mult", "Min inside-bar range (× ATR)", "float",
                      default=0.3, min=0.05, max=2.0, step=0.05),
            ParamSpec("trigger_buffer_atr_mult", "Trigger buffer (× ATR)", "float",
                      default=0.05, min=0.0, max=1.0, step=0.01),
            ParamSpec("max_age_bars", "Max order age (bars)", "int",
                      default=5, min=1, max=30, step=1),
        ],
    ),
    "ha_trend": StrategySpec(
        key="ha_trend",
        label="Heikin Ashi trend follower",
        cls=HeikinAshiTrend,
        warmup_bars=20,
        description=(
            "Enter when N consecutive Heikin Ashi candles are the same colour. "
            "Exit when HA candle colour flips. Higher N = stricter trend filter."
        ),
        params=[
            ParamSpec("consecutive_bars", "Consecutive HA bars", "int",
                      default=3, min=2, max=10, step=1),
            ParamSpec("atr_period", "ATR period", "int", default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "triple_ema": StrategySpec(
        key="triple_ema",
        label="Triple EMA stack",
        cls=TripleEma,
        warmup_bars=55,
        description=(
            "Long when EMAs are stacked up (fast > mid > slow). Short when "
            "stacked down. Exit when stack breaks. Filter for trending markets."
        ),
        params=[
            ParamSpec("fast", "Fast EMA", "int", default=9, min=3, max=30, step=1),
            ParamSpec("mid", "Mid EMA", "int", default=21, min=10, max=60, step=1),
            ParamSpec("slow", "Slow EMA", "int", default=50, min=20, max=200, step=5),
            ParamSpec("atr_period", "ATR period", "int", default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),

    # ---- Phase 15 additions --------------------------------------------
    "pivot_reversal": StrategySpec(
        key="pivot_reversal",
        label="Pivot reversal (S1/R1 fade)",
        cls=PivotReversal,
        warmup_bars=20,
        description=(
            "Day-trade. At session open, compute classic floor-trader pivots "
            "from yesterday's H/L/C. Arm limit BUY at S1 and limit SELL at R1. "
            "Stop past S2/R2, target the central pivot P. First-fill-wins; "
            "flat by session close."
        ),
        params=[
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.3, min=0.0, max=2.0, step=0.05,
                      help="Cushion past S2/R2 for stop placement."),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("max_age_bars", "Max order age (bars)", "int",
                      default=20, min=2, max=80, step=1,
                      help="Cancel unfilled limits after this many bars."),
        ],
    ),
    "adx_trend": StrategySpec(
        key="adx_trend",
        label="ADX-filtered SMA trend",
        cls=AdxTrend,
        warmup_bars=55,
        description=(
            "SMA crossover GATED by ADX > threshold and +DI/-DI directional "
            "agreement. Only takes trend trades when ADX confirms an actual "
            "trend is present. Helps avoid the classic SMA-crossover chop pain."
        ),
        params=[
            ParamSpec("fast", "Fast SMA period", "int", default=20, min=5, max=100, step=1),
            ParamSpec("slow", "Slow SMA period", "int", default=50, min=20, max=300, step=5),
            ParamSpec("adx_period", "ADX period", "int", default=14, min=5, max=50, step=1),
            ParamSpec("adx_threshold", "ADX min threshold", "float",
                      default=25.0, min=10.0, max=50.0, step=1.0,
                      help="ADX must be ≥ this to take the cross. 20-25 typical."),
            ParamSpec("atr_period", "ATR period", "int", default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "psar_flip": StrategySpec(
        key="psar_flip",
        label="Parabolic SAR flip",
        cls=ParabolicSarFlip,
        warmup_bars=30,
        description=(
            "Welles Wilder's stop-and-reverse system. Enter on SAR flip, "
            "stop at current SAR value, exit on opposite flip. Optional ADX "
            "filter to skip flips in chop."
        ),
        params=[
            ParamSpec("af_start", "AF start (acceleration)", "float",
                      default=0.02, min=0.005, max=0.1, step=0.005,
                      help="Initial acceleration factor. Wilder's default 0.02."),
            ParamSpec("af_step", "AF step", "float",
                      default=0.02, min=0.005, max=0.1, step=0.005),
            ParamSpec("af_max", "AF max", "float",
                      default=0.2, min=0.05, max=0.5, step=0.05),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.1, min=0.0, max=1.0, step=0.05),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("adx_min", "ADX gate (0 = off)", "float",
                      default=0.0, min=0.0, max=50.0, step=1.0,
                      help="Skip SAR flips when ADX is below this. 0 disables the filter."),
            ParamSpec("adx_period", "ADX period", "int",
                      default=14, min=5, max=50, step=1),
        ],
    ),
    "keltner_break": StrategySpec(
        key="keltner_break",
        label="Keltner channel breakout",
        cls=KeltnerBreakout,
        warmup_bars=35,
        description=(
            "Trend-continuation: enter LONG when close breaks above the Keltner "
            "upper channel (EMA + ATR mult), SHORT on lower break. Stop at the "
            "channel midline, target an R-multiple. Smoother than BB breakout "
            "because the bands ride ATR not std-dev."
        ),
        params=[
            ParamSpec("ema_period", "EMA period (midline)", "int",
                      default=20, min=5, max=100, step=1),
            ParamSpec("atr_period", "ATR period (band width)", "int",
                      default=10, min=5, max=50, step=1),
            ParamSpec("mult", "Channel multiplier", "float",
                      default=2.0, min=1.0, max=4.0, step=0.25,
                      help="Bands = EMA ± mult × ATR."),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.2, min=0.0, max=2.0, step=0.05),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
        ],
    ),
    "overnight_range": StrategySpec(
        key="overnight_range",
        label="Overnight range breakout",
        cls=OvernightRangeBreakout,
        warmup_bars=30,
        description=(
            "Day-trade. At today's session open, arm pending stop orders at "
            "the high (long) and low (short) of the previous N bars. Captures "
            "gap-and-go moves and breaks of yesterday's range. One trade per day."
        ),
        params=[
            ParamSpec("lookback_bars", "Lookback bars (prior session)", "int",
                      default=30, min=5, max=200, step=5,
                      help="How many bars before today's first session bar to scan for the overnight high/low."),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.3, min=0.0, max=2.0, step=0.05),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("max_age_bars", "Pending order max age (bars)", "int",
                      default=12, min=2, max=60, step=1,
                      help="Cancel unfilled stops after this many bars."),
        ],
    ),
    "mfi_extremes": StrategySpec(
        key="mfi_extremes",
        label="MFI extremes mean reversion",
        cls=MfiExtremes,
        warmup_bars=25,
        description=(
            "Volume-weighted RSI cousin. Long when MFI dipped oversold then "
            "hooks up; short on overbought hook down. Exit when MFI crosses "
            "back through 50 or stop/target. Note: volume here is the index "
            "underlying volume from Yahoo — a proxy, not your broker's flow."
        ),
        params=[
            ParamSpec("mfi_period", "MFI period", "int",
                      default=14, min=2, max=50, step=1),
            ParamSpec("oversold", "Oversold threshold", "float",
                      default=20.0, min=5.0, max=40.0, step=1.0),
            ParamSpec("overbought", "Overbought threshold", "float",
                      default=80.0, min=60.0, max=95.0, step=1.0),
            ParamSpec("exit_level", "Exit MFI level", "float",
                      default=50.0, min=40.0, max=60.0, step=1.0),
            ParamSpec("atr_period", "ATR period", "int",
                      default=14, min=5, max=50, step=1),
            ParamSpec("atr_stop_mult", "ATR stop multiplier", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("stop_buffer_atr_mult", "Stop buffer (× ATR)", "float",
                      default=0.1, min=0.0, max=2.0, step=0.05),
            ParamSpec("r_target", "Target (R)", "float",
                      default=2.0, min=0.5, max=5.0, step=0.25),
            ParamSpec("lookback_for_extreme", "Hook lookback (bars)", "int",
                      default=5, min=2, max=20, step=1,
                      help="How recently MFI must have hit the extreme before the hook."),
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
