"""
Central configuration for the backtester.

All cost assumptions live here so they are easy to audit and stress-test.
The cost model is now PER-INSTRUMENT and SCALE-INVARIANT:

  - Each known instrument has a calibrated profile (real IG spread bet defaults
    at the time of writing — UPDATE these if your broker's spreads change).
  - Unknown instruments fall back to a `spread_bps` (basis points of price)
    model so backtests on AAPL ($200) and BTC ($60k) both use sensible costs
    automatically. Without this, a flat "1.5 points" spread is 1.5 bps on FTSE
    10k but 75 bps on AAPL 200 — wildly wrong.
  - When the data source provides REAL bid/ask spreads per bar (IG demo,
    EODHD intraday), those win over any profile estimate.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# --- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CACHE = PROJECT_ROOT / "data" / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_CACHE.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Cost model ---------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    """
    Cost assumptions for a SINGLE instrument.

    Spread:
      Prefer real per-bar bid/ask (set on the bar via Spread column) if known.
      Else use `spread_points` if set (good for major indices with stable spread).
      Else use `spread_bps × price / 10_000` (scale-invariant fallback).

    Slippage on stops/markets:
      If real bar spread known: scales with it (slip_spread_multiplier × spread).
      Else: `slippage_points` if set, else `slippage_bps × price / 10_000`.
      Floored at `min_slippage_points`.

    Financing:
      Long pays sonia_annual + admin_annual per year on notional.
      Short receives sonia_annual - admin_annual (can be a credit when rates are high).

    All numbers below are IG-calibrated retail spread bet defaults at the time
    of writing (May 2026). Override per instrument or per run as needed.
    """
    instrument: str = "UK100"

    # Spread (one-way, applied round-trip on entry+exit).
    spread_points: float | None = 1.5     # absolute points — used when set
    spread_bps: float | None = None       # basis points of price — fallback

    # Slippage on stops + market orders.
    slippage_points: float | None = 0.5
    slippage_bps: float | None = None
    min_slippage_points: float = 0.2
    slip_spread_multiplier: float = 0.5

    # Financing (annualised, applied daily on overnight notional).
    sonia_annual: float = 0.0525
    admin_annual: float = 0.025

    # Guaranteed-stop premium (only applied if strategy explicitly opts in).
    guaranteed_stop_premium_pts: float = 3.0

    # ---- Financing -----------------------------------------------------
    @property
    def long_funding_annual(self) -> float:
        return self.sonia_annual + self.admin_annual

    @property
    def short_funding_annual(self) -> float:
        return self.sonia_annual - self.admin_annual

    def overnight_charge(self, notional_gbp: float, is_long: bool, days: int = 1) -> float:
        """Daily financing charge in GBP. Positive = cost, negative = credit."""
        rate = self.long_funding_annual if is_long else -self.short_funding_annual
        return notional_gbp * rate * days / 365.0

    # ---- Spread / slippage --------------------------------------------
    def effective_spread_pts(self, price: float | None = None,
                             bar_spread: float | None = None) -> float:
        """
        Effective spread for this fill, in points.

        Resolution order:
          1. Real bar spread (IG/EODHD bid/ask) — wins if available
          2. spread_points (if set) — for instruments with stable spread
          3. spread_bps × price (scale-invariant fallback)
          4. 1.0 last-ditch
        """
        if bar_spread is not None and bar_spread > 0:
            return float(bar_spread)
        if self.spread_points is not None:
            return self.spread_points
        if self.spread_bps is not None and price is not None and price > 0:
            return price * self.spread_bps / 10_000.0
        return 1.0

    def effective_slippage_pts(self, bar_spread: float | None = None,
                               price: float | None = None) -> float:
        """
        Slippage in points. Scales with real bar spread when known.
        Falls back to slippage_points, then slippage_bps × price, then min.

        Note: arg order is (bar_spread, price) for backward-compat with
        existing tests that call effective_slippage_pts(spread).
        """
        if bar_spread is not None and bar_spread > 0:
            return max(self.min_slippage_points,
                       self.slip_spread_multiplier * float(bar_spread))
        if self.slippage_points is not None:
            return self.slippage_points
        if self.slippage_bps is not None and price is not None and price > 0:
            return max(self.min_slippage_points,
                       price * self.slippage_bps / 10_000.0)
        return self.min_slippage_points


# --- Realistic IG retail spread-bet cost profiles -----------------------
# Calibrated against IG's published "minimum spread" values (May 2026).
# These are CONSERVATIVE (slightly above the minimum, since you rarely fill
# at the absolute minimum spread except on highly liquid bars).
#
# For instruments not listed, the loader picks DEFAULT (bps-based) so any
# new asset works out of the box.
PROFILES: dict[str, CostModel] = {
    # --- Indices (spread in absolute points, stable across price level) ---
    "UK100":    CostModel(instrument="UK100",    spread_points=1.5, slippage_points=0.5, min_slippage_points=0.2),
    "US500":    CostModel(instrument="US500",    spread_points=0.5, slippage_points=0.2, min_slippage_points=0.1),
    "US100":    CostModel(instrument="US100",    spread_points=1.0, slippage_points=0.4, min_slippage_points=0.2),
    "DJI":      CostModel(instrument="DJI",      spread_points=2.0, slippage_points=1.0, min_slippage_points=0.5),
    "GER40":    CostModel(instrument="GER40",    spread_points=1.2, slippage_points=0.5, min_slippage_points=0.2),
    "JPN225":   CostModel(instrument="JPN225",   spread_points=7.0, slippage_points=3.0, min_slippage_points=1.0),
    "FRA40":    CostModel(instrument="FRA40",    spread_points=1.5, slippage_points=0.6, min_slippage_points=0.3),
    # --- Forex (spread in pip-points, scales differently per pair) --------
    "EURUSD":   CostModel(instrument="EURUSD",   spread_points=0.6, slippage_points=0.2, min_slippage_points=0.1),
    "GBPUSD":   CostModel(instrument="GBPUSD",   spread_points=0.9, slippage_points=0.3, min_slippage_points=0.1),
    "USDJPY":   CostModel(instrument="USDJPY",   spread_points=0.7, slippage_points=0.3, min_slippage_points=0.1),
    # --- Crypto (spread in points, large absolute but small in bps) -------
    "BTC":      CostModel(instrument="BTC",      spread_points=40.0, slippage_points=20.0, min_slippage_points=10.0),
    "ETH":      CostModel(instrument="ETH",      spread_points=3.0,  slippage_points=1.5,  min_slippage_points=0.5),
    # --- Single-stock CFDs / ETFs (bps-based, scales with price) ----------
    # ETFs trade tight — ISF.LSE realistic spread is ~5 bps including IG markup.
    "ETF":      CostModel(instrument="ETF",      spread_points=None, spread_bps=5.0,
                          slippage_points=None,  slippage_bps=2.0,   min_slippage_points=0.01),
    "STOCK":    CostModel(instrument="STOCK",    spread_points=None, spread_bps=10.0,
                          slippage_points=None,  slippage_bps=4.0,   min_slippage_points=0.01),
    # --- Default fallback for unknown instruments --------------------------
    # Conservative 10 bps spread + 5 bps slippage. Roughly matches the cost
    # of trading a mid-cap stock CFD with retail IG.
    "DEFAULT":  CostModel(instrument="DEFAULT",  spread_points=None, spread_bps=10.0,
                          slippage_points=None,  slippage_bps=5.0,   min_slippage_points=0.01),
}


def profile_for(instrument: str | None) -> CostModel:
    """
    Look up a cost profile by instrument name (case-insensitive).

    Recognises common ticker/symbol patterns:
      - "UK100", "UKX", "FTSE"  → UK100 profile
      - "US500", "SPX"          → US500 profile
      - "ISF.LSE", "VOD.L"      → ETF / STOCK profile (bps-based)
      - "BTC-USD", "BTCUSD"     → BTC profile
      - anything else           → DEFAULT (bps-based, scale-invariant)

    This means: if you backtest on AAPL, ISF.LSE, BTC, or anything custom,
    costs auto-adapt to price. You no longer get the "1.5pt = 17bps on
    ISF.LSE" bug that wiped out the FVG backtest at the start.
    """
    if instrument is None:
        return PROFILES["DEFAULT"]
    s = instrument.upper().strip()

    # Direct hit
    if s in PROFILES:
        return PROFILES[s]

    # Common aliases
    if s in ("UKX", "UKX.INDX", "FTSE", "FTSE100", "^FTSE", "^UKX"):
        return PROFILES["UK100"]
    if s in ("SPX", "^SPX", "^GSPC", "SP500"):
        return PROFILES["US500"]
    if s in ("NDX", "^NDX", "^IXIC", "NAS100", "NASDAQ100"):
        return PROFILES["US100"]
    if s in ("^DJI", "WALL", "WALL30", "DOW"):
        return PROFILES["DJI"]
    if s in ("DAX", "DAX40", "^GDAXI"):
        return PROFILES["GER40"]
    if s in ("NIKKEI", "N225", "^N225"):
        return PROFILES["JPN225"]
    if s.startswith("BTC") or s.endswith("BTC"):
        return PROFILES["BTC"]
    if s.startswith("ETH") or s.endswith("ETH"):
        return PROFILES["ETH"]

    # ETF-ish suffixes
    if s.endswith(".LSE") or s.endswith(".L") or s.endswith(".AS"):
        return PROFILES["ETF"]

    # Default: bps-based, scale-invariant
    return PROFILES["DEFAULT"]


# --- Account / risk defaults --------------------------------------------
@dataclass(frozen=True)
class AccountConfig:
    starting_balance_gbp: float = 10_000.0
    risk_per_trade_pct: float = 0.01      # risk 1% of equity per trade
    max_concurrent_positions: int = 1     # start strict; relax only with reason
    leverage_cap: float = 20.0            # safety cap; FCA retail limit on indices is 20:1
    min_stake_per_point: float = 0.10     # IG minimum bet size (most markets)


# Active config — defaults to UK100 for backward-compat with existing code/tests.
COSTS = PROFILES["UK100"]
ACCOUNT = AccountConfig()


def with_profile(instrument: str) -> CostModel:
    """Convenience: get the profile for an instrument, ready to pass to Broker."""
    return profile_for(instrument)
