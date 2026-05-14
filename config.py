"""
Central configuration for the UK 100 backtester.

All cost assumptions live here so they are easy to audit and stress-test.
The defaults are deliberately CONSERVATIVE for IG spread bet on UK 100 —
real costs may be lower in calm markets, but we'd rather over-estimate
costs in backtest than under-estimate them and trade a fake edge.
"""
from dataclasses import dataclass
from pathlib import Path

# --- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CACHE = PROJECT_ROOT / "data" / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_CACHE.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --- IG spread bet UK 100 cost model ------------------------------------
@dataclass(frozen=True)
class CostModel:
    """
    Cost assumptions for IG spread bet on UK 100 (FTSE 100).

    Spread: IG quotes 1pt during UK cash hours, widens overnight/weekends.
            We use 1.5pt as a flat conservative average.
    Financing: IG charges (SONIA + admin) / 365 per day on long positions,
               (SONIA - admin) on shorts (can be a credit when rates are high).
               Admin commonly cited as 2.5%. SONIA assumed 5.25% as of May 2026
               — UPDATE this if BoE base rate changes materially.
    """
    spread_points: float = 1.5            # one-way; round-trip cost = spread * stake
    sonia_annual: float = 0.0525           # current SONIA estimate
    admin_annual: float = 0.025            # IG admin component (commonly cited 2.5%)
    slippage_points: float = 0.5           # per side, on top of spread, for stops/market orders
    guaranteed_stop_premium_pts: float = 3.0  # only applied if strategy uses guaranteed stops

    @property
    def long_funding_annual(self) -> float:
        return self.sonia_annual + self.admin_annual

    @property
    def short_funding_annual(self) -> float:
        # Note: can be negative (credit) when SONIA > admin, which it currently is NOT
        # because admin (2.5%) < SONIA (5.25%), so shorts CURRENTLY receive a credit.
        # Formula: short pays SONIA - admin if positive, receives if negative.
        return self.sonia_annual - self.admin_annual

    def overnight_charge(self, notional_gbp: float, is_long: bool, days: int = 1) -> float:
        """Daily financing charge in GBP. Positive = cost, negative = credit."""
        rate = self.long_funding_annual if is_long else -self.short_funding_annual
        return notional_gbp * rate * days / 365.0


# --- Account / risk defaults --------------------------------------------
@dataclass(frozen=True)
class AccountConfig:
    starting_balance_gbp: float = 10_000.0
    risk_per_trade_pct: float = 0.01      # risk 1% of equity per trade
    max_concurrent_positions: int = 1     # start strict; relax only with reason
    leverage_cap: float = 20.0            # safety cap; FCA retail limit on indices is 20:1


COSTS = CostModel()
ACCOUNT = AccountConfig()
