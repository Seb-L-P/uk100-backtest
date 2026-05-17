"""
DEPRECATED — superseded by backtest/graph.py (DecisionGraph).

This module previously hosted the confluence wrapper. Its functionality is
now provided by the decision-graph framework, which generalises the concept
(trigger / supporters / vetoes / per-trade score recording / risk scaling)
into the engine itself.

Existing callers should migrate:
  ConfluenceWrapper(strategy, engine)  →  GraphOrchestrator(graph)

The shim below exists only so old imports don't crash; it raises a clear
error pointing at the replacement.
"""
from __future__ import annotations


def _removed(*args, **kwargs):
    raise RuntimeError(
        "backtest.confluence has been replaced by backtest.graph. "
        "Use DecisionGraph + GraphOrchestrator instead."
    )


# Symbols kept for compatibility — they all raise on use.
ConfluenceEngine = _removed
ConfluenceWrapper = _removed
HtfTrendConfluence = _removed
HtfFvgConfluence = _removed
HtfSwingLevelConfluence = _removed
HtfMomentumConfluence = _removed
build_engine = _removed
