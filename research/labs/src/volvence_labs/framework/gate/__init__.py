"""Gate module: Two-Gate + SGM e-value for SHADOW→ACTIVE promotion decisions.

The gate module is independent of individual probes. It takes an ExperimentReport
and produces a PromotionDecision (approve / hold / reject).
"""

from .two_gate import TwoGate, CapacityBound, ValidationMargin
from .sgm import SGMGate, EValue
from .aggregator import GateAggregator, GateDecision, PromotionDecision

__all__ = [
    "TwoGate",
    "CapacityBound",
    "ValidationMargin",
    "SGMGate",
    "EValue",
    "GateAggregator",
    "GateDecision",
    "PromotionDecision",
]
