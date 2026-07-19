"""Two-Gate mechanism: VC capacity bound + validation margin.

Gate 1 (Capacity Bound): Ensures the probe's hypothesis class has bounded
complexity. Based on Vapnik's VC dimension theory — if the effective VC
dimension of the probe's output space exceeds a threshold, the probe cannot
be promoted (too many degrees of freedom → overfitting risk).

Gate 2 (Validation Margin): Ensures the probe's readouts on held-out seeds
exceed a minimum margin above the baseline cell. This is a simple δ-test:
    mean(probe_on) - mean(baseline) > δ

Both gates must pass for promotion to be approved.

References:
- Vapnik (1998) Statistical Learning Theory
- Bartlett & Mendelson (2002) Rademacher complexity bounds
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CapacityBound:
    """Result of the VC capacity check.

    vc_dim_estimate: estimated effective VC dimension of the probe's output
    threshold: maximum allowed VC dimension for promotion
    passed: whether vc_dim_estimate <= threshold
    """
    vc_dim_estimate: float
    threshold: float
    passed: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class ValidationMargin:
    """Result of the validation margin check.

    margin: mean(probe_on) - mean(baseline) on held-out seeds
    delta: minimum required margin
    passed: whether margin >= delta
    """
    margin: float
    delta: float
    passed: bool
    n_probe_on: int
    n_baseline: int
    metric_name: str
    details: dict[str, Any]


class TwoGate:
    """Two-Gate mechanism for promotion decisions.

    Usage:
        gate = TwoGate(vc_threshold=50.0, margin_delta=0.01)
        cap = gate.check_capacity(report)
        margin = gate.check_margin(report, metric="accuracy")
        both_pass = cap.passed and margin.passed
    """

    def __init__(
        self,
        *,
        vc_threshold: float = 100.0,
        margin_delta: float = 0.01,
        conservative_margin: float = 2.0,  # safety factor
    ):
        self.vc_threshold = vc_threshold
        self.margin_delta = margin_delta
        self.conservative_margin = conservative_margin

    def check_capacity(
        self,
        n_parameters: int,
        n_samples: int,
        n_outputs: int,
    ) -> CapacityBound:
        """Estimate VC dimension and check against threshold.

        Uses the heuristic: VC_dim ≈ min(n_parameters, n_outputs * log(n_samples))
        This is a conservative upper bound for neural-network-like hypothesis classes.

        For probes that are purely read-only (no trainable parameters), VC_dim = 0.
        """
        if n_parameters == 0:
            # Read-only probe: no hypothesis class complexity
            vc_estimate = 0.0
        else:
            # Bartlett-style bound: VC ≈ O(params / margin^2)
            # Simplified: VC ≈ min(params, outputs * log(samples))
            vc_estimate = min(
                float(n_parameters),
                float(n_outputs) * math.log(max(n_samples, 2)),
            )

        passed = vc_estimate <= self.vc_threshold
        return CapacityBound(
            vc_dim_estimate=vc_estimate,
            threshold=self.vc_threshold,
            passed=passed,
            details={
                "n_parameters": n_parameters,
                "n_samples": n_samples,
                "n_outputs": n_outputs,
            },
        )

    def check_margin(
        self,
        probe_on_values: list[float],
        baseline_values: list[float],
        *,
        metric_name: str = "primary",
        higher_is_better: bool = True,
    ) -> ValidationMargin:
        """Check if probe_on exceeds baseline by at least delta.

        Uses a conservative margin: actual_delta = margin_delta * conservative_margin.
        """
        if not probe_on_values or not baseline_values:
            return ValidationMargin(
                margin=0.0,
                delta=self.margin_delta * self.conservative_margin,
                passed=False,
                n_probe_on=len(probe_on_values),
                n_baseline=len(baseline_values),
                metric_name=metric_name,
                details={"error": "insufficient data"},
            )

        probe_mean = sum(probe_on_values) / len(probe_on_values)
        base_mean = sum(baseline_values) / len(baseline_values)

        if higher_is_better:
            margin = probe_mean - base_mean
        else:
            margin = base_mean - probe_mean  # lower is better → flip

        effective_delta = self.margin_delta * self.conservative_margin
        passed = margin >= effective_delta

        # Also compute standard error for informational purposes
        if len(probe_on_values) > 1:
            probe_var = sum((x - probe_mean) ** 2 for x in probe_on_values) / (len(probe_on_values) - 1)
            probe_se = math.sqrt(probe_var / len(probe_on_values))
        else:
            probe_se = 0.0

        if len(baseline_values) > 1:
            base_var = sum((x - base_mean) ** 2 for x in baseline_values) / (len(baseline_values) - 1)
            base_se = math.sqrt(base_var / len(baseline_values))
        else:
            base_se = 0.0

        return ValidationMargin(
            margin=margin,
            delta=effective_delta,
            passed=passed,
            n_probe_on=len(probe_on_values),
            n_baseline=len(baseline_values),
            metric_name=metric_name,
            details={
                "probe_mean": probe_mean,
                "base_mean": base_mean,
                "probe_se": probe_se,
                "base_se": base_se,
                "effective_delta": effective_delta,
            },
        )
