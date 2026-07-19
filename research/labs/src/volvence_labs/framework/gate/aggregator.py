"""Gate aggregator: combines Two-Gate + SGM into a PromotionDecision.

Takes an ExperimentReport and produces a final decision:
- approve: both gates pass, e-value exceeds threshold
- hold: evidence is accumulating but not yet sufficient
- reject: capacity bound violated OR Goodhart detected OR margin negative
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from .two_gate import TwoGate, CapacityBound, ValidationMargin
from .sgm import SGMGate, EValue, HoeffdingBound


class GateDecision(enum.Enum):
    APPROVE = "approve"
    HOLD = "hold"
    REJECT = "reject"


@dataclass(frozen=True)
class PromotionDecision:
    """Final promotion decision with full audit trail."""
    decision: GateDecision
    probe_id: str
    from_level: str
    to_level: str
    capacity: Optional[CapacityBound]
    margin: Optional[ValidationMargin]
    sgm_summary: dict[str, Any]
    hoeffding_summary: dict[str, Any]
    reason: str
    evidence_run_ids: list[str]


class GateAggregator:
    """Aggregates gate checks into a PromotionDecision.

    Usage:
        agg = GateAggregator()
        decision = agg.evaluate(
            probe_id="refusal-direction-v1",
            units=report.units,
            metric_name="accuracy",
            n_parameters=0,  # read-only probe
        )
    """

    def __init__(
        self,
        *,
        vc_threshold: float = 100.0,
        margin_delta: float = 0.01,
        sgm_delta: float = 0.01,
        sgm_sigma: float = 1.0,
        sgm_alpha: float = 0.05,
        conservative_margin: float = 2.0,
        higher_is_better: bool = True,
    ):
        self.two_gate = TwoGate(
            vc_threshold=vc_threshold,
            margin_delta=margin_delta,
            conservative_margin=conservative_margin,
        )
        self.sgm = SGMGate(
            delta=sgm_delta,
            sigma=sgm_sigma,
            alpha=sgm_alpha,
        )
        self.hoeffding = HoeffdingBound(alpha=sgm_alpha)
        self.higher_is_better = higher_is_better

    def evaluate(
        self,
        probe_id: str,
        units: list,
        *,
        metric_name: str = "accuracy",
        n_parameters: int = 0,
        from_level: str = "shadow",
        to_level: str = "active",
        mode: str = "relative",
        absolute_threshold: float = 0.8,
    ) -> PromotionDecision:
        """Evaluate all gates and produce a PromotionDecision.

        Args:
            probe_id: probe being evaluated
            units: list of UnitReport objects from ExperimentReport
            metric_name: which metric to use for margin/SGM checks
            n_parameters: number of trainable parameters (0 for read-only probes)
            from_level: current wiring level
            to_level: target wiring level
            mode: "relative" (probe_on vs baseline) or "absolute" (probe_on vs threshold)
            absolute_threshold: for mode="absolute", the minimum acceptable value
        """
        # Extract metrics by cell
        probe_on_values: list[float] = []
        baseline_values: list[float] = []
        counterfactual_values: list[float] = []
        all_run_ids: list[str] = []
        n_samples = 0

        for unit in units:
            if not unit.ok:
                continue
            all_run_ids.append(unit.run_id)
            n_samples += 1
            metric_val = unit.metrics.get(metric_name)
            if metric_val is None:
                continue

            if unit.cell == "probe_on":
                probe_on_values.append(metric_val)
            elif unit.cell == "baseline":
                baseline_values.append(metric_val)
            elif unit.cell == "counterfactual":
                counterfactual_values.append(metric_val)

        # Gate 1: Capacity bound
        n_outputs = len(probe_on_values) + len(baseline_values)
        capacity = self.two_gate.check_capacity(n_parameters, n_samples, n_outputs)

        # Gate 2: Validation margin
        # In "absolute" mode, compare probe_on (or baseline) against a fixed threshold
        # rather than comparing probe_on vs baseline.
        if mode == "absolute":
            # For absolute mode: use baseline values as the "evidence" pool
            # and compare against absolute_threshold
            evidence_values = baseline_values if baseline_values else probe_on_values
            synthetic_baseline = [absolute_threshold] * len(evidence_values)
            margin = self.two_gate.check_margin(
                evidence_values,
                synthetic_baseline,
                metric_name=metric_name,
                higher_is_better=self.higher_is_better,
            )
            # SGM: compare each observation against the threshold
            self.sgm.reset()
            for v in evidence_values:
                self.sgm.update(v, absolute_threshold)
                self.hoeffding.update(v, absolute_threshold)
        else:
            margin = self.two_gate.check_margin(
                probe_on_values,
                baseline_values,
                metric_name=metric_name,
                higher_is_better=self.higher_is_better,
            )
            # SGM e-value (paired observations)
            self.sgm.reset()
            paired_count = min(len(probe_on_values), len(baseline_values))
            if paired_count > 0:
                self.sgm.update_batch(
                    probe_on_values[:paired_count],
                    baseline_values[:paired_count],
                )
                for p, b in zip(probe_on_values[:paired_count], baseline_values[:paired_count]):
                    self.hoeffding.update(p, b)

        sgm_summary = self.sgm.summary()
        hoeffding_summary = {
            "n": self.hoeffding.n,
            "mean_diff": self.hoeffding.mean_diff,
            "lower_bound": self.hoeffding.lower_bound(),
            "upper_bound": self.hoeffding.upper_bound(),
            "confidence_width": self.hoeffding.confidence_width(),
            "is_significant": self.hoeffding.is_significant(0.0),
        }

        # Goodhart check: if counterfactual cell performs BETTER than probe_on,
        # it suggests the metric is being gamed.
        goodhart_detected = False
        if counterfactual_values and probe_on_values:
            cf_mean = sum(counterfactual_values) / len(counterfactual_values)
            po_mean = sum(probe_on_values) / len(probe_on_values)
            if self.higher_is_better:
                goodhart_detected = cf_mean > po_mean * 1.1  # 10% better = suspicious
            else:
                goodhart_detected = cf_mean < po_mean * 0.9

        # Decision logic
        if not capacity.passed:
            decision = GateDecision.REJECT
            reason = f"capacity bound violated: VC={capacity.vc_dim_estimate:.1f} > {capacity.threshold}"
        elif goodhart_detected:
            decision = GateDecision.REJECT
            reason = "Goodhart detected: counterfactual outperforms probe_on"
        elif margin.margin < 0:
            decision = GateDecision.REJECT
            reason = f"negative margin: probe_on is worse than baseline ({margin.margin:.4f})"
        elif not margin.passed:
            decision = GateDecision.HOLD
            reason = f"margin insufficient: {margin.margin:.4f} < {margin.delta:.4f}"
        elif sgm_summary["decision"] == "approve":
            decision = GateDecision.APPROVE
            reason = (
                f"all gates pass: capacity OK, margin={margin.margin:.4f}>{margin.delta:.4f}, "
                f"e-value={sgm_summary['e_value']:.2f}>{sgm_summary['threshold']:.1f}"
            )
        else:
            decision = GateDecision.HOLD
            reason = (
                f"margin passes but e-value insufficient: "
                f"{sgm_summary['e_value']:.2f} < {sgm_summary['threshold']:.1f} "
                f"(need more seeds/runs)"
            )

        return PromotionDecision(
            decision=decision,
            probe_id=probe_id,
            from_level=from_level,
            to_level=to_level,
            capacity=capacity,
            margin=margin,
            sgm_summary=sgm_summary,
            hoeffding_summary=hoeffding_summary,
            reason=reason,
            evidence_run_ids=all_run_ids,
        )
