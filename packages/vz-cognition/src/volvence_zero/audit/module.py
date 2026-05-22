"""AuditModule skeleton (architecture-uplift A5 / T11).

The actual N8 audit-agent tool loop / risk-score computation / 8-attack
acceptance test live in the OA-4 业务 packet (out of scope for A5). This
module defines the publisher contract so:

- ModificationGate can declare an opt-in dependency on ``audit`` once T11
  接口扩展 is in place
- DATA_CONTRACT §6 has a real owner to register against
- WiringLevel SHADOW / ACTIVE switching can be plumbed via FinalRolloutConfig
  with the standard mechanism

Default wiring is ``WiringLevel.SHADOW`` per
[`docs/specs/audit-owner.md`](../../../../../../docs/specs/audit-owner.md)
§WiringLevel — empty AuditSnapshot is published but does not flow into
正式 upstream.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from volvence_zero.audit.types import AuditSnapshot, AuditToolTrace
from volvence_zero.runtime.kernel import (
    RuntimeModule,
    Snapshot,
    WiringLevel,
)

__all__ = ["AuditModule"]


class AuditModule(RuntimeModule[AuditSnapshot]):
    """Audit owner: publish AuditSnapshot for ModificationGate evidence.

    A5 阶段 implementation: empty snapshot publisher. Real audit-agent tool
    loop is OA-4 business packet scope.
    """

    slot_name = "audit"
    owner = "AuditModule"
    value_type = AuditSnapshot
    dependencies = ("evaluation", "credit")
    default_wiring_level = WiringLevel.SHADOW

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[AuditSnapshot]:
        """Publish read-only audit evidence from public evaluation/credit snapshots.

        fail-loudly: missing declared dependency surfaces through kernel
        UpstreamView (DependencyGuard).
        """
        evaluation_snapshot = upstream["evaluation"]
        credit_snapshot = upstream["credit"]
        from volvence_zero.credit.gate import CreditSnapshot
        from volvence_zero.evaluation.types import EvaluationSnapshot

        evaluation_value = (
            evaluation_snapshot.value
            if isinstance(evaluation_snapshot.value, EvaluationSnapshot)
            else None
        )
        credit_value = (
            credit_snapshot.value
            if isinstance(credit_snapshot.value, CreditSnapshot)
            else None
        )
        alert_risk = 0.0
        persona_drift = 0.0
        if evaluation_value is not None:
            for alert in evaluation_value.structured_alerts:
                if alert.severity == "CRITICAL":
                    alert_risk = max(alert_risk, 1.0)
                elif alert.severity == "HIGH":
                    alert_risk = max(alert_risk, 0.75)
            metric_values = {
                score.metric_name: score.value
                for score in evaluation_value.turn_scores + evaluation_value.session_scores
            }
            persona_drift = max(0.0, min(1.0, metric_values.get("persona_geometry_drift", 0.0)))
        control_effort = (
            credit_value.least_control_readout.control_effort
            if credit_value is not None and credit_value.least_control_readout is not None
            else 0.0
        )
        risk_score = max(alert_risk, persona_drift, control_effort)
        if risk_score >= 0.75:
            threshold_decision = "hard-block"
        elif risk_score >= 0.35:
            threshold_decision = "soft-warn"
        else:
            threshold_decision = "pass"
        tool_traces = (
            AuditToolTrace(
                tool_name="benchmark_runner",
                tool_args_summary="evaluation_snapshot",
                tool_output_summary=f"alert_risk={alert_risk:.3f}",
                duration_ms=0,
                succeeded=True,
            ),
            AuditToolTrace(
                tool_name="persona_drift_probe",
                tool_args_summary="evaluation.persona_geometry_drift",
                tool_output_summary=f"persona_drift={persona_drift:.3f}",
                duration_ms=0,
                succeeded=True,
            ),
            AuditToolTrace(
                tool_name="least_control_probe",
                tool_args_summary="credit.least_control_readout",
                tool_output_summary=f"control_effort={control_effort:.3f}",
                duration_ms=0,
                succeeded=True,
            ),
        )
        value = AuditSnapshot(
            audit_id=f"audit:{self._version + 1}",
            timestamp_ms=evaluation_snapshot.timestamp_ms,
            proposal_id=None,
            risk_score=risk_score,
            transcript=(
                "audit owner consumed evaluation + credit public readouts",
                f"threshold_decision={threshold_decision}",
            ),
            tool_traces=tool_traces,
            detected_attack_classes=(),
            threshold_decision=threshold_decision,
            description=(
                f"Audit readout risk={risk_score:.3f} "
                f"alert={alert_risk:.3f} persona={persona_drift:.3f} "
                f"control={control_effort:.3f}."
            ),
        )
        return self.publish(value)
