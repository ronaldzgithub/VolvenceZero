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

from volvence_zero.audit.types import AuditSnapshot
from volvence_zero.runtime.kernel import (
    RuntimeModule,
    Snapshot,
    WiringLevel,
)

__all__ = ["AuditModule"]


_EMPTY_AUDIT_SNAPSHOT = AuditSnapshot(
    audit_id="",
    timestamp_ms=0,
    proposal_id=None,
    risk_score=0.0,
    transcript=(),
    tool_traces=(),
    detected_attack_classes=(),
    threshold_decision="pass",
    description="audit_owner skeleton (A5 / T11 packet); OA-4 will populate.",
)


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
        """Publish an empty audit snapshot.

        fail-loudly: missing declared dependency surfaces through kernel
        UpstreamView (DependencyGuard).
        """
        evaluation_snapshot = upstream["evaluation"]
        credit_snapshot = upstream["credit"]
        # Skeleton: deferred to OA-4 packet
        del evaluation_snapshot
        del credit_snapshot
        return self.publish(_EMPTY_AUDIT_SNAPSHOT)
