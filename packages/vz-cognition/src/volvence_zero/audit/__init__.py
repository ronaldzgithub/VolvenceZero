"""Audit owner package (architecture-uplift A5).

This package is the staged-gate-evidence owner described in
[`docs/specs/audit-owner.md`](../../../../../../docs/specs/audit-owner.md).

A5 落地范围（T11 packet）:
- :class:`AuditSnapshot` frozen schema + dependency types
- :class:`AuditModule` skeleton (default ``WiringLevel.DISABLED``;
  ``process()`` returns empty snapshot)
- Contract surface for ``evaluate_gate_reasons`` extension

OA-4 后续业务 packet 落地范围（不在 A5）:
- N8 风格 audit-agent tool loop
- risk score 计算
- 8 类 attack 验收
- transcript 记录
"""

from volvence_zero.audit.types import (
    AuditDetectedAttackClass,
    AuditSnapshot,
    AuditToolTrace,
)
from volvence_zero.audit.module import AuditModule

__all__ = [
    "AuditDetectedAttackClass",
    "AuditModule",
    "AuditSnapshot",
    "AuditToolTrace",
]
