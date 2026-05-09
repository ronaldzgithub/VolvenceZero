"""DLaaS platform-tier typed dataclass surface.

Stable wire-format types shared across `dlaas-platform-*` wheels. This
package is the foundation of the platform tier: zero `vz-*` and zero
`lifeform-*` imports, pure stdlib.

Public exports cover the two surfaces touched by Slice 1:

* Runtime envelope: :class:`InteractionEnvelope`, :class:`InteractionType`,
  :class:`InteractionMode`, :class:`OutputAct`, :class:`FeedbackPayload`,
  :class:`OutputContract`.

Later slices add: TenantSpec / ShellSpec / AssetSpec / TemplateSpec /
TemplateVersionSpec / ContractSpec / FocusPersonSpec / IdentityLinkSpec /
HandoffTicketSpec / ExamQuestionSpec / ExamRunSpec / LaunchLicenseSpec.
"""

from __future__ import annotations

from dlaas_platform_contracts.envelope import (
    DEFAULT_PROTOCOL_VERSION,
    FeedbackPayload,
    InteractionEnvelope,
    InteractionMode,
    InteractionType,
    OutputAct,
    OutputContract,
)

__all__ = (
    "DEFAULT_PROTOCOL_VERSION",
    "FeedbackPayload",
    "InteractionEnvelope",
    "InteractionMode",
    "InteractionType",
    "OutputAct",
    "OutputContract",
)
