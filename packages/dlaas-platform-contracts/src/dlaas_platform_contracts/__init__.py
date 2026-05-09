"""DLaaS platform-tier typed dataclass surface.

Stable wire-format types shared across `dlaas-platform-*` wheels. This
package is the foundation of the platform tier: zero `vz-*` (besides
the typed taxonomy in ``vz-contracts``) and zero `lifeform-*` imports.

Public exports:

* Runtime envelope (Slice 1): :class:`InteractionEnvelope`,
  :class:`InteractionType`, :class:`InteractionMode`,
  :class:`OutputAct`, :class:`FeedbackPayload`,
  :class:`OutputContract`.
* Dispatch vocabulary (Slice 2): :class:`FeedbackValence`,
  :class:`ObservationType`, :class:`CommandName` +
  :func:`feedback_valence_to_outcome_kind`.
* Control-plane resources (Slice 3 / 4): :class:`TenantSpec`,
  :class:`ShellSpec` / :class:`ShellKind`, :class:`AssetSpec`,
  :class:`TemplateSpec` / :class:`TemplateStatus` /
  :class:`TemplateActivationStatus` /
  :class:`TemplateVersionSpec` /
  :class:`TemplateAssetLinkSpec` /
  :class:`ReadinessReport`,
  :class:`ContractSpec` / :class:`ContractStatus`,
  :class:`FocusPersonSpec`, :class:`IdentityLinkSpec`,
  :class:`HandoffTicketSpec` / :class:`HandoffStatus`.
"""

from __future__ import annotations

from dlaas_platform_contracts.dispatch_vocab import (
    CommandName,
    FeedbackValence,
    ObservationType,
    feedback_valence_to_outcome_kind,
)
from dlaas_platform_contracts.envelope import (
    DEFAULT_PROTOCOL_VERSION,
    FeedbackPayload,
    InteractionEnvelope,
    InteractionMode,
    InteractionType,
    OutputAct,
    OutputContract,
)
from dlaas_platform_contracts.eval import (
    AudienceProfileSpec,
    ExamQuestionSpec,
    ExamRunSpec,
    ExamRunStatus,
    ExamSubmissionScore,
    LaunchLicenseSpec,
    RubricEntry,
)
from dlaas_platform_contracts.resources import (
    AssetSpec,
    ContractSpec,
    ContractStatus,
    FocusPersonSpec,
    HandoffStatus,
    HandoffTicketSpec,
    IdentityLinkSpec,
    ReadinessReport,
    ShellKind,
    ShellSpec,
    TemplateActivationStatus,
    TemplateAssetLinkSpec,
    TemplateSpec,
    TemplateStatus,
    TemplateVersionSpec,
    TenantSpec,
)

__all__ = (
    "AssetSpec",
    "AudienceProfileSpec",
    "CommandName",
    "ContractSpec",
    "ContractStatus",
    "DEFAULT_PROTOCOL_VERSION",
    "ExamQuestionSpec",
    "ExamRunSpec",
    "ExamRunStatus",
    "ExamSubmissionScore",
    "FeedbackPayload",
    "FeedbackValence",
    "FocusPersonSpec",
    "HandoffStatus",
    "HandoffTicketSpec",
    "IdentityLinkSpec",
    "InteractionEnvelope",
    "InteractionMode",
    "InteractionType",
    "LaunchLicenseSpec",
    "ObservationType",
    "OutputAct",
    "OutputContract",
    "ReadinessReport",
    "RubricEntry",
    "ShellKind",
    "ShellSpec",
    "TemplateActivationStatus",
    "TemplateAssetLinkSpec",
    "TemplateSpec",
    "TemplateStatus",
    "TemplateVersionSpec",
    "TenantSpec",
    "feedback_valence_to_outcome_kind",
)
