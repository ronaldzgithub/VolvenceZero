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

from dlaas_platform_contracts.adoption import (
    AdoptionConfig,
    MemoryPolicySelection,
    OpsPolicySelection,
    ProtocolSelection,
    SubstrateSelection,
    ToolPolicySelection,
    TrainingPolicySelection,
    VerticalSelection,
)
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
from dlaas_platform_contracts.lifecycle import (
    InstanceLifecycleState,
    InstanceStatus,
    SleepRequest,
    WakeRequest,
)
from dlaas_platform_contracts.intake import (
    AssetIntakeDecision,
    AssetIntakeIntent,
    AssetIntakeRequest,
    AssetMediaKind,
)
from dlaas_platform_contracts.governance import (
    ArtifactKind,
    ArtifactRecord,
    AuditEvent,
    BillingEvent,
    ConsentRecord,
    DataExportJob,
    DataJobStatus,
    DeletionJob,
    EvalGateDecision,
    EvalRun,
    EvalRunStatus,
    EventStreamRecord,
    EvidenceTrace,
    PolicySnapshot,
    PromotionDecision,
    QuotaSnapshot,
    UsageRecord,
    WebhookSubscription,
)
from dlaas_platform_contracts.observability import (
    ExplainTrace,
    LifeBlueprint,
    ReadoutBundle,
    ReadoutView,
    SnapshotExportRequest,
)
from dlaas_platform_contracts.resources import (
    AssetSpec,
    CitationPolicy,
    ContractSpec,
    ContractStatus,
    CoveragePolicy,
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
from dlaas_platform_contracts.training import (
    ProtocolSubmission,
    ProtocolSubmissionSourceType,
    TrainingJob,
    TrainingJobStatus,
    TrainingJobType,
)

__all__ = (
    "AdoptionConfig",
    "ArtifactKind",
    "ArtifactRecord",
    "AssetIntakeDecision",
    "AssetIntakeIntent",
    "AssetIntakeRequest",
    "AssetMediaKind",
    "AssetSpec",
    "AuditEvent",
    "AudienceProfileSpec",
    "BillingEvent",
    "CitationPolicy",
    "CommandName",
    "ConsentRecord",
    "ContractSpec",
    "ContractStatus",
    "CoveragePolicy",
    "DataExportJob",
    "DataJobStatus",
    "DeletionJob",
    "EvalGateDecision",
    "EvalRun",
    "EvalRunStatus",
    "DEFAULT_PROTOCOL_VERSION",
    "ExamQuestionSpec",
    "ExamRunSpec",
    "ExamRunStatus",
    "ExamSubmissionScore",
    "EventStreamRecord",
    "EvidenceTrace",
    "ExplainTrace",
    "FeedbackPayload",
    "FeedbackValence",
    "FocusPersonSpec",
    "HandoffStatus",
    "HandoffTicketSpec",
    "IdentityLinkSpec",
    "InstanceLifecycleState",
    "InstanceStatus",
    "InteractionEnvelope",
    "InteractionMode",
    "InteractionType",
    "LaunchLicenseSpec",
    "LifeBlueprint",
    "MemoryPolicySelection",
    "ObservationType",
    "OpsPolicySelection",
    "OutputAct",
    "OutputContract",
    "PolicySnapshot",
    "ProtocolSelection",
    "ProtocolSubmission",
    "ProtocolSubmissionSourceType",
    "PromotionDecision",
    "QuotaSnapshot",
    "ReadinessReport",
    "ReadoutBundle",
    "ReadoutView",
    "RubricEntry",
    "ShellKind",
    "ShellSpec",
    "SleepRequest",
    "SnapshotExportRequest",
    "SubstrateSelection",
    "TemplateActivationStatus",
    "TemplateAssetLinkSpec",
    "TemplateSpec",
    "TemplateStatus",
    "TemplateVersionSpec",
    "TenantSpec",
    "ToolPolicySelection",
    "TrainingJob",
    "TrainingJobStatus",
    "TrainingJobType",
    "TrainingPolicySelection",
    "UsageRecord",
    "VerticalSelection",
    "WakeRequest",
    "WebhookSubscription",
    "feedback_valence_to_outcome_kind",
)
