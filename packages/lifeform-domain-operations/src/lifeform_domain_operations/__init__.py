"""AutoCompany-facing Operations Brain vertical and product facade."""

from __future__ import annotations

from typing import Any

from lifeform_domain_operations.operations_brain import (
    EmptyOperationsAdviceProvider,
    OperationsAdviceProvider,
    OperationsBrainConflictError,
    OperationsBrainController,
    OperationsBrainError,
    OperationsBrainLineageError,
    OperationsBrainMemoryContractError,
    OperationsBrainReadOnlyError,
    OperationsBrainSettlementPendingError,
)
from lifeform_domain_operations.operations_brain_contracts import (
    ADVICE_SCHEMA_VERSION,
    CONTEXT_PACK_SCHEMA_VERSION,
    CONTEXT_REQUEST_SCHEMA_VERSION,
    OUTCOME_RECEIPT_SCHEMA_VERSION,
    OUTCOME_REPORT_SCHEMA_VERSION,
    OperationsAdviceCandidate,
    OperationsAdviceKind,
    OperationsAdviceSnapshot,
    OperationsExecutionOutcome,
    OperationsConstraint,
    OperationsConstraintKind,
    OperationsContextPackSnapshot,
    OperationsContextRequest,
    OperationsCostBreakdown,
    OperationsObjectiveResult,
    OperationsDecisionKind,
    OperationsDecisionPoint,
    OperationsEstimateRange,
    OperationsEvidenceClass,
    OperationsEvidenceRef,
    OperationsEvidenceRole,
    OperationsFact,
    OperationsFactKind,
    OperationsMetricObservation,
    OperationsOutcomeKind,
    OperationsOutcomeReceipt,
    OperationsOutcomeReport,
    OperationsOutcomeRoute,
    OperationsOutcomeVerdict,
    OperationsRecalledExperience,
    OperationsOperatingWindow,
    OperationsReversibility,
    OperationsRiskLevel,
    OperationsSettlementState,
    OperationsUncertainty,
)
from lifeform_domain_operations.operations_pack import build_operations_package


def build_operations_lifeform(
    *,
    config: object | None = None,
    substrate_runtime: Any = None,
    identity_provider: Any = None,
) -> Any:
    """Build a Lifeform with reviewed Operations Brain domain priors.

    v1 adds no second cognitive owner, bespoke vitals, or substrate update.
    All online adaptation remains in the existing Memory, PE/credit, semantic,
    regime, and temporal owners.
    """

    from dataclasses import replace

    from lifeform_core import Lifeform, LifeformConfig

    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": False}
    if substrate_runtime is not None:
        brain_overrides["substrate_mode"] = "injected"
    base_config = replace(
        base_config,
        brain_config=replace(base_config.brain_config, **brain_overrides),
    ).with_domain_experience((build_operations_package(),))
    return Lifeform(
        base_config,
        substrate_runtime=substrate_runtime,
        identity_provider=identity_provider,
    )


__all__ = (
    "ADVICE_SCHEMA_VERSION",
    "CONTEXT_PACK_SCHEMA_VERSION",
    "CONTEXT_REQUEST_SCHEMA_VERSION",
    "EmptyOperationsAdviceProvider",
    "OUTCOME_RECEIPT_SCHEMA_VERSION",
    "OUTCOME_REPORT_SCHEMA_VERSION",
    "OperationsAdviceCandidate",
    "OperationsAdviceKind",
    "OperationsAdviceProvider",
    "OperationsAdviceSnapshot",
    "OperationsBrainConflictError",
    "OperationsBrainController",
    "OperationsBrainError",
    "OperationsBrainLineageError",
    "OperationsBrainMemoryContractError",
    "OperationsBrainReadOnlyError",
    "OperationsBrainSettlementPendingError",
    "OperationsExecutionOutcome",
    "OperationsConstraint",
    "OperationsConstraintKind",
    "OperationsContextPackSnapshot",
    "OperationsContextRequest",
    "OperationsCostBreakdown",
    "OperationsObjectiveResult",
    "OperationsDecisionKind",
    "OperationsDecisionPoint",
    "OperationsEstimateRange",
    "OperationsEvidenceClass",
    "OperationsEvidenceRef",
    "OperationsEvidenceRole",
    "OperationsFact",
    "OperationsFactKind",
    "OperationsMetricObservation",
    "OperationsOutcomeKind",
    "OperationsOutcomeReceipt",
    "OperationsOutcomeReport",
    "OperationsOutcomeRoute",
    "OperationsOutcomeVerdict",
    "OperationsRecalledExperience",
    "OperationsOperatingWindow",
    "OperationsReversibility",
    "OperationsRiskLevel",
    "OperationsSettlementState",
    "OperationsUncertainty",
    "build_operations_lifeform",
    "build_operations_package",
)
