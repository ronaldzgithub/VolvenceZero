"""Foundry-facing Venture Brain vertical and product facade."""

from __future__ import annotations

from typing import Any

from lifeform_domain_venture.venture_brain import (
    EmptyVentureAdviceProvider,
    VentureAdviceProvider,
    VentureBrainConflictError,
    VentureBrainController,
    VentureBrainError,
    VentureBrainLineageError,
    VentureBrainMemoryContractError,
    VentureBrainReadOnlyError,
    VentureBrainSettlementPendingError,
)
from lifeform_domain_venture.foundry_public_contract import (
    FOUNDRY_PUBLIC_CONTRACT_ID_PREFIX,
    FOUNDRY_PUBLIC_CONTRACT_SCHEMA_VERSION,
    build_foundry_public_contract,
    load_foundry_public_contract_fixture,
    load_foundry_public_contract_json_schema,
    validate_foundry_public_contract,
)
from lifeform_domain_venture.venture_brain_contracts import (
    ADVICE_SCHEMA_VERSION,
    CONTEXT_PACK_SCHEMA_VERSION,
    CONTEXT_REQUEST_SCHEMA_VERSION,
    OUTCOME_RECEIPT_SCHEMA_VERSION,
    OUTCOME_REPORT_SCHEMA_VERSION,
    VentureAdviceCandidate,
    VentureAdviceKind,
    VentureAdviceSnapshot,
    VentureCommercialOutcome,
    VentureConstraint,
    VentureConstraintKind,
    VentureContextPackSnapshot,
    VentureContextRequest,
    VentureCostBreakdown,
    VentureCustomerResult,
    VentureDecisionKind,
    VentureDecisionPoint,
    VentureEstimateRange,
    VentureEvidenceClass,
    VentureEvidenceRef,
    VentureEvidenceRole,
    VentureFact,
    VentureFactKind,
    VentureOutcomeKind,
    VentureOutcomeReceipt,
    VentureOutcomeReport,
    VentureOutcomeRoute,
    VentureOutcomeVerdict,
    VentureRecalledExperience,
    VentureResourceWindow,
    VentureReversibility,
    VentureRiskLevel,
    VentureSettlementState,
    VentureUncertainty,
)
from lifeform_domain_venture.venture_pack import build_venture_package


def build_venture_lifeform(
    *,
    config: object | None = None,
    substrate_runtime: Any = None,
    identity_provider: Any = None,
) -> Any:
    """Build a Lifeform with reviewed Venture Brain domain priors.

    v1 adds no bespoke controller, vitals, or substrate update. All online
    adaptation remains in the existing Memory, PE/credit, semantic, regime,
    and temporal owners.
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
    ).with_domain_experience((build_venture_package(),))
    return Lifeform(
        base_config,
        substrate_runtime=substrate_runtime,
        identity_provider=identity_provider,
    )


__all__ = (
    "ADVICE_SCHEMA_VERSION",
    "CONTEXT_PACK_SCHEMA_VERSION",
    "CONTEXT_REQUEST_SCHEMA_VERSION",
    "EmptyVentureAdviceProvider",
    "FOUNDRY_PUBLIC_CONTRACT_ID_PREFIX",
    "FOUNDRY_PUBLIC_CONTRACT_SCHEMA_VERSION",
    "OUTCOME_RECEIPT_SCHEMA_VERSION",
    "OUTCOME_REPORT_SCHEMA_VERSION",
    "VentureAdviceCandidate",
    "VentureAdviceKind",
    "VentureAdviceProvider",
    "VentureAdviceSnapshot",
    "VentureBrainConflictError",
    "VentureBrainController",
    "VentureBrainError",
    "VentureBrainLineageError",
    "VentureBrainMemoryContractError",
    "VentureBrainReadOnlyError",
    "VentureBrainSettlementPendingError",
    "VentureCommercialOutcome",
    "VentureConstraint",
    "VentureConstraintKind",
    "VentureContextPackSnapshot",
    "VentureContextRequest",
    "VentureCostBreakdown",
    "VentureCustomerResult",
    "VentureDecisionKind",
    "VentureDecisionPoint",
    "VentureEstimateRange",
    "VentureEvidenceClass",
    "VentureEvidenceRef",
    "VentureEvidenceRole",
    "VentureFact",
    "VentureFactKind",
    "VentureOutcomeKind",
    "VentureOutcomeReceipt",
    "VentureOutcomeReport",
    "VentureOutcomeRoute",
    "VentureOutcomeVerdict",
    "VentureRecalledExperience",
    "VentureResourceWindow",
    "VentureReversibility",
    "VentureRiskLevel",
    "VentureSettlementState",
    "VentureUncertainty",
    "build_foundry_public_contract",
    "build_venture_lifeform",
    "build_venture_package",
    "load_foundry_public_contract_fixture",
    "load_foundry_public_contract_json_schema",
    "validate_foundry_public_contract",
)
