"""Read-only evidence that multiple learned action families stay distinct."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from json import JSONDecodeError
from typing import Protocol

from volvence_zero.application import CaseActionAbstractionPromotion


class BehaviorFamilyPromptKind(str, Enum):
    """Exact protocol lanes admitted to the real-provider evidence run."""

    ACTION_ABSTRACTION = "action_abstraction"
    ACTION_GENERALIZATION_AUDIT = "action_generalization_audit"
    ACTION_APPLICABILITY = "action_applicability"
    EXCLUDED_OTHER = "excluded_other"


class _TextGenerationProvider(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = ...,
        temperature: float = ...,
    ) -> str: ...


@dataclass(frozen=True)
class BehaviorFamilyProviderTrace:
    call_index: int
    provider_id: str
    prompt_kind: BehaviorFamilyPromptKind
    delegated_to_provider: bool
    prompt_sha256: str
    response_sha256: str
    prompt_text: str
    response_text: str
    response_is_json_object: bool
    response_is_non_empty_json_object: bool


class ActionEvidenceOnlyTextProvider:
    """Audited provider boundary for the real-model evidence lane.

    ``LLMSemanticProposalRuntime`` shares one text provider across semantic
    owners.  This wrapper delegates only the three exact action-evidence
    protocols to the real model and returns the explicit no-proposal object
    for every other protocol.  The dispatch classifies protocol preambles,
    never user or chapter language.
    """

    _ACTION_ABSTRACTION_PREAMBLE = (
        "You are the background-slow semantic decoder for a "
        "CaseMemory owner."
    )
    _ACTION_APPLICABILITY_PREAMBLE = (
        "You are the turn-time semantic applicability evaluator "
        "for a CaseMemory owner."
    )
    _ACTION_GENERALIZATION_AUDIT_PREAMBLE = (
        "You are the independent second-pass semantic generalization "
        "auditor for a"
    )

    def __init__(
        self,
        *,
        provider: _TextGenerationProvider,
        provider_id: str,
    ) -> None:
        if not provider_id.strip():
            raise ValueError("provider_id must be non-empty")
        self._provider = provider
        self._provider_id = provider_id
        self._traces: list[BehaviorFamilyProviderTrace] = []

    @property
    def traces(self) -> tuple[BehaviorFamilyProviderTrace, ...]:
        return tuple(self._traces)

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        if prompt.startswith(self._ACTION_ABSTRACTION_PREAMBLE):
            prompt_kind = BehaviorFamilyPromptKind.ACTION_ABSTRACTION
        elif prompt.startswith(
            self._ACTION_GENERALIZATION_AUDIT_PREAMBLE
        ):
            prompt_kind = (
                BehaviorFamilyPromptKind.ACTION_GENERALIZATION_AUDIT
            )
        elif prompt.startswith(self._ACTION_APPLICABILITY_PREAMBLE):
            prompt_kind = BehaviorFamilyPromptKind.ACTION_APPLICABILITY
        else:
            prompt_kind = BehaviorFamilyPromptKind.EXCLUDED_OTHER

        delegated = prompt_kind is not BehaviorFamilyPromptKind.EXCLUDED_OTHER
        response = (
            self._provider.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if delegated
            else "{}"
        )
        cleaned_response = response.strip()
        if (
            cleaned_response.startswith("```")
            and cleaned_response.endswith("```")
        ):
            lines = cleaned_response.splitlines()
            cleaned_response = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(cleaned_response)
        except JSONDecodeError:
            payload = None
        is_json_object = isinstance(payload, dict)
        self._traces.append(
            BehaviorFamilyProviderTrace(
                call_index=len(self._traces),
                provider_id=self._provider_id,
                prompt_kind=prompt_kind,
                delegated_to_provider=delegated,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                response_sha256=hashlib.sha256(
                    response.encode()
                ).hexdigest(),
                prompt_text=prompt if delegated else "",
                response_text=response if delegated else "",
                response_is_json_object=is_json_object,
                response_is_non_empty_json_object=(
                    is_json_object and bool(payload)
                ),
            )
        )
        return response


@dataclass(frozen=True)
class BehaviorFamilyRoutingObservation:
    case_id: str
    expected_schema_id: str
    selected_schema_id: str | None
    source_state_digest_verified: bool
    outcome_feedback_submitted: bool
    evaluation_feedback_submitted: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("case_id", self.case_id),
            ("expected_schema_id", self.expected_schema_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if (
            self.selected_schema_id is not None
            and not self.selected_schema_id.strip()
        ):
            raise ValueError(
                "selected_schema_id must be non-empty when present"
            )


@dataclass(frozen=True)
class BehaviorFamilyPortfolioReport:
    suite_id: str
    expected_schema_ids: tuple[str, ...]
    promoted_schema_ids: tuple[str, ...]
    promoted_family_ids: tuple[str, ...]
    promotion_count: int
    generalization_audited_promotion_count: int
    distinct_family_count: int
    pending_family_count: int
    routing_case_count: int
    correct_routing_count: int
    gate_statuses: tuple[tuple[str, str], ...]
    multi_family_owner_supported: bool
    claim_status: str
    description: str


@dataclass(frozen=True)
class RealProviderBehaviorEvidenceReport:
    provider_id: str
    delegated_abstraction_call_count: int
    delegated_generalization_audit_call_count: int
    delegated_applicability_call_count: int
    excluded_other_call_count: int
    non_empty_structured_response_count: int
    gate_statuses: tuple[tuple[str, str], ...]
    real_provider_supported: bool
    claim_status: str
    description: str


def evaluate_behavior_family_portfolio(
    *,
    suite_id: str,
    expected_schema_ids: tuple[str, ...],
    promotions: tuple[CaseActionAbstractionPromotion, ...],
    routing_observations: tuple[
        BehaviorFamilyRoutingObservation, ...
    ],
    pending_family_ids: tuple[str, ...],
) -> BehaviorFamilyPortfolioReport:
    """Verify owner separation and held-out routing without learning feedback."""

    if not suite_id.strip():
        raise ValueError("suite_id must be non-empty")
    if (
        len(expected_schema_ids) < 2
        or any(not item.strip() for item in expected_schema_ids)
        or len(set(expected_schema_ids)) != len(expected_schema_ids)
    ):
        raise ValueError(
            "expected_schema_ids must contain at least two unique values"
        )
    if (
        any(not family_id.strip() for family_id in pending_family_ids)
        or len(set(pending_family_ids)) != len(pending_family_ids)
    ):
        raise ValueError(
            "pending_family_ids must contain unique non-empty values"
        )
    promoted_schema_ids = tuple(
        promotion.schema_id for promotion in promotions
    )
    if len(set(promoted_schema_ids)) != len(promoted_schema_ids):
        raise ValueError("portfolio contains duplicate promoted schema ids")
    promoted_family_ids = tuple(
        promotion.action_family_id for promotion in promotions
    )
    source_outcome_ids = tuple(
        outcome_id
        for promotion in promotions
        for outcome_id in promotion.source_outcome_ids
    )
    source_outcomes_disjoint = (
        len(source_outcome_ids) == len(set(source_outcome_ids))
    )
    generalization_audited_promotion_count = sum(
        promotion.generalization_audit_passed
        for promotion in promotions
    )
    promotions_generalization_audited = (
        bool(promotions)
        and generalization_audited_promotion_count == len(promotions)
    )
    route_case_ids = tuple(
        observation.case_id for observation in routing_observations
    )
    if len(route_case_ids) != len(set(route_case_ids)):
        raise ValueError("portfolio contains duplicate routing case ids")
    route_expectations = tuple(
        observation.expected_schema_id
        for observation in routing_observations
    )
    exact_promotion_coverage = (
        set(promoted_schema_ids) == set(expected_schema_ids)
        and len(promoted_schema_ids) == len(expected_schema_ids)
    )
    distinct_family_ownership = (
        len(set(promoted_family_ids)) == len(expected_schema_ids)
    )
    pending_families_isolated = not (
        set(pending_family_ids) & set(promoted_family_ids)
    )
    exact_routing_coverage = (
        set(route_expectations) == set(expected_schema_ids)
        and len(route_expectations) == len(expected_schema_ids)
    )
    correct_routing_count = sum(
        observation.selected_schema_id
        == observation.expected_schema_id
        for observation in routing_observations
    )
    routing_separation = (
        exact_routing_coverage
        and correct_routing_count == len(expected_schema_ids)
    )
    source_integrity = all(
        observation.source_state_digest_verified
        for observation in routing_observations
    )
    no_feedback = all(
        not observation.outcome_feedback_submitted
        and not observation.evaluation_feedback_submitted
        for observation in routing_observations
    )
    gates = (
        (
            "exact_promotion_coverage",
            "pass" if exact_promotion_coverage else "fail",
        ),
        (
            "distinct_family_ownership",
            "pass" if distinct_family_ownership else "fail",
        ),
        (
            "source_outcomes_disjoint",
            "pass" if source_outcomes_disjoint else "fail",
        ),
        (
            "promotions_generalization_audited",
            "pass" if promotions_generalization_audited else "fail",
        ),
        (
            "pending_families_isolated",
            "pass" if pending_families_isolated else "fail",
        ),
        (
            "held_out_routing_separation",
            "pass" if routing_separation else "fail",
        ),
        (
            "source_integrity",
            "pass" if source_integrity else "fail",
        ),
        (
            "no_evaluation_feedback",
            "pass" if no_feedback else "fail",
        ),
    )
    supported = all(status == "pass" for _gate, status in gates)
    return BehaviorFamilyPortfolioReport(
        suite_id=suite_id,
        expected_schema_ids=expected_schema_ids,
        promoted_schema_ids=promoted_schema_ids,
        promoted_family_ids=promoted_family_ids,
        promotion_count=len(promotions),
        generalization_audited_promotion_count=(
            generalization_audited_promotion_count
        ),
        distinct_family_count=len(set(promoted_family_ids)),
        pending_family_count=len(pending_family_ids),
        routing_case_count=len(routing_observations),
        correct_routing_count=correct_routing_count,
        gate_statuses=gates,
        multi_family_owner_supported=supported,
        claim_status=(
            "multi-family-owner-diagnostic-pass"
            if supported
            else "diagnostic-fail"
        ),
        description=(
            "Read-only portfolio proof for multiple owner-promoted action "
            "families. It does not publish reward or behavior-fidelity "
            "scores and cannot upgrade an external-validation claim."
        ),
    )


def evaluate_real_provider_behavior_evidence(
    *,
    provider_id: str,
    traces: tuple[BehaviorFamilyProviderTrace, ...],
    portfolio: BehaviorFamilyPortfolioReport,
) -> RealProviderBehaviorEvidenceReport:
    """Verify that owner evidence was produced by a scoped real provider."""

    if not provider_id.strip():
        raise ValueError("provider_id must be non-empty")
    if not traces:
        raise ValueError("traces must be non-empty")
    if any(trace.provider_id != provider_id for trace in traces):
        raise ValueError("all traces must use the declared provider_id")
    if tuple(trace.call_index for trace in traces) != tuple(
        range(len(traces))
    ):
        raise ValueError("trace call indexes must be contiguous from zero")

    abstraction_traces = tuple(
        trace
        for trace in traces
        if trace.prompt_kind is BehaviorFamilyPromptKind.ACTION_ABSTRACTION
    )
    applicability_traces = tuple(
        trace
        for trace in traces
        if trace.prompt_kind is BehaviorFamilyPromptKind.ACTION_APPLICABILITY
    )
    generalization_traces = tuple(
        trace
        for trace in traces
        if trace.prompt_kind
        is BehaviorFamilyPromptKind.ACTION_GENERALIZATION_AUDIT
    )
    excluded_traces = tuple(
        trace
        for trace in traces
        if trace.prompt_kind is BehaviorFamilyPromptKind.EXCLUDED_OTHER
    )
    delegated_traces = (
        abstraction_traces
        + generalization_traces
        + applicability_traces
    )
    delegated_only_action_protocols = (
        all(trace.delegated_to_provider for trace in delegated_traces)
        and all(
            not trace.delegated_to_provider for trace in excluded_traces
        )
    )
    structured_outputs = all(
        trace.response_is_json_object for trace in delegated_traces
    )
    abstraction_outputs = (
        portfolio.promotion_count >= 2
        and len(abstraction_traces) >= portfolio.promotion_count
        and sum(
            trace.response_is_non_empty_json_object
            for trace in abstraction_traces
        )
        >= portfolio.promotion_count
    )
    applicability_outputs = (
        portfolio.routing_case_count >= 2
        and len(applicability_traces) >= portfolio.routing_case_count
        and sum(
            trace.response_is_non_empty_json_object
            for trace in applicability_traces
        )
        >= portfolio.routing_case_count
    )
    generalization_outputs = (
        portfolio.promotion_count >= 2
        and portfolio.generalization_audited_promotion_count
        == portfolio.promotion_count
        and len(generalization_traces) >= portfolio.promotion_count
        and sum(
            trace.response_is_non_empty_json_object
            for trace in generalization_traces
        )
        >= portfolio.promotion_count
    )
    gates = (
        (
            "action_protocol_scope",
            "pass" if delegated_only_action_protocols else "fail",
        ),
        (
            "structured_provider_outputs",
            "pass" if structured_outputs else "fail",
        ),
        (
            "provider_abstraction_consumed",
            "pass" if abstraction_outputs else "fail",
        ),
        (
            "provider_generalization_audit_consumed",
            "pass" if generalization_outputs else "fail",
        ),
        (
            "provider_applicability_consumed",
            "pass" if applicability_outputs else "fail",
        ),
        (
            "owner_portfolio_pass",
            "pass" if portfolio.multi_family_owner_supported else "fail",
        ),
    )
    supported = all(status == "pass" for _gate, status in gates)
    return RealProviderBehaviorEvidenceReport(
        provider_id=provider_id,
        delegated_abstraction_call_count=len(abstraction_traces),
        delegated_generalization_audit_call_count=len(
            generalization_traces
        ),
        delegated_applicability_call_count=len(applicability_traces),
        excluded_other_call_count=len(excluded_traces),
        non_empty_structured_response_count=sum(
            trace.response_is_non_empty_json_object
            for trace in delegated_traces
        ),
        gate_statuses=gates,
        real_provider_supported=supported,
        claim_status=(
            "real-structured-provider-diagnostic-pass"
            if supported
            else "diagnostic-fail"
        ),
        description=(
            "Provider-provenance diagnostic for action abstraction, "
            "second-pass generalization audit, and held-out applicability. "
            "Protocol-scoped non-action calls are excluded from the real "
            "model; evaluation remains read-only and this result is not an "
            "external blind-validation claim."
        ),
    )


__all__ = [
    "ActionEvidenceOnlyTextProvider",
    "BehaviorFamilyPortfolioReport",
    "BehaviorFamilyPromptKind",
    "BehaviorFamilyProviderTrace",
    "BehaviorFamilyRoutingObservation",
    "RealProviderBehaviorEvidenceReport",
    "evaluate_behavior_family_portfolio",
    "evaluate_real_provider_behavior_evidence",
]
