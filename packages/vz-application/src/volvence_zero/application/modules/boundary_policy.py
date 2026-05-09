"""BoundaryPolicyModule (debt #9 wave 2).

R5 application-tier owner: derives ``BoundaryPolicySnapshot``
from regime + retrieval + case-memory + domain-knowledge
snapshots; chooses the active boundary decision (allow /
warn / refer-out / refuse) per turn.

Wave 2 of debt #9 split: this was lines 3404-3579 of the
original monolithic ``runtime.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemoryEntry, MemorySnapshot, Track
from volvence_zero.runtime import RuntimeModule, RuntimePlaceholderValue, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    CommonGroundSnapshot,
    ConversationalRoleSnapshot,
    FeelingAboutOtherSnapshot,
    GroupSnapshot,
    IntentAboutOtherSnapshot,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
)
from volvence_zero.application.retrieval_readout import (
    RetrievalControlReadoutInputs,
    RetrievalControlReadoutParameters,
    RetrievalReadoutCheckpoint,
    RetrievalControlReadoutStrategy,
)

if TYPE_CHECKING:
    from volvence_zero.prediction.error import PredictionErrorSnapshot
    from volvence_zero.regime import RegimeSnapshot
    from volvence_zero.temporal_types import TemporalAbstractionSnapshot


from volvence_zero.application.scoring_helpers import clamp01 as _clamp

from volvence_zero.application.types import *  # noqa: F401,F403 -- typed surface
from volvence_zero.application.runtime_helpers import *  # noqa: F401,F403
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState  # noqa: F401


class BoundaryPolicyModule(RuntimeModule[BoundaryPolicySnapshot]):
    slot_name = "boundary_policy"
    owner = "BoundaryPolicyModule"
    value_type = BoundaryPolicySnapshot
    dependencies = ("retrieval_policy", "domain_knowledge", "regime", "prediction_error", "boundary_consent")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[BoundaryPolicySnapshot]:
        from volvence_zero.prediction.error import PredictionErrorSnapshot
        from volvence_zero.regime import RegimeSnapshot

        retrieval_policy = upstream["retrieval_policy"].value
        domain_knowledge = upstream["domain_knowledge"].value
        regime_snapshot = upstream["regime"].value
        prediction_error = upstream["prediction_error"].value
        boundary_consent = upstream["boundary_consent"].value
        if not isinstance(retrieval_policy, RetrievalPolicySnapshot):
            raise TypeError("retrieval_policy must publish RetrievalPolicySnapshot.")
        if not isinstance(domain_knowledge, DomainKnowledgeSnapshot):
            raise TypeError("domain_knowledge must publish DomainKnowledgeSnapshot.")
        if not isinstance(regime_snapshot, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        if not isinstance(prediction_error, PredictionErrorSnapshot):
            raise TypeError("prediction_error must publish PredictionErrorSnapshot.")
        from volvence_zero.semantic_state import BoundaryConsentSnapshot

        boundary_consent_snapshot = (
            boundary_consent if isinstance(boundary_consent, BoundaryConsentSnapshot) else None
        )
        trigger_reasons: list[str] = []
        retrieval_risk_score = dict(RISK_BAND_ANCHORS).get(retrieval_policy.risk_band, 0.18)
        prediction_risk_score = _clamp(
            min(max(prediction_error.error.magnitude, 0.0) / 4.0, 1.0) * 0.58
            + _clamp(-prediction_error.error.relationship_error) * 0.42
        )
        effective_risk_score = max(retrieval_risk_score, prediction_risk_score)
        effective_risk_band = _nearest_anchor_value(effective_risk_score, RISK_BAND_ANCHORS)
        if prediction_risk_score >= 0.82:
            trigger_reasons.append("prediction-error-critical")
        elif prediction_risk_score >= 0.58:
            trigger_reasons.append("prediction-error-high")
        clarification_score = _clamp(
            (1.0 if retrieval_policy.jurisdiction_required else 0.0) * 0.40
            + min(len(domain_knowledge.unresolved_conflicts), 3) / 3.0 * 0.45
            + (1.0 if retrieval_policy.citation_required else 0.0) * 0.15
            + retrieval_policy.clarification_bias * 0.20
        )
        clarification_required = clarification_score >= 0.5
        if clarification_required:
            trigger_reasons.append("jurisdiction-clarification-required")
        refer_out_score = _clamp(
            effective_risk_score * 0.74
            + clarification_score * 0.16
            + retrieval_policy.refer_out_bias * 0.16
            + _application_brief(retrieval_policy.regime_id).domain_bonus("refer_out")
        )
        refer_out_required = refer_out_score >= 0.84
        if refer_out_required:
            trigger_reasons.append("refer-out-required")
        citation_required = retrieval_policy.citation_required
        if citation_required:
            trigger_reasons.append("citation-required")
        if boundary_consent_snapshot is not None and boundary_consent_snapshot.missing_consents:
            clarification_required = True
            trigger_reasons.append("consent-clarification-required")
        if boundary_consent_snapshot is not None and boundary_consent_snapshot.denied_boundaries:
            refer_out_required = True
            trigger_reasons.append("consent-boundary-denied")
        answer_depth_score = _clamp(
            effective_risk_score * 0.28
            + clarification_score * 0.30
            + (1.0 if citation_required else 0.0) * 0.24
            + retrieval_policy.answer_depth_bias * 0.18
            + _application_brief(retrieval_policy.regime_id).domain_bonus("answer_depth")
        )
        answer_depth_limit = _nearest_anchor_value(answer_depth_score, ANSWER_DEPTH_ANCHORS)
        professional_scope = (
            ProfessionalScope.PROFESSIONAL_HANDOFF
            if refer_out_required
            else ProfessionalScope.DOMAIN_INFORMATION
            if citation_required or retrieval_policy.jurisdiction_required
            else ProfessionalScope.GENERAL_SUPPORT
        )
        required_disclaimers: list[str] = []
        if retrieval_policy.jurisdiction_required:
            required_disclaimers.append("jurisdiction-variance")
        if clarification_required:
            required_disclaimers.append("clarify-before-concluding")
        if refer_out_required:
            required_disclaimers.append("professional-handoff")
        blocked_topics: tuple[str, ...] = ("definitive-domain-conclusion",) if citation_required else ()
        if self._rare_heavy_state is not None and self._rare_heavy_state.boundary_prior_hints:
            matching_hints = tuple(
                hint
                for hint in self._rare_heavy_state.boundary_prior_hints
                if (
                    hint.regime_id is None
                    or hint.regime_id == retrieval_policy.regime_id
                )
                and (
                    not hint.trigger_reasons
                    or set(hint.trigger_reasons).intersection(trigger_reasons)
                )
            )
            if matching_hints:
                clarification_required = clarification_required or any(
                    hint.clarification_required for hint in matching_hints
                )
                refer_out_required = refer_out_required or any(
                    hint.refer_out_required for hint in matching_hints
                )
                if any(hint.answer_depth_limit_hint == "high-level-only" for hint in matching_hints):
                    answer_depth_limit = "high-level-only"
                elif (
                    answer_depth_limit != "high-level-only"
                    and any(hint.answer_depth_limit_hint == "support-first" for hint in matching_hints)
                ):
                    answer_depth_limit = "support-first"
                blocked_topics = _dedupe(
                    blocked_topics
                    + tuple(topic for hint in matching_hints for topic in hint.blocked_topics)
                )
                required_disclaimers = list(
                    _dedupe(
                        tuple(required_disclaimers)
                        + tuple(
                            disclaimer
                            for hint in matching_hints
                            for disclaimer in hint.required_disclaimers
                        )
                    )
                )
                trigger_reasons.append("boundary-prior-consumed")
                professional_scope = (
                    ProfessionalScope.PROFESSIONAL_HANDOFF
                    if refer_out_required
                    else professional_scope
                )
        decision = BoundaryDecision(
            decision_id=f"boundary:{regime_snapshot.active_regime.regime_id}:{answer_depth_limit}",
            risk_band=effective_risk_band,
            professional_scope=professional_scope,
            answer_depth_limit=answer_depth_limit,
            citation_required=citation_required,
            clarification_required=clarification_required,
            refer_out_required=refer_out_required,
            blocked_topics=blocked_topics,
            required_disclaimers=tuple(required_disclaimers),
            description=(
                f"Boundary policy regime={regime_snapshot.active_regime.regime_id} "
                f"risk={effective_risk_band.value} scope={professional_scope.value} depth={answer_depth_limit}."
            ),
        )
        return self.publish(
            BoundaryPolicySnapshot(
                active_decision=decision,
                trigger_reasons=tuple(trigger_reasons),
                description=(
                    f"Boundary policy published with clarification={clarification_required} "
                    f"refer_out={refer_out_required} and {len(domain_knowledge.hits)} knowledge hits."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[BoundaryPolicySnapshot]:
        raise NotImplementedError("BoundaryPolicyModule should be driven by runtime upstream state.")

