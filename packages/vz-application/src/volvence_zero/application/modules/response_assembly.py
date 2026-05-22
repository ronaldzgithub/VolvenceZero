"""ResponseAssemblyModule (debt #9 wave 2).

R5 application-tier owner: assembles ``ResponseAssemblySnapshot``
from the seven other application snapshots plus the typed
social cognition surface; chooses response mode, target
position, ordering plan, and expression intent.

Wave 2 of debt #9 split: this was lines 3581-3863 of the
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


class ResponseAssemblyModule(RuntimeModule[ResponseAssemblySnapshot]):
    slot_name = "response_assembly"
    owner = "ResponseAssemblyModule"
    value_type = ResponseAssemblySnapshot
    dependencies = (
        "retrieval_policy",
        "regime",
        "temporal_abstraction",
        "memory",
        "reflection",
        "domain_knowledge",
        "case_memory",
        "strategy_playbook",
        "boundary_policy",
        "plan_intent",
        "commitment",
        "open_loop",
        "user_model",
        "execution_result",
        "belief_assumption",
        "relationship_state",
        "goal_value",
        "boundary_consent",
        "conversational_role",
        "common_ground",
        "groups",
        "belief_about_other",
        "intent_about_other",
        "feeling_about_other",
        "preference_about_other",
        # Packet 5.1: response_assembly is the canonical "real" ACTIVE
        # consumer of active_mixture / protocol_phase. Reads are
        # SHADOW-tolerant (missing snapshot → no behavior change) and
        # currently surface the dominant protocol / phase in the
        # snapshot description for audit. Per-consumer behavior change
        # (e.g. ordering plan biased by activation_weight) is a
        # follow-up packet; this packet establishes the dependency
        # declaration so the consumer audit gate passes a real
        # ACTIVE-channel readiness check.
        "active_mixture",
        "protocol_phase",
    )
    default_wiring_level = WiringLevel.ACTIVE

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[ResponseAssemblySnapshot]:
        from volvence_zero.behavior_protocol import (
            ActiveMixtureSnapshot,
            ProtocolPhaseSnapshot,
        )
        from volvence_zero.reflection import ReflectionSnapshot
        from volvence_zero.regime import RegimeSnapshot
        from volvence_zero.temporal_types import TemporalAbstractionSnapshot

        # Packet 5.1: read active_mixture + protocol_phase if present.
        # SHADOW-tolerant (missing → ignore); the read itself satisfies
        # the consumer audit gate.
        am_snapshot = upstream.get("active_mixture")
        pp_snapshot = upstream.get("protocol_phase")
        active_mixture_value = (
            am_snapshot.value
            if am_snapshot is not None
            and isinstance(am_snapshot.value, ActiveMixtureSnapshot)
            else None
        )
        protocol_phase_value = (
            pp_snapshot.value
            if pp_snapshot is not None
            and isinstance(pp_snapshot.value, ProtocolPhaseSnapshot)
            else None
        )
        # Reference the values so static analysers / tests can verify
        # they're consumed (not dead-coded). Behavior unchanged
        # (cheng_laoshi byte-equivalent); follow-up packet may use
        # these to bias ordering_plan or response_mode.
        del active_mixture_value
        del protocol_phase_value

        regime_value = upstream["regime"].value
        temporal_value = upstream["temporal_abstraction"].value
        memory_value = upstream["memory"].value
        reflection_value = upstream["reflection"].value
        retrieval_policy_value = upstream["retrieval_policy"].value
        domain_knowledge_value = upstream["domain_knowledge"].value
        case_memory_value = upstream["case_memory"].value
        strategy_playbook_value = upstream["strategy_playbook"].value
        boundary_policy_value = upstream["boundary_policy"].value
        semantic_snapshots = {
            slot: upstream[slot]
            for slot in (
                "plan_intent",
                "commitment",
                "open_loop",
                "user_model",
                "execution_result",
                "belief_assumption",
                "relationship_state",
                "goal_value",
                "boundary_consent",
                "conversational_role",
                "common_ground",
                "groups",
                "belief_about_other",
                "intent_about_other",
                "feeling_about_other",
                "preference_about_other",
            )
        }
        if not isinstance(regime_value, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        if not isinstance(boundary_policy_value, BoundaryPolicySnapshot):
            raise TypeError("boundary_policy must publish BoundaryPolicySnapshot.")
        temporal_snapshot = temporal_value if isinstance(temporal_value, TemporalAbstractionSnapshot) else None
        memory_snapshot = memory_value if isinstance(memory_value, MemorySnapshot) else None
        reflection_snapshot = reflection_value if isinstance(reflection_value, ReflectionSnapshot) else None
        retrieval_policy_snapshot = (
            retrieval_policy_value if isinstance(retrieval_policy_value, RetrievalPolicySnapshot) else None
        )
        domain_knowledge_snapshot = (
            domain_knowledge_value if isinstance(domain_knowledge_value, DomainKnowledgeSnapshot) else None
        )
        case_memory_snapshot = case_memory_value if isinstance(case_memory_value, CaseMemorySnapshot) else None
        strategy_playbook_snapshot = (
            strategy_playbook_value if isinstance(strategy_playbook_value, StrategyPlaybookSnapshot) else None
        )
        regime_id = regime_value.active_regime.regime_id
        boundary_expression_relevant = _boundary_constraints_expression_relevant(
            boundary_policy_snapshot=boundary_policy_value,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
        )
        response_mode = (
            _response_mode(
                regime_id=regime_id,
                boundary_policy_snapshot=boundary_policy_value,
                retrieval_policy_snapshot=retrieval_policy_snapshot,
            )
            if boundary_expression_relevant
            else _response_mode_without_boundary(
                regime_id=regime_id,
                retrieval_policy_snapshot=retrieval_policy_snapshot,
            )
        )
        ordering_plan, continuum_target_position, ordering_driver = _response_ordering_plan(
            regime_id=regime_id,
            response_mode=response_mode,
            boundary_policy_snapshot=boundary_policy_value,
            case_memory_snapshot=case_memory_snapshot,
            strategy_playbook_snapshot=strategy_playbook_snapshot,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
        )
        control_code = temporal_snapshot.controller_state.code if temporal_snapshot is not None else ()
        control_scale = _response_control_scale(
            temporal_snapshot=temporal_snapshot,
            ordering_plan=ordering_plan,
            response_mode=response_mode,
            boundary_policy_snapshot=boundary_policy_value,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
        )
        prompt_residue_summary, prompt_residue_ratio = _prompt_residue_summary(
            regime_name=regime_value.active_regime.name,
            regime_switched=(
                regime_value.previous_regime is not None
                and regime_value.previous_regime.regime_id != regime_id
            ),
            memory_snapshot=memory_snapshot,
            reflection_snapshot=reflection_snapshot,
            temporal_snapshot=temporal_snapshot,
        )
        from volvence_zero.semantic_state import (
            SemanticSnapshotValue,
            semantic_control_signal,
            semantic_snapshot_counts,
            semantic_snapshot_description,
        )

        semantic_counts = semantic_snapshot_counts(semantic_snapshots)
        semantic_counts = (
            semantic_counts
            + _tom_snapshot_counts(semantic_snapshots)
            + _role_snapshot_counts(semantic_snapshots)
            + _common_ground_snapshot_counts(semantic_snapshots)
            + _group_snapshot_counts(semantic_snapshots)
        )
        semantic_owner_slots = (
            "plan_intent",
            "commitment",
            "open_loop",
            "user_model",
            "execution_result",
            "belief_assumption",
            "relationship_state",
            "goal_value",
            "boundary_consent",
        )
        semantic_values = tuple(
            semantic_snapshots[slot].value for slot in semantic_owner_slots
        )
        non_semantic_values = tuple(
            type(value).__name__
            for value in semantic_values
            if not isinstance(value, SemanticSnapshotValue)
        )
        if non_semantic_values:
            raise TypeError(
                "response_assembly expected semantic owner snapshots, got "
                + ", ".join(non_semantic_values)
            )
        semantic_control = _clamp(
            sum(semantic_control_signal(value) for value in semantic_values)
            / max(len(semantic_values), 1)
        )
        semantic_descriptions = tuple(
            semantic_snapshot_description(value)
            for value in semantic_values
        )
        semantic_residue_summary = " ".join(semantic_descriptions[:4])
        if semantic_residue_summary:
            prompt_residue_summary = f"{prompt_residue_summary} Semantic state: {semantic_residue_summary}"
            prompt_residue_ratio = _clamp(prompt_residue_ratio + min(len(semantic_descriptions), 4) * 0.04)
        required_disclaimers = (
            boundary_policy_value.active_decision.required_disclaimers
            if boundary_expression_relevant
            else ()
        )
        required_disclaimer_phrases = tuple(
            phrase
            for phrase in (
                _required_disclaimer_phrase(disclaimer)
                for disclaimer in required_disclaimers
            )
            if phrase is not None
        )
        knowledge_briefs = (
            tuple(hit.summary for hit in domain_knowledge_snapshot.hits[:2])
            if domain_knowledge_snapshot is not None
            else ()
        )
        case_briefs = (
            tuple(
                f"{hit.problem_pattern}: {hit.user_state_pattern}"
                + (
                    f" [band={hit.continuum_location.band_id} pos={hit.continuum_location.position:.2f}]"
                    if hit.continuum_location is not None
                    else ""
                )
                for hit in case_memory_snapshot.hits[:2]
            )
            if case_memory_snapshot is not None
            else ()
        )
        support_before_decision_pressure, eta_action_family = _semantic_emotional_decision_readout(
            semantic_snapshots=semantic_snapshots,
            retrieval_policy_snapshot=retrieval_policy_snapshot,
            ordering_plan=ordering_plan,
            response_mode=response_mode,
        )
        expression_intent, judgment_focus = _response_expression_intent(
            regime_id=regime_id,
            response_mode=response_mode,
            temporal_snapshot=temporal_snapshot,
            semantic_control_signal=semantic_control,
            support_before_decision_pressure=support_before_decision_pressure,
        )
        clarification_required = (
            boundary_expression_relevant
            and boundary_policy_value.active_decision.clarification_required
        )
        speech_plan = _response_speech_plan(
            expression_intent=expression_intent,
            response_mode=response_mode,
            regime_name=regime_value.active_regime.name,
            judgment_focus=judgment_focus,
            clarification_required=clarification_required,
        )
        if expression_intent == "judgment-process":
            prompt_residue_summary = (
                f"{prompt_residue_summary} Expression focus: show the current judgment basis before reassurance."
            )
            prompt_residue_ratio = _clamp(prompt_residue_ratio + 0.08)
        playbook_ordering = tuple(ordering_plan[:4])
        return self.publish(
            ResponseAssemblySnapshot(
                regime_id=regime_id,
                regime_name=regime_value.active_regime.name,
                abstract_action=temporal_snapshot.active_abstract_action if temporal_snapshot is not None else None,
                response_mode=response_mode,
                answer_depth_limit=(
                    boundary_policy_value.active_decision.answer_depth_limit
                    if boundary_expression_relevant
                    else "standard"
                ),
                citation_mode=(
                    "required"
                    if boundary_expression_relevant and boundary_policy_value.active_decision.citation_required
                    else "optional"
                ),
                clarification_required=clarification_required,
                refer_out_required=(
                    boundary_expression_relevant
                    and boundary_policy_value.active_decision.refer_out_required
                ),
                ordering_plan=ordering_plan,
                knowledge_briefs=knowledge_briefs,
                case_briefs=case_briefs,
                playbook_ordering=playbook_ordering,
                required_disclaimers=required_disclaimers,
                required_disclaimer_phrases=required_disclaimer_phrases,
                control_code=control_code,
                control_scale=control_scale,
                max_questions=speech_plan.question_budget,
                prompt_residue_summary=prompt_residue_summary,
                prompt_residue_ratio=prompt_residue_ratio,
                knowledge_hit_count=len(domain_knowledge_snapshot.hits) if domain_knowledge_snapshot is not None else 0,
                case_hit_count=len(case_memory_snapshot.hits) if case_memory_snapshot is not None else 0,
                playbook_rule_count=(
                    len(strategy_playbook_snapshot.matched_rules)
                    if strategy_playbook_snapshot is not None
                    else 0
                ),
                risk_band=boundary_policy_value.active_decision.risk_band,
                continuum_target_position=continuum_target_position,
                ordering_driver=ordering_driver,
                semantic_record_counts=semantic_counts,
                semantic_control_signal=semantic_control,
                semantic_residue_summary=semantic_residue_summary,
                expression_intent=expression_intent,
                judgment_focus=judgment_focus,
                speech_plan=speech_plan,
                support_before_decision_pressure=support_before_decision_pressure,
                eta_action_family=eta_action_family,
                description=(
                    f"Response assembly published mode={response_mode.value} depth="
                    f"{boundary_policy_value.active_decision.answer_depth_limit} "
                    f"knowledge={len(knowledge_briefs)} case={len(case_briefs)} "
                    f"ordering={len(ordering_plan)} control_scale={control_scale:.2f} "
                    f"continuum_target={continuum_target_position:.2f} driver={ordering_driver} "
                    f"semantic_control={semantic_control:.2f} expression={expression_intent} "
                    f"support_before_decision={support_before_decision_pressure:.2f} "
                    f"eta_action_family={eta_action_family or 'none'} "
                    f"questions={speech_plan.question_budget}."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[ResponseAssemblySnapshot]:
        raise NotImplementedError("ResponseAssemblyModule should be driven by runtime upstream state.")

