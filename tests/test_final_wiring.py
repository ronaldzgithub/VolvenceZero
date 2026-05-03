from __future__ import annotations

import asyncio

from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    ApplicationPriorUpdate,
    ApplicationRareHeavyCheckpoint,
    ApplicationRareHeavyState,
    BoundaryDecision,
    BoundaryPolicySnapshot,
    CaseMemoryPriorUpdate,
    CaseMemorySnapshot,
    CaseMemoryRecord,
    DomainKnowledgeCheckpoint,
    DomainKnowledgePriorUpdate,
    DomainKnowledgeRecord,
    EvidenceStrength,
    ExperienceConsolidationSnapshot,
    ExperienceFastPriorActionBias,
    ExperienceFastPriorFamilyBias,
    ExperienceFastPriorSnapshot,
    KnowledgeCitation,
    KnowledgeHit,
    KnowledgeReviewDecision,
    KnowledgeReviewStatus,
    KnowledgeSourceKind,
    KnowledgeSourceType,
    ExternalKnowledgeCandidate,
    PlaybookRule,
    ProfessionalScope,
    ResponseMode,
    RetrievalReadoutCheckpoint,
    RetrievalReadoutPriorUpdate,
    RiskBand,
    StrategyPlaybookPriorUpdate,
)
from volvence_zero.application.experience_layers import (
    ApplicationPriorProposalBuilder,
    ApplicationPriorProposalInputs,
)
from volvence_zero.application.knowledge_channels import (
    apply_knowledge_review_decisions,
    build_conversation_knowledge_candidates,
)
from volvence_zero.application.retrieval_readout import (
    RetrievalControlReadoutInputs,
    RetrievalControlReadoutParameters,
    RetrievalControlReadoutStrategy,
)
from volvence_zero.application.runtime import _response_ordering_plan
from volvence_zero.credit.gate import CreditSnapshot, GateDecision, ModificationGate, SelfModificationRecord
from volvence_zero.evaluation import EvaluationScore
from volvence_zero.integration import _apply_application_prior_writeback, FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.joint_loop import ScheduledJointLoopResult
from volvence_zero.memory import (
    build_default_memory_store,
    CMSTowerConsolidationUpdate,
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    Track,
)
from volvence_zero.prediction import PredictionErrorModule
from volvence_zero.reflection import WritebackMode
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import SEMANTIC_OWNER_SLOTS
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    BeliefAboutOtherSnapshot,
    CommonGroundSnapshot,
    ConversationalRoleSnapshot,
    FeelingAboutOtherSnapshot,
    GroupSnapshot,
    IntentAboutOtherSnapshot,
    MultiPartyIdentitySnapshot,
    PreferenceAboutOtherSnapshot,
    SocialPredictionError,
    SocialPredictionErrorSnapshot,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialPredictionSnapshot,
    SocialScopeKind,
)
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
)
from volvence_zero.temporal import ControllerState, FullLearnedTemporalPolicy, TemporalAbstractionSnapshot
from volvence_zero.dual_track import DualTrackSnapshot, TrackState
from volvence_zero.environment import (
    EnvironmentActorRef,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentFrame,
)


def test_retrieval_control_readout_stays_compact_and_adjusts_domains():
    readout = RetrievalControlReadoutStrategy().build(
        RetrievalControlReadoutInputs(
            regime_id="emotional_support",
            abstract_action="stabilize_controller",
            knowledge_domains=("family_transition",),
            experience_domains=("general_guidance_patterns",),
            world_weight=0.32,
            self_weight=0.71,
            continuum_position=0.78,
            continuum_frequency=0.25,
            continuum_slow_share=0.51,
            continuum_reconstruction_pressure=0.62,
            delayed_payoff_signal=0.58,
            sequence_payoff_signal=0.54,
            playbook_knowledge_hint=0.34,
            playbook_experience_hint=0.73,
            knowledge_weight_bias=-0.08,
            experience_weight_bias=0.12,
            regime_fast_prior_bias=0.05,
            action_fast_prior_bias=0.08,
            family_fast_prior_bias=0.06,
            action_family_version=1,
            switch_gate=0.18,
            continuum_active_band_ids=("background-slow", "tower-readout"),
        )
    )

    assert readout.knowledge_domains == ("family_transition",)
    assert "stabilization_patterns" in readout.experience_domains
    assert readout.experience_weight > readout.knowledge_weight
    assert "compact ETA/application control" in readout.description


def test_retrieval_readout_parameters_accept_bounded_slow_prior_updates():
    base = RetrievalControlReadoutParameters.default()
    updated = base.updated_from_slow_prior(
        strength=0.82,
        attribution_count=5,
        sequence_count=3,
        regime_bias=0.18,
        action_bias=0.24,
        family_bias=0.16,
        knowledge_weight_bias=-0.10,
        experience_weight_bias=0.14,
    )

    assert updated != base
    assert updated.experience_weight_head.bias > base.experience_weight_head.bias
    assert updated.knowledge_weight_head.bias < base.knowledge_weight_head.bias
    assert updated.stabilization_domain_head.bias > base.stabilization_domain_head.bias


def test_application_prior_proposal_builder_stays_owner_side_and_typed():
    knowledge_hits = (
        KnowledgeHit(
            hit_id="knowledge:family-transition:1",
            domain="family_transition",
            topic_tags=("family", "transition"),
            jurisdiction_tags=("local-law-sensitive",),
            freshness_label="seed-current",
            confidence=0.72,
            evidence_strength=EvidenceStrength.MEDIUM,
            summary="High-level family transition guidance.",
            conflict_markers=(),
            citations=(
                KnowledgeCitation(
                    citation_id="knowledge:family-transition:1:primary",
                    source_type=KnowledgeSourceType.OFFICIAL_GUIDE,
                    title="Family transition basics",
                    locator="phase1-seed",
                    snippet="Confirm local specifics before conclusions.",
                    url=None,
                ),
            ),
            description="Seed knowledge hit for owner-side proposal test.",
        ),
    )
    conversation_knowledge_candidates = build_conversation_knowledge_candidates(
        knowledge_hits=knowledge_hits,
        context_session_id="ctx-1",
        source_wave_id="wave-1",
        source_turn_index=3,
        boundary_trigger_reasons=("citation-required", "jurisdiction-clarification-required"),
    )
    proposal = ApplicationPriorProposalBuilder().build(
        inputs=ApplicationPriorProposalInputs(
            job_id="job-1",
            closed_at_turn=3,
            regime_id="emotional_support",
            knowledge_domains=("family_transition",),
            experience_domains=("stabilization_patterns",),
            case_problem_patterns=("family-transition-high-emotion",),
            case_risk_markers=("risk-medium", "child-impact"),
            boundary_trigger_reasons=("citation-required", "jurisdiction-clarification-required"),
            knowledge_weight=0.42,
            experience_weight=0.68,
            case_hit_count=2,
            mean_experience_quality=0.74,
            knowledge_hits=knowledge_hits,
            conversation_knowledge_candidates=conversation_knowledge_candidates,
        )
    )

    assert proposal is not None
    assert proposal.case_memory_updates
    assert proposal.strategy_playbook_updates
    assert proposal.boundary_policy_updates
    assert proposal.domain_knowledge_updates
    assert proposal.case_memory_updates[0].target.startswith("application.case_memory.records.")
    assert proposal.strategy_playbook_updates[0].rule.recommended_ordering
    assert proposal.boundary_policy_updates[0].hint.trigger_reasons


def test_application_prior_writeback_applies_retrieval_readout_checkpoint_owner_side():
    rare_heavy_state = ApplicationRareHeavyState()
    prior_update = ApplicationPriorUpdate(
        source_session_post_job_id="job:retrieval-readout",
        retrieval_readout_updates=(
            RetrievalReadoutPriorUpdate(
                update_id="update:retrieval-readout",
                target="application.retrieval_readout.checkpoint",
                checkpoint=RetrievalReadoutCheckpoint(
                    checkpoint_id="retrieval-readout:job",
                    parameters=RetrievalControlReadoutParameters.default().updated_from_slow_prior(
                        strength=0.72,
                        attribution_count=4,
                        sequence_count=2,
                        regime_bias=0.14,
                        action_bias=0.18,
                        family_bias=0.12,
                        knowledge_weight_bias=-0.08,
                        experience_weight_bias=0.10,
                    ),
                    confidence=0.74,
                    description="Session-post retrieval readout checkpoint.",
                    source_session_post_job_id="job:retrieval-readout",
                    source_attribution_ids=("attr:1", "attr:2"),
                    source_sequence_ids=("seq:1",),
                    mean_retrieval_mix_alignment=0.68,
                    mean_regime_alignment=0.66,
                    mean_action_alignment=0.64,
                    mean_sequence_payoff=0.72,
                ),
                confidence=0.74,
                description="Promote retrieval readout checkpoint from delayed evidence.",
            ),
        ),
        description="Application prior update with retrieval readout checkpoint.",
    )

    operations, blocks, audits, report = _apply_application_prior_writeback(
        prior_update=prior_update,
        domain_knowledge_store=ApplicationDomainKnowledgeStore(),
        case_memory_store=ApplicationCaseMemoryStore(),
        application_rare_heavy_state=rare_heavy_state,
        credit_snapshot=None,
        timestamp_ms=10,
        checkpoint_id="checkpoint:test",
        apply_enabled=True,
        blocked_reason="allow",
    )

    assert report is not None
    assert not blocks
    assert any(op.startswith("application-prior:retrieval-readout:") for op in operations)
    assert rare_heavy_state.retrieval_readout_checkpoint is not None
    assert rare_heavy_state.retrieval_readout_checkpoint.checkpoint_id == "retrieval-readout:job"
    assert rare_heavy_state.retrieval_readout_checkpoint.source_attribution_ids == ("attr:1", "attr:2")
    assert rare_heavy_state.retrieval_readout_checkpoint.source_sequence_ids == ("seq:1",)
    assert audits


def test_final_wiring_turn_builds_expected_active_and_shadow_chain():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="final-model",
                feature_surface=(FeatureSignal(name="final_context", values=(0.5,), source="adapter"),),
            ),
            session_id="s1",
            wave_id="w1",
        )
    )

    assert result.acceptance_report.passed is True
    assert "substrate" in result.active_snapshots
    assert "memory" in result.active_snapshots
    assert "retrieval_policy" in result.active_snapshots
    assert "domain_knowledge" in result.active_snapshots
    assert "boundary_policy" in result.active_snapshots
    assert "response_assembly" in result.active_snapshots
    assert "dual_track" in result.active_snapshots
    assert "evaluation" in result.active_snapshots
    assert "regime" in result.active_snapshots
    assert "credit" in result.active_snapshots
    assert "reflection" in result.active_snapshots
    assert "temporal_abstraction" in result.active_snapshots
    assert "substrate_self_mod" in result.active_snapshots
    assert "case_memory" in result.active_snapshots
    assert "strategy_playbook" in result.active_snapshots
    assert "experience_fast_prior" in result.active_snapshots
    for slot_name in SEMANTIC_OWNER_SLOTS:
        assert slot_name in result.active_snapshots
    assert result.temporal_runtime_state is not None
    assert result.temporal_runtime_state.mode == "full-learned"
    metric_names = {score.metric_name for score in result.active_snapshots["evaluation"].value.turn_scores}
    assert "retrieval_quality" in metric_names
    assert "knowledge_hit_count" in metric_names
    assert "boundary_clarification_triggered" in metric_names
    assert "reflection_usefulness" in metric_names
    assert "fallback_reliance" in metric_names
    assert "temporal_action_commitment" in metric_names
    assert "memory_tower_depth" in metric_names
    assert "memory_tower_alignment" in metric_names
    assert "tower_consolidation_activity" in metric_names
    assert "continuum_frequency_coverage" in metric_names
    assert "continuum_reconstruction_capacity" in metric_names
    assert "substrate_online_fast_change_rate" in metric_names
    assert "substrate_online_fast_gate_preview" in metric_names
    assert "substrate_online_fast_optimizer_norm" in metric_names
    assert "substrate_online_fast_recommended" in metric_names
    assert "substrate_online_fast_runtime_evidence_strength" in metric_names
    assert "substrate_online_fast_proposal_readiness" in metric_names
    assert "plan_continuity" in metric_names
    assert "commitment_honoring" in metric_names
    assert "open_loop_closure_pressure" in metric_names
    assert "user_model_stability" in metric_names
    assert "execution_grounding" in metric_names
    assert "belief_verification" in metric_names
    assert "relationship_continuity" in metric_names
    assert "goal_alignment" in metric_names
    assert "consent_compliance" in metric_names
    assert "semantic_spine_coverage" in metric_names
    assert "cognitive_loop_readiness" in metric_names
    assert any(
        "cognitive_loop_readiness" in recommendation
        for recommendation in result.acceptance_report.recommendations
    )


def test_final_wiring_publishes_multi_party_identity_active_scaffold():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="social-identity-active-model",
                feature_surface=(
                    FeatureSignal(name="social_identity_context", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="social-identity-session",
            wave_id="social-identity-wave",
        )
    )

    assert "multi_party_identity" in result.active_snapshots
    snapshot = result.active_snapshots["multi_party_identity"]
    value = snapshot.value
    assert isinstance(value, MultiPartyIdentitySnapshot)
    assert value.active_speaker_id == PRIMARY_INTERLOCUTOR_ID
    assert value.subject_ids == (PRIMARY_INTERLOCUTOR_ID,)
    assert value.addressee_ids == (SELF_INTERLOCUTOR_ID,)
    assert value.audience_ids == (SELF_INTERLOCUTOR_ID,)
    assert value.identity_predictions == ()
    assert "R16 SHADOW scaffold" in value.description


def test_final_wiring_multi_party_identity_consumes_environment_frame():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="environment-frame-model",
                feature_surface=(
                    FeatureSignal(name="environment_frame", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="environment-frame-session",
            wave_id="environment-frame-wave",
            environment_event=EnvironmentEvent(
                event_id="env-frame-1",
                event_kind=EnvironmentEventKind.USER_INPUT,
                trigger_kind="user_input",
                frame=EnvironmentFrame(
                    actor=EnvironmentActorRef(actor_id="alice"),
                    active_speaker_id="alice",
                    addressee_ids=(SELF_INTERLOCUTOR_ID,),
                    subject_ids=("alice",),
                    audience_ids=(SELF_INTERLOCUTOR_ID, "alice"),
                ),
                scene_id="scene-1",
                timestamp_ms=1,
                provenance="test",
                payload_summary="Alice asks a question.",
            ),
        )
    )

    snapshot = result.active_snapshots["multi_party_identity"]
    value = snapshot.value
    assert isinstance(value, MultiPartyIdentitySnapshot)
    assert value.active_speaker_id == "alice"
    assert value.subject_ids == ("alice",)
    assert value.audience_ids == (SELF_INTERLOCUTOR_ID, "alice")
    assert "env-frame-1" in value.description


def test_final_wiring_multi_party_identity_kill_switch_rolls_back_to_disabled():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(kill_switches=frozenset({"multi_party_identity"})),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="environment-frame-kill-switch-model",
                feature_surface=(
                    FeatureSignal(name="environment_frame", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="environment-frame-kill-switch-session",
            wave_id="environment-frame-kill-switch-wave",
        )
    )

    assert "multi_party_identity" not in result.active_snapshots
    assert "multi_party_identity" not in result.shadow_snapshots


def test_final_wiring_publishes_empty_social_prediction_active_scaffolds():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="social-prediction-active-model",
                feature_surface=(
                    FeatureSignal(name="social_prediction_context", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="social-prediction-session",
            wave_id="social-prediction-wave",
        )
    )

    assert "social_prediction" in result.active_snapshots
    assert "social_prediction_error" in result.active_snapshots
    prediction_value = result.active_snapshots["social_prediction"].value
    error_value = result.active_snapshots["social_prediction_error"].value
    assert isinstance(prediction_value, SocialPredictionSnapshot)
    assert isinstance(error_value, SocialPredictionErrorSnapshot)
    assert prediction_value.predictions == ()
    assert error_value.errors == ()
    assert "scope=default-or-missing" in prediction_value.description
    assert "memory_visibility_pe=0" in error_value.description


def test_final_wiring_social_prediction_kill_switch_rolls_back_to_disabled():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                kill_switches=frozenset({"social_prediction", "social_prediction_error"})
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="social-prediction-kill-switch-model",
                feature_surface=(
                    FeatureSignal(name="social_prediction_context", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="social-prediction-kill-switch-session",
            wave_id="social-prediction-kill-switch-wave",
        )
    )

    assert "social_prediction" not in result.active_snapshots
    assert "social_prediction_error" not in result.active_snapshots
    assert "social_prediction" not in result.shadow_snapshots
    assert "social_prediction_error" not in result.shadow_snapshots


def test_final_wiring_publishes_empty_tom_owner_shadow_scaffolds():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="tom-shadow-model",
                feature_surface=(
                    FeatureSignal(name="tom_context", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="tom-shadow-session",
            wave_id="tom-shadow-wave",
        )
    )

    expected = {
        "belief_about_other": BeliefAboutOtherSnapshot,
        "intent_about_other": IntentAboutOtherSnapshot,
        "feeling_about_other": FeelingAboutOtherSnapshot,
        "preference_about_other": PreferenceAboutOtherSnapshot,
    }
    for slot_name, expected_type in expected.items():
        assert slot_name not in result.active_snapshots
        value = result.shadow_snapshots[slot_name].value
        assert isinstance(value, expected_type)
        assert value.records == ()
        assert value.active_predictions == ()
        assert value.control_signal == 0.0
        assert "R17 SHADOW scaffold" in value.description


def test_final_wiring_publishes_conversational_role_active_scaffold():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="role-active-model",
                feature_surface=(
                    FeatureSignal(name="role_context", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="role-active-session",
            wave_id="role-active-wave",
        )
    )

    assert "conversational_role" in result.active_snapshots
    value = result.active_snapshots["conversational_role"].value
    assert isinstance(value, ConversationalRoleSnapshot)
    assert value.active_speaker_id == PRIMARY_INTERLOCUTOR_ID
    assert value.addressee_ids == (SELF_INTERLOCUTOR_ID,)
    assert value.subject_ids == (PRIMARY_INTERLOCUTOR_ID,)
    assert value.witness_ids == ()
    assert value.overhearer_ids == ()
    assert value.group_audience_ids == ()
    assert value.active_predictions == ()
    assert "R18 SHADOW scaffold" in value.description


def test_final_wiring_conversational_role_consumes_environment_frame():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="role-environment-frame-model",
                feature_surface=(
                    FeatureSignal(name="role_environment_frame", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="role-environment-session",
            wave_id="role-environment-wave",
            environment_event=EnvironmentEvent(
                event_id="role-env-frame-1",
                event_kind=EnvironmentEventKind.USER_INPUT,
                trigger_kind="user_input",
                frame=EnvironmentFrame(
                    actor=EnvironmentActorRef(actor_id="alice"),
                    active_speaker_id="alice",
                    addressee_ids=("bob",),
                    subject_ids=("carol",),
                    audience_ids=("bob", "alice"),
                ),
                scene_id="scene-role-1",
                timestamp_ms=1,
                provenance="test",
                payload_summary="Alice tells Bob about Carol.",
            ),
        )
    )

    value = result.active_snapshots["conversational_role"].value
    assert isinstance(value, ConversationalRoleSnapshot)
    assert value.active_speaker_id == "alice"
    assert value.addressee_ids == ("bob",)
    assert value.subject_ids == ("carol",)
    assert value.witness_ids == ()
    assert value.overhearer_ids == ()
    assert value.group_audience_ids == ()
    assert "role-env-frame-1" in value.description
    assert len(value.active_predictions) == 1
    prediction = value.active_predictions[0]
    assert prediction.kind is SocialPredictionKind.ROLE_ASSIGNMENT
    assert prediction.prediction_id == "role-env-frame-1:role-assignment"
    assert prediction.scope_id == "alice"
    assert prediction.subject_ids == ("carol",)
    assert prediction.audience_ids == ("bob", "alice")


def test_final_wiring_conversational_role_kill_switch_disables_active_role():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(kill_switches=frozenset({"conversational_role"})),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="role-kill-switch-model",
                feature_surface=(
                    FeatureSignal(name="role_kill_switch", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="role-kill-switch-session",
            wave_id="role-kill-switch-wave",
        )
    )

    assert "conversational_role" not in result.active_snapshots
    assert "conversational_role" not in result.shadow_snapshots


def test_response_assembly_surfaces_conversational_role_prediction_count_diagnostically():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(conversational_role=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="role-count-model",
                feature_surface=(
                    FeatureSignal(name="role_count_context", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="role-count-session",
            wave_id="role-count-wave",
            environment_event=EnvironmentEvent(
                event_id="role-count-frame-1",
                event_kind=EnvironmentEventKind.USER_INPUT,
                trigger_kind="user_input",
                frame=EnvironmentFrame(
                    actor=EnvironmentActorRef(actor_id="alice"),
                    active_speaker_id="alice",
                    addressee_ids=("bob",),
                    subject_ids=("carol",),
                    audience_ids=("bob", "alice"),
                ),
                scene_id="scene-role-count",
                timestamp_ms=1,
                provenance="test",
                payload_summary="Alice tells Bob about Carol.",
            ),
        )
    )

    response_assembly = result.active_snapshots["response_assembly"].value
    counts = dict(response_assembly.semantic_record_counts)
    assert counts["conversational_role"] == 1
    assert "conversational_role" not in response_assembly.semantic_residue_summary
    assert response_assembly.expression_intent


def test_final_wiring_publishes_common_ground_shadow_scaffold():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="common-ground-shadow-model",
                feature_surface=(
                    FeatureSignal(name="common_ground_context", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="common-ground-shadow-session",
            wave_id="common-ground-shadow-wave",
        )
    )

    assert "common_ground" not in result.active_snapshots
    value = result.shadow_snapshots["common_ground"].value
    assert isinstance(value, CommonGroundSnapshot)
    assert value.dyad_atoms == ()
    assert value.group_atoms == ()
    assert value.active_predictions == ()
    assert value.control_signal == 0.0
    assert "R19 SHADOW scaffold" in value.description


def test_final_wiring_publishes_groups_shadow_scaffold():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="groups-shadow-model",
                feature_surface=(
                    FeatureSignal(name="groups_context", values=(0.5,), source="adapter"),
                ),
            ),
            session_id="groups-shadow-session",
            wave_id="groups-shadow-wave",
        )
    )

    assert "groups" not in result.active_snapshots
    value = result.shadow_snapshots["groups"].value
    assert isinstance(value, GroupSnapshot)
    assert value.groups == ()
    assert value.active_group_id is None
    assert value.joint_attention == ()
    assert value.joint_commitments == ()
    assert value.group_regime_id is None
    assert value.active_predictions == ()
    assert "R20 SHADOW scaffold" in value.description


def test_final_wiring_records_social_prediction_errors_into_credit_ledger():
    social_error = SocialPredictionError(
        error_id="social-pe:alice-bob:runtime",
        prediction_id="social-pred:alice-memory-visible-to-bob",
        kind=SocialPredictionKind.MEMORY_VISIBILITY,
        outcome=SocialPredictionOutcome.DISCONFIRMED,
        magnitude=0.64,
        owner="MultiPartyIdentityModule",
        scope_kind=SocialScopeKind.INTERLOCUTOR,
        scope_id="alice",
        evidence=("Alice-scoped preference entered Bob audience context.",),
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="social-pe-credit-model",
                feature_surface=(
                    FeatureSignal(name="social_pe_context", values=(0.5,), source="adapter"),
                ),
            ),
            social_prediction_errors=(social_error,),
            session_id="social-pe-credit-session",
            wave_id="social-pe-credit-wave",
        )
    )

    credit = result.active_snapshots["credit"].value
    social_credits = [
        record
        for record in credit.recent_credits
        if record.level == "social_prediction_error"
    ]
    assert len(social_credits) == 1
    assert social_credits[0].source_event == "social_pe:memory_visibility"
    assert social_credits[0].credit_value == -0.64
    assert "scope=interlocutor:alice" in social_credits[0].context


def test_final_wiring_phase1_slots_publish_compact_knowledge_and_boundary_state():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="phase1-knowledge-model",
                feature_surface=(FeatureSignal(name="phase1_context", values=(0.55,), source="adapter"),),
            ),
            session_id="phase1-session",
            wave_id="phase1-wave",
        )
    )

    retrieval_policy = result.active_snapshots["retrieval_policy"].value
    domain_knowledge = result.active_snapshots["domain_knowledge"].value
    boundary_policy = result.active_snapshots["boundary_policy"].value
    response_assembly = result.active_snapshots["response_assembly"].value

    assert retrieval_policy.knowledge_domains
    assert domain_knowledge.hits
    assert domain_knowledge.active_domains == retrieval_policy.knowledge_domains
    assert boundary_policy.active_decision.risk_band.value in {"low", "medium", "high", "critical"}
    assert isinstance(boundary_policy.trigger_reasons, tuple)
    assert response_assembly.answer_depth_limit == boundary_policy.active_decision.answer_depth_limit
    assert response_assembly.knowledge_hit_count == len(domain_knowledge.hits)
    assert response_assembly.ordering_plan


def test_final_wiring_phase2_case_memory_publishes_sibling_case_hits():
    memory_store = MemoryStore()
    memory_store.write(
        request=MemoryWriteRequest(
            content="I feel overwhelmed about divorce and want the smallest next step first.",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
            tags=("divorce", "overwhelmed"),
        ),
        timestamp_ms=1,
    )
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(case_memory=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="phase2-case-model",
                feature_surface=(FeatureSignal(name="phase2_context", values=(0.61,), source="adapter"),),
            ),
            memory_store=memory_store,
            session_id="phase2-session",
            wave_id="phase2-wave",
        )
    )

    case_memory = result.active_snapshots["case_memory"].value
    metric_names = {score.metric_name for score in result.active_snapshots["evaluation"].value.turn_scores}

    assert case_memory.hits
    assert case_memory.active_problem_patterns
    assert case_memory.continuum_profile_id is not None
    assert case_memory.active_band_ids
    assert "case_hit_count" in metric_names
    assert "case_relevance_mean" in metric_names
    assert "application_continuum_case_coverage" in metric_names


def test_final_wiring_phase3_strategy_playbook_publishes_rules_from_case_memory():
    memory_store = MemoryStore()
    memory_store.write(
        request=MemoryWriteRequest(
            content="I feel overwhelmed about divorce and need the smallest next step with calm support.",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.85,
            tags=("divorce", "support"),
        ),
        timestamp_ms=1,
    )
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                case_memory=WiringLevel.ACTIVE,
                strategy_playbook=WiringLevel.ACTIVE,
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="phase3-playbook-model",
                feature_surface=(FeatureSignal(name="phase3_context", values=(0.64,), source="adapter"),),
            ),
            memory_store=memory_store,
            session_id="phase3-session",
            wave_id="phase3-wave",
        )
    )

    strategy_playbook = result.active_snapshots["strategy_playbook"].value
    metric_names = {score.metric_name for score in result.active_snapshots["evaluation"].value.turn_scores}

    assert strategy_playbook.matched_rules
    assert strategy_playbook.matched_problem_patterns
    assert strategy_playbook.matched_rules[0].recommended_ordering
    assert strategy_playbook.continuum_profile_id is not None
    assert strategy_playbook.active_band_ids
    assert "playbook_match_count" in metric_names
    assert "playbook_confidence_mean" in metric_names
    assert "application_continuum_playbook_transfer" in metric_names


def test_final_wiring_exposes_active_experience_fast_prior_contract():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="experience-fast-prior-model",
                feature_surface=(FeatureSignal(name="experience_fast_prior_context", values=(0.63,), source="adapter"),),
            ),
            session_id="experience-fast-prior-session",
            wave_id="experience-fast-prior-wave",
        )
    )

    assert "experience_fast_prior" in result.active_snapshots
    experience_fast_prior = result.active_snapshots["experience_fast_prior"].value
    assert experience_fast_prior.prior_strength == 0.0
    assert experience_fast_prior.source_attribution_ids == ()


def test_final_wiring_surfaces_upstream_experience_consolidation_with_session_owned_active_level():
    upstream_consolidation = Snapshot(
        slot_name="experience_consolidation",
        owner="ExperienceConsolidationModule",
        version=3,
        timestamp_ms=7,
        value=ExperienceConsolidationSnapshot(
            source_session_post_job_id="job:seeded",
            promoted_case_count=0,
            playbook_delta_count=0,
            boundary_delta_count=0,
            deltas=(),
            description="Seeded consolidation snapshot from session-owned surface.",
        ),
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="experience-consolidation-upstream-model",
                feature_surface=(FeatureSignal(name="experience_consolidation_context", values=(0.52,), source="adapter"),),
            ),
            upstream_snapshots={"experience_consolidation": upstream_consolidation},
            session_id="experience-consolidation-upstream-session",
            wave_id="experience-consolidation-upstream-wave",
        )
    )

    assert result.active_snapshots["experience_consolidation"] is upstream_consolidation
    assert result.active_snapshots["experience_fast_prior"].value.prior_strength == 0.0


def test_final_wiring_temporal_owner_consumes_upstream_experience_fast_prior():
    upstream_fast_prior = Snapshot(
        slot_name="experience_fast_prior",
        owner="ExperienceFastPriorModule",
        version=1,
        timestamp_ms=1,
        value=ExperienceFastPriorSnapshot(
            regime_biases=(),
            knowledge_weight_bias=0.0,
            experience_weight_bias=0.0,
            action_biases=(
                ExperienceFastPriorActionBias(
                    abstract_action="unassigned_action",
                    bias=0.18,
                    source_attribution_ids=("attr:1",),
                    description="Injected action bias for temporal owner test.",
                ),
            ),
            family_biases=(
                ExperienceFastPriorFamilyBias(
                    action_family_version=0,
                    continuation_bias=0.14,
                    source_attribution_ids=("attr:1",),
                    description="Injected family continuation bias for temporal owner test.",
                ),
            ),
            sequence_biases=(),
            prior_strength=0.42,
            source_attribution_ids=("attr:1",),
            source_sequence_ids=(),
            description="Injected experience fast prior for temporal owner-side consumption.",
        ),
    )
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(kill_switches=frozenset({"experience_fast_prior"})),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="temporal-fast-prior-model",
                feature_surface=(FeatureSignal(name="temporal_fast_prior_context", values=(0.58,), source="adapter"),),
            ),
            upstream_snapshots={"experience_fast_prior": upstream_fast_prior},
            session_id="temporal-fast-prior-session",
            wave_id="temporal-fast-prior-wave",
        )
    )

    assert result.temporal_runtime_state is not None
    assert result.temporal_runtime_state.fast_prior_strength > 0.0
    assert result.temporal_runtime_state.fast_prior_switch_pressure_delta != 0.0
    metric_names = {score.metric_name for score in result.active_snapshots["evaluation"].value.turn_scores}
    assert "temporal_fast_prior_strength" in metric_names
    assert "temporal_fast_prior_switch_pressure" in metric_names


def test_final_wiring_retrieval_policy_surfaces_shared_control_advisories():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="shared-readout-model",
                feature_surface=(
                    FeatureSignal(name="support_weighted_signal", values=(0.79,), source="adapter"),
                    FeatureSignal(name="repair_context_signal", values=(0.62,), source="adapter"),
                ),
            ),
            session_id="shared-readout-session",
            wave_id="shared-readout-wave",
        )
    )

    retrieval_policy = result.active_snapshots["retrieval_policy"].value
    assert retrieval_policy.response_mode_hint in {"support", "clarify", "structure", "refer-out"}
    assert 0.0 <= retrieval_policy.clarification_bias <= 1.0
    assert 0.0 <= retrieval_policy.refer_out_bias <= 1.0
    assert 0.0 <= retrieval_policy.answer_depth_bias <= 1.0
    assert 0.0 <= retrieval_policy.continuum_target_position_hint <= 1.0
    assert retrieval_policy.ordering_driver_hint
    assert retrieval_policy.ordering_bias


def test_final_wiring_phase3_prefers_case_derived_playbook_ordering_before_template():
    case_store = ApplicationCaseMemoryStore(
        records=(
            CaseMemoryRecord(
                case_id="case:case-derived:1",
                domain="stabilization_patterns",
                problem_pattern="family-transition-high-emotion",
                user_state_pattern="high-emotional-load",
                risk_markers=("risk-medium", "child-impact"),
                track_tags=("self",),
                regime_tags=("emotional_support",),
                intervention_ordering=("stabilize", "split_axes", "smallest_next_step"),
                outcome_label="improved",
                delayed_signal_count=4,
                escalation_observed=False,
                repair_observed=True,
                confidence=0.84,
                relevance_score=0.9,
                description="Case-derived ordering should outrank fallback template.",
            ),
        )
    )
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                case_memory=WiringLevel.ACTIVE,
                strategy_playbook=WiringLevel.ACTIVE,
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="phase3-case-derived-playbook",
                feature_surface=(FeatureSignal(name="phase3_case_context", values=(0.68,), source="adapter"),),
            ),
            case_memory_store=case_store,
            session_id="phase3-case-derived-session",
            wave_id="phase3-case-derived-wave",
        )
    )

    strategy_playbook = result.active_snapshots["strategy_playbook"].value
    assert strategy_playbook.matched_rules
    assert strategy_playbook.matched_rules[0].rule_id.startswith("playbook:case-derived:")


def test_final_wiring_response_assembly_uses_continuum_target_to_prefers_clarify_first():
    case_store = ApplicationCaseMemoryStore(
        records=(
            CaseMemoryRecord(
                case_id="case:assembly-structure",
                domain="stabilization_patterns",
                problem_pattern="family-transition-high-emotion",
                user_state_pattern="high-emotional-load",
                risk_markers=("risk-medium",),
                track_tags=("self",),
                regime_tags=("emotional_support",),
                intervention_ordering=("narrow_scope", "option_compare", "smallest_next_step"),
                outcome_label="stable",
                delayed_signal_count=1,
                escalation_observed=False,
                repair_observed=False,
                confidence=0.76,
                relevance_score=0.86,
                description="Response assembly should still stabilize first when continuum target is slow.",
                continuum_profile_id="memory-profile",
                continuum_band_id="online-fast",
                continuum_position=0.18,
                continuum_update_frequency=1.0,
                reconstruction_source="artifact-anchor",
            ),
        )
    )
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                case_memory=WiringLevel.ACTIVE,
                strategy_playbook=WiringLevel.ACTIVE,
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="response-assembly-slow-target",
                feature_surface=(FeatureSignal(name="support_weighted_signal", values=(0.86,), source="adapter"),),
            ),
            case_memory_store=case_store,
            session_id="response-assembly-slow-session",
            wave_id="response-assembly-slow-wave",
        )
    )

    response_assembly = result.active_snapshots["response_assembly"].value
    assert response_assembly.ordering_plan[0] == "clarify_goal"
    assert response_assembly.ordering_driver == "continuum-clarify-first"
    assert response_assembly.continuum_target_position < 0.66


def test_response_ordering_plan_prefers_stabilize_when_target_is_support_first():
    ordering_plan, target_position, ordering_driver = _response_ordering_plan(
        regime_id="emotional_support",
        response_mode=ResponseMode.SUPPORT,
        boundary_policy_snapshot=BoundaryPolicySnapshot(
            active_decision=BoundaryDecision(
                decision_id="boundary:test",
                risk_band=RiskBand.MEDIUM,
                professional_scope=ProfessionalScope.GENERAL_SUPPORT,
                answer_depth_limit="support-first",
                citation_required=False,
                clarification_required=False,
                refer_out_required=False,
                blocked_topics=(),
                required_disclaimers=(),
                description="test boundary",
            ),
            trigger_reasons=(),
            description="test",
        ),
        case_memory_snapshot=CaseMemorySnapshot(
            retrieval_policy_id="policy:test",
            hits=(),
            active_problem_patterns=(),
            active_risk_markers=(),
            description="test case memory",
            continuum_profile_id="memory-profile",
            active_band_ids=("background-slow",),
            mean_continuum_position=0.82,
        ),
        strategy_playbook_snapshot=None,
    )

    assert ordering_plan[0] == "stabilize"
    assert ordering_driver == "continuum-support-first"
    assert target_position >= 0.66


def test_final_wiring_response_assembly_prefers_clarify_when_boundary_requires_it():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="response-assembly-clarify-target",
                feature_surface=(
                    FeatureSignal(name="family_transition_signal", values=(0.74,), source="adapter"),
                    FeatureSignal(name="procedure_signal", values=(0.69,), source="adapter"),
                ),
            ),
            session_id="response-assembly-clarify-session",
            wave_id="response-assembly-clarify-wave",
        )
    )

    response_assembly = result.active_snapshots["response_assembly"].value
    if response_assembly.clarification_required:
        assert response_assembly.ordering_plan[0] in {"clarify_goal", "stabilize"}
        assert response_assembly.ordering_driver in {"continuum-clarify-first", "continuum-support-clarify"}


def test_response_assembly_does_not_surface_low_relevance_boundary_residue():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="response-assembly-introspection",
                feature_surface=(),
            ),
            user_input=(
                "I want to feel whether there is a real mind-like process turning here. "
                "Do not just comfort me; tell me how you are judging what I need right now."
            ),
            session_id="response-assembly-introspection-session",
            wave_id="response-assembly-introspection-wave",
        )
    )

    retrieval_policy = result.active_snapshots["retrieval_policy"].value
    boundary_policy = result.active_snapshots["boundary_policy"].value
    response_assembly = result.active_snapshots["response_assembly"].value

    assert retrieval_policy.knowledge_weight < 0.38
    assert boundary_policy.active_decision.required_disclaimers
    assert response_assembly.response_mode is not ResponseMode.CLARIFY
    assert not response_assembly.clarification_required
    assert response_assembly.citation_mode == "optional"
    assert response_assembly.required_disclaimers == ()
    assert response_assembly.required_disclaimer_phrases == ()


def test_response_assembly_marks_guided_exploration_for_judgment_process_expression():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="judgment-expression-substrate",
                feature_surface=(FeatureSignal(name="exploration_signal", values=(0.82,), source="adapter"),),
            ),
            session_id="test-judgment-expression",
            user_input=(
                "I want to see whether you really have a thinking brain. "
                "Tell me how you are judging what I need right now."
            ),
        )
    )

    response_assembly = result.active_snapshots["response_assembly"].value

    assert response_assembly.regime_id == "guided_exploration"
    assert response_assembly.expression_intent == "judgment-process"
    assert response_assembly.judgment_focus
    assert response_assembly.speech_plan is not None
    assert response_assembly.speech_plan.cue
    assert response_assembly.speech_plan.inferred_need
    assert response_assembly.speech_plan.response_adjustment
    assert response_assembly.speech_plan.question_budget == response_assembly.max_questions
    assert "Expression focus" in response_assembly.prompt_residue_summary


def test_final_wiring_retrieval_mix_absorbs_rare_heavy_playbook_prior():
    baseline = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="retrieval-mix-baseline",
                feature_surface=(
                    FeatureSignal(name="career_decision_signal", values=(0.62,), source="adapter"),
                    FeatureSignal(name="offer_tradeoff_signal", values=(0.71,), source="adapter"),
                ),
            ),
            session_id="retrieval-mix-session",
            wave_id="baseline-wave",
        )
    )

    rare_heavy_state = ApplicationRareHeavyState()
    rare_heavy_state.import_rare_heavy_state(
        ApplicationRareHeavyCheckpoint(
            checkpoint_id="experience-playbook-prior",
            domain_template_biases=(),
            case_clusters=(),
            distilled_playbook_rules=(
                PlaybookRule(
                    rule_id="playbook:eta-prior",
                        problem_pattern="career_decision_tradeoff",
                        recommended_regime="guided_exploration",
                        recommended_ordering=("narrow_scope", "option_compare", "smallest_next_step"),
                    recommended_pacing="gradual",
                    avoid_patterns=("procedure-dump-too-early",),
                    knowledge_weight_hint=0.22,
                        experience_weight_hint=0.82,
                        applicability_scope=("guided_exploration",),
                        confidence=0.86,
                        description="Distilled playbook prior for exploration-weighted retrieval mix.",
                ),
            ),
            description="rare-heavy playbook prior",
        )
    )
    with_prior = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="retrieval-mix-prior",
                feature_surface=(
                    FeatureSignal(name="career_decision_signal", values=(0.62,), source="adapter"),
                    FeatureSignal(name="offer_tradeoff_signal", values=(0.71,), source="adapter"),
                ),
            ),
            application_rare_heavy_state=rare_heavy_state,
            session_id="retrieval-mix-session",
            wave_id="prior-wave",
        )
    )

    baseline_policy = baseline.active_snapshots["retrieval_policy"].value
    prior_policy = with_prior.active_snapshots["retrieval_policy"].value

    assert prior_policy.knowledge_weight < baseline_policy.knowledge_weight
    assert prior_policy.experience_weight > baseline_policy.experience_weight


def test_final_wiring_retrieval_mix_uses_continuum_profile_as_first_class_input():
    fast_biased_store = build_default_memory_store(latent_dim=8)
    slow_biased_store = build_default_memory_store(latent_dim=8)
    fast_biased_store.learned_core.apply_tower_consolidation(
        update=CMSTowerConsolidationUpdate(
            online_signal=tuple(1.0 for _ in range(fast_biased_store.learned_core.dim)),
            description="bias toward fast continuum bands",
        ),
        timestamp_ms=1,
    )
    slow_biased_store.learned_core.apply_tower_consolidation(
        update=CMSTowerConsolidationUpdate(
            session_signal=tuple(0.8 for _ in range(slow_biased_store.learned_core.dim)),
            background_signal=tuple(1.0 for _ in range(slow_biased_store.learned_core.dim)),
            description="bias toward slow continuum bands",
        ),
        timestamp_ms=1,
    )

    fast_biased = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="continuum-fast-model",
                feature_surface=(
                    FeatureSignal(name="support_signal", values=(0.72,), source="adapter"),
                    FeatureSignal(name="decision_signal", values=(0.54,), source="adapter"),
                ),
            ),
            memory_store=fast_biased_store,
            session_id="continuum-mix-session",
            wave_id="fast-wave",
        )
    )
    slow_biased = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="continuum-slow-model",
                feature_surface=(
                    FeatureSignal(name="support_signal", values=(0.72,), source="adapter"),
                    FeatureSignal(name="decision_signal", values=(0.54,), source="adapter"),
                ),
            ),
            memory_store=slow_biased_store,
            session_id="continuum-mix-session",
            wave_id="slow-wave",
        )
    )

    fast_policy = fast_biased.active_snapshots["retrieval_policy"].value
    slow_policy = slow_biased.active_snapshots["retrieval_policy"].value

    assert slow_policy.experience_weight > fast_policy.experience_weight
    assert "continuum_position=" in slow_policy.intent_description


def test_final_wiring_playbook_ranking_prefers_hits_closer_to_target_continuum_position():
    case_store = ApplicationCaseMemoryStore(
        records=(
            CaseMemoryRecord(
                case_id="case:continuum-fast",
                domain="stabilization_patterns",
                problem_pattern="family-transition-high-emotion",
                user_state_pattern="high-emotional-load",
                risk_markers=("risk-medium",),
                track_tags=("self",),
                regime_tags=("emotional_support",),
                intervention_ordering=("jump_to_procedure", "smallest_next_step"),
                outcome_label="stable",
                delayed_signal_count=1,
                escalation_observed=False,
                repair_observed=False,
                confidence=0.72,
                relevance_score=0.86,
                description="Fast-band case should lose under support-first continuum ranking.",
                continuum_profile_id="memory-profile",
                continuum_band_id="online-fast",
                continuum_position=0.16,
                continuum_update_frequency=1.0,
                reconstruction_source="artifact-anchor",
            ),
            CaseMemoryRecord(
                case_id="case:continuum-slow",
                domain="stabilization_patterns",
                problem_pattern="family-transition-high-emotion",
                user_state_pattern="high-emotional-load",
                risk_markers=("risk-medium", "child-impact"),
                track_tags=("self",),
                regime_tags=("emotional_support",),
                intervention_ordering=("stabilize", "split_axes", "smallest_next_step"),
                outcome_label="improved",
                delayed_signal_count=4,
                escalation_observed=False,
                repair_observed=True,
                confidence=0.82,
                relevance_score=0.82,
                description="Slow-band case should win because it aligns with support-first continuum target.",
                continuum_profile_id="memory-profile",
                continuum_band_id="background-slow",
                continuum_position=0.84,
                continuum_update_frequency=0.25,
                reconstruction_source="slow-to-fast-reuse",
            ),
        )
    )
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                case_memory=WiringLevel.ACTIVE,
                strategy_playbook=WiringLevel.ACTIVE,
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="continuum-playbook-ranking",
                feature_surface=(FeatureSignal(name="support_first_signal", values=(0.83,), source="adapter"),),
            ),
            case_memory_store=case_store,
            session_id="continuum-playbook-session",
            wave_id="continuum-playbook-wave",
        )
    )

    strategy_playbook = result.active_snapshots["strategy_playbook"].value
    assert strategy_playbook.matched_rules
    assert strategy_playbook.matched_rules[0].recommended_ordering == (
        "stabilize",
        "split_axes",
        "smallest_next_step",
    )
    assert strategy_playbook.matched_rules[0].continuum_band_id == "background-slow"


def test_final_wiring_honors_kill_switches():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(kill_switches=frozenset({"reflection", "temporal"})),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="kill-switch-model",
                feature_surface=(FeatureSignal(name="kill_switch_context", values=(0.4,), source="adapter"),),
            ),
            session_id="s1",
            wave_id="w1",
        )
    )

    assert "reflection" not in result.shadow_snapshots
    assert "temporal_abstraction" not in result.shadow_snapshots
    assert "reflection" in result.acceptance_report.disabled_slots
    assert "temporal" in result.acceptance_report.disabled_slots


def test_final_wiring_allows_active_widening_but_reports_caution():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(
                reflection=WiringLevel.ACTIVE,
                temporal=WiringLevel.ACTIVE,
            ),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="wide-model",
                feature_surface=(FeatureSignal(name="wide_context", values=(0.8,), source="adapter"),),
            ),
            session_id="s1",
            wave_id="w1",
        )
    )

    assert result.acceptance_report.passed is True
    assert "reflection" in result.active_snapshots
    assert "temporal_abstraction" in result.active_snapshots
    assert result.acceptance_report.recommendations


def test_final_wiring_can_apply_bounded_writeback_when_enabled():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(reflection=WiringLevel.ACTIVE, temporal=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="apply-model",
                feature_surface=(FeatureSignal(name="apply_context", values=(0.9,), source="adapter"),),
            ),
            memory_store=MemoryStore(),
            reflection_mode=WritebackMode.APPLY,
            session_id="s1",
            wave_id="w2",
        )
    )

    assert result.writeback_result is not None
    assert result.writeback_result.description


def test_final_wiring_can_defer_slow_writeback_into_session_post_request():
    policy = FullLearnedTemporalPolicy()
    before = policy.export_parameters()

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(reflection=WiringLevel.ACTIVE, temporal=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="deferred-apply-model",
                feature_surface=(FeatureSignal(name="deferred_context", values=(0.9,), source="adapter"),),
            ),
            memory_store=MemoryStore(),
            reflection_mode=WritebackMode.APPLY,
            temporal_policy=policy,
            session_id="s-deferred",
            wave_id="w-deferred",
            apply_slow_writeback=False,
        )
    )
    after = policy.export_parameters()

    assert result.writeback_result is None
    assert result.session_post_writeback_request is not None
    assert result.session_post_writeback_request.context_session_id == "s-deferred"
    assert after == before


def test_final_wiring_applies_reflection_temporal_prior_update_to_owner_policy():
    policy = FullLearnedTemporalPolicy()
    before = policy.export_parameters()

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(reflection=WiringLevel.ACTIVE, temporal=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="temporal-writeback-model",
                feature_surface=(FeatureSignal(name="temporal_writeback_context", values=(0.85,), source="adapter"),),
            ),
            memory_store=MemoryStore(),
            reflection_mode=WritebackMode.APPLY,
            temporal_policy=policy,
            session_id="s-temporal",
            wave_id="w-temporal",
        )
    )
    after = policy.export_parameters()

    assert result.writeback_result is not None
    assert after != before
    assert any(operation.startswith("temporal-prior:") for operation in result.writeback_result.applied_operations)
    modification_targets = {record.target for record in result.active_snapshots["credit"].value.recent_modifications}
    assert any(target.startswith("metacontroller.temporal_prior.") for target in modification_targets)


def test_final_wiring_application_prior_helper_supports_partial_credit_block():
    case_store = ApplicationCaseMemoryStore()
    rare_heavy_state = ApplicationRareHeavyState()
    credit_snapshot = CreditSnapshot(
        recent_credits=(),
        recent_modifications=(
            SelfModificationRecord(
                target="application.strategy_playbook.rules.family-transition-high-emotion",
                gate=ModificationGate.BACKGROUND,
                decision=GateDecision.BLOCK,
                old_value_hash="before",
                new_value_hash="before",
                justification="Seeded block for playbook target.",
                timestamp_ms=1,
                is_reversible=True,
            ),
        ),
        cumulative_credit_by_level=(),
        description="Seeded credit snapshot for application prior partial block.",
    )
    prior_update = ApplicationPriorUpdate(
        source_session_post_job_id="slow-loop:test",
        case_memory_updates=(
            CaseMemoryPriorUpdate(
                update_id="case-update",
                target="application.case_memory.records.family-transition-high-emotion",
                record=CaseMemoryRecord(
                    case_id="case:slow-loop:test:family-transition-high-emotion",
                    domain="stabilization_patterns",
                    problem_pattern="family-transition-high-emotion",
                    user_state_pattern="slow-loop-promoted",
                    risk_markers=("risk-medium",),
                    track_tags=("self",),
                    regime_tags=("emotional_support",),
                    intervention_ordering=("stabilize", "split_axes", "smallest_next_step"),
                    outcome_label="improved",
                    delayed_signal_count=2,
                    escalation_observed=False,
                    repair_observed=False,
                    confidence=0.78,
                    relevance_score=0.81,
                    description="Promoted case prior for helper partial-block test.",
                ),
                confidence=0.78,
                description="Apply case prior.",
            ),
        ),
        strategy_playbook_updates=(
            StrategyPlaybookPriorUpdate(
                update_id="playbook-update",
                target="application.strategy_playbook.rules.family-transition-high-emotion",
                rule=PlaybookRule(
                    rule_id="playbook:slow-loop:test",
                    problem_pattern="family-transition-high-emotion",
                    recommended_regime="emotional_support",
                    recommended_ordering=("stabilize", "split_axes", "smallest_next_step"),
                    recommended_pacing="gradual",
                    avoid_patterns=("procedure-dump-too-early",),
                    knowledge_weight_hint=0.35,
                    experience_weight_hint=0.76,
                    applicability_scope=("emotional_support",),
                    confidence=0.8,
                    description="Promoted playbook prior for helper partial-block test.",
                ),
                confidence=0.8,
                description="Apply playbook prior.",
            ),
        ),
        description="Application prior update for helper partial-block test.",
    )

    applied_operations, blocked_operations, audits, report = _apply_application_prior_writeback(
        prior_update=prior_update,
        domain_knowledge_store=ApplicationDomainKnowledgeStore(),
        case_memory_store=case_store,
        application_rare_heavy_state=rare_heavy_state,
        credit_snapshot=credit_snapshot,
        timestamp_ms=2,
        checkpoint_id="helper-checkpoint",
        apply_enabled=True,
        blocked_reason="allow",
    )

    assert applied_operations == ("application-prior:case-memory:case:slow-loop:test:family-transition-high-emotion",)
    assert blocked_operations == (
        "application-prior:block:application.strategy_playbook.rules.family-transition-high-emotion:credit-gate-block",
    )
    assert len(case_store.records) == 1
    assert not rare_heavy_state.distilled_playbook_rules
    assert report is not None
    assert report.applied_targets == ("application.case_memory.records.family-transition-high-emotion",)
    assert report.blocked_targets == ("application.strategy_playbook.rules.family-transition-high-emotion",)
    assert {audit.target for audit in audits} == {
        "application.case_memory.records.family-transition-high-emotion",
        "application.strategy_playbook.rules.family-transition-high-emotion",
    }


def test_application_prior_writeback_applies_domain_knowledge_owner_side():
    domain_store = ApplicationDomainKnowledgeStore()
    prior_update = ApplicationPriorUpdate(
        source_session_post_job_id="job:domain-knowledge",
        domain_knowledge_updates=(
            DomainKnowledgePriorUpdate(
                update_id="update:domain-knowledge",
                target="application.domain_knowledge.records.family_transition.knowledge-family-transition-1",
                record=DomainKnowledgeRecord(
                    record_id="knowledge:slow-loop:job:knowledge-family-transition-1:1",
                    domain="family_transition",
                    topic_tags=("family", "transition"),
                    jurisdiction_tags=("local-law-sensitive",),
                    source_type="official-guide",
                    title="Family transition promoted knowledge",
                    locator="phase1-seed",
                    summary="Promoted family transition knowledge from delayed evidence.",
                    snippet="Keep local specifics explicit before conclusions.",
                    freshness_label="session-post-promoted",
                    confidence=0.83,
                    evidence_strength="medium",
                ),
                confidence=0.81,
                description="Promote domain knowledge from delayed evidence.",
                source_kind=KnowledgeSourceKind.EXTERNAL_IMPORT,
                citation_ids=("knowledge:slow-loop:job:knowledge-family-transition-1:1:primary",),
            ),
        ),
        description="Application prior update with domain knowledge.",
    )

    operations, blocks, audits, report = _apply_application_prior_writeback(
        prior_update=prior_update,
        domain_knowledge_store=domain_store,
        case_memory_store=ApplicationCaseMemoryStore(),
        application_rare_heavy_state=ApplicationRareHeavyState(),
        credit_snapshot=None,
        timestamp_ms=11,
        checkpoint_id="checkpoint:knowledge",
        apply_enabled=True,
        blocked_reason="allow",
    )

    assert report is not None
    assert not blocks
    assert any(op.startswith("application-prior:domain-knowledge:") for op in operations)
    assert any(record.record_id == "knowledge:slow-loop:job:knowledge-family-transition-1:1" for record in domain_store.records)
    assert audits


def test_application_prior_writeback_blocks_domain_knowledge_when_review_not_approved():
    domain_store = ApplicationDomainKnowledgeStore()
    prior_update = ApplicationPriorUpdate(
        source_session_post_job_id="job:domain-knowledge-shadow",
        domain_knowledge_updates=(
            DomainKnowledgePriorUpdate(
                update_id="update:domain-knowledge-shadow",
                target="application.domain_knowledge.records.test.shadow",
                record=DomainKnowledgeRecord(
                    record_id="knowledge:shadow:1",
                    domain="family_transition",
                    topic_tags=("family",),
                    jurisdiction_tags=("general",),
                    source_type="internal-guide",
                    title="Shadow",
                    locator="phase1-seed",
                    summary="Shadow record",
                    snippet="Shadow",
                    freshness_label="shadow",
                    confidence=0.5,
                    evidence_strength="low",
                ),
                confidence=0.5,
                description="Shadow domain knowledge prior.",
                source_kind=KnowledgeSourceKind.CONVERSATION,
                citation_ids=("c-1",),
                review_status=KnowledgeReviewStatus.SHADOW,
            ),
        ),
        description="Shadow knowledge prior update.",
    )
    operations, blocks, audits, report = _apply_application_prior_writeback(
        prior_update=prior_update,
        domain_knowledge_store=domain_store,
        case_memory_store=ApplicationCaseMemoryStore(),
        application_rare_heavy_state=ApplicationRareHeavyState(),
        credit_snapshot=None,
        timestamp_ms=11,
        checkpoint_id="checkpoint:knowledge-shadow",
        apply_enabled=True,
        retrieval_apply_enabled=True,
        blocked_reason="allow",
    )
    assert report is not None
    assert "application.domain_knowledge.records.test.shadow" in report.blocked_targets
    assert not any(record.record_id == "knowledge:shadow:1" for record in domain_store.records)
    assert audits


def test_application_prior_writeback_blocks_conversation_domain_knowledge_when_citation_missing():
    domain_store = ApplicationDomainKnowledgeStore()
    prior_update = ApplicationPriorUpdate(
        source_session_post_job_id="job:domain-knowledge-no-cite",
        domain_knowledge_updates=(
            DomainKnowledgePriorUpdate(
                update_id="update:domain-knowledge-no-cite",
                target="application.domain_knowledge.records.test.no-cite",
                record=DomainKnowledgeRecord(
                    record_id="knowledge:no-cite:1",
                    domain="family_transition",
                    topic_tags=("family",),
                    jurisdiction_tags=("general",),
                    source_type="internal-guide",
                    title="No cite",
                    locator="phase1-seed",
                    summary="No cite",
                    snippet="No cite",
                    freshness_label="test",
                    confidence=0.7,
                    evidence_strength="medium",
                ),
                confidence=0.7,
                description="Conversation prior without citation ids.",
                source_kind=KnowledgeSourceKind.CONVERSATION,
                citation_ids=(),
                review_status=KnowledgeReviewStatus.APPROVED,
            ),
        ),
        description="Conversation knowledge prior missing citations.",
    )
    operations, blocks, audits, report = _apply_application_prior_writeback(
        prior_update=prior_update,
        domain_knowledge_store=domain_store,
        case_memory_store=ApplicationCaseMemoryStore(),
        application_rare_heavy_state=ApplicationRareHeavyState(),
        credit_snapshot=None,
        timestamp_ms=12,
        checkpoint_id="checkpoint:knowledge-no-cite",
        apply_enabled=True,
        retrieval_apply_enabled=True,
        blocked_reason="allow",
    )
    assert report is not None
    assert report.blocked_targets
    assert not any(record.record_id == "knowledge:no-cite:1" for record in domain_store.records)


def test_application_prior_writeback_captures_domain_knowledge_pre_checkpoint_when_requested():
    domain_store = ApplicationDomainKnowledgeStore()
    captured: list[DomainKnowledgeCheckpoint] = []
    prior_update = ApplicationPriorUpdate(
        source_session_post_job_id="job:domain-knowledge-checkpoint",
        domain_knowledge_updates=(
            DomainKnowledgePriorUpdate(
                update_id="update:domain-knowledge-checkpoint",
                target="application.domain_knowledge.records.test.checkpoint",
                record=DomainKnowledgeRecord(
                    record_id="knowledge:checkpoint:1",
                    domain="family_transition",
                    topic_tags=("family",),
                    jurisdiction_tags=("general",),
                    source_type="official-guide",
                    title="Checkpointed",
                    locator="phase1-seed",
                    summary="Checkpointed import",
                    snippet="Checkpointed import",
                    freshness_label="import",
                    confidence=0.77,
                    evidence_strength="medium",
                ),
                confidence=0.76,
                description="Import with pre-checkpoint capture.",
                source_kind=KnowledgeSourceKind.EXTERNAL_IMPORT,
                citation_ids=("knowledge:checkpoint:1:primary",),
            ),
        ),
        description="Checkpoint capture test.",
    )
    _operations, _blocks, _audits, report = _apply_application_prior_writeback(
        prior_update=prior_update,
        domain_knowledge_store=domain_store,
        case_memory_store=ApplicationCaseMemoryStore(),
        application_rare_heavy_state=ApplicationRareHeavyState(),
        credit_snapshot=None,
        timestamp_ms=13,
        checkpoint_id="checkpoint:knowledge-pre",
        apply_enabled=True,
        retrieval_apply_enabled=True,
        blocked_reason="allow",
        domain_knowledge_pre_checkpoint_out=captured,
    )
    assert report is not None
    assert captured
    pre = captured[0]
    assert any(record.record_id == "knowledge:checkpoint:1" for record in domain_store.records)
    domain_store.restore_checkpoint(pre)
    assert not any(record.record_id == "knowledge:checkpoint:1" for record in domain_store.records)


def test_apply_knowledge_review_decisions_skips_non_approved():
    external = (
        ExternalKnowledgeCandidate(
            candidate_id="ext-1",
            source_label="manual-entry",
            domain="career_decision",
            topic_tags=("career",),
            jurisdiction_tags=("general",),
            source_type="reviewed-article",
            title="Career framing",
            locator="manual-1",
            summary="Frame trade-offs explicitly.",
            snippet="Trade-offs",
            freshness_label="current",
            confidence=0.7,
            evidence_strength="medium",
        ),
    )
    decisions = (
        KnowledgeReviewDecision(
            candidate_id="ext-1",
            review_status=KnowledgeReviewStatus.REJECTED,
            reviewer_id="human",
            confidence=0.0,
            note="Low quality",
        ),
    )
    reviewed = apply_knowledge_review_decisions(candidates=external, decisions=decisions)
    assert reviewed == ()


def test_final_wiring_can_apply_bounded_writeback_from_shadow_reflection():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(reflection=WiringLevel.SHADOW, temporal=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="shadow-apply-model",
                feature_surface=(FeatureSignal(name="shadow_apply_context", values=(0.7,), source="adapter"),),
            ),
            memory_store=MemoryStore(),
            reflection_mode=WritebackMode.APPLY,
            session_id="s1",
            wave_id="w-shadow-apply",
        )
    )

    assert "reflection" in result.shadow_snapshots
    assert result.writeback_result is not None
    assert result.writeback_source == "shadow"


def test_session_post_writeback_keeps_memory_consolidation_when_structure_is_judge_blocked():
    from volvence_zero.evaluation.backbone import EvaluationReport
    from volvence_zero.integration import SessionPostWritebackRequest, apply_session_post_writeback_request
    from volvence_zero.reflection import (
        ConsolidationScore,
        MemoryConsolidation,
        PolicyConsolidation,
        ReflectionSnapshot,
    )

    memory_store = build_default_memory_store(latent_dim=4)
    promoted = memory_store.write(
        MemoryWriteRequest(
            content="repair this durable lesson",
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            tags=("repair",),
            strength=0.9,
        ),
        timestamp_ms=5,
    )
    request = SessionPostWritebackRequest(
        context_session_id="judge-blocked-memory-session",
        source_wave_id="wave-4",
        session_report=EvaluationReport(
            report_id="report-1",
            report_type="session",
            timestamp_ms=10,
            session_ids=("judge-blocked-memory-session",),
            scores_by_family=(),
            alerts=(),
            trends=(),
            recommendations=(),
            description="minimal session report",
        ),
        reflection_snapshot=ReflectionSnapshot(
            memory_consolidation=MemoryConsolidation(
                new_durable_entries=(),
                promoted_entries=(promoted.entry_id,),
                decayed_entries=(),
                beliefs_updated=("reinforce:self:repair",),
            ),
            policy_consolidation=PolicyConsolidation(
                controller_updates=(),
                strategy_priors_updated=(),
                regime_effectiveness_updated=(),
            ),
            consolidation_score=ConsolidationScore(
                promotion_score=0.7,
                decay_score=0.2,
                threshold_delta=0.0,
                strategy_gain=0.2,
                regime_effectiveness_gain=0.1,
                confidence=0.8,
                description="judge-blocked structure but safe memory apply",
            ),
            interaction_trace_summary="trace",
            tensions_identified=("support-tension",),
            lessons_extracted=("repair lesson",),
            writeback_mode=WritebackMode.APPLY.value,
            review_required=False,
            description="reflection snapshot",
        ),
        credit_snapshot=None,
        evolution_judgement=None,
        cross_session_verdict="hold",
        writeback_source="shadow",
        reflection_apply_enabled=True,
        structural_writeback_allowed=False,
        checkpoint_id="judge-blocked-memory-checkpoint",
        description="judge-blocked memory request",
    )

    writeback_result, temporal_audits = apply_session_post_writeback_request(
        request=request,
        memory_store=memory_store,
        temporal_policy=FullLearnedTemporalPolicy(),
        regime_module=None,
    )

    assert writeback_result is not None
    assert any(operation.startswith("tower-consolidation:") for operation in writeback_result.applied_operations)
    assert "evolution-judge-block:structural-only" in writeback_result.blocked_operations
    assert temporal_audits == ()
    metrics = dict(memory_store.snapshot(retrieved_entries=()).lifecycle_metrics)
    assert metrics["tower_consolidation_count"] >= 1.0


def test_final_wiring_merges_joint_kernel_scores_into_published_evaluation():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(reflection=WiringLevel.ACTIVE, temporal=WiringLevel.ACTIVE),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="joint-kernel-model",
                feature_surface=(FeatureSignal(name="joint_kernel_context", values=(0.75,), source="adapter"),),
            ),
            memory_store=MemoryStore(),
            joint_loop_result=ScheduledJointLoopResult(
                turn_index=1,
                schedule_action="full-cycle",
                cycle_report=None,
                kernel_scores=(
                    EvaluationScore(
                        family="abstraction",
                        metric_name="abstract_action_usefulness",
                        value=0.74,
                        confidence=0.6,
                        evidence="Injected kernel evidence for test.",
                    ),
                ),
                ssl_prediction_loss=0.0,
                ssl_kl_loss=0.0,
                metacontroller_state=None,
                cms_description="test cms",
                owner_path="test-joint-loop",
                schedule_telemetry=(
                    ("ssl_interval", 1),
                    ("rl_interval", 1),
                    ("pe_pressure_x1000", 620),
                    ("family_stability_x1000", 710),
                    ("rollback_risk_x1000", 180),
                    ("transition_pressure_x1000", 330),
                    ("substrate_pressure_x1000", 410),
                    ("rare_heavy_pressure_x1000", 520),
                    ("rl_batch_target", 2),
                    ("pending_batch_count", 1),
                ),
                description="scheduled result for final wiring test",
            ),
            session_id="s-kernel",
            wave_id="w-kernel",
        )
    )

    turn_scores = {score.metric_name: score for score in result.active_snapshots["evaluation"].value.turn_scores}
    assert "abstract_action_usefulness" in turn_scores
    assert "scheduler_pe_pressure" in turn_scores
    assert "scheduler_discipline" in turn_scores
    assert turn_scores["abstract_action_usefulness"].value == 0.74
    credit_events = {record.source_event for record in result.active_snapshots["credit"].value.recent_credits}
    assert "evaluation:abstract_action_usefulness" in credit_events


def test_final_wiring_can_seed_memory_query_from_previous_wave_snapshots():
    memory_store = MemoryStore()
    memory_store.write(
        request=MemoryWriteRequest(
            content="repair_controller maintain a calm supportive tone while planning next steps",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
            tags=("repair", "support"),
        ),
        timestamp_ms=1,
    )
    prior_temporal = Snapshot(
        slot_name="temporal_abstraction",
        owner="TemporalModule",
        version=1,
        timestamp_ms=1,
        value=TemporalAbstractionSnapshot(
            controller_state=ControllerState(
                code=(0.2, 0.8, 0.4),
                code_dim=3,
                switch_gate=0.7,
                is_switching=True,
                steps_since_switch=1,
            ),
            active_abstract_action="repair_controller",
            controller_params_hash="hash",
            description="prior temporal",
        ),
    )
    prior_dual_track = Snapshot(
        slot_name="dual_track",
        owner="DualTrackModule",
        version=1,
        timestamp_ms=1,
        value=DualTrackSnapshot(
            world_track=TrackState(
                track=Track.WORLD,
                active_goals=("planning next steps",),
                recent_credits=(),
                controller_code=(0.6, 0.4),
                tension_level=0.4,
            ),
            self_track=TrackState(
                track=Track.SELF,
                active_goals=("maintain a calm supportive tone",),
                recent_credits=(),
                controller_code=(0.3, 0.7),
                tension_level=0.6,
            ),
            cross_track_tension=0.2,
            description="prior dual track",
        ),
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="seeded-model",
                feature_surface=(FeatureSignal(name="neutral_context", values=(0.1,), source="adapter"),),
            ),
            memory_store=memory_store,
            upstream_snapshots={
                "temporal_abstraction": prior_temporal,
                "dual_track": prior_dual_track,
            },
            session_id="s1",
            wave_id="w-seeded",
        )
    )

    retrieved_contents = tuple(entry.content for entry in result.active_snapshots["memory"].value.retrieved_entries)
    assert "repair_controller maintain a calm supportive tone while planning next steps" in retrieved_contents


def test_final_wiring_exposes_prediction_error_and_reflection_promotion_fields():
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="prediction-model",
                feature_surface=(FeatureSignal(name="prediction_context", values=(0.6,), source="adapter"),),
            ),
            session_id="prediction-session",
            wave_id="wave-1",
        )
    )

    assert result.prediction_error_snapshot is not None
    assert isinstance(result.reflection_promotion_eligible, bool)
    assert isinstance(result.reflection_promotion_reason, str)
    assert "prediction_error" in result.active_snapshots
    turn_metrics = {score.metric_name for score in result.active_snapshots["evaluation"].value.turn_scores}
    assert "prediction_error_bootstrap" in turn_metrics or "prediction_error_magnitude" in turn_metrics


def test_final_wiring_prediction_error_metrics_remain_owner_readouts_on_second_turn():
    prediction_module = PredictionErrorModule()
    first = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="prediction-readout-model",
                feature_surface=(FeatureSignal(name="prediction_context", values=(0.6,), source="adapter"),),
            ),
            prediction_module=prediction_module,
            session_id="prediction-session",
            wave_id="wave-1",
        )
    )
    second = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="prediction-readout-model",
                feature_surface=(
                    FeatureSignal(name="semantic_task_pull", values=(0.9,), source="adapter"),
                    FeatureSignal(name="semantic_support_pull", values=(0.8,), source="adapter"),
                    FeatureSignal(name="semantic_repair_pull", values=(0.7,), source="adapter"),
                    FeatureSignal(name="semantic_exploration_pull", values=(0.5,), source="adapter"),
                    FeatureSignal(name="semantic_directive_pull", values=(0.8,), source="adapter"),
                ),
            ),
            prediction_module=prediction_module,
            session_id="prediction-session",
            wave_id="wave-2",
        )
    )

    del first
    turn_scores = {score.metric_name: score for score in second.active_snapshots["evaluation"].value.turn_scores}
    assert "prediction_error_magnitude" in turn_scores
    assert "prediction_error_reward" in turn_scores
    assert "predictive_accuracy" in turn_scores
    assert "PE-owner" in turn_scores["prediction_error_magnitude"].evidence
    assert "prediction_confidence" in turn_scores["predictive_accuracy"].evidence


def test_reused_prediction_module_receives_current_action_context():
    prediction_module = PredictionErrorModule()
    environment_event = EnvironmentEvent(
        event_id="env:event:prediction-context",
        event_kind=EnvironmentEventKind.USER_INPUT,
        trigger_kind="user-message",
        frame=EnvironmentFrame(
            actor=EnvironmentActorRef(actor_id="primary"),
            active_speaker_id="primary",
            addressee_ids=("self",),
            subject_ids=("primary",),
            audience_ids=("self",),
        ),
        scene_id="scene:prediction-context",
        timestamp_ms=10,
        provenance="test",
    )

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="prediction-context-model",
                feature_surface=(FeatureSignal(name="prediction_context", values=(0.6,), source="adapter"),),
            ),
            prediction_module=prediction_module,
            environment_event=environment_event,
            environment_outcome_id="tool:context:outcome",
            session_id="prediction-context-session",
            wave_id="wave-context",
        )
    )

    prediction_snapshot = result.active_snapshots["prediction_error"].value
    assert prediction_snapshot.action_context.environment_event_id == "env:event:prediction-context"
    assert prediction_snapshot.action_context.environment_outcome_id == "tool:context:outcome"


# ---------------------------------------------------------------------------
# Phase 4 W10.1 — Reflection promotion evaluation
# ---------------------------------------------------------------------------

def test_reflection_accuracy_populated_in_evaluation_snapshot():
    """Verify reflection_accuracy is populated in EvaluationSnapshot
    during run_final_wiring_turn when a ReflectionModule is present."""
    from volvence_zero.evaluation import EvaluationSnapshot

    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="ref-acc-model",
                feature_surface=(FeatureSignal(name="ctx", values=(0.5,), source="test"),),
            ),
            session_id="ref-acc-session",
            wave_id="w1",
        )
    )

    eval_snap = result.active_snapshots["evaluation"].value
    assert isinstance(eval_snap, EvaluationSnapshot)
    assert isinstance(eval_snap.reflection_accuracy, float)
    assert eval_snap.reflection_accuracy >= 0.0


def test_reflection_promotion_eligible_rejects_insufficient_data():
    """Promotion should be rejected when too few proposal outcomes exist."""
    from volvence_zero.evaluation import EvaluationSnapshot
    from volvence_zero.integration import reflection_promotion_eligible
    from volvence_zero.reflection import ReflectionEngine, WritebackMode

    engine = ReflectionEngine(writeback_mode=WritebackMode.PROPOSAL_ONLY)
    eval_snap = EvaluationSnapshot(
        turn_scores=(),
        session_scores=(),
        alerts=(),
        reflection_accuracy=0.0,
        description="empty",
    )

    eligible, reason = reflection_promotion_eligible(
        evaluation_snapshot=eval_snap,
        reflection_engine=engine,
    )
    assert eligible is False
    assert "insufficient" in reason


def test_reflection_promotion_eligible_accepts_high_accuracy():
    """Promotion should be accepted when accuracy is high enough."""
    from volvence_zero.evaluation import EvaluationSnapshot
    from volvence_zero.integration import reflection_promotion_eligible
    from volvence_zero.reflection import ReflectionEngine, WritebackMode, ProposalOutcomeEntry

    engine = ReflectionEngine(writeback_mode=WritebackMode.PROPOSAL_ONLY)
    for i in range(8):
        engine._proposal_outcome_ledger.append(
            ProposalOutcomeEntry(
                bundle_scope="single_family",
                proposal_types=("merge",),
                bundle_confidence=0.8,
                pre_metric_snapshot=(("stability", 0.5),),
                post_metric_snapshot=(("stability", 0.6),),
                metric_delta=0.1,
                success=True,
            )
        )

    eval_snap = EvaluationSnapshot(
        turn_scores=(),
        session_scores=(),
        alerts=(),
        reflection_accuracy=engine.proposal_success_rate,
        description="high accuracy",
    )

    eligible, reason = reflection_promotion_eligible(
        evaluation_snapshot=eval_snap,
        reflection_engine=engine,
    )
    assert eligible is True
    assert "eligible" in reason
