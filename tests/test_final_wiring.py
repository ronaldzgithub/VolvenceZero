from __future__ import annotations

import asyncio

from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    ApplicationPriorUpdate,
    ApplicationRareHeavyCheckpoint,
    ApplicationRareHeavyState,
    BoundaryDecision,
    BoundaryPolicySnapshot,
    CaseMemoryPriorUpdate,
    CaseMemorySnapshot,
    CaseMemoryRecord,
    ExperienceFastPriorActionBias,
    ExperienceFastPriorFamilyBias,
    ExperienceFastPriorSnapshot,
    PlaybookRule,
    ProfessionalScope,
    ResponseMode,
    RiskBand,
    StrategyPlaybookPriorUpdate,
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
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
)
from volvence_zero.temporal import ControllerState, FullLearnedTemporalPolicy, TemporalAbstractionSnapshot
from volvence_zero.dual_track import DualTrackSnapshot, TrackState


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
    assert "case_memory" not in result.active_snapshots
    assert "strategy_playbook" not in result.active_snapshots
    assert "case_memory" in result.acceptance_report.disabled_slots
    assert "strategy_playbook" in result.acceptance_report.disabled_slots
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


def test_final_wiring_exposes_shadow_experience_fast_prior_contract():
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

    assert "experience_fast_prior" in result.shadow_snapshots
    experience_fast_prior = result.shadow_snapshots["experience_fast_prior"].value
    assert experience_fast_prior.prior_strength == 0.0
    assert experience_fast_prior.source_attribution_ids == ()


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


def test_final_wiring_retrieval_mix_absorbs_rare_heavy_playbook_prior():
    baseline = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="retrieval-mix-baseline",
                feature_surface=(
                    FeatureSignal(name="semantic_support_pull", values=(0.78,), source="adapter"),
                    FeatureSignal(name="semantic_repair_pull", values=(0.56,), source="adapter"),
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
                    problem_pattern="family-transition-high-emotion",
                    recommended_regime="repair_and_deescalation",
                    recommended_ordering=("stabilize", "split_axes", "smallest_next_step"),
                    recommended_pacing="gradual",
                    avoid_patterns=("procedure-dump-too-early",),
                    knowledge_weight_hint=0.22,
                    experience_weight_hint=0.82,
                    applicability_scope=("repair_and_deescalation",),
                    confidence=0.86,
                    description="Distilled playbook prior for support-first retrieval mix.",
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
                    FeatureSignal(name="semantic_support_pull", values=(0.78,), source="adapter"),
                    FeatureSignal(name="semantic_repair_pull", values=(0.56,), source="adapter"),
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
