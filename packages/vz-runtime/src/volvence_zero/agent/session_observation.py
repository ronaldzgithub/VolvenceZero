"""Substrate adapter / training trace / turn-result composer mixin.

Debt #9 wave 1 split: this mixin owns the leaf observation surface --
methods that translate the raw runtime snapshot graph into the
typed dataclasses the public API surfaces, plus the substrate
adapter factory and training-trace assembly helpers used by
``run_turn``. The biggest method here is ``_to_turn_result``
(~440 lines), which assembles ``AgentTurnResult`` from the post-turn
``FinalIntegrationResult`` plus the joint-loop / rare-heavy / online-
fast results.

It is a pure ``class`` with no ``__init__`` and no state of its own.
All instance attributes it reads (``self._substrate_adapter_factory``,
``self._default_residual_runtime``, ``self._previous_substrate_snapshot``,
``self._session_id``, ``self._turn_index``, ``self._config``,
``self._dialogue_pe_continued_evidence_enabled``,
``self._dialogue_commitment_outcome_evidence_enabled``,
``self._dialogue_trace_store``, ``self._evaluation_backbone``,
``self._response_synthesizer``) are owned by
``AgentSessionRunner.__init__``.

Cross-mixin call surface:
``_to_turn_result`` calls ``self._publish_session_post_snapshot``,
``self._publish_experience_consolidation_snapshot``, and
``self._publish_experience_fast_prior_snapshot`` -- all live in
``SessionWritebackPhaseMixin`` and resolve via standard MRO.
``AgentTurnResult``, ``OnlineFastSubstrateTurnResult``, and
``RareHeavyTurnResult`` live in ``session.py`` and are imported
lazily inside the method that constructs them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from volvence_zero.agent.dialogue_outcome_producers import (
    commitment_outcome_evidence_from_commitment,
    pe_continued_evidence_from_prediction_error,
)
from volvence_zero.agent.response import ResponseContext
from volvence_zero.agent.session_helpers import (
    repair_expression_advisory_from_snapshots as _repair_expression_advisory_from_snapshots,
)
from volvence_zero.agent.session_post_slow_loop import SessionPostSlowLoopQueueState
from volvence_zero.application.runtime import (
    BoundaryPolicySnapshot,
    CaseMemorySnapshot,
    DomainKnowledgeSnapshot,
    ResponseAssemblySnapshot,
    StrategyPlaybookSnapshot,
)
from volvence_zero.credit.gate import (
    derive_dialogue_outcome_credit_records,
    extend_credit_snapshot,
)
from volvence_zero.dialogue_trace import DialogueOutcomeEvidence
from volvence_zero.environment import EnvironmentEvent
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.integration import FinalIntegrationResult
from volvence_zero.joint_loop import ScheduledJointLoopResult
from volvence_zero.memory import MemorySnapshot
from volvence_zero.planning import ImaginationResult, imagine
from volvence_zero.reflection import ReflectionSnapshot, WritebackResult
from volvence_zero.regime import RegimeSnapshot
from volvence_zero.runtime import Snapshot
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    MultiPartyIdentitySnapshot,
)
from volvence_zero.substrate import (
    OpenWeightResidualStreamSubstrateAdapter,
    SubstrateAdapter,
    SubstrateSnapshot,
    TraceStep,
    TrainingTrace,
    build_training_trace,
)
from volvence_zero.temporal import TemporalAbstractionSnapshot


if TYPE_CHECKING:
    from volvence_zero.agent.session import (
        AgentTurnResult,
        OnlineFastSubstrateTurnResult,
        RareHeavyTurnResult,
    )


class SessionObservationMixin:
    """Methods that build the substrate adapter / training trace / turn result.

    See module docstring for the full list of ``self._*`` attributes
    this mixin assumes ``AgentSessionRunner.__init__`` has set, and
    for the cross-mixin callsites that resolve via MRO.
    """

    def _build_substrate_adapter(self, *, user_input: str) -> SubstrateAdapter:
        if self._substrate_adapter_factory is not None:
            return self._substrate_adapter_factory(user_input, self._turn_index)
        return OpenWeightResidualStreamSubstrateAdapter(
            runtime=self._default_residual_runtime,
            default_source_text=user_input,
        )

    def _build_training_trace_from_substrate(self, *, user_input: str) -> TrainingTrace:
        """Build a training trace from real substrate data when available.

        When a previous turn produced a real substrate snapshot, construct
        the trace from its residual sequence.  Otherwise fall back to the
        simulated ``build_training_trace``.
        """
        prev = self._previous_substrate_snapshot
        if prev is None or not prev.residual_sequence:
            return build_training_trace(
                trace_id=f"{self._session_id}:joint:{self._turn_index}",
                source_text=user_input,
            )
        steps = tuple(
            TraceStep(
                step=rs.step,
                token=rs.token,
                feature_surface=rs.feature_surface,
                residual_activations=rs.residual_activations,
            )
            for rs in prev.residual_sequence
        )
        return TrainingTrace(
            trace_id=f"{self._session_id}:real:{self._turn_index}",
            source_text=user_input,
            steps=steps,
        )

    def _to_turn_result(
        self,
        *,
        user_input: str,
        wave_id: str,
        environment_event: EnvironmentEvent,
        integration_result: FinalIntegrationResult,
        joint_result: ScheduledJointLoopResult,
        imagination_result: ImaginationResult | None = None,
        online_fast_substrate_result: "OnlineFastSubstrateTurnResult | None" = None,
        rare_heavy_result: "RareHeavyTurnResult | None" = None,
        deferred_writeback_result: WritebackResult | None = None,
        queue_state: SessionPostSlowLoopQueueState | None = None,
    ) -> "AgentTurnResult":
        from volvence_zero.agent.session import AgentTurnResult

        active_regime = None
        regime_snapshot = integration_result.active_snapshots.get("regime") or integration_result.shadow_snapshots.get(
            "regime"
        )
        if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot):
            active_regime = regime_snapshot.value.active_regime.regime_id
        regime_switched = bool(
            regime_snapshot is not None
            and isinstance(regime_snapshot.value, RegimeSnapshot)
            and regime_snapshot.value.previous_regime is not None
            and regime_snapshot.value.previous_regime.regime_id != active_regime
        )

        active_abstract_action = None
        temporal_switch_gate = 0.0
        temporal_is_switching = False
        metacontroller_state = integration_result.temporal_runtime_state
        temporal_snapshot = integration_result.active_snapshots.get(
            "temporal_abstraction"
        ) or integration_result.shadow_snapshots.get("temporal_abstraction")
        if temporal_snapshot is not None and isinstance(
            temporal_snapshot.value, TemporalAbstractionSnapshot
        ):
            active_abstract_action = temporal_snapshot.value.active_abstract_action
            temporal_switch_gate = temporal_snapshot.value.controller_state.switch_gate
            temporal_is_switching = temporal_snapshot.value.controller_state.is_switching

        evaluation_alerts: tuple[str, ...] = ()
        prediction_error = None
        evaluated_prediction = None
        actual_outcome = None
        next_prediction = None
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if evaluation_snapshot is not None and isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            evaluation_alerts = evaluation_snapshot.value.alerts
        if integration_result.prediction_error_snapshot is not None:
            evaluated_prediction = integration_result.prediction_error_snapshot.evaluated_prediction
            actual_outcome = integration_result.prediction_error_snapshot.actual_outcome
            next_prediction = integration_result.prediction_error_snapshot.next_prediction
            prediction_error = integration_result.prediction_error_snapshot.error

        memory_retrieval_count = 0
        nested_profile_active = False
        nested_context_reset_applied = False
        nested_context_reset_total_count = 0
        slow_to_fast_init_benefit = 0.0
        slow_to_fast_target_distance_before = 0.0
        slow_to_fast_target_distance_after = 0.0
        slow_to_fast_target_alignment_gain = 0.0
        learned_memory_primary = False
        artifact_consolidation_count = 0
        tower_consolidation_count = 0
        learned_recall_count = 0
        learned_recall_confidence = 0.0
        learned_recall_core_guided = False
        memory_tower_depth = 0
        memory_tower_alignment = 0.0
        memory_tower_profile_id = ""
        runtime_backbone_evidence_active = False
        runtime_backbone_signal_norm = 0.0
        runtime_backbone_signal_quality = 0.0
        runtime_backbone_signal_strength = 0.0
        runtime_backbone_hook_coverage = 0.0
        fast_memory_signal_norm = 0.0
        fast_memory_runtime_alignment = 0.0
        memory_snapshot = integration_result.active_snapshots.get("memory")
        if memory_snapshot is not None and isinstance(memory_snapshot.value, MemorySnapshot):
            memory_retrieval_count = len(memory_snapshot.value.retrieved_entries)
            lifecycle_metrics = dict(memory_snapshot.value.lifecycle_metrics)
            nested_profile_active = lifecycle_metrics.get("nested_profile_active", 0.0) > 0.0
            nested_context_reset_applied = lifecycle_metrics.get("last_nested_reset_applied", 0.0) > 0.0
            nested_context_reset_total_count = int(lifecycle_metrics.get("nested_context_reset_count", 0.0))
            slow_to_fast_init_benefit = lifecycle_metrics.get("slow_to_fast_init_benefit", 0.0)
            slow_to_fast_target_distance_before = lifecycle_metrics.get("slow_to_fast_target_distance_before", 0.0)
            slow_to_fast_target_distance_after = lifecycle_metrics.get("slow_to_fast_target_distance_after", 0.0)
            slow_to_fast_target_alignment_gain = lifecycle_metrics.get("slow_to_fast_target_alignment_gain", 0.0)
            learned_memory_primary = lifecycle_metrics.get("learned_memory_primary", 0.0) > 0.0
            artifact_consolidation_count = int(lifecycle_metrics.get("artifact_consolidation_count", 0.0))
            tower_consolidation_count = int(lifecycle_metrics.get("tower_consolidation_count", 0.0))
            learned_recall_count = int(lifecycle_metrics.get("learned_recall_count", 0.0))
            learned_recall_confidence = lifecycle_metrics.get("last_learned_recall_confidence", 0.0)
            learned_recall_core_guided = lifecycle_metrics.get("last_learned_recall_driver_is_core", 0.0) > 0.0
            memory_tower_depth = int(lifecycle_metrics.get("last_memory_tower_depth", 0.0))
            memory_tower_alignment = lifecycle_metrics.get("last_memory_tower_alignment", 0.0)
            runtime_backbone_signal_norm = lifecycle_metrics.get("last_runtime_backbone_signal_norm", 0.0)
            runtime_backbone_signal_quality = lifecycle_metrics.get("last_runtime_backbone_signal_quality", 0.0)
            runtime_backbone_signal_strength = lifecycle_metrics.get("last_runtime_backbone_signal_strength", 0.0)
            runtime_backbone_hook_coverage = lifecycle_metrics.get("last_runtime_backbone_hook_coverage", 0.0)
            runtime_backbone_evidence_active = (
                lifecycle_metrics.get("last_runtime_backbone_residual_stream_active", 0.0) > 0.0
                and runtime_backbone_signal_quality > 0.0
            )
            fast_memory_signal_norm = lifecycle_metrics.get("last_fast_memory_signal_norm", 0.0)
            fast_memory_runtime_alignment = lifecycle_metrics.get("last_fast_memory_runtime_alignment", 0.0)
            cms_state = memory_snapshot.value.cms_state
            if cms_state is not None and cms_state.tower_profile is not None:
                memory_tower_profile_id = cms_state.tower_profile.profile_id

        substrate_model_id = None
        substrate_runtime_origin = getattr(self._default_residual_runtime, "runtime_origin", None)
        substrate_fallback_active = bool(getattr(self._default_residual_runtime, "fallback_active", False))
        substrate_capture_source = getattr(self._default_residual_runtime, "capture_source", None)
        substrate_residual_sequence_length = 0
        substrate_snapshot = integration_result.active_snapshots.get("substrate")
        if substrate_snapshot is not None and isinstance(substrate_snapshot.value, SubstrateSnapshot):
            substrate_model_id = substrate_snapshot.value.model_id
            substrate_residual_sequence_length = len(substrate_snapshot.value.residual_sequence)

        reflection_lesson_count = 0
        reflection_tension_count = 0
        primary_reflection_lesson = None
        primary_reflection_tension = None
        reflection_snapshot = integration_result.active_snapshots.get("reflection") or integration_result.shadow_snapshots.get(
            "reflection"
        )
        if reflection_snapshot is not None and isinstance(reflection_snapshot.value, ReflectionSnapshot):
            reflection_lesson_count = len(reflection_snapshot.value.lessons_extracted)
            reflection_tension_count = len(reflection_snapshot.value.tensions_identified)
            primary_reflection_lesson = next(iter(reflection_snapshot.value.lessons_extracted), None)
            primary_reflection_tension = next(iter(reflection_snapshot.value.tensions_identified), None)

        response_assembly_snapshot = integration_result.active_snapshots.get("response_assembly")
        response_assembly = (
            response_assembly_snapshot.value
            if response_assembly_snapshot is not None and isinstance(response_assembly_snapshot.value, ResponseAssemblySnapshot)
            else None
        )
        domain_knowledge_snapshot = integration_result.active_snapshots.get("domain_knowledge")
        case_memory_snapshot = integration_result.active_snapshots.get("case_memory")
        strategy_playbook_snapshot = integration_result.active_snapshots.get("strategy_playbook")
        boundary_policy_snapshot = integration_result.active_snapshots.get("boundary_policy")
        multi_party_identity_snapshot = integration_result.active_snapshots.get(
            "multi_party_identity"
        ) or integration_result.shadow_snapshots.get("multi_party_identity")
        active_speaker_id = PRIMARY_INTERLOCUTOR_ID
        addressee_ids = (SELF_INTERLOCUTOR_ID,)
        subject_ids = (PRIMARY_INTERLOCUTOR_ID,)
        audience_ids = (SELF_INTERLOCUTOR_ID,)
        if (
            multi_party_identity_snapshot is not None
            and isinstance(multi_party_identity_snapshot.value, MultiPartyIdentitySnapshot)
        ):
            identity_scope = multi_party_identity_snapshot.value
            active_speaker_id = identity_scope.active_speaker_id
            addressee_ids = identity_scope.addressee_ids
            subject_ids = identity_scope.subject_ids
            audience_ids = identity_scope.audience_ids

        repair_advisory = _repair_expression_advisory_from_snapshots(
            integration_result.shadow_snapshots,
            active_snapshots=integration_result.active_snapshots,
        )
        # W3 SSOT: per-regime expression brief is owned by the
        # regime module (RegimeIdentity.expression_brief). The
        # synthesizer reads context.regime_expression_brief to pick
        # variant prose instead of branching on regime_id strings.
        if regime_snapshot is not None and isinstance(
            regime_snapshot.value, RegimeSnapshot
        ):
            regime_value = regime_snapshot.value
            regime_name_value = regime_value.active_regime.name
            regime_expression_brief = regime_value.active_regime.expression_brief
        else:
            from volvence_zero.regime import (
                ExpressionBrief as _ExpressionBriefDefault,
            )
            regime_name_value = "current context"
            regime_expression_brief = _ExpressionBriefDefault()
        response = self._response_synthesizer.synthesize(
            context=ResponseContext(
                regime_id=active_regime,
                regime_name=regime_name_value,
                regime_switched=regime_switched,
                abstract_action=active_abstract_action,
                alert_count=len(evaluation_alerts),
                temporal_switch_gate=temporal_switch_gate,
                temporal_is_switching=temporal_is_switching,
                reflection_lesson_count=reflection_lesson_count,
                reflection_tension_count=reflection_tension_count,
                reflection_writeback_applied=bool(
                    integration_result.writeback_result is not None
                    and integration_result.writeback_result.applied_operations
                ),
                primary_reflection_lesson=primary_reflection_lesson,
                primary_reflection_tension=primary_reflection_tension,
                joint_schedule_action=joint_result.schedule_action,
                user_input=user_input,
                active_speaker_id=active_speaker_id,
                addressee_ids=addressee_ids,
                subject_ids=subject_ids,
                audience_ids=audience_ids,
                repair_advisory=repair_advisory,
                regime_expression_brief=regime_expression_brief,
            ),
            assembly=response_assembly,
        )
        effective_writeback_result = deferred_writeback_result or integration_result.writeback_result
        effective_queue_state = queue_state or self.session_post_queue_state
        session_post_snapshot = self._publish_session_post_snapshot()
        experience_consolidation_snapshot = self._publish_experience_consolidation_snapshot()
        experience_fast_prior_snapshot = self._publish_experience_fast_prior_snapshot()
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if (
            evaluation_snapshot is not None
            and isinstance(evaluation_snapshot.value, EvaluationSnapshot)
            and (
                experience_consolidation_snapshot.value.delayed_outcome_ledger
                or experience_consolidation_snapshot.value.sequence_payoffs
            )
        ):
            enriched_evaluation = self._evaluation_backbone.record_learning_evidence(
                session_id=self.active_context_session_id,
                wave_id=wave_id,
                timestamp_ms=evaluation_snapshot.timestamp_ms + 20,
                base_snapshot=evaluation_snapshot.value,
                memory_snapshot=memory_snapshot.value if memory_snapshot is not None and isinstance(memory_snapshot.value, MemorySnapshot) else None,
                reflection_snapshot=reflection_snapshot.value if reflection_snapshot is not None else None,
                writeback_result=integration_result.writeback_result,
                joint_loop_result=joint_result,
                regime_snapshot=regime_snapshot.value if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot) else None,
                domain_knowledge_snapshot=(
                    domain_knowledge_snapshot.value
                    if domain_knowledge_snapshot is not None and isinstance(domain_knowledge_snapshot.value, DomainKnowledgeSnapshot)
                    else None
                ),
                case_memory_snapshot=(
                    case_memory_snapshot.value
                    if case_memory_snapshot is not None and isinstance(case_memory_snapshot.value, CaseMemorySnapshot)
                    else None
                ),
                strategy_playbook_snapshot=(
                    strategy_playbook_snapshot.value
                    if strategy_playbook_snapshot is not None and isinstance(strategy_playbook_snapshot.value, StrategyPlaybookSnapshot)
                    else None
                ),
                boundary_policy_snapshot=(
                    boundary_policy_snapshot.value
                    if boundary_policy_snapshot is not None and isinstance(boundary_policy_snapshot.value, BoundaryPolicySnapshot)
                    else None
                ),
                experience_fast_prior_snapshot=experience_fast_prior_snapshot.value,
                response_assembly_snapshot=response_assembly,
                delayed_outcome_ledger=experience_consolidation_snapshot.value.delayed_outcome_ledger,
                sequence_payoffs=experience_consolidation_snapshot.value.sequence_payoffs,
            )
            integration_result.active_snapshots["evaluation"] = Snapshot(
                slot_name=evaluation_snapshot.slot_name,
                owner=evaluation_snapshot.owner,
                version=evaluation_snapshot.version + 1,
                timestamp_ms=evaluation_snapshot.timestamp_ms + 20,
                value=enriched_evaluation,
            )
        active_snapshots = dict(integration_result.active_snapshots)
        shadow_snapshots = dict(integration_result.shadow_snapshots)
        if self._config.is_active("session_post_slow_loop"):
            active_snapshots["session_post_slow_loop"] = session_post_snapshot
        else:
            shadow_snapshots["session_post_slow_loop"] = session_post_snapshot
        if self._config.is_active("experience_consolidation"):
            active_snapshots["experience_consolidation"] = experience_consolidation_snapshot
        else:
            shadow_snapshots["experience_consolidation"] = experience_consolidation_snapshot
        if self._config.is_active("experience_fast_prior"):
            active_snapshots["experience_fast_prior"] = experience_fast_prior_snapshot
        else:
            shadow_snapshots["experience_fast_prior"] = experience_fast_prior_snapshot

        outcome_evidence: tuple[DialogueOutcomeEvidence, ...] = ()
        if self._dialogue_pe_continued_evidence_enabled:
            pe_snapshot = active_snapshots.get("prediction_error") or shadow_snapshots.get(
                "prediction_error"
            )
            outcome_evidence = (
                *outcome_evidence,
                *pe_continued_evidence_from_prediction_error(
                    prediction_error_snapshot=(
                        pe_snapshot.value if pe_snapshot is not None else None
                    ),
                    wave_id=wave_id,
                ),
            )
        if self._dialogue_commitment_outcome_evidence_enabled:
            commitment_snapshot = active_snapshots.get("commitment") or shadow_snapshots.get(
                "commitment"
            )
            outcome_evidence = (
                *outcome_evidence,
                *commitment_outcome_evidence_from_commitment(
                    commitment_snapshot=(
                        commitment_snapshot.value if commitment_snapshot is not None else None
                    ),
                    wave_id=wave_id,
                    current_turn_index=self._turn_index,
                ),
            )

        dialogue_trace, dialogue_outcome_resolution = self._dialogue_trace_store.record_action(
            session_id=self.active_context_session_id,
            wave_id=wave_id,
            turn_index=self._turn_index,
            environment_event=environment_event,
            active_regime=active_regime,
            active_abstract_action=active_abstract_action,
            response_text=response.text,
            response_rationale=response.rationale,
            next_prediction=next_prediction,
            evaluated_prediction=evaluated_prediction,
            actual_outcome=actual_outcome,
            prediction_error=prediction_error,
            outcome_evidence=outcome_evidence,
        )
        dialogue_trace_snapshot = self._dialogue_trace_store.snapshot()
        if outcome_evidence:
            credit_snapshot = active_snapshots.get("credit") or shadow_snapshots.get("credit")
            if credit_snapshot is not None:
                dialogue_credit_records = derive_dialogue_outcome_credit_records(
                    outcome_evidence=outcome_evidence,
                    timestamp_ms=max(
                        credit_snapshot.timestamp_ms + 1,
                        self._turn_index,
                    ),
                )
                if dialogue_credit_records:
                    extended_credit = extend_credit_snapshot(
                        credit_snapshot=credit_snapshot.value,
                        extra_records=dialogue_credit_records,
                    )
                    new_credit_snapshot = Snapshot(
                        slot_name="credit",
                        owner=credit_snapshot.owner,
                        version=credit_snapshot.version + 1,
                        timestamp_ms=max(
                            credit_snapshot.timestamp_ms + 1,
                            self._turn_index,
                        ),
                        value=extended_credit,
                    )
                    if "credit" in active_snapshots:
                        active_snapshots["credit"] = new_credit_snapshot
                    else:
                        shadow_snapshots["credit"] = new_credit_snapshot

        return AgentTurnResult(
            session_id=self.active_context_session_id,
            wave_id=wave_id,
            user_input=user_input,
            active_snapshots=active_snapshots,
            shadow_snapshots=shadow_snapshots,
            acceptance_passed=integration_result.acceptance_report.passed,
            acceptance_issues=integration_result.acceptance_report.issues,
            active_regime=active_regime,
            active_abstract_action=active_abstract_action,
            metacontroller_state=metacontroller_state,
            evaluation_alerts=evaluation_alerts,
            evaluated_prediction=evaluated_prediction,
            actual_outcome=actual_outcome,
            next_prediction=next_prediction,
            prediction_error=prediction_error,
            bounded_writeback_applied=bool(
                effective_writeback_result is not None
                and effective_writeback_result.applied_operations
            ),
            writeback_source=integration_result.writeback_source,
            writeback_operations=effective_writeback_result.applied_operations
            if effective_writeback_result is not None
            else (),
            writeback_blocks=effective_writeback_result.blocked_operations
            if effective_writeback_result is not None
            else (),
            joint_schedule_action=joint_result.schedule_action,
            joint_learning_summary=joint_result.description,
            joint_cycle_report=joint_result.cycle_report,
            default_continual_learning_surface=joint_result.default_continual_learning_surface,
            response=response,
            event_count=integration_result.event_count,
            environment_event_id=environment_event.event_id,
            environment_event_kind=environment_event.event_kind.value,
            environment_trigger_kind=environment_event.trigger_kind,
            dialogue_trace=dialogue_trace,
            dialogue_outcome_resolution=dialogue_outcome_resolution,
            dialogue_trace_snapshot=dialogue_trace_snapshot,
            active_speaker_id=active_speaker_id,
            addressee_ids=addressee_ids,
            subject_ids=subject_ids,
            audience_ids=audience_ids,
            substrate_model_id=substrate_model_id,
            substrate_runtime_origin=substrate_runtime_origin,
            substrate_fallback_active=substrate_fallback_active,
            substrate_capture_source=substrate_capture_source,
            substrate_residual_sequence_length=substrate_residual_sequence_length,
            reflection_promotion_eligible=integration_result.reflection_promotion_eligible,
            reflection_promotion_reason=integration_result.reflection_promotion_reason,
            imagination_result=imagination_result,
            rare_heavy_result=rare_heavy_result,
            evolution_judgement=integration_result.evolution_judgement,
            cross_session_verdict=integration_result.cross_session_verdict,
            nested_profile_active=nested_profile_active,
            nested_context_reset_applied=nested_context_reset_applied,
            nested_context_reset_total_count=nested_context_reset_total_count,
            slow_to_fast_init_benefit=slow_to_fast_init_benefit,
            slow_to_fast_target_distance_before=slow_to_fast_target_distance_before,
            slow_to_fast_target_distance_after=slow_to_fast_target_distance_after,
            slow_to_fast_target_alignment_gain=slow_to_fast_target_alignment_gain,
            learned_memory_primary=learned_memory_primary,
            artifact_consolidation_count=artifact_consolidation_count,
            tower_consolidation_count=tower_consolidation_count,
            learned_recall_count=learned_recall_count,
            learned_recall_confidence=learned_recall_confidence,
            learned_recall_core_guided=learned_recall_core_guided,
            memory_tower_depth=memory_tower_depth,
            memory_tower_alignment=memory_tower_alignment,
            memory_tower_profile_id=memory_tower_profile_id,
            runtime_backbone_evidence_active=runtime_backbone_evidence_active,
            runtime_backbone_signal_norm=runtime_backbone_signal_norm,
            runtime_backbone_signal_quality=runtime_backbone_signal_quality,
            runtime_backbone_signal_strength=runtime_backbone_signal_strength,
            runtime_backbone_hook_coverage=runtime_backbone_hook_coverage,
            fast_memory_signal_norm=fast_memory_signal_norm,
            fast_memory_runtime_alignment=fast_memory_runtime_alignment,
            session_post_pending_job_count=effective_queue_state.pending_job_count,
            session_post_completed_job_count=effective_queue_state.completed_job_count,
            session_post_last_completed_job_id=effective_queue_state.last_completed_job_id,
            online_fast_substrate_result=online_fast_substrate_result,
        )

    def _run_imagination(self, integration_result: FinalIntegrationResult) -> ImaginationResult | None:
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        dual_track_snapshot = integration_result.active_snapshots.get("dual_track")
        regime_snapshot = integration_result.active_snapshots.get("regime")
        if (
            evaluation_snapshot is None
            or not isinstance(evaluation_snapshot.value, EvaluationSnapshot)
            or dual_track_snapshot is None
        ):
            return None
        from volvence_zero.dual_track import DualTrackSnapshot as DTS

        if not isinstance(dual_track_snapshot.value, DTS):
            return None
        regime_value = (
            regime_snapshot.value
            if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot)
            else None
        )
        metacontroller_state = integration_result.temporal_runtime_state
        prior_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
        prior_std: tuple[float, ...] = (0.1, 0.1, 0.1)
        action_family_centroids: tuple[tuple[str, tuple[float, ...]], ...] = ()
        if metacontroller_state is not None:
            prior_mean = metacontroller_state.prior_mean or prior_mean
            prior_std = metacontroller_state.prior_std or prior_std
            action_family_centroids = tuple(
                (summary.family_id, (summary.stability, summary.switch_bias, summary.competition_score))
                for summary in metacontroller_state.action_family_summaries
                if summary.support >= 2
            )
        previous_prediction = (
            integration_result.prediction_error_snapshot.next_prediction
            if integration_result.prediction_error_snapshot is not None
            else None
        )
        return imagine(
            current_substrate=self._previous_substrate_snapshot,
            current_evaluation=evaluation_snapshot.value,
            current_dual_track=dual_track_snapshot.value,
            current_regime=regime_value,
            previous_prediction=previous_prediction,
            action_family_centroids=action_family_centroids,
            prior_mean=prior_mean,
            prior_std=prior_std,
        )


__all__ = ["SessionObservationMixin"]
