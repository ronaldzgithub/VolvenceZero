"""Rare-heavy + online-fast substrate self-modification mixin for ``AgentSessionRunner``.

Debt #9 wave 1 split: this is the largest mixin in the split. It owns
the entire training-phase surface: training-trace and substrate-batch
windowing, rare-heavy training-bundle assembly, rare-heavy pipeline
runners, pre-import replay evaluation, online-fast substrate self-mod
gating + apply + rollback, delayed-rollback evaluation evidence, and
the public-API-facing ``_maybe_apply_rare_heavy`` orchestrator that
``run_turn`` calls when the joint loop recommends a rare-heavy review.

The grouping intentionally tracks the existing ``TRAINING WRITEBACK
PHASE`` boundary that debt #8 (``joint_loop`` and runtime main chain
share owner instances) already pins via contract test. Keeping all
of this in one mixin preserves that pinning -- the writeback boundary
remains a single import target.

It is a pure ``class`` with no ``__init__`` and no state of its own.
All instance attributes it reads (``self._recent_training_traces``,
``self._recent_substrate_batches``, ``self._recent_rare_heavy_examples``,
``self._rare_heavy_trace_window``, ``self._rare_heavy_min_traces``,
``self._rare_heavy_cooldown_turns``, ``self._rare_heavy_pipeline_config``,
``self._rare_heavy_enabled``, ``self._joint_loop``,
``self._memory_store``, ``self._evaluation_backbone``,
``self._upstream_snapshots``, ``self._application_rare_heavy_state``,
``self._session_id``, ``self._turn_index``, ``self._config``,
``self._default_residual_runtime``,
``self._prediction_error_readout_only``,
``self._primary_prediction_error_dominance_enabled``,
``self._last_online_fast_import_checkpoint``,
``self._last_rare_heavy_import_checkpoint``,
``self._last_rare_heavy_turn_index``) are owned by
``AgentSessionRunner.__init__``.

Cross-mixin call surface: ``_build_rare_heavy_replay_runner`` calls
``AgentSessionRunner(...)`` to spawn an isolated replay session; the
runtime import is lazy (inside the method) to avoid a circular
import with ``session.py``. Result dataclasses
(``RareHeavyTrainingExample``, ``RareHeavyTrainingBundle``,
``RareHeavyPreImportEvaluation``, ``RareHeavyTurnResult``,
``OnlineFastSubstrateTurnResult``, ``AgentTurnResult``) live in
``session.py`` and are also imported lazily inside methods that
construct them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from volvence_zero.application.runtime import (
    ApplicationCaseCluster,
    ApplicationRareHeavyCheckpoint,
    CaseMemorySnapshot,
    DomainKnowledgeSnapshot,
    StrategyPlaybookSnapshot,
)
from volvence_zero.credit.gate import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    SelfModificationRecord,
    evaluate_gate,
    extend_credit_snapshot,
)
from volvence_zero.evaluation import (
    EvaluationScore,
    EvaluationSnapshot,
    EvolutionDecision,
)
from volvence_zero.integration import FinalIntegrationResult
from volvence_zero.joint_loop import (
    JointLoopSchedule,
    PipelineConfig,
    RareHeavyArtifact,
    SSLRLTrainingPipeline,
    ScheduledJointLoopResult,
)
from volvence_zero.memory import MemorySnapshot, MemoryStore
from volvence_zero.reflection import WritebackMode
from volvence_zero.runtime import Snapshot
from volvence_zero.substrate import (
    SubstrateSelfModSnapshot,
    SubstrateSnapshot,
    TrainingTrace,
)
from volvence_zero.temporal import (
    DualTrackRareHeavySnapshot,
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    clone_full_learned_temporal_policy,
)


if TYPE_CHECKING:
    from volvence_zero.agent.session import (
        AgentSessionRunner,
        AgentTurnResult,
        OnlineFastSubstrateTurnResult,
        RareHeavyPreImportEvaluation,
        RareHeavyTrainingBundle,
        RareHeavyTrainingExample,
        RareHeavyTurnResult,
    )


class SessionTrainingPhaseMixin:
    """Methods that own the training / rare-heavy / online-fast paths.

    See module docstring for the full list of ``self._*`` attributes
    this mixin assumes ``AgentSessionRunner.__init__`` has set, and
    for the lazy-import pattern used to avoid circular imports with
    ``session.py``.
    """

    def _record_training_trace(self, trace: TrainingTrace) -> None:
        self._recent_training_traces.append(trace)
        if len(self._recent_training_traces) > self._rare_heavy_trace_window:
            del self._recent_training_traces[:-self._rare_heavy_trace_window]

    def _record_substrate_batch(self, batch: tuple[SubstrateSnapshot, ...]) -> None:
        if not batch:
            return
        self._recent_substrate_batches.append(batch)
        if len(self._recent_substrate_batches) > self._rare_heavy_trace_window:
            del self._recent_substrate_batches[:-self._rare_heavy_trace_window]

    def _record_rare_heavy_example(
        self,
        *,
        wave_id: str,
        user_input: str,
        trace: TrainingTrace,
        substrate_batch: tuple[SubstrateSnapshot, ...],
    ) -> None:
        from volvence_zero.agent.session import RareHeavyTrainingExample

        if not substrate_batch:
            return
        self._recent_rare_heavy_examples.append(
            RareHeavyTrainingExample(
                turn_index=self._turn_index,
                wave_id=wave_id,
                source_text=user_input,
                trace=trace,
                substrate_batch=substrate_batch,
                description=(
                    f"Rare-heavy example turn={self._turn_index} wave={wave_id} "
                    f"trace_steps={len(trace.steps)} substrate_steps={len(substrate_batch)}."
                ),
            )
        )
        if len(self._recent_rare_heavy_examples) > self._rare_heavy_trace_window:
            del self._recent_rare_heavy_examples[:-self._rare_heavy_trace_window]

    def _substrate_batch_from_snapshot(self, snapshot: SubstrateSnapshot) -> tuple[SubstrateSnapshot, ...]:
        if not snapshot.residual_sequence:
            return (snapshot,)
        return tuple(
            SubstrateSnapshot(
                model_id=snapshot.model_id,
                is_frozen=snapshot.is_frozen,
                surface_kind=snapshot.surface_kind,
                token_logits=snapshot.token_logits,
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                residual_sequence=(step,),
                unavailable_fields=snapshot.unavailable_fields,
                description=f"{snapshot.description} rare-heavy-step={step.step}",
            )
            for step in snapshot.residual_sequence
        )

    def _effective_rare_heavy_pipeline_config(self) -> PipelineConfig:
        policy_n_z = self._joint_loop.temporal_policy.parameter_store.n_z
        if self._rare_heavy_pipeline_config.n_z == policy_n_z:
            return self._rare_heavy_pipeline_config
        return replace(self._rare_heavy_pipeline_config, n_z=policy_n_z)

    def _build_rare_heavy_training_bundle(self) -> "RareHeavyTrainingBundle":
        from volvence_zero.agent.session import RareHeavyTrainingBundle

        examples = tuple(self._recent_rare_heavy_examples[-self._rare_heavy_trace_window :])
        if not examples:
            return RareHeavyTrainingBundle(
                examples=(),
                trace_count=0,
                substrate_batch_count=0,
                aligned_example_count=0,
                alignment_ratio=0.0,
                mean_trace_step_count=0.0,
                mean_sequence_length=0.0,
                mean_residual_magnitude=0.0,
                description="No aligned rare-heavy training examples are available.",
            )
        trace_count = len(examples)
        substrate_batch_count = sum(1 for example in examples if example.substrate_batch)
        aligned_example_count = sum(
            1
            for example in examples
            if example.trace.steps and example.substrate_batch
        )
        alignment_ratio = aligned_example_count / max(trace_count, 1)
        mean_trace_step_count = sum(len(example.trace.steps) for example in examples) / trace_count
        flattened_substrates = tuple(
            snapshot
            for example in examples
            for snapshot in example.substrate_batch
        )
        mean_sequence_length = (
            sum(max(len(snapshot.residual_sequence), 1) for snapshot in flattened_substrates)
            / max(len(flattened_substrates), 1)
            if flattened_substrates
            else 0.0
        )
        residual_values = tuple(
            abs(value)
            for snapshot in flattened_substrates
            for activation in snapshot.residual_activations
            for value in activation.activation
        )
        mean_residual_magnitude = (
            sum(residual_values) / len(residual_values)
            if residual_values
            else 0.0
        )
        return RareHeavyTrainingBundle(
            examples=examples,
            trace_count=trace_count,
            substrate_batch_count=substrate_batch_count,
            aligned_example_count=aligned_example_count,
            alignment_ratio=alignment_ratio,
            mean_trace_step_count=mean_trace_step_count,
            mean_sequence_length=mean_sequence_length,
            mean_residual_magnitude=mean_residual_magnitude,
            description=(
                f"Rare-heavy bundle examples={trace_count} aligned={aligned_example_count} "
                f"alignment={alignment_ratio:.2f} mean_trace_steps={mean_trace_step_count:.2f} "
                f"mean_sequence_len={mean_sequence_length:.2f}."
            ),
        )

    def _clone_memory_store_for_rare_heavy(self) -> MemoryStore:
        checkpoint = self._joint_loop.memory_store.export_rare_heavy_state(
            checkpoint_id=f"{self._session_id}:rare-heavy-seed:{self._turn_index}"
        )
        source_core = self._joint_loop.memory_store.learned_core
        learned_core = source_core.clone_empty() if source_core is not None else None
        cloned_store = MemoryStore(learned_core=learned_core)
        cloned_store.import_rare_heavy_state(checkpoint)
        return cloned_store

    def _build_rare_heavy_pipeline(
        self,
        *,
        source_policy: FullLearnedTemporalPolicy,
    ) -> SSLRLTrainingPipeline:
        policy_n_z = source_policy.parameter_store.n_z
        cloned_policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(n_z=policy_n_z),
        )
        cloned_policy.apply_rare_heavy_snapshot(
            source_policy.export_rare_heavy_snapshot()
        )
        return SSLRLTrainingPipeline(
            config=self._effective_rare_heavy_pipeline_config(),
            policy=cloned_policy,
            memory_store=self._clone_memory_store_for_rare_heavy(),
            residual_runtime=(
                self._joint_loop.residual_runtime.clone_for_rare_heavy()
                if self._joint_loop.residual_runtime is not None
                else None
            ),
        )

    def _build_rare_heavy_replay_runner(
        self,
        *,
        label: str,
        artifact: RareHeavyArtifact | None = None,
    ) -> "AgentSessionRunner":
        from volvence_zero.agent.session import AgentSessionRunner

        runner = AgentSessionRunner(
            session_id=f"{self._session_id}:rare-heavy-replay:{label}",
            config=self._config,
            memory_store=self._clone_memory_store_for_rare_heavy(),
            reflection_mode=WritebackMode.PROPOSAL_ONLY,
            world_temporal_policy=clone_full_learned_temporal_policy(self._joint_loop.world_temporal_policy),
            self_temporal_policy=clone_full_learned_temporal_policy(self._joint_loop.self_temporal_policy),
            default_residual_runtime=(
                self.residual_runtime.clone_for_rare_heavy()
                if self.residual_runtime is not None
                else self._default_residual_runtime
            ),
            joint_schedule=JointLoopSchedule(ssl_interval=0, rl_interval=0),
            rare_heavy_enabled=False,
            external_prediction_error_drive=False,
            prediction_error_readout_only=self._prediction_error_readout_only,
            primary_prediction_error_dominance_enabled=self._primary_prediction_error_dominance_enabled,
        )
        runner._application_rare_heavy_state.import_rare_heavy_state(
            self._application_rare_heavy_state.export_rare_heavy_state(
                checkpoint_id=f"{runner.session_id}:application-seed"
            )
        )
        if artifact is not None:
            runner.apply_rare_heavy_artifact(
                artifact,
                checkpoint_id=f"{runner.session_id}:candidate-import",
            )
        return runner

    def _build_application_rare_heavy_checkpoint(
        self,
        *,
        artifact_id: str,
    ) -> ApplicationRareHeavyCheckpoint:
        domain_snapshot = self._upstream_snapshots.get("domain_knowledge")
        case_snapshot = self._upstream_snapshots.get("case_memory")
        playbook_snapshot = self._upstream_snapshots.get("strategy_playbook")
        domain_biases: list[tuple[str, float]] = []
        if domain_snapshot is not None and isinstance(domain_snapshot.value, DomainKnowledgeSnapshot):
            for domain in domain_snapshot.value.active_domains:
                domain_biases.append((domain, 0.72))
        case_clusters: list[ApplicationCaseCluster] = []
        if case_snapshot is not None and isinstance(case_snapshot.value, CaseMemorySnapshot):
            for index, pattern in enumerate(case_snapshot.value.active_problem_patterns[:3], start=1):
                matched_hits = tuple(hit for hit in case_snapshot.value.hits if hit.problem_pattern == pattern)
                mean_relevance = (
                    sum(hit.relevance_score for hit in matched_hits) / max(len(matched_hits), 1)
                    if matched_hits
                    else 0.5
                )
                risk_markers = matched_hits[0].risk_markers if matched_hits else ()
                case_clusters.append(
                    ApplicationCaseCluster(
                        cluster_id=f"{artifact_id}:cluster:{index}",
                        problem_pattern=pattern,
                        exemplar_count=max(len(matched_hits), 1),
                        mean_relevance=mean_relevance,
                        risk_markers=risk_markers,
                        description=f"Rare-heavy distilled cluster for pattern={pattern}.",
                    )
                )
        distilled_playbook_rules = ()
        if playbook_snapshot is not None and isinstance(playbook_snapshot.value, StrategyPlaybookSnapshot):
            distilled_playbook_rules = playbook_snapshot.value.matched_rules
        if not domain_biases and not case_clusters and not distilled_playbook_rules:
            return self._application_rare_heavy_state.export_rare_heavy_state(
                checkpoint_id=f"{artifact_id}:application"
            )
        return ApplicationRareHeavyCheckpoint(
            checkpoint_id=f"{artifact_id}:application",
            domain_template_biases=tuple(domain_biases),
            case_clusters=tuple(case_clusters),
            distilled_playbook_rules=tuple(distilled_playbook_rules),
            description=(
                f"Application rare-heavy checkpoint distilled {len(domain_biases)} domain biases, "
                f"{len(case_clusters)} case clusters, and {len(distilled_playbook_rules)} playbook rules."
            ),
        )

    @staticmethod
    def _rare_heavy_replay_score(result: "AgentTurnResult") -> float:
        evaluation_snapshot = result.active_snapshots.get("evaluation")
        if evaluation_snapshot is None or not isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            return 0.0
        turn_scores = evaluation_snapshot.value.turn_scores
        if not turn_scores:
            return 0.0
        mean_score = sum(score.value for score in turn_scores) / len(turn_scores)
        alert_penalty = min(len(result.evaluation_alerts) * 0.05, 0.20)
        acceptance_bonus = 0.05 if result.acceptance_passed else -0.05
        return max(0.0, min(1.0, mean_score - alert_penalty + acceptance_bonus))

    async def _evaluate_rare_heavy_candidate(
        self,
        *,
        artifact: RareHeavyArtifact,
        bundle: "RareHeavyTrainingBundle",
    ) -> "RareHeavyPreImportEvaluation":
        from volvence_zero.agent.session import RareHeavyPreImportEvaluation

        replay_examples = bundle.examples[-min(len(bundle.examples), 3) :]
        if not replay_examples:
            return RareHeavyPreImportEvaluation(
                accepted=False,
                case_count=0,
                baseline_mean_score=0.0,
                candidate_mean_score=0.0,
                mean_score_delta=0.0,
                worst_score_delta=0.0,
                positive_fraction=0.0,
                judgement="not-run",
                reasons=("no-replay-examples",),
                description="Rare-heavy pre-import replay skipped because no aligned examples were available.",
            )
        try:
            baseline_runner = self._build_rare_heavy_replay_runner(label="baseline")
            candidate_runner = self._build_rare_heavy_replay_runner(label="candidate", artifact=artifact)
        except (TypeError, ValueError, RuntimeError) as exc:
            return RareHeavyPreImportEvaluation(
                accepted=False,
                case_count=len(replay_examples),
                baseline_mean_score=0.0,
                candidate_mean_score=0.0,
                mean_score_delta=0.0,
                worst_score_delta=0.0,
                positive_fraction=0.0,
                judgement="runner-build-failed",
                reasons=("candidate-runner-build-failed",),
                description=f"Rare-heavy pre-import replay failed while building candidate runners: {exc}",
            )
        baseline_scores: list[float] = []
        candidate_scores: list[float] = []
        for example in replay_examples:
            baseline_result = await baseline_runner.run_turn(example.source_text)
            candidate_result = await candidate_runner.run_turn(example.source_text)
            baseline_scores.append(self._rare_heavy_replay_score(baseline_result))
            candidate_scores.append(self._rare_heavy_replay_score(candidate_result))
        deltas = tuple(
            candidate - baseline
            for baseline, candidate in zip(baseline_scores, candidate_scores, strict=True)
        )
        baseline_mean_score = sum(baseline_scores) / len(baseline_scores)
        candidate_mean_score = sum(candidate_scores) / len(candidate_scores)
        mean_score_delta = sum(deltas) / len(deltas)
        worst_score_delta = min(deltas, default=0.0)
        positive_fraction = (
            sum(1 for delta in deltas if delta > 0.0) / len(deltas)
            if deltas
            else 0.0
        )
        session_report = candidate_runner.build_current_session_report()
        judgement_label = "not-run"
        reasons: list[str] = []
        if session_report is not None:
            replay_suite = candidate_runner.evaluation_backbone.run_default_evolution_benchmark(
                timestamp_ms=max(candidate_runner.turn_index, 1) + 1,
            )
            judgement = candidate_runner.evaluation_backbone.judge_evolution_candidate(
                replay_suite_result=replay_suite,
                session_report=session_report,
            )
            judgement_label = judgement.decision.value
            if judgement.decision is EvolutionDecision.ROLLBACK:
                reasons.append("evolution-judge-rollback")
        if mean_score_delta <= 0.0:
            reasons.append("pre-import-mean-score-nonpositive")
        if worst_score_delta < -0.05:
            reasons.append("pre-import-worst-case-regressed")
        if positive_fraction < 0.5:
            reasons.append("pre-import-positive-fraction-too-low")
        accepted = not reasons
        return RareHeavyPreImportEvaluation(
            accepted=accepted,
            case_count=len(replay_examples),
            baseline_mean_score=baseline_mean_score,
            candidate_mean_score=candidate_mean_score,
            mean_score_delta=mean_score_delta,
            worst_score_delta=worst_score_delta,
            positive_fraction=positive_fraction,
            judgement=judgement_label,
            reasons=tuple(reasons),
            description=(
                f"Rare-heavy pre-import replay over {len(replay_examples)} cases produced "
                f"baseline_mean={baseline_mean_score:.3f}, candidate_mean={candidate_mean_score:.3f}, "
                f"mean_delta={mean_score_delta:.3f}, worst_delta={worst_score_delta:.3f}, "
                f"positive_fraction={positive_fraction:.3f}, judgement={judgement_label}."
            ),
        )

    def _append_online_fast_credit_audit(
        self,
        *,
        integration_result: FinalIntegrationResult,
        record: SelfModificationRecord,
    ) -> None:
        credit_snapshot = integration_result.active_snapshots.get("credit")
        if credit_snapshot is None:
            return
        extended = extend_credit_snapshot(
            credit_snapshot=credit_snapshot.value,
            extra_modifications=(record,),
        )
        integration_result.active_snapshots["credit"] = Snapshot(
            slot_name="credit",
            owner="CreditModule",
            version=credit_snapshot.version + 1,
            timestamp_ms=max(credit_snapshot.timestamp_ms + 1, self._turn_index),
            value=extended,
        )

    def _append_online_fast_evaluation_evidence(
        self,
        *,
        integration_result: FinalIntegrationResult,
        wave_id: str,
        result: "OnlineFastSubstrateTurnResult",
    ) -> None:
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if evaluation_snapshot is None or not isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            return
        enriched = self._evaluation_backbone.record_external_scores(
            session_id=self.active_context_session_id,
            wave_id=wave_id,
            timestamp_ms=max(evaluation_snapshot.timestamp_ms + 1, self._turn_index),
            base_snapshot=evaluation_snapshot.value,
            scores=(
                EvaluationScore(
                    family="learning",
                    metric_name="substrate_online_fast_proposed",
                    value=1.0 if result.recommended else 0.0,
                    confidence=0.8,
                    evidence=result.description,
                ),
                EvaluationScore(
                    family="learning",
                    metric_name="substrate_online_fast_applied",
                    value=1.0 if result.applied else 0.0,
                    confidence=0.8,
                    evidence=result.description,
                ),
                EvaluationScore(
                    family="learning",
                    metric_name="substrate_online_fast_optimizer_norm",
                    value=result.optimizer_state_norm,
                    confidence=0.7,
                    evidence=result.optimizer_state_description,
                ),
                EvaluationScore(
                    family="learning",
                    metric_name="substrate_online_fast_experimental_mode",
                    value=1.0 if result.experimental_live_mutation else 0.0,
                    confidence=0.82,
                    evidence=(
                        "Derived from session/runtime capability for experimental bounded live mutation."
                    ),
                ),
                EvaluationScore(
                    family="safety",
                    metric_name="substrate_online_fast_rollback_integrity",
                    value=1.0
                    if (not result.rollback_reason or result.rollback_applied)
                    else 0.0,
                    confidence=0.78,
                    evidence=(
                        result.description
                        if not result.rollback_reason
                        else (
                            f"rollback_reason={result.rollback_reason}, "
                            f"rollback_ops={len(result.rollback_operations)}."
                        )
                    ),
                ),
                EvaluationScore(
                    family="safety",
                    metric_name="substrate_online_fast_review_or_revert_safe",
                    value=1.0 if (not result.applied or result.rollback_applied or result.experimental_live_mutation) else 0.8,
                    confidence=0.72,
                    evidence=(
                        f"gate_decision={result.gate_decision}, experimental={result.experimental_live_mutation}, "
                        f"rollback={result.rollback_applied}."
                    ),
                ),
            ),
            description_suffix="Session owner appended online-fast substrate apply evidence.",
        )
        integration_result.active_snapshots["evaluation"] = Snapshot(
            slot_name="evaluation",
            owner="EvaluationModule",
            version=evaluation_snapshot.version + 1,
            timestamp_ms=max(evaluation_snapshot.timestamp_ms + 1, self._turn_index),
            value=enriched,
        )

    def _refresh_memory_snapshot_after_online_fast_evidence(
        self,
        *,
        integration_result: FinalIntegrationResult,
    ) -> None:
        memory_snapshot = integration_result.active_snapshots.get("memory")
        if memory_snapshot is None or not isinstance(memory_snapshot.value, MemorySnapshot):
            return
        integration_result.active_snapshots["memory"] = Snapshot(
            slot_name=memory_snapshot.slot_name,
            owner=memory_snapshot.owner,
            version=memory_snapshot.version + 1,
            timestamp_ms=max(memory_snapshot.timestamp_ms + 1, self._turn_index),
            value=self._memory_store.snapshot(
                retrieved_entries=memory_snapshot.value.retrieved_entries,
            ),
        )

    def _delayed_substrate_rollback_reasons(
        self,
        *,
        integration_result: FinalIntegrationResult,
    ) -> tuple[str, ...]:
        del integration_result
        return ()

    def _append_delayed_rollback_evaluation_evidence(
        self,
        *,
        integration_result: FinalIntegrationResult,
        reasons: tuple[str, ...],
        operations: tuple[str, ...],
    ) -> None:
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        if evaluation_snapshot is None or not isinstance(evaluation_snapshot.value, EvaluationSnapshot):
            return
        enriched = self._evaluation_backbone.record_external_scores(
            session_id=self.active_context_session_id,
            wave_id=f"wave-{self._turn_index}",
            timestamp_ms=max(evaluation_snapshot.timestamp_ms + 1, self._turn_index),
            base_snapshot=evaluation_snapshot.value,
            scores=(
                EvaluationScore(
                    family="safety",
                    metric_name="substrate_delayed_rollback_applied",
                    value=1.0 if operations else 0.0,
                    confidence=0.82,
                    evidence=(
                        f"Derived from delayed rollback reasons={reasons} "
                        f"and operation_count={len(operations)}."
                    ),
                ),
            ),
            description_suffix="Session owner appended delayed substrate rollback evidence.",
        )
        integration_result.active_snapshots["evaluation"] = Snapshot(
            slot_name="evaluation",
            owner="EvaluationModule",
            version=evaluation_snapshot.version + 1,
            timestamp_ms=max(evaluation_snapshot.timestamp_ms + 1, self._turn_index),
            value=enriched,
        )

    def _maybe_apply_delayed_substrate_rollback(
        self,
        *,
        integration_result: FinalIntegrationResult,
    ) -> tuple[str, ...]:
        reasons = tuple(self._delayed_substrate_rollback_reasons(integration_result=integration_result))
        if not reasons:
            return ()
        operations: list[str] = []
        if self._last_online_fast_import_checkpoint is not None:
            operations.extend(
                self._joint_loop.rollback_online_fast_substrate_import(
                    self._last_online_fast_import_checkpoint
                )
            )
            self._last_online_fast_import_checkpoint = None
        if self._last_rare_heavy_import_checkpoint is not None:
            operations.extend(
                self.rollback_rare_heavy_import(self._last_rare_heavy_import_checkpoint)
            )
            self._last_rare_heavy_import_checkpoint = None
        applied = tuple(operations)
        self._append_delayed_rollback_evaluation_evidence(
            integration_result=integration_result,
            reasons=reasons,
            operations=applied,
        )
        return applied

    def _maybe_apply_online_fast_substrate_self_mod(
        self,
        *,
        wave_id: str,
        joint_result: ScheduledJointLoopResult,
        integration_result: FinalIntegrationResult,
    ) -> "OnlineFastSubstrateTurnResult | None":
        from volvence_zero.agent.session import OnlineFastSubstrateTurnResult

        snapshot = integration_result.active_snapshots.get("substrate_self_mod") or integration_result.shadow_snapshots.get(
            "substrate_self_mod"
        )
        if snapshot is None or not isinstance(snapshot.value, SubstrateSelfModSnapshot):
            return None
        substrate_self_mod = snapshot.value
        if not substrate_self_mod.recommended or substrate_self_mod.checkpoint is None:
            return None
        evaluation_snapshot = integration_result.active_snapshots.get("evaluation")
        evaluation_value = (
            evaluation_snapshot.value
            if evaluation_snapshot is not None and isinstance(evaluation_snapshot.value, EvaluationSnapshot)
            else None
        )
        gate_decision = GateDecision.BLOCK
        if evaluation_value is not None:
            gate_decision = evaluate_gate(
                proposal=ModificationProposal(
                    target=substrate_self_mod.target,
                    desired_gate=ModificationGate.ONLINE,
                    old_value_hash="substrate.online_fast:pre",
                    new_value_hash=substrate_self_mod.checkpoint_hash,
                    justification=substrate_self_mod.description,
                    validation_delta=substrate_self_mod.proposal_readiness - 0.5,
                    capacity_cost=min(substrate_self_mod.parameter_change_rate, 1.0),
                    rollback_evidence=substrate_self_mod.checkpoint.checkpoint_id,
                ),
                evaluation_snapshot=evaluation_value,
            )
        if not joint_result.substrate_online_fast_due:
            return OnlineFastSubstrateTurnResult(
                recommended=True,
                applied=False,
                gate_decision="schedule-not-due",
                applied_operations=(),
                blocked_operations=("online-fast:schedule-not-due",),
                rollback_operations=(),
                parameter_change_rate=substrate_self_mod.parameter_change_rate,
                optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
                source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
                optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
                fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
                description="Online-fast substrate self-mod proposal was present, but schedule was not due.",
                experimental_live_mutation=self.residual_runtime.supports_live_substrate_mutation,
            )
        if not self.residual_runtime.supports_live_substrate_mutation:
            self._append_online_fast_credit_audit(
                integration_result=integration_result,
                record=SelfModificationRecord(
                    target=substrate_self_mod.target,
                    gate=ModificationGate.ONLINE,
                    decision=GateDecision.BLOCK,
                    old_value_hash="substrate.online_fast:pre",
                    new_value_hash=substrate_self_mod.checkpoint_hash,
                    justification=(
                        "Frozen-substrate doctrine kept the online-fast substrate proposal in review-only mode. "
                        f"{substrate_self_mod.description}"
                    ),
                    timestamp_ms=self._turn_index,
                    is_reversible=True,
                    checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                    lineage_hash=substrate_self_mod.checkpoint.fast_state_hash,
                    proposal_hash=substrate_self_mod.checkpoint_hash,
                ),
            )
            blocked_result = OnlineFastSubstrateTurnResult(
                recommended=True,
                applied=False,
                gate_decision="frozen-substrate-doctrine",
                applied_operations=(),
                blocked_operations=("online-fast:frozen-substrate-doctrine",),
                rollback_operations=(),
                parameter_change_rate=substrate_self_mod.parameter_change_rate,
                optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
                source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
                optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
                fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
                description=(
                    "Online-fast substrate self-mod proposal stayed review-only because the live runtime "
                    "is operating under the frozen-substrate doctrine."
                ),
                experimental_live_mutation=False,
            )
            self._append_online_fast_evaluation_evidence(
                integration_result=integration_result,
                wave_id=wave_id,
                result=blocked_result,
            )
            if substrate_self_mod.checkpoint.fast_memory_signal:
                self._memory_store.observe_fast_memory_signal(
                    signal=substrate_self_mod.checkpoint.fast_memory_signal,
                    timestamp_ms=max(self._turn_index, 1),
                )
                self._refresh_memory_snapshot_after_online_fast_evidence(
                    integration_result=integration_result,
                )
            return blocked_result
        if gate_decision is GateDecision.BLOCK:
            self._append_online_fast_credit_audit(
                integration_result=integration_result,
                record=SelfModificationRecord(
                    target=substrate_self_mod.target,
                    gate=ModificationGate.ONLINE,
                    decision=GateDecision.BLOCK,
                    old_value_hash="substrate.online_fast:pre",
                    new_value_hash=substrate_self_mod.checkpoint_hash,
                    justification=substrate_self_mod.description,
                    timestamp_ms=self._turn_index,
                    is_reversible=True,
                    checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                    lineage_hash=substrate_self_mod.checkpoint.fast_state_hash,
                    proposal_hash=substrate_self_mod.checkpoint_hash,
                ),
            )
            blocked_result = OnlineFastSubstrateTurnResult(
                recommended=True,
                applied=False,
                gate_decision=gate_decision.value,
                applied_operations=(),
                blocked_operations=("online-fast:evaluation-gate-block",),
                rollback_operations=(),
                parameter_change_rate=substrate_self_mod.parameter_change_rate,
                optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
                source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
                optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
                fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
                description="Online-fast substrate self-mod proposal was blocked by the ONLINE evaluation gate.",
                experimental_live_mutation=self.residual_runtime.supports_live_substrate_mutation,
            )
            self._append_online_fast_evaluation_evidence(
                integration_result=integration_result,
                wave_id=wave_id,
                result=blocked_result,
            )
            if substrate_self_mod.checkpoint.fast_memory_signal:
                self._memory_store.observe_fast_memory_signal(
                    signal=substrate_self_mod.checkpoint.fast_memory_signal,
                    timestamp_ms=max(self._turn_index, 1),
                )
                self._refresh_memory_snapshot_after_online_fast_evidence(
                    integration_result=integration_result,
                )
            return blocked_result
        import_result = self._joint_loop.apply_online_fast_substrate_checkpoint(
            substrate_self_mod.checkpoint,
            checkpoint_id=f"{self._session_id}:{wave_id}:online-fast-substrate",
        )
        self._last_online_fast_import_checkpoint = import_result.checkpoint
        prior_checkpoint = import_result.checkpoint.substrate_checkpoint
        if substrate_self_mod.checkpoint.fast_memory_signal:
            self._memory_store.observe_fast_memory_signal(
                signal=substrate_self_mod.checkpoint.fast_memory_signal,
                timestamp_ms=max(self._turn_index, 1),
            )
            self._refresh_memory_snapshot_after_online_fast_evidence(
                integration_result=integration_result,
            )
        self._append_online_fast_credit_audit(
            integration_result=integration_result,
            record=SelfModificationRecord(
                target=substrate_self_mod.target,
                gate=ModificationGate.ONLINE,
                decision=GateDecision.ALLOW,
                old_value_hash=prior_checkpoint.checkpoint_id if prior_checkpoint is not None else "none",
                new_value_hash=substrate_self_mod.checkpoint.checkpoint_id,
                justification=substrate_self_mod.description,
                timestamp_ms=self._turn_index,
                is_reversible=True,
                checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
                lineage_hash=substrate_self_mod.checkpoint.fast_state_hash,
                proposal_hash=substrate_self_mod.checkpoint_hash,
            ),
        )
        rollback_reason = ""
        rollback_operations: tuple[str, ...] = ()
        if (
            substrate_self_mod.parameter_change_rate > 0.85
            and substrate_self_mod.optimizer_state_norm > 0.85
        ):
            rollback_reason = "online-fast-integrity-guard"
            rollback_operations = self._joint_loop.rollback_online_fast_substrate_import(
                import_result.checkpoint
            )
            self._last_online_fast_import_checkpoint = None
            self._append_online_fast_credit_audit(
                integration_result=integration_result,
                record=SelfModificationRecord(
                    target=substrate_self_mod.target,
                    gate=ModificationGate.ONLINE,
                    decision=GateDecision.BLOCK,
                    old_value_hash=substrate_self_mod.checkpoint.checkpoint_id,
                    new_value_hash=prior_checkpoint.checkpoint_id if prior_checkpoint is not None else "none",
                    justification=(
                        "Online-fast substrate self-mod proposal was rolled back by the session integrity guard. "
                        f"{substrate_self_mod.description}"
                    ),
                    timestamp_ms=self._turn_index,
                    is_reversible=True,
                    checkpoint_id=(
                        prior_checkpoint.checkpoint_id if prior_checkpoint is not None else substrate_self_mod.checkpoint.checkpoint_id
                    ),
                    lineage_hash=substrate_self_mod.checkpoint.fast_state_hash,
                    proposal_hash=substrate_self_mod.checkpoint_hash,
                ),
            )
        applied_result = OnlineFastSubstrateTurnResult(
            recommended=True,
            applied=not bool(rollback_operations),
            gate_decision=(
                "allowed-then-rolled-back" if rollback_operations else gate_decision.value
            ),
            applied_operations=import_result.applied_operations,
            blocked_operations=(),
            rollback_operations=rollback_operations,
            parameter_change_rate=substrate_self_mod.parameter_change_rate,
            optimizer_state_norm=substrate_self_mod.optimizer_state_norm,
            checkpoint_id=substrate_self_mod.checkpoint.checkpoint_id,
            fast_state_hash=substrate_self_mod.checkpoint.fast_state_hash,
            source_fast_state_hash=substrate_self_mod.checkpoint.source_fast_state_hash,
            optimizer_state_description=substrate_self_mod.checkpoint.optimizer_state_description,
            fast_memory_signal=substrate_self_mod.checkpoint.fast_memory_signal,
            description=(
                import_result.description
                if not rollback_operations
                else (
                    f"{import_result.description} Session integrity guard rolled the substrate back "
                    f"via {len(rollback_operations)} owner-side operations."
                )
            ),
            experimental_live_mutation=True,
            rollback_applied=bool(rollback_operations),
            rollback_reason=rollback_reason,
        )
        self._append_online_fast_evaluation_evidence(
            integration_result=integration_result,
            wave_id=wave_id,
            result=applied_result,
        )
        return applied_result

    async def _maybe_apply_rare_heavy(
        self,
        *,
        wave_id: str,
        joint_result: ScheduledJointLoopResult,
        pre_turn_world_temporal_snapshot: object,
        pre_turn_self_temporal_snapshot: object,
    ) -> "RareHeavyTurnResult | None":
        from volvence_zero.agent.session import RareHeavyTurnResult

        if not joint_result.rare_heavy_review_recommended:
            return None
        if not self._rare_heavy_enabled:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode="not-run",
                description="Rare-heavy review was recommended, but session owner has rare-heavy execution disabled.",
                import_decision="disabled",
                reject_reason="rare-heavy-disabled",
            )
        turns_since_last_import = self._turn_index - self._last_rare_heavy_turn_index
        if self._last_rare_heavy_turn_index and turns_since_last_import < self._rare_heavy_cooldown_turns:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode="not-run",
                description=(
                    f"Rare-heavy review was recommended, but cooldown is active "
                    f"({turns_since_last_import}/{self._rare_heavy_cooldown_turns} turns since last import)."
                ),
                import_decision="skipped-cooldown",
                reject_reason="cooldown-active",
            )
        bundle = self._build_rare_heavy_training_bundle()
        if bundle.aligned_example_count < self._rare_heavy_min_traces:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode="not-run",
                description=(
                    f"Rare-heavy review was recommended, but only {bundle.aligned_example_count} aligned examples are available; "
                    f"need {self._rare_heavy_min_traces}. {bundle.description}"
                ),
                import_decision="skipped-insufficient-alignment",
                reject_reason="insufficient-aligned-examples",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
            )
        world_pipeline = self._build_rare_heavy_pipeline(
            source_policy=self._joint_loop.world_temporal_policy
        )
        self_pipeline = self._build_rare_heavy_pipeline(
            source_policy=self._joint_loop.self_temporal_policy
        )
        traces = tuple(example.trace for example in bundle.examples)
        substrate_batches = tuple(example.substrate_batch for example in bundle.examples)
        try:
            world_pipeline_result = world_pipeline.run_pipeline(
                traces=traces,
                substrate_steps_per_trace=substrate_batches if substrate_batches else None,
            )
            self_pipeline_result = self_pipeline.run_pipeline(
                traces=traces,
                substrate_steps_per_trace=substrate_batches if substrate_batches else None,
            )
        except RuntimeError as exc:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=None,
                applied_operations=(),
                substrate_status="incompatible",
                substrate_training_mode="unsupported",
                description=(
                    f"Rare-heavy pipeline failed closed during substrate training/import preparation: {exc}"
                ),
                import_decision="pipeline-failed-closed",
                reject_reason="pipeline-failed-closed",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
            )
        world_artifact = world_pipeline.export_rare_heavy_artifact(
            artifact_id=f"{self._session_id}:{wave_id}:rare-heavy:world"
        )
        self_artifact = self_pipeline.export_rare_heavy_artifact(
            artifact_id=f"{self._session_id}:{wave_id}:rare-heavy:self"
        )
        artifact = RareHeavyArtifact(
            artifact_id=f"{self._session_id}:{wave_id}:rare-heavy",
            owner_path=world_artifact.owner_path,
            created_at_ms=world_artifact.created_at_ms,
            temporal_snapshot=DualTrackRareHeavySnapshot(
                world_snapshot=world_artifact.temporal_snapshot,
                self_snapshot=self_artifact.temporal_snapshot,
                description=(
                    f"Dual rare-heavy snapshot world={world_artifact.artifact_id} "
                    f"self={self_artifact.artifact_id}."
                ),
            ),
            memory_checkpoint=world_artifact.memory_checkpoint,
            substrate_checkpoint=world_artifact.substrate_checkpoint,
            transition_step=max(world_artifact.transition_step, self_artifact.transition_step),
            final_ssl_loss=(world_artifact.final_ssl_loss + self_artifact.final_ssl_loss) / 2.0,
            final_total_reward=(world_artifact.final_total_reward + self_artifact.final_total_reward) / 2.0,
            description=(
                f"Dual-track rare-heavy artifact world={world_artifact.artifact_id} "
                f"self={self_artifact.artifact_id}."
            ),
            training_evidence=world_artifact.training_evidence,
            application_checkpoint=self._build_application_rare_heavy_checkpoint(
                artifact_id=f"{self._session_id}:{wave_id}:rare-heavy"
            ),
        )
        combined_rl_steps = (
            world_pipeline_result.rl_steps_completed + self_pipeline_result.rl_steps_completed
        )
        combined_substrate_mode = (
            world_pipeline_result.substrate_training_mode
            if world_pipeline_result.substrate_training_mode == self_pipeline_result.substrate_training_mode
            else f"{world_pipeline_result.substrate_training_mode}+{self_pipeline_result.substrate_training_mode}"
        )
        if combined_rl_steps <= 0:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="skipped",
                substrate_training_mode=combined_substrate_mode,
                description=(
                    f"Rare-heavy pipeline exported {artifact.artifact_id}, but no offline RL steps completed; "
                    f"skipping import. world={world_pipeline_result.description} self={self_pipeline_result.description}"
                ),
                import_decision="skipped-no-offline-rl",
                reject_reason="no-offline-rl-steps",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
                candidate_adapter_parameter_count=(
                    artifact.substrate_checkpoint.adapter_parameter_count
                    if artifact.substrate_checkpoint is not None
                    else 0
                ),
                candidate_adapter_training_loss=(
                    artifact.substrate_checkpoint.adapter_training_loss
                    if artifact.substrate_checkpoint is not None
                    else 0.0
                ),
            )
        pre_import_evaluation = await self._evaluate_rare_heavy_candidate(
            artifact=artifact,
            bundle=bundle,
        )
        if not pre_import_evaluation.accepted:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="rejected",
                substrate_training_mode=combined_substrate_mode,
                description=(
                    f"{world_pipeline_result.description} {self_pipeline_result.description} "
                    f"{pre_import_evaluation.description}"
                ),
                import_decision="rejected-pre-import",
                reject_reason=",".join(pre_import_evaluation.reasons),
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
                candidate_adapter_parameter_count=(
                    artifact.substrate_checkpoint.adapter_parameter_count
                    if artifact.substrate_checkpoint is not None
                    else 0
                ),
                candidate_adapter_training_loss=(
                    artifact.substrate_checkpoint.adapter_training_loss
                    if artifact.substrate_checkpoint is not None
                    else 0.0
                ),
                pre_import_case_count=pre_import_evaluation.case_count,
                pre_import_mean_score_delta=pre_import_evaluation.mean_score_delta,
                pre_import_worst_score_delta=pre_import_evaluation.worst_score_delta,
                pre_import_positive_fraction=pre_import_evaluation.positive_fraction,
                pre_import_passed=pre_import_evaluation.accepted,
                pre_import_judgement=pre_import_evaluation.judgement,
            )
        if not self.residual_runtime.supports_live_substrate_mutation:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="review-only",
                substrate_training_mode=combined_substrate_mode,
                description=(
                    f"{world_pipeline_result.description} {self_pipeline_result.description} "
                    "Rare-heavy candidate stayed review-only because the live runtime is enforcing the "
                    "frozen-substrate doctrine."
                ),
                import_decision="blocked-by-doctrine",
                reject_reason="frozen-substrate-doctrine",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
                candidate_adapter_parameter_count=(
                    artifact.substrate_checkpoint.adapter_parameter_count
                    if artifact.substrate_checkpoint is not None
                    else 0
                ),
                candidate_adapter_training_loss=(
                    artifact.substrate_checkpoint.adapter_training_loss
                    if artifact.substrate_checkpoint is not None
                    else 0.0
                ),
                pre_import_case_count=pre_import_evaluation.case_count,
                pre_import_mean_score_delta=pre_import_evaluation.mean_score_delta,
                pre_import_worst_score_delta=pre_import_evaluation.worst_score_delta,
                pre_import_positive_fraction=pre_import_evaluation.positive_fraction,
                pre_import_passed=pre_import_evaluation.accepted,
                pre_import_judgement=pre_import_evaluation.judgement,
            )
        try:
            import_result = self.apply_rare_heavy_artifact(
                artifact,
                checkpoint_id=f"{self._session_id}:{wave_id}:rare-heavy-import",
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return RareHeavyTurnResult(
                recommended=True,
                applied=False,
                artifact_id=artifact.artifact_id,
                applied_operations=(),
                substrate_status="incompatible",
                substrate_training_mode=combined_substrate_mode,
                description=(
                    f"{world_pipeline_result.description} {self_pipeline_result.description} "
                    f"Rare-heavy import failed closed: {exc}"
                ),
                import_decision="import-failed-closed",
                reject_reason="import-failed-closed",
                bundle_alignment_ratio=bundle.alignment_ratio,
                bundle_trace_count=bundle.trace_count,
                bundle_substrate_batch_count=bundle.substrate_batch_count,
                bundle_mean_trace_step_count=bundle.mean_trace_step_count,
                bundle_mean_sequence_length=bundle.mean_sequence_length,
                bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
                candidate_adapter_parameter_count=(
                    artifact.substrate_checkpoint.adapter_parameter_count
                    if artifact.substrate_checkpoint is not None
                    else 0
                ),
                candidate_adapter_training_loss=(
                    artifact.substrate_checkpoint.adapter_training_loss
                    if artifact.substrate_checkpoint is not None
                    else 0.0
                ),
                pre_import_case_count=pre_import_evaluation.case_count,
                pre_import_mean_score_delta=pre_import_evaluation.mean_score_delta,
                pre_import_worst_score_delta=pre_import_evaluation.worst_score_delta,
                pre_import_positive_fraction=pre_import_evaluation.positive_fraction,
                pre_import_passed=pre_import_evaluation.accepted,
                pre_import_judgement=pre_import_evaluation.judgement,
            )
        self._last_rare_heavy_import_checkpoint = replace(
            import_result.checkpoint,
            world_temporal_snapshot=pre_turn_world_temporal_snapshot,
            self_temporal_snapshot=pre_turn_self_temporal_snapshot,
        )
        self._last_rare_heavy_turn_index = self._turn_index
        return RareHeavyTurnResult(
            recommended=True,
            applied=True,
            artifact_id=artifact.artifact_id,
            applied_operations=import_result.applied_operations,
            substrate_status="imported",
            substrate_training_mode=combined_substrate_mode,
            description=(
                f"{world_pipeline_result.description} {self_pipeline_result.description} "
                f"{import_result.description}"
            ),
            import_decision="imported",
            bundle_alignment_ratio=bundle.alignment_ratio,
            bundle_trace_count=bundle.trace_count,
            bundle_substrate_batch_count=bundle.substrate_batch_count,
            bundle_mean_trace_step_count=bundle.mean_trace_step_count,
            bundle_mean_sequence_length=bundle.mean_sequence_length,
            bundle_mean_residual_magnitude=bundle.mean_residual_magnitude,
            candidate_adapter_parameter_count=(
                artifact.substrate_checkpoint.adapter_parameter_count
                if artifact.substrate_checkpoint is not None
                else 0
            ),
            candidate_adapter_training_loss=(
                artifact.substrate_checkpoint.adapter_training_loss
                if artifact.substrate_checkpoint is not None
                else 0.0
            ),
            pre_import_case_count=pre_import_evaluation.case_count,
            pre_import_mean_score_delta=pre_import_evaluation.mean_score_delta,
            pre_import_worst_score_delta=pre_import_evaluation.worst_score_delta,
            pre_import_positive_fraction=pre_import_evaluation.positive_fraction,
            pre_import_passed=pre_import_evaluation.accepted,
            pre_import_judgement=pre_import_evaluation.judgement,
        )


__all__ = ["SessionTrainingPhaseMixin"]
