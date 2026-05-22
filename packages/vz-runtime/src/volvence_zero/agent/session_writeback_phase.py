"""Session-post slow-loop + experience writeback mixin for ``AgentSessionRunner``.

Debt #9 wave 1 split: this mixin owns the methods that build,
schedule, run, and consume session-post slow-loop jobs plus the
follow-up experience publication and delayed-evidence accounting.
These methods are coupled by a shared lifecycle: a slow-loop job is
constructed from the current upstream snapshot graph, drained on
context boundary or explicit request, then its result is reflected
back into evaluation backbone delayed evidence + the credit snapshot
+ the experience consolidation / fast prior snapshots.

It is a pure ``class`` with no ``__init__`` and no state of its own.
All instance attributes it reads (``self._evaluation_backbone``,
``self._upstream_snapshots``, ``self._session_post_queue``,
``self._session_post_module``, ``self._session_post_snapshot``,
``self._experience_consolidation_module``,
``self._experience_consolidation_snapshot``,
``self._experience_fast_prior_module``,
``self._experience_fast_prior_snapshot``,
``self._memory_store``, ``self._world_temporal_policy``,
``self._regime_module``, ``self._domain_knowledge_store``,
``self._case_memory_store``, ``self._application_rare_heavy_state``,
``self._completed_session_reports``, ``self._recent_training_traces``,
``self._recent_substrate_batches``, ``self._previous_prediction_error``,
``self._last_session_post_writeback_request``,
``self._session_post_lock``, ``self._turn_index``) are owned by
``AgentSessionRunner.__init__``.

This mixin also closes a pre-existing latent regression in
``session.py``: ``derive_learning_evidence_credit_records`` was
referenced inside ``_record_application_delayed_evidence`` but never
imported in the original module (the call path is rare and tests did
not exercise it). The mixin imports it explicitly here.
"""

from __future__ import annotations

from dataclasses import replace

from volvence_zero.application.experience_layers import (
    ApplicationPriorProposalBuilder,
    ApplicationPriorProposalInputs,
)
from volvence_zero.application.knowledge_channels import (
    build_conversation_knowledge_candidates,
)
from volvence_zero.application.runtime import (
    ApplicationOutcomeAttribution,
    ApplicationPriorUpdate,
    ApplicationPriorWritebackReport,
    ApplicationSequencePayoff,
    BoundaryPolicySnapshot,
    CaseMemorySnapshot,
    DelayedCreditSummary,
    DomainKnowledgeSnapshot,
    ExperienceConsolidationSnapshot,
    ExperienceFastPriorSnapshot,
    KnowledgeHit,
    RetrievalPolicySnapshot,
    StrategyPlaybookSnapshot,
)
from volvence_zero.application.storage import (
    ProvisionalReconcileResult,
    ProvisionalReconcileThresholds,
)
from volvence_zero.agent.session_helpers import (
    abstract_action_alignment as _abstract_action_alignment,
    application_outcome_score as _application_outcome_score,
    clamp01 as _clamp,
    experience_deltas_from_prior_update as _experience_deltas_from_prior_update,
    regime_alignment as _regime_alignment,
    retrieval_mix_alignment as _retrieval_mix_alignment,
)
from volvence_zero.agent.session_post_slow_loop import (
    SessionPostSlowLoopJob,
    SessionPostSlowLoopResult,
    SessionPostSlowLoopSnapshot,
)
from volvence_zero.credit.gate import (
    CreditSnapshot,
    derive_learning_evidence_credit_records,
    extend_credit_snapshot,
)
from volvence_zero.evaluation import (
    EvaluationReport,
    EvaluationScore,
    EvaluationSnapshot,
)
from volvence_zero.integration import (
    _apply_application_prior_writeback,
    apply_session_post_writeback_request,
)
from volvence_zero.reflection import WritebackResult
from volvence_zero.regime import RegimeSnapshot
from volvence_zero.runtime import Snapshot, validate_snapshot_contract
from volvence_zero.temporal import TemporalAbstractionSnapshot


# Per-mixin module-level singleton: only the writeback path needs this
# proposal builder, so it lives here rather than in ``session.py``
# (mechanical move rule 5: per-mixin private helpers move with the
# mixin).
_APPLICATION_PRIOR_PROPOSAL_BUILDER = ApplicationPriorProposalBuilder()


class SessionWritebackPhaseMixin:
    """Methods that build / run / consume session-post slow-loop jobs.

    See module docstring for the full list of ``self._*`` attributes
    this mixin assumes ``AgentSessionRunner.__init__`` has set.
    """

    def _record_application_delayed_evidence(
        self,
        *,
        completed_results: tuple[SessionPostSlowLoopResult, ...],
    ) -> None:
        for result in completed_results:
            if (
                not result.delayed_outcome_ledger
                and not result.sequence_payoffs
                and not result.application_prior_audits
            ):
                continue
            if result.delayed_outcome_ledger or result.sequence_payoffs:
                self._evaluation_backbone.record_application_delayed_evidence(
                    session_id=result.context_session_id,
                    wave_id=result.job_id,
                    timestamp_ms=max(result.closed_at_turn, 1) + 1000,
                    base_snapshot=EvaluationSnapshot(
                        turn_scores=(),
                        session_scores=(),
                        alerts=(),
                        description="Application delayed evidence baseline.",
                    ),
                    delayed_outcome_ledger=result.delayed_outcome_ledger,
                    sequence_payoffs=result.sequence_payoffs,
                )
            if self._upstream_snapshots.get("credit") is not None:
                credit_snapshot = self._upstream_snapshots["credit"]
                if credit_snapshot is not None:
                    delayed_scores: tuple[EvaluationScore, ...] = ()
                    if result.delayed_outcome_ledger:
                        delayed_scores = delayed_scores + (
                            EvaluationScore(
                                family="learning",
                                metric_name="delayed_retrieval_mix_alignment",
                                value=_clamp(
                                    sum(item.retrieval_mix_alignment for item in result.delayed_outcome_ledger)
                                    / len(result.delayed_outcome_ledger)
                                ),
                                confidence=0.7,
                                evidence=result.delayed_outcome_ledger[-1].description,
                            ),
                            EvaluationScore(
                                family="learning",
                                metric_name="delayed_regime_alignment",
                                value=_clamp(
                                    sum(item.regime_alignment for item in result.delayed_outcome_ledger)
                                    / len(result.delayed_outcome_ledger)
                                ),
                                confidence=0.7,
                                evidence=result.delayed_outcome_ledger[-1].description,
                            ),
                            EvaluationScore(
                                family="abstraction",
                                metric_name="delayed_abstract_action_alignment",
                                value=_clamp(
                                    sum(item.abstract_action_alignment for item in result.delayed_outcome_ledger)
                                    / len(result.delayed_outcome_ledger)
                                ),
                                confidence=0.7,
                                evidence=result.delayed_outcome_ledger[-1].description,
                            ),
                        )
                    if result.sequence_payoffs:
                        delayed_scores = delayed_scores + (
                            EvaluationScore(
                                family="learning",
                                metric_name="regime_sequence_payoff",
                                value=_clamp(
                                    sum(item.rolling_payoff for item in result.sequence_payoffs)
                                    / len(result.sequence_payoffs)
                                ),
                                confidence=0.68,
                                evidence=result.sequence_payoffs[-1].description,
                            ),
                        )
                    delayed_credit_records = derive_learning_evidence_credit_records(
                        evaluation_snapshot=EvaluationSnapshot(
                            turn_scores=(),
                            session_scores=delayed_scores,
                            alerts=(),
                            description="session delayed credit snapshot",
                        ),
                        timestamp_ms=max(result.closed_at_turn, 1) + 1001,
                    )
                    if delayed_credit_records or result.application_prior_audits:
                        extended_credit = extend_credit_snapshot(
                            credit_snapshot=credit_snapshot.value,
                            extra_records=delayed_credit_records,
                            extra_modifications=result.application_prior_audits,
                        )
                        self._upstream_snapshots["credit"] = validate_snapshot_contract(
                            snapshot=Snapshot(
                                slot_name="credit",
                                owner=credit_snapshot.owner,
                                version=credit_snapshot.version + 1,
                                timestamp_ms=credit_snapshot.timestamp_ms + 1,
                                value=extended_credit,
                            ),
                            expected_slot="credit",
                            expected_owner="CreditModule",
                            expected_value_type=CreditSnapshot,
                        )

    def _experience_eta_signals(self) -> dict[str, float]:
        signals: dict[str, float] = {}
        fast_prior_snapshot = self.experience_fast_prior_snapshot
        if fast_prior_snapshot is not None:
            fast_prior = fast_prior_snapshot.value
            signals["experience_fast_prior_strength"] = fast_prior.prior_strength
            signals["delayed_fast_prior_available"] = 1.0 if (
                fast_prior.source_attribution_ids or fast_prior.source_sequence_ids
            ) else 0.0
            retrieval_snapshot = self._upstream_snapshots.get("retrieval_policy")
            if retrieval_snapshot is not None and isinstance(retrieval_snapshot.value, RetrievalPolicySnapshot):
                signals["experience_retrieval_mix_bias"] = _clamp(
                    0.5 + (fast_prior.experience_weight_bias - fast_prior.knowledge_weight_bias)
                )
                signals["delayed_retrieval_mix_alignment"] = _clamp(
                    0.5 + (fast_prior.experience_weight_bias - fast_prior.knowledge_weight_bias) * 0.5
                )
            regime_snapshot = self._upstream_snapshots.get("regime")
            if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot):
                active_regime_id = regime_snapshot.value.active_regime.regime_id
                matching_regime_bias = next(
                    (
                        item.bias
                        for item in fast_prior.regime_biases
                        if item.regime_id == active_regime_id
                    ),
                    0.0,
                )
                signals["experience_regime_bias"] = _clamp(0.5 + matching_regime_bias)
                signals["delayed_regime_alignment"] = _clamp(0.5 + matching_regime_bias * 0.5)
            temporal_snapshot = self._upstream_snapshots.get("temporal_abstraction")
            if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot):
                active_action = temporal_snapshot.value.active_abstract_action
                action_bias = next(
                    (
                        item.bias
                        for item in fast_prior.action_biases
                        if item.abstract_action == active_action
                    ),
                    0.0,
                )
                signals["experience_action_bias"] = _clamp(0.5 + action_bias)
                family_version = temporal_snapshot.value.action_family_version
                family_bias = next(
                    (
                        item.continuation_bias
                        for item in fast_prior.family_biases
                        if item.action_family_version == family_version
                    ),
                    0.0,
                )
                signals["experience_action_family_bias"] = _clamp(0.5 + family_bias)
                signals["delayed_abstract_action_alignment"] = _clamp(
                    0.5 + ((action_bias + family_bias) / 2.0) * 0.5
                )
            if fast_prior.sequence_biases:
                signals["regime_sequence_payoff"] = _clamp(
                    0.5
                    + sum(item.payoff_bias for item in fast_prior.sequence_biases) / len(fast_prior.sequence_biases) * 0.5
                )
        case_snapshot = self._upstream_snapshots.get("case_memory")
        if case_snapshot is not None and isinstance(case_snapshot.value, CaseMemorySnapshot):
            if case_snapshot.value.hits:
                mean_relevance = _clamp(
                    sum(hit.relevance_score for hit in case_snapshot.value.hits) / len(case_snapshot.value.hits)
                )
                signals["experience_case_strength"] = mean_relevance
                signals["experience_case_support_prior"] = _clamp(case_snapshot.value.support_prior)
                signals["experience_case_task_prior"] = _clamp(case_snapshot.value.task_prior)
                signals["experience_case_continuum_position"] = _clamp(case_snapshot.value.mean_continuum_position)
                signals["experience_case_continuum_band_coverage"] = _clamp(
                    len(case_snapshot.value.active_band_ids) / 4.0
                )
        playbook_snapshot = self._upstream_snapshots.get("strategy_playbook")
        if playbook_snapshot is not None and isinstance(playbook_snapshot.value, StrategyPlaybookSnapshot):
            if playbook_snapshot.value.matched_rules:
                matched_rules = playbook_snapshot.value.matched_rules
                signals["experience_playbook_strength"] = _clamp(
                    sum(rule.confidence for rule in matched_rules) / len(matched_rules)
                )
                signals["experience_playbook_knowledge_hint"] = _clamp(
                    sum(rule.knowledge_weight_hint for rule in matched_rules) / len(matched_rules)
                )
                signals["experience_playbook_experience_hint"] = _clamp(
                    sum(rule.experience_weight_hint for rule in matched_rules) / len(matched_rules)
                )
                signals["experience_playbook_support_prior"] = _clamp(playbook_snapshot.value.support_prior)
                signals["experience_playbook_task_prior"] = _clamp(playbook_snapshot.value.task_prior)
                signals["experience_playbook_band_coverage"] = _clamp(
                    len(playbook_snapshot.value.active_band_ids) / 4.0
                )
        prior_strength_values = tuple(
            signals[key]
            for key in (
                "experience_case_strength",
                "experience_playbook_strength",
                "delayed_retrieval_mix_alignment",
                "regime_sequence_payoff",
                "experience_fast_prior_strength",
                "experience_action_family_bias",
            )
            if key in signals
        )
        if prior_strength_values:
            signals["experience_control_prior_strength"] = _clamp(
                sum(prior_strength_values) / len(prior_strength_values)
            )
        return signals

    async def drain_session_post_slow_loop(self) -> tuple[SessionPostSlowLoopResult, ...]:
        self._session_post_queue.schedule()
        await self._session_post_queue.wait_for_idle()
        results = self._session_post_queue.consume_completed_results()
        self._record_application_delayed_evidence(completed_results=results)
        self._upstream_snapshots["session_post_slow_loop"] = self._publish_session_post_snapshot(completed_results=results)
        self._upstream_snapshots["experience_consolidation"] = self._publish_experience_consolidation_snapshot(
            completed_results=results
        )
        self._upstream_snapshots["experience_fast_prior"] = self._publish_experience_fast_prior_snapshot()
        # Rupture-and-Repair M4/M6: persist the memory store after the
        # slow-loop writeback has settled. Scoped stores (built via
        # ``volvence_zero.memory.identity.build_scoped_memory_store``)
        # carry a filesystem persistence backend; unscoped stores are
        # in-memory only and ``save_to_backend`` becomes a no-op.
        self._memory_store.save_to_backend()
        return results

    def reconcile_case_memory_provisional(
        self,
        *,
        now_tick: int,
        thresholds: ProvisionalReconcileThresholds | None = None,
    ) -> ProvisionalReconcileResult:
        """Sweep CANDIDATE / PROVISIONAL case_memory records (Gap 4 slice 2a).

        Scene-boundary hook: typically called by the lifeform layer's
        ``end_scene`` AFTER ``drain_session_post_slow_loop`` so that any
        provisional records the slow loop wrote during scene close are
        part of the decision set. Returns the full
        ``ProvisionalReconcileResult`` (promoted / retired / expired
        case_ids + per-decision audit tuple) so callers can surface it
        for observability \u2014 the runner does NOT silently swallow the
        outcome.

        ``now_tick`` is the lifeform clock; the kernel never advances
        this itself. Records with ``expires_at_tick <= now_tick`` are
        retired; others go through the promote / retire-by-weakness
        decision table in ``ApplicationCaseMemoryStore``.

        This is a scene-level sweep \u2014 intentionally synchronous and
        bounded. Mid-turn cheap expiry and async mid-reflection
        workers are slice-2b concerns (see
        ``docs/specs/thinking-loop.md``).
        """
        return self._case_memory_store.reconcile_provisional_cases(
            now_tick=now_tick,
            thresholds=thresholds,
        )

    def _maybe_build_current_session_report(self) -> EvaluationReport | None:
        records = tuple(
            record
            for record in self._evaluation_backbone.records
            if record.session_id == self.active_context_session_id
        )
        if not records:
            return None
        return self._evaluation_backbone.build_session_report(
            session_id=self.active_context_session_id,
            timestamp_ms=max(record.timestamp_ms for record in records) + 1,
        )

    def build_current_session_report(self) -> EvaluationReport | None:
        return self._maybe_build_current_session_report()

    def _build_session_post_slow_loop_job(
        self,
        *,
        active_report: EvaluationReport | None,
    ) -> SessionPostSlowLoopJob | None:
        request = self._last_session_post_writeback_request
        if (
            active_report is None
            or request is None
            or request.context_session_id != self.active_context_session_id
        ):
            return None
        prediction_error_summary: tuple[tuple[str, float], ...] = ()
        if self._previous_prediction_error is not None:
            prediction_error_summary = (
                ("task_error", self._previous_prediction_error.task_error),
                ("relationship_error", self._previous_prediction_error.relationship_error),
                ("regime_error", self._previous_prediction_error.regime_error),
                ("action_error", self._previous_prediction_error.action_error),
                ("magnitude", self._previous_prediction_error.magnitude),
                ("signed_reward", self._previous_prediction_error.signed_reward),
            )
        case_problem_patterns: tuple[str, ...] = ()
        case_risk_markers: tuple[str, ...] = ()
        case_band_ids: tuple[str, ...] = ()
        case_mean_continuum_position = 0.0
        knowledge_domains: tuple[str, ...] = ()
        knowledge_hits: tuple[KnowledgeHit, ...] = ()
        boundary_trigger_reasons: tuple[str, ...] = ()
        regime_id: str | None = None
        abstract_action: str | None = None
        action_family_version = 0
        retrieval_policy_id: str | None = None
        knowledge_weight = 0.0
        experience_weight = 0.0
        experience_domains: tuple[str, ...] = ()
        regime_sequence: tuple[str, ...] = ()
        case_hit_count = 0
        playbook_rule_count = 0
        playbook_band_ids: tuple[str, ...] = ()
        continuum_profile_id: str | None = None
        retrieval_fast_prior_strength = 0.0
        retrieval_fast_prior_attribution_count = 0
        retrieval_fast_prior_sequence_count = 0
        retrieval_regime_bias = 0.0
        retrieval_action_bias = 0.0
        retrieval_family_bias = 0.0
        retrieval_knowledge_weight_bias = 0.0
        retrieval_experience_weight_bias = 0.0
        case_snapshot = self._upstream_snapshots.get("case_memory")
        if case_snapshot is not None and isinstance(case_snapshot.value, CaseMemorySnapshot):
            case_problem_patterns = case_snapshot.value.active_problem_patterns
            case_risk_markers = case_snapshot.value.active_risk_markers
            case_hit_count = len(case_snapshot.value.hits)
            case_band_ids = case_snapshot.value.active_band_ids
            case_mean_continuum_position = case_snapshot.value.mean_continuum_position
            continuum_profile_id = case_snapshot.value.continuum_profile_id
        knowledge_snapshot = self._upstream_snapshots.get("domain_knowledge")
        if knowledge_snapshot is not None and isinstance(knowledge_snapshot.value, DomainKnowledgeSnapshot):
            knowledge_domains = knowledge_snapshot.value.active_domains
            knowledge_hits = knowledge_snapshot.value.hits
        retrieval_policy_snapshot = self._upstream_snapshots.get("retrieval_policy")
        if retrieval_policy_snapshot is not None:
            retrieval_policy_value = retrieval_policy_snapshot.value
            retrieval_policy_id = f"policy:{hash(retrieval_policy_value.intent_description) & 0xFFFF:04x}"
            regime_id = retrieval_policy_value.regime_id
            abstract_action = retrieval_policy_value.abstract_action
            knowledge_weight = retrieval_policy_value.knowledge_weight
            experience_weight = retrieval_policy_value.experience_weight
            experience_domains = retrieval_policy_value.experience_domains
        boundary_snapshot = self._upstream_snapshots.get("boundary_policy")
        if boundary_snapshot is not None and isinstance(boundary_snapshot.value, BoundaryPolicySnapshot):
            boundary_trigger_reasons = boundary_snapshot.value.trigger_reasons
        regime_snapshot = self._upstream_snapshots.get("regime")
        if regime_snapshot is not None and isinstance(regime_snapshot.value, RegimeSnapshot):
            regime_id = regime_id or regime_snapshot.value.active_regime.regime_id
            if regime_snapshot.value.previous_regime is not None:
                regime_sequence = (
                    regime_snapshot.value.previous_regime.regime_id,
                    regime_snapshot.value.active_regime.regime_id,
                )
            else:
                regime_sequence = (regime_snapshot.value.active_regime.regime_id,)
        temporal_snapshot = self._upstream_snapshots.get("temporal_abstraction")
        if temporal_snapshot is not None and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot):
            abstract_action = abstract_action or temporal_snapshot.value.active_abstract_action
            action_family_version = temporal_snapshot.value.action_family_version
        playbook_snapshot = self._upstream_snapshots.get("strategy_playbook")
        if playbook_snapshot is not None and isinstance(playbook_snapshot.value, StrategyPlaybookSnapshot):
            playbook_rule_count = len(playbook_snapshot.value.matched_rules)
            playbook_band_ids = playbook_snapshot.value.active_band_ids
            continuum_profile_id = continuum_profile_id or playbook_snapshot.value.continuum_profile_id
        fast_prior_snapshot = self.experience_fast_prior_snapshot
        if fast_prior_snapshot is not None:
            fast_prior = fast_prior_snapshot.value
            retrieval_fast_prior_strength = fast_prior.prior_strength
            retrieval_fast_prior_attribution_count = len(fast_prior.source_attribution_ids)
            retrieval_fast_prior_sequence_count = len(fast_prior.source_sequence_ids)
            retrieval_knowledge_weight_bias = fast_prior.knowledge_weight_bias
            retrieval_experience_weight_bias = fast_prior.experience_weight_bias
            if regime_id is not None:
                retrieval_regime_bias = next(
                    (item.bias for item in fast_prior.regime_biases if item.regime_id == regime_id),
                    0.0,
                )
            if abstract_action is not None:
                retrieval_action_bias = next(
                    (item.bias for item in fast_prior.action_biases if item.abstract_action == abstract_action),
                    0.0,
                )
            if action_family_version > 0:
                retrieval_family_bias = next(
                    (
                        item.continuation_bias
                        for item in fast_prior.family_biases
                        if item.action_family_version == action_family_version
                    ),
                    0.0,
                )
        conversation_knowledge_candidates = build_conversation_knowledge_candidates(
            knowledge_hits=knowledge_hits,
            context_session_id=request.context_session_id,
            source_wave_id=request.source_wave_id,
            source_turn_index=self._turn_index,
            boundary_trigger_reasons=boundary_trigger_reasons,
        )
        job = SessionPostSlowLoopJob(
            job_id=f"{request.context_session_id}:slow-loop:{self._turn_index}",
            context_session_id=request.context_session_id,
            closed_at_turn=self._turn_index,
            session_report=active_report,
            prior_session_report_count=len(self._completed_session_reports),
            trace_count=len(self._recent_training_traces),
            substrate_batch_count=len(self._recent_substrate_batches),
            prediction_error_summary=prediction_error_summary,
            writeback_request=request,
            description=(
                f"Session-post slow loop job for {request.context_session_id} with "
                f"{len(self._recent_training_traces)} traces and {len(self._recent_substrate_batches)} substrate batches."
            ),
            case_problem_patterns=case_problem_patterns,
            case_risk_markers=case_risk_markers,
            knowledge_domains=knowledge_domains,
            knowledge_hits=knowledge_hits,
            conversation_knowledge_candidates=conversation_knowledge_candidates,
            boundary_trigger_reasons=boundary_trigger_reasons,
            regime_id=regime_id,
            abstract_action=abstract_action,
            action_family_version=action_family_version,
            retrieval_policy_id=retrieval_policy_id,
            knowledge_weight=knowledge_weight,
            experience_weight=experience_weight,
            experience_domains=experience_domains,
            regime_sequence=regime_sequence,
            case_hit_count=case_hit_count,
            playbook_rule_count=playbook_rule_count,
            continuum_profile_id=continuum_profile_id,
            case_band_ids=case_band_ids,
            case_mean_continuum_position=case_mean_continuum_position,
            playbook_band_ids=playbook_band_ids,
            retrieval_fast_prior_strength=retrieval_fast_prior_strength,
            retrieval_fast_prior_attribution_count=retrieval_fast_prior_attribution_count,
            retrieval_fast_prior_sequence_count=retrieval_fast_prior_sequence_count,
            retrieval_regime_bias=retrieval_regime_bias,
            retrieval_action_bias=retrieval_action_bias,
            retrieval_family_bias=retrieval_family_bias,
            retrieval_knowledge_weight_bias=retrieval_knowledge_weight_bias,
            retrieval_experience_weight_bias=retrieval_experience_weight_bias,
            semantic_state_descriptions=request.semantic_state_descriptions,
        )
        self._last_session_post_writeback_request = None
        return job

    async def _run_session_post_slow_loop_job(
        self,
        job: SessionPostSlowLoopJob,
    ) -> SessionPostSlowLoopResult:
        async with self._session_post_lock:
            writeback_result, _ = apply_session_post_writeback_request(
                request=job.writeback_request,
                memory_store=self._memory_store,
                temporal_policy=self._world_temporal_policy,
                regime_module=self._regime_module,
            )
        prediction_summary = dict(job.prediction_error_summary)
        reward = float(prediction_summary.get("signed_reward", 0.0))
        magnitude = float(prediction_summary.get("magnitude", 0.0))
        relationship_error = float(prediction_summary.get("relationship_error", 0.0))
        regime_error = float(prediction_summary.get("regime_error", 0.0))
        outcome_score = _application_outcome_score(
            reward=reward,
            magnitude=magnitude,
            relationship_error=relationship_error,
        )
        retrieval_mix_alignment = _retrieval_mix_alignment(
            regime_id=job.regime_id,
            knowledge_weight=job.knowledge_weight,
            experience_weight=job.experience_weight,
        )
        regime_alignment = _regime_alignment(
            regime_id=job.regime_id,
            outcome_score=outcome_score,
            relationship_error=relationship_error,
            regime_error=regime_error,
            magnitude=magnitude,
        )
        abstract_action_alignment = _abstract_action_alignment(
            regime_id=job.regime_id,
            abstract_action=job.abstract_action,
            action_family_version=job.action_family_version,
            outcome_score=outcome_score,
        )
        dominant_band_id = (
            job.case_band_ids[0]
            if job.case_band_ids
            else job.playbook_band_ids[0]
            if job.playbook_band_ids
            else None
        )
        continuum_alignment = _clamp(
            0.5
            + (
                (1.0 - abs(job.case_mean_continuum_position - 0.5))
                * 0.35
                + retrieval_mix_alignment * 0.35
                + abstract_action_alignment * 0.30
            )
            / 2.0
        )
        delayed_outcome_ledger = (
            ApplicationOutcomeAttribution(
                attribution_id=f"{job.job_id}:outcome",
                source_context_session_id=job.context_session_id,
                source_wave_id=job.writeback_request.source_wave_id,
                regime_id=job.regime_id,
                abstract_action=job.abstract_action,
                action_family_version=job.action_family_version,
                retrieval_policy_id=job.retrieval_policy_id,
                knowledge_weight=job.knowledge_weight,
                experience_weight=job.experience_weight,
                retrieval_mix_alignment=retrieval_mix_alignment,
                regime_alignment=regime_alignment,
                abstract_action_alignment=abstract_action_alignment,
                outcome_score=outcome_score,
                resolved_turn_index=job.closed_at_turn,
                continuum_profile_id=job.continuum_profile_id,
                dominant_band_id=dominant_band_id,
                mean_continuum_position=job.case_mean_continuum_position,
                continuum_alignment=continuum_alignment,
                description=(
                    f"Delayed application outcome for regime={job.regime_id} abstract_action={job.abstract_action} "
                    f"knowledge_weight={job.knowledge_weight:.2f} experience_weight={job.experience_weight:.2f} "
                    f"reward={reward:.2f} magnitude={magnitude:.2f}."
                ),
            ),
        )
        sequence_payoffs = (
            ApplicationSequencePayoff(
                sequence_id=f"{job.job_id}:sequence",
                regime_sequence=job.regime_sequence or ((job.regime_id,) if job.regime_id is not None else ()),
                action_family_version=job.action_family_version,
                sample_count=1,
                rolling_payoff=outcome_score,
                latest_outcome=outcome_score,
                continuum_profile_id=job.continuum_profile_id,
                dominant_band_id=dominant_band_id,
                mean_continuum_position=job.case_mean_continuum_position,
                description=(
                    f"Sequence payoff for regime_sequence={job.regime_sequence or ((job.regime_id,) if job.regime_id is not None else ())} "
                    f"family_version={job.action_family_version}."
                ),
            ),
        )
        delayed_credit_summary = DelayedCreditSummary(
            summary_id=f"{job.job_id}:delayed-credit-summary",
            regime_id=job.regime_id,
            abstract_action=job.abstract_action,
            action_family_version=job.action_family_version,
            retrieval_policy_id=job.retrieval_policy_id,
            knowledge_weight=job.knowledge_weight,
            experience_weight=job.experience_weight,
            retrieval_mix_alignment=retrieval_mix_alignment,
            regime_alignment=regime_alignment,
            abstract_action_alignment=abstract_action_alignment,
            outcome_score=outcome_score,
            sequence_payoff=sequence_payoffs[0].rolling_payoff,
            continuum_alignment=continuum_alignment,
            attribution_count=len(delayed_outcome_ledger),
            sequence_count=len(sequence_payoffs),
            continuum_profile_id=job.continuum_profile_id,
            dominant_band_id=dominant_band_id,
            mean_continuum_position=job.case_mean_continuum_position,
            description=(
                f"Delayed credit summary for regime={job.regime_id} abstract_action={job.abstract_action} "
                f"family_version={job.action_family_version} outcome={outcome_score:.2f} "
                f"mix_alignment={retrieval_mix_alignment:.2f} sequence_payoff={sequence_payoffs[0].rolling_payoff:.2f}."
            ),
        )
        mean_experience_quality = _clamp(
            (
                retrieval_mix_alignment
                + regime_alignment
                + abstract_action_alignment
                + outcome_score
            )
            / 4.0
        )
        application_prior_update = _APPLICATION_PRIOR_PROPOSAL_BUILDER.build(
            inputs=ApplicationPriorProposalInputs(
                job_id=job.job_id,
                closed_at_turn=job.closed_at_turn,
                regime_id=job.regime_id,
                knowledge_domains=job.knowledge_domains,
                experience_domains=job.experience_domains,
                case_problem_patterns=job.case_problem_patterns,
                case_risk_markers=job.case_risk_markers,
                boundary_trigger_reasons=job.boundary_trigger_reasons,
                knowledge_weight=job.knowledge_weight,
                experience_weight=job.experience_weight,
                case_hit_count=job.case_hit_count,
                mean_experience_quality=mean_experience_quality,
                knowledge_hits=job.knowledge_hits,
                conversation_knowledge_candidates=job.conversation_knowledge_candidates,
                retrieval_readout_checkpoint=self._application_rare_heavy_state.retrieval_readout_checkpoint,
                retrieval_fast_prior_strength=max(job.retrieval_fast_prior_strength, mean_experience_quality),
                retrieval_fast_prior_attribution_count=max(job.retrieval_fast_prior_attribution_count, 1),
                retrieval_fast_prior_sequence_count=max(job.retrieval_fast_prior_sequence_count, 1),
                retrieval_regime_bias=max(job.retrieval_regime_bias, regime_alignment - 0.5),
                retrieval_action_bias=max(job.retrieval_action_bias, abstract_action_alignment - 0.5),
                retrieval_family_bias=max(job.retrieval_family_bias, outcome_score - 0.5),
                retrieval_knowledge_weight_bias=(
                    job.retrieval_knowledge_weight_bias - max(retrieval_mix_alignment - 0.5, 0.0) * 0.5
                ),
                retrieval_experience_weight_bias=(
                    job.retrieval_experience_weight_bias + max(retrieval_mix_alignment - 0.5, 0.0) * 0.5
                ),
                retrieval_source_attribution_ids=tuple(
                    item.attribution_id for item in delayed_outcome_ledger
                ),
                retrieval_source_sequence_ids=tuple(
                    item.sequence_id for item in sequence_payoffs
                ),
                retrieval_mean_retrieval_mix_alignment=retrieval_mix_alignment,
                retrieval_mean_regime_alignment=regime_alignment,
                retrieval_mean_action_alignment=abstract_action_alignment,
                retrieval_mean_sequence_payoff=(
                    sum(item.rolling_payoff for item in sequence_payoffs) / len(sequence_payoffs)
                    if sequence_payoffs
                    else 0.0
                ),
            )
        )
        application_apply_enabled = (
            job.writeback_request.reflection_apply_enabled
            and job.writeback_request.structural_writeback_allowed
            and mean_experience_quality >= 0.52
        )
        retrieval_checkpoint_apply_enabled = (
            job.writeback_request.reflection_apply_enabled
            and mean_experience_quality >= 0.45
        )
        if not job.writeback_request.reflection_apply_enabled:
            application_block_reason = "writeback-mode-not-apply"
        elif not job.writeback_request.structural_writeback_allowed:
            application_block_reason = "evolution-judge-block"
        elif mean_experience_quality < 0.52:
            application_block_reason = "experience-quality-below-threshold"
        else:
            application_block_reason = "allow"
        async with self._session_post_lock:
            (
                application_prior_ops,
                application_prior_blocks,
                application_prior_audits,
                application_prior_writeback_report,
            ) = _apply_application_prior_writeback(
                prior_update=application_prior_update,
                domain_knowledge_store=self._domain_knowledge_store,
                case_memory_store=self._case_memory_store,
                application_rare_heavy_state=self._application_rare_heavy_state,
                credit_snapshot=job.writeback_request.credit_snapshot,
                timestamp_ms=max(job.closed_at_turn, 1) + 2,
                checkpoint_id=job.writeback_request.checkpoint_id,
                apply_enabled=application_apply_enabled,
                retrieval_apply_enabled=retrieval_checkpoint_apply_enabled,
                blocked_reason=application_block_reason,
            )
        if application_prior_writeback_report is None:
            application_prior_writeback_report = ApplicationPriorWritebackReport(
                proposed_target_count=0,
                applied_targets=(),
                blocked_targets=(),
                audit_record_count=0,
                description="No application prior update was proposed for this slow-loop result.",
            )
        if writeback_result is not None and (application_prior_ops or application_prior_blocks):
            writeback_result = replace(
                writeback_result,
                applied_operations=writeback_result.applied_operations + application_prior_ops,
                blocked_operations=writeback_result.blocked_operations + application_prior_blocks,
                description=(
                    f"{writeback_result.description} application_prior_ops={len(application_prior_ops)} "
                    f"application_prior_blocks={len(application_prior_blocks)}."
                ),
            )
        elif application_prior_ops or application_prior_blocks:
            writeback_result = WritebackResult(
                applied_operations=application_prior_ops,
                blocked_operations=application_prior_blocks,
                checkpoint=None,
                description=(
                    f"Session-post application prior writeback produced {len(application_prior_ops)} applied ops "
                    f"and {len(application_prior_blocks)} blocked ops."
                ),
            )
        experience_deltas = _experience_deltas_from_prior_update(
            prior_update=application_prior_update,
            blocked_targets=application_prior_writeback_report.blocked_targets,
        )
        return SessionPostSlowLoopResult(
            job_id=job.job_id,
            context_session_id=job.context_session_id,
            closed_at_turn=job.closed_at_turn,
            writeback_result=writeback_result,
            applied=bool(writeback_result is not None and writeback_result.applied_operations),
            blocked=bool(writeback_result is not None and writeback_result.blocked_operations),
            description=(
                f"Session-post slow loop finished for {job.context_session_id} "
                f"applied={bool(writeback_result is not None and writeback_result.applied_operations)} "
                f"blocked={bool(writeback_result is not None and writeback_result.blocked_operations)} "
                f"application_promotion={'allow' if application_apply_enabled else 'blocked'} "
                f"experience_quality={mean_experience_quality:.2f} "
                f"semantic_state={len(job.semantic_state_descriptions)}."
            ),
            experience_deltas=experience_deltas,
            delayed_outcome_ledger=delayed_outcome_ledger,
            sequence_payoffs=sequence_payoffs,
            delayed_credit_summary=delayed_credit_summary,
            conversation_knowledge_candidates=job.conversation_knowledge_candidates,
            application_prior_update=application_prior_update,
            application_prior_writeback_report=application_prior_writeback_report,
            application_prior_audits=application_prior_audits,
            continuum_profile_id=job.continuum_profile_id,
            case_band_ids=job.case_band_ids,
            playbook_band_ids=job.playbook_band_ids,
            semantic_state_descriptions=job.semantic_state_descriptions,
        )

    def _publish_session_post_snapshot(
        self,
        *,
        completed_results: tuple[SessionPostSlowLoopResult, ...] = (),
    ) -> Snapshot[SessionPostSlowLoopSnapshot]:
        self._session_post_snapshot = self._session_post_module.publish_snapshot(
            queue_state=self.session_post_queue_state,
            completed_results=completed_results,
        )
        return self._session_post_snapshot

    def _publish_experience_consolidation_snapshot(
        self,
        *,
        completed_results: tuple[SessionPostSlowLoopResult, ...] = (),
    ) -> Snapshot[ExperienceConsolidationSnapshot]:
        if not completed_results and self._experience_consolidation_snapshot is not None:
            return self._experience_consolidation_snapshot
        self._experience_consolidation_snapshot = self._experience_consolidation_module.publish_snapshot(
            completed_results=completed_results,
        )
        return self._experience_consolidation_snapshot

    def _publish_experience_fast_prior_snapshot(self) -> Snapshot[ExperienceFastPriorSnapshot]:
        experience_consolidation = (
            self._experience_consolidation_snapshot.value
            if self._experience_consolidation_snapshot is not None
            else None
        )
        self._experience_fast_prior_snapshot = self._experience_fast_prior_module.publish_snapshot(
            experience_consolidation_snapshot=experience_consolidation,
        )
        return self._experience_fast_prior_snapshot

    def _collect_session_post_writeback_result(self) -> WritebackResult | None:
        completed = self._session_post_queue.consume_completed_results()
        if completed:
            self._record_application_delayed_evidence(completed_results=completed)
            self._upstream_snapshots["session_post_slow_loop"] = self._publish_session_post_snapshot(
                completed_results=completed
            )
            self._upstream_snapshots["experience_consolidation"] = self._publish_experience_consolidation_snapshot(
                completed_results=completed
            )
            self._upstream_snapshots["experience_fast_prior"] = self._publish_experience_fast_prior_snapshot()
        else:
            self._upstream_snapshots["session_post_slow_loop"] = self._publish_session_post_snapshot()
            self._upstream_snapshots["experience_consolidation"] = self._publish_experience_consolidation_snapshot()
            self._upstream_snapshots["experience_fast_prior"] = self._publish_experience_fast_prior_snapshot()
        if not completed:
            return None
        applied_operations: tuple[str, ...] = ()
        blocked_operations: tuple[str, ...] = ()
        checkpoint = None
        descriptions: list[str] = []
        for result in completed:
            if result.writeback_result is None:
                descriptions.append(result.description)
                continue
            applied_operations = applied_operations + result.writeback_result.applied_operations
            blocked_operations = blocked_operations + result.writeback_result.blocked_operations
            if result.writeback_result.checkpoint is not None:
                checkpoint = result.writeback_result.checkpoint
            descriptions.append(result.writeback_result.description)
        return WritebackResult(
            applied_operations=applied_operations,
            blocked_operations=blocked_operations,
            checkpoint=checkpoint,
            description=" ".join(descriptions),
        )


__all__ = ["SessionWritebackPhaseMixin"]
