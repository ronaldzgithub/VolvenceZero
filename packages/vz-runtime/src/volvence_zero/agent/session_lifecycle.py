"""Public input/output surface mixin for ``AgentSessionRunner``.

Debt #9 wave 1 split: this mixin owns the methods that other systems
call into to drive the session from the outside -- dialogue trace
exporters, dialogue outcome submission, semantic event enqueueing,
environment outcome stashing, context lifecycle, and the public
artifact-level rare-heavy API.

It is a pure ``class`` with no ``__init__`` and no state of its own.
All instance attributes it reads (``self._dialogue_trace_store``,
``self._dialogue_external_outcome_module``, ``self._pending_*``,
``self._upstream_snapshots``, ``self._memory_store``,
``self._joint_loop``, ``self._application_rare_heavy_state``,
``self._domain_knowledge_store``, ``self._case_memory_store``,
``self._session_post_queue``, ``self._completed_session_reports``,
``self._context_index``, ``self._turn_index``, ``self._session_id``,
``self._previous_*``, ``self._recommended_z``,
``self._recent_substrate_batches``, ``self._recent_rare_heavy_examples``)
are owned by ``AgentSessionRunner.__init__``. The mixin must be mixed
into a concrete class that defines ``__init__`` first so the MRO
keeps state setup in the canonical place.

Cross-mixin call surface: ``begin_new_context`` calls
``self._maybe_build_current_session_report`` /
``self._build_session_post_slow_loop_job`` /
``self._publish_session_post_snapshot`` -- these live in
``SessionWritebackPhaseMixin`` and resolve via standard MRO.
"""

from __future__ import annotations

from dataclasses import replace

from volvence_zero.application import RetrievalPolicySnapshot
from volvence_zero.environment import EnvironmentOutcome
from volvence_zero.application.knowledge_channels import (
    domain_knowledge_prior_updates_from_reviewed,
)
from volvence_zero.application.runtime import ApplicationPriorUpdate
from volvence_zero.agent.dialogue_outcome_producers import (
    structural_outcome_evidence_from_external,
)
from volvence_zero.agent.conditioning_lineage import (
    resolve_conditioning_lineage_for_outcome,
)
from volvence_zero.credit.gate import CreditSnapshot
from volvence_zero.dialogue_trace import (
    ConditioningLineage,
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
    DialogueOutcomeEvidence,
    DialogueOutcomeResolution,
)
from volvence_zero.integration import _apply_application_prior_writeback
from volvence_zero.joint_loop import (
    RareHeavyArtifact,
    RareHeavyImportCheckpoint,
    RareHeavyImportResult,
)
from volvence_zero.prediction.error import PredictionErrorSnapshot
from volvence_zero.agent.session_post_slow_loop import (
    SettledEnvironmentAction,
)
from volvence_zero.semantic_state import (
    ExternalSemanticEvent,
    ExternalSemanticEventBatch,
)
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastRequest,
    PreferenceActionForecastRuntime,
)
from volvence_zero.social_cognition import PreferenceActionForecast
from volvence_zero.temporal import (
    TemporalAbstractionSnapshot,
    TemporalActionAdvisoryProposal,
)
from volvence_zero.runtime import WiringLevel


class SessionLifecycleMixin:
    """Methods that drive the session lifecycle from the outside.

    See module docstring for the full list of ``self._*`` attributes
    this mixin assumes ``AgentSessionRunner.__init__`` has set.
    """

    def export_dialogue_trace_replay_artifact(self) -> dict[str, object]:
        return self._dialogue_trace_store.export_replay_artifact()

    def export_snapshot_replay_artifact(self) -> dict[str, object]:
        snapshots = tuple(
            {
                "slot_name": slot_name,
                "owner": snapshot.owner,
                "version": snapshot.version,
                "description": getattr(snapshot.value, "description", ""),
            }
            for slot_name, snapshot in sorted(self._upstream_snapshots.items())
        )
        prediction_snapshot = self._upstream_snapshots.get("prediction_error")
        prediction_value = (
            prediction_snapshot.value
            if prediction_snapshot is not None
            and isinstance(prediction_snapshot.value, PredictionErrorSnapshot)
            else None
        )
        temporal_snapshot = self._upstream_snapshots.get("temporal_abstraction")
        temporal_value = (
            temporal_snapshot.value
            if temporal_snapshot is not None
            and isinstance(temporal_snapshot.value, TemporalAbstractionSnapshot)
            else None
        )
        credit_snapshot = self._upstream_snapshots.get("credit")
        credit_value = (
            credit_snapshot.value
            if credit_snapshot is not None
            and isinstance(credit_snapshot.value, CreditSnapshot)
            else None
        )
        action_context = (
            prediction_value.action_context if prediction_value is not None else None
        )
        action_replay = {
            "prediction_error": (
                {
                    "turn_index": prediction_value.turn_index,
                    "bootstrap": prediction_value.bootstrap,
                    "task_error": prediction_value.error.task_error,
                    "relationship_error": prediction_value.error.relationship_error,
                    "regime_error": prediction_value.error.regime_error,
                    "action_error": prediction_value.error.action_error,
                    "magnitude": prediction_value.error.magnitude,
                    "signed_reward": prediction_value.error.signed_reward,
                    "description": prediction_value.description,
                }
                if prediction_value is not None
                else None
            ),
            "action_context": (
                {
                    "segment_id": action_context.segment_id,
                    "abstract_action_id": action_context.abstract_action_id,
                    "z_t_digest": action_context.z_t_digest,
                    "regime_id": action_context.regime_id,
                    "affordance_name": action_context.affordance_name,
                    "environment_event_id": action_context.environment_event_id,
                    "environment_outcome_id": action_context.environment_outcome_id,
                    "environment_task_progress": action_context.environment_task_progress,
                    "environment_action_payoff": action_context.environment_action_payoff,
                    "environment_outcome_terminal": action_context.environment_outcome_terminal,
                }
                if action_context is not None
                else None
            ),
            "closed_segments": (
                tuple(
                    {
                        "segment_id": segment.segment_id,
                        "open_turn_index": segment.open_turn_index,
                        "close_turn_index": segment.close_turn_index,
                        "abstract_action_id": segment.abstract_action_id,
                        "z_t_digest": segment.z_t_digest,
                        "affordance_name": segment.affordance_name,
                    }
                    for segment in temporal_value.closed_segments
                )
                if temporal_value is not None
                else ()
            ),
            "credit_records": (
                tuple(
                    {
                        "level": record.level,
                        "track": record.track.value,
                        "source_event": record.source_event,
                        "credit_value": record.credit_value,
                        "context": record.context,
                    }
                    for record in credit_value.recent_credits
                )
                if credit_value is not None
                else ()
            ),
            "description": (
                "Action replay evidence exported from existing prediction_error, "
                "temporal_abstraction, and credit snapshots."
            ),
        }
        return {
            "session_id": self.active_context_session_id,
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
            "action_replay": action_replay,
            "dialogue_trace": self.export_dialogue_trace_replay_artifact(),
            "description": (
                "Snapshot replay artifact exported from existing runtime "
                "snapshots; no trace-specific runtime schema was introduced."
            ),
        }

    def attach_dialogue_outcome_evidence(
        self,
        evidence: tuple[DialogueOutcomeEvidence, ...],
    ) -> DialogueOutcomeResolution | None:
        return self._dialogue_trace_store.attach_outcome_evidence_to_last_trace(evidence)

    async def preview_preference_action_forecast(
        self,
        *,
        request: PreferenceActionForecastRequest,
        runtime: PreferenceActionForecastRuntime,
    ) -> PreferenceActionForecast | None:
        """Run the preference owner pre-action SHADOW phase.

        This facade exists for the closed-alpha relationship controller.  It
        invokes the actual owner against the session-held ``SocialRecordStore``;
        the caller receives only the frozen forecast and cannot construct or
        mutate owner state.  The upcoming canonical turn publishes the same
        pending forecast through normal propagation.  No expression, PE,
        credit, or evaluator surface is touched here.
        """

        expected_turn = self._turn_index + 1
        if request.turn_index != expected_turn:
            raise ValueError(
                "preference forecast preview must target the upcoming turn "
                f"{expected_turn}, got {request.turn_index}"
            )
        if request.session_scope != self._session_id:
            raise ValueError(
                "preference forecast preview session_scope must match runner session"
            )
        owner = PreferenceAboutOtherModule(
            wiring_level=WiringLevel.SHADOW,
            record_store=self._social_record_store,
            turn_index=request.turn_index,
            action_forecast_runtime=runtime,
            action_forecast_request=request,
        )
        snapshot = (await owner.process({})).value
        matches = tuple(
            forecast
            for forecast in snapshot.action_forecasts
            if forecast.decision_id == request.decision_id
        )
        if len(matches) > 1:
            raise RuntimeError(
                "preference owner published multiple forecasts for one decision"
            )
        return matches[0] if matches else None

    def stage_self_temporal_action_advisory(
        self,
        advisory: TemporalActionAdvisoryProposal,
    ) -> None:
        """Stage one owner-validated advisory for the upcoming turn."""

        if self._pending_self_temporal_action_advisory is not None:
            raise RuntimeError(
                "a self-temporal action advisory is already pending consumption"
            )
        forecasts = {
            item.forecast_id: item
            for item in self._social_record_store.preference_action_forecasts
        }
        try:
            forecast = forecasts[advisory.prediction_id]
        except KeyError as exc:
            raise ValueError(
                "self-temporal advisory must reference a pending owner forecast"
            ) from exc
        if advisory.decision_id != forecast.decision_id:
            raise ValueError("self-temporal advisory decision lineage mismatch")
        action_ids = {
            candidate.action_id for candidate in forecast.candidate_predictions
        }
        if advisory.action_id not in action_ids:
            raise ValueError("self-temporal advisory action is outside forecast surface")
        self._pending_self_temporal_action_advisory = advisory

    def submit_dialogue_outcome(
        self,
        *,
        kind: DialogueExternalOutcomeKind,
        source: DialogueExternalOutcomeEvidenceSource = DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence: float = 0.9,
        turn_index: int | None = None,
        evidence_ref: str | None = None,
        description: str = "",
        action_turn_index: int | None = None,
        forecast_id: str = "",
        decision_id: str = "",
        action_id: str = "",
        typing_qualification_id: str = "",
        typing_qualification_sha256: str = "",
        typing_runtime_id: str = "",
        typing_schema_version: str = "",
    ) -> DialogueExternalOutcomeEvidence:
        """Submit a typed external dialogue outcome (Rupture-and-Repair M2).

        This is the single legal entry point for external dialogue
        outcomes (user explicit feedback, environment outcomes, human
        review, or gated LLM proposals). It:

        * appends a ``DialogueExternalOutcomeEvidence`` to the
          ``DialogueExternalOutcomeModule`` buffer — the only path into
          the ``dialogue_external_outcome`` snapshot slot;
        * attaches a *structural* ``DialogueOutcomeEvidence`` to the
          most recent dialogue trace via the existing trace-resolution
          path, so replay artifacts also see the outcome;
        * does NOT write memory, regime, or PE state directly. All
          downstream effects arise from PE / regime / rupture_state
          consuming the published snapshot on the next turn (R8).

        ``turn_index`` defaults to the current turn index, which is
        correct for outcomes attached to the turn that just finished
        (or is about to run when ``submit`` is called before
        ``run_turn``). Pass an explicit value when an external reviewer
        retrospectively scores a past turn.

        ``action_turn_index`` names the turn whose *action* is being
        evaluated, which is what bank attribution joins on. It defaults to
        ``turn_index`` because on this surface the two coincide. Surfaces
        where they do not -- the HTTP feedback endpoint binds ``turn_index``
        to the consuming turn, one ahead -- must pass it explicitly, or the
        outcome will be credited to the wrong turn's bank set.
        """

        resolved_turn = int(turn_index) if turn_index is not None else max(
            0, int(self._turn_index)
        )
        # On this surface ``turn_index`` already means "the turn whose action
        # this evaluates", so it doubles as the attribution key unless a caller
        # separates them explicitly (the HTTP surface does, because there
        # ``turn_index`` means the consuming turn instead).
        resolved_action_turn = (
            int(action_turn_index) if action_turn_index is not None else resolved_turn
        )
        ref = evidence_ref or f"{source.value}:{kind.value}:turn-{resolved_turn}"
        evidence = DialogueExternalOutcomeEvidence(
            evidence_id=f"external:{source.value}:{kind.value}:{resolved_turn}:{ref}",
            turn_index=resolved_turn,
            kind=kind,
            source=source,
            confidence=float(confidence),
            evidence_ref=ref,
            description=description,
            # Attribution join key. ``_session_id`` rather than
            # ``active_context_session_id``: a context reset must not sever the
            # link between an outcome and the banks that produced the action,
            # and an outcome often arrives after such a reset.
            session_scope=self._session_id,
            action_turn_index=resolved_action_turn,
            forecast_id=forecast_id,
            decision_id=decision_id,
            action_id=action_id,
            typing_qualification_id=typing_qualification_id,
            typing_qualification_sha256=typing_qualification_sha256,
            typing_runtime_id=typing_runtime_id,
            typing_schema_version=typing_schema_version,
        )
        self._dialogue_external_outcome_module.append_evidence(evidence)
        structural = structural_outcome_evidence_from_external(evidence)
        if structural is not None:
            # Target the turn whose action is being evaluated, not the latest
            # one. These differ whenever a reviewer scores a past turn, and
            # attaching to the wrong trace would misattribute the outcome to a
            # different bank set entirely.
            self._dialogue_trace_store.attach_outcome_evidence_to_turn(
                turn_index=resolved_action_turn,
                evidence=(structural,),
            )
        lineage = self.resolve_dialogue_outcome_attribution(evidence)
        if (
            lineage is not None
            and lineage not in self._pending_external_outcome_lineages
        ):
            self._pending_external_outcome_lineages.append(lineage)
        return evidence

    def resolve_dialogue_outcome_attribution(
        self,
        evidence: DialogueExternalOutcomeEvidence,
    ) -> ConditioningLineage | None:
        """Resolve an external outcome to the bank lineage of its action turn.

        Read-only audit surface over this session's own dialogue trace: given
        evidence previously produced by :meth:`submit_dialogue_outcome` (or
        reconstructed from a durable sink), it answers "which conditioning
        banks, at which state versions, shaped the action this outcome is
        rating". Returns ``None`` for unattributable evidence, a trimmed or
        unrecorded turn, or a turn where no bank was live -- see
        :func:`resolve_conditioning_lineage_for_outcome` for the exact
        semantics. It never mutates trace, PE, or credit state.
        """

        return resolve_conditioning_lineage_for_outcome(
            evidence=evidence,
            trace_snapshot=self._dialogue_trace_store.snapshot(),
        )

    def enqueue_semantic_events(
        self,
        events: ExternalSemanticEventBatch | tuple[ExternalSemanticEvent, ...],
    ) -> tuple[str, ...]:
        event_tuple = events.events if isinstance(events, ExternalSemanticEventBatch) else events
        self._pending_semantic_events.extend(event_tuple)
        if len(self._pending_semantic_events) > 64:
            del self._pending_semantic_events[:-64]
        return tuple(event.event_id for event in event_tuple)

    def _drain_pending_semantic_events(self) -> tuple[ExternalSemanticEvent, ...]:
        events = tuple(self._pending_semantic_events)
        self._pending_semantic_events.clear()
        return events

    def remember_environment_outcome(self, outcome_id: str) -> None:
        """Compatibility path for producers that only have an outcome id."""

        self._pending_environment_outcome_id = outcome_id

    def submit_environment_outcome(self, outcome: EnvironmentOutcome) -> None:
        """Buffer one complete immutable outcome for next-turn PE settlement."""

        if self._pending_environment_outcome is not None:
            raise RuntimeError(
                "pending EnvironmentOutcome must be consumed before submitting another"
            )
        temporal_snapshot = self._upstream_snapshots.get(
            "temporal_abstraction"
        )
        temporal_value = (
            temporal_snapshot.value
            if temporal_snapshot is not None
            and isinstance(
                temporal_snapshot.value,
                TemporalAbstractionSnapshot,
            )
            else None
        )
        retrieval_snapshot = self._upstream_snapshots.get(
            "retrieval_policy"
        )
        retrieval_value = (
            retrieval_snapshot.value
            if retrieval_snapshot is not None
            and isinstance(
                retrieval_snapshot.value,
                RetrievalPolicySnapshot,
            )
            else None
        )
        action_family_id = (
            retrieval_value.abstract_action
            if (
                retrieval_value is not None
                and retrieval_value.abstract_action is not None
            )
            else temporal_value.active_abstract_action
            if temporal_value is not None
            else ""
        )
        self._pending_environment_outcome = SettledEnvironmentAction(
            outcome=outcome,
            action_family_id=action_family_id,
            action_family_version=(
                temporal_value.action_family_version
                if temporal_value is not None and action_family_id
                else 0
            ),
            controller_code_digest=(
                tuple(
                    round(value, 6)
                    for value in temporal_value.controller_state.code[:8]
                )
                if temporal_value is not None and action_family_id
                else ()
            ),
        )
        self._pending_environment_outcome_id = ""
        self._pending_environment_prediction_id = outcome.prediction_id or ""

    def _consume_pending_environment_outcome(
        self,
    ) -> SettledEnvironmentAction | None:
        outcome = self._pending_environment_outcome
        self._pending_environment_outcome = None
        return outcome

    def _consume_pending_environment_outcome_id(self) -> str:
        outcome_id = self._pending_environment_outcome_id
        self._pending_environment_outcome_id = ""
        return outcome_id

    def remember_environment_prediction_id(self, prediction_id: str) -> None:
        """Buffer a plan_ref / prediction_id for the next-turn PE action context.

        Packet A (long-horizon-closure): paired with
        ``remember_environment_outcome`` so the AffordanceInvoker can
        thread its caller-supplied ``plan_ref`` all the way to
        ``PredictionActionContext.prediction_id`` on the next turn.
        Empty string means "no plan_ref was supplied" (back-compat path).
        """
        self._pending_environment_prediction_id = prediction_id or ""

    def _consume_pending_environment_prediction_id(self) -> str:
        prediction_id = self._pending_environment_prediction_id
        self._pending_environment_prediction_id = ""
        return prediction_id

    def persist_owners(self) -> tuple[str, ...]:
        """Packet D (long-horizon-closure): export the runner-owned
        hydratable owners (currently just SemanticStateStore) and
        write to the configured ``OwnerHydrationStore``.

        Returns the tuple of persisted owner names for observability.
        Returns ``()`` when no hydration store is wired (DISABLED
        wiring or no persistence backend).

        Caller (BrainSession.persist_owners or LifeformSession.end_scene)
        is responsible for orchestrating the call timing — never in
        the middle of a turn / propagate, only at scene boundaries.
        """
        if self._owner_hydration_store is None:
            return ()
        persisted: list[str] = []
        self._owner_hydration_store.export_and_save_owner(
            self._semantic_state_store, "semantic_state"
        )
        persisted.append("semantic_state")
        self._owner_hydration_store.export_and_save_owner(
            self._regime_module, "regime"
        )
        persisted.append("regime")
        self._owner_hydration_store.export_and_save_owner(
            self._prediction_module, "prediction_error_heads"
        )
        persisted.append("prediction_error_heads")
        self._owner_hydration_store.export_and_save_owner(
            self._credit_module, "credit_heads"
        )
        persisted.append("credit_heads")
        self._owner_hydration_store.export_and_save_owner(
            self._dual_track_gate_learner, "dual_track_gate_learner"
        )
        persisted.append("dual_track_gate_learner")
        self._owner_hydration_store.export_and_save_owner(
            self._social_record_store, "social_record_store"
        )
        persisted.append("social_record_store")
        self._owner_hydration_store.save_snapshot(
            self._joint_loop.export_learning_persistence_snapshot(
                include_runtime_replay=False,
            )
        )
        persisted.append("joint_loop.learning")
        self._owner_hydration_store.export_and_save_owner(
            self._reflection_consolidation_learner,
            "reflection.consolidation_score",
        )
        persisted.append("reflection.consolidation_score")
        return tuple(persisted)

    def begin_new_context(self, *, reason: str = "manual") -> tuple[str, ...]:
        operations: list[str] = []
        active_report = self._maybe_build_current_session_report()
        session_post_job = self._build_session_post_slow_loop_job(
            active_report=active_report,
        )
        self._settled_environment_outcomes = []
        if active_report is not None:
            self._completed_session_reports.append(active_report)
            operations.append(f"session-report:checkpoint:{self.active_context_session_id}")
        operations.extend(
            self._memory_store.reset_nested_context(
                reason=reason,
                timestamp_ms=max(self._turn_index, 1),
            )
        )
        self._context_index += 1
        self._upstream_snapshots = {}
        self._previous_substrate_snapshot = None
        self._online_training_substrate_snapshots = []
        self._previous_personal_conditioning_snapshot = None
        self._previous_prediction_reward = 0.0
        self._previous_prediction_magnitude = 0.0
        self._previous_prediction_error = None
        self._recommended_z = None
        self._recent_substrate_batches = []
        self._recent_rare_heavy_examples = []
        if session_post_job is not None:
            self._session_post_queue.enqueue(session_post_job)
            self._session_post_queue.schedule()
            operations.append(f"session-post-slow-loop:enqueued:{session_post_job.job_id}")
        self._publish_session_post_snapshot()
        return tuple(operations)

    def apply_rare_heavy_artifact(
        self,
        artifact: RareHeavyArtifact,
        *,
        checkpoint_id: str | None = None,
    ) -> RareHeavyImportResult:
        application_checkpoint = self._application_rare_heavy_state.export_rare_heavy_state(
            checkpoint_id=f"{checkpoint_id or artifact.artifact_id}:application-preimport"
        )
        result = self._joint_loop.apply_rare_heavy_artifact(
            artifact,
            checkpoint_id=checkpoint_id,
        )
        application_operations: tuple[str, ...] = ()
        if artifact.application_checkpoint is not None:
            application_operations = self._application_rare_heavy_state.import_rare_heavy_state(
                artifact.application_checkpoint
            )
        reset_operations = self._memory_store.reset_nested_context(
            reason="rare-heavy-import",
            timestamp_ms=max(self._turn_index, 1),
        )
        result = replace(
            result,
            checkpoint=replace(result.checkpoint, application_checkpoint=application_checkpoint),
        )
        knowledge_import_operations: tuple[str, ...] = ()
        if artifact.application_checkpoint is not None and artifact.application_checkpoint.reviewed_knowledge_candidates:
            reviewed = artifact.application_checkpoint.reviewed_knowledge_candidates
            knowledge_updates = domain_knowledge_prior_updates_from_reviewed(
                job_id=f"{artifact.artifact_id}:rare-heavy-knowledge-import",
                reviewed=reviewed,
            )
            if knowledge_updates:
                knowledge_prior = ApplicationPriorUpdate(
                    source_session_post_job_id=f"{artifact.artifact_id}:rare-heavy-knowledge-import",
                    domain_knowledge_updates=knowledge_updates,
                    description="Rare-heavy reviewed knowledge import batch (owner-side gated writeback).",
                )
                (
                    knowledge_import_operations,
                    _knowledge_blocks,
                    _knowledge_audits,
                    _knowledge_report,
                ) = _apply_application_prior_writeback(
                    prior_update=knowledge_prior,
                    domain_knowledge_store=self._domain_knowledge_store,
                    case_memory_store=self._case_memory_store,
                    application_rare_heavy_state=self._application_rare_heavy_state,
                    credit_snapshot=None,
                    timestamp_ms=max(self._turn_index, 1) + 3,
                    checkpoint_id=checkpoint_id or artifact.artifact_id,
                    apply_enabled=True,
                    retrieval_apply_enabled=True,
                    blocked_reason="allow",
                )
        tail_operations = application_operations + knowledge_import_operations + reset_operations
        if not tail_operations:
            return result
        return replace(
            result,
            applied_operations=result.applied_operations + tail_operations,
            description=(
                f"{result.description} "
                f"{'Application rare-heavy state imported. ' if application_operations else ''}"
                f"{'Reviewed domain knowledge import applied. ' if knowledge_import_operations else ''}"
                f"{'Nested context reset applied after import.' if reset_operations else ''}"
            ).strip(),
        )

    def review_rare_heavy_artifact(
        self,
        artifact: RareHeavyArtifact,
        *,
        checkpoint_id: str | None = None,
    ) -> RareHeavyImportResult:
        application_checkpoint = self._application_rare_heavy_state.export_rare_heavy_state(
            checkpoint_id=f"{checkpoint_id or artifact.artifact_id}:application-review"
        )
        result = self._joint_loop.review_rare_heavy_artifact(
            artifact,
            checkpoint_id=checkpoint_id,
        )
        return replace(
            result,
            checkpoint=replace(result.checkpoint, application_checkpoint=application_checkpoint),
            description=(
                f"{result.description} "
                "Application rare-heavy state remained under session-owned review."
            ).strip(),
        )

    def rollback_rare_heavy_import(
        self,
        checkpoint: RareHeavyImportCheckpoint,
    ) -> tuple[str, ...]:
        operations = list(self._joint_loop.rollback_rare_heavy_import(checkpoint))
        if checkpoint.application_checkpoint is not None:
            operations.extend(self._application_rare_heavy_state.restore_rare_heavy_state(checkpoint.application_checkpoint))
        return tuple(operations)


__all__ = ["SessionLifecycleMixin"]
