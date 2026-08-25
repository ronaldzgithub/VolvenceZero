"""Subjective chapter live-through driver.

The driver is an orchestrator over the existing public session surface:
``submit_semantic_events`` / ``run_turn`` / ``submit_environment_outcome`` /
``end_scene``. It does not own memory, semantic state, PE, credit, temporal
control, or regime.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from lifeform_core import Lifeform, LifeformSession, TurnTriggerKind
from volvence_zero.dialogue_trace import DialogueExternalOutcomeEvidenceSource
from volvence_zero.environment import (
    EnvironmentEventKind,
    EnvironmentMeasurement,
    EnvironmentOutcome,
)

from lifeform_domain_character.chapter_experience import (
    ChapterCoverageKind,
    ChapterLiveThroughLedger,
    ReviewedChapterExperience,
)
from lifeform_domain_character.replay import (
    SceneReplayRecord,
    _REGISTER_TO_OUTCOME,
    _drive_levels_from_session,
    _pe_magnitude,
    _truncate,
)


@dataclass(frozen=True)
class ChapterReplayRecord:
    chapter_id: str
    chapter_index: int
    coverage: str
    semantic_events_submitted: int
    scenes_processed: int
    total_pe_signal: float
    final_drive_levels: tuple[tuple[str, float], ...]
    notes: tuple[str, ...] = ()
    semantic_events_verified: int = 0
    semantic_owner_slots_verified: tuple[str, ...] = ()
    semantic_events_applied_after_outcome: bool = False
    environment_outcomes_settled: int = 0
    runtime_lineage_matches: int = 0
    runtime_transitions: int = 0
    internal_rl_policy_updates: int = 0
    slow_loop_applied_operations: int = 0
    application_owner_targets_updated: tuple[str, ...] = ()
    memory_entry_delta: int = 0
    bake_verified: bool = False
    verification_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChapterSceneBakeEvidence:
    """Snapshot-derived proof that one reviewed scene became experience."""

    scene_id: str
    prediction_id: str
    environment_event_id: str
    environment_outcome_id: str
    evaluated_prediction_id: str
    pe_environment_outcome_id: str
    pe_magnitude: float
    world_z_t: tuple[float, ...]
    self_z_t: tuple[float, ...]
    runtime_replay_wiring: str
    runtime_settled_delta: int
    runtime_transition_delta: int
    runtime_lineage_match_delta: int
    internal_rl_backend: str
    internal_rl_policy_update_applied: bool
    internal_rl_policy_epochs: int
    internal_rl_total_reward: float
    internal_rl_rollback_reasons: tuple[str, ...]
    memory_entry_delta: int
    slow_loop_completed: bool
    slow_loop_applied_operations: tuple[str, ...]
    application_owner_targets_updated: tuple[str, ...]
    delayed_credit_published: bool
    bake_verified: bool
    verification_failures: tuple[str, ...] = ()
    experienced_action_family_ids: tuple[str, ...] = ()
    schema_free_action_family_persisted: bool = False


@dataclass(frozen=True)
class ChapterLiveThroughReport:
    ledger_character_id: str
    chapters_processed: int
    experienced_chapters: int
    learned_chapters: int
    not_known_chapters: int
    no_change_chapters: int
    per_chapter: tuple[ChapterReplayRecord, ...]
    per_scene: tuple[SceneReplayRecord, ...]
    total_pe_signal: float
    final_vitals: tuple[tuple[str, float], ...]
    per_scene_evidence: tuple[ChapterSceneBakeEvidence, ...] = ()
    semantic_events_verified: int = 0
    runtime_lineage_matches: int = 0
    runtime_transitions: int = 0
    internal_rl_policy_updates: int = 0
    slow_loop_applied_operations: int = 0
    application_owner_targets_updated: tuple[str, ...] = ()
    memory_entry_delta: int = 0
    proof_gates: tuple[tuple[str, str], ...] = ()
    bake_succeeded: bool = False
    verification_failures: tuple[str, ...] = ()

    def to_evidence_payload(self) -> dict[str, Any]:
        payload = _to_jsonable(self)
        if not isinstance(payload, dict):
            raise RuntimeError("chapter bake evidence payload must be an object")
        return payload

    def require_success(self) -> None:
        if not self.bake_succeeded:
            raise RuntimeError(
                "chapter live-through bake did not satisfy proof gates: "
                + "; ".join(self.verification_failures)
            )


class ChapterLiveThroughDriver:
    """Replay a reviewed chapter ledger through one continuous session."""

    def __init__(
        self,
        *,
        scene_end_drains_slow_loop: bool = True,
        max_internal_rl_integration_turns: int = 3,
        evidence_confidence: float = 0.9,
        evidence_source: DialogueExternalOutcomeEvidenceSource = (
            DialogueExternalOutcomeEvidenceSource.HUMAN_REVIEW
        ),
    ) -> None:
        if not 0.0 <= evidence_confidence <= 1.0:
            raise ValueError(
                "evidence_confidence must be in [0, 1], got "
                f"{evidence_confidence!r}"
            )
        if not 1 <= max_internal_rl_integration_turns <= 6:
            raise ValueError(
                "max_internal_rl_integration_turns must be in [1, 6], got "
                f"{max_internal_rl_integration_turns!r}"
            )
        self._scene_end_drains_slow_loop = scene_end_drains_slow_loop
        self._max_internal_rl_integration_turns = (
            max_internal_rl_integration_turns
        )
        self._evidence_confidence = float(evidence_confidence)
        self._evidence_source = evidence_source

    async def run_ledger_async(
        self,
        *,
        ledger: ChapterLiveThroughLedger,
        lifeform: Lifeform,
        session_id: str | None = None,
        progress_path: Path | None = None,
        resume: bool = False,
    ) -> ChapterLiveThroughReport:
        session = lifeform.create_session(
            session_id=session_id or f"live-through-{ledger.character_id}"
        )
        if resume:
            existing_progress = _load_progress(
                progress_path=progress_path,
                source_sha256=ledger.source_sha256,
            )
        else:
            # A fresh (non-resume) run must not append onto a stale
            # progress ledger — later resume loads would mix runs.
            existing_progress = {}
            if progress_path is not None and progress_path.exists():
                progress_path.unlink()
        per_chapter: list[ChapterReplayRecord] = list(existing_progress.values())
        per_scene: list[SceneReplayRecord] = []
        per_scene_evidence: list[ChapterSceneBakeEvidence] = []
        total_pe = sum(record.total_pe_signal for record in per_chapter)
        for chapter in ledger.chapters:
            if chapter.chapter_id in existing_progress:
                continue
            (
                chapter_record,
                scene_records,
                scene_evidence,
            ) = await self.run_chapter_async(
                chapter=chapter,
                session=session,
            )
            per_chapter.append(chapter_record)
            per_scene.extend(scene_records)
            per_scene_evidence.extend(scene_evidence)
            total_pe += chapter_record.total_pe_signal
            _append_progress(
                progress_path=progress_path,
                source_sha256=ledger.source_sha256,
                record=chapter_record,
            )
        failures = tuple(
            failure
            for record in per_chapter
            for failure in record.verification_failures
        )
        all_chapters_verified = (
            len(per_chapter) == len(ledger.chapters)
            and all(record.bake_verified for record in per_chapter)
        )
        semantic_events_submitted = sum(
            record.semantic_events_submitted for record in per_chapter
        )
        semantic_events_verified = sum(
            record.semantic_events_verified for record in per_chapter
        )
        scenes_processed = sum(record.scenes_processed for record in per_chapter)
        settled_outcomes = sum(
            record.environment_outcomes_settled for record in per_chapter
        )
        runtime_lineage_matches = sum(
            record.runtime_lineage_matches for record in per_chapter
        )
        runtime_transitions = sum(
            record.runtime_transitions for record in per_chapter
        )
        internal_rl_updates = sum(
            record.internal_rl_policy_updates for record in per_chapter
        )
        slow_loop_operations = sum(
            record.slow_loop_applied_operations for record in per_chapter
        )
        application_targets = tuple(
            dict.fromkeys(
                target
                for record in per_chapter
                for target in record.application_owner_targets_updated
            )
        )
        memory_entry_delta = sum(
            record.memory_entry_delta for record in per_chapter
        )
        proof_gates = (
            (
                "full_chapter_coverage",
                "pass" if len(per_chapter) == len(ledger.chapters) else "fail",
            ),
            (
                "semantic_events_reached_declared_owners_after_outcome",
                (
                    "pass"
                    if (
                        semantic_events_verified == semantic_events_submitted
                        and all(
                            record.semantic_events_applied_after_outcome
                            for record in per_chapter
                        )
                    )
                    else "fail"
                ),
            ),
            (
                "canonical_outcomes_settled_by_prediction_error",
                "pass" if settled_outcomes == scenes_processed else "fail",
            ),
            (
                "runtime_replay_lineage_reached_internal_rl",
                (
                    "pass"
                    if scenes_processed == 0
                    or (
                        runtime_lineage_matches >= scenes_processed * 2
                        and runtime_transitions >= scenes_processed * 2
                    )
                    else "fail"
                ),
            ),
            (
                "internal_rl_updated_z_policy",
                (
                    "pass"
                    if scenes_processed == 0
                    or internal_rl_updates == scenes_processed
                    else "fail"
                ),
            ),
            (
                "background_slow_memory_and_policy_integration",
                (
                    "pass"
                    if scenes_processed == 0
                    or (
                        slow_loop_operations > 0
                        and memory_entry_delta > 0
                        and bool(application_targets)
                    )
                    else "fail"
                ),
            ),
        )
        bake_succeeded = all_chapters_verified and all(
            status == "pass" for _gate, status in proof_gates
        )
        if not bake_succeeded and not failures:
            failures = tuple(
                f"{gate}:{status}"
                for gate, status in proof_gates
                if status != "pass"
            )
        return ChapterLiveThroughReport(
            ledger_character_id=ledger.character_id,
            chapters_processed=len(per_chapter),
            experienced_chapters=sum(
                1 for c in ledger.chapters if c.coverage is ChapterCoverageKind.EXPERIENCED
            ),
            learned_chapters=sum(
                1 for c in ledger.chapters if c.coverage is ChapterCoverageKind.LEARNED
            ),
            not_known_chapters=sum(
                1 for c in ledger.chapters if c.coverage is ChapterCoverageKind.NOT_KNOWN
            ),
            no_change_chapters=sum(
                1 for c in ledger.chapters if c.coverage is ChapterCoverageKind.NO_CHANGE
            ),
            per_chapter=tuple(per_chapter),
            per_scene=tuple(per_scene),
            total_pe_signal=total_pe,
            final_vitals=_drive_levels_from_session(session),
            per_scene_evidence=tuple(per_scene_evidence),
            semantic_events_verified=semantic_events_verified,
            runtime_lineage_matches=runtime_lineage_matches,
            runtime_transitions=runtime_transitions,
            internal_rl_policy_updates=internal_rl_updates,
            slow_loop_applied_operations=slow_loop_operations,
            application_owner_targets_updated=application_targets,
            memory_entry_delta=memory_entry_delta,
            proof_gates=proof_gates,
            bake_succeeded=bake_succeeded,
            verification_failures=failures,
        )

    def run_ledger(
        self,
        *,
        ledger: ChapterLiveThroughLedger,
        lifeform: Lifeform,
        session_id: str | None = None,
        progress_path: Path | None = None,
        resume: bool = False,
    ) -> ChapterLiveThroughReport:
        return asyncio.run(
            self.run_ledger_async(
                ledger=ledger,
                lifeform=lifeform,
                session_id=session_id,
                progress_path=progress_path,
                resume=resume,
            )
        )

    async def run_chapter_async(
        self,
        *,
        chapter: ReviewedChapterExperience,
        session: LifeformSession,
    ) -> tuple[
        ChapterReplayRecord,
        tuple[SceneReplayRecord, ...],
        tuple[ChapterSceneBakeEvidence, ...],
    ]:
        submitted_ids: tuple[str, ...] = ()
        semantic_verified_ids: tuple[str, ...] = ()
        semantic_owner_slots: tuple[str, ...] = ()

        scene_records: list[SceneReplayRecord] = []
        scene_evidence: list[ChapterSceneBakeEvidence] = []
        chapter_pe = 0.0
        for scene in chapter.scenes:
            record, evidence = await self._replay_one_scene(
                scene=scene,
                session=session,
            )
            scene_records.append(record)
            scene_evidence.append(evidence)
            chapter_pe += record.pe_magnitude

        # Reviewed semantic conclusions are acquired only after the chapter's
        # lived choices and outcomes. Applying them before replay would leak
        # future knowledge into the decision turn and would not be live-through.
        if chapter.semantic_events:
            submitted_ids = session.submit_semantic_events(
                chapter.semantic_event_bundle().to_external_batch()
            )
            await session.run_turn(
                f"回看第 {chapter.chapter_index} 章已经发生的经历，整理内在变化。",
                trigger_kind=TurnTriggerKind.APPRENTICE,
            )
            semantic_verified_ids, semantic_owner_slots = (
                _verify_semantic_owner_delivery(
                    events=chapter.semantic_events,
                    session=session,
                )
            )

        if chapter.semantic_events or chapter.coverage in {
            ChapterCoverageKind.LEARNED,
            ChapterCoverageKind.NO_CHANGE,
        }:
            await session.end_scene(
                reason=f"chapter-live-through:{chapter.chapter_id}:{chapter.coverage.value}",
                drain_slow_loop=self._scene_end_drains_slow_loop,
            )

        verification_failures: list[str] = []
        missing_semantic_ids = tuple(
            event_id
            for event_id in submitted_ids
            if event_id not in semantic_verified_ids
        )
        if missing_semantic_ids:
            verification_failures.append(
                "semantic-events-not-visible-in-declared-owner:"
                + ",".join(missing_semantic_ids)
            )
        for evidence in scene_evidence:
            verification_failures.extend(
                f"{evidence.scene_id}:{failure}"
                for failure in evidence.verification_failures
            )
        chapter_verified = not verification_failures
        application_targets = tuple(
            dict.fromkeys(
                target
                for evidence in scene_evidence
                for target in evidence.application_owner_targets_updated
            )
        )
        return (
            ChapterReplayRecord(
                chapter_id=chapter.chapter_id,
                chapter_index=chapter.chapter_index,
                coverage=chapter.coverage.value,
                semantic_events_submitted=len(submitted_ids),
                scenes_processed=len(scene_records),
                total_pe_signal=chapter_pe,
                final_drive_levels=_drive_levels_from_session(session),
                notes=chapter.notes,
                semantic_events_verified=len(semantic_verified_ids),
                semantic_owner_slots_verified=semantic_owner_slots,
                semantic_events_applied_after_outcome=True,
                environment_outcomes_settled=sum(
                    int(
                        evidence.evaluated_prediction_id
                        == evidence.prediction_id
                        and evidence.pe_environment_outcome_id
                        == evidence.environment_outcome_id
                    )
                    for evidence in scene_evidence
                ),
                runtime_lineage_matches=sum(
                    evidence.runtime_lineage_match_delta
                    for evidence in scene_evidence
                ),
                runtime_transitions=sum(
                    evidence.runtime_transition_delta
                    for evidence in scene_evidence
                ),
                internal_rl_policy_updates=sum(
                    int(evidence.internal_rl_policy_update_applied)
                    for evidence in scene_evidence
                ),
                slow_loop_applied_operations=sum(
                    len(evidence.slow_loop_applied_operations)
                    for evidence in scene_evidence
                ),
                application_owner_targets_updated=application_targets,
                memory_entry_delta=sum(
                    evidence.memory_entry_delta
                    for evidence in scene_evidence
                ),
                bake_verified=chapter_verified,
                verification_failures=tuple(verification_failures),
            ),
            tuple(scene_records),
            tuple(scene_evidence),
        )

    async def _replay_one_scene(
        self,
        *,
        scene: Any,
        session: LifeformSession,
    ) -> tuple[SceneReplayRecord, ChapterSceneBakeEvidence]:
        await session.run_turn(
            scene.setting,
            trigger_kind=TurnTriggerKind.APPRENTICE,
        )
        decision_result = await session.run_turn(
            scene.decision_point,
            trigger_kind=TurnTriggerKind.USER_INPUT,
        )
        prediction = decision_result.next_prediction
        if prediction is None or not prediction.prediction_id:
            raise RuntimeError(
                "chapter live-through decision turn did not publish an "
                f"owner-issued prediction id for scene {scene.scene_id!r}"
            )
        if not decision_result.environment_event_id:
            raise RuntimeError(
                "chapter live-through decision turn did not publish a "
                f"canonical environment event id for scene {scene.scene_id!r}"
            )
        prediction_id = prediction.prediction_id
        canonical_action_id = f"chapter-replay:{scene.scene_id}:canonical-action"
        environment_outcome_id = (
            f"chapter-replay:{scene.scene_id}:outcome:{prediction_id}"
        )
        runtime_before = decision_result.runtime_replay_report
        settled_before = (
            runtime_before.settled_count if runtime_before is not None else 0
        )
        transitions_before = (
            runtime_before.transition_count if runtime_before is not None else 0
        )
        lineage_before = (
            runtime_before.lineage_match_count if runtime_before is not None else 0
        )
        world_z_t = _controller_code(
            decision_result.track_z_t_codes,
            "world",
        )
        self_z_t = _controller_code(
            decision_result.track_z_t_codes,
            "self",
        )
        memory_entries_before = session.memory_entry_count()
        outcome_kind = _REGISTER_TO_OUTCOME[scene.emotional_register]
        session.submit_dialogue_outcome(
            kind=outcome_kind,
            source=self._evidence_source,
            confidence=self._evidence_confidence,
            evidence_ref=f"chapter-replay:{scene.scene_id}:{scene.canonical_action[:80]}",
            description=(
                f"Chapter live-through outcome for {scene.scene_id}; "
                f"canonical: {scene.canonical_outcome[:160]}"
            ),
        )
        # A scene terminal is an externally reviewable fact, not a reward.
        # action_payoff intentionally remains unset: PE owns the comparison and
        # Internal RL consumes the PE owner's ActualOutcome, never a character-
        # vertical reward formula.
        session.submit_environment_outcome(
            EnvironmentOutcome(
                outcome_id=environment_outcome_id,
                event_id=decision_result.environment_event_id,
                outcome_kind=EnvironmentEventKind.SCENE_EVENT,
                action_id=canonical_action_id,
                status="observed",
                summary=scene.canonical_action,
                detail=(
                    f"Reviewed canonical outcome for {scene.scene_id}: "
                    f"{scene.canonical_outcome}"
                ),
                confidence=self._evidence_confidence,
                prediction_id=prediction_id,
                evidence=(scene.evidence_locator,),
                measurement=EnvironmentMeasurement(
                    task_progress=1.0,
                    terminal=True,
                ),
                action_schema=scene.canonical_action_schema,
                situation_summary=(
                    f"{scene.setting}\nDecision point: "
                    f"{scene.decision_point}"
                ),
            )
        )
        assimilation_result = await session.run_turn(
            (
                f"我实际采取了行动：{scene.canonical_action}\n"
                f"随后发生的结果：{scene.canonical_outcome}"
            ),
            trigger_kind=TurnTriggerKind.APPRENTICE,
        )
        integration_result = None
        learning_cycle = None
        for _attempt in range(self._max_internal_rl_integration_turns):
            integration_result = await session.run_turn(
                "把这次选择、实际结果和由此产生的落差整合为本章经验。",
                trigger_kind=TurnTriggerKind.APPRENTICE,
            )
            candidate_cycle = integration_result.joint_cycle_report
            if (
                candidate_cycle is not None
                and candidate_cycle.backend_name == "runtime-replay"
                and candidate_cycle.policy_update_applied
            ):
                learning_cycle = candidate_cycle
                break
            replay_report = integration_result.runtime_replay_report
            if replay_report is None or replay_report.wiring_level != "active":
                break
        if integration_result is None:
            raise RuntimeError("chapter live-through integration turn did not run")
        if learning_cycle is None:
            candidate_cycle = integration_result.joint_cycle_report
            if candidate_cycle is not None:
                learning_cycle = candidate_cycle

        await session.end_scene(
            reason=f"chapter-live-through-scene-end:{scene.scene_id}",
            drain_slow_loop=self._scene_end_drains_slow_loop,
        )
        slow_results = session.latest_session_post_slow_loop_results
        slow_applied_operations = tuple(
            operation
            for result in slow_results
            if result.writeback_result is not None
            for operation in result.writeback_result.applied_operations
        )
        application_targets = tuple(
            dict.fromkeys(
                target
                for result in slow_results
                if result.application_prior_writeback_report is not None
                for target in result.application_prior_writeback_report.applied_targets
            )
        )
        delayed_credit_published = any(
            result.delayed_credit_summary is not None for result in slow_results
        )
        experienced_action_family_ids = tuple(
            dict.fromkeys(
                family_id
                for result in slow_results
                for family_id in result.experienced_action_family_ids
            )
        )
        schema_free_action_family_persisted = any(
            result.schema_free_action_family_ids
            and result.application_prior_writeback_report is not None
            and any(
                target.startswith(
                    "application.case_memory.records.experienced-action."
                )
                for target in (
                    result.application_prior_writeback_report.applied_targets
                )
            )
            for result in slow_results
        )
        runtime_after = integration_result.runtime_replay_report
        settled_delta = max(
            0,
            (
                runtime_after.settled_count if runtime_after is not None else 0
            )
            - settled_before,
        )
        transition_delta = max(
            0,
            (
                runtime_after.transition_count
                if runtime_after is not None
                else 0
            )
            - transitions_before,
        )
        lineage_delta = max(
            0,
            (
                runtime_after.lineage_match_count
                if runtime_after is not None
                else 0
            )
            - lineage_before,
        )
        evaluated_prediction_id = (
            assimilation_result.evaluated_prediction.prediction_id
            if assimilation_result.evaluated_prediction is not None
            else ""
        )
        actual_outcome = assimilation_result.actual_outcome
        pe_outcome_id = (
            actual_outcome.action_context.environment_outcome_id
            if actual_outcome is not None
            else ""
        )
        memory_entry_delta = max(
            0,
            session.memory_entry_count() - memory_entries_before,
        )
        failures: list[str] = []
        if evaluated_prediction_id != prediction_id:
            failures.append("prediction-error-did-not-settle-issued-prediction")
        if pe_outcome_id != environment_outcome_id:
            failures.append("prediction-error-missing-environment-outcome-lineage")
        runtime_wiring = (
            runtime_after.wiring_level if runtime_after is not None else "missing"
        )
        if runtime_wiring != "active":
            failures.append("internal-rl-runtime-replay-not-active")
        if settled_delta < 2 or lineage_delta < 2 or transition_delta < 2:
            failures.append("world-self-runtime-replay-lineage-incomplete")
        if (
            learning_cycle is None
            or learning_cycle.backend_name != "runtime-replay"
        ):
            failures.append("internal-rl-did-not-consume-runtime-replay")
        if learning_cycle is None or not learning_cycle.policy_update_applied:
            failures.append("internal-rl-policy-update-not-applied")
        if memory_entry_delta <= 0:
            failures.append("memory-owner-did-not-grow")
        if not delayed_credit_published:
            failures.append("background-slow-delayed-credit-missing")
        if not any(
            operation.startswith("temporal-prior:")
            for operation in slow_applied_operations
        ):
            failures.append("background-slow-policy-integration-missing")
        if not application_targets:
            failures.append("application-owner-integration-missing")

        scene_record = SceneReplayRecord(
            scene_id=scene.scene_id,
            phase_label=scene.phase_label,
            predicted_action_snippet=_truncate(decision_result.response.text),
            canonical_action=scene.canonical_action,
            outcome_kind=outcome_kind.value,
            pe_magnitude=(
                assimilation_result.prediction_error.magnitude
                if assimilation_result.prediction_error is not None
                else _pe_magnitude(assimilation_result.active_snapshots)
            ),
            active_regime=decision_result.active_regime,
            drive_level_after=_drive_levels_from_session(session),
        )
        scene_evidence = ChapterSceneBakeEvidence(
            scene_id=scene.scene_id,
            prediction_id=prediction_id,
            environment_event_id=decision_result.environment_event_id,
            environment_outcome_id=environment_outcome_id,
            evaluated_prediction_id=evaluated_prediction_id,
            pe_environment_outcome_id=pe_outcome_id,
            pe_magnitude=scene_record.pe_magnitude,
            world_z_t=world_z_t,
            self_z_t=self_z_t,
            runtime_replay_wiring=runtime_wiring,
            runtime_settled_delta=settled_delta,
            runtime_transition_delta=transition_delta,
            runtime_lineage_match_delta=lineage_delta,
            internal_rl_backend=(
                learning_cycle.backend_name
                if learning_cycle is not None
                else "missing"
            ),
            internal_rl_policy_update_applied=bool(
                learning_cycle is not None
                and learning_cycle.backend_name == "runtime-replay"
                and learning_cycle.policy_update_applied
            ),
            internal_rl_policy_epochs=(
                learning_cycle.policy_epochs_executed
                if learning_cycle is not None
                else 0
            ),
            internal_rl_total_reward=(
                learning_cycle.total_reward
                if learning_cycle is not None
                else 0.0
            ),
            internal_rl_rollback_reasons=(
                learning_cycle.rollback_reasons
                if learning_cycle is not None
                else ()
            ),
            memory_entry_delta=memory_entry_delta,
            slow_loop_completed=bool(slow_results),
            slow_loop_applied_operations=slow_applied_operations,
            application_owner_targets_updated=application_targets,
            delayed_credit_published=delayed_credit_published,
            bake_verified=not failures,
            verification_failures=tuple(failures),
            experienced_action_family_ids=experienced_action_family_ids,
            schema_free_action_family_persisted=(
                schema_free_action_family_persisted
            ),
        )
        return scene_record, scene_evidence


def _verify_semantic_owner_delivery(
    *,
    events: tuple[Any, ...],
    session: LifeformSession,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    deliveries = session.semantic_event_delivery(
        tuple((event.event_id, event.target_slot) for event in events)
    )
    return (
        tuple(delivery.event_id for delivery in deliveries),
        tuple(
            dict.fromkeys(delivery.target_slot for delivery in deliveries)
        ),
    )


def _controller_code(
    track_z_t_codes: tuple[tuple[str, tuple[float, ...]], ...],
    track_name: str,
) -> tuple[float, ...]:
    for name, code in track_z_t_codes:
        if name == track_name:
            return tuple(float(value) for value in code)
    return ()


__all__ = [
    "ChapterLiveThroughDriver",
    "ChapterLiveThroughReport",
    "ChapterReplayRecord",
    "ChapterSceneBakeEvidence",
]


def _load_progress(
    *,
    progress_path: Path | None,
    source_sha256: str,
) -> dict[str, ChapterReplayRecord]:
    if progress_path is None or not progress_path.exists():
        return {}
    completed: dict[str, ChapterReplayRecord] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("source_sha256") != source_sha256:
            raise ValueError(
                "chapter replay progress source_sha256 mismatch: "
                f"expected {source_sha256}, got {payload.get('source_sha256')}"
            )
        record_raw = payload.get("record")
        if not isinstance(record_raw, dict):
            raise ValueError("chapter replay progress line missing record object")
        record = ChapterReplayRecord(
            chapter_id=str(record_raw["chapter_id"]),
            chapter_index=int(record_raw["chapter_index"]),
            coverage=str(record_raw["coverage"]),
            semantic_events_submitted=int(record_raw["semantic_events_submitted"]),
            scenes_processed=int(record_raw["scenes_processed"]),
            total_pe_signal=float(record_raw["total_pe_signal"]),
            final_drive_levels=tuple(
                (str(name), float(level))
                for name, level in record_raw.get("final_drive_levels", ())
            ),
            notes=tuple(str(note) for note in record_raw.get("notes", ())),
            semantic_events_verified=int(
                record_raw.get("semantic_events_verified", 0)
            ),
            semantic_owner_slots_verified=tuple(
                str(slot)
                for slot in record_raw.get(
                    "semantic_owner_slots_verified",
                    (),
                )
            ),
            semantic_events_applied_after_outcome=bool(
                record_raw.get("semantic_events_applied_after_outcome", False)
            ),
            environment_outcomes_settled=int(
                record_raw.get("environment_outcomes_settled", 0)
            ),
            runtime_lineage_matches=int(
                record_raw.get("runtime_lineage_matches", 0)
            ),
            runtime_transitions=int(
                record_raw.get("runtime_transitions", 0)
            ),
            internal_rl_policy_updates=int(
                record_raw.get("internal_rl_policy_updates", 0)
            ),
            slow_loop_applied_operations=int(
                record_raw.get("slow_loop_applied_operations", 0)
            ),
            application_owner_targets_updated=tuple(
                str(target)
                for target in record_raw.get(
                    "application_owner_targets_updated",
                    (),
                )
            ),
            memory_entry_delta=int(record_raw.get("memory_entry_delta", 0)),
            bake_verified=bool(record_raw.get("bake_verified", False)),
            verification_failures=tuple(
                str(failure)
                for failure in record_raw.get("verification_failures", ())
            ),
        )
        completed[record.chapter_id] = record
    return completed


def _append_progress(
    *,
    progress_path: Path | None,
    source_sha256: str,
    record: ChapterReplayRecord,
) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_sha256": source_sha256,
        "record": _to_jsonable(record),
    }
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    return str(value)
