"""Subjective chapter live-through driver.

The driver is an orchestrator over the existing public session surface:
``submit_semantic_events`` / ``run_turn`` / ``submit_dialogue_outcome`` /
``end_scene``. It does not own memory, semantic state, PE, or regime.
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


class ChapterLiveThroughDriver:
    """Replay a reviewed chapter ledger through one continuous session."""

    def __init__(
        self,
        *,
        scene_end_drains_slow_loop: bool = True,
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
        self._scene_end_drains_slow_loop = scene_end_drains_slow_loop
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
        total_pe = sum(record.total_pe_signal for record in per_chapter)
        for chapter in ledger.chapters:
            if chapter.chapter_id in existing_progress:
                continue
            chapter_record, scene_records = await self.run_chapter_async(
                chapter=chapter,
                session=session,
            )
            per_chapter.append(chapter_record)
            per_scene.extend(scene_records)
            total_pe += chapter_record.total_pe_signal
            _append_progress(
                progress_path=progress_path,
                source_sha256=ledger.source_sha256,
                record=chapter_record,
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
    ) -> tuple[ChapterReplayRecord, tuple[SceneReplayRecord, ...]]:
        submitted_ids: tuple[str, ...] = ()
        if chapter.semantic_events:
            submitted_ids = session.submit_semantic_events(
                chapter.semantic_event_bundle().to_external_batch()
            )
            # Drain the queued semantic events through the normal turn path.
            await session.run_turn(
                f"整理第 {chapter.chapter_index} 章已经审核的内在变化。",
                trigger_kind=TurnTriggerKind.APPRENTICE,
            )

        scene_records: list[SceneReplayRecord] = []
        chapter_pe = 0.0
        for scene in chapter.scenes:
            record = await self._replay_one_scene(scene=scene, session=session)
            scene_records.append(record)
            chapter_pe += record.pe_magnitude

        if chapter.coverage in {ChapterCoverageKind.LEARNED, ChapterCoverageKind.NO_CHANGE}:
            await session.end_scene(
                reason=f"chapter-live-through:{chapter.chapter_id}:{chapter.coverage.value}",
                drain_slow_loop=self._scene_end_drains_slow_loop,
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
            ),
            tuple(scene_records),
        )

    async def _replay_one_scene(
        self,
        *,
        scene: Any,
        session: LifeformSession,
    ) -> SceneReplayRecord:
        await session.run_turn(
            scene.setting,
            trigger_kind=TurnTriggerKind.APPRENTICE,
        )
        decision_result = await session.run_turn(
            scene.decision_point,
            trigger_kind=TurnTriggerKind.USER_INPUT,
        )
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
        await session.end_scene(
            reason=f"chapter-live-through-scene-end:{scene.scene_id}",
            drain_slow_loop=self._scene_end_drains_slow_loop,
        )
        return SceneReplayRecord(
            scene_id=scene.scene_id,
            phase_label=scene.phase_label,
            predicted_action_snippet=_truncate(decision_result.response.text),
            canonical_action=scene.canonical_action,
            outcome_kind=outcome_kind.value,
            pe_magnitude=_pe_magnitude(decision_result.active_snapshots),
            active_regime=decision_result.active_regime,
            drive_level_after=_drive_levels_from_session(session),
        )


__all__ = [
    "ChapterLiveThroughDriver",
    "ChapterLiveThroughReport",
    "ChapterReplayRecord",
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
