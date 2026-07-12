"""LLM-assisted chapter live-through extraction with mandatory review.

The LLM only proposes chapter artifacts. The returned candidate is not
accepted by runtime code until ``review_chapter_ledger`` is called with a
non-empty reviewer id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from json import JSONDecodeError
from typing import Any, Protocol

from lifeform_domain_character.chapter_artifacts import SourceChapter
from lifeform_domain_character.chapter_experience import (
    ChapterCoverageKind,
    ChapterLiveThroughLedger,
    CharacterSemanticEvent,
    ReviewedChapterExperience,
)
from lifeform_domain_character.narrative import NarrativeScene
from volvence_zero.semantic_state import SemanticProposalOperation


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


@dataclass(frozen=True)
class ChapterLedgerCandidate:
    character_id: str
    source_title: str
    source_sha256: str
    chapters: tuple[ReviewedChapterExperience, ...]
    failed_chapters: tuple[str, ...]
    requires_review: tuple[str, ...]
    prompt_version: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.character_id.strip():
            raise ValueError("ChapterLedgerCandidate.character_id must be non-empty")
        if not self.source_sha256.strip():
            raise ValueError("ChapterLedgerCandidate.source_sha256 must be non-empty")


def load_chapter_live_through_prompt() -> str:
    return (
        files("lifeform_domain_character")
        .joinpath("prompts", "chapter_live_through.system.md")
        .read_text(encoding="utf-8")
    )


def load_chapter_live_through_schema() -> str:
    return (
        files("lifeform_domain_character")
        .joinpath("schemas", "chapter_live_through.schema.json")
        .read_text(encoding="utf-8")
    )


def extract_chapter_ledger_candidate(
    *,
    chapters: tuple[SourceChapter, ...],
    llm_runtime: _GenerateProtocol,
    character_id: str,
    character_name: str,
    source_title: str,
    source_sha256: str,
    max_new_tokens: int = 4096,
) -> ChapterLedgerCandidate:
    if not chapters:
        raise ValueError("extract_chapter_ledger_candidate: chapters must be non-empty")
    prompt_template = load_chapter_live_through_prompt()
    schema_text = load_chapter_live_through_schema()
    reviewed: list[ReviewedChapterExperience] = []
    failures: list[str] = []
    requires_review: list[str] = []
    notes: list[str] = []
    for chapter in chapters:
        prompt = _render_prompt(
            template=prompt_template,
            schema_text=schema_text,
            character_id=character_id,
            character_name=character_name,
            source_title=source_title,
            chapter=chapter,
        )
        try:
            raw = llm_runtime.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            parsed = _parse_chapter_response(
                raw,
                source_chapter=chapter,
                reviewed_by="llm-candidate",
                source_provenance=f"llm-candidate:{source_title}:{chapter.chapter_id}",
            )
        except (JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            failures.append(chapter.chapter_id)
            notes.append(f"{chapter.chapter_id}: {type(exc).__name__}: {exc}")
            continue
        reviewed.append(parsed)
        if _chapter_requires_review(parsed):
            requires_review.append(parsed.chapter_id)
    return ChapterLedgerCandidate(
        character_id=character_id,
        source_title=source_title,
        source_sha256=source_sha256,
        chapters=tuple(sorted(reviewed, key=lambda item: item.chapter_index)),
        failed_chapters=tuple(failures),
        requires_review=tuple(requires_review),
        prompt_version="chapter_live_through.system.v1",
        notes=tuple(notes),
    )


def review_chapter_ledger(
    candidate: ChapterLedgerCandidate,
    *,
    reviewer: str,
    expected_chapters: tuple[SourceChapter, ...],
) -> ChapterLiveThroughLedger:
    if not reviewer.strip():
        raise ValueError("review_chapter_ledger: reviewer must be non-empty")
    if reviewer.strip() in {"llm-candidate", "operator-review-required"}:
        raise ValueError(
            "review_chapter_ledger: reviewer must be a real accountable "
            f"reviewer id, got placeholder {reviewer!r}"
        )
    if candidate.failed_chapters:
        raise ValueError(
            "review_chapter_ledger: candidate has failed chapters "
            f"{candidate.failed_chapters!r}; re-run or mark them explicitly."
        )
    expected_ids = tuple(chapter.chapter_id for chapter in expected_chapters)
    actual_ids = tuple(chapter.chapter_id for chapter in candidate.chapters)
    if actual_ids != expected_ids:
        raise ValueError(
            "review_chapter_ledger: chapter coverage mismatch; "
            f"expected {expected_ids!r}, got {actual_ids!r}"
        )
    by_id = {chapter.chapter_id: chapter for chapter in expected_chapters}
    reviewed_chapters: list[ReviewedChapterExperience] = []
    for chapter in candidate.chapters:
        source = by_id[chapter.chapter_id]
        if f"sha256:{source.text_sha256}" not in chapter.evidence_locator:
            raise ValueError(
                "review_chapter_ledger: evidence_locator for "
                f"{chapter.chapter_id!r} must include sha256:{source.text_sha256}"
            )
        reviewed_chapters.append(
            ReviewedChapterExperience(
                chapter_id=chapter.chapter_id,
                chapter_index=chapter.chapter_index,
                chapter_title=chapter.chapter_title,
                coverage=chapter.coverage,
                evidence_locator=chapter.evidence_locator,
                reviewed_by=reviewer,
                source_provenance=chapter.source_provenance,
                epistemic_cutoff_locator=chapter.epistemic_cutoff_locator,
                known_facts=chapter.known_facts,
                excluded_facts=chapter.excluded_facts,
                scenes=chapter.scenes,
                semantic_events=chapter.semantic_events,
                notes=chapter.notes + (f"reviewed-from:{candidate.prompt_version}",),
            )
        )
    return ChapterLiveThroughLedger(
        character_id=candidate.character_id,
        source_title=candidate.source_title,
        source_sha256=candidate.source_sha256,
        chapters=tuple(reviewed_chapters),
        reviewed_by=reviewer,
    )


def _render_prompt(
    *,
    template: str,
    schema_text: str,
    character_id: str,
    character_name: str,
    source_title: str,
    chapter: SourceChapter,
) -> str:
    return "\n\n".join(
        (
            template,
            "JSON schema:",
            schema_text,
            f"Character id: {character_id}",
            f"Character name: {character_name}",
            f"Source title: {source_title}",
            f"Chapter id: {chapter.chapter_id}",
            f"Chapter index: {chapter.chapter_index}",
            f"Chapter title: {chapter.title}",
            f"Chapter sha256: {chapter.text_sha256}",
            "Chapter text:",
            chapter.text,
        )
    )


def _parse_chapter_response(
    raw: str,
    *,
    source_chapter: SourceChapter,
    reviewed_by: str,
    source_provenance: str,
) -> ReviewedChapterExperience:
    payload = _loads_json_object(raw)
    chapters_raw = payload.get("chapters")
    if not isinstance(chapters_raw, list) or len(chapters_raw) != 1:
        raise ValueError("chapter response must contain exactly one chapter")
    item = chapters_raw[0]
    if not isinstance(item, dict):
        raise ValueError("chapter item must be an object")
    chapter_id = str(item["chapter_id"])
    if chapter_id != source_chapter.chapter_id:
        raise ValueError(
            f"chapter_id mismatch: expected {source_chapter.chapter_id!r}, got {chapter_id!r}"
        )
    scenes = tuple(_scene_from_json(scene) for scene in item.get("scenes", ()))
    semantic_events = tuple(
        _semantic_event_from_json(event)
        for event in item.get("semantic_events", ())
    )
    evidence_locator = str(item["evidence_locator"])
    if f"sha256:{source_chapter.text_sha256}" not in evidence_locator:
        evidence_locator = f"{evidence_locator} sha256:{source_chapter.text_sha256}"
    return ReviewedChapterExperience(
        chapter_id=chapter_id,
        chapter_index=int(item["chapter_index"]),
        chapter_title=str(item["chapter_title"]),
        coverage=ChapterCoverageKind(str(item["coverage"])),
        evidence_locator=evidence_locator,
        reviewed_by=reviewed_by,
        source_provenance=source_provenance,
        epistemic_cutoff_locator=str(item["epistemic_cutoff_locator"]),
        known_facts=tuple(str(fact) for fact in item.get("known_facts", ())),
        excluded_facts=tuple(str(fact) for fact in item.get("excluded_facts", ())),
        scenes=scenes,
        semantic_events=semantic_events,
        notes=(f"llm-candidate:{source_chapter.chapter_id}",),
    )


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed


def _scene_from_json(raw: Any) -> NarrativeScene:
    if not isinstance(raw, dict):
        raise ValueError("scene must be an object")
    return NarrativeScene(
        scene_id=str(raw["scene_id"]),
        phase_label=str(raw["phase_label"]),
        setting=str(raw["setting"]),
        decision_point=str(raw["decision_point"]),
        canonical_action=str(raw["canonical_action"]),
        canonical_outcome=str(raw["canonical_outcome"]),
        emotional_register=str(raw["emotional_register"]),
        risk_markers=tuple(str(item) for item in raw.get("risk_markers", ())),
        expected_regime=raw.get("expected_regime"),
        evidence_locator=str(raw["evidence_locator"]),
    )


def _semantic_event_from_json(raw: Any) -> CharacterSemanticEvent:
    if not isinstance(raw, dict):
        raise ValueError("semantic_event must be an object")
    return CharacterSemanticEvent(
        event_id=str(raw["event_id"]),
        target_slot=str(raw["target_slot"]),
        operation=SemanticProposalOperation(str(raw["operation"])),
        summary=str(raw["summary"]),
        detail=str(raw["detail"]),
        confidence=float(raw["confidence"]),
        evidence_locator=str(raw["evidence_locator"]),
        control_signal=float(raw.get("control_signal", 0.0)),
        requires_confirmation=bool(raw.get("requires_confirmation", False)),
    )


def _chapter_requires_review(chapter: ReviewedChapterExperience) -> bool:
    return (
        chapter.coverage is ChapterCoverageKind.EXPERIENCED
        and not chapter.scenes
    ) or any(event.confidence < 0.7 for event in chapter.semantic_events)


__all__ = [
    "ChapterLedgerCandidate",
    "extract_chapter_ledger_candidate",
    "load_chapter_live_through_prompt",
    "load_chapter_live_through_schema",
    "review_chapter_ledger",
]
