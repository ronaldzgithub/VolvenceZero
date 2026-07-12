"""Chapter ledger IO and TXT scaffold helpers.

The helpers here are operator tooling. They identify structural chapter
boundaries and produce a review scaffold; they do not infer character
behaviour from novel text.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from volvence_zero.semantic_state import SemanticProposalOperation

from lifeform_domain_character.chapter_experience import (
    ChapterCoverageKind,
    ChapterLiveThroughLedger,
    CharacterSemanticEvent,
    ReviewedChapterExperience,
)
from lifeform_domain_character.narrative import NarrativeScene


_CHAPTER_HEADING_RE = re.compile(r"^\s*第[一二三四五六七八九十百千零〇两\d]+[章节回].*$")


@dataclass(frozen=True)
class SourceChapter:
    chapter_id: str
    chapter_index: int
    title: str
    text_sha256: str
    char_count: int
    text: str


def read_text_with_detected_encoding(path: Path) -> tuple[str, str, str]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding), encoding, digest
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}:{exc.reason}")
    raise UnicodeDecodeError(
        "unknown",
        data,
        0,
        min(len(data), 1),
        "unable to decode text with utf-8-sig/utf-8/gb18030/gbk: "
        + "; ".join(errors),
    )


def split_source_chapters(text: str) -> tuple[SourceChapter, ...]:
    """Split TXT by structural Chinese chapter headings.

    When no heading is found, the whole document becomes chapter 0 so the
    operator still gets an explicit review item instead of a silent skip.
    """

    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if _CHAPTER_HEADING_RE.match(line):
            headings.append((index, line.strip()))
    if not headings:
        stripped = text.strip()
        return (
            SourceChapter(
                chapter_id="ch-0",
                chapter_index=0,
                title="Document",
                text_sha256=hashlib.sha256(stripped.encode("utf-8")).hexdigest(),
                char_count=len(stripped),
                text=stripped,
            ),
        )

    chapters: list[SourceChapter] = []
    for pos, (start, title) in enumerate(headings):
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        chapters.append(
            SourceChapter(
                chapter_id=f"ch-{pos}",
                chapter_index=pos,
                title=title,
                text_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                char_count=len(body),
                text=body,
            )
        )
    return tuple(chapters)


def build_review_scaffold(
    *,
    novel_path: Path,
    character_id: str,
    source_title: str,
    reviewed_by: str,
) -> ChapterLiveThroughLedger:
    text, encoding, source_sha256 = read_text_with_detected_encoding(novel_path)
    chapters = split_source_chapters(text)
    experiences = tuple(
        ReviewedChapterExperience(
            chapter_id=chapter.chapter_id,
            chapter_index=chapter.chapter_index,
            chapter_title=chapter.title,
            coverage=ChapterCoverageKind.NO_CHANGE,
            evidence_locator=f"{chapter.chapter_id} sha256:{chapter.text_sha256}",
            reviewed_by=reviewed_by,
            source_provenance=f"{novel_path.as_posix()} encoding:{encoding}",
            epistemic_cutoff_locator=chapter.chapter_id,
            notes=(f"review-required: {chapter.char_count} chars",),
        )
        for chapter in chapters
    )
    return ChapterLiveThroughLedger(
        character_id=character_id,
        source_title=source_title,
        source_sha256=source_sha256,
        chapters=experiences,
        reviewed_by=reviewed_by,
    )


def write_ledger_json(ledger: ChapterLiveThroughLedger, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(ledger), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_ledger_json(path: Path) -> ChapterLiveThroughLedger:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ChapterLiveThroughLedger JSON must be an object")
    return _ledger_from_json(raw)


def _ledger_from_json(raw: dict[str, Any]) -> ChapterLiveThroughLedger:
    chapters = tuple(_experience_from_json(item) for item in raw.get("chapters", ()))
    return ChapterLiveThroughLedger(
        character_id=str(raw["character_id"]),
        source_title=str(raw["source_title"]),
        source_sha256=str(raw["source_sha256"]),
        chapters=chapters,
        reviewed_by=str(raw["reviewed_by"]),
    )


def _experience_from_json(raw: dict[str, Any]) -> ReviewedChapterExperience:
    scenes = tuple(_scene_from_json(item) for item in raw.get("scenes", ()))
    events = tuple(_semantic_event_from_json(item) for item in raw.get("semantic_events", ()))
    return ReviewedChapterExperience(
        chapter_id=str(raw["chapter_id"]),
        chapter_index=int(raw["chapter_index"]),
        chapter_title=str(raw["chapter_title"]),
        coverage=ChapterCoverageKind(str(raw["coverage"])),
        evidence_locator=str(raw["evidence_locator"]),
        reviewed_by=str(raw["reviewed_by"]),
        source_provenance=str(raw["source_provenance"]),
        epistemic_cutoff_locator=str(raw["epistemic_cutoff_locator"]),
        known_facts=tuple(str(item) for item in raw.get("known_facts", ())),
        excluded_facts=tuple(str(item) for item in raw.get("excluded_facts", ())),
        scenes=scenes,
        semantic_events=events,
        notes=tuple(str(item) for item in raw.get("notes", ())),
    )


def _scene_from_json(raw: dict[str, Any]) -> NarrativeScene:
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


def _semantic_event_from_json(raw: dict[str, Any]) -> CharacterSemanticEvent:
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


__all__ = [
    "SourceChapter",
    "build_review_scaffold",
    "read_ledger_json",
    "read_text_with_detected_encoding",
    "split_source_chapters",
    "write_ledger_json",
]
