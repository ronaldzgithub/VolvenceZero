"""Reviewed chapter-level subjective experience artifacts.

This module is the character vertical's contract for "live through the
book" work. It deliberately stores reviewed, structured artifacts only:
raw novel text may be used by an offline extraction/review step, but the
runtime replay path consumes these frozen records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from volvence_zero.semantic_state import (
    ExternalSemanticEventBatch,
    GenericSemanticEvent,
    SEMANTIC_OWNER_SLOTS,
    SemanticEventAdapter,
    SemanticProposal,
    SemanticProposalOperation,
)

from lifeform_domain_character.narrative import NarrativeScene


class ChapterCoverageKind(str, Enum):
    """How a chapter relates to the character's subjective timeline."""

    EXPERIENCED = "experienced"
    LEARNED = "learned"
    NOT_KNOWN = "not-known"
    NO_CHANGE = "no-change"


@dataclass(frozen=True)
class CharacterSemanticEvent:
    """A reviewed semantic-state proposal carried by a chapter artifact.

    The event is not an owner. It is a typed proposal source consumed by
    :class:`CharacterChapterSemanticAdapter`, then merged by the existing
    ``SemanticStateStore`` single writer.
    """

    event_id: str
    target_slot: str
    operation: SemanticProposalOperation
    summary: str
    detail: str
    confidence: float
    evidence_locator: str
    control_signal: float = 0.0
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        _require_non_empty("event_id", self.event_id)
        if self.target_slot not in SEMANTIC_OWNER_SLOTS:
            raise ValueError(
                "CharacterSemanticEvent.target_slot must be one of "
                f"{SEMANTIC_OWNER_SLOTS!r}, got {self.target_slot!r}"
            )
        _require_non_empty("summary", self.summary)
        _require_non_empty("detail", self.detail)
        _require_non_empty("evidence_locator", self.evidence_locator)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "CharacterSemanticEvent.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )

    def to_generic_event(self) -> GenericSemanticEvent:
        return _to_generic_event(self)


@dataclass(frozen=True)
class CharacterSemanticEventBundle:
    """Semantic events reviewed for one chapter."""

    chapter_id: str
    events: tuple[CharacterSemanticEvent, ...]
    reviewed_by: str
    source_provenance: str

    def __post_init__(self) -> None:
        _require_non_empty("chapter_id", self.chapter_id)
        _require_non_empty("reviewed_by", self.reviewed_by)
        _require_non_empty("source_provenance", self.source_provenance)
        _check_unique("CharacterSemanticEvent.event_id", tuple(e.event_id for e in self.events))

    def to_external_batch(self) -> ExternalSemanticEventBatch:
        """Return a batch suitable for ``LifeformSession.submit_semantic_events``."""

        return ExternalSemanticEventBatch(
            events=tuple(event.to_generic_event() for event in self.events),
            source="character-chapter",
            description=(
                f"Reviewed character semantic events for {self.chapter_id} "
                f"from {self.source_provenance}."
            ),
        )


@dataclass(frozen=True)
class ReviewedChapterExperience:
    """One chapter's reviewed subjective experience for a character.

    ``coverage`` is mandatory so every chapter has an audit record. Only
    ``EXPERIENCED`` chapters may carry replay scenes; chapters marked
    ``NOT_KNOWN`` must not carry semantic events or scenes because they
    describe material outside the character's epistemic boundary.
    """

    chapter_id: str
    chapter_index: int
    chapter_title: str
    coverage: ChapterCoverageKind
    evidence_locator: str
    reviewed_by: str
    source_provenance: str
    epistemic_cutoff_locator: str
    known_facts: tuple[str, ...] = ()
    excluded_facts: tuple[str, ...] = ()
    scenes: tuple[NarrativeScene, ...] = ()
    semantic_events: tuple[CharacterSemanticEvent, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("chapter_id", self.chapter_id)
        if self.chapter_index < 0:
            raise ValueError("ReviewedChapterExperience.chapter_index must be >= 0")
        _require_non_empty("chapter_title", self.chapter_title)
        _require_non_empty("evidence_locator", self.evidence_locator)
        _require_non_empty("reviewed_by", self.reviewed_by)
        _require_non_empty("source_provenance", self.source_provenance)
        _require_non_empty("epistemic_cutoff_locator", self.epistemic_cutoff_locator)
        if self.coverage is ChapterCoverageKind.NOT_KNOWN:
            if self.scenes or self.semantic_events or self.known_facts:
                raise ValueError(
                    "NOT_KNOWN chapters may only carry audit notes/excluded_facts; "
                    "they cannot seed replay, known facts, or semantic owner proposals."
                )
        if self.coverage is not ChapterCoverageKind.EXPERIENCED and self.scenes:
            raise ValueError(
                "Only EXPERIENCED chapters may carry NarrativeScene replay records."
            )
        _check_unique(
            "ReviewedChapterExperience.scene_id",
            tuple(scene.scene_id for scene in self.scenes),
        )
        _check_unique(
            "ReviewedChapterExperience.semantic_event_id",
            tuple(event.event_id for event in self.semantic_events),
        )

    def semantic_event_bundle(self) -> CharacterSemanticEventBundle:
        return CharacterSemanticEventBundle(
            chapter_id=self.chapter_id,
            events=self.semantic_events,
            reviewed_by=self.reviewed_by,
            source_provenance=self.source_provenance,
        )


def _to_generic_event(event: CharacterSemanticEvent) -> GenericSemanticEvent:
    return GenericSemanticEvent(
        event_id=event.event_id,
        target_slot=event.target_slot,
        operation=event.operation,
        summary=event.summary,
        detail=event.detail,
        confidence=event.confidence,
        evidence=event.evidence_locator,
        control_signal=event.control_signal,
        requires_confirmation=event.requires_confirmation,
    )


@dataclass(frozen=True)
class ChapterLiveThroughLedger:
    """Reviewed, ordered subjective ledger for one character."""

    character_id: str
    source_title: str
    source_sha256: str
    chapters: tuple[ReviewedChapterExperience, ...]
    reviewed_by: str

    def __post_init__(self) -> None:
        _require_non_empty("character_id", self.character_id)
        _require_non_empty("source_title", self.source_title)
        _require_non_empty("source_sha256", self.source_sha256)
        _require_non_empty("reviewed_by", self.reviewed_by)
        _check_unique(
            "ChapterLiveThroughLedger.chapter_id",
            tuple(chapter.chapter_id for chapter in self.chapters),
        )
        indexes = tuple(chapter.chapter_index for chapter in self.chapters)
        if indexes != tuple(sorted(indexes)):
            raise ValueError(
                "ChapterLiveThroughLedger.chapters must be ordered by chapter_index"
            )


class CharacterChapterSemanticAdapter(SemanticEventAdapter):
    """Adapter from reviewed character chapter events to semantic proposals."""

    def adapt(
        self,
        *,
        event: object,
        target_slot: str,
        turn_index: int,
    ) -> tuple[SemanticProposal, ...]:
        if not isinstance(event, CharacterSemanticEvent):
            return ()
        if event.target_slot != target_slot:
            return ()
        return (
            SemanticProposal(
                proposal_id=f"{event.event_id}:turn-{turn_index}",
                target_slot=event.target_slot,
                operation=event.operation,
                summary=event.summary,
                detail=event.detail,
                confidence=event.confidence,
                evidence=event.evidence_locator,
                control_signal=event.control_signal,
                requires_confirmation=event.requires_confirmation,
            ),
        )


def _require_non_empty(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _check_unique(field_name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique, got {values!r}")


__all__ = [
    "ChapterCoverageKind",
    "ChapterLiveThroughLedger",
    "CharacterChapterSemanticAdapter",
    "CharacterSemanticEvent",
    "CharacterSemanticEventBundle",
    "ReviewedChapterExperience",
]
