from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from volvence_zero.memory import (
    FileSystemPersistenceBackend,
    PersistenceBackend,
    deserialize_checkpoint,
    serialize_checkpoint,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    ascii_buffer: list[str] = []
    compact = "".join(char for char in text.lower() if not char.isspace())
    for char in text.lower():
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        if ascii_buffer:
            tokens.add("".join(ascii_buffer))
            ascii_buffer.clear()
        if not char.isspace():
            tokens.add(char)
    if ascii_buffer:
        tokens.add("".join(ascii_buffer))
    for index in range(len(compact) - 1):
        tokens.add(compact[index : index + 2])
    return tokens


@dataclass(frozen=True)
class DomainKnowledgeRecord:
    record_id: str
    domain: str
    topic_tags: tuple[str, ...]
    jurisdiction_tags: tuple[str, ...]
    source_type: str
    title: str
    locator: str
    summary: str
    snippet: str
    freshness_label: str
    confidence: float
    evidence_strength: str
    conflict_markers: tuple[str, ...] = ()
    url: str | None = None


@dataclass(frozen=True)
class DomainKnowledgeCheckpoint:
    checkpoint_id: str
    records: tuple[DomainKnowledgeRecord, ...]


@dataclass(frozen=True)
class CaseMemoryRecord:
    case_id: str
    domain: str
    problem_pattern: str
    user_state_pattern: str
    risk_markers: tuple[str, ...]
    track_tags: tuple[str, ...]
    regime_tags: tuple[str, ...]
    intervention_ordering: tuple[str, ...]
    outcome_label: str
    delayed_signal_count: int
    escalation_observed: bool
    repair_observed: bool
    confidence: float
    relevance_score: float
    description: str


@dataclass(frozen=True)
class CaseMemoryCheckpoint:
    checkpoint_id: str
    records: tuple[CaseMemoryRecord, ...]


def _reconstruct_domain_knowledge_checkpoint(parsed: dict[str, object]) -> DomainKnowledgeCheckpoint | None:
    try:
        records_raw = parsed.get("records", [])
        records = tuple(
            DomainKnowledgeRecord(
                record_id=str(record["record_id"]),
                domain=str(record["domain"]),
                topic_tags=tuple(str(tag) for tag in record.get("topic_tags", ())),
                jurisdiction_tags=tuple(str(tag) for tag in record.get("jurisdiction_tags", ())),
                source_type=str(record["source_type"]),
                title=str(record["title"]),
                locator=str(record["locator"]),
                summary=str(record["summary"]),
                snippet=str(record["snippet"]),
                freshness_label=str(record["freshness_label"]),
                confidence=float(record["confidence"]),
                evidence_strength=str(record["evidence_strength"]),
                conflict_markers=tuple(str(tag) for tag in record.get("conflict_markers", ())),
                url=str(record["url"]) if record.get("url") is not None else None,
            )
            for record in records_raw
        )
        return DomainKnowledgeCheckpoint(
            checkpoint_id=str(parsed.get("checkpoint_id", "domain-knowledge-restored")),
            records=records,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _reconstruct_case_memory_checkpoint(parsed: dict[str, object]) -> CaseMemoryCheckpoint | None:
    try:
        records_raw = parsed.get("records", [])
        records = tuple(
            CaseMemoryRecord(
                case_id=str(record["case_id"]),
                domain=str(record["domain"]),
                problem_pattern=str(record["problem_pattern"]),
                user_state_pattern=str(record["user_state_pattern"]),
                risk_markers=tuple(str(tag) for tag in record.get("risk_markers", ())),
                track_tags=tuple(str(tag) for tag in record.get("track_tags", ())),
                regime_tags=tuple(str(tag) for tag in record.get("regime_tags", ())),
                intervention_ordering=tuple(str(step) for step in record.get("intervention_ordering", ())),
                outcome_label=str(record["outcome_label"]),
                delayed_signal_count=int(record["delayed_signal_count"]),
                escalation_observed=bool(record["escalation_observed"]),
                repair_observed=bool(record["repair_observed"]),
                confidence=float(record["confidence"]),
                relevance_score=float(record["relevance_score"]),
                description=str(record["description"]),
            )
            for record in records_raw
        )
        return CaseMemoryCheckpoint(
            checkpoint_id=str(parsed.get("checkpoint_id", "case-memory-restored")),
            records=records,
        )
    except (KeyError, TypeError, ValueError):
        return None


DEFAULT_DOMAIN_KNOWLEDGE_RECORDS: tuple[DomainKnowledgeRecord, ...] = (
    DomainKnowledgeRecord(
        record_id="knowledge:family-transition:1",
        domain="family_transition",
        topic_tags=("family", "transition", "procedure"),
        jurisdiction_tags=("local-law-sensitive",),
        source_type="official-guide",
        title="Family transition basics",
        locator="phase1-seed",
        summary=(
            "Separate emotional stabilization from legal or procedural next steps, and keep any child-safety "
            "or jurisdiction-sensitive guidance explicitly bounded."
        ),
        snippet="High-level family transition guidance; local specifics must be confirmed before conclusions.",
        freshness_label="seed-current",
        confidence=0.72,
        evidence_strength="medium",
        conflict_markers=(),
    ),
    DomainKnowledgeRecord(
        record_id="knowledge:professional-process:1",
        domain="professional_process",
        topic_tags=("professional", "process", "bounded-advice"),
        jurisdiction_tags=("local-law-sensitive",),
        source_type="official-guide",
        title="Professional process basics",
        locator="phase1-seed",
        summary=(
            "Use sourced high-level process guidance first, and avoid definitive professional conclusions "
            "before local specifics are confirmed."
        ),
        snippet="Process guidance should stay sourced and bounded until jurisdiction is known.",
        freshness_label="seed-current",
        confidence=0.70,
        evidence_strength="medium",
        conflict_markers=(),
    ),
    DomainKnowledgeRecord(
        record_id="knowledge:career-decision:1",
        domain="career_decision",
        topic_tags=("career", "tradeoff", "next-step"),
        jurisdiction_tags=("general",),
        source_type="reviewed-article",
        title="Career decision framing",
        locator="phase1-seed",
        summary=(
            "Frame trade-offs explicitly, reduce ambiguity, and prefer the smallest next step over a full "
            "life-plan answer."
        ),
        snippet="Break career decisions into trade-offs and smallest next actions.",
        freshness_label="seed-current",
        confidence=0.68,
        evidence_strength="medium",
        conflict_markers=(),
    ),
)


class ApplicationDomainKnowledgeStore:
    def __init__(
        self,
        *,
        records: tuple[DomainKnowledgeRecord, ...] = DEFAULT_DOMAIN_KNOWLEDGE_RECORDS,
        persistence_backend: PersistenceBackend | None = None,
    ) -> None:
        self._records: dict[str, DomainKnowledgeRecord] = {record.record_id: record for record in records}
        self._persistence_backend = persistence_backend
        self._persistence_version = 0

    @property
    def records(self) -> tuple[DomainKnowledgeRecord, ...]:
        return tuple(self._records.values())

    def query(
        self,
        *,
        domains: tuple[str, ...],
        query_text: str,
        jurisdiction_required: bool,
        limit: int = 3,
    ) -> tuple[DomainKnowledgeRecord, ...]:
        query_tokens = _tokenize(query_text)
        scored: list[tuple[float, DomainKnowledgeRecord]] = []
        for record in self._records.values():
            if domains and record.domain not in domains:
                continue
            summary_tokens = _tokenize(record.summary + " " + record.snippet + " " + " ".join(record.topic_tags))
            overlap = len(query_tokens & summary_tokens)
            jurisdiction_bonus = 0.08 if jurisdiction_required and "local-law-sensitive" in record.jurisdiction_tags else 0.0
            score = record.confidence + overlap * 0.04 + jurisdiction_bonus
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].record_id))
        return tuple(record for _, record in scored[:limit])

    def create_checkpoint(self, *, checkpoint_id: str) -> DomainKnowledgeCheckpoint:
        return DomainKnowledgeCheckpoint(
            checkpoint_id=checkpoint_id,
            records=self.records,
        )

    def restore_checkpoint(self, checkpoint: DomainKnowledgeCheckpoint) -> None:
        self._records = {record.record_id: record for record in checkpoint.records}

    def save_to_backend(self, *, key: str = "application/domain_knowledge") -> bool:
        if self._persistence_backend is None:
            return False
        checkpoint = self.create_checkpoint(checkpoint_id=f"persist-{key}")
        self._persistence_version += 1
        self._persistence_backend.save_checkpoint(
            key=key,
            data=serialize_checkpoint(checkpoint),
            version=self._persistence_version,
        )
        return True

    def load_from_backend(self, *, key: str = "application/domain_knowledge") -> bool:
        if self._persistence_backend is None:
            return False
        result = self._persistence_backend.load_checkpoint(key=key)
        if result is None:
            return False
        data, version = result
        parsed = deserialize_checkpoint(data)
        if not parsed:
            return False
        checkpoint = _reconstruct_domain_knowledge_checkpoint(parsed)
        if checkpoint is None:
            return False
        self.restore_checkpoint(checkpoint)
        self._persistence_version = version
        return True


class ApplicationCaseMemoryStore:
    def __init__(
        self,
        *,
        records: tuple[CaseMemoryRecord, ...] = (),
        persistence_backend: PersistenceBackend | None = None,
    ) -> None:
        self._records: dict[str, CaseMemoryRecord] = {record.case_id: record for record in records}
        self._persistence_backend = persistence_backend
        self._persistence_version = 0

    @property
    def records(self) -> tuple[CaseMemoryRecord, ...]:
        return tuple(self._records.values())

    def upsert_records(self, records: Iterable[CaseMemoryRecord]) -> None:
        for record in records:
            existing = self._records.get(record.case_id)
            if existing is None or record.relevance_score >= existing.relevance_score:
                self._records[record.case_id] = record

    def query(
        self,
        *,
        experience_domains: tuple[str, ...],
        regime_id: str | None,
        risk_band: str,
        limit: int = 3,
    ) -> tuple[CaseMemoryRecord, ...]:
        scored: list[tuple[float, CaseMemoryRecord]] = []
        for record in self._records.values():
            score = record.relevance_score
            if experience_domains and record.domain in experience_domains:
                score += 0.12
            if regime_id is not None and regime_id in record.regime_tags:
                score += 0.08
            if f"risk-{risk_band}" in record.risk_markers:
                score += 0.06
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].case_id))
        return tuple(record for _, record in scored[:limit])

    def create_checkpoint(self, *, checkpoint_id: str) -> CaseMemoryCheckpoint:
        return CaseMemoryCheckpoint(
            checkpoint_id=checkpoint_id,
            records=self.records,
        )

    def restore_checkpoint(self, checkpoint: CaseMemoryCheckpoint) -> None:
        self._records = {record.case_id: record for record in checkpoint.records}

    def save_to_backend(self, *, key: str = "application/case_memory") -> bool:
        if self._persistence_backend is None:
            return False
        checkpoint = self.create_checkpoint(checkpoint_id=f"persist-{key}")
        self._persistence_version += 1
        self._persistence_backend.save_checkpoint(
            key=key,
            data=serialize_checkpoint(checkpoint),
            version=self._persistence_version,
        )
        return True

    def load_from_backend(self, *, key: str = "application/case_memory") -> bool:
        if self._persistence_backend is None:
            return False
        result = self._persistence_backend.load_checkpoint(key=key)
        if result is None:
            return False
        data, version = result
        parsed = deserialize_checkpoint(data)
        if not parsed:
            return False
        checkpoint = _reconstruct_case_memory_checkpoint(parsed)
        if checkpoint is None:
            return False
        self.restore_checkpoint(checkpoint)
        self._persistence_version = version
        return True


def build_filesystem_persistence_backend(*, base_dir: str, max_versions: int = 5) -> PersistenceBackend:
    return FileSystemPersistenceBackend(base_dir=base_dir, max_versions=max_versions)


def build_default_domain_knowledge_store(
    *,
    persistence_backend: PersistenceBackend | None = None,
) -> ApplicationDomainKnowledgeStore:
    store = ApplicationDomainKnowledgeStore(persistence_backend=persistence_backend)
    if persistence_backend is not None:
        loaded = store.load_from_backend()
        if not loaded:
            store.save_to_backend()
    return store


def build_default_case_memory_store(
    *,
    persistence_backend: PersistenceBackend | None = None,
) -> ApplicationCaseMemoryStore:
    store = ApplicationCaseMemoryStore(persistence_backend=persistence_backend)
    if persistence_backend is not None:
        store.load_from_backend()
    return store
