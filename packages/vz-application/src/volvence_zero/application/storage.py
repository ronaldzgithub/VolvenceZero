from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from volvence_zero.memory import (
    FileSystemPersistenceBackend,
    PersistenceBackend,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from volvence_zero.semantic_embedding import (
    stub_cosine_similarity as _cosine_similarity,
    stub_semantic_embedding as _semantic_embedding,
)


class CaseLifecycle(str, Enum):
    """Per-record lifecycle stage for case memory entries (Gap 4).

    ``VALIDATED`` is the default — it preserves backwards compatibility
    for every case record that existed before the lifecycle field was
    introduced. ``CANDIDATE`` / ``PROVISIONAL`` / ``RETIRED`` are the
    new states unlocked by the thinking-loop mid-session path:

    * ``CANDIDATE`` — reflection writeback produced a weak prior that
      didn't meet the full promotion threshold; may still be used for
      exploration but is lower-weight than ``VALIDATED`` for retrieval.
    * ``PROVISIONAL`` — strong-enough to inject into retrieval with an
      explicit TTL; will be promoted to ``VALIDATED`` or retired at
      ``reconcile_provisional_cases`` time.
    * ``VALIDATED`` — passed the full threshold (matches the EmoGPT
      "pattern" evidence gate: min_total_records >= 2,
      min_same_polarity >= 2, min_mean_abs_reward >= 0.15). Default
      lifecycle for all pre-Gap-4 records.
    * ``RETIRED`` — superseded, expired, or explicitly reconciled as
      a failed provisional. Retrieval must skip retired records.

    See ``docs/specs/thinking-loop.md`` for the transition semantics
    enforced by ``reconcile_provisional_cases``.
    """

    CANDIDATE = "candidate"
    PROVISIONAL = "provisional"
    VALIDATED = "validated"
    RETIRED = "retired"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


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
    continuum_profile_id: str | None = None
    continuum_band_id: str | None = None
    continuum_position: float = 0.0
    continuum_update_frequency: float = 0.0
    reconstruction_source: str = "direct"
    # Gap 4 / docs/specs/thinking-loop.md additions. Defaults preserve
    # backwards compat for every record that existed before the
    # lifecycle field was introduced.
    lifecycle: CaseLifecycle = CaseLifecycle.VALIDATED
    ttl_seconds: int | None = None
    expires_at_tick: int | None = None
    provisional_origin: str = ""

    def __post_init__(self) -> None:
        # Ownership invariants:
        # - CANDIDATE / PROVISIONAL records MAY carry a ttl and an origin.
        # - A non-VALIDATED lifecycle with neither a ttl nor an origin is
        #   probably a bug: something produced a weak record without
        #   identifying where it came from.
        # - VALIDATED / RETIRED records must NOT carry an unresolved ttl
        #   (fully validated or retired means lifecycle timer is no
        #   longer meaningful).
        if self.lifecycle in {CaseLifecycle.CANDIDATE, CaseLifecycle.PROVISIONAL}:
            has_ttl = self.ttl_seconds is not None or self.expires_at_tick is not None
            if not has_ttl and not self.provisional_origin:
                raise ValueError(
                    f"CaseMemoryRecord.lifecycle={self.lifecycle.value!r} "
                    f"requires either a ttl_seconds / expires_at_tick or "
                    f"a non-empty provisional_origin; both missing."
                )
        if self.ttl_seconds is not None and self.ttl_seconds < 0:
            raise ValueError(
                f"CaseMemoryRecord.ttl_seconds must be >= 0 when set, "
                f"got {self.ttl_seconds!r}"
            )
        if self.expires_at_tick is not None and self.expires_at_tick < 0:
            raise ValueError(
                f"CaseMemoryRecord.expires_at_tick must be >= 0 when set, "
                f"got {self.expires_at_tick!r}"
            )

    def is_available_for_retrieval(self) -> bool:
        """Return True iff this record should appear in retrieval hits.

        ``RETIRED`` records are always hidden. Other lifecycle stages
        are available; consumers that want stricter filtering (e.g.
        "validated only") should apply their own additional gate.
        """
        return self.lifecycle is not CaseLifecycle.RETIRED


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
                continuum_profile_id=(
                    str(record["continuum_profile_id"])
                    if record.get("continuum_profile_id") is not None
                    else None
                ),
                continuum_band_id=(
                    str(record["continuum_band_id"])
                    if record.get("continuum_band_id") is not None
                    else None
                ),
                continuum_position=float(record.get("continuum_position", 0.0)),
                continuum_update_frequency=float(record.get("continuum_update_frequency", 0.0)),
                reconstruction_source=str(record.get("reconstruction_source", "direct")),
                # Gap 4: lifecycle fields. Records serialised before the
                # lifecycle was introduced come back as VALIDATED (the
                # default) with no ttl / origin \u2014 matching fresh
                # construction semantics.
                lifecycle=CaseLifecycle(
                    str(record.get("lifecycle", CaseLifecycle.VALIDATED.value))
                ),
                ttl_seconds=(
                    int(record["ttl_seconds"])
                    if record.get("ttl_seconds") is not None
                    else None
                ),
                expires_at_tick=(
                    int(record["expires_at_tick"])
                    if record.get("expires_at_tick") is not None
                    else None
                ),
                provisional_origin=str(record.get("provisional_origin", "")),
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

    def upsert_records(self, records: Iterable[DomainKnowledgeRecord]) -> None:
        for record in records:
            existing = self._records.get(record.record_id)
            if existing is None or record.confidence >= existing.confidence:
                self._records[record.record_id] = record

    def remove_records_by_id_prefix(self, prefix: str) -> int:
        """Packet 6.9: drop records whose ``record_id`` starts with ``prefix``."""

        to_remove = [rid for rid in self._records if rid.startswith(prefix)]
        for rid in to_remove:
            del self._records[rid]
        return len(to_remove)

    def query(
        self,
        *,
        domains: tuple[str, ...],
        query_text: str,
        jurisdiction_required: bool,
        limit: int = 3,
    ) -> tuple[DomainKnowledgeRecord, ...]:
        query_embedding = _semantic_embedding(query_text)
        scored: list[tuple[float, DomainKnowledgeRecord]] = []
        for record in self._records.values():
            if domains and record.domain not in domains:
                continue
            record_embedding = _semantic_embedding(
                " ".join(
                    (
                        record.title,
                        record.summary,
                        record.snippet,
                        " ".join(record.topic_tags),
                        " ".join(record.jurisdiction_tags),
                    )
                )
            )
            semantic_similarity = _clamp((_cosine_similarity(query_embedding, record_embedding) + 1.0) / 2.0)
            jurisdiction_bonus = 0.08 if jurisdiction_required and "local-law-sensitive" in record.jurisdiction_tags else 0.0
            score = record.confidence * 0.52 + semantic_similarity * 0.40 + jurisdiction_bonus
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


@dataclass(frozen=True)
class ProvisionalReconcileThresholds:
    """Threshold pack controlling promote / retire decisions.

    Kept as a frozen dataclass so callers can construct a scenario-
    specific set (e.g. tighter thresholds for coding vertical) without
    touching the store. Defaults match the EmoGPT PRD \u00a75.10 "pattern"
    evidence gate (``min_total_records >= 2``,
    ``min_same_polarity >= 2``, ``min_mean_abs_reward >= 0.15``),
    adapted to the single-record-at-a-time shape of ``CaseMemoryRecord``
    via proxies: relevance_score as evidence strength, repair_observed
    as polarity alignment.
    """

    promote_min_relevance: float = 0.55
    promote_min_confidence: float = 0.50
    promote_requires_repair_observed: bool = False
    retire_max_relevance: float = 0.25


@dataclass(frozen=True)
class ProvisionalReconcileDecision:
    """Single decision emitted by ``reconcile_provisional_cases``.

    Consumers (reflection writeback audit, family report) can iterate
    the decision list to explain why a record moved; nothing else is
    needed for explainability.
    """

    case_id: str
    previous_lifecycle: CaseLifecycle
    new_lifecycle: CaseLifecycle
    reason: str

    @property
    def changed(self) -> bool:
        return self.previous_lifecycle is not self.new_lifecycle


@dataclass(frozen=True)
class ProvisionalReconcileResult:
    promoted: tuple[str, ...]
    retired: tuple[str, ...]
    expired: tuple[str, ...]
    decisions: tuple[ProvisionalReconcileDecision, ...]


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

    def remove_records_by_id_prefix(self, prefix: str) -> int:
        """Packet 6.9: drop records whose ``case_id`` starts with ``prefix``."""

        to_remove = [cid for cid in self._records if cid.startswith(prefix)]
        for cid in to_remove:
            del self._records[cid]
        return len(to_remove)

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
            # Retired records are invisible to retrieval. Candidates /
            # provisionals ARE visible; the store does not second-guess
            # the caller's preference for weak-prior material.
            if not record.is_available_for_retrieval():
                continue
            score = record.relevance_score
            if experience_domains and record.domain in experience_domains:
                score += 0.12
            if regime_id is not None and regime_id in record.regime_tags:
                score += 0.08
            if f"risk-{risk_band}" in record.risk_markers:
                score += 0.06
            # Provisional / candidate records retrieve at a slightly
            # dampened score so validated prior records rank higher
            # when relevance is otherwise tied. No magic ordering \u2014
            # retired records are hidden above and all others are
            # dampened uniformly by lifecycle.
            if record.lifecycle is CaseLifecycle.PROVISIONAL:
                score *= 0.75
            elif record.lifecycle is CaseLifecycle.CANDIDATE:
                score *= 0.50
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].case_id))
        return tuple(record for _, record in scored[:limit])

    def reconcile_provisional_cases(
        self,
        *,
        now_tick: int,
        thresholds: ProvisionalReconcileThresholds | None = None,
    ) -> ProvisionalReconcileResult:
        """Sweep CANDIDATE / PROVISIONAL records by TTL + evidence.

        Pure in the sense that the decision table is deterministic given
        ``(record_state, now_tick, thresholds)`` \u2014 the store is mutated
        in place but the result contains the full decision list so
        downstream audit / family-report consumers can explain every
        promotion or retirement. VALIDATED / RETIRED records are
        untouched.

        Decision order:

        1. **Expire by tick**: any CANDIDATE / PROVISIONAL with
           ``expires_at_tick <= now_tick`` \u2192 RETIRED (reason
           ``"ttl-expired"``).
        2. **Promote**: PROVISIONAL records that meet the thresholds
           \u2192 VALIDATED (reason ``"promoted-by-thresholds"``). TTL and
           origin are cleared on promotion because the record is no
           longer a weak prior.
        3. **Retire by weakness**: CANDIDATE / PROVISIONAL records
           below ``retire_max_relevance`` \u2192 RETIRED (reason
           ``"retired-by-thresholds"``).
        4. **Hold**: everything else stays in its current lifecycle.
        """
        threshold_pack = thresholds or ProvisionalReconcileThresholds()
        decisions: list[ProvisionalReconcileDecision] = []
        promoted: list[str] = []
        retired: list[str] = []
        expired: list[str] = []
        for case_id, record in list(self._records.items()):
            if record.lifecycle in {CaseLifecycle.VALIDATED, CaseLifecycle.RETIRED}:
                continue
            previous = record.lifecycle
            # Expire by tick first.
            if record.expires_at_tick is not None and record.expires_at_tick <= now_tick:
                new_record = self._retire_record(record)
                self._records[case_id] = new_record
                expired.append(case_id)
                decisions.append(
                    ProvisionalReconcileDecision(
                        case_id=case_id,
                        previous_lifecycle=previous,
                        new_lifecycle=CaseLifecycle.RETIRED,
                        reason="ttl-expired",
                    )
                )
                continue
            # Promote (PROVISIONAL only).
            if (
                record.lifecycle is CaseLifecycle.PROVISIONAL
                and record.relevance_score >= threshold_pack.promote_min_relevance
                and record.confidence >= threshold_pack.promote_min_confidence
                and (
                    not threshold_pack.promote_requires_repair_observed
                    or record.repair_observed
                )
            ):
                new_record = self._promote_record(record)
                self._records[case_id] = new_record
                promoted.append(case_id)
                decisions.append(
                    ProvisionalReconcileDecision(
                        case_id=case_id,
                        previous_lifecycle=previous,
                        new_lifecycle=CaseLifecycle.VALIDATED,
                        reason="promoted-by-thresholds",
                    )
                )
                continue
            # Retire by weakness.
            if record.relevance_score <= threshold_pack.retire_max_relevance:
                new_record = self._retire_record(record)
                self._records[case_id] = new_record
                retired.append(case_id)
                decisions.append(
                    ProvisionalReconcileDecision(
                        case_id=case_id,
                        previous_lifecycle=previous,
                        new_lifecycle=CaseLifecycle.RETIRED,
                        reason="retired-by-thresholds",
                    )
                )
                continue
        return ProvisionalReconcileResult(
            promoted=tuple(promoted),
            retired=tuple(retired),
            expired=tuple(expired),
            decisions=tuple(decisions),
        )

    @staticmethod
    def _promote_record(record: CaseMemoryRecord) -> CaseMemoryRecord:
        # Clearing ttl / expires_at_tick signals that lifecycle is no
        # longer a timer \u2014 the record is fully validated now.
        # ``provisional_origin`` is kept as audit trail.
        return replace(
            record,
            lifecycle=CaseLifecycle.VALIDATED,
            ttl_seconds=None,
            expires_at_tick=None,
        )

    @staticmethod
    def _retire_record(record: CaseMemoryRecord) -> CaseMemoryRecord:
        # Retired records keep ttl / origin for audit trail; retrieval
        # hides them via ``is_available_for_retrieval``.
        return replace(record, lifecycle=CaseLifecycle.RETIRED)

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
