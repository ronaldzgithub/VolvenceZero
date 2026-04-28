"""Typed helpers for dual knowledge ingestion channels (conversation + external/reviewed).

These are owner-side helpers: they only construct immutable proposals (`ConversationKnowledgeCandidate`,
`DomainKnowledgePriorUpdate`, …) for downstream gated writeback. They must not write stores directly.
"""

from __future__ import annotations

from volvence_zero.application.runtime import (
    ConversationKnowledgeCandidate,
    DomainKnowledgePriorUpdate,
    DomainKnowledgeRecord,
    ExternalKnowledgeCandidate,
    KnowledgeHit,
    KnowledgeReviewDecision,
    KnowledgeReviewStatus,
    KnowledgeSourceKind,
    ReviewedKnowledgeCandidate,
)


def infer_knowledge_hit_is_fallback(hit: KnowledgeHit) -> bool:
    """Heuristic aligned with `DomainKnowledgeModule` store vs surface-fallback hits."""
    if "Fallback knowledge hit" in hit.description:
        return True
    return any(citation.locator == "surface-fallback" for citation in hit.citations)


def build_conversation_knowledge_candidates(
    *,
    knowledge_hits: tuple[KnowledgeHit, ...],
    context_session_id: str,
    source_wave_id: str,
    source_turn_index: int,
    boundary_trigger_reasons: tuple[str, ...],
    max_candidates: int = 3,
) -> tuple[ConversationKnowledgeCandidate, ...]:
    """Build audit-friendly conversation-side candidates from turn-time knowledge hits.

    - Store-backed hits: promotable when citations exist and boundary alignment holds.
    - Surface-fallback / citationless hits: `SHADOW` (must not be persisted as durable facts).
    """
    boundary_aligned = "refer-out-required" not in boundary_trigger_reasons
    built: list[ConversationKnowledgeCandidate] = []
    for hit in knowledge_hits[:max_candidates]:
        is_fallback = infer_knowledge_hit_is_fallback(hit)
        has_citation = bool(hit.citations)
        if is_fallback or not has_citation:
            review_status = KnowledgeReviewStatus.SHADOW
        elif not boundary_aligned:
            review_status = KnowledgeReviewStatus.SHADOW
        else:
            review_status = KnowledgeReviewStatus.APPROVED

        citation_ids = tuple(citation.citation_id for citation in hit.citations)
        candidate_id = f"{context_session_id}:{source_wave_id}:turn-{source_turn_index}:knowledge-candidate:{hit.hit_id}"
        built.append(
            ConversationKnowledgeCandidate(
                candidate_id=candidate_id,
                source_context_session_id=context_session_id,
                source_wave_id=source_wave_id,
                source_turn_index=source_turn_index,
                turn_reference=f"turn-{source_turn_index}",
                domain=hit.domain,
                knowledge_hit_id=hit.hit_id,
                citation_ids=citation_ids,
                summary=hit.summary,
                confidence=hit.confidence,
                boundary_aligned=boundary_aligned,
                review_status=review_status,
                is_fallback_hit=is_fallback,
                description=(
                    f"Conversation knowledge candidate for hit={hit.hit_id} domain={hit.domain} "
                    f"review={review_status.value} fallback={is_fallback}."
                ),
            )
        )
    return tuple(built)


def apply_knowledge_review_decisions(
    *,
    candidates: tuple[ExternalKnowledgeCandidate, ...],
    decisions: tuple[KnowledgeReviewDecision, ...],
) -> tuple[ReviewedKnowledgeCandidate, ...]:
    """Minimal review queue: map external candidates to reviewed records using explicit decisions."""
    decision_by_id = {item.candidate_id: item for item in decisions}
    reviewed: list[ReviewedKnowledgeCandidate] = []
    for candidate in candidates:
        decision = decision_by_id.get(candidate.candidate_id)
        if decision is None:
            continue
        if decision.review_status is not KnowledgeReviewStatus.APPROVED:
            continue
        record_id = f"knowledge:external:{candidate.candidate_id}"
        reviewed.append(
            ReviewedKnowledgeCandidate(
                candidate_id=candidate.candidate_id,
                source_kind=KnowledgeSourceKind.EXTERNAL_IMPORT,
                review_status=KnowledgeReviewStatus.APPROVED,
                record=DomainKnowledgeRecord(
                    record_id=record_id,
                    domain=candidate.domain,
                    topic_tags=candidate.topic_tags,
                    jurisdiction_tags=candidate.jurisdiction_tags,
                    source_type=candidate.source_type,
                    title=candidate.title,
                    locator=candidate.locator,
                    summary=candidate.summary,
                    snippet=candidate.snippet,
                    freshness_label=candidate.freshness_label,
                    confidence=max(candidate.confidence, decision.confidence),
                    evidence_strength=candidate.evidence_strength,
                    conflict_markers=candidate.conflict_markers,
                    url=candidate.url,
                ),
                source_candidate_ids=(candidate.candidate_id,),
                review_note=decision.note,
                confidence=max(candidate.confidence, decision.confidence),
                supersedes_record_id=decision.supersedes_record_id,
            )
        )
    return tuple(reviewed)


def domain_knowledge_prior_updates_from_reviewed(
    *,
    job_id: str,
    reviewed: tuple[ReviewedKnowledgeCandidate, ...],
) -> tuple[DomainKnowledgePriorUpdate, ...]:
    """Turn reviewed knowledge artifacts into typed owner-side prior updates."""
    updates: list[DomainKnowledgePriorUpdate] = []
    for index, item in enumerate(reviewed, start=1):
        if item.review_status is not KnowledgeReviewStatus.APPROVED:
            continue
        stable = item.candidate_id.replace(":", "-")
        updates.append(
            DomainKnowledgePriorUpdate(
                update_id=f"{job_id}:knowledge-import:{stable}:{index}",
                target=f"application.domain_knowledge.records.{item.record.domain}.{stable}",
                record=item.record,
                confidence=item.confidence,
                description=item.review_note or f"Import reviewed knowledge candidate={item.candidate_id}.",
                source_kind=item.source_kind,
                source_candidate_ids=item.source_candidate_ids,
                review_status=item.review_status,
                citation_ids=(f"{item.record.record_id}:primary",),
                supersedes_record_id=item.supersedes_record_id,
            )
        )
    return tuple(updates)
