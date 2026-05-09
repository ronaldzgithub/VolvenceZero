"""DomainKnowledgeModule (debt #9 wave 2).

R5 application-tier owner: surfaces ``DomainKnowledgeSnapshot``
with active domains + retrieved hits derived from the retrieval
policy snapshot and the application domain knowledge store.

Wave 2 of debt #9 split: this was lines 2818-2958 of the
original monolithic ``runtime.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.dual_track import DualTrackSnapshot
from volvence_zero.memory import MemoryEntry, MemorySnapshot, Track
from volvence_zero.runtime import RuntimeModule, RuntimePlaceholderValue, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    CommonGroundSnapshot,
    ConversationalRoleSnapshot,
    FeelingAboutOtherSnapshot,
    GroupSnapshot,
    IntentAboutOtherSnapshot,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
)
from volvence_zero.application.retrieval_readout import (
    RetrievalControlReadoutInputs,
    RetrievalControlReadoutParameters,
    RetrievalReadoutCheckpoint,
    RetrievalControlReadoutStrategy,
)

if TYPE_CHECKING:
    from volvence_zero.prediction.error import PredictionErrorSnapshot
    from volvence_zero.regime import RegimeSnapshot
    from volvence_zero.temporal_types import TemporalAbstractionSnapshot


from volvence_zero.application.scoring_helpers import clamp01 as _clamp

from volvence_zero.application.types import *  # noqa: F401,F403 -- typed surface
from volvence_zero.application.runtime_helpers import *  # noqa: F401,F403
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState  # noqa: F401


class DomainKnowledgeModule(RuntimeModule[DomainKnowledgeSnapshot]):
    slot_name = "domain_knowledge"
    owner = "DomainKnowledgeModule"
    value_type = DomainKnowledgeSnapshot
    dependencies = ("retrieval_policy", "memory", "dual_track", "regime")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        rare_heavy_state: ApplicationRareHeavyState | None = None,
        store: ApplicationDomainKnowledgeStore | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._rare_heavy_state = rare_heavy_state
        self._store = store

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[DomainKnowledgeSnapshot]:
        from volvence_zero.regime import RegimeSnapshot

        retrieval_policy = upstream["retrieval_policy"].value
        memory_snapshot = upstream["memory"].value
        dual_track_snapshot = upstream["dual_track"].value
        regime_snapshot = upstream["regime"].value
        if not isinstance(retrieval_policy, RetrievalPolicySnapshot):
            raise TypeError("retrieval_policy must publish RetrievalPolicySnapshot.")
        if not isinstance(dual_track_snapshot, DualTrackSnapshot):
            raise TypeError("dual_track must publish DualTrackSnapshot.")
        if not isinstance(regime_snapshot, RegimeSnapshot):
            raise TypeError("regime must publish RegimeSnapshot.")
        memory_value = memory_snapshot if isinstance(memory_snapshot, MemorySnapshot) else None
        memory_text = _memory_text(memory_value)
        unresolved_conflicts: list[str] = []
        if retrieval_policy.jurisdiction_required and not _has_jurisdiction_context(memory_text):
            unresolved_conflicts.append("jurisdiction-unspecified")
        hits: list[KnowledgeHit] = []
        domain_biases = dict(self._rare_heavy_state.domain_template_biases) if self._rare_heavy_state is not None else {}
        records = (
            self._store.query(
                domains=retrieval_policy.knowledge_domains,
                query_text=f"{memory_text} {retrieval_policy.intent_description}",
                jurisdiction_required=retrieval_policy.jurisdiction_required,
                limit=3,
            )
            if self._store is not None
            else ()
        )
        if records:
            for record in records:
                confidence = _clamp(record.confidence + domain_biases.get(record.domain, 0.0) * 0.08)
                hits.append(
                    KnowledgeHit(
                        hit_id=record.record_id,
                        domain=record.domain,
                        topic_tags=record.topic_tags,
                        jurisdiction_tags=record.jurisdiction_tags,
                        freshness_label=record.freshness_label,
                        confidence=confidence,
                        evidence_strength=EvidenceStrength(record.evidence_strength),
                        summary=record.summary,
                        conflict_markers=record.conflict_markers,
                        citations=(
                            KnowledgeCitation(
                                citation_id=f"{record.record_id}:primary",
                                source_type=KnowledgeSourceType(record.source_type),
                                title=record.title,
                                locator=record.locator,
                                snippet=record.snippet,
                                url=record.url,
                            ),
                        ),
                        description=(
                            f"Knowledge record {record.record_id} aligned to regime={retrieval_policy.regime_id} "
                            f"and world_weight={retrieval_policy.world_weight:.2f}."
                        ),
                    )
                )
        else:
            for index, domain in enumerate(retrieval_policy.knowledge_domains[:3], start=1):
                source_type = _domain_source_type(domain)
                confidence = _clamp(
                    0.48
                    + retrieval_policy.knowledge_weight * 0.35
                    + (0.05 if memory_text else 0.0)
                    + domain_biases.get(domain, 0.0) * 0.08
                )
                hit = KnowledgeHit(
                    hit_id=f"{domain}:{index}",
                    domain=domain,
                    topic_tags=_domain_topic_tags(domain),
                    jurisdiction_tags=_domain_jurisdiction_tags(
                        domain,
                        jurisdiction_required=retrieval_policy.jurisdiction_required,
                    ),
                    freshness_label="surface-fallback-current",
                    confidence=confidence,
                    evidence_strength=(
                        EvidenceStrength.HIGH if confidence >= 0.8 else
                        EvidenceStrength.MEDIUM if confidence >= 0.6 else
                        EvidenceStrength.LOW
                    ),
                    summary=_domain_summary(domain, regime_id=retrieval_policy.regime_id),
                    conflict_markers=("jurisdiction-unspecified",)
                    if "jurisdiction-unspecified" in unresolved_conflicts and domain in {"family_transition", "professional_process"}
                    else (),
                    citations=(
                        KnowledgeCitation(
                            citation_id=f"{domain}:primary",
                            source_type=source_type,
                            title=f"{domain.replace('_', ' ')} guidance",
                            locator="surface-fallback",
                            snippet=_domain_summary(domain, regime_id=retrieval_policy.regime_id),
                            url=None,
                        ),
                    ),
                    description=(
                        f"Fallback knowledge hit for {domain} aligned to regime={retrieval_policy.regime_id} "
                        f"and world_weight={retrieval_policy.world_weight:.2f}; compact evidence only."
                    ),
                )
                hits.append(hit)
        retrieval_policy_id = f"policy:{hash(retrieval_policy.intent_description) & 0xFFFF:04x}"
        return self.publish(
            DomainKnowledgeSnapshot(
                retrieval_policy_id=retrieval_policy_id,
                active_domains=retrieval_policy.knowledge_domains,
                hits=tuple(hits),
                citation_required=retrieval_policy.citation_required,
                jurisdiction_required=retrieval_policy.jurisdiction_required,
                unresolved_conflicts=tuple(unresolved_conflicts),
                description=(
                    f"Domain knowledge produced {len(hits)} compact hits for regime={regime_snapshot.active_regime.regime_id} "
                    f"with citation_required={retrieval_policy.citation_required}."
                ),
            )
        )

    async def process_standalone(self, **kwargs: Any) -> Snapshot[DomainKnowledgeSnapshot]:
        raise NotImplementedError("DomainKnowledgeModule should be driven by RetrievalPolicySnapshot.")

