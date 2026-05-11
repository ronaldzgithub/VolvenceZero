"""Application-tier rare-heavy state container (debt #9 wave 2).

Owns the long-lived rare-heavy slow-loop bias state that the
training pipeline accumulates and replays into the application
modules: domain template biases, distilled case clusters,
distilled playbook rules, boundary prior hints, retrieval
control readout checkpoint, and reviewed knowledge candidates.

Wave 2 of debt #9 split: this was lines 2240-2366 of the
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

from volvence_zero.application.types import (
    ApplicationCaseCluster,
    ApplicationRareHeavyCheckpoint,
    BoundaryPriorHint,
    PlaybookRule,
    ReviewedKnowledgeCandidate,
)

class ApplicationRareHeavyState:
    def __init__(self) -> None:
        self._domain_template_biases: dict[str, float] = {}
        self._case_clusters: tuple[ApplicationCaseCluster, ...] = ()
        self._distilled_playbook_rules: tuple[PlaybookRule, ...] = ()
        self._boundary_prior_hints: tuple[BoundaryPriorHint, ...] = ()
        self._retrieval_readout_checkpoint: RetrievalReadoutCheckpoint | None = None
        self._reviewed_knowledge_candidates: tuple[ReviewedKnowledgeCandidate, ...] = ()

    @property
    def domain_template_biases(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self._domain_template_biases.items()))

    @property
    def case_clusters(self) -> tuple[ApplicationCaseCluster, ...]:
        return self._case_clusters

    @property
    def distilled_playbook_rules(self) -> tuple[PlaybookRule, ...]:
        return self._distilled_playbook_rules

    @property
    def boundary_prior_hints(self) -> tuple[BoundaryPriorHint, ...]:
        return self._boundary_prior_hints

    @property
    def retrieval_readout_checkpoint(self) -> RetrievalReadoutCheckpoint | None:
        return self._retrieval_readout_checkpoint

    @property
    def reviewed_knowledge_candidates(self) -> tuple[ReviewedKnowledgeCandidate, ...]:
        return self._reviewed_knowledge_candidates

    def upsert_distilled_playbook_rules(self, rules: tuple[PlaybookRule, ...]) -> tuple[str, ...]:
        by_pattern = {rule.problem_pattern: rule for rule in self._distilled_playbook_rules}
        for rule in rules:
            existing = by_pattern.get(rule.problem_pattern)
            if existing is None or rule.confidence >= existing.confidence:
                by_pattern[rule.problem_pattern] = rule
        self._distilled_playbook_rules = tuple(
            sorted(by_pattern.values(), key=lambda rule: (rule.problem_pattern, rule.rule_id))
        )
        return tuple(f"application-playbook-upsert:{rule.problem_pattern}" for rule in rules)

    def upsert_boundary_prior_hints(self, hints: tuple[BoundaryPriorHint, ...]) -> tuple[str, ...]:
        by_key = {
            (hint.regime_id, hint.trigger_reasons): hint
            for hint in self._boundary_prior_hints
        }
        for hint in hints:
            key = (hint.regime_id, hint.trigger_reasons)
            existing = by_key.get(key)
            if existing is None or hint.confidence >= existing.confidence:
                by_key[key] = hint
        self._boundary_prior_hints = tuple(
            sorted(
                by_key.values(),
                key=lambda hint: (
                    hint.regime_id or "",
                    ",".join(hint.trigger_reasons),
                    hint.hint_id,
                ),
            )
        )
        return tuple(
            f"application-boundary-hint-upsert:{hint.regime_id or 'shared'}:{len(hint.trigger_reasons)}"
            for hint in hints
        )

    def remove_boundary_prior_hints_by_id_prefix(self, prefix: str) -> int:
        """Packet 6.9: drop boundary hints whose ``hint_id`` starts with ``prefix``.

        Returns the number of removed hints. Used by
        ``ProtocolRegistryModule.unload_protocol`` to clean up
        protocol-driven entries on unload / rollback.
        """

        before = len(self._boundary_prior_hints)
        self._boundary_prior_hints = tuple(
            h for h in self._boundary_prior_hints
            if not h.hint_id.startswith(prefix)
        )
        return before - len(self._boundary_prior_hints)

    def remove_distilled_playbook_rules_by_id_prefix(self, prefix: str) -> int:
        """Packet 6.9: drop playbook rules whose ``rule_id`` starts with ``prefix``."""

        before = len(self._distilled_playbook_rules)
        self._distilled_playbook_rules = tuple(
            r for r in self._distilled_playbook_rules
            if not r.rule_id.startswith(prefix)
        )
        return before - len(self._distilled_playbook_rules)

    def apply_retrieval_readout_checkpoint(self, checkpoint: RetrievalReadoutCheckpoint) -> tuple[str, ...]:
        existing = self._retrieval_readout_checkpoint
        if existing is None or checkpoint.confidence >= existing.confidence:
            self._retrieval_readout_checkpoint = checkpoint
            return ("application-retrieval-readout-checkpoint-upsert",)
        return ("application-retrieval-readout-checkpoint-skip-lower-confidence",)

    def export_rare_heavy_state(self, *, checkpoint_id: str) -> ApplicationRareHeavyCheckpoint:
        return ApplicationRareHeavyCheckpoint(
            checkpoint_id=checkpoint_id,
            domain_template_biases=self.domain_template_biases,
            case_clusters=self._case_clusters,
            distilled_playbook_rules=self._distilled_playbook_rules,
            boundary_prior_hints=self._boundary_prior_hints,
            reviewed_knowledge_candidates=self._reviewed_knowledge_candidates,
            continuum_profile_id=None,
            retrieval_readout_checkpoint=self._retrieval_readout_checkpoint,
            description=(
                f"Application rare-heavy checkpoint with {len(self._domain_template_biases)} domain biases, "
                f"{len(self._case_clusters)} case clusters, {len(self._distilled_playbook_rules)} playbook rules, "
                f"{len(self._boundary_prior_hints)} boundary prior hints, "
                f"{len(self._reviewed_knowledge_candidates)} reviewed knowledge candidates, and "
                f"{'a' if self._retrieval_readout_checkpoint is not None else 'no'} retrieval readout checkpoint."
            ),
        )

    def import_rare_heavy_state(self, checkpoint: ApplicationRareHeavyCheckpoint) -> tuple[str, ...]:
        self._domain_template_biases = dict(checkpoint.domain_template_biases)
        self._case_clusters = checkpoint.case_clusters
        self._distilled_playbook_rules = checkpoint.distilled_playbook_rules
        self._boundary_prior_hints = checkpoint.boundary_prior_hints
        self._retrieval_readout_checkpoint = checkpoint.retrieval_readout_checkpoint
        self._reviewed_knowledge_candidates = checkpoint.reviewed_knowledge_candidates
        return (
            "rare-heavy:application-domain-refresh",
            "rare-heavy:application-case-clusters-import",
            "rare-heavy:application-playbook-import",
            "rare-heavy:application-boundary-import",
            "rare-heavy:application-retrieval-readout-import",
            "rare-heavy:application-reviewed-knowledge-import",
        )

    def restore_rare_heavy_state(self, checkpoint: ApplicationRareHeavyCheckpoint) -> tuple[str, ...]:
        self._domain_template_biases = dict(checkpoint.domain_template_biases)
        self._case_clusters = checkpoint.case_clusters
        self._distilled_playbook_rules = checkpoint.distilled_playbook_rules
        self._boundary_prior_hints = checkpoint.boundary_prior_hints
        self._retrieval_readout_checkpoint = checkpoint.retrieval_readout_checkpoint
        self._reviewed_knowledge_candidates = checkpoint.reviewed_knowledge_candidates
        return (
            "rare-heavy:application-domain-rollback",
            "rare-heavy:application-case-clusters-rollback",
            "rare-heavy:application-playbook-rollback",
            "rare-heavy:application-boundary-rollback",
            "rare-heavy:application-retrieval-readout-rollback",
            "rare-heavy:application-reviewed-knowledge-rollback",
        )

