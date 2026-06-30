"""ApprenticeshipProtocolAlignmentModule (DRAFT Packet 1, SHADOW).

Protocol-layer instantiation of reliable active apprenticeship learning
(Hanneke, Yang, Wang & Song, ALT 2025). Where the vz-cognition
``apprenticeship_alignment`` owner compares operator guidance against
open-ended cognition text (a heuristic, no finite option set), this
application-tier owner compares the SAME guidance constraints against
the currently-active, compiled protocol artifacts — ``strategy_playbook``
(``PlaybookRule``), ``domain_knowledge`` (``KnowledgeHit``), and
``boundary_policy`` — which ARE a finite structured option set. That is
the layer where the reliability / eluder notions are well-defined
(see ``docs/specs/apprenticeship-alignment-protocol-layer-draft.md``).

Boundary rationale: this owner lives in vz-application because it must
read the compiled application-tier protocol content; vz-cognition
cannot import vz-application by tier order. It consumes the enriched
``ApprenticeshipAlignmentSnapshot.guidance_constraints`` (enriched
publisher, R8) so it does NOT re-extract guidance.

Packet 1 scope: SHADOW-only readout. No PE overlay, no belief / protocol
revision. Comparison uses protocol STRUCTURAL fields for the verdict;
character-bigram token overlap is used only for candidate RECALL.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.apprenticeship import (
    ApprenticeshipAlignmentSnapshot,
    ConstraintLevel,
    IntentConstraint,
)
from volvence_zero.runtime import (
    RuntimeModule,
    RuntimePlaceholderValue,
    Snapshot,
    WiringLevel,
)
from volvence_zero.semantic_embedding import stub_semantic_tokens as _semantic_tokens

from volvence_zero.application.types import (
    ApprenticeshipProtocolAlignmentSnapshot,
    BoundaryPolicySnapshot,
    DomainKnowledgeSnapshot,
    PlaybookRule,
    ProtocolAlignmentRef,
    StrategyPlaybookSnapshot,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_semantic_tokens(text)) if text else frozenset()


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    if inter == 0:
        return 0.0
    return inter / len(left | right)


class _ProtocolThresholds:
    # Token-overlap floors are RECALL floors only (the verdict is then
    # decided on structural fields). Protocol problem-pattern phrasing
    # differs from operator phrasing, so the cover floor is looser than
    # the content-layer owner's.
    cover: float = 0.30
    avoid_conflict: float = 0.30
    novelty_surprise: float = 0.45


_RELATION_COVERED = "covered"
_RELATION_NOVEL = "novel"
_RELATION_CONFLICT = "conflict"

_LAYER_STRATEGY = "strategy"
_LAYER_KNOWLEDGE = "knowledge"


def _idle_snapshot(reason: str) -> ApprenticeshipProtocolAlignmentSnapshot:
    return ApprenticeshipProtocolAlignmentSnapshot(
        version_space_status="idle",
        reliability="idle",
        in_agreement_region=False,
        guidance_surprise=0.0,
        matched_protocol_count=0,
        alignment_refs=(),
        contradiction_refs=(),
        description=f"Apprenticeship protocol alignment idle: {reason}.",
    )


class _StrategyTarget:
    __slots__ = ("rule_id", "pattern_tokens", "avoid_tokens")

    def __init__(self, rule: PlaybookRule) -> None:
        self.rule_id = rule.rule_id
        self.pattern_tokens = _tokens(
            f"{rule.problem_pattern} {' '.join(rule.recommended_ordering)}"
        )
        self.avoid_tokens = _tokens(" ".join(rule.avoid_patterns))


class _KnowledgeTarget:
    __slots__ = ("hit_id", "tokens", "has_conflict")

    def __init__(self, *, hit_id: str, text: str, has_conflict: bool) -> None:
        self.hit_id = hit_id
        self.tokens = _tokens(text)
        self.has_conflict = has_conflict


class ApprenticeshipProtocolAlignmentModule(
    RuntimeModule[ApprenticeshipProtocolAlignmentSnapshot]
):
    slot_name = "apprenticeship_protocol_alignment"
    owner = "ApprenticeshipProtocolAlignmentModule"
    value_type = ApprenticeshipProtocolAlignmentSnapshot
    dependencies = (
        "apprenticeship_alignment",
        "strategy_playbook",
        "domain_knowledge",
        "boundary_policy",
        "regime",
    )
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        thresholds: _ProtocolThresholds | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._thresholds = thresholds or _ProtocolThresholds()

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ApprenticeshipProtocolAlignmentSnapshot]:
        alignment = self._value(upstream.get("apprenticeship_alignment"))
        strategy = self._value(upstream.get("strategy_playbook"))
        knowledge = self._value(upstream.get("domain_knowledge"))
        return self.publish(
            self._run(alignment=alignment, strategy=strategy, knowledge=knowledge)
        )

    async def process_standalone(
        self, **kwargs: Any
    ) -> Snapshot[ApprenticeshipProtocolAlignmentSnapshot]:
        return self.publish(
            self._run(
                alignment=kwargs.get("apprenticeship_alignment"),
                strategy=kwargs.get("strategy_playbook"),
                knowledge=kwargs.get("domain_knowledge"),
            )
        )

    @staticmethod
    def _value(snapshot: Snapshot[Any] | None) -> Any:
        if snapshot is None:
            return None
        value = snapshot.value
        if isinstance(value, RuntimePlaceholderValue):
            return None
        return value

    def _run(
        self,
        *,
        alignment: Any,
        strategy: Any,
        knowledge: Any,
    ) -> ApprenticeshipProtocolAlignmentSnapshot:
        if not isinstance(alignment, ApprenticeshipAlignmentSnapshot):
            return _idle_snapshot("no upstream apprenticeship_alignment snapshot")
        constraints = alignment.guidance_constraints
        if not constraints:
            return _idle_snapshot("no guidance constraints this turn")

        strategy_targets: list[_StrategyTarget] = []
        if isinstance(strategy, StrategyPlaybookSnapshot):
            strategy_targets = [_StrategyTarget(rule) for rule in strategy.matched_rules]
        knowledge_targets: list[_KnowledgeTarget] = []
        if isinstance(knowledge, DomainKnowledgeSnapshot):
            knowledge_targets = [
                _KnowledgeTarget(
                    hit_id=hit.hit_id,
                    text=f"{hit.summary} {' '.join(hit.topic_tags)}",
                    has_conflict=bool(hit.conflict_markers),
                )
                for hit in knowledge.hits
            ]
        matched_protocol_count = len(strategy_targets) + len(knowledge_targets)

        alignment_refs: list[ProtocolAlignmentRef] = []
        contradiction_refs: list[ProtocolAlignmentRef] = []
        novel_count = 0
        covered_count = 0

        for constraint in constraints:
            ref = self._classify_constraint(
                constraint=constraint,
                strategy_targets=strategy_targets,
                knowledge_targets=knowledge_targets,
            )
            alignment_refs.append(ref)
            if ref.relation == _RELATION_CONFLICT:
                contradiction_refs.append(ref)
            elif ref.relation == _RELATION_NOVEL:
                novel_count += 1
            else:
                covered_count += 1

        total = len(constraints)
        guidance_surprise = round(novel_count / total, 4) if total else 0.0
        has_conflict = bool(contradiction_refs)
        in_agreement = (
            total > 0 and covered_count == total and not has_conflict
        )
        if has_conflict:
            status = "inconsistent"
        elif guidance_surprise >= self._thresholds.novelty_surprise:
            status = "shrinking"
        else:
            status = "consistent"
        reliability = "reliable" if in_agreement else "deferring"

        description = (
            f"Apprenticeship protocol alignment: status={status} "
            f"reliability={reliability} surprise={guidance_surprise:.2f} "
            f"agreement={in_agreement} matched_protocols={matched_protocol_count} "
            f"covered={covered_count} novel={novel_count} "
            f"conflicts={len(contradiction_refs)}."
        )
        return ApprenticeshipProtocolAlignmentSnapshot(
            version_space_status=status,
            reliability=reliability,
            in_agreement_region=in_agreement,
            guidance_surprise=guidance_surprise,
            matched_protocol_count=matched_protocol_count,
            alignment_refs=tuple(alignment_refs),
            contradiction_refs=tuple(contradiction_refs),
            description=description,
        )

    def _classify_constraint(
        self,
        *,
        constraint: IntentConstraint,
        strategy_targets: list[_StrategyTarget],
        knowledge_targets: list[_KnowledgeTarget],
    ) -> ProtocolAlignmentRef:
        tokens = _tokens(constraint.target_key or constraint.statement)

        # --- strategy layer (richest structural fields) ----------------
        best_rule_overlap = 0.0
        best_rule_id = ""
        best_avoid_overlap = 0.0
        best_avoid_id = ""
        for target in strategy_targets:
            overlap = _jaccard(tokens, target.pattern_tokens)
            if overlap > best_rule_overlap:
                best_rule_overlap = overlap
                best_rule_id = target.rule_id
            avoid_overlap = _jaccard(tokens, target.avoid_tokens)
            if avoid_overlap > best_avoid_overlap:
                best_avoid_overlap = avoid_overlap
                best_avoid_id = target.rule_id

        # Conflict: operator endorses (polarity >= 0) something an active
        # rule explicitly lists in avoid_patterns — a structural-field
        # contradiction, not a text-similarity guess.
        if best_avoid_overlap >= self._thresholds.avoid_conflict and constraint.polarity >= 0:
            return ProtocolAlignmentRef(
                guidance_constraint_id=constraint.constraint_id,
                layer=_LAYER_STRATEGY,
                relation=_RELATION_CONFLICT,
                target_ref=best_avoid_id,
                severity=round(_clamp(best_avoid_overlap * constraint.confidence), 4),
                description=(
                    f"guidance endorses an avoid_pattern of active rule "
                    f"{best_avoid_id} (overlap={best_avoid_overlap:.2f})"
                ),
            )
        # Conflict: operator negates (polarity < 0) an active rule.
        if constraint.polarity < 0 and best_rule_overlap >= self._thresholds.cover:
            return ProtocolAlignmentRef(
                guidance_constraint_id=constraint.constraint_id,
                layer=_LAYER_STRATEGY,
                relation=_RELATION_CONFLICT,
                target_ref=best_rule_id,
                severity=round(_clamp(best_rule_overlap * constraint.confidence), 4),
                description=(
                    f"guidance negates active rule {best_rule_id} "
                    f"(overlap={best_rule_overlap:.2f})"
                ),
            )
        if best_rule_overlap >= self._thresholds.cover:
            return ProtocolAlignmentRef(
                guidance_constraint_id=constraint.constraint_id,
                layer=_LAYER_STRATEGY,
                relation=_RELATION_COVERED,
                target_ref=best_rule_id,
                severity=round(_clamp(best_rule_overlap), 4),
                description=(
                    f"guidance reinforces active rule {best_rule_id} "
                    f"(overlap={best_rule_overlap:.2f})"
                ),
            )

        # --- knowledge layer -------------------------------------------
        best_hit_overlap = 0.0
        best_hit_id = ""
        best_hit_conflict = False
        for target in knowledge_targets:
            overlap = _jaccard(tokens, target.tokens)
            if overlap > best_hit_overlap:
                best_hit_overlap = overlap
                best_hit_id = target.hit_id
                best_hit_conflict = target.has_conflict
        if best_hit_overlap >= self._thresholds.cover:
            if best_hit_conflict:
                return ProtocolAlignmentRef(
                    guidance_constraint_id=constraint.constraint_id,
                    layer=_LAYER_KNOWLEDGE,
                    relation=_RELATION_CONFLICT,
                    target_ref=best_hit_id,
                    severity=round(_clamp(best_hit_overlap * constraint.confidence), 4),
                    description=(
                        f"guidance touches conflict-marked knowledge "
                        f"{best_hit_id} (overlap={best_hit_overlap:.2f})"
                    ),
                )
            return ProtocolAlignmentRef(
                guidance_constraint_id=constraint.constraint_id,
                layer=_LAYER_KNOWLEDGE,
                relation=_RELATION_COVERED,
                target_ref=best_hit_id,
                severity=round(_clamp(best_hit_overlap), 4),
                description=(
                    f"guidance covered by knowledge {best_hit_id} "
                    f"(overlap={best_hit_overlap:.2f})"
                ),
            )

        # --- novel (disagreement region) ------------------------------
        level = (
            _LAYER_STRATEGY
            if constraint.level == ConstraintLevel.ABSTRACT.value
            else _LAYER_KNOWLEDGE
        )
        return ProtocolAlignmentRef(
            guidance_constraint_id=constraint.constraint_id,
            layer=level,
            relation=_RELATION_NOVEL,
            target_ref="",
            severity=1.0,
            description="guidance not covered by any active protocol element",
        )
