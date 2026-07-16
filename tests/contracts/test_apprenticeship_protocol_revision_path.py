"""A1 (#90 residue) contract tests: protocol-alignment PE overlay + revision path.

Covers the two follow-up paths from
``docs/specs/apprenticeship-alignment-protocol-layer-draft.md``:

* Step 2b resolution — the snapshot publishes a bounded PE-shaped
  overlay readout (``pe_overlay_magnitude`` / ``pe_overlay_source``)
  derived from structural verdicts; application-side consumption only.
* Step 3 — conflicts against protocol-lineage artifacts produce typed
  ``ProtocolRevisionProposal``s (conservative WEIGHT_DECAY / L3 /
  1-turn window), and ``ProtocolRevisionQueueModule`` routes them
  through the R10 gate into the human-review queue (never auto-apply),
  with per-proposal-id dedup across turns.
"""

from __future__ import annotations

import asyncio

from volvence_zero.apprenticeship import (
    ApprenticeshipAlignmentSnapshot,
    IntentConstraint,
)
from volvence_zero.application.modules.apprenticeship_protocol_alignment import (
    ApprenticeshipProtocolAlignmentModule,
)
from volvence_zero.application.types import (
    ApprenticeshipProtocolAlignmentSnapshot,
    PlaybookRule,
    StrategyPlaybookSnapshot,
)
from volvence_zero.behavior_protocol import (
    ProtocolRevisionChangeKind,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.protocol_runtime import ProtocolRevisionQueueModule
from volvence_zero.runtime import Snapshot


def _alignment_snapshot(
    constraints: tuple[IntentConstraint, ...],
) -> ApprenticeshipAlignmentSnapshot:
    return ApprenticeshipAlignmentSnapshot(
        version_space_status="consistent",
        consistency_margin=1.0,
        reliability="reliable",
        in_agreement_region=True,
        guidance_surprise=0.0,
        active_constraint_count=len(constraints),
        mismatch_refs=(),
        contradiction_findings=(),
        revision_proposal_refs=(),
        description="test alignment snapshot",
        guidance_constraints=constraints,
    )


def _constraint(
    *, constraint_id: str = "c-1", target_key: str, polarity: int = 1
) -> IntentConstraint:
    return IntentConstraint(
        constraint_id=constraint_id,
        statement=target_key,
        level="abstract",
        polarity=polarity,
        target_key=target_key,
        confidence=0.9,
        source_turn=1,
    )


def _rule(*, rule_id: str, avoid: str) -> PlaybookRule:
    return PlaybookRule(
        rule_id=rule_id,
        problem_pattern="grief support pacing",
        recommended_regime=None,
        recommended_ordering=("listen", "validate"),
        recommended_pacing="slow",
        avoid_patterns=(avoid,),
        knowledge_weight_hint=0.5,
        experience_weight_hint=0.5,
        applicability_scope=(),
        confidence=0.8,
        description="test rule",
    )


def _playbook(rules: tuple[PlaybookRule, ...]) -> StrategyPlaybookSnapshot:
    return StrategyPlaybookSnapshot(
        matched_problem_patterns=("grief support pacing",),
        matched_rules=rules,
        description="test playbook",
    )


def _run_module(
    alignment: ApprenticeshipAlignmentSnapshot,
    playbook: StrategyPlaybookSnapshot,
) -> ApprenticeshipProtocolAlignmentSnapshot:
    module = ApprenticeshipProtocolAlignmentModule()
    snapshot = asyncio.run(
        module.process_standalone(
            apprenticeship_alignment=alignment,
            strategy_playbook=playbook,
        )
    )
    return snapshot.value


_AVOID_TEXT = "procedure dump too early"
_LINEAGE_RULE_ID = "protocol:cheng-laoshi:playbook:pace-rule"


def test_lineage_conflict_emits_conservative_revision_proposal() -> None:
    value = _run_module(
        _alignment_snapshot((_constraint(target_key=_AVOID_TEXT),)),
        _playbook((_rule(rule_id=_LINEAGE_RULE_ID, avoid=_AVOID_TEXT),)),
    )

    assert value.version_space_status == "inconsistent"
    assert value.contradiction_refs
    assert len(value.revision_proposals) == 1
    proposal = value.revision_proposals[0]
    assert proposal.target_protocol_id == "cheng-laoshi"
    assert proposal.target_entry_id == "pace-rule"
    assert proposal.target_field is ProtocolRevisionTargetField.STRATEGY_PRIOR
    assert proposal.change_kind is ProtocolRevisionChangeKind.WEIGHT_DECAY
    assert proposal.required_review_level is ReviewLevel.L3
    # 1-turn window => the R10 gate can never auto-approve this.
    assert proposal.evidence.observation_window_turns == 1
    assert "protocol_alignment_conflict" in proposal.evidence.pe_signature


def test_non_lineage_conflict_emits_no_proposal_but_keeps_verdict() -> None:
    value = _run_module(
        _alignment_snapshot((_constraint(target_key=_AVOID_TEXT),)),
        _playbook((_rule(rule_id="playbook:case-derived:grief:0", avoid=_AVOID_TEXT),)),
    )

    assert value.contradiction_refs
    assert value.revision_proposals == ()


def test_pe_overlay_is_bounded_and_zero_when_covered() -> None:
    conflict_value = _run_module(
        _alignment_snapshot((_constraint(target_key=_AVOID_TEXT),)),
        _playbook((_rule(rule_id=_LINEAGE_RULE_ID, avoid=_AVOID_TEXT),)),
    )
    assert 0.0 < conflict_value.pe_overlay_magnitude <= 1.0
    assert conflict_value.pe_overlay_source

    covered_value = _run_module(
        _alignment_snapshot(
            (_constraint(target_key="grief support pacing listen validate"),)
        ),
        _playbook((_rule(rule_id=_LINEAGE_RULE_ID, avoid="unrelated topic entirely"),)),
    )
    assert covered_value.in_agreement_region
    assert covered_value.pe_overlay_magnitude == 0.0
    assert covered_value.revision_proposals == ()


def _queue_upstream(
    value: ApprenticeshipProtocolAlignmentSnapshot,
) -> dict[str, Snapshot]:
    return {
        "apprenticeship_protocol_alignment": Snapshot(
            slot_name="apprenticeship_protocol_alignment",
            owner="ApprenticeshipProtocolAlignmentModule",
            version=1,
            timestamp_ms=0,
            value=value,
        )
    }


def test_queue_router_routes_alignment_proposals_to_human_queue() -> None:
    value = _run_module(
        _alignment_snapshot((_constraint(target_key=_AVOID_TEXT),)),
        _playbook((_rule(rule_id=_LINEAGE_RULE_ID, avoid=_AVOID_TEXT),)),
    )
    router = ProtocolRevisionQueueModule()

    routed = asyncio.run(router.process(_queue_upstream(value))).value

    assert routed.auto_applied_count == 0
    assert len(routed.newly_routed) == 1
    entry = routed.newly_routed[0]
    assert entry.outcome == "queued_for_human"
    assert entry.target_protocol_id == "cheng-laoshi"
    assert routed.pending_count == 1
    assert len(router.queue) == 1

    # Same proposal republished next turn is deduped, not re-queued.
    routed_again = asyncio.run(router.process(_queue_upstream(value))).value
    assert routed_again.newly_routed == ()
    assert routed_again.pending_count == 1


def test_queue_router_dependency_declares_alignment_slot() -> None:
    assert (
        "apprenticeship_protocol_alignment"
        in ProtocolRevisionQueueModule.dependencies
    )
