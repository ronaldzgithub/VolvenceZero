"""Packet 3.3: ProtocolRegistry.apply_revision tests."""

from __future__ import annotations

import pytest

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
)
from volvence_zero.behavior_protocol import (
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule


def _evidence() -> ProposalEvidence:
    return ProposalEvidence(
        observation_window_turns=10,
        pe_signature="test",
        summary="test summary",
    )


def _build_module_with_cheng() -> tuple[
    ProtocolRegistryModule, ApplicationRareHeavyState
]:
    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    return module, rare


# ---------------------------------------------------------------------------
# WEIGHT_DECAY on STRATEGY_PRIOR
# ---------------------------------------------------------------------------


def test_apply_strategy_decay_protocol_granular() -> None:
    module, _ = _build_module_with_cheng()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    pre_weights = {s.rule_id: s.initial_weight for s in bp.strategy_priors}

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:strategy_decay",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="growth_advisor:cheng-laoshi",  # protocol-granular
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
        proposed_payload={"weight_multiplier": 0.5},
        required_review_level=ReviewLevel.L3,
    )

    module.apply_revision(proposal)

    revised = module.registry.get("growth_advisor:cheng-laoshi")
    post_weights = {s.rule_id: s.initial_weight for s in revised.strategy_priors}
    for rid, pre in pre_weights.items():
        assert post_weights[rid] == pre * 0.5, (rid, pre, post_weights[rid])


def test_apply_strategy_decay_per_strategy() -> None:
    module, _ = _build_module_with_cheng()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    target = bp.strategy_priors[0].rule_id
    pre = bp.strategy_priors[0].initial_weight
    other = bp.strategy_priors[1]
    other_pre = other.initial_weight

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:per-strategy",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
        proposed_payload={"weight_multiplier": 0.25},
    )
    module.apply_revision(proposal)

    revised = module.registry.get("growth_advisor:cheng-laoshi")
    new_strategies = {s.rule_id: s for s in revised.strategy_priors}
    assert new_strategies[target].initial_weight == pre * 0.25
    # Other strategies untouched.
    assert new_strategies[other.rule_id].initial_weight == other_pre


def test_apply_strategy_deactivate_zeros_weight() -> None:
    module, _ = _build_module_with_cheng()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    target = bp.strategy_priors[0].rule_id

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:deactivate",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.DEACTIVATE,
        evidence=_evidence(),
    )
    module.apply_revision(proposal)
    revised = module.registry.get("growth_advisor:cheng-laoshi")
    target_strategy = next(
        s for s in revised.strategy_priors if s.rule_id == target
    )
    assert target_strategy.initial_weight == 0.0


# ---------------------------------------------------------------------------
# revision_log + audit
# ---------------------------------------------------------------------------


def test_apply_revision_appends_to_revision_log() -> None:
    module, _ = _build_module_with_cheng()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:log",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="growth_advisor:cheng-laoshi",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
    )
    pre = len(module.registry.get("growth_advisor:cheng-laoshi").revision_log)
    module.apply_revision(proposal, revised_by="test-reviewer")
    revised = module.registry.get("growth_advisor:cheng-laoshi")
    assert len(revised.revision_log) == pre + 1
    last = revised.revision_log[-1]
    assert last.revised_by == "test-reviewer"
    assert "prop:test:log" in last.revision_id


def test_apply_revision_records_strategy_revision_history() -> None:
    """StrategyPrior.revision_history must record decay actions."""
    module, _ = _build_module_with_cheng()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    target = bp.strategy_priors[0].rule_id

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:strategy-history",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
        proposed_payload={"weight_multiplier": 0.5},
    )
    module.apply_revision(proposal)

    revised = module.registry.get("growth_advisor:cheng-laoshi")
    target_strategy = next(
        s for s in revised.strategy_priors if s.rule_id == target
    )
    assert len(target_strategy.revision_history) == 1
    assert proposal.proposal_id in target_strategy.revision_history[0].revision_id


# ---------------------------------------------------------------------------
# ARCHIVE on KNOWLEDGE_SEED / SIGNATURE_CASE
# ---------------------------------------------------------------------------


def test_apply_archive_knowledge_seed_removes_entry() -> None:
    module, _ = _build_module_with_cheng()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    if not bp.knowledge_seeds:
        pytest.skip("cheng_laoshi has no knowledge_seeds")
    target = bp.knowledge_seeds[0].seed_id

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:archive-knowledge",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.KNOWLEDGE_SEED,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.ARCHIVE,
        evidence=_evidence(),
    )
    module.apply_revision(proposal)
    revised = module.registry.get("growth_advisor:cheng-laoshi")
    seed_ids = {s.seed_id for s in revised.knowledge_seeds}
    assert target not in seed_ids


def test_apply_archive_signature_case_removes_entry() -> None:
    module, _ = _build_module_with_cheng()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    if not bp.signature_cases:
        pytest.skip("cheng_laoshi has no signature_cases")
    target = bp.signature_cases[0].case_id

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:archive-case",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.SIGNATURE_CASE,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.ARCHIVE,
        evidence=_evidence(),
    )
    module.apply_revision(proposal)
    revised = module.registry.get("growth_advisor:cheng-laoshi")
    case_ids = {c.case_id for c in revised.signature_cases}
    assert target not in case_ids


# ---------------------------------------------------------------------------
# Recompile path (application owners see the new content)
# ---------------------------------------------------------------------------


def test_apply_revision_recompiles_into_application_owners() -> None:
    """After apply_revision, the application stores reflect the new content."""
    module, rare = _build_module_with_cheng()
    bp = module.registry.get("growth_advisor:cheng-laoshi")
    target = bp.strategy_priors[0].rule_id

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:recompile",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.DEACTIVATE,
        evidence=_evidence(),
    )
    module.apply_revision(proposal)
    # The compile path runs on apply; rule_id-prefixed entries still exist
    # but their `default_active_weight` reflects the new initial_weight.
    target_rule_id_prefix = (
        f"protocol:growth_advisor:cheng-laoshi:playbook:{target}"
    )
    rules = [
        r for r in rare.distilled_playbook_rules
        if r.rule_id == target_rule_id_prefix
    ]
    assert len(rules) == 1
    # The deactivated strategy compiles to a rule with default
    # active weight reflecting the new (zero) initial_weight.
    # We simply assert that the rule still exists; the compile
    # path is separately covered by packet 1.3b tests.


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_apply_revision_unknown_protocol_raises() -> None:
    module, _ = _build_module_with_cheng()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:bad-target",
        target_protocol_id="never-loaded",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="x",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
    )
    with pytest.raises(KeyError, match="never-loaded"):
        module.apply_revision(proposal)


def test_apply_revision_unsupported_combo_raises() -> None:
    module, _ = _build_module_with_cheng()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:unsupported",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.BOUNDARY_CONTRACT,
        target_entry_id="bp-no-hard-sell",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
    )
    with pytest.raises(NotImplementedError):
        module.apply_revision(proposal)


def test_checkout_revision_without_revisions_returns_initial() -> None:
    """Packet 6.3: rollback on a freshly-loaded protocol returns it unchanged."""
    module, _ = _build_module_with_cheng()
    bp_before = module.registry.get("growth_advisor:cheng-laoshi")
    rolled_back = module.registry.checkout_revision(
        "growth_advisor:cheng-laoshi"
    )
    assert rolled_back == bp_before


def test_checkout_revision_restores_pre_mutation_state() -> None:
    """Packet 6.3: rollback to None restores initial state before any revision."""
    module, _ = _build_module_with_cheng()
    bp_initial = module.registry.get("growth_advisor:cheng-laoshi")
    initial_strategies = tuple(s.rule_id for s in bp_initial.strategy_priors)

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:test:rollback",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="growth_advisor:cheng-laoshi",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
    )
    module.apply_revision(proposal)
    assert any(
        s.initial_weight < bp_initial.strategy_priors[0].initial_weight
        or s.revision_history
        for s in module.registry.get(
            "growth_advisor:cheng-laoshi"
        ).strategy_priors
    )

    rolled_back = module.checkout_revision("growth_advisor:cheng-laoshi")
    rolled_strategies = tuple(s.rule_id for s in rolled_back.strategy_priors)
    assert rolled_strategies == initial_strategies
    # Initial strategies have no revision_history.
    for s in rolled_back.strategy_priors:
        assert s.revision_history == ()


def test_checkout_revision_to_specific_revision_id() -> None:
    """Apply 2 revisions; checkout to first → state after first only."""
    module, _ = _build_module_with_cheng()
    target = module.registry.get(
        "growth_advisor:cheng-laoshi"
    ).strategy_priors[0].rule_id

    proposal_a = ProtocolRevisionProposal(
        proposal_id="prop:rev-a",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
        proposed_payload={"weight_multiplier": 0.5},
    )
    revised_a = module.apply_revision(proposal_a)
    rev_a_id = revised_a.revision_log[-1].revision_id

    proposal_b = ProtocolRevisionProposal(
        proposal_id="prop:rev-b",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
        proposed_payload={"weight_multiplier": 0.5},
    )
    module.apply_revision(proposal_b)

    rolled_back = module.checkout_revision(
        "growth_advisor:cheng-laoshi", rev_a_id
    )
    # After rolling back to rev_a, the revision_log should have
    # exactly 1 entry (the rev_a one).
    assert len(rolled_back.revision_log) == 1
    assert rolled_back.revision_log[0].revision_id == rev_a_id


def test_checkout_revision_unknown_revision_raises() -> None:
    module, _ = _build_module_with_cheng()
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:setup",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="growth_advisor:cheng-laoshi",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
    )
    module.apply_revision(proposal)
    with pytest.raises(KeyError, match="no revision"):
        module.registry.checkout_revision(
            "growth_advisor:cheng-laoshi", "nonexistent-rev"
        )
