"""Load path E2E: ``ProtocolRegistryModule.load_protocol`` →
``ApplicationRareHeavyState.boundary_prior_hints``.

Pins the contract that loading a ``BehaviorProtocol`` into a
``ProtocolRegistryModule`` constructed with an injected
``ApplicationRareHeavyState`` causes the protocol's boundary
contracts to materialise as ``BoundaryPriorHint`` records on the
state.

This is the first end-to-end check that the compile path actually
flows into the application owners. A future ``boundary_policy``
matched-control test (packet 1.2+) reads the same state to verify
behaviour parity; if this load path silently fails, that test
would also silently pass on empty state. So the gate here is
"things actually got pushed".

What is asserted:

* Without a state injected, the registry-only path works as before
  (packet 1.0 behaviour preserved).
* With a state injected, ``cheng_laoshi`` load → 4 hints appear in
  ``state.boundary_prior_hints``.
* Hint count grows correctly when multiple protocols are loaded.
* Apply failure (e.g. application validator rejects a malformed
  hint) rolls the registry back so a half-loaded protocol never
  lingers.
* Unload fails fast for already-applied protocols (packet 1.2
  deferred semantics).
"""

from __future__ import annotations

from dataclasses import replace as _replace

import pytest

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    BoundaryContract,
    BoundarySeverity,
    ReviewLevel,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule


def _cheng_laoshi_bp() -> BehaviorProtocol:
    return growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())


# ---------------------------------------------------------------------------
# Path 1: no state injected — registry-only behaviour preserved
# ---------------------------------------------------------------------------


def test_load_without_state_only_touches_registry() -> None:
    module = ProtocolRegistryModule()
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert len(module.registry) == 1
    assert module.registry.get(bp.protocol_id) is bp


# ---------------------------------------------------------------------------
# Path 2: state injected — boundary hints flow through
# ---------------------------------------------------------------------------


def test_load_with_state_pushes_boundary_hints() -> None:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert len(state.boundary_prior_hints) == len(bp.boundary_contracts)


def test_pushed_hints_have_namespaced_ids() -> None:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    for hint in state.boundary_prior_hints:
        assert hint.hint_id.startswith(f"protocol:{bp.protocol_id}:boundary:")


def test_pushed_hints_match_protocol_boundary_ids() -> None:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    expected_boundary_ids = {c.boundary_id for c in bp.boundary_contracts}
    actual_boundary_ids = {
        h.hint_id.rsplit(":", 1)[-1] for h in state.boundary_prior_hints
    }
    assert actual_boundary_ids == expected_boundary_ids


# ---------------------------------------------------------------------------
# Packet 1.3b: load_protocol also pushes playbook rules
# ---------------------------------------------------------------------------


def test_load_with_state_pushes_playbook_rules() -> None:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert len(state.distilled_playbook_rules) == len(bp.strategy_priors)


def test_pushed_playbook_rules_have_namespaced_ids() -> None:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    for rule in state.distilled_playbook_rules:
        assert rule.rule_id.startswith(f"protocol:{bp.protocol_id}:playbook:")


def test_pushed_playbook_rules_match_protocol_rule_ids() -> None:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    expected_rule_ids = {p.rule_id for p in bp.strategy_priors}
    actual_rule_ids = {
        r.rule_id.rsplit(":", 1)[-1] for r in state.distilled_playbook_rules
    }
    assert actual_rule_ids == expected_rule_ids


def test_load_pushes_both_boundary_and_playbook_in_one_call() -> None:
    """Packet 1.3b: a single ``load_protocol`` call applies both
    boundary AND playbook artifacts in one transactional unit.

    cheng_laoshi has 4 boundaries + 16 strategies → state should
    have both populated after a single load.
    """

    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert len(state.boundary_prior_hints) == 4
    assert len(state.distilled_playbook_rules) == 16


# ---------------------------------------------------------------------------
# Packet 1.4a: load_protocol pushes knowledge records when store injected
# ---------------------------------------------------------------------------


def test_load_with_domain_knowledge_store_pushes_records() -> None:
    from volvence_zero.application.storage import (
        ApplicationDomainKnowledgeStore,
    )

    knowledge_store = ApplicationDomainKnowledgeStore(records=())
    module = ProtocolRegistryModule(domain_knowledge_store=knowledge_store)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert len(knowledge_store.records) == len(bp.knowledge_seeds)


def test_pushed_knowledge_records_have_namespaced_ids() -> None:
    from volvence_zero.application.storage import (
        ApplicationDomainKnowledgeStore,
    )

    knowledge_store = ApplicationDomainKnowledgeStore(records=())
    module = ProtocolRegistryModule(domain_knowledge_store=knowledge_store)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    for record in knowledge_store.records:
        assert record.record_id.startswith(
            f"protocol:{bp.protocol_id}:knowledge:"
        )


def test_pushed_knowledge_records_match_protocol_seed_ids() -> None:
    from volvence_zero.application.storage import (
        ApplicationDomainKnowledgeStore,
    )

    knowledge_store = ApplicationDomainKnowledgeStore(records=())
    module = ProtocolRegistryModule(domain_knowledge_store=knowledge_store)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    expected_seed_ids = {s.seed_id for s in bp.knowledge_seeds}
    actual_seed_ids = {
        r.record_id.rsplit(":", 1)[-1] for r in knowledge_store.records
    }
    assert actual_seed_ids == expected_seed_ids


def test_load_pushes_all_three_artifact_kinds_in_one_call() -> None:
    """Packet 1.4a: a single ``load_protocol`` call applies boundary
    + playbook + domain-knowledge artifacts together.

    cheng_laoshi: 4 boundaries + 16 strategies + 16 knowledge seeds.
    """

    from volvence_zero.application.storage import (
        ApplicationDomainKnowledgeStore,
    )

    state = ApplicationRareHeavyState()
    knowledge_store = ApplicationDomainKnowledgeStore(records=())
    module = ProtocolRegistryModule(
        application_rare_heavy_state=state,
        domain_knowledge_store=knowledge_store,
    )
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert len(state.boundary_prior_hints) == 4
    assert len(state.distilled_playbook_rules) == 16
    assert len(knowledge_store.records) == 16


def test_protocol_with_no_knowledge_seeds_does_not_write_knowledge_store() -> None:
    from volvence_zero.application.storage import (
        ApplicationDomainKnowledgeStore,
    )

    knowledge_store = ApplicationDomainKnowledgeStore(records=())
    module = ProtocolRegistryModule(domain_knowledge_store=knowledge_store)
    bp = _replace(_cheng_laoshi_bp(), knowledge_seeds=())
    module.load_protocol(bp)
    assert knowledge_store.records == ()


def test_unload_after_knowledge_only_apply_raises() -> None:
    """Unload deferred semantics cover knowledge-only protocols.

    Even when a protocol pushes only domain-knowledge records (no
    boundary, no strategy), unload must fail-loud — the store has
    no per-key remove API.
    """

    from volvence_zero.application.storage import (
        ApplicationDomainKnowledgeStore,
    )

    knowledge_store = ApplicationDomainKnowledgeStore(records=())
    module = ProtocolRegistryModule(domain_knowledge_store=knowledge_store)
    bp = _replace(
        _cheng_laoshi_bp(),
        boundary_contracts=(),
        strategy_priors=(),
        signature_cases=(),
    )
    module.load_protocol(bp)
    with pytest.raises(NotImplementedError, match="cannot remove"):
        module.unload_protocol(bp.protocol_id)


# ---------------------------------------------------------------------------
# Packet 1.4b: load_protocol pushes case records when store injected
# ---------------------------------------------------------------------------


def test_load_with_case_memory_store_pushes_records() -> None:
    from volvence_zero.application.storage import (
        ApplicationCaseMemoryStore,
    )

    case_store = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(case_memory_store=case_store)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert len(case_store.records) == len(bp.signature_cases)


def test_pushed_case_records_have_namespaced_ids() -> None:
    from volvence_zero.application.storage import (
        ApplicationCaseMemoryStore,
    )

    case_store = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(case_memory_store=case_store)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    for record in case_store.records:
        assert record.case_id.startswith(
            f"protocol:{bp.protocol_id}:case:"
        )


def test_pushed_case_records_match_protocol_case_ids() -> None:
    from volvence_zero.application.storage import (
        ApplicationCaseMemoryStore,
    )

    case_store = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(case_memory_store=case_store)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    expected_case_ids = {c.case_id for c in bp.signature_cases}
    actual_case_ids = {
        r.case_id.rsplit(":", 1)[-1] for r in case_store.records
    }
    assert actual_case_ids == expected_case_ids


def test_load_pushes_all_four_artifact_kinds_in_one_call() -> None:
    """Packet 1.4b: a single ``load_protocol`` call applies boundary
    + playbook + domain-knowledge + case-memory artifacts together.

    cheng_laoshi: 4 boundaries + 16 strategies + 16 knowledge_seeds
    + 12 signature_cases.
    """

    from volvence_zero.application.storage import (
        ApplicationCaseMemoryStore,
        ApplicationDomainKnowledgeStore,
    )

    state = ApplicationRareHeavyState()
    knowledge_store = ApplicationDomainKnowledgeStore(records=())
    case_store = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=state,
        domain_knowledge_store=knowledge_store,
        case_memory_store=case_store,
    )
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert len(state.boundary_prior_hints) == 4
    assert len(state.distilled_playbook_rules) == 16
    assert len(knowledge_store.records) == 16
    assert len(case_store.records) == 12


def test_protocol_with_no_signature_cases_does_not_write_case_store() -> None:
    from volvence_zero.application.storage import (
        ApplicationCaseMemoryStore,
    )

    case_store = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(case_memory_store=case_store)
    bp = _replace(_cheng_laoshi_bp(), signature_cases=())
    module.load_protocol(bp)
    assert case_store.records == ()


def test_unload_after_case_only_apply_raises() -> None:
    """Unload deferred semantics cover case-only protocols too."""

    from volvence_zero.application.storage import (
        ApplicationCaseMemoryStore,
    )

    case_store = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(case_memory_store=case_store)
    bp = _replace(
        _cheng_laoshi_bp(),
        boundary_contracts=(),
        strategy_priors=(),
        knowledge_seeds=(),
    )
    module.load_protocol(bp)
    with pytest.raises(NotImplementedError, match="cannot remove"):
        module.unload_protocol(bp.protocol_id)


# ---------------------------------------------------------------------------
# Packet 1.3b: empty-strategy short-circuit (mirrors empty-boundary case)
# ---------------------------------------------------------------------------


def test_protocol_with_no_strategy_priors_does_not_write_playbook_state() -> None:
    """A protocol with empty ``strategy_priors`` should leave
    ``distilled_playbook_rules`` untouched. Mirrors the
    empty-boundary case for ``boundary_prior_hints``.
    """

    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _replace(_cheng_laoshi_bp(), strategy_priors=())
    module.load_protocol(bp)
    assert state.distilled_playbook_rules == ()
    # Boundaries are still pushed
    assert len(state.boundary_prior_hints) == 4


def test_unload_after_strategy_only_apply_raises() -> None:
    """Unload deferred semantics cover strategy-only protocols too.

    The deferral keys on "any application artifact applied" — so a
    protocol that only pushed playbook rules also fails fast on
    unload. (Boundaries empty + strategies present is a contrived
    case but the guard must still fire.)
    """

    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _replace(_cheng_laoshi_bp(), boundary_contracts=())
    module.load_protocol(bp)
    with pytest.raises(NotImplementedError, match="cannot remove"):
        module.unload_protocol(bp.protocol_id)


# ---------------------------------------------------------------------------
# Path 3: empty boundary_contracts means no application-state writes
# ---------------------------------------------------------------------------


def test_protocol_with_no_boundary_contracts_does_not_write_state() -> None:
    """A protocol with no boundary contracts should not touch the state.

    This is important for ``unload_protocol`` semantics: if no
    artifacts were applied, unload is allowed (returns to registry-
    only delete). See ``unload`` deferred-NotImplementedError test
    below for the contrasting case.
    """

    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _replace(_cheng_laoshi_bp(), boundary_contracts=())
    module.load_protocol(bp)
    assert state.boundary_prior_hints == ()


# ---------------------------------------------------------------------------
# Path 4: rollback on apply failure (registry stays consistent)
# ---------------------------------------------------------------------------


class _FailingState:
    """Test double whose ``upsert_boundary_prior_hints`` always raises.

    Used to verify the load-time rollback path: if the application
    apply step throws, the registry must NOT keep the half-loaded
    protocol.
    """

    @property
    def boundary_prior_hints(self):
        return ()

    def upsert_boundary_prior_hints(self, hints):  # noqa: D401, ARG002
        raise RuntimeError("simulated apply failure")


def test_apply_failure_rolls_back_registry() -> None:
    state = _FailingState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    with pytest.raises(RuntimeError, match="simulated apply failure"):
        module.load_protocol(bp)
    assert module.registry.get(bp.protocol_id) is None
    assert len(module.registry) == 0


# ---------------------------------------------------------------------------
# Path 5: unload semantics (deferred for applied protocols)
# ---------------------------------------------------------------------------


def test_unload_after_apply_raises_not_implemented() -> None:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    with pytest.raises(NotImplementedError, match="cannot remove"):
        module.unload_protocol(bp.protocol_id)


def test_unload_when_no_state_works_as_registry_delete() -> None:
    """Without a state injected, unload is the trivial registry delete."""

    module = ProtocolRegistryModule()
    bp = _cheng_laoshi_bp()
    module.load_protocol(bp)
    assert module.unload_protocol(bp.protocol_id) is True
    assert module.unload_protocol(bp.protocol_id) is False


def test_unload_when_protocol_had_no_application_artifacts_works() -> None:
    """If a protocol had no application artifacts (no boundaries,
    strategies, knowledge seeds, or signature cases pushed),
    unload is allowed even with stores injected.

    Packet 1.2 only checked boundaries; packet 1.3b broadened to
    cover strategies; packet 1.4a broadened to cover knowledge;
    packet 1.4b broadens to cover cases. The deferred semantics
    only block unload of protocols that actually pushed entries.
    """

    from volvence_zero.application.storage import (
        ApplicationCaseMemoryStore,
        ApplicationDomainKnowledgeStore,
    )

    state = ApplicationRareHeavyState()
    knowledge_store = ApplicationDomainKnowledgeStore(records=())
    case_store = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=state,
        domain_knowledge_store=knowledge_store,
        case_memory_store=case_store,
    )
    bp = _replace(
        _cheng_laoshi_bp(),
        boundary_contracts=(),
        strategy_priors=(),
        knowledge_seeds=(),
        signature_cases=(),
    )
    module.load_protocol(bp)
    assert module.unload_protocol(bp.protocol_id) is True


# ---------------------------------------------------------------------------
# Path 6: multiple protocols stack hints (no collision)
# ---------------------------------------------------------------------------


def _make_unique_test_protocol(suffix: str) -> BehaviorProtocol:
    """Build a synthetic protocol with one unique boundary.

    Used to assert the application state correctly merges hints
    from multiple protocols. We use unique trigger_reasons to
    avoid the (regime_id, trigger_reasons) merge key collision in
    ``upsert_boundary_prior_hints``.
    """

    base = _cheng_laoshi_bp()
    unique_contract = BoundaryContract(
        boundary_id=f"bp-test-{suffix}",
        description=f"test boundary {suffix}",
        trigger_reasons=(f"trigger-{suffix}",),
        severity=BoundarySeverity.HARD_BLOCK,
        review_level=ReviewLevel.L3,
    )
    return _replace(
        base,
        protocol_id=f"test:{suffix}",
        boundary_contracts=(unique_contract,),
    )


def test_loading_two_protocols_stacks_hints_in_state() -> None:
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    a = _make_unique_test_protocol("a")
    b = _make_unique_test_protocol("b")
    module.load_protocol(a)
    module.load_protocol(b)

    hint_ids = {h.hint_id for h in state.boundary_prior_hints}
    assert "protocol:test:a:boundary:bp-test-a" in hint_ids
    assert "protocol:test:b:boundary:bp-test-b" in hint_ids
    assert len(state.boundary_prior_hints) == 2
