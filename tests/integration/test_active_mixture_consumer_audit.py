"""Packet 4.1: active_mixture consumer audit.

Asserts that the ``active_mixture`` slot, now ACTIVE-published by
default (packet 4.0), has at least one effective downstream
consumer in the production runtime — i.e. the slot is not just
"published into the void". The audit accepts either of two
evidence channels:

1. A kernel ``RuntimeModule`` whose ``dependencies`` tuple
   includes ``"active_mixture"`` (direct snapshot read).
2. ``protocol:`` lineage-prefixed entries observed in the
   canonical application owners (boundary_policy /
   strategy_playbook / domain_knowledge / case_memory) after
   loading the cheng_laoshi protocol — i.e. the compile path
   is feeding application owners that themselves are read by
   ACTIVE consumers.

Channel 2 is the current production reality (packet 1.2–1.4b
shipped the compile path, no kernel module yet reads
``active_mixture`` directly). Channel 1 is the future-proof
escape hatch: when a metacontroller / response-assembly module
starts reading ``active_mixture`` directly (e.g. packet 1.5c-iv
hypothetical), this test still passes.

Exactly one of the two channels MUST yield evidence; if neither
does, the ACTIVE wiring is dangling and packet 4.0's promotion
was premature.
"""

from __future__ import annotations

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
)
from volvence_zero.integration.final_wiring import (
    FinalRolloutConfig,
    build_final_runtime_modules,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule
from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter


def _direct_active_mixture_readers() -> list[str]:
    """List kernel modules whose ``dependencies`` include ``active_mixture``."""

    config = FinalRolloutConfig()
    adapter = PlaceholderSubstrateAdapter(model_id="active-mixture-audit")
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=adapter,
    )
    readers: list[str] = []
    for module in modules:
        deps = getattr(module, "dependencies", ())
        if "active_mixture" in deps:
            readers.append(module.slot_name)
    return readers


def _protocol_lineage_in_application_owners() -> dict[str, list[str]]:
    """Run the protocol compile path and collect protocol-prefixed IDs.

    Returns a mapping owner-name → list of protocol-prefixed IDs.
    Empty mapping (or all-empty values) means the compile path
    isn't producing artifacts (which would fail the audit).
    """

    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)

    return {
        "boundary_policy": [
            h.hint_id for h in rare.boundary_prior_hints
            if h.hint_id.startswith("protocol:")
        ],
        "strategy_playbook": [
            r.rule_id for r in rare.distilled_playbook_rules
            if r.rule_id.startswith("protocol:")
        ],
        "domain_knowledge": [
            r.record_id for r in knowledge.records
            if r.record_id.startswith("protocol:")
        ],
        "case_memory": [
            r.case_id for r in case_memory.records
            if r.case_id.startswith("protocol:")
        ],
    }


def test_at_least_one_active_consumer_evidence_channel() -> None:
    """Channel 1 OR channel 2 must yield evidence.

    If both are empty, ACTIVE wiring is dangling.
    """

    direct_readers = _direct_active_mixture_readers()
    indirect = _protocol_lineage_in_application_owners()
    indirect_total = sum(len(v) for v in indirect.values())

    assert direct_readers or indirect_total > 0, (
        "ACTIVE active_mixture has no effective consumer. "
        f"direct_readers={direct_readers}, indirect={indirect}. "
        "Either a kernel module must declare 'active_mixture' in "
        "its dependencies, or load_protocol must feed application "
        "owners with protocol-prefixed lineage."
    )


def test_indirect_channel_covers_all_four_artifact_families() -> None:
    """Compile path must feed all four artifact families.

    Currently the compile path is the dominant evidence channel,
    so this test pins that all four artifact targets actually
    receive entries from cheng_laoshi. Regression here would
    indicate the compile path is dropping a family.
    """

    indirect = _protocol_lineage_in_application_owners()
    assert indirect["boundary_policy"], indirect
    assert indirect["strategy_playbook"], indirect
    assert indirect["domain_knowledge"], indirect
    assert indirect["case_memory"], indirect


def test_audit_documents_current_evidence_channel() -> None:
    """Audit log: which channel(s) actually fire today?

    This test always passes; it is here as a documentation
    surface so that anyone reading the test output sees the
    evidence shape on the current production path. When packet
    1.5c-iv (or similar) lights up direct active_mixture readers,
    this test will continue to pass and the print will reflect
    the new shape.
    """

    direct = _direct_active_mixture_readers()
    indirect = _protocol_lineage_in_application_owners()
    indirect_counts = {k: len(v) for k, v in indirect.items()}
    print(
        f"\n[active_mixture consumer audit] direct_readers={direct}, "
        f"indirect_lineage_counts={indirect_counts}"
    )
    # Pure documentation; no assertion.
    assert True
