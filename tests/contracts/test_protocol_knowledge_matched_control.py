"""Matched-control: protocol-driven vs vertical-driven knowledge records.

Mirror of ``test_protocol_boundary_matched_control.py`` and
``test_protocol_strategy_matched_control.py`` for the
knowledge / domain-knowledge compile path. Closes the knowledge
half of spec ``docs/specs/protocol-runtime.md`` SHADOW → ACTIVE
checklist condition 7 ("BehaviorProtocol → application owners
compile path").

The vertical compile (existing
``DomainExperiencePackage.knowledge_records`` flow producing
``DomainKnowledgeRecord`` records via
``compiler._knowledge_record``) and the protocol compile (packet
1.4a ``compile_protocol_to_application_artifacts`` producing
``DomainKnowledgeRecord`` records via
``protocol_runtime.compiler._knowledge_seed_to_domain_record``)
must produce records that are byte-for-byte identical *modulo*
the lineage prefix on ``record_id``:

* Vertical: ``rid-growth-advisor:cheng-laoshi:knowledge:persona-identity-growth-planner``
* Protocol: ``protocol:growth_advisor:cheng-laoshi:knowledge:persona-identity-growth-planner``

Once that holds, the downstream ``DomainKnowledgeModule``
execution is provably identical because it reads the same
``ApplicationDomainKnowledgeStore`` via ``store.query(...)``,
filtering by ``record_id`` only as an opaque key — not by lineage
prefix.

If a future packet adds a new field to ``KnowledgeSeed`` /
``DomainKnowledgeRecord`` and forgets to thread it through both
compile paths, this test catches the drift.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from lifeform_domain_growth_advisor.compiler import build_growth_advisor_package
from volvence_zero.application.storage import (
    ApplicationDomainKnowledgeStore,
    DomainKnowledgeRecord,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule


_NORMALIZED_RECORD_ID = "<lineage-normalized>"


def _populate_via_vertical_path() -> ApplicationDomainKnowledgeStore:
    """Populate the domain-knowledge store via the existing vertical compile."""

    store = ApplicationDomainKnowledgeStore(records=())
    package = build_growth_advisor_package(build_cheng_laoshi_profile())
    store.upsert_records(package.knowledge_records)
    return store


def _populate_via_protocol_path() -> ApplicationDomainKnowledgeStore:
    """Populate the domain-knowledge store via the packet 1.4a protocol compile."""

    store = ApplicationDomainKnowledgeStore(records=())
    module = ProtocolRegistryModule(domain_knowledge_store=store)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)
    return store


def _normalize_record(record: DomainKnowledgeRecord) -> DomainKnowledgeRecord:
    """Strip the ``record_id`` lineage prefix for cross-path comparison."""

    return _replace(record, record_id=_NORMALIZED_RECORD_ID)


def _sort_key(record: DomainKnowledgeRecord) -> str:
    """Stable sort key independent of record_id (which differs by path).

    Uses ``locator`` since cheng_laoshi seeds have unique locators
    (``profile:cheng-laoshi:persona-identity`` etc.) — easy to sort
    by and stable across paths.
    """

    return record.locator


# ---------------------------------------------------------------------------
# Sanity: both paths produce 16 records (cheng_laoshi has 16 knowledge_seeds)
# ---------------------------------------------------------------------------


def test_both_paths_produce_sixteen_records() -> None:
    store_v = _populate_via_vertical_path()
    store_p = _populate_via_protocol_path()
    assert len(store_v.records) == 16
    assert len(store_p.records) == 16


# ---------------------------------------------------------------------------
# Record-set equivalence (modulo record_id lineage prefix)
# ---------------------------------------------------------------------------


def test_protocol_path_produces_same_records_as_vertical_path() -> None:
    """The matched-control gate: every ``DomainKnowledgeRecord``
    field *except* ``record_id`` must match across paths.

    If this fires, either (a) ``KnowledgeSeed`` schema has grown a
    field that ``GrowthAdvisorKnowledgeSeed`` doesn't have (or vice
    versa), or (b) one of the two compile functions is dropping a
    field. Both are contract violations.
    """

    store_v = _populate_via_vertical_path()
    store_p = _populate_via_protocol_path()

    norm_v = sorted(
        (_normalize_record(r) for r in store_v.records),
        key=_sort_key,
    )
    norm_p = sorted(
        (_normalize_record(r) for r in store_p.records),
        key=_sort_key,
    )
    assert norm_v == norm_p


# ---------------------------------------------------------------------------
# Lineage prefix invariant on record_id (the only path-dependent field)
# ---------------------------------------------------------------------------


def test_vertical_record_ids_use_growth_advisor_prefix() -> None:
    store_v = _populate_via_vertical_path()
    for record in store_v.records:
        assert record.record_id.startswith(
            "rid-growth-advisor:cheng-laoshi:knowledge:"
        ), record.record_id


def test_protocol_record_ids_use_protocol_prefix() -> None:
    store_p = _populate_via_protocol_path()
    for record in store_p.records:
        assert record.record_id.startswith(
            "protocol:growth_advisor:cheng-laoshi:knowledge:"
        ), record.record_id


def test_each_seed_id_appears_once_per_path() -> None:
    store_v = _populate_via_vertical_path()
    store_p = _populate_via_protocol_path()

    def seed_ids(store: ApplicationDomainKnowledgeStore) -> set[str]:
        return {r.record_id.rsplit(":", 1)[-1] for r in store.records}

    expected_v = seed_ids(store_v)
    expected_p = seed_ids(store_p)
    assert expected_v == expected_p
    assert len(expected_v) == 16


# ---------------------------------------------------------------------------
# Per-field passthrough (extra granularity for diagnosability)
# ---------------------------------------------------------------------------


def _records_keyed_by_seed_id(
    store: ApplicationDomainKnowledgeStore,
) -> dict[str, DomainKnowledgeRecord]:
    return {r.record_id.rsplit(":", 1)[-1]: r for r in store.records}


def test_per_field_equivalence_across_paths() -> None:
    """Diagnostic-friendly per-field check.

    Failure mode: if the aggregate equality test above fires, this
    test pinpoints which field diverged — useful when
    ``KnowledgeSeed`` or ``DomainKnowledgeRecord`` grows a new
    field.
    """

    by_id_v = _records_keyed_by_seed_id(_populate_via_vertical_path())
    by_id_p = _records_keyed_by_seed_id(_populate_via_protocol_path())

    assert set(by_id_v) == set(by_id_p)

    diverged_fields: dict[str, set[str]] = {}
    for seed_id in by_id_v:
        r_v = by_id_v[seed_id]
        r_p = by_id_p[seed_id]
        for field_name in (
            "domain",
            "topic_tags",
            "jurisdiction_tags",
            "source_type",
            "title",
            "locator",
            "summary",
            "snippet",
            "freshness_label",
            "confidence",
            "evidence_strength",
            "conflict_markers",
            "url",
        ):
            if getattr(r_v, field_name) != getattr(r_p, field_name):
                diverged_fields.setdefault(seed_id, set()).add(field_name)
    assert not diverged_fields, (
        f"Per-field divergence between vertical and protocol paths: "
        f"{diverged_fields}. Either KnowledgeSeed is missing a field "
        f"that GrowthAdvisorKnowledgeSeed has, or one of the compilers "
        f"is dropping a field."
    )


# ---------------------------------------------------------------------------
# Apply order does not affect outcome
# ---------------------------------------------------------------------------


def test_loading_protocol_after_vertical_does_not_break_state() -> None:
    """Defensive: simultaneous use of both paths in production
    (e.g. cheng_laoshi loaded both as DomainExperiencePackage AND
    as BehaviorProtocol) must not corrupt the store.

    The merge key ``record_id`` differs between paths (lineage
    prefix), so both sets of records coexist; the store ends up
    with 32 records (16 vertical + 16 protocol). This is
    intentional: lineage divergence is the audit signal that two
    sources are seeding the same content. Downstream
    ``DomainKnowledgeModule`` queries by domain / topic_tags etc.,
    not by ``record_id`` — so it sees both copies and either
    works.
    """

    store = ApplicationDomainKnowledgeStore(records=())

    # Apply vertical first
    package = build_growth_advisor_package(build_cheng_laoshi_profile())
    store.upsert_records(package.knowledge_records)
    assert len(store.records) == 16

    # Then apply protocol on top of same store
    module = ProtocolRegistryModule(domain_knowledge_store=store)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)

    # Lineage prefix differs → 16 + 16 = 32 records (no merge)
    assert len(store.records) == 32
