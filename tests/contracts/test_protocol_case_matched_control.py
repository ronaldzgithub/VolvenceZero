"""Matched-control: protocol-driven vs vertical-driven case records.

Mirror of ``test_protocol_boundary_matched_control.py`` /
``test_protocol_strategy_matched_control.py`` /
``test_protocol_knowledge_matched_control.py`` for the case_memory
compile path. Closes the case half of spec
``docs/specs/protocol-runtime.md`` SHADOW → ACTIVE checklist
condition 7 ("BehaviorProtocol → application owners compile path"),
**fully retiring the condition**.

The vertical compile (existing
``DomainExperiencePackage.case_records`` flow producing
``CaseMemoryRecord`` records via ``compiler._case_record``) and the
protocol compile (packet 1.4b
``compile_protocol_to_application_artifacts`` producing
``CaseMemoryRecord`` records via
``protocol_runtime.compiler._signature_case_to_case_record``) must
produce records that are byte-for-byte identical *modulo* the
lineage prefix on ``case_id``:

* Vertical: ``rid-growth-advisor:case:day1-icebreaker-no-followup``
* Protocol: ``protocol:growth_advisor:cheng-laoshi:case:day1-icebreaker-no-followup``

Once that holds, the downstream ``CaseMemoryModule`` execution is
provably identical because it queries the same
``ApplicationCaseMemoryStore`` with no ``case_id``-shape branching.

If a future packet adds a new field to ``SignatureCase`` /
``CaseMemoryRecord`` and forgets to thread it through both compile
paths, this test catches the drift.
"""

from __future__ import annotations

from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from lifeform_domain_growth_advisor.compiler import build_growth_advisor_package
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    CaseMemoryRecord,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule


_NORMALIZED_CASE_ID = "<lineage-normalized>"


def _populate_via_vertical_path() -> ApplicationCaseMemoryStore:
    """Populate the case-memory store via the existing vertical compile."""

    store = ApplicationCaseMemoryStore()
    package = build_growth_advisor_package(build_cheng_laoshi_profile())
    store.upsert_records(package.case_records)
    return store


def _populate_via_protocol_path() -> ApplicationCaseMemoryStore:
    """Populate the case-memory store via the packet 1.4b protocol compile."""

    store = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(case_memory_store=store)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)
    return store


def _normalize_record(record: CaseMemoryRecord) -> CaseMemoryRecord:
    """Strip the ``case_id`` lineage prefix for cross-path comparison."""

    return _replace(record, case_id=_NORMALIZED_CASE_ID)


def _sort_key(record: CaseMemoryRecord) -> str:
    """Stable sort key independent of case_id (which differs by path).

    Uses ``problem_pattern`` since cheng_laoshi cases have unique
    problem patterns — easy to sort and stable across paths.
    """

    return record.problem_pattern


# ---------------------------------------------------------------------------
# Sanity: both paths produce 12 records (cheng_laoshi has 12 signature_cases)
# ---------------------------------------------------------------------------


def test_both_paths_produce_twelve_records() -> None:
    store_v = _populate_via_vertical_path()
    store_p = _populate_via_protocol_path()
    assert len(store_v.records) == 12
    assert len(store_p.records) == 12


# ---------------------------------------------------------------------------
# Record-set equivalence (modulo case_id lineage prefix)
# ---------------------------------------------------------------------------


def test_protocol_path_produces_same_records_as_vertical_path() -> None:
    """The matched-control gate: every ``CaseMemoryRecord`` field
    *except* ``case_id`` must match across paths.

    If this fires, either (a) ``SignatureCase`` schema has grown a
    field that ``GrowthAdvisorSignatureCase`` doesn't have (or vice
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
# Lineage prefix invariant on case_id (the only path-dependent field)
# ---------------------------------------------------------------------------


def test_vertical_case_ids_use_growth_advisor_prefix() -> None:
    store_v = _populate_via_vertical_path()
    for record in store_v.records:
        assert record.case_id.startswith("rid-growth-advisor:case:"), record.case_id


def test_protocol_case_ids_use_protocol_prefix() -> None:
    store_p = _populate_via_protocol_path()
    for record in store_p.records:
        assert record.case_id.startswith(
            "protocol:growth_advisor:cheng-laoshi:case:"
        ), record.case_id


def test_each_case_id_appears_once_per_path() -> None:
    store_v = _populate_via_vertical_path()
    store_p = _populate_via_protocol_path()

    def case_ids(store: ApplicationCaseMemoryStore) -> set[str]:
        return {r.case_id.rsplit(":", 1)[-1] for r in store.records}

    expected_v = case_ids(store_v)
    expected_p = case_ids(store_p)
    assert expected_v == expected_p
    assert len(expected_v) == 12


# ---------------------------------------------------------------------------
# Per-field passthrough (extra granularity for diagnosability)
# ---------------------------------------------------------------------------


def _records_keyed_by_case_id(
    store: ApplicationCaseMemoryStore,
) -> dict[str, CaseMemoryRecord]:
    return {r.case_id.rsplit(":", 1)[-1]: r for r in store.records}


def test_per_field_equivalence_across_paths() -> None:
    """Diagnostic-friendly per-field check.

    Failure mode: if the aggregate equality test above fires, this
    test pinpoints which field diverged — useful when
    ``SignatureCase`` or ``CaseMemoryRecord`` grows a new field.
    """

    by_id_v = _records_keyed_by_case_id(_populate_via_vertical_path())
    by_id_p = _records_keyed_by_case_id(_populate_via_protocol_path())

    assert set(by_id_v) == set(by_id_p)

    diverged_fields: dict[str, set[str]] = {}
    for case_id in by_id_v:
        r_v = by_id_v[case_id]
        r_p = by_id_p[case_id]
        for field_name in (
            "domain",
            "problem_pattern",
            "user_state_pattern",
            "risk_markers",
            "track_tags",
            "regime_tags",
            "intervention_ordering",
            "outcome_label",
            "delayed_signal_count",
            "escalation_observed",
            "repair_observed",
            "confidence",
            "relevance_score",
            "description",
            "continuum_profile_id",
            "continuum_band_id",
            "continuum_position",
            "continuum_update_frequency",
            "reconstruction_source",
            "lifecycle",
            "ttl_seconds",
            "expires_at_tick",
            "provisional_origin",
        ):
            if getattr(r_v, field_name) != getattr(r_p, field_name):
                diverged_fields.setdefault(case_id, set()).add(field_name)
    assert not diverged_fields, (
        f"Per-field divergence between vertical and protocol paths: "
        f"{diverged_fields}. Either SignatureCase is missing a field "
        f"that GrowthAdvisorSignatureCase has, or one of the compilers "
        f"is dropping a field, or fixture-uptake metadata defaults "
        f"diverged from the vertical hardcodes."
    )


# ---------------------------------------------------------------------------
# Apply order does not affect outcome
# ---------------------------------------------------------------------------


def test_loading_protocol_after_vertical_does_not_break_state() -> None:
    """Defensive: simultaneous use of both paths in production
    (e.g. cheng_laoshi loaded both as DomainExperiencePackage AND
    as BehaviorProtocol) must not corrupt the store.

    The merge key ``case_id`` differs between paths (lineage
    prefix), so both sets of records coexist; the store ends up
    with 24 records (12 vertical + 12 protocol). Lineage divergence
    is the audit signal that two sources are seeding the same
    content. Downstream ``CaseMemoryModule`` queries by domain /
    problem_pattern / regime_tags etc., not by ``case_id`` — so it
    sees both copies and either works.
    """

    store = ApplicationCaseMemoryStore()

    # Apply vertical first
    package = build_growth_advisor_package(build_cheng_laoshi_profile())
    store.upsert_records(package.case_records)
    assert len(store.records) == 12

    # Then apply protocol on top of same store
    module = ProtocolRegistryModule(case_memory_store=store)
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)

    # Lineage prefix differs → 12 + 12 = 24 records (no merge)
    assert len(store.records) == 24
