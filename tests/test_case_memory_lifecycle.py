"""Unit tests for Gap 4's case-memory lifecycle and reconcile pass.

Covers the Gap-4 slice-1 foundation: ``CaseLifecycle`` enum, lifecycle
fields on ``CaseMemoryRecord``, and
``ApplicationCaseMemoryStore.reconcile_provisional_cases``. No
scheduler, no workers \u2014 those are slice 2.

Invariants validated:

* **Backwards compatibility**: records constructed without lifecycle
  fields default to ``VALIDATED`` with no TTL; retrieval is unchanged
  for legacy records.
* **Construction invariants**: non-VALIDATED lifecycle without a
  ttl/origin fails loudly; negative ttl / expires_at_tick fails
  loudly.
* **Retrieval filtering**: RETIRED records are invisible to
  ``query()``; PROVISIONAL / CANDIDATE records retrieve at a dampened
  score.
* **Reconcile decision table**: expire-by-tick wins over promote;
  retire-by-weakness is a separate path; VALIDATED / RETIRED are
  untouched.
* **Checkpoint round-trip**: new lifecycle fields survive
  serialization / restore.
"""

from __future__ import annotations

import tempfile

import pytest

from volvence_zero.application import (
    ApplicationCaseMemoryStore,
    CaseLifecycle,
    CaseMemoryRecord,
    ProvisionalReconcileDecision,
    ProvisionalReconcileResult,
    ProvisionalReconcileThresholds,
    build_filesystem_persistence_backend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case(
    case_id: str,
    *,
    lifecycle: CaseLifecycle = CaseLifecycle.VALIDATED,
    relevance_score: float = 0.5,
    confidence: float = 0.5,
    repair_observed: bool = False,
    ttl_seconds: int | None = None,
    expires_at_tick: int | None = None,
    provisional_origin: str = "",
) -> CaseMemoryRecord:
    return CaseMemoryRecord(
        case_id=case_id,
        domain="relationship_continuity",
        problem_pattern="overload-then-decision",
        user_state_pattern="emotionally-overloaded",
        risk_markers=("risk-medium",),
        track_tags=("self",),
        regime_tags=("emotional_support",),
        intervention_ordering=("acknowledge", "clarify"),
        outcome_label="stable",
        delayed_signal_count=1,
        escalation_observed=False,
        repair_observed=repair_observed,
        confidence=confidence,
        relevance_score=relevance_score,
        description=f"Synthetic case {case_id} for lifecycle tests.",
        lifecycle=lifecycle,
        ttl_seconds=ttl_seconds,
        expires_at_tick=expires_at_tick,
        provisional_origin=provisional_origin,
    )


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_default_lifecycle_is_validated_and_no_ttl() -> None:
    record = _case("legacy-1")
    assert record.lifecycle is CaseLifecycle.VALIDATED
    assert record.ttl_seconds is None
    assert record.expires_at_tick is None
    assert record.provisional_origin == ""
    assert record.is_available_for_retrieval()


def test_retired_record_is_hidden_from_retrieval() -> None:
    store = ApplicationCaseMemoryStore(
        records=(
            _case("validated-1", relevance_score=0.3),
            _case(
                "retired-1",
                lifecycle=CaseLifecycle.RETIRED,
                relevance_score=0.95,
                ttl_seconds=1800,
                provisional_origin="mid-reflection:abc",
            ),
        )
    )
    hits = store.query(
        experience_domains=("relationship_continuity",),
        regime_id="emotional_support",
        risk_band="medium",
        limit=5,
    )
    hit_ids = {hit.case_id for hit in hits}
    assert "validated-1" in hit_ids
    assert "retired-1" not in hit_ids


def test_provisional_records_retrieve_at_dampened_score() -> None:
    """Same relevance_score: VALIDATED outranks PROVISIONAL outranks CANDIDATE."""
    store = ApplicationCaseMemoryStore(
        records=(
            _case("validated-a", relevance_score=0.5),
            _case(
                "provisional-a",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.5,
                ttl_seconds=1800,
            ),
            _case(
                "candidate-a",
                lifecycle=CaseLifecycle.CANDIDATE,
                relevance_score=0.5,
                provisional_origin="mid-reflection:abc",
            ),
        )
    )
    hits = store.query(
        experience_domains=("relationship_continuity",),
        regime_id="emotional_support",
        risk_band="medium",
        limit=5,
    )
    hit_ids = [hit.case_id for hit in hits]
    # Order is deterministic: validated first, then provisional, then candidate.
    assert hit_ids.index("validated-a") < hit_ids.index("provisional-a")
    assert hit_ids.index("provisional-a") < hit_ids.index("candidate-a")


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------


def test_candidate_without_ttl_or_origin_is_rejected() -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        _case("invalid-candidate", lifecycle=CaseLifecycle.CANDIDATE)


def test_provisional_with_origin_only_is_accepted() -> None:
    record = _case(
        "valid-provisional",
        lifecycle=CaseLifecycle.PROVISIONAL,
        provisional_origin="reflection:abc",
    )
    assert record.lifecycle is CaseLifecycle.PROVISIONAL


def test_negative_ttl_is_rejected() -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        _case(
            "invalid-ttl",
            lifecycle=CaseLifecycle.PROVISIONAL,
            ttl_seconds=-1,
            provisional_origin="x",
        )


def test_negative_expires_at_tick_is_rejected() -> None:
    with pytest.raises(ValueError, match="expires_at_tick"):
        _case(
            "invalid-expires",
            lifecycle=CaseLifecycle.PROVISIONAL,
            expires_at_tick=-5,
            provisional_origin="x",
        )


# ---------------------------------------------------------------------------
# Reconcile decision table
# ---------------------------------------------------------------------------


def test_reconcile_expires_by_tick_even_if_otherwise_promotable() -> None:
    """TTL expiry wins over promotion \u2014 if time ran out, retire, don't
    promote.
    """
    store = ApplicationCaseMemoryStore(
        records=(
            _case(
                "p-expired",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.9,  # would otherwise promote
                confidence=0.9,
                expires_at_tick=100,
                ttl_seconds=1800,
            ),
        )
    )
    result = store.reconcile_provisional_cases(now_tick=101)
    assert "p-expired" in result.expired
    assert "p-expired" not in result.promoted
    refreshed = {r.case_id: r for r in store.records}
    assert refreshed["p-expired"].lifecycle is CaseLifecycle.RETIRED


def test_reconcile_promotes_provisional_when_thresholds_pass() -> None:
    store = ApplicationCaseMemoryStore(
        records=(
            _case(
                "p-promote",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.80,
                confidence=0.70,
                ttl_seconds=1800,
                provisional_origin="reflection:promote",
            ),
        )
    )
    result = store.reconcile_provisional_cases(now_tick=10)
    assert "p-promote" in result.promoted
    refreshed = {r.case_id: r for r in store.records}
    assert refreshed["p-promote"].lifecycle is CaseLifecycle.VALIDATED
    # Timer fields cleared on promotion; origin retained for audit.
    assert refreshed["p-promote"].ttl_seconds is None
    assert refreshed["p-promote"].expires_at_tick is None
    assert refreshed["p-promote"].provisional_origin == "reflection:promote"


def test_reconcile_retires_weak_records() -> None:
    store = ApplicationCaseMemoryStore(
        records=(
            _case(
                "p-weak",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.15,
                confidence=0.80,
                ttl_seconds=1800,
            ),
        )
    )
    result = store.reconcile_provisional_cases(now_tick=10)
    assert "p-weak" in result.retired
    refreshed = {r.case_id: r for r in store.records}
    assert refreshed["p-weak"].lifecycle is CaseLifecycle.RETIRED


def test_reconcile_leaves_validated_and_retired_untouched() -> None:
    store = ApplicationCaseMemoryStore(
        records=(
            _case("v-steady", lifecycle=CaseLifecycle.VALIDATED, relevance_score=0.9),
            _case(
                "r-done",
                lifecycle=CaseLifecycle.RETIRED,
                relevance_score=0.05,
                ttl_seconds=1800,
                provisional_origin="old-reflection",
            ),
        )
    )
    result = store.reconcile_provisional_cases(now_tick=9999)
    assert result.promoted == ()
    assert result.retired == ()
    assert result.expired == ()
    refreshed = {r.case_id: r for r in store.records}
    assert refreshed["v-steady"].lifecycle is CaseLifecycle.VALIDATED
    assert refreshed["r-done"].lifecycle is CaseLifecycle.RETIRED


def test_reconcile_emits_decision_records_for_audit() -> None:
    store = ApplicationCaseMemoryStore(
        records=(
            _case(
                "p-promote",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.8,
                confidence=0.7,
                ttl_seconds=1800,
            ),
            _case(
                "p-expired",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.9,
                confidence=0.9,
                expires_at_tick=50,
                ttl_seconds=1800,
            ),
            _case(
                "p-weak",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.1,
                confidence=0.9,
                ttl_seconds=1800,
            ),
        )
    )
    result = store.reconcile_provisional_cases(now_tick=100)
    decisions_by_case = {d.case_id: d for d in result.decisions}
    assert decisions_by_case["p-promote"].reason == "promoted-by-thresholds"
    assert decisions_by_case["p-expired"].reason == "ttl-expired"
    assert decisions_by_case["p-weak"].reason == "retired-by-thresholds"
    for decision in result.decisions:
        assert isinstance(decision, ProvisionalReconcileDecision)
        assert decision.changed is True


def test_reconcile_respects_custom_thresholds() -> None:
    """A stricter threshold pack keeps the record provisional."""
    store = ApplicationCaseMemoryStore(
        records=(
            _case(
                "p-borderline",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.56,
                confidence=0.50,
                ttl_seconds=1800,
            ),
        )
    )
    # Strict thresholds: promote needs relevance >= 0.80.
    strict = ProvisionalReconcileThresholds(
        promote_min_relevance=0.80,
        promote_min_confidence=0.50,
    )
    result = store.reconcile_provisional_cases(now_tick=10, thresholds=strict)
    assert result.promoted == ()
    # Still not weak enough to retire (default retire_max_relevance=0.25):
    assert result.retired == ()
    refreshed = {r.case_id: r for r in store.records}
    assert refreshed["p-borderline"].lifecycle is CaseLifecycle.PROVISIONAL


def test_reconcile_returns_empty_result_when_nothing_to_do() -> None:
    store = ApplicationCaseMemoryStore(records=())
    result = store.reconcile_provisional_cases(now_tick=10)
    assert isinstance(result, ProvisionalReconcileResult)
    assert result.promoted == ()
    assert result.retired == ()
    assert result.expired == ()
    assert result.decisions == ()


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_preserves_lifecycle_fields() -> None:
    original = ApplicationCaseMemoryStore(
        records=(
            _case("v-basic"),
            _case(
                "p-active",
                lifecycle=CaseLifecycle.PROVISIONAL,
                relevance_score=0.5,
                ttl_seconds=1800,
                expires_at_tick=200,
                provisional_origin="reflection:origin-1",
            ),
            _case(
                "r-archived",
                lifecycle=CaseLifecycle.RETIRED,
                relevance_score=0.05,
                ttl_seconds=1800,
                provisional_origin="reflection:origin-2",
            ),
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = build_filesystem_persistence_backend(base_dir=tmpdir)
        original = ApplicationCaseMemoryStore(
            records=original.records,
            persistence_backend=backend,
        )
        assert original.save_to_backend(key="lifecycle_test") is True
        restored = ApplicationCaseMemoryStore(persistence_backend=backend)
        assert restored.load_from_backend(key="lifecycle_test") is True
        by_id = {r.case_id: r for r in restored.records}
        assert by_id["v-basic"].lifecycle is CaseLifecycle.VALIDATED
        assert by_id["p-active"].lifecycle is CaseLifecycle.PROVISIONAL
        assert by_id["p-active"].ttl_seconds == 1800
        assert by_id["p-active"].expires_at_tick == 200
        assert by_id["p-active"].provisional_origin == "reflection:origin-1"
        assert by_id["r-archived"].lifecycle is CaseLifecycle.RETIRED


# ---------------------------------------------------------------------------
# Sanity: CaseLifecycle enum is exhaustive
# ---------------------------------------------------------------------------


def test_case_lifecycle_values_are_exhaustive() -> None:
    known = {
        CaseLifecycle.CANDIDATE,
        CaseLifecycle.PROVISIONAL,
        CaseLifecycle.VALIDATED,
        CaseLifecycle.RETIRED,
    }
    assert set(CaseLifecycle) == known, (
        "CaseLifecycle set changed; update reconcile_provisional_cases, "
        "is_available_for_retrieval dampening, and this test."
    )
