"""PersonalConditioningModule owner unit tests (State KV P0-a).

The module previously had zero direct coverage; these tests pin the
owner contract documented in ``docs/specs/personal-conditioning.md``:
strict cold-start behaviour, the frozen 16-coordinate mapping, source
lineage fingerprinting, and loud failure on upstream type drift.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from companion_standard.semantic_state import SemanticRecord
import pytest

from volvence_zero.personal_conditioning import PersonalConditioningModule
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.runtime import Snapshot
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    GoalValueSnapshot,
    RelationshipStateSnapshot,
    UserModelSnapshot,
)


def _record(record_id: str) -> SemanticRecord:
    return SemanticRecord(
        record_id=record_id,
        summary=f"summary {record_id}",
        detail=f"detail {record_id}",
        confidence=0.9,
        status="active",
        source_turn=1,
        evidence=f"evidence {record_id}",
    )


def _user_model(*, populated: bool) -> UserModelSnapshot:
    return UserModelSnapshot(
        stable_preferences=(_record("pref-1"),) if populated else (),
        working_style_hints=(),
        sensitive_boundaries=(),
        durable_goals=(),
        stability_score=0.7,
        control_signal=0.1,
        description="user model",
        overwhelm_pattern_strength=0.3,
    )


def _relationship(*, populated: bool) -> RelationshipStateSnapshot:
    return RelationshipStateSnapshot(
        trust_level=0.6,
        continuity_level=0.5,
        repair_pressure=0.2,
        rapport_signals=(_record("rapport-1"),) if populated else (),
        relational_tensions=(),
        control_signal=0.1,
        description="relationship state",
        emotional_load=0.4,
        repair_need=0.2,
        attunement_gap=0.1,
    )


def _goal_value(*, populated: bool) -> GoalValueSnapshot:
    return GoalValueSnapshot(
        explicit_goals=(_record("goal-1"),) if populated else (),
        value_priorities=(),
        tradeoff_notes=(),
        active_goal_id="goal-1" if populated else None,
        alignment_score=0.8,
        control_signal=0.1,
        description="goal value",
        value_conflict=0.2,
        decision_readiness=0.6,
        reversibility_need=0.3,
    )


def _boundary(*, populated: bool) -> BoundaryConsentSnapshot:
    return BoundaryConsentSnapshot(
        granted_consents=(_record("consent-1"),) if populated else (),
        missing_consents=(),
        denied_boundaries=(),
        memory_consent="granted",
        external_action_consent="granted",
        compliance_score=0.9,
        control_signal=0.1,
        description="boundary consent",
        autonomy_risk=0.1,
        consent_clarity=0.8,
        overreach_risk=0.05,
    )


def _upstream(
    *,
    populated: bool = True,
    versions: tuple[int, int, int, int] = (1, 1, 1, 1),
    overrides: dict[str, Any] | None = None,
) -> dict[str, Snapshot[Any]]:
    values: dict[str, Any] = {
        "user_model": _user_model(populated=populated),
        "relationship_state": _relationship(populated=populated),
        "goal_value": _goal_value(populated=populated),
        "boundary_consent": _boundary(populated=populated),
    }
    if overrides:
        values.update(overrides)
    return {
        slot: Snapshot(
            slot_name=slot,
            owner=f"{slot}-owner",
            version=version,
            timestamp_ms=0,
            value=values[slot],
        )
        for slot, version in zip(
            ("user_model", "relationship_state", "goal_value", "boundary_consent"),
            versions,
            strict=True,
        )
    }


def _process(upstream: dict[str, Snapshot[Any]]):
    module = PersonalConditioningModule()
    return asyncio.run(module.process(upstream))


def test_cold_start_publishes_all_zero_vector_and_zero_confidence() -> None:
    snapshot = _process(_upstream(populated=False))
    value = snapshot.value

    assert value.is_cold_start is True
    assert value.confidence == 0.0
    assert value.state_vector == tuple(
        0.0 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS
    )


def test_vector_coordinates_map_to_frozen_labels() -> None:
    value = _process(_upstream()).value

    assert value.is_cold_start is False
    assert value.schema_version == PERSONAL_CONDITIONING_SCHEMA_VERSION
    assert value.vector_labels == PERSONAL_CONDITIONING_VECTOR_LABELS
    coordinates = dict(zip(value.vector_labels, value.state_vector, strict=True))
    assert coordinates["user_stability"] == pytest.approx(0.7)
    assert coordinates["user_overwhelm"] == pytest.approx(0.3)
    assert coordinates["relationship_trust"] == pytest.approx(0.6)
    assert coordinates["relationship_continuity"] == pytest.approx(0.5)
    assert coordinates["goal_alignment"] == pytest.approx(0.8)
    assert coordinates["goal_decision_readiness"] == pytest.approx(0.6)
    assert coordinates["boundary_compliance"] == pytest.approx(0.9)
    assert coordinates["boundary_consent_clarity"] == pytest.approx(0.8)
    assert 0.0 < value.confidence <= 1.0


def test_source_versions_and_fingerprint_track_upstream_changes() -> None:
    baseline = _process(_upstream()).value
    assert baseline.source_versions == (
        ("user_model", 1),
        ("relationship_state", 1),
        ("goal_value", 1),
        ("boundary_consent", 1),
    )

    bumped_version = _process(_upstream(versions=(2, 1, 1, 1))).value
    assert bumped_version.source_versions[0] == ("user_model", 2)
    assert bumped_version.source_fingerprint != baseline.source_fingerprint

    changed_value = _process(
        _upstream(
            overrides={
                "user_model": replace(
                    _user_model(populated=True), stability_score=0.2
                )
            }
        )
    ).value
    assert changed_value.source_fingerprint != baseline.source_fingerprint


def test_same_upstream_yields_stable_fingerprint() -> None:
    first = _process(_upstream()).value
    second = _process(_upstream()).value
    assert first.source_fingerprint == second.source_fingerprint


def test_cold_start_has_empty_rendered_statement() -> None:
    value = _process(_upstream(populated=False)).value
    assert value.is_cold_start is True
    assert value.rendered_statement == ""


def test_cold_start_contract_rejects_rendered_statement() -> None:
    with pytest.raises(ValueError, match="must not carry a rendered statement"):
        PersonalConditioningSnapshot(
            schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
            state_vector=tuple(
                0.0 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS
            ),
            vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
            source_versions=(),
            source_fingerprint="cold-start",
            confidence=0.0,
            is_cold_start=True,
            description="cold start",
            rendered_statement="unsupported inferred state",
        )


def test_rendered_statement_is_deterministic_and_matches_readout() -> None:
    first = _process(_upstream()).value
    second = _process(_upstream()).value

    assert first.rendered_statement
    assert first.rendered_statement == second.rendered_statement
    # The statement carries the same information as the vector: every
    # coordinate value appears verbatim.
    for coordinate in first.state_vector:
        assert f"({coordinate:.2f})" in first.rendered_statement
    assert f"confidence {first.confidence:.2f}" in first.rendered_statement


def test_rendered_statement_never_leaks_semantic_record_text() -> None:
    """Privacy posture: rendering derives only from the typed readout.

    None of the upstream SemanticRecord prose (summary / detail /
    evidence / record ids) may appear in the rendered statement.
    """
    value = _process(_upstream()).value

    assert value.rendered_statement
    for fragment in (
        "summary pref-1",
        "detail pref-1",
        "evidence pref-1",
        "pref-1",
        "rapport-1",
        "goal-1",
        "consent-1",
        "user model",
        "relationship state",
        "goal value",
        "boundary consent",
    ):
        assert fragment not in value.rendered_statement


def test_upstream_type_drift_fails_loud() -> None:
    upstream = _upstream()
    upstream["user_model"] = Snapshot(
        slot_name="user_model",
        owner="user_model-owner",
        version=1,
        timestamp_ms=0,
        value="not a user model snapshot",
    )
    module = PersonalConditioningModule()
    with pytest.raises(TypeError, match="UserModelSnapshot"):
        asyncio.run(module.process(upstream))
