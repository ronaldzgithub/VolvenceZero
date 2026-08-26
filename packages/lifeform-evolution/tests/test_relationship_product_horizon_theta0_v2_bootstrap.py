from __future__ import annotations

from collections import Counter
import json
import pathlib
from types import SimpleNamespace

import pytest

import lifeform_evolution.relationship_product_horizon_theta0_v2_bootstrap as subject
from lifeform_domain_emogpt.lab.relationship_product_pulse import (
    RelationshipProductForcedActionRole,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RELATIONSHIP_ACTION_GATE_FEATURE_ORDER,
    RelationshipActionGateCheckpoint,
)


def test_protocol_freezes_adaptive_boundary_and_rejects_type_aliases(
    tmp_path: pathlib.Path,
) -> None:
    loaded = (
        subject.load_relationship_product_horizon_theta0_v2_bootstrap_protocol()
    )
    assert loaded.payload["evidence_tier"] == "development"
    assert loaded.payload["inherited_lineage"]["adaptive_double_use_declared"] is True
    assert loaded.payload["inherited_lineage"]["source_v3_unseen_evidence"] is False
    assert loaded.payload["terminal_gates"]["campaign_authorized_on_success"] is False
    assert (
        loaded.payload["terminal_gates"]["terminal_parameter_nonzero_required"]
        is True
    )
    assert loaded.payload["claims"]["learnable_effect"] is False
    assert loaded.payload["claims"]["steerable_effect"] is False

    protocol = json.loads(
        subject.relationship_product_horizon_theta0_v2_bootstrap_protocol_path().read_text(
            encoding="utf-8"
        )
    )
    variants = (
        (("forced_schedule", "entry_count"), True),
        (("forced_schedule", "role_depends_on_outcome"), 0),
        (("bootstrap", "credit_batch_operator"), "different_operator"),
        (("bootstrap", "online_gate_credit_apply"), 0),
        (("terminal_gates", "credit_count"), False),
        (("causal_firewall", "oracle_input_to_learning"), 0),
        (("claims", "campaign_execution_authorized"), 0),
        (("claim_boundary",), {}),
    )
    for index, (path, replacement) in enumerate(variants):
        mutated = json.loads(json.dumps(protocol))
        cursor = mutated
        for part in path[:-1]:
            cursor = cursor[part]
        cursor[path[-1]] = replacement
        candidate = tmp_path / f"protocol-{index}.json"
        candidate.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(ValueError):
            subject.load_relationship_product_horizon_theta0_v2_bootstrap_protocol(
                candidate
            )

    mutated = json.loads(json.dumps(protocol))
    mutated["bootstrap"]["unregistered_override"] = True
    candidate = tmp_path / "nested-extra-key.json"
    candidate.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="bootstrap fields do not match schema"):
        subject.load_relationship_product_horizon_theta0_v2_bootstrap_protocol(
            candidate
        )

    for token in ("NaN", "Infinity", "-Infinity"):
        candidate = tmp_path / f"nonfinite-{token.replace('-', 'minus')}.json"
        candidate.write_text(f'{{"value":{token}}}', encoding="utf-8")
        with pytest.raises(ValueError, match="non-finite JSON constant"):
            subject.load_relationship_product_horizon_theta0_v2_bootstrap_protocol(
                candidate
            )


def test_schedule_is_frozen_position_only_and_balanced_by_root_and_column() -> None:
    roles = tuple(
        subject._forced_role(
            root_index=root_index,
            decision_index=decision_index,
        )
        for root_index in range(8)
        for decision_index in range(24)
    )
    counts = Counter(roles)
    assert counts == {
        RelationshipProductForcedActionRole.OWNER_RECOMMENDATION: 96,
        RelationshipProductForcedActionRole.NEUTRAL_NOOP: 96,
    }
    for root_index in range(8):
        root = roles[root_index * 24 : (root_index + 1) * 24]
        assert Counter(root) == {
            RelationshipProductForcedActionRole.OWNER_RECOMMENDATION: 12,
            RelationshipProductForcedActionRole.NEUTRAL_NOOP: 12,
        }
    for decision_index in range(24):
        column = roles[decision_index::24]
        assert Counter(column) == {
            RelationshipProductForcedActionRole.OWNER_RECOMMENDATION: 4,
            RelationshipProductForcedActionRole.NEUTRAL_NOOP: 4,
        }


def test_real_source_schedule_matches_frozen_artifact_id() -> None:
    from lifeform_domain_emogpt.lab.relationship_product_pilot_source_v2 import (
        build_relationship_product_pilot_public_view,
        load_relationship_product_pilot_source_protocol,
    )

    protocol = (
        subject.load_relationship_product_horizon_theta0_v2_bootstrap_protocol()
    )
    public_view = build_relationship_product_pilot_public_view(
        load_relationship_product_pilot_source_protocol()
    )
    dependencies = subject._Dependencies(
        protocol=protocol,
        inherited=SimpleNamespace(public_view=public_view),
        base_theta0=None,
    )
    schedule = subject._build_forced_schedule(dependencies)

    assert schedule.artifact_id == protocol.payload["forced_schedule"][
        "expected_schedule_artifact_id"
    ]
    assert len(schedule.entries) == 192


def test_public_join_index_uses_the_owner_session_id_and_rejects_duplicates() -> None:
    rows = [
        {"session_id": f"session-{index:03d}", "join_row_id": f"row-{index:03d}"}
        for index in range(224)
    ]
    indexed = subject._index_public_join({"rows": rows})
    assert len(indexed) == 224
    assert indexed["session-000"]["join_row_id"] == "row-000"

    duplicate = [*rows[:-1], rows[0]]
    with pytest.raises(ValueError, match="session identity is not unique"):
        subject._index_public_join({"rows": duplicate})


def test_zero_terminal_checkpoint_is_a_sealable_failure_not_an_artifact_error() -> None:
    checkpoint = RelationshipActionGateCheckpoint(
        artifact_id="test-gate",
        artifact_version=1,
        weights=tuple(0.0 for _ in RELATIONSHIP_ACTION_GATE_FEATURE_ORDER),
        bias=0.0,
        update_count=192,
        processed_credit_ids=(),
        pending_decisions=(),
    )
    assert (
        subject._create_candidate_theta0(
            post_checkpoint=checkpoint,
            base_theta0=SimpleNamespace(learning_rate=0.25, max_abs_parameter=4.0),
            batch_id="relationship-action-gate-credit-batch-sha256:" + "0" * 64,
        )
        is None
    )
