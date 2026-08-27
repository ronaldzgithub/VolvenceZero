from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lifeform_domain_emogpt.lab.contracts import canonical_json
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    RelationshipProductHorizonPublicView,
    build_relationship_product_horizon_evaluator_bundle,
    build_relationship_product_horizon_public_view,
)
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v5 import (
    RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION,
    build_relationship_product_horizon_source_v5_projections,
    load_relationship_product_horizon_source_v5_protocol,
    relationship_product_horizon_source_v5_protocol_path,
    relationship_product_horizon_source_v5_reader_text_inventory,
)


@pytest.fixture(scope="module")
def source_v5() -> tuple[object, RelationshipProductHorizonPublicView, object]:
    protocol = load_relationship_product_horizon_source_v5_protocol()
    public, evaluator = build_relationship_product_horizon_source_v5_projections(protocol)
    return protocol, public, evaluator


def _reader_texts(public: RelationshipProductHorizonPublicView) -> set[str]:
    return {
        *(
            item.user_utterance
            for root in public.roots
            for item in root.onboarding_sessions
        ),
        *(
            item.current_input
            for root in public.roots
            for item in root.decision_sessions
        ),
    }


def test_source_v5_freezes_new_public_and_sealed_identities(
    source_v5: tuple[object, RelationshipProductHorizonPublicView, object],
) -> None:
    protocol, public, evaluator = source_v5
    path = relationship_product_horizon_source_v5_protocol_path()

    assert len(path.read_bytes()) == 17_908
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "33623a4409d3e5419207340e08bd90462b6b1675afb089433b89fbdb2d859134"
    )
    assert protocol.schema_version == RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION
    assert protocol.protocol_id == "71dc200630bf09ee66ce47b9f45460f30ec14cd3ff4e08366c7946497babad9b"
    assert public.public_plan_sha256 == (
        "bab2ff2291b95d4eef6107a58ebf4575b08490775bee71b2ad99a5b029e09f6c"
    )
    assert evaluator.sealed_bundle_sha256 == (
        "01026dd6ec00c58762e8d62263d3a114a954c04c8e9ff6cbd7e09ed7054ab0ab"
    )
    assert protocol.engine_protocol.protocol_id == protocol.protocol_id
    assert len(public.roots) == 112
    assert len(evaluator.root_manifests) == 112
    assert len(evaluator.onboarding_sessions) == 448
    assert len(evaluator.decision_sessions) == 5_376
    assert min(root.public_source_characters for root in public.roots) >= 12_000
    assert min(root.public_source_utf8_bytes for root in public.roots) >= 30_000
    firewall = json.loads(path.read_bytes())["firewall"]
    assert firewall["public_historical_onboarding_outcome_record_count"] == 448
    assert firewall["sealed_evaluator_onboarding_truth_row_count"] == 448
    assert firewall["sealed_evaluator_decision_row_count"] == 5_376
    assert firewall["spent_v4_sealed_bundle_identity_rebuild_for_exclusion_only"] is True
    assert firewall["spent_v4_credit_record_read_count"] == 0
    assert firewall["source_v4_outcome_or_credit_used_to_generate_or_select_v5"] is False
    assert firewall["decision_action_outcome_settlement_count"] == 0
    assert firewall["prediction_error_count"] == 0
    assert firewall["credit_count"] == 0
    assert firewall["gate_update_count"] == 0
    assert firewall["embedding_inference_count"] == 0
    assert firewall["unseen_evidence_authorized"] is False
    assert firewall["dgp_independence_established"] is False


def test_source_v5_is_exactly_disjoint_from_spent_source_v4(
    source_v5: tuple[object, RelationshipProductHorizonPublicView, object],
) -> None:
    _protocol, public, evaluator = source_v5
    spent_public = build_relationship_product_horizon_public_view()
    spent_evaluator = build_relationship_product_horizon_evaluator_bundle()

    assert not (_reader_texts(public) & _reader_texts(spent_public))
    assert len(_reader_texts(spent_public)) == 1_881
    protocol = load_relationship_product_horizon_source_v5_protocol()
    assert len(
        relationship_product_horizon_source_v5_reader_text_inventory(
            public, protocol=protocol
        )
    ) == 3_946
    assert sum(
        len(root.onboarding_sessions) + len(root.decision_sessions) for root in public.roots
    ) == 5_824
    with pytest.raises(ValueError, match="public lineage drifted"):
        relationship_product_horizon_source_v5_reader_text_inventory(
            spent_public, protocol=protocol
        )

    identity_pairs = (
        ({item.subject_id for item in public.roots}, {item.subject_id for item in spent_public.roots}),
        (
            {item.root_seed for item in evaluator.root_manifests},
            {item.root_seed for item in spent_evaluator.root_manifests},
        ),
        (
            {item.tape_seed for item in evaluator.root_manifests},
            {item.tape_seed for item in spent_evaluator.root_manifests},
        ),
        (
            {item.world_clone_id for item in evaluator.root_manifests},
            {item.world_clone_id for item in spent_evaluator.root_manifests},
        ),
        (
            {item.causal_tape_signature for item in evaluator.root_manifests},
            {item.causal_tape_signature for item in spent_evaluator.root_manifests},
        ),
        (
            {item.environment_seed for item in evaluator.decision_sessions},
            {item.environment_seed for item in spent_evaluator.decision_sessions},
        ),
    )
    assert all(not (left & right) for left, right in identity_pairs)
    assert len({item.public_trajectory_sha256 for item in evaluator.root_manifests}) == 112
    assert len({item.causal_tape_signature for item in evaluator.root_manifests}) == 112
    assert len({item.environment_seed for item in evaluator.decision_sessions}) == 5_376


def test_source_v5_public_and_evaluator_join_without_truth_leakage(
    source_v5: tuple[object, RelationshipProductHorizonPublicView, object],
) -> None:
    _protocol, public, evaluator = source_v5
    assert tuple(root.subject_id for root in public.roots) == tuple(
        item.subject_id for item in evaluator.root_manifests
    )
    assert tuple(root.public_trajectory_sha256 for root in public.roots) == tuple(
        item.public_trajectory_sha256 for item in evaluator.root_manifests
    )
    restored = RelationshipProductHorizonPublicView.from_payload(public.to_sut_payload())
    assert restored == public

    forbidden_keys = {
        "condition_id",
        "environment_seed",
        "policy_id",
        "policy_mode",
        "preferred_action_id",
        "root_seed",
        "scene_id",
        "segment_id",
        "surface_recipe_id",
        "tape_seed",
        "world_clone_id",
    }

    public_strings: set[str] = set()

    def assert_public(value: object) -> None:
        if isinstance(value, dict):
            assert not (set(value) & forbidden_keys)
            for child in value.values():
                assert_public(child)
        elif isinstance(value, list):
            for child in value:
                assert_public(child)
        elif isinstance(value, str):
            public_strings.add(value)

    assert_public(public.to_sut_payload())
    for meta_literal in (
        "模型",
        "评估器",
        "隐藏状态",
        "系统",
        "跨会话",
        "长上下文",
        "评测",
        "测试",
        "提示词",
        "记忆是否",
        "恢复和读取",
        "检验",
        "与关系判断无关",
        "只用于说明注意力背景",
        "隐藏的语义标签",
    ):
        assert not any(meta_literal in value for value in public_strings)


def test_source_v5_build_is_model_free_and_deterministic(
    source_v5: tuple[object, RelationshipProductHorizonPublicView, object],
) -> None:
    protocol, public, evaluator = source_v5
    rebuilt_public, rebuilt_evaluator = build_relationship_product_horizon_source_v5_projections(
        protocol
    )
    assert rebuilt_public == public
    assert rebuilt_evaluator == evaluator


def test_source_v5_loader_fails_loudly_on_lineage_or_catalog_drift(tmp_path: Path) -> None:
    canonical_path = relationship_product_horizon_source_v5_protocol_path()
    raw = canonical_path.read_bytes()

    payload = json.loads(raw)
    payload["firewall"]["formal_evidence_authorized"] = True
    drifted = tmp_path / "source-v5.json"
    drifted.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="firewall drifted"):
        load_relationship_product_horizon_source_v5_protocol(drifted)

    payload = json.loads(raw)
    payload["base_source"]["protocol_id"] = "f" * 64
    drifted.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="base causal engine pin drifted"):
        load_relationship_product_horizon_source_v5_protocol(drifted)

    payload = json.loads(raw)
    payload["base_source"]["spent_public_plan_sha256"] = "e" * 64
    drifted.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="base causal engine pin drifted"):
        load_relationship_product_horizon_source_v5_protocol(drifted)

    payload = json.loads(raw)
    surfaces = payload["rendering"]["catalog"]["condition_surfaces"]
    surfaces["connection_under_exclusion"][1] = surfaces["connection_under_exclusion"][0]
    drifted.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="must contain unique strings"):
        load_relationship_product_horizon_source_v5_protocol(drifted)

    payload = json.loads(raw)
    payload["rendering"]["catalog"]["neutral_contexts"][0] = (
        "这是一句向系统透露测试目标的文本。"
    )
    drifted.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="meta-evaluation literal"):
        load_relationship_product_horizon_source_v5_protocol(drifted)

    drifted.write_bytes(raw.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="LF-only"):
        load_relationship_product_horizon_source_v5_protocol(drifted)

    duplicate = raw.replace(
        b'{\n  "schema_version":',
        b'{\n  "schema_version": "duplicate",\n  "schema_version":',
        1,
    )
    drifted.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_relationship_product_horizon_source_v5_protocol(drifted)
