from __future__ import annotations

import json
import pathlib

import pytest

import lifeform_evolution.relationship_product_horizon_transductive_public_opportunity as subject


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SOURCE_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_source_v4_admission_20260826_b3988b21"
)
_READER_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_development_reader_20260826_"
    "pa1ea1e30fd7b_ce8272ce7d3da"
)
_THETA_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_theta0_v2_bootstrap_20260826_"
    "pdfefb9faa240_c66c2d83a"
)


def test_protocol_is_transductive_fail_closed_and_never_authorizes_campaign(
    tmp_path: pathlib.Path,
) -> None:
    loaded = (
        subject.load_relationship_product_horizon_transductive_public_opportunity_protocol()
    )
    assert loaded.payload["adaptive_lineage"]["transductive"] is True
    assert loaded.payload["adaptive_lineage"]["unseen"] is False
    assert loaded.payload["source_v4_public"]["sealed_file_read_count"] == 0
    assert (
        loaded.payload["terminal_gates"][
            "collection_prefix_protocol_freeze_authorized_on_success"
        ]
        is True
    )
    assert (
        loaded.payload["terminal_gates"][
            "collection_prefix_execution_authorized_on_success"
        ]
        is False
    )
    assert loaded.payload["terminal_gates"]["campaign_authorized_on_success"] is False
    assert loaded.payload["claims"]["learnable_effect"] is False
    assert loaded.payload["claims"]["steerable_effect"] is False

    protocol = json.loads(
        subject.relationship_product_horizon_transductive_public_opportunity_protocol_path().read_text(
            encoding="utf-8"
        )
    )
    variants = (
        (("adaptive_lineage", "unseen"), True),
        (("source_v4_public", "sealed_file_read_count"), False),
        (("scan", "total_probe_count"), True),
        (("paired_witness", "same_owner_prestate_required"), False),
        (
            (
                "terminal_gates",
                "future_collection_first_preaction_projection_schema_version",
            ),
            "drifted",
        ),
        (("terminal_gates", "campaign_authorized_on_success"), True),
        (("claims", "collection_prefix_protocol_freeze_authorized"), True),
        (("causal_firewall", "environment_settlement_count"), False),
        (("claims", "collection_prefix_execution_authorized"), 0),
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
            subject.load_relationship_product_horizon_transductive_public_opportunity_protocol(
                candidate
            )

    mutated = json.loads(json.dumps(protocol))
    mutated["scan"]["hidden_subset"] = "post_hoc"
    candidate = tmp_path / "extra-key.json"
    candidate.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="scan fields do not match schema"):
        subject.load_relationship_product_horizon_transductive_public_opportunity_protocol(
            candidate
        )


def test_probe_partition_and_public_index_buckets_are_exact() -> None:
    categories = [subject._probe_category(index) for index in range(48)]
    assert categories.count("reachable_first_preaction") == 1
    assert categories.count("collection_stress") == 7
    assert categories.count("evaluation_stress") == 40

    category_counts = {
        category: sum(
            subject._probe_category(decision_index) == category
            for _root_index in range(112)
            for decision_index in range(48)
        )
        for category in set(categories)
    }
    assert category_counts == {
        "reachable_first_preaction": 112,
        "collection_stress": 784,
        "evaluation_stress": 4480,
    }
    bucket_counts = {
        bucket: sum(
            subject._public_index_bucket(decision_index) == bucket
            for _root_index in range(112)
            for decision_index in range(48)
        )
        for bucket in {subject._public_index_bucket(index) for index in range(48)}
    }
    assert bucket_counts == {
        f"public_index_bucket_{index}": 896 for index in range(6)
    }


def test_narrow_dependency_loader_reads_no_sealed_or_training_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[pathlib.Path] = []
    original = pathlib.Path.read_bytes

    def tracked(path: pathlib.Path) -> bytes:
        resolved = path.resolve()
        reads.append(resolved)
        return original(path)

    monkeypatch.setattr(pathlib.Path, "read_bytes", tracked)
    dependencies = subject._load_dependencies(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
        theta0_v2_root=_THETA_ROOT,
    )

    assert dependencies.public_view.public_plan_sha256 == (
        "f46336a95aeac2c7be60616388a31333fb45e46cd18320dae2f9bd25179a86d6"
    )
    assert dependencies.frozen_policy.checkpoint.content_sha256 == (
        "d94aa0312e96b184b944503d272cd0ffda48bee36abacb3674475dccdf64ab07"
    )
    assert dependencies.frozen_policy.policy_id == (
        "relationship-action-gate-frozen-policy-sha256:"
        "06e46a9c281e32d9e4e0a53c536188fa6b839ed5c3843a5ea4c7124962f3aebe"
    )
    assert dependencies.frozen_policy.checkpoint.update_count == 0
    assert not dependencies.frozen_policy.checkpoint.processed_credit_ids
    assert not dependencies.frozen_policy.checkpoint.pending_decisions

    forbidden_parts = {
        "sealed",
        "training_inputs.json",
        "condition_training_labels.json",
        "challenge_labels.json",
        "group_split.json",
        "credit_batch.json",
        "apply_receipt.json",
        "withhold_receipt.json",
        "forced_batch_trace.jsonl",
    }
    assert not any(forbidden_parts & set(path.parts) for path in reads)
    assert {path.name for path in reads} <= {
        "relationship_product_horizon_transductive_public_opportunity_v1.json",
        "manifest.json",
        "source_plan.json",
        "embedding_table.json",
        "reader_artifact.json",
        "theta0_artifact.json",
    }


def test_all_noop_scan_has_an_explicit_terminal_failure() -> None:
    reasons = subject._terminal_failure_reasons(
        nonnoop_counts_by_category={
            "reachable_first_preaction": 0,
            "collection_stress": 0,
            "evaluation_stress": 0,
        },
        nonnoop_root_counts_by_category={
            "reachable_first_preaction": 0,
            "collection_stress": 0,
            "evaluation_stress": 0,
        },
        witness_pass_count=0,
    )
    assert reasons == (
        "reachable_first_temporal_delivered_nonnoop_below_one",
        "evaluation_stress_temporal_delivered_nonnoop_below_one",
        "evaluation_stress_temporal_delivered_nonnoop_root_below_one",
        "paired_witness_pass_count_not_two",
    )
