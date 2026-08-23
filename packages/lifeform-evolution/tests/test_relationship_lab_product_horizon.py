from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys

import pytest

import lifeform_evolution.relationship_lab_product_horizon as campaign
from lifeform_domain_emogpt.lab import RelationshipAction, sha256_json
from lifeform_domain_emogpt.lab.relationship_product_pilot_source import (
    build_relationship_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol,
)
from lifeform_evolution.relationship_lab_baseline import (
    DEFAULT_STATELESS_MODEL_ID,
    StatelessActionCompletion,
)
from lifeform_evolution.relationship_lab_product_baselines import (
    FrozenProductChatMessage,
    ProductBaselineTokenBudget,
    RelationshipProductBaselineSuite,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    PrecomputedPublicEmbeddingRecord,
    PrecomputedPublicEmbeddingTable,
    write_precomputed_public_embedding_table,
)


class _FakeExactTokenizer:
    tokenizer_id = "fake-test-only/qwen-tokenizer"

    def count_message_tokens(
        self,
        *,
        messages: tuple[FrozenProductChatMessage, ...],
    ) -> int:
        return 3 + sum(2 + len(message.content.split()) for message in messages)


class _FakePolicy:
    model_id = DEFAULT_STATELESS_MODEL_ID
    tokenizer_id = _FakeExactTokenizer.tokenizer_id
    max_new_tokens = 64
    weights_sha256 = sha256_json("fake-product-policy-weights")
    prompt_sha256 = sha256_json("fake-product-policy-prompt")
    generation_config_sha256 = sha256_json("fake-product-policy-generation")

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        raise AssertionError("product baselines must call choose_from_messages")

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        prompt_tokens = 3 + sum(2 + len(message["content"].split()) for message in messages)
        action = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
        return StatelessActionCompletion(
            raw_output=json.dumps(
                {"action_id": action.value},
                separators=(",", ":"),
            ),
            chosen_action_id=action,
            prompt_tokens=prompt_tokens,
            completion_tokens=2,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class _FakeSemanticEmbedder:
    name = "fake-test-only/product-baseline-semantic"

    def embed(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return (
            (digest[0] + 1) / 256.0,
            (digest[1] + 1) / 256.0,
            (digest[2] + 1) / 256.0,
        )


def _fake_table(path: pathlib.Path) -> None:
    source = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(source)
    selection = campaign.RelationshipProductCampaignSelection(
        subject_count=1,
        onboarding_session_count=4,
        decision_session_count=2,
    )
    records = []
    embedder = _FakeSemanticEmbedder()
    for text in campaign.relationship_product_required_semantic_texts(
        public,
        selection=selection,
    ):
        records.append(
            PrecomputedPublicEmbeddingRecord(
                text=text,
                embedding_hex=tuple(value.hex() for value in embedder.embed(text)),
            )
        )
    records.sort(key=lambda item: (item.text_sha256, item.text))
    write_precomputed_public_embedding_table(
        PrecomputedPublicEmbeddingTable(
            source_embedder_name="fake-test-only/product-horizon",
            embedding_width=3,
            records=tuple(records),
        ),
        path=path,
    )


@pytest.fixture(scope="module")
def smoke_campaign(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    root = tmp_path_factory.mktemp("relationship-product-horizon")
    table_path = root / "fake-public-embedding-table.json"
    _fake_table(table_path)
    output = root / "campaign"
    suite = RelationshipProductBaselineSuite(
        policy=_FakePolicy(),
        token_counter=_FakeExactTokenizer(),
        token_budget=ProductBaselineTokenBudget(
            context_window_tokens=32768,
            generation_reserve_tokens=64,
        ),
        semantic_embedder=_FakeSemanticEmbedder(),
    )
    previous = os.environ.get("PYTHONNOUSERSITE")
    os.environ["PYTHONNOUSERSITE"] = "1"
    try:
        campaign.run_relationship_product_horizon_campaign(
            output_dir=output,
            public_embedding_table_path=table_path,
            worker_script=pathlib.Path(__file__).resolve().parents[3]
            / "scripts"
            / "run_relationship_lab_product_horizon.py",
            python_executable=sys.executable,
            baseline_suite=suite,
            selection=campaign.RelationshipProductCampaignSelection(
                subject_count=1,
                onboarding_session_count=4,
                decision_session_count=2,
            ),
            allow_test_semantic_backend=True,
            max_workers=5,
            worker_timeout_seconds=60.0,
        )
    finally:
        if previous is None:
            os.environ.pop("PYTHONNOUSERSITE", None)
        else:
            os.environ["PYTHONNOUSERSITE"] = previous
    return output


def test_protocol_and_truth_firewall_are_strict() -> None:
    protocol = campaign.load_relationship_product_horizon_protocol()
    assert protocol.subject_count == 8
    assert protocol.decision_sessions_per_subject == 24
    assert protocol.context_window_tokens == 32768
    assert protocol.source_protocol_id == ("048b73d4a412b4444fb469be0d9daa6d2a26e9920c743804da8f36dc331691ae")
    with pytest.raises(ValueError, match="leaked sealed evaluator keys"):
        campaign._assert_truth_firewall({"public": {}, "preferred_action_id": "x"})


def test_cross_process_smoke_covers_restart_world_clone_and_state_contrasts(
    smoke_campaign: pathlib.Path,
) -> None:
    report = campaign.validate_relationship_product_horizon_campaign(
        output_dir=smoke_campaign,
    )
    assert report["typed_control_executed"] is True
    assert report["strong_baselines_executed"] is True
    assert report["fresh_process_per_volvence_logical_session"] is True
    assert report["volvence_logical_session_count"] == 30
    assert len(report["worker_request_artifact_ids"]) == 30
    assert len(set(report["worker_request_artifact_ids"])) == 30
    assert report["distinct_child_pid_count"] > 1
    assert report["residual_steerable"] is False
    assert report["four_able_complete"] is False
    assert report["production_active"] is False
    assert report["os_security_boundary"] is False

    chain_files = sorted((smoke_campaign / "chains").glob("*/*/chain.json"))
    chains = [json.loads(path.read_text(encoding="utf-8")) for path in chain_files]
    assert len({chain["world_clone_id"] for chain in chains}) == 1
    frozen = next(chain for chain in chains if chain["arm_id"] == "appendable_frozen_onboarding")
    frozen_pre_hashes = {decision["pre_owner_snapshot_sha256"] for decision in frozen["decisions"]}
    assert len(frozen_pre_hashes) == 1
    assert [decision["gate_update_count_after"] for decision in frozen["decisions"]] == [1, 2]
    full = next(chain for chain in chains if chain["arm_id"] == "volvence_full")
    assert full["decisions"][1]["pre_owner_snapshot_sha256"] == full["decisions"][0]["post_owner_snapshot_sha256"]
    readable = next(chain for chain in chains if chain["arm_id"] == "readable_permuted")
    first_full_recommendation = full["decisions"][0]["recommended_action_id"]
    first_readable_recommendation = readable["decisions"][0]["recommended_action_id"]
    expected_permutation = {
        RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value: (
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value
        ),
        RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value: (
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value
        ),
        RelationshipAction.NEUTRAL_NOOP.value: RelationshipAction.NEUTRAL_NOOP.value,
    }
    assert first_readable_recommendation == expected_permutation[first_full_recommendation]
    withheld = next(chain for chain in chains if chain["arm_id"] == "credit_withheld")
    assert [decision["gate_update_count_after"] for decision in withheld["decisions"]] == [0, 0]
    assert all(decision["credit_applied_to_gate"] is False for decision in withheld["decisions"])
    noop = next(chain for chain in chains if chain["arm_id"] == "strict_noop")
    assert all(
        decision["selected_action_id"] == RelationshipAction.NEUTRAL_NOOP.value for decision in noop["decisions"]
    )


def test_manifest_tamper_is_rejected(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(smoke_campaign, tampered)
    report_path = tampered / "report.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest file tree/hash mismatch"):
        campaign.validate_relationship_product_horizon_campaign(
            output_dir=tampered,
        )


def test_self_consistent_surface_rehash_cannot_forge_recomputed_metrics(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    tampered = tmp_path / "self-consistent-tamper"
    shutil.copytree(smoke_campaign, tampered)
    report_path = tampered / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_core = {key: value for key, value in report.items() if key != "artifact_id"}
    report_core["typed_control_effect_observed"] = not report_core["typed_control_effect_observed"]
    forged_report = {
        **report_core,
        "artifact_id": sha256_json(report_core),
    }
    report_path.write_bytes((campaign.canonical_json(forged_report) + "\n").encode("utf-8"))

    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_core = {key: value for key, value in manifest.items() if key != "artifact_id"}
    manifest_core["report_artifact_id"] = forged_report["artifact_id"]
    for entry in manifest_core["files"]:
        if entry["path"] == "report.json":
            raw = report_path.read_bytes()
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
            entry["bytes"] = len(raw)
            break
    forged_manifest = {
        **manifest_core,
        "artifact_id": sha256_json(manifest_core),
    }
    manifest_path.write_bytes((campaign.canonical_json(forged_manifest) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="offline chain/metric recomputation"):
        campaign.validate_relationship_product_horizon_campaign(
            output_dir=tampered,
        )


def test_resealed_chain_environment_forgery_is_recomputed_and_rejected(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    tampered = tmp_path / "resealed-chain-tamper"
    shutil.copytree(smoke_campaign, tampered)
    chain_path = next(
        path for path in sorted((tampered / "chains").glob("*/*/chain.json")) if path.parent.name == "volvence_full"
    )
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    chain_core = {key: value for key, value in chain.items() if key != "artifact_id"}
    chain_core["decisions"][0]["rendered_user_reaction"] = "forged public reaction"
    forged_chain = {**chain_core, "artifact_id": sha256_json(chain_core)}
    chain_path.write_bytes((campaign.canonical_json(forged_chain) + "\n").encode("utf-8"))

    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_core = {key: value for key, value in manifest.items() if key != "artifact_id"}
    relative_chain_path = chain_path.relative_to(tampered).as_posix()
    for entry in manifest_core["files"]:
        if entry["path"] == relative_chain_path:
            raw = chain_path.read_bytes()
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
            entry["bytes"] = len(raw)
            break
    else:  # pragma: no cover - fixture manifest invariant
        raise AssertionError("typed chain missing from manifest")
    forged_manifest = {
        **manifest_core,
        "artifact_id": sha256_json(manifest_core),
    }
    manifest_path.write_bytes((campaign.canonical_json(forged_manifest) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="does not recompute from the environment"):
        campaign.validate_relationship_product_horizon_campaign(
            output_dir=tampered,
        )


def test_resealed_protocol_cannot_replace_packaged_preregistration(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    tampered = tmp_path / "resealed-protocol-tamper"
    shutil.copytree(smoke_campaign, tampered)
    protocol_path = tampered / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["analysis"]["development_directional_effect_floor"] = 0.0
    protocol_path.write_bytes((campaign.canonical_json(protocol) + "\n").encode("utf-8"))
    _reseal_manifest_file(
        root=tampered,
        changed_path=protocol_path,
        protocol_id=sha256_json(protocol),
    )
    with pytest.raises(ValueError, match="packaged preregistration SSOT"):
        campaign.validate_relationship_product_horizon_campaign(
            output_dir=tampered,
        )


def test_resealed_baseline_raw_output_cannot_self_report_valid_action(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    tampered = tmp_path / "resealed-raw-output-tamper"
    shutil.copytree(smoke_campaign, tampered)
    chain_path = next(
        path
        for path in sorted((tampered / "chains").glob("*/*/chain.json"))
        if path.parent.name == "native_full_history"
    )
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    result = chain["decisions"][0]["baseline_result"]
    completion = result["action_completion"]
    completion["raw_output"] = "garbage-not-json"
    completion_core = {key: value for key, value in completion.items() if key != "artifact_id"}
    completion["artifact_id"] = sha256_json(completion_core)
    result_core = {key: value for key, value in result.items() if key != "artifact_id"}
    result["artifact_id"] = sha256_json(result_core)
    chain_core = {key: value for key, value in chain.items() if key != "artifact_id"}
    chain["artifact_id"] = sha256_json(chain_core)
    chain_path.write_bytes((campaign.canonical_json(chain) + "\n").encode("utf-8"))
    _reseal_manifest_file(root=tampered, changed_path=chain_path)
    with pytest.raises(ValueError, match="strictly reproduce chosen/valid"):
        campaign.validate_relationship_product_horizon_campaign(
            output_dir=tampered,
        )


def _reseal_manifest_file(
    *,
    root: pathlib.Path,
    changed_path: pathlib.Path,
    protocol_id: str | None = None,
) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_core = {key: value for key, value in manifest.items() if key != "artifact_id"}
    if protocol_id is not None:
        manifest_core["protocol_id"] = protocol_id
    relative_path = changed_path.relative_to(root).as_posix()
    for entry in manifest_core["files"]:
        if entry["path"] == relative_path:
            raw = changed_path.read_bytes()
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
            entry["bytes"] = len(raw)
            break
    else:  # pragma: no cover - fixture manifest invariant
        raise AssertionError(f"changed file missing from manifest: {relative_path}")
    forged_manifest = {
        **manifest_core,
        "artifact_id": sha256_json(manifest_core),
    }
    manifest_path.write_bytes((campaign.canonical_json(forged_manifest) + "\n").encode("utf-8"))
