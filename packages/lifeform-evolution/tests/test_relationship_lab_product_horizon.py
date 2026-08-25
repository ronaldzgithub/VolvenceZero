from __future__ import annotations

from collections.abc import Iterator
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
    build_relationship_product_pilot_evaluator_bundle,
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
    load_precomputed_public_embedding_table,
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


def _fake_table(
    path: pathlib.Path,
    *,
    protocol: campaign.RelationshipProductHorizonProtocol,
) -> None:
    source = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(source)
    records = []
    embedder = _FakeSemanticEmbedder()
    for text in campaign.relationship_product_required_semantic_texts(
        public,
        protocol=protocol,
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


def _validate_v2_campaign(output_dir: pathlib.Path) -> dict[str, object]:
    protocol = campaign.load_relationship_product_horizon_protocol(
        output_dir / "protocol.json"
    )
    return dict(
        campaign.validate_relationship_product_horizon_campaign(
            output_dir=output_dir,
            expected_protocol_id=protocol.protocol_id,
        )
    )


def _campaign_protocol(
    output_dir: pathlib.Path,
) -> campaign.RelationshipProductHorizonProtocol:
    return campaign.load_relationship_product_horizon_protocol(
        output_dir / "protocol.json"
    )


def _write_non_authorizing_current_tree_test_protocol(
    path: pathlib.Path,
) -> campaign.RelationshipProductHorizonProtocol:
    repository_root = pathlib.Path(__file__).resolve().parents[3]
    tree_summary, _ = campaign._local_execution_source_tree(
        repository_root=repository_root
    )
    payload = json.loads(
        campaign.relationship_product_horizon_protocol_path("v2").read_text(
            encoding="utf-8"
        )
    )
    payload["execution"]["local_execution_source_tree"] = tree_summary
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return campaign.load_relationship_product_horizon_protocol(path)


def _write_minimal_local_source_repository(
    root: pathlib.Path,
    *,
    python_bytes: bytes = b"VALUE = 1\n",
    active_protocol_bytes: bytes = b'{"version":1}\n',
) -> None:
    python_path = root / "packages" / "demo" / "src" / "demo" / "module.py"
    python_path.parent.mkdir(parents=True)
    python_path.write_bytes(python_bytes)
    for relative_path in (
        *campaign._LOCAL_EXECUTION_SOURCE_ENTRYPOINTS,
        *campaign._LOCAL_EXECUTION_RESOURCE_PATHS,
    ):
        path = root / pathlib.PurePosixPath(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"resource:{relative_path}\n".encode())
    active_protocol_path = root / pathlib.PurePosixPath(
        campaign._LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH
    )
    active_protocol_path.parent.mkdir(parents=True, exist_ok=True)
    active_protocol_path.write_bytes(active_protocol_bytes)


@pytest.fixture(scope="module")
def smoke_campaign(tmp_path_factory: pytest.TempPathFactory) -> Iterator[pathlib.Path]:
    root = tmp_path_factory.mktemp("relationship-product-horizon")
    table_path = root / "fake-public-embedding-table.json"
    protocol_path = root / "current-tree.non_authorizing_test.json"
    protocol = _write_non_authorizing_current_tree_test_protocol(protocol_path)
    _fake_table(table_path, protocol=protocol)
    assert len(load_precomputed_public_embedding_table(table_path).records) == 30
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
    previous_test_protocol = os.environ.get(
        campaign._NON_AUTHORIZING_TEST_PROTOCOL_ENV
    )
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ[campaign._NON_AUTHORIZING_TEST_PROTOCOL_ENV] = str(
        protocol_path.resolve()
    )
    try:
        campaign.run_relationship_product_horizon_campaign(
            output_dir=output,
            public_embedding_table_path=table_path,
            protocol_path=protocol_path,
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
        yield output
    finally:
        if previous is None:
            os.environ.pop("PYTHONNOUSERSITE", None)
        else:
            os.environ["PYTHONNOUSERSITE"] = previous
        if previous_test_protocol is None:
            os.environ.pop(campaign._NON_AUTHORIZING_TEST_PROTOCOL_ENV, None)
        else:
            os.environ[
                campaign._NON_AUTHORIZING_TEST_PROTOCOL_ENV
            ] = previous_test_protocol


def test_v1_v2_protocol_registry_semantic_tables_and_truth_firewall_are_strict() -> None:
    v1_path = campaign.relationship_product_horizon_protocol_path("v1")
    v2_path = campaign.relationship_product_horizon_protocol_path("v2")
    assert campaign.relationship_product_horizon_protocol_paths() == (
        v2_path,
        v1_path,
    )

    v1 = campaign.load_relationship_product_horizon_protocol(v1_path)
    v2 = campaign.load_relationship_product_horizon_protocol(v2_path)
    assert campaign.load_relationship_product_horizon_protocol() == v2
    assert campaign._registered_product_protocol_for_id(v1.protocol_id) == v1
    assert campaign._registered_product_protocol_for_id(v2.protocol_id) == v2

    assert v1.schema_version == campaign.RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V1
    assert v1.protocol_id == (
        "13cceb4cc36970e40431e59056f51534e19b86b4e6f4ec616e2fce39f91a7bb1"
    )
    assert v1.volvence_arms == (
        "volvence_full",
        "appendable_frozen_onboarding",
        "readable_permuted",
        "credit_withheld",
        "strict_noop",
    )
    assert v1.semantic_table_record_count == 28
    assert v1.persists_full_forecast is False
    assert v1.condition_reader_artifact_id is None
    assert v1.baseline_constrained_action_choice is False

    assert v2.schema_version == campaign.RELATIONSHIP_PRODUCT_HORIZON_SCHEMA_VERSION_V2
    assert v2.volvence_arms == (
        "volvence_full",
        "appendable_frozen_onboarding",
        "readable_unnamed_legacy",
        "credit_withheld",
        "strict_noop",
    )
    assert v2.semantic_table_record_count == 30
    assert v2.persists_full_forecast is True
    assert v2.condition_reader_artifact_id == (
        campaign.relationship_product_condition_reader_artifact().artifact_id
    )
    assert v2.baseline_constrained_action_choice is True

    source = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(source)
    v1_texts = campaign.relationship_product_required_semantic_texts(
        public,
        protocol=v1,
    )
    v2_texts = campaign.relationship_product_required_semantic_texts(
        public,
        protocol=v2,
    )
    prototype_texts = {
        item.summary
        for item in campaign.relationship_product_condition_reader_artifact().prototypes
    }
    assert len(v1_texts) == 28
    assert len(v2_texts) == 30
    assert set(v2_texts) - set(v1_texts) == prototype_texts

    assert v2.subject_count == 8
    assert v2.decision_sessions_per_subject == 24
    assert v2.context_window_tokens == 32768
    assert v2.source_protocol_id == ("048b73d4a412b4444fb469be0d9daa6d2a26e9920c743804da8f36dc331691ae")
    with pytest.raises(ValueError, match="leaked sealed evaluator keys"):
        campaign._assert_truth_firewall({"public": {}, "preferred_action_id": "x"})


@pytest.mark.parametrize("protocol_revision", ("v1", "v2"))
def test_historical_campaign_protocols_revalidate_exact_legacy_source_binding(
    protocol_revision: str,
) -> None:
    source = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(source)
    evaluator = build_relationship_product_pilot_evaluator_bundle(source)
    protocol = campaign.load_relationship_product_horizon_protocol(
        campaign.relationship_product_horizon_protocol_path(protocol_revision)
    )

    campaign._validate_campaign_source_binding(protocol, source, public, evaluator)


def test_v2_local_source_tree_normalizes_newlines_but_preserves_eof_and_detaches_protocol(
    tmp_path: pathlib.Path,
) -> None:
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    no_eof_root = tmp_path / "no-eof"
    _write_minimal_local_source_repository(
        lf_root,
        python_bytes=b"VALUE = 1\nNEXT = 2\n",
        active_protocol_bytes=b'{"version":1}\n',
    )
    _write_minimal_local_source_repository(
        crlf_root,
        python_bytes=b"VALUE = 1\r\nNEXT = 2\r\n",
        active_protocol_bytes=b'{"version":2}\n',
    )
    _write_minimal_local_source_repository(
        no_eof_root,
        python_bytes=b"VALUE = 1\nNEXT = 2",
        active_protocol_bytes=b'{"version":1}\n',
    )

    lf_summary, _ = campaign._local_execution_source_tree(repository_root=lf_root)
    crlf_summary, _ = campaign._local_execution_source_tree(
        repository_root=crlf_root
    )
    no_eof_summary, _ = campaign._local_execution_source_tree(
        repository_root=no_eof_root
    )

    assert lf_summary == crlf_summary
    assert lf_summary["tree_sha256"] != no_eof_summary["tree_sha256"]

    active_protocol = lf_root / pathlib.PurePosixPath(
        campaign._LOCAL_EXECUTION_ACTIVE_PROTOCOL_PATH
    )
    active_protocol.write_bytes(b'{"version":999}\n')
    changed_protocol_summary, _ = campaign._local_execution_source_tree(
        repository_root=lf_root
    )
    assert changed_protocol_summary == lf_summary


def test_v2_protocol_uses_tree_authority_without_legacy_raw_source_pins() -> None:
    payload = json.loads(
        campaign.relationship_product_horizon_protocol_path("v2").read_text(
            encoding="utf-8"
        )
    )
    execution = payload["execution"]
    assert "execution_source_sha256s" not in execution
    assert execution["local_execution_source_tree"]["tree_sha256"] == (
        campaign.load_relationship_product_horizon_protocol(
            campaign.relationship_product_horizon_protocol_path("v2")
        ).local_execution_source_tree_sha256
    )


def test_historical_v2_protocol_rejects_current_checkout_source_tree(
    tmp_path: pathlib.Path,
) -> None:
    protocol = campaign.load_relationship_product_horizon_protocol(
        campaign.relationship_product_horizon_protocol_path("v2")
    )
    repository_root = pathlib.Path(__file__).resolve().parents[3]

    with pytest.raises(ValueError, match="local Python source tree differs from protocol pin"):
        campaign._publish_execution_source_bundle(
            root=tmp_path / "historical-v2-rejection",
            protocol=protocol,
            campaign_cli=repository_root
            / "scripts"
            / "run_relationship_lab_product_horizon.py",
        )


def test_v2_child_environment_removes_python_path_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHOME", "C:/forged-python-home")
    monkeypatch.setenv("PYTHONPATH", "C:/forged-import-root")
    environment = campaign._child_environment()
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_v2_baseline_dispatcher_identity_requires_one_shared_process() -> None:
    shared_attestation = campaign._with_artifact_id({"process_pid": 4123})
    resident_chain = {
        "dispatcher_startup_attestation": shared_attestation,
        "decisions": [
            {"baseline_execution_backend": "resident_jsonl_dispatcher"}
        ],
    }
    summary = campaign._baseline_dispatcher_identity_summary(
        [resident_chain, json.loads(json.dumps(resident_chain))]
    )
    assert summary == {
        "startup_artifact_ids": [shared_attestation["artifact_id"]],
        "process_pids": [4123],
        "single_resident_dispatcher_verified": True,
    }

    second_attestation = campaign._with_artifact_id({"process_pid": 4124})
    second_chain = {
        "dispatcher_startup_attestation": second_attestation,
        "decisions": [
            {"baseline_execution_backend": "resident_jsonl_dispatcher"}
        ],
    }
    with pytest.raises(ValueError, match="do not share one resident"):
        campaign._baseline_dispatcher_identity_summary(
            [resident_chain, second_chain]
        )

    injected_summary = campaign._baseline_dispatcher_identity_summary(
        [
            {
                "dispatcher_startup_attestation": None,
                "decisions": [
                    {"baseline_execution_backend": "injected_resident_suite"}
                ],
            }
        ]
    )
    assert injected_summary == {
        "startup_artifact_ids": [],
        "process_pids": [],
        "single_resident_dispatcher_verified": False,
    }


def test_cross_process_smoke_covers_restart_world_clone_and_state_contrasts(
    smoke_campaign: pathlib.Path,
) -> None:
    report = _validate_v2_campaign(smoke_campaign)
    assert report["typed_control_executed"] is True
    assert report["strong_baselines_executed"] is True
    assert (
        report["fresh_process_launch_receipt_per_volvence_logical_session"]
        is True
    )
    assert report["volvence_logical_session_count"] == 30
    assert len(report["worker_request_artifact_ids"]) == 30
    assert len(set(report["worker_request_artifact_ids"])) == 30
    assert report["distinct_child_pid_count"] > 1
    assert report["residual_steerable"] is False
    assert report["four_able_complete"] is False
    assert report["production_active"] is False
    assert report["os_security_boundary"] is False
    assert report["horizon_durability_pass"] is False
    assert report["baseline_dispatcher_startup_artifact_ids"] == []
    assert report["baseline_dispatcher_process_pids"] == []
    assert report["baseline_single_resident_dispatcher_verified"] is False

    chain_files = sorted((smoke_campaign / "chains").glob("*/*/chain.json"))
    chains = [json.loads(path.read_text(encoding="utf-8")) for path in chain_files]
    assert len({chain["world_clone_id"] for chain in chains}) == 1
    frozen = next(chain for chain in chains if chain["arm_id"] == "appendable_frozen_onboarding")
    frozen_pre_hashes = {decision["pre_owner_snapshot_sha256"] for decision in frozen["decisions"]}
    assert len(frozen_pre_hashes) == 1
    assert [decision["gate_update_count_after"] for decision in frozen["decisions"]] == [1, 2]
    full = next(chain for chain in chains if chain["arm_id"] == "volvence_full")
    assert full["decisions"][1]["pre_owner_snapshot_sha256"] == full["decisions"][0]["post_owner_snapshot_sha256"]
    readable = next(
        chain
        for chain in chains
        if chain["arm_id"] == "readable_unnamed_legacy"
    )
    protocol = _campaign_protocol(smoke_campaign)
    for decision in full["decisions"]:
        request = json.loads(
            (smoke_campaign / decision["request_path"]).read_text(encoding="utf-8")
        )
        preaction = json.loads(
            (smoke_campaign / decision["preaction_receipt_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert request["named_reader"] == "prototype_named_condition_readout"
        assert preaction["schema_version"] == (
            campaign.RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2
        )
        readout = preaction["frozen_forecast"]["forecast"]["condition_readout"]
        assert readout is not None
        assert readout["reader_artifact_id"] == protocol.condition_reader_artifact_id
        assert readout["source_observation_sha256"] == hashlib.sha256(
            request["session"]["current_input"].encode("utf-8")
        ).hexdigest()
        lineage = preaction["execution_source_lineage"]
        assert lineage["local_execution_source_tree_sha256"] == (
            protocol.local_execution_source_tree_sha256
        )
        assert lineage["worker_script_repository_path"] == (
            "scripts/run_relationship_lab_product_horizon.py"
        )
        assert all(
            origin["repository_path"].startswith("packages/")
            for origin in lineage["critical_module_origins"]
        )
        assert lineage["volvence_zero_namespace_search_locations"]
        assert lineage["loaded_local_module_origins"]
        assert all(
            origin["module_name"]
            == campaign._module_name_for_repository_path(
                origin["repository_path"]
            )
            for origin in lineage["loaded_local_module_origins"]
        )
        postaction = json.loads(
            (smoke_campaign / decision["postaction_receipt_path"]).read_text(
                encoding="utf-8"
            )
        )
        post_lineage = postaction["execution_source_lineage"]
        assert post_lineage["local_execution_source_tree_sha256"] == (
            protocol.local_execution_source_tree_sha256
        )
        assert post_lineage["loaded_local_module_origins"]
    for decision in readable["decisions"]:
        request = json.loads(
            (smoke_campaign / decision["request_path"]).read_text(encoding="utf-8")
        )
        preaction = json.loads(
            (smoke_campaign / decision["preaction_receipt_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert request["named_reader"] == "legacy_unnamed_semantic_similarity"
        assert preaction["schema_version"] == (
            campaign.RELATIONSHIP_PRODUCT_PREACTION_RECEIPT_SCHEMA_VERSION_V2
        )
        assert preaction["frozen_forecast"]["forecast"]["condition_readout"] is None
    withheld = next(chain for chain in chains if chain["arm_id"] == "credit_withheld")
    assert [decision["gate_update_count_after"] for decision in withheld["decisions"]] == [0, 0]
    assert all(decision["credit_applied_to_gate"] is False for decision in withheld["decisions"])
    noop = next(chain for chain in chains if chain["arm_id"] == "strict_noop")
    assert all(
        decision["selected_action_id"] == RelationshipAction.NEUTRAL_NOOP.value for decision in noop["decisions"]
    )


def test_v2_invalid_baseline_completions_fail_closed_in_report(
    smoke_campaign: pathlib.Path,
) -> None:
    protocol = _campaign_protocol(smoke_campaign)
    report = json.loads((smoke_campaign / "report.json").read_text(encoding="utf-8"))
    chain_files = sorted((smoke_campaign / "chains").glob("*/*/chain.json"))
    chains = [json.loads(path.read_text(encoding="utf-8")) for path in chain_files]
    typed_chains = [
        chain for chain in chains if chain["arm_id"] in protocol.volvence_arms
    ]
    invalid_baseline_chains = [
        json.loads(json.dumps(chain))
        for chain in chains
        if chain["arm_id"] in {"native_full_history", "selective_rag"}
    ]
    for chain in invalid_baseline_chains:
        for decision in chain["decisions"]:
            decision["valid_completion"] = False

    invalid_report = campaign._build_report(
        root=smoke_campaign,
        protocol=protocol,
        source_protocol_id=report["source_protocol_id"],
        public_plan_sha256=report["public_plan_sha256"],
        sealed_bundle_sha256=report["sealed_bundle_sha256"],
        embedding_table_artifact_id=report["embedding_table_artifact_id"],
        semantic_backend=report["semantic_backend"],
        embedding_table_fresh_process_reobserved=report[
            "embedding_table_fresh_process_reobserved"
        ],
        execution_source_bundle_artifact_id=report[
            "execution_source_bundle_artifact_id"
        ],
        selection=campaign.RelationshipProductCampaignSelection(
            subject_count=1,
            onboarding_session_count=4,
            decision_session_count=2,
        ),
        typed_chains=typed_chains,
        baseline_chains=invalid_baseline_chains,
    )

    assert invalid_report["baseline_completion_count"] == 4
    assert invalid_report["baseline_valid_completion_count"] == 0
    assert invalid_report["baseline_all_completions_valid"] is False
    baseline_comparisons = {
        item["comparator"]: item
        for item in invalid_report["paired_comparisons"]
        if item["comparator"] in {"native_full_history", "selective_rag"}
    }
    baseline_safety = {
        item["comparator"]: item
        for item in invalid_report["safety_noninferiority_comparisons"]
        if item["comparator"] in {"native_full_history", "selective_rag"}
    }
    assert set(baseline_comparisons) == {
        "native_full_history",
        "selective_rag",
    }
    assert all(
        item
        == {
            "comparator": comparator,
            "status": "invalid_output",
            "mean_paired_effect": None,
            "subjects_with_positive_effect": None,
            "directional_pass": False,
        }
        for comparator, item in baseline_comparisons.items()
    )
    assert all(
        item
        == {
            "comparator": comparator,
            "status": "invalid_output",
            "mean_full_safety_rate_increase": None,
            "noninferiority_pass": False,
        }
        for comparator, item in baseline_safety.items()
    )
    assert invalid_report["both_strong_baseline_directional_pass"] is False
    assert invalid_report["safety_noninferiority_pass"] is False
    assert invalid_report["stage_two_admission_candidate"] is False


def test_v2_product_stage_requires_direct_mechanism_closure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    protocol = campaign.load_relationship_product_horizon_protocol()
    incomplete_mechanism = {"direct_mechanism_evidence_complete": False}

    def _incomplete_mechanism_summary(**_: object) -> dict[str, bool]:
        return incomplete_mechanism

    monkeypatch.setattr(
        campaign,
        "_v2_mechanism_evidence_summary",
        _incomplete_mechanism_summary,
    )

    typed_chains: list[dict[str, object]] = []
    baseline_chains: list[dict[str, object]] = []
    for arm_index, arm_id in enumerate(protocol.volvence_arms):
        for subject_index in range(8):
            subject_scope = f"subject-{subject_index:02d}"
            positive = arm_id == "volvence_full"
            decisions = [
                {
                    "decision_index": decision_index,
                    "positive_outcome": positive,
                    "preferred_action_match": positive,
                    "typed_outcome_id": "felt_heard" if positive else "missed",
                    "launch_identity_sha256": sha256_json(
                        [arm_id, subject_scope, "decision", decision_index]
                    ),
                    "request_artifact_id": sha256_json(
                        [arm_id, subject_scope, "request", decision_index]
                    ),
                    "child_pid": 10_000
                    + arm_index * 1_000
                    + subject_index * 24
                    + decision_index,
                }
                for decision_index in range(24)
            ]
            onboarding_receipts = [
                {
                    "launch_identity_sha256": sha256_json(
                        [arm_id, subject_scope, "onboarding", session_index]
                    ),
                    "request_artifact_id": sha256_json(
                        [arm_id, subject_scope, "onboarding-request", session_index]
                    ),
                    "child_pid": 100_000
                    + arm_index * 1_000
                    + subject_index * 4
                    + session_index,
                }
                for session_index in range(4)
            ]
            typed_chains.append(
                {
                    "arm_id": arm_id,
                    "subject_scope": subject_scope,
                    "world_clone_id": sha256_json(["world", subject_scope]),
                    "decisions": decisions,
                    "onboarding_receipts": onboarding_receipts,
                }
            )

    dispatcher_startup = campaign._with_artifact_id({"process_pid": 6204})
    for arm_id in ("native_full_history", "selective_rag"):
        arm_prompt_sha256 = (
            protocol.baseline_native_prompt_sha256
            if arm_id == "native_full_history"
            else protocol.baseline_rag_prompt_sha256
        )
        for subject_index in range(8):
            subject_scope = f"subject-{subject_index:02d}"
            baseline_chains.append(
                {
                    "arm_id": arm_id,
                    "subject_scope": subject_scope,
                    "world_clone_id": sha256_json(["world", subject_scope]),
                    "dispatcher_startup_attestation": dispatcher_startup,
                    "decisions": [
                        {
                            "decision_index": decision_index,
                            "positive_outcome": False,
                            "preferred_action_match": False,
                            "typed_outcome_id": "missed",
                            "valid_completion": True,
                            "baseline_execution_backend": (
                                "resident_jsonl_dispatcher"
                            ),
                            "baseline_result": {
                                "context_receipt": {
                                    "generation_config_sha256": (
                                        protocol.baseline_generation_config_sha256
                                    ),
                                    "model_id": protocol.baseline_model_id,
                                    "weights_sha256": (
                                        protocol.baseline_model_weights_sha256
                                    ),
                                    "tokenizer_id": protocol.baseline_tokenizer_id,
                                    "arm_prompt_sha256": arm_prompt_sha256,
                                }
                            },
                        }
                        for decision_index in range(24)
                    ],
                }
            )

    report = campaign._build_report(
        root=tmp_path,
        protocol=protocol,
        source_protocol_id=sha256_json("source-protocol"),
        public_plan_sha256=sha256_json("public-plan"),
        sealed_bundle_sha256=sha256_json("sealed-bundle"),
        embedding_table_artifact_id=protocol.semantic_table_artifact_id,
        semantic_backend="bge_m3_precomputed_public_table",
        embedding_table_fresh_process_reobserved=True,
        execution_source_bundle_artifact_id=sha256_json("source-bundle"),
        selection=campaign.RelationshipProductCampaignSelection(
            subject_count=8,
            onboarding_session_count=4,
            decision_session_count=24,
        ),
        typed_chains=typed_chains,
        baseline_chains=baseline_chains,
    )

    assert report["all_targeted_intervention_directional_pass"] is True
    assert report["both_strong_baseline_directional_pass"] is True
    assert report["baseline_instrument_valid"] is True
    assert report["baseline_all_completions_valid"] is True
    assert report["safety_noninferiority_pass"] is True
    assert report["horizon_durability_pass"] is True
    assert report["mechanism_evidence"] == incomplete_mechanism
    assert report["internal_typed_control_ablation_effect_observed"] is False
    assert report["stage_two_admission_candidate"] is False
    assert report["product_stage_two_effect_observed"] is False


def test_manifest_tamper_is_rejected(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(smoke_campaign, tampered)
    report_path = tampered / "report.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest file tree/hash mismatch"):
        _validate_v2_campaign(tampered)


def test_external_expected_protocol_id_rejects_artifact_self_anchor(
    smoke_campaign: pathlib.Path,
) -> None:
    protocol = _campaign_protocol(smoke_campaign)
    assert campaign.validate_relationship_product_horizon_campaign(
        output_dir=smoke_campaign,
        expected_protocol_id=protocol.protocol_id,
    )["protocol_id"] == protocol.protocol_id
    with pytest.raises(ValueError, match="requires an external expected protocol id"):
        campaign.validate_relationship_product_horizon_campaign(
            output_dir=smoke_campaign,
        )
    with pytest.raises(ValueError, match="external expected protocol id"):
        campaign.validate_relationship_product_horizon_campaign(
            output_dir=smoke_campaign,
            expected_protocol_id="0" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra", "missing or extra files"),
        (
            "missing",
            "run_relationship_lab_product_horizon.py|missing or extra files",
        ),
        ("tamper", "file drifted"),
    ),
)
def test_v2_execution_source_bundle_rejects_extra_missing_and_tampered_files(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
    mutation: str,
    message: str,
) -> None:
    root = tmp_path / mutation
    (root / "inputs").mkdir(parents=True)
    shutil.copytree(
        smoke_campaign / "inputs" / "execution_sources",
        root / "inputs" / "execution_sources",
    )
    shutil.copy2(smoke_campaign / "protocol.json", root / "protocol.json")
    worker = (
        root
        / "inputs"
        / "execution_sources"
        / "tree"
        / "scripts"
        / "run_relationship_lab_product_horizon.py"
    )
    if mutation == "extra":
        extra = worker.with_name("unregistered_extra.py")
        extra.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    elif mutation == "missing":
        worker.unlink()
    elif mutation == "tamper":
        worker.write_bytes(worker.read_bytes() + b" ")
    else:  # pragma: no cover - parametrization invariant
        raise AssertionError(mutation)

    with pytest.raises((FileNotFoundError, ValueError), match=message):
        campaign._validate_execution_source_bundle(
            root=root,
            protocol=_campaign_protocol(smoke_campaign),
        )


def test_v2_execution_source_bundle_rejects_tree_external_hardlink(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    root = tmp_path / "external-hardlink"
    (root / "inputs").mkdir(parents=True)
    shutil.copytree(
        smoke_campaign / "inputs" / "execution_sources",
        root / "inputs" / "execution_sources",
    )
    shutil.copy2(smoke_campaign / "protocol.json", root / "protocol.json")
    worker = (
        root
        / "inputs"
        / "execution_sources"
        / "tree"
        / "scripts"
        / "run_relationship_lab_product_horizon.py"
    )
    outside_alias = root / "worker-outside-tree.py"
    worker.replace(outside_alias)
    os.link(outside_alias, worker)
    assert worker.stat().st_nlink >= 2

    with pytest.raises(ValueError, match="rejects hard-linked files"):
        campaign._validate_execution_source_bundle(
            root=root,
            protocol=_campaign_protocol(smoke_campaign),
        )


def test_v2_worker_lineage_rejects_resealed_module_name_path_forgery(
    smoke_campaign: pathlib.Path,
) -> None:
    chain_path = next(
        path
        for path in sorted((smoke_campaign / "chains").glob("*/*/chain.json"))
        if path.parent.name == "volvence_full"
    )
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    preaction = json.loads(
        (
            smoke_campaign
            / chain["decisions"][0]["preaction_receipt_path"]
        ).read_text(encoding="utf-8")
    )
    lineage = json.loads(json.dumps(preaction["execution_source_lineage"]))
    critical_names = {
        item["module_name"] for item in lineage["critical_module_origins"]
    }
    forged_origin = next(
        item
        for item in lineage["loaded_local_module_origins"]
        if item["module_name"] not in critical_names
    )
    forged_origin["module_name"] = f"{forged_origin['module_name']}.__forged__"
    lineage["loaded_local_module_origins"].sort(
        key=lambda item: item["module_name"].encode("utf-8")
    )
    forged_lineage = campaign._with_artifact_id(
        {key: value for key, value in lineage.items() if key != "artifact_id"}
    )
    execution_source_bundle = json.loads(
        (
            smoke_campaign / "inputs" / "execution_sources" / "bundle.json"
        ).read_text(encoding="utf-8")
    )

    with pytest.raises(ValueError, match="name/path mapping drifted"):
        campaign._validate_v2_worker_source_lineage(
            forged_lineage,
            root=smoke_campaign,
            protocol=_campaign_protocol(smoke_campaign),
            execution_source_bundle_artifact_id=execution_source_bundle[
                "artifact_id"
            ],
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
    report_core["internal_typed_control_ablation_effect_observed"] = not report_core[
        "internal_typed_control_ablation_effect_observed"
    ]
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
        _validate_v2_campaign(tampered)


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
        _validate_v2_campaign(tampered)


def test_resealed_protocol_cannot_replace_packaged_preregistration(
    smoke_campaign: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    tampered = tmp_path / "resealed-protocol-tamper"
    shutil.copytree(smoke_campaign, tampered)
    protocol_path = tampered / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["analysis"]["development_directional_effect_floor"] = 0.06
    protocol_path.write_bytes((campaign.canonical_json(protocol) + "\n").encode("utf-8"))
    _reseal_manifest_file(
        root=tampered,
        changed_path=protocol_path,
        protocol_id=sha256_json(protocol),
    )
    with pytest.raises(ValueError, match="packaged preregistration SSOT"):
        _validate_v2_campaign(tampered)


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
        _validate_v2_campaign(tampered)


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
