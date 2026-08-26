from __future__ import annotations

import asyncio
import copy
import json
import pathlib
import runpy
import sys

import pytest

import lifeform_evolution.relationship_product_horizon_development_campaign as subject
from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind
from volvence_zero.social import social_record_store_persistence_sha256


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
_SCANNER_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_transductive_public_opportunity_20260826_"
    "p4471c9ab49bc_c0ffda0a1976d"
)
_DYNAMIC_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_dynamic_collection_prefix_20260826_"
    "p47cea5fae3be_cc275bd908afd"
)
_FORCED_ROOT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_forced_common_batch_20260827_"
    "pdd0d28a72f9e_c5d028d9ce41b"
)


@pytest.fixture(scope="module")
def dependencies() -> subject._Dependencies:
    return subject._load_dependencies(
        source_v4_admission_root=_SOURCE_ROOT,
        reader_root=_READER_ROOT,
        theta0_v2_root=_THETA_ROOT,
        scanner_root=_SCANNER_ROOT,
        dynamic_root=_DYNAMIC_ROOT,
        forced_common_batch_root=_FORCED_ROOT,
    )


def test_protocol_freezes_no_rehearsal_three_arm_development_boundary(
    tmp_path: pathlib.Path,
) -> None:
    protocol = subject.load_relationship_product_horizon_development_campaign_protocol()
    assert protocol.protocol_id == subject._EXPECTED_PROTOCOL_ID
    assert protocol.raw_sha256 == subject._EXPECTED_PROTOCOL_RAW_SHA256
    assert protocol.payload["design"]["root_count"] == 112
    assert protocol.payload["design"]["arm_ids"] == [
        "full",
        "frozen_theta0",
        "strict_noop",
    ]
    assert not protocol.payload["design"]["rehearsal_enabled"]
    assert not protocol.payload["design"]["rehearsal_required"]
    assert protocol.payload["runtime_order"]["model_invocation_count"] == 0
    assert protocol.payload["runtime_order"]["cuda_execution_count"] == 0
    assert not protocol.payload["claims"]["confirmatory_effect_tested"]
    assert not protocol.payload["claims"]["learnable_effect"]
    assert not protocol.payload["claims"]["steerable_effect"]

    mutated = json.loads(protocol.raw_bytes)
    mutated["design"]["rehearsal_enabled"] = True
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol identity drifted"):
        subject.load_relationship_product_horizon_development_campaign_protocol(path)


def test_protocol_record_field_digest_matches_implementation() -> None:
    protocol = subject.load_relationship_product_horizon_development_campaign_protocol()
    field_sets = {
        key: sorted(value) for key, value in subject._TRACE_FIELDS.items()
    }
    field_sets["root_arm_terminal_state"] = sorted(
        subject._TERMINAL_STATE_FIELDS
    )
    assert subject.sha256_json(field_sets) == protocol.payload["trace_contract"][
        "record_field_sets_sha256"
    ]


def test_campaign_plan_is_public_complete_and_sealed_free(
    dependencies: subject._Dependencies,
) -> None:
    plan = subject._build_campaign_plan(dependencies=dependencies)
    assert plan["plan_id"] == subject.sha256_json(
        {key: value for key, value in plan.items() if key != "plan_id"}
    )
    assert plan["root_count"] == 112
    assert len(plan["roots"]) == 112
    assert [
        tuple(root["arm_order"]) for root in plan["roots"][:6]
    ] == [tuple(item.value for item in order) for order in subject._ARM_ORDERS]
    assert [
        sum(index % 6 == permutation for index in range(112))
        for permutation in range(6)
    ] == [19, 19, 19, 19, 18, 18]
    assert all(len(root["evaluation_decisions"]) == 40 for root in plan["roots"])
    forbidden_keys = {
        "condition_id",
        "policy_id",
        "preferred_action_id",
        "typed_outcome_id",
        "environment_seed",
        "sealed_evaluator_bundle",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert forbidden_keys.isdisjoint(keys(plan))


def test_bootstrap_counter_stream_and_status_boundaries() -> None:
    first = subject._bootstrap_root_indices(
        replicate_index=0,
        root_count=112,
        seed_hex="6a09e667f3bcc909",
        domain="relationship-product-horizon-development-campaign-bootstrap-v1",
    )
    second = subject._bootstrap_root_indices(
        replicate_index=1,
        root_count=112,
        seed_hex="6a09e667f3bcc909",
        domain="relationship-product-horizon-development-campaign-bootstrap-v1",
    )
    assert len(first) == len(second) == 112
    assert first[:12] == subject._BOOTSTRAP_REPLICATE_ZERO_FIRST_TWELVE
    assert first != second
    assert all(0 <= value < 112 for value in (*first, *second))
    protocol = subject.load_relationship_product_horizon_development_campaign_protocol()
    assert list(first[:12]) == protocol.payload["bootstrap"][
        "replicate_zero_first_twelve_root_indices"
    ]
    assert subject._effect_class(0.0, 0.0) == "directionally_nonpositive"
    assert subject._effect_class(0.049, 0.01) == (
        "directionally_positive_below_practical_floor"
    )
    assert subject._effect_class(0.05, 0.0) == (
        "at_or_above_practical_floor_interval_inconclusive"
    )
    assert subject._effect_class(0.05, 0.001) == (
        "at_or_above_practical_floor_positive_bound"
    )


def test_contrast_mechanism_and_terminal_claims_fail_closed() -> None:
    mechanism = {
        "learnable_actual_action_divergence_count": 1,
        "steerable_actual_action_divergence_count": 0,
        "frozen_theta0_physical_nonnoop_count": 0,
        "full_learned_policy_differs_from_cold_root_count": 112,
    }
    assert not subject._contrast_mechanism_valid(
        contrast_id=subject._LEARNABLE_CONTRAST_ID,
        mechanism=mechanism,
    )
    mechanism["steerable_actual_action_divergence_count"] = 1
    mechanism["frozen_theta0_physical_nonnoop_count"] = 1
    assert subject._contrast_mechanism_valid(
        contrast_id=subject._LEARNABLE_CONTRAST_ID,
        mechanism=mechanism,
    )
    assert subject._contrast_mechanism_valid(
        contrast_id=subject._STEERABLE_CONTRAST_ID,
        mechanism=mechanism,
    )

    claims = subject._terminal_claims(
        {
            subject._LEARNABLE_CONTRAST_ID: {
                "mechanism_valid": True,
                "development_go_candidate": True,
            },
            subject._STEERABLE_CONTRAST_ID: {
                "mechanism_valid": False,
                "development_go_candidate": False,
            },
        }
    )
    protocol_claims = subject.load_relationship_product_horizon_development_campaign_protocol().payload[
        "claims"
    ]
    assert set(claims) == set(protocol_claims)
    assert not claims["development_campaign_execution_authorized"]
    assert not claims["power_prereg_design_authorized"]
    assert claims["development_go_candidate_by_contrast"] == {
        subject._LEARNABLE_CONTRAST_ID: True,
        subject._STEERABLE_CONTRAST_ID: False,
    }
    assert claims["power_prereg_design_authorized_by_contrast"] == {
        subject._LEARNABLE_CONTRAST_ID: False,
        subject._STEERABLE_CONTRAST_ID: False,
    }

    claims = subject._terminal_claims(
        {
            subject._LEARNABLE_CONTRAST_ID: {
                "mechanism_valid": True,
                "development_go_candidate": True,
            },
            subject._STEERABLE_CONTRAST_ID: {
                "mechanism_valid": True,
                "development_go_candidate": False,
            },
        }
    )
    assert claims["power_prereg_design_authorized"]
    assert claims["power_prereg_design_authorized_by_contrast"] == {
        subject._LEARNABLE_CONTRAST_ID: True,
        subject._STEERABLE_CONTRAST_ID: False,
    }


def test_persisted_transition_binding_distinguishes_apply_from_withhold() -> None:
    initialization = {"batch_id": "batch", "batch_receipt_id": "receipt"}
    assert subject._expected_persisted_transition_binding(
        arm=subject.RelationshipProductHorizonCampaignArm.FULL,
        initialization=initialization,
    ) == ("batch", "receipt")
    assert subject._expected_persisted_transition_binding(
        arm=subject.RelationshipProductHorizonCampaignArm.FROZEN_THETA0,
        initialization=initialization,
    ) == (None, None)
    assert subject._expected_persisted_transition_binding(
        arm=subject.RelationshipProductHorizonCampaignArm.STRICT_NOOP,
        initialization=initialization,
    ) == (None, None)


def test_cli_summary_uses_terminal_claim_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: pathlib.Path,
) -> None:
    namespace = runpy.run_path(
        str(_REPO_ROOT / "scripts/run_relationship_product_horizon_development_campaign.py")
    )
    manifest = {
        "status": "development_campaign_completed_stop_no_effect_claim",
        "protocol_id": "p",
        "artifact_id": "a",
        "trace_row_count": 36066,
        "terminal_state_row_count": 336,
        "claims": {
            "development_contrast_estimated": True,
            "power_prereg_design_authorized": False,
            "confirmatory_effect_tested": False,
        },
    }
    monkeypatch.setitem(
        namespace["main"].__globals__,
        "materialize_relationship_product_horizon_development_campaign",
        lambda **_kwargs: manifest,
    )
    shared = [
        "--source-v4-admission-root",
        str(tmp_path / "source"),
        "--reader-root",
        str(tmp_path / "reader"),
        "--theta0-v2-root",
        str(tmp_path / "theta"),
        "--scanner-root",
        str(tmp_path / "scanner"),
        "--dynamic-root",
        str(tmp_path / "dynamic"),
        "--forced-common-batch-root",
        str(tmp_path / "forced"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_relationship_product_horizon_development_campaign.py",
            "materialize",
            *shared,
            "--implementation-git-commit",
            "0" * 40,
        ],
    )
    assert namespace["main"]() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["power_prereg_design_authorized"] is False
    assert "new_power_and_prereg_protocol_design_authorized" not in summary


def test_streaming_sink_is_create_only_fsync_bound_and_has_no_linear_cache(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "trace.jsonl"
    sink = subject._CreateOnlyStreamingJsonlSink(path)
    receipt = sink.append_many_fsync(
        (
            {"schema_version": "unit.v1", "record_type": "one"},
            {"schema_version": "unit.v1", "record_type": "two"},
        )
    )
    assert receipt.start_index == 0
    assert receipt.end_index == 1
    assert len(receipt.row_ids) == 2
    assert sink.row_count == 2
    assert sink.raw_bytes == path.stat().st_size
    assert not hasattr(sink, "_chunks")
    sink.close()
    with pytest.raises(FileExistsError):
        subject._CreateOnlyStreamingJsonlSink(path)

    failed_path = tmp_path / "failed.jsonl"
    failed = subject._CreateOnlyStreamingJsonlSink(failed_path)
    monkeypatch.setattr(subject.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync failed")))
    with pytest.raises(OSError, match="fsync failed"):
        failed.append_many_fsync(
            ({"schema_version": "unit.v1", "record_type": "not-durable"},)
        )
    failed.close()


def test_one_real_triplet_settles_only_after_durable_barrier(
    dependencies: subject._Dependencies,
    tmp_path: pathlib.Path,
) -> None:
    root_input = dependencies.inputs.roots[0]
    root = root_input.public_root
    decision = root.decision_sessions[8]
    order = subject._arm_order(0)
    initializations = {
        item.arm_id: item for item in root_input.fresh_arm_initializations()
    }
    prepared = []
    pre_rows = []
    for physical_index, arm in enumerate(order):
        initialization = initializations[arm]
        owner = initialization.owner_persistence_snapshot
        owner_sha = social_record_store_persistence_sha256(owner)
        preaction = asyncio.run(
            subject.prepare_relationship_product_frozen_preaction(
                request=subject._request(root=root, decision=decision),
                owner_persistence_snapshot=owner,
                forecast_runtime=initialization.forecast_runtime,
                frozen_policy=initialization.frozen_policy,
                executor_disposition=initialization.executor_disposition,
                authorization=subject._authorization(
                    protocol_id=dependencies.protocol.protocol_id,
                    root_sequence_index=0,
                    arm_id=arm,
                    initialization=initialization,
                ),
                substrate_snapshot=subject._placeholder_substrate(),
            )
        )
        prepared.append((arm, preaction))
        pre_rows.append(
            subject._preaction_payload(
                root_sequence_index=0,
                root=root,
                decision=decision,
                arm_id=arm,
                physical_arm_order_index=physical_index,
                owner_input_sha256=owner_sha,
                initialization=initialization,
                preaction=preaction,
            )
        )
    pre_by_arm = {row["arm_id"]: row for row in pre_rows}
    assert pre_by_arm["full"]["transition_batch_id"] is not None
    assert pre_by_arm["full"]["transition_receipt_id"] is not None
    for arm_id in ("frozen_theta0", "strict_noop"):
        assert pre_by_arm[arm_id]["transition_batch_id"] is None
        assert pre_by_arm[arm_id]["transition_receipt_id"] is None
    opened: list[bool] = []

    class Environment:
        def settle(
            self,
            *,
            public_root: object,
            public_decision: object,
            selected_action: RelationshipAction,
        ) -> subject.dynamic.RelationshipProductHorizonSelectedBranchOutcome:
            del public_decision
            return subject.dynamic.RelationshipProductHorizonSelectedBranchOutcome(
                environment_subject_id=public_root.subject_id,
                selected_action=selected_action,
                typed_outcome=DialogueExternalOutcomeKind.HELPED,
                rendered_user_reaction="typed unit reaction",
                environment_evidence_ref=(
                    f"unit-environment:{selected_action.value}"
                ),
                environment_version="unit.v1",
                commitment_id=subject.cal._sha256_text(selected_action.value),
            )

    def open_environment() -> Environment:
        opened.append(True)
        return Environment()

    sink = subject._CreateOnlyStreamingJsonlSink(tmp_path / "triplet.jsonl")
    durable_rows = sink.append_many_fsync(tuple(pre_rows))
    assert opened == []
    barrier = subject._mint_preaction_barrier(
        sink=sink,
        protocol_id=dependencies.protocol.protocol_id,
        root_sequence_index=0,
        decision_index=8,
        arm_order=order,
        preactions=tuple(prepared),
        durable_preactions=durable_rows,
    )
    assert opened == []
    owner = subject._OnceOnlySelectedBranchSettlementOwner(
        environment_opener=open_environment
    )
    settlements = asyncio.run(
        owner.settle_triplet(
            durable=subject._DurablePreactions(
                barrier=barrier,
                preactions=tuple(prepared),
            ),
            root=root,
            decision=decision,
            root_sequence_index=0,
        )
    )
    sink.close()
    assert opened == [True]
    assert len(settlements) == 3
    assert all(not item.settled.credit_applied_to_gate for item in settlements)
    assert all(item.settled.evaluation_gate_update_delta == 0 for item in settlements)
    for physical_index, item in enumerate(settlements):
        pre_payload = pre_by_arm[item.arm_id.value]
        forecast = subject._validate_persisted_executor_receipt(
            preaction=pre_payload,
            protocol_id=dependencies.protocol.protocol_id,
            root_sequence_index=0,
            arm=item.arm_id,
            initialization=initializations[item.arm_id],
        )
        post_payload = subject._postaction_payload(
            protocol_id=dependencies.protocol.protocol_id,
            barrier=barrier,
            root_sequence_index=0,
            root=root,
            decision=decision,
            physical_arm_order_index=physical_index,
            item=item,
        )
        assert subject._validate_persisted_settlement_join(
            postaction=post_payload,
            preaction=pre_payload,
            forecast=forecast,
            root=root,
            decision=decision,
            root_sequence_index=0,
        ) is DialogueExternalOutcomeKind.HELPED
    outcomes_by_action: dict[RelationshipAction, object] = {}
    for item in settlements:
        action = RelationshipAction(item.settled.preaction.delivered_action_id)
        prior = outcomes_by_action.setdefault(action, item.outcome)
        assert prior is item.outcome

    full_pre = copy.deepcopy(pre_by_arm["full"])
    receipt = full_pre["executor_receipt"]
    receipt["executor_status"] = "strict_noop"
    receipt_core = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = (
        f"{subject._EXECUTOR_RECEIPT_PREFIX}"
        f"{subject._pulse_payload_sha256(receipt_core)}"
    )
    full_pre["executor_receipt_id"] = receipt["receipt_id"]
    full_pre["executor_status"] = "strict_noop"
    with pytest.raises(ValueError, match="executor"):
        subject._validate_persisted_executor_receipt(
            preaction=full_pre,
            protocol_id=dependencies.protocol.protocol_id,
            root_sequence_index=0,
            arm=subject.RelationshipProductHorizonCampaignArm.FULL,
            initialization=initializations[
                subject.RelationshipProductHorizonCampaignArm.FULL
            ],
        )

    full_item = next(
        item
        for item in settlements
        if item.arm_id is subject.RelationshipProductHorizonCampaignArm.FULL
    )
    full_post = subject._postaction_payload(
        protocol_id=dependencies.protocol.protocol_id,
        barrier=barrier,
        root_sequence_index=0,
        root=root,
        decision=decision,
        physical_arm_order_index=order.index(full_item.arm_id),
        item=full_item,
    )
    tampered_post = copy.deepcopy(full_post)
    tampered_post["settlement"]["observed_outcome_id"] = "missed"
    with pytest.raises(ValueError, match="settlement"):
        subject._validate_persisted_settlement_join(
            postaction=tampered_post,
            preaction=pre_by_arm["full"],
            forecast=subject._validate_persisted_executor_receipt(
                preaction=pre_by_arm["full"],
                protocol_id=dependencies.protocol.protocol_id,
                root_sequence_index=0,
                arm=subject.RelationshipProductHorizonCampaignArm.FULL,
                initialization=initializations[
                    subject.RelationshipProductHorizonCampaignArm.FULL
                ],
            ),
            root=root,
            decision=decision,
            root_sequence_index=0,
        )
    strict = next(
        item
        for item in settlements
        if item.arm_id
        is subject.RelationshipProductHorizonCampaignArm.STRICT_NOOP
    )
    assert strict.settled.preaction.delivered_action_id == (
        RelationshipAction.NEUTRAL_NOOP.value
    )
    with pytest.raises(ValueError, match="barrier or campaign slot drifted"):
        asyncio.run(
            owner.settle_triplet(
                durable=subject._DurablePreactions(
                    barrier=barrier,
                    preactions=tuple(prepared),
                ),
                root=root,
                decision=decision,
                root_sequence_index=0,
            )
        )
