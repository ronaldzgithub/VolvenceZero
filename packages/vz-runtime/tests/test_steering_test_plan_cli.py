from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from volvence_zero.agent.steering_promotion_gate import SteeringComponent
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_steering_promotion_test_plan as promotion_plan  # noqa: E402
import run_dialogue_steering_test_plan as dialogue_plan  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_a1_bundle(root: Path, *, passed: bool = True) -> Path:
    preregistration_sha256 = "b" * 64
    contrasts = (
        "correct-user-state-vs-stateless",
        "correct-user-state-vs-swapped-user-state",
        "correct-user-state-vs-shuffled-history",
        "sleep-consolidation-vs-no-sleep",
    )
    gates = {
        f"{contrast}:{suffix}": passed
        for contrast in contrasts
        for suffix in ("n-plus-one-coverage", "n-plus-one-primary-effect")
    }
    result = {
        "schema_version": "seven-day-companion-ablation.v2",
        "preregistration_sha256": preregistration_sha256,
        "run_count": 36,
        "case_count": 6,
        "daily_readouts": [],
        "comparisons": [
            {
                "contrast_id": contrast,
                "expected_pair_count": 6,
                "complete_n_plus_one_pair_count": 6,
                "n_plus_one_prediction_gain_mean": 0.03 if passed else -0.01,
                "n_plus_one_prediction_gain_ci95": (
                    [0.01, 0.05] if passed else [-0.03, 0.01]
                ),
            }
            for contrast in contrasts
        ],
        "gates": gates,
        "passed": passed,
        "claim_scope": "simulated-user-real-lifecycle-only",
        "production_promotion_authorized": False,
        "evaluation_writeback_allowed": False,
    }
    result_path = root / "ablation_results.json"
    verdict_path = root / "promotion_verdict.json"
    _write_json(result_path, result)
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "seven-day-companion-ablation.v2",
            "preregistration_sha256": preregistration_sha256,
            "arm_schedule": list(dialogue_plan.A1_ARM_SCHEDULE),
            "case_count": 6,
            "run_count": 36,
            "required_files": [
                "ablation_results.json",
                "daily_metrics.jsonl",
                "promotion_verdict.json",
                "report.md",
            ],
            "claim_scope": "simulated-user-real-lifecycle-only",
        },
    )
    _write_json(
        verdict_path,
        {
            "schema_version": "seven-day-companion-verdict.v1",
            "passed": passed,
            "claim_scope": "simulated-user-real-lifecycle-only",
            "external_human_value_claim_allowed": False,
            "production_promotion_authorized": False,
            "evaluation_writeback_allowed": False,
            "failed_gates": [name for name, ok in gates.items() if not ok],
        },
    )
    _write_json(
        root / "audit" / "independent_audit.json",
        {
            "schema_version": "seven-day-companion-independent-audit.v1",
            "passed": True,
            "preregistration_sha256": preregistration_sha256,
            "execution_source_snapshot": {"tree_sha256": "c" * 64},
            "ablation_results_sha256": _sha256(result_path),
            "promotion_verdict_sha256": _sha256(verdict_path),
            "counts": {
                "cases": 6,
                "runs": 36,
                "turns": 1260,
                "http_errors": 0,
            },
            "checks": {
                "full_source_tree_revalidated": True,
                "exact_preregistered_matrix": True,
                "frozen_user_turn_replay": True,
                "matched_arm_inputs": True,
                "restart_identity_chain": True,
                "state_archive_digests": True,
                "measurement_checkpoint_digests": True,
                "state_source_selection": True,
                "pilot_transcript_digests": True,
                "service_session_evidence": True,
                "evaluation_recomputed_exactly": True,
                "production_promotion_authorized": False,
                "evaluation_writeback_allowed": False,
            },
            "claim_scope": "simulated-user-real-lifecycle-only",
        },
    )
    return result_path


def _args(tmp_path: Path) -> argparse.Namespace:
    c3_preregistration = tmp_path / "c3-prereg.json"
    _write_json(
        c3_preregistration,
        {
            "schema_version": "dialogue-steering-formal-prereg.v1",
            "claim": "fixture",
            "run_configuration": {
                "model_id": "frozen-model",
                "model_weights_sha256": "a" * 64,
                "steering_layer_index": 20,
                "activation_width": 896,
                "max_length": 768,
                "runtime_max_new_tokens": 16,
            },
        },
    )
    return argparse.Namespace(
        c3_preregistration=c3_preregistration,
        c3_output=tmp_path / "c3-output",
        preregistration=tmp_path / "b3-prereg.json",
        output=tmp_path / "b3-output",
        bootstrap_resamples=100,
    )


def test_c3_a1_attestation_requires_exact_audited_n_plus_one_formal(
    tmp_path: Path,
) -> None:
    result_path = _write_a1_bundle(tmp_path / "a1", passed=False)

    attestation = dialogue_plan._a1_attestation(result_path)

    assert attestation["schema_version"] == (
        "a1-n-plus-one-formal-attestation.v1"
    )
    assert attestation["run_count"] == 36
    assert attestation["n_plus_one_primary_passed"] is False
    assert attestation["ablation_result_sha256"] == _sha256(result_path)


def test_c3_a1_attestation_rejects_token_only_or_unaudited_results(
    tmp_path: Path,
) -> None:
    token_only = tmp_path / "token-only.json"
    _write_json(token_only, {"n+1": "complete", "status": "pass"})
    with pytest.raises(ValueError, match="complete 36-run"):
        dialogue_plan._a1_attestation(token_only)

    result_path = _write_a1_bundle(tmp_path / "a1")
    (result_path.parent / "audit" / "independent_audit.json").unlink()
    with pytest.raises(FileNotFoundError, match="independent audit"):
        dialogue_plan._a1_attestation(result_path)


def test_c3_provenance_is_resolved_next_to_extracted_corpus(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "data/external/msc/v0.1/extracted"
    corpus.mkdir(parents=True)
    provenance = corpus.parent / "DOWNLOAD_PROVENANCE.json"
    _write_json(
        provenance,
        {"schema_version": "msc-download-provenance.v1"},
    )

    assert dialogue_plan._msc_provenance_path(corpus) == provenance

    provenance.unlink()
    misplaced = corpus.parents[1] / "DOWNLOAD_PROVENANCE.json"
    _write_json(
        misplaced,
        {"schema_version": "msc-download-provenance.v1"},
    )
    with pytest.raises(FileNotFoundError, match="next to the corpus"):
        dialogue_plan._msc_provenance_path(corpus)


def test_c3_context_resume_rejects_artifact_lineage_drift() -> None:
    bundle = SimpleNamespace(
        reader=SimpleNamespace(
            artifact_id="reader-v1",
            model_id="model-v1",
            model_weights_sha256="a" * 64,
        ),
        executor=SimpleNamespace(
            artifact_id="executor-v1",
            layer_index=20,
            control_norm_cap_ratio=0.25,
        ),
        gate=SimpleNamespace(artifact_id="gate-v1", policy_version=3),
        sensor_off_executor=SimpleNamespace(
            artifact_id="executor-unconditional-v1"
        ),
    )
    shadow = SimpleNamespace(
        reader_artifact_id="reader-v1",
        executor_artifact_id="executor-v1",
        gate_policy_artifact_id="gate-v1",
        gate_policy_version=3,
        sensor_off_executor_artifact_id="executor-unconditional-v1",
        source_model_id="model-v1",
        source_model_weights_sha256="a" * 64,
        layer_index=20,
        residual_norm=2.0,
        control_norm_cap=0.5,
    )
    samples = (SimpleNamespace(steering_shadow=shadow),)

    dialogue_plan._validate_shadow_bundle_lineage(samples, bundle=bundle)
    samples_with_drift = (
        SimpleNamespace(
            steering_shadow=SimpleNamespace(
                **(vars(shadow) | {"executor_artifact_id": "stale-executor"})
            )
        ),
    )
    with pytest.raises(ValueError, match="artifact/model lineage drift"):
        dialogue_plan._validate_shadow_bundle_lineage(
            samples_with_drift,
            bundle=bundle,
        )


def test_b3_preregistration_is_idempotent_and_binds_independent_controls(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)

    assert promotion_plan._preregister(args) == 0
    first = args.preregistration.read_bytes()
    assert promotion_plan._preregister(args) == 0

    assert args.preregistration.read_bytes() == first
    payload = json.loads(first)
    configuration = payload["run_configuration"]
    assert configuration["gate_off_controls"] == ["noop", "always_on"]
    assert configuration["sensor_off_control"] == (
        "matched-budget-unconditional-operator"
    )
    assert configuration["promotion_order"] == [
        "steering_sensor",
        "steering_executor",
        "steering_gate",
    ]
    assert {
        "packages/lifeform-expression/src/lifeform_expression/llm_synthesizer.py",
        "packages/lifeform-service/src/lifeform_service/app.py",
        "packages/lifeform-service/src/lifeform_service/steering_activation.py",
        "packages/vz-contracts/src/volvence_zero/runtime/kernel.py",
        "packages/vz-cognition/src/volvence_zero/steering_sensor.py",
        "packages/vz-runtime/src/volvence_zero/agent/response.py",
        "packages/vz-runtime/src/volvence_zero/agent/session.py",
        "packages/vz-runtime/src/volvence_zero/brain.py",
        "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
        "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
        "packages/vz-substrate/src/volvence_zero/steering_executor.py",
        "packages/vz-temporal/src/volvence_zero/steering_gate.py",
    } <= set(configuration["source_sha256"])
    assert "learned-active-eta-off-gate" in payload["forbidden_substitutions"]


def test_b3_preregistration_rejects_post_c3_result_retrofit(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    _write_json(args.c3_output / "report.json", {"observed": True})

    with pytest.raises(ValueError, match="must precede every C3 formal artifact"):
        promotion_plan._preregister(args)

    assert not args.preregistration.exists()


def test_b3_preflight_does_not_treat_missing_c3_results_as_passed_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    args = _args(tmp_path)
    promotion_plan._preregister(args)
    capsys.readouterr()

    assert promotion_plan._preflight(args) == 2
    payload = json.loads(capsys.readouterr().out)

    assert payload["passed"] is False
    assert not any(payload["c3_artifacts_present"].values())
    assert payload["c3_artifacts_valid"] is False


def test_b3_activation_plan_changes_exactly_one_valid_field_per_rollout(
    tmp_path: Path,
) -> None:
    verdict = SimpleNamespace(
        eligible_prefix=(
            SteeringComponent.SENSOR,
            SteeringComponent.EXECUTOR,
            SteeringComponent.GATE,
        ),
        rollback_order=(
            SteeringComponent.GATE,
            SteeringComponent.EXECUTOR,
            SteeringComponent.SENSOR,
        ),
    )
    plan = promotion_plan._activation_plan(
        verdict,
        candidate_bundle_path=tmp_path / "candidate.json",
        candidate_bundle_sha256="a" * 64,
        deployment_contract={
            "model_id": "frozen-model",
            "model_weights_sha256": "a" * 64,
            "steering_layer_index": 20,
            "activation_width": 896,
            "substrate_max_length": 768,
            "generation_max_new_tokens": 16,
            "generation_temperature": 0.0,
            "fail_on_truncation": True,
        },
    )
    state = {
        "steering_sensor": "shadow",
        "steering_executor": "shadow",
        "steering_gate": "shadow",
        "steering_ungated_action": "blocked",
    }

    def validate(step: dict[str, object]) -> None:
        flip = step["single_field_flip"]
        assert isinstance(flip, dict)
        assert state[flip["field"]] == flip["from"]
        state[flip["field"]] = flip["to"]
        assert step["rollout_values_after_flip"] == state
        FinalRolloutConfig(
            steering_sensor=WiringLevel(state["steering_sensor"]),
            steering_executor=WiringLevel(state["steering_executor"]),
            steering_gate=WiringLevel(state["steering_gate"]),
            steering_ungated_action=state["steering_ungated_action"],
        )

    for step in plan["steps"]:
        validate(step)
    assert state == {
        "steering_sensor": "active",
        "steering_executor": "active",
        "steering_gate": "active",
        "steering_ungated_action": "blocked",
    }
    for step in plan["rollback_steps"]:
        validate(step)
    assert state == {
        "steering_sensor": "shadow",
        "steering_executor": "shadow",
        "steering_gate": "shadow",
        "steering_ungated_action": "blocked",
    }
    assert plan["deployment_contract"]["substrate_max_length"] == 768
    all_steps = (*plan["steps"], *plan["rollback_steps"])
    assert all(
        step["rollout_values_after_flip"]["steering_ungated_action"]
        != "steer"
        for step in all_steps
    )
