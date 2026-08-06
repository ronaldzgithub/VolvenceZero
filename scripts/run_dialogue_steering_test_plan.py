"""C3 control plane: preregister and run real-dialogue steering transfer.

Formal data are MSC SHADOW turns.  Raw text is used transiently by the service
and target publisher, but checkpoints retain only normalized residual values,
hash lineage, owner observations, latency, and PE settlements.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Iterable

from companion_bench.msc_corpus import MSCDyad, load_msc_split
from companion_bench.msc_runtime_collection import (
    MSCFullRuntimeCollectedSample,
    collect_msc_full_runtime_contexts,
    msc_runtime_scope_ids,
    parse_msc_collected_samples,
    serialize_msc_collected_samples,
)
from companion_bench.prediction_research import (
    MSCNextTurnExample,
    build_msc_next_turn_examples,
)
from companion_test_plan_common import exclusive_mps_lock, require_mps
from run_msc_prediction_research import (
    MSC_STEERING_SEMANTIC_PROPOSAL_CHANNEL,
    _acquire_output_lock,
    _running_msc_runtime_service,
    _stable_subset,
    _substrate_context_limit,
)
from volvence_zero.agent.dialogue_steering_evidence import (
    DialogueSteeringThresholds,
    DialogueSteeringTraceDataset,
    DialogueSteeringTraceRow,
    run_dialogue_steering_evidence,
)
from volvence_zero.agent.eta_conditional_steering_screen import (
    ACTION_PROMPT_SUFFIX,
)
from volvence_zero.agent.eta_conflict_instrument import (
    build_conflict_junction_rows,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_rate_distortion_evidence import _action_options
from volvence_zero.agent.steering_artifact_training import (
    fit_steering_artifact_bundle,
)
from volvence_zero.prediction import (
    ForwardRepresentationBatch,
    PredictionErrorModule,
    settle_steering_terminal_prediction_error,
)
from volvence_zero.steering_contracts import SteeringArtifactBundle
from volvence_zero.substrate import (
    SubstrateFingerprint,
    SubstrateForwardRepresentationPublisher,
    build_transformers_runtime_with_fallback,
    fingerprint_model_weight_files,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PREREG_SCHEMA = "dialogue-steering-formal-prereg.v1"
PLAN_ID = "dialogue-steering-c3-formal.v1"
C3_DEFAULT_SUBSTRATE_MAX_LENGTH = 32768
A1_RESULT_SCHEMA = "seven-day-companion-ablation.v2"
A1_AUDIT_SCHEMA = "seven-day-companion-independent-audit.v1"
A1_ARM_SCHEDULE = (
    "correct-user-state",
    "stateless",
    "swapped-user-state",
    "shuffled-history",
    "sleep-consolidation",
    "no-sleep",
)
SOURCE_FILES = (
    "packages/companion-bench/src/companion_bench/msc_corpus.py",
    "packages/companion-bench/src/companion_bench/prediction_research.py",
    "packages/companion-bench/src/companion_bench/msc_runtime_collection.py",
    "packages/lifeform-evolution/src/lifeform_evolution/seven_day_companion.py",
    "packages/lifeform-expression/src/lifeform_expression/llm_synthesizer.py",
    "packages/lifeform-service/src/lifeform_service/app.py",
    "packages/lifeform-service/src/lifeform_service/cli.py",
    "packages/lifeform-service/src/lifeform_service/companion_evidence_profile.py",
    "packages/lifeform-service/src/lifeform_service/msc_runtime_collector.py",
    "packages/lifeform-service/src/lifeform_service/steering_activation.py",
    "packages/lifeform-service/src/lifeform_service/verticals.py",
    "packages/vz-contracts/src/volvence_zero/steering_contracts.py",
    "packages/vz-contracts/src/volvence_zero/runtime/kernel.py",
    "packages/vz-cognition/src/volvence_zero/steering_sensor.py",
    "packages/vz-cognition/src/volvence_zero/prediction/error.py",
    "packages/vz-cognition/src/volvence_zero/prediction/forward_representation.py",
    "packages/vz-cognition/src/volvence_zero/credit/gate.py",
    "packages/vz-substrate/src/volvence_zero/substrate/forward_representation.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
    "packages/vz-substrate/src/volvence_zero/steering_executor.py",
    "packages/vz-temporal/src/volvence_zero/steering_gate.py",
    "packages/vz-runtime/src/volvence_zero/agent/dialogue_steering_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/steering_artifact_training.py",
    "packages/vz-runtime/src/volvence_zero/agent/response.py",
    "packages/vz-runtime/src/volvence_zero/agent/session.py",
    "packages/vz-runtime/src/volvence_zero/brain.py",
    "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
    "scripts/companion_test_plan_common.py",
    "scripts/msc_prediction_checkpoint.py",
    "scripts/run_msc_prediction_research.py",
    "scripts/run_dialogue_steering_test_plan.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _msc_provenance_path(msc_root: Path) -> Path:
    path = msc_root.resolve().parent / "DOWNLOAD_PROVENANCE.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"C3 requires MSC DOWNLOAD_PROVENANCE.json next to the corpus: {path}"
        )
    payload = _load_json_object(path, label="MSC download provenance")
    if payload.get("schema_version") != "msc-download-provenance.v1":
        raise ValueError("C3 MSC download provenance schema is unsupported")
    return path


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: object) -> None:
    _write_immutable(path, _canonical_bytes(value))


def _write_gzip(path: Path, payload: str) -> None:
    _write_immutable(path, gzip.compress(payload.encode("utf-8"), mtime=0))


def _read_gzip(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8")


def _batches(indices: tuple[int, ...], size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _a1_attestation(path: Path) -> dict[str, object]:
    result_path = (
        path.resolve() / "ablation_results.json"
        if path.is_dir()
        else path.resolve()
    )
    if not result_path.is_file():
        raise FileNotFoundError(f"A1 formal result is missing: {result_path}")
    root = result_path.parent
    result = _load_json_object(result_path, label="A1 ablation result")
    gates = result.get("gates")
    comparisons = result.get("comparisons")
    if (
        result.get("schema_version") != A1_RESULT_SCHEMA
        or result.get("run_count") != 36
        or result.get("case_count") != 6
        or result.get("claim_scope") != "simulated-user-real-lifecycle-only"
        or result.get("production_promotion_authorized") is not False
        or result.get("evaluation_writeback_allowed") is not False
        or not isinstance(result.get("passed"), bool)
        or not _valid_sha256(result.get("preregistration_sha256"))
        or not isinstance(gates, dict)
        or not gates
        or not all(
            isinstance(name, str) and isinstance(passed, bool)
            for name, passed in gates.items()
        )
        or result.get("passed") is not all(gates.values())
        or not isinstance(comparisons, list)
        or len(comparisons) != 4
    ):
        raise ValueError("A1 result is not a complete 36-run N+1 formal verdict")
    expected_contrasts = {
        "correct-user-state-vs-stateless",
        "correct-user-state-vs-swapped-user-state",
        "correct-user-state-vs-shuffled-history",
        "sleep-consolidation-vs-no-sleep",
    }
    observed_contrasts = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("A1 comparison must be an object")
        contrast_id = comparison.get("contrast_id")
        if not isinstance(contrast_id, str):
            raise ValueError("A1 comparison contrast_id is invalid")
        observed_contrasts.add(contrast_id)
        mean = comparison.get("n_plus_one_prediction_gain_mean")
        interval = comparison.get("n_plus_one_prediction_gain_ci95")
        if (
            comparison.get("expected_pair_count") != 6
            or comparison.get("complete_n_plus_one_pair_count") != 6
            or not isinstance(mean, (int, float))
            or isinstance(mean, bool)
            or not math.isfinite(float(mean))
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in interval
            )
        ):
            raise ValueError("A1 comparison lacks complete N+1 effect evidence")
    expected_gate_names = {
        f"{contrast_id}:{suffix}"
        for contrast_id in expected_contrasts
        for suffix in ("n-plus-one-coverage", "n-plus-one-primary-effect")
    }
    if observed_contrasts != expected_contrasts or set(gates) != expected_gate_names:
        raise ValueError("A1 N+1 contrast/gate matrix drift")

    manifest_path = root / "manifest.json"
    verdict_path = root / "promotion_verdict.json"
    manifest = _load_json_object(manifest_path, label="A1 manifest")
    verdict = _load_json_object(verdict_path, label="A1 promotion verdict")
    preregistration_sha256 = result["preregistration_sha256"]
    required_files = manifest.get("required_files")
    if (
        manifest.get("schema_version") != A1_RESULT_SCHEMA
        or manifest.get("preregistration_sha256") != preregistration_sha256
        or manifest.get("arm_schedule") != list(A1_ARM_SCHEDULE)
        or manifest.get("case_count") != 6
        or manifest.get("run_count") != 36
        or manifest.get("claim_scope") != "simulated-user-real-lifecycle-only"
        or not isinstance(required_files, list)
        or not all(isinstance(value, str) for value in required_files)
        or set(required_files)
        != {
            "ablation_results.json",
            "daily_metrics.jsonl",
            "promotion_verdict.json",
            "report.md",
        }
    ):
        raise ValueError("A1 manifest does not bind the exact formal matrix")
    expected_failed = {name for name, passed in gates.items() if not passed}
    failed_gates = verdict.get("failed_gates")
    if (
        verdict.get("schema_version") != "seven-day-companion-verdict.v1"
        or verdict.get("passed") is not result["passed"]
        or verdict.get("claim_scope") != "simulated-user-real-lifecycle-only"
        or verdict.get("external_human_value_claim_allowed") is not False
        or verdict.get("production_promotion_authorized") is not False
        or verdict.get("evaluation_writeback_allowed") is not False
        or not isinstance(failed_gates, list)
        or not all(isinstance(value, str) for value in failed_gates)
        or len(set(failed_gates)) != len(failed_gates)
        or set(failed_gates) != expected_failed
    ):
        raise ValueError("A1 promotion verdict differs from the N+1 result")

    audit_candidates = (
        root / "audit" / "independent_audit.json",
        root / "independent_audit.json",
    )
    audit_paths = tuple(
        candidate for candidate in audit_candidates if candidate.is_file()
    )
    if not audit_paths:
        raise FileNotFoundError("A1 independent audit is missing")
    if (
        len(audit_paths) > 1
        and audit_paths[0].read_bytes() != audit_paths[1].read_bytes()
    ):
        raise ValueError("A1 independent audit copies differ")
    audit_path = audit_paths[0]
    audit = _load_json_object(audit_path, label="A1 independent audit")
    counts = audit.get("counts")
    checks = audit.get("checks")
    source_snapshot = audit.get("execution_source_snapshot")
    required_true_checks = {
        "full_source_tree_revalidated",
        "exact_preregistered_matrix",
        "frozen_user_turn_replay",
        "matched_arm_inputs",
        "restart_identity_chain",
        "state_archive_digests",
        "measurement_checkpoint_digests",
        "state_source_selection",
        "pilot_transcript_digests",
        "service_session_evidence",
        "evaluation_recomputed_exactly",
    }
    if (
        audit.get("schema_version") != A1_AUDIT_SCHEMA
        or audit.get("passed") is not True
        or audit.get("preregistration_sha256") != preregistration_sha256
        or audit.get("ablation_results_sha256") != _sha256(result_path)
        or audit.get("promotion_verdict_sha256") != _sha256(verdict_path)
        or audit.get("claim_scope") != "simulated-user-real-lifecycle-only"
        or not isinstance(counts, dict)
        or counts.get("cases") != 6
        or counts.get("runs") != 36
        or counts.get("turns") != 1260
        or counts.get("http_errors") != 0
        or not isinstance(checks, dict)
        or any(checks.get(name) is not True for name in required_true_checks)
        or checks.get("production_promotion_authorized") is not False
        or checks.get("evaluation_writeback_allowed") is not False
        or not isinstance(source_snapshot, dict)
        or not _valid_sha256(source_snapshot.get("tree_sha256"))
    ):
        raise ValueError("A1 independent audit does not attest the formal result")
    return {
        "schema_version": "a1-n-plus-one-formal-attestation.v1",
        "output_root": str(root),
        "ablation_result_sha256": _sha256(result_path),
        "manifest_sha256": _sha256(manifest_path),
        "promotion_verdict_sha256": _sha256(verdict_path),
        "independent_audit_sha256": _sha256(audit_path),
        "preregistration_sha256": preregistration_sha256,
        "execution_source_tree_sha256": source_snapshot["tree_sha256"],
        "n_plus_one_primary_passed": result["passed"],
        "case_count": 6,
        "run_count": 36,
    }


def _formal_configuration(args: argparse.Namespace) -> dict[str, object]:
    model_source = args.model_source.resolve()
    substrate_context_limit = _validated_substrate_context_limit(
        model_source=model_source,
        configured_max_length=args.max_length,
    )
    weights_sha256 = fingerprint_model_weight_files(model_source)
    thresholds = asdict(DialogueSteeringThresholds())
    corpus_provenance = _msc_provenance_path(args.msc_root)
    return {
        "schema_version": "dialogue-steering-run-configuration.v1",
        "msc_root": str(args.msc_root.resolve()),
        "corpus_provenance_sha256": _sha256(corpus_provenance),
        "license_policy": "noncommercial-research-only",
        "model_id": args.model_id,
        "model_source": str(model_source),
        "model_weights_sha256": weights_sha256,
        "substrate_device": args.substrate_device,
        "substrate_model_dtype": args.substrate_model_dtype,
        "steering_layer_index": 20,
        "target_layer_indices": [11, 12, 13],
        "service_layer_indices": [11, 12, 13, 20],
        "activation_width": 896,
        "substrate_context_limit": substrate_context_limit,
        "max_length": args.max_length,
        "runtime_max_new_tokens": args.runtime_max_new_tokens,
        "runtime_semantic_proposal_channel": (
            MSC_STEERING_SEMANTIC_PROPOSAL_CHANNEL
        ),
        "temporal_n_z": 3,
        "train_dyads": args.train_dyads,
        "validation_dyads": args.validation_dyads,
        "head_n_z": 3,
        "head_seed": 1501,
        "head_epochs": args.head_epochs,
        "head_batch_size": args.head_batch_size,
        "head_learning_rate": 0.003,
        "gate_seed_schedule": list(args.seeds),
        "policy_restarts": 4,
        "max_online_episodes": 1200,
        "eval_every": 80,
        "gate_learning_rate": 0.05,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_confidence": 0.95,
        "thresholds": thresholds,
        "artifact_fit": {
            "corpus_seed": 20260802,
            "objective_count": 8,
            "corridor_count": 2,
            "extra_edge_probability": 0.35,
            "train_routes": 64,
            "heldout_routes": 24,
            "train_lengths": [2, 3],
            "heldout_lengths": [3, 4],
            "rank": 8,
            "executor_updates": 80,
            "executor_learning_rate": 0.01,
            "reader_ridge_lambda": 10.0,
            "batch_size": 32,
        },
        "source_sha256": _source_hashes(),
    }


def _validated_substrate_context_limit(
    *,
    model_source: Path,
    configured_max_length: int,
) -> int:
    context_limit = _substrate_context_limit(model_source)
    if configured_max_length != context_limit:
        raise ValueError(
            "C3 formal requires --max-length to equal the frozen substrate "
            "context limit so complete runtime prompts cannot be truncated: "
            f"configured={configured_max_length}, declared={context_limit}"
        )
    return context_limit


def _load_preregistration(
    path: Path,
    *,
    expected_configuration: dict[str, object],
    expected_a1: dict[str, object],
) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PREREG_SCHEMA:
        raise ValueError("C3 formal preregistration schema is invalid")
    if payload.get("run_configuration") != expected_configuration:
        raise ValueError("C3 formal run configuration drifted from preregistration")
    if payload.get("a1_n_plus_one_formal") != expected_a1:
        raise ValueError("C3 A1 prerequisite attestation drift")
    return payload, _sha256(path)


def _fit_bundle(
    *,
    args: argparse.Namespace,
    preregistration_sha256: str,
    output: Path,
    configuration: dict[str, object],
) -> tuple[SteeringArtifactBundle, bool]:
    bundle_path = output / "steering_artifact_bundle.json"
    fit_path = output / "steering_artifact_fit.json"
    if bundle_path.exists() != fit_path.exists():
        raise RuntimeError("C3 steering artifact fit checkpoint is partial")
    if bundle_path.exists() and fit_path.exists():
        bundle = SteeringArtifactBundle.from_json(
            bundle_path.read_text(encoding="utf-8")
        )
        report = json.loads(fit_path.read_text(encoding="utf-8"))
        if (
            bundle.reader.source_preregistration_sha256
            != preregistration_sha256
            or bundle.reader.model_id != args.model_id
            or bundle.reader.model_weights_sha256
            != configuration["model_weights_sha256"]
            or bundle.reader.layer_index != configuration["steering_layer_index"]
            or bundle.reader.residual_width != configuration["activation_width"]
            or bundle.sensor_off_executor is None
            or report.get("bundle_sha256") != _sha256(bundle_path)
        ):
            raise ValueError("C3 resumed steering artifact lineage drift")
        return bundle, bool(report["prerequisite_passed"])
    model_source = Path(str(configuration["model_source"]))
    runtime = build_transformers_runtime_with_fallback(
        model_id=args.model_id,
        model_source=str(model_source),
        device=args.substrate_device,
        model_dtype=args.substrate_model_dtype,
        layer_indices=(20,),
        activation_width=896,
        max_length=args.max_length,
        fail_on_truncation=True,
        local_files_only=True,
        fallback_mode="deny",
        runtime_mode="strict-local",
        expected_model_weights_sha256=str(configuration["model_weights_sha256"]),
    )
    corpus = generate_eta_proof_corpus(
        seed=20260802,
        objective_count=8,
        corridor_count=2,
        extra_edge_probability=0.35,
        train_route_count=64,
        heldout_route_count=24,
        train_lengths=(2, 3),
        heldout_lengths=(3, 4),
    )
    probe_rows = build_conflict_junction_rows(corpus, split="train")
    scorer = runtime.build_steered_action_scorer(
        action_options=_action_options(corpus.environment),
        injection_layer_index=20,
        prompt_suffix="",
        max_length=args.max_length,
        control_norm_ratio=0.25,
        probe_texts=tuple(
            row.observation_text + ACTION_PROMPT_SUFFIX for row in probe_rows[:16]
        ),
        joint_training=False,
        prefix_cache=True,
    )
    result = fit_steering_artifact_bundle(
        corpus=corpus,
        runtime=runtime,
        scorer=scorer,
        model_weights_sha256=str(configuration["model_weights_sha256"]),
        source_preregistration_sha256=preregistration_sha256,
        progress=lambda message: print(f"[artifact-fit] {message}", flush=True),
    )
    _write_immutable(bundle_path, (result.bundle.to_json() + "\n").encode("utf-8"))
    _write_json(
        fit_path,
        {
            "schema_version": "steering-artifact-fit-evidence.v1",
            "report": asdict(result.report),
            "prerequisite_passed": result.report.prerequisite_passed,
            "bundle_sha256": _sha256(bundle_path),
        },
    )
    del scorer, runtime
    gc.collect()
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result.bundle, result.report.prerequisite_passed


def _checkpoint_path(root: Path, split: str, dyad: MSCDyad) -> Path:
    digest = hashlib.sha256(dyad.dyad_id.encode("utf-8")).hexdigest()
    return root / "contexts" / split / f"{digest}.json.gz"


def _validate_shadow_bundle_lineage(
    samples: tuple[MSCFullRuntimeCollectedSample, ...],
    *,
    bundle: SteeringArtifactBundle,
) -> None:
    """Bind every resumable SHADOW row to the exact frozen artifact bundle."""

    sensor_off = bundle.sensor_off_executor
    if sensor_off is None:
        raise ValueError("C3 bundle lacks its sensor-off control artifact")
    if not samples:
        raise ValueError("C3 context checkpoint contains no samples")
    for sample in samples:
        shadow = sample.steering_shadow
        if shadow is None:
            raise ValueError("C3 context checkpoint lacks steering SHADOW data")
        if (
            shadow.reader_artifact_id != bundle.reader.artifact_id
            or shadow.executor_artifact_id != bundle.executor.artifact_id
            or shadow.gate_policy_artifact_id != bundle.gate.artifact_id
            or shadow.gate_policy_version != bundle.gate.policy_version
            or shadow.sensor_off_executor_artifact_id
            != sensor_off.artifact_id
            or shadow.source_model_id != bundle.reader.model_id
            or shadow.source_model_weights_sha256
            != bundle.reader.model_weights_sha256
            or shadow.layer_index != bundle.executor.layer_index
            or not math.isclose(
                shadow.control_norm_cap,
                bundle.executor.control_norm_cap_ratio * shadow.residual_norm,
                rel_tol=1e-8,
                abs_tol=1e-8,
            )
        ):
            raise ValueError(
                "C3 context checkpoint artifact/model lineage drift"
            )


def _collect_contexts(
    *,
    args: argparse.Namespace,
    output: Path,
    bundle: SteeringArtifactBundle,
    bundle_path: Path,
    split_dyads: dict[str, tuple[MSCDyad, ...]],
    configuration: dict[str, object],
) -> dict[str, tuple[MSCFullRuntimeCollectedSample, ...]]:
    collected: dict[tuple[str, str], tuple[MSCFullRuntimeCollectedSample, ...]] = {}
    missing: list[tuple[str, MSCDyad]] = []
    for split, dyads in split_dyads.items():
        for dyad in dyads:
            path = _checkpoint_path(output, split, dyad)
            expected = build_msc_next_turn_examples((dyad,))
            if path.exists():
                samples = parse_msc_collected_samples(_read_gzip(path))
                if tuple(item.sample_id for item in samples) != tuple(
                    item.sample_id for item in expected
                ):
                    raise ValueError("C3 context checkpoint sample drift")
                _validate_shadow_bundle_lineage(samples, bundle=bundle)
                collected[(split, dyad.dyad_id)] = samples
            else:
                missing.append((split, dyad))
    if missing:
        user_ids = tuple(msc_runtime_scope_ids(dyad)[1] for _, dyad in missing)
        with _running_msc_runtime_service(
            repository_root=REPOSITORY_ROOT,
            user_ids=user_ids,
            substrate_model=args.model_id,
            substrate_model_source=args.model_source.resolve(),
            substrate_device=args.substrate_device,
            substrate_model_dtype=args.substrate_model_dtype,
            substrate_layer_indices=(11, 12, 13, 20),
            substrate_activation_width=896,
            substrate_max_length=args.max_length,
            substrate_weights_sha256=str(configuration["model_weights_sha256"]),
            temporal_n_z=3,
            max_new_tokens=args.runtime_max_new_tokens,
            startup_timeout_s=args.runtime_startup_timeout,
            evidence_profile="msc-steering-shadow-collector-v1",
            steering_bundle_path=bundle_path.resolve(),
        ) as (service_factory, profile):
            _write_json(output / "service_profile_attestation.json", profile)
            for index, (split, dyad) in enumerate(missing, start=1):
                print(
                    f"[c3-shadow] {index}/{len(missing)} {split} "
                    f"scope={msc_runtime_scope_ids(dyad)[0][:12]}",
                    flush=True,
                )
                samples = collect_msc_full_runtime_contexts(
                    (dyad,), service_factory=service_factory
                )
                _validate_shadow_bundle_lineage(samples, bundle=bundle)
                _write_gzip(
                    _checkpoint_path(output, split, dyad),
                    serialize_msc_collected_samples(samples),
                )
                collected[(split, dyad.dyad_id)] = samples
    return {
        split: tuple(
            sample
            for dyad in dyads
            for sample in collected[(split, dyad.dyad_id)]
        )
        for split, dyads in split_dyads.items()
    }


def _make_batch(
    *,
    batch_id: str,
    examples: tuple[MSCNextTurnExample, ...],
    contexts: tuple[tuple[float, ...], ...],
    targets: tuple[tuple[float, ...], ...],
    persistence: tuple[tuple[float, ...], ...],
    indices: tuple[int, ...],
    lineage,
) -> ForwardRepresentationBatch:
    return ForwardRepresentationBatch(
        batch_id=batch_id,
        sample_ids=tuple(examples[index].sample_id for index in indices),
        context_representations=tuple(contexts[index] for index in indices),
        target_representations=tuple(targets[index] for index in indices),
        persistence_representations=tuple(persistence[index] for index in indices),
        history_turns=tuple(examples[index].history_turns for index in indices),
        target_lineage=lineage,
    )


def _build_trace_dataset(
    *,
    args: argparse.Namespace,
    output: Path,
    bundle: SteeringArtifactBundle,
    examples: dict[str, tuple[MSCNextTurnExample, ...]],
    samples: dict[str, tuple[MSCFullRuntimeCollectedSample, ...]],
    configuration: dict[str, object],
) -> DialogueSteeringTraceDataset:
    trace_path = output / "dialogue_steering_trace.json.gz"
    if trace_path.exists():
        dataset = DialogueSteeringTraceDataset.from_json(_read_gzip(trace_path))
        if dataset.bundle_id != bundle.bundle_id:
            raise ValueError("C3 resumed trace bundle lineage drift")
        for split, rows in (
            ("train", dataset.train_rows),
            ("validation", dataset.validation_rows),
        ):
            if tuple(row.sample_id for row in rows) != tuple(
                example.sample_id for example in examples[split]
            ):
                raise ValueError(f"C3 resumed {split} trace sample drift")
        sensor_off = bundle.sensor_off_executor
        if sensor_off is None or any(
            row.reader_artifact_id != bundle.reader.artifact_id
            or row.executor_artifact_id != bundle.executor.artifact_id
            or row.sensor_off_executor_artifact_id != sensor_off.artifact_id
            or row.source_model_id != bundle.reader.model_id
            or row.source_model_weights_sha256
            != bundle.reader.model_weights_sha256
            for row in (*dataset.train_rows, *dataset.validation_rows)
        ):
            raise ValueError("C3 resumed trace artifact/model lineage drift")
        return dataset
    runtime = build_transformers_runtime_with_fallback(
        model_id=args.model_id,
        model_source=str(args.model_source.resolve()),
        device=args.substrate_device,
        model_dtype=args.substrate_model_dtype,
        layer_indices=(11, 12, 13),
        activation_width=896,
        max_length=args.max_length,
        fail_on_truncation=True,
        local_files_only=True,
        fallback_mode="deny",
        runtime_mode="strict-local",
        expected_model_weights_sha256=str(configuration["model_weights_sha256"]),
    )
    fingerprint = SubstrateFingerprint(
        model_id=args.model_id,
        version=runtime.model_version,
        weights_sha256=str(configuration["model_weights_sha256"]),
    )
    publisher = SubstrateForwardRepresentationPublisher(
        runtime, model_fingerprint=fingerprint
    )
    all_examples = (*examples["train"], *examples["validation"])
    sample_sources = tuple(
        value
        for example in all_examples
        for value in (
            (f"{example.sample_id}:target", example.target_text),
            (f"{example.sample_id}:persistence", example.latest_text),
        )
    )
    target_snapshot = publisher.publish(
        sample_sources,
        progress=lambda sample_id, index, total: (
            print(f"[c3-target] {index}/{total} {sample_id}", flush=True)
            if index == total or index % 64 == 0
            else None
        ),
    )
    vectors = {
        row.sample_id: row.values for row in target_snapshot.representations
    }
    targets = {
        split: tuple(vectors[f"{item.sample_id}:target"] for item in rows)
        for split, rows in examples.items()
    }
    persistence = {
        split: tuple(vectors[f"{item.sample_id}:persistence"] for item in rows)
        for split, rows in examples.items()
    }
    if any(
        sample.steering_shadow is None
        for split_samples in samples.values()
        for sample in split_samples
    ):
        raise RuntimeError("C3 trace contains a partial steering owner chain")
    if any(
        len(samples[split]) != len(examples[split])
        or tuple(sample.sample_id for sample in samples[split])
        != tuple(example.sample_id for example in examples[split])
        for split in ("train", "validation")
    ):
        raise ValueError("C3 runtime context/example alignment drift")
    noop_contexts = {
        split: tuple(
            sample.steering_shadow.noop_context.values
            for sample in split_samples
            if sample.steering_shadow is not None
        )
        for split, split_samples in samples.items()
    }
    action_contexts = {
        split: tuple(
            sample.steering_shadow.action_context.values
            for sample in split_samples
            if sample.steering_shadow is not None
        )
        for split, split_samples in samples.items()
    }
    sensor_off_contexts = {
        split: tuple(
            sample.steering_shadow.sensor_off_action_context.values
            for sample in split_samples
            if sample.steering_shadow is not None
            and sample.steering_shadow.sensor_off_action_context is not None
        )
        for split, split_samples in samples.items()
    }
    if any(
        len(sensor_off_contexts[split]) != len(examples[split])
        for split in ("train", "validation")
    ):
        raise RuntimeError("C3 formal trace lacks the sensor-off counterfactual")
    module = PredictionErrorModule()
    module.configure_forward_representation_head(
        input_dim=len(noop_contexts["train"][0]),
        target_dim=len(targets["train"][0]),
        n_z=3,
        seed=1501,
        learning_rate=0.003,
        device="cpu",
    )
    rng = random.Random(1501)
    for epoch in range(args.head_epochs):
        order = list(range(len(examples["train"])))
        rng.shuffle(order)
        for batch_index, indices in enumerate(
            _batches(tuple(order), args.head_batch_size)
        ):
            module.process_forward_representation_batch(
                _make_batch(
                    batch_id=f"c3-head-train:e{epoch}:b{batch_index}",
                    examples=examples["train"],
                    contexts=noop_contexts["train"],
                    targets=targets["train"],
                    persistence=persistence["train"],
                    indices=indices,
                    lineage=target_snapshot.lineage,
                ),
                update=True,
            )
    trace_by_split: dict[str, tuple[DialogueSteeringTraceRow, ...]] = {}
    for split in ("train", "validation"):
        rows: list[DialogueSteeringTraceRow] = []
        for index, (example, sample) in enumerate(
            zip(examples[split], samples[split], strict=True)
        ):
            shadow = sample.steering_shadow
            if shadow is None:
                raise RuntimeError("C3 trace construction lacks steering SHADOW data")
            indices = (index,)
            action_snapshot = module.process_forward_representation_batch(
                _make_batch(
                    batch_id=f"c3:{split}:{example.sample_id}:steer",
                    examples=examples[split],
                    contexts=action_contexts[split],
                    targets=targets[split],
                    persistence=persistence[split],
                    indices=indices,
                    lineage=target_snapshot.lineage,
                ),
                update=False,
            )
            noop_snapshot = module.process_forward_representation_batch(
                _make_batch(
                    batch_id=f"c3:{split}:{example.sample_id}:noop",
                    examples=examples[split],
                    contexts=noop_contexts[split],
                    targets=targets[split],
                    persistence=persistence[split],
                    indices=indices,
                    lineage=target_snapshot.lineage,
                ),
                update=False,
            )
            sensor_off_snapshot = module.process_forward_representation_batch(
                _make_batch(
                    batch_id=f"c3:{split}:{example.sample_id}:sensor-off",
                    examples=examples[split],
                    contexts=sensor_off_contexts[split],
                    targets=targets[split],
                    persistence=persistence[split],
                    indices=indices,
                    lineage=target_snapshot.lineage,
                ),
                update=False,
            )
            if (
                action_snapshot.zero_norm_prediction_count
                or noop_snapshot.zero_norm_prediction_count
                or sensor_off_snapshot.zero_norm_prediction_count
            ):
                raise RuntimeError("C3 forward head produced zero-norm predictions")
            episode_id = f"msc:{split}:{example.sample_id}"
            settlement = settle_steering_terminal_prediction_error(
                episode_id=episode_id,
                decision_ids=(shadow.decision_id,),
                action_snapshot=action_snapshot,
                noop_snapshot=noop_snapshot,
            )
            rows.append(
                DialogueSteeringTraceRow(
                    sample_id=example.sample_id,
                    split=split,
                    episode_id=episode_id,
                    cluster_id=example.dyad_id,
                    session_index=example.session_index,
                    observations=shadow.observations,
                    terminal_prediction_error=settlement,
                    reader_artifact_id=shadow.reader_artifact_id,
                    executor_artifact_id=shadow.executor_artifact_id,
                    source_model_id=shadow.source_model_id,
                    source_model_weights_sha256=(
                        shadow.source_model_weights_sha256
                    ),
                    shadow_hook_latency_ms=(
                        sample.interval_steering_hook_latency_ms
                    ),
                    end_to_end_latency_ms=sample.interval_latency_ms,
                    shadow_owner_chain_complete=True,
                    shadow_hook_executed=True,
                    free_bias_present=False,
                    zero_code_strict_noop=True,
                    raw_text_retained=False,
                    evaluation_writeback_allowed=False,
                    sensor_off_executor_artifact_id=(
                        shadow.sensor_off_executor_artifact_id
                    ),
                    control_norm=shadow.control_norm,
                    control_norm_cap=shadow.control_norm_cap,
                    sensor_off_control_norm=shadow.sensor_off_control_norm,
                    sensor_off_mean_squared_error=(
                        sensor_off_snapshot.mean_squared_error
                    ),
                    sensor_off_cosine_similarity=(
                        sensor_off_snapshot.mean_cosine_similarity
                    ),
                )
            )
            if (index + 1) % 100 == 0 or index + 1 == len(examples[split]):
                print(
                    f"[c3-pe] {split} {index + 1}/{len(examples[split])}",
                    flush=True,
                )
        trace_by_split[split] = tuple(rows)
    head_fingerprint = trace_by_split["train"][0].terminal_prediction_error.prediction_head_fingerprint
    dataset = DialogueSteeringTraceDataset(
        schema_version="dialogue-steering-trace-dataset.v1",
        bundle_id=bundle.bundle_id,
        prediction_head_fingerprint=head_fingerprint,
        train_rows=trace_by_split["train"],
        validation_rows=trace_by_split["validation"],
        raw_text_retained=False,
        evaluation_writeback_allowed=False,
        description=(
            "Text-free MSC SHADOW owner observations and PE-owned matched "
            "steer/noop terminal settlements."
        ),
    )
    _write_gzip(trace_path, dataset.to_json())
    del publisher, runtime, module
    gc.collect()
    return dataset


def _completed_formal_result(
    *,
    output: Path,
    preregistration_sha256: str,
    a1: dict[str, object],
    configuration: dict[str, object],
) -> dict[str, object] | None:
    manifest_path = output / "artifact_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "dialogue-steering-formal-manifest.v1"
        or manifest.get("completed") is not True
        or manifest.get("formal_claim_allowed") is not True
        or manifest.get("preregistration_sha256")
        != preregistration_sha256
        or manifest.get("a1_n_plus_one_formal") != a1
        or manifest.get("run_configuration") != configuration
        or manifest.get("raw_text_retained") is not False
        or manifest.get("evaluation_writeback_allowed") is not False
    ):
        raise ValueError("existing C3 formal manifest is invalid or drifted")
    paths = {
        "bundle": output / "steering_artifact_bundle.json",
        "trace": output / "dialogue_steering_trace.json.gz",
        "report": output / "report.json",
    }
    for name, path in paths.items():
        if not path.is_file() or manifest.get(f"{name}_sha256") != _sha256(path):
            raise ValueError(f"existing C3 {name} artifact drift")
    return manifest


def _run_formal(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    configuration = _formal_configuration(args)
    a1 = _a1_attestation(args.seven_day_formal_report.resolve())
    _, prereg_sha = _load_preregistration(
        args.preregistration.resolve(),
        expected_configuration=configuration,
        expected_a1=a1,
    )
    completed = _completed_formal_result(
        output=output,
        preregistration_sha256=prereg_sha,
        a1=a1,
        configuration=configuration,
    )
    if completed is not None:
        print(json.dumps(completed, indent=2))
        return 0
    output_lock = _acquire_output_lock(output)
    with ExitStack() as stack:
        stack.callback(output_lock.close)
        stack.enter_context(exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID))
        require_mps()
        output.mkdir(parents=True, exist_ok=True)
        split_dyads = {}
        for split, limit in (
            ("train", args.train_dyads),
            ("validation", args.validation_dyads),
        ):
            dyads, _ = load_msc_split(args.msc_root, split=split, strict=True)
            split_dyads[split] = _stable_subset(dyads, limit)
        examples = {
            split: build_msc_next_turn_examples(dyads)
            for split, dyads in split_dyads.items()
        }
        if len(examples["validation"]) < DialogueSteeringThresholds().min_real_trace_turns:
            raise ValueError(
                "C3 preregistered validation subset has fewer than 500 turns"
            )
        bundle, fit_passed = _fit_bundle(
            args=args,
            preregistration_sha256=prereg_sha,
            output=output,
            configuration=configuration,
        )
        bundle_path = output / "steering_artifact_bundle.json"
        samples = _collect_contexts(
            args=args,
            output=output,
            bundle=bundle,
            bundle_path=bundle_path,
            split_dyads=split_dyads,
            configuration=configuration,
        )
        dataset = _build_trace_dataset(
            args=args,
            output=output,
            bundle=bundle,
            examples=examples,
            samples=samples,
            configuration=configuration,
        )
        report = run_dialogue_steering_evidence(
            train_rows=dataset.train_rows,
            validation_rows=dataset.validation_rows,
            preregistration_sha256=prereg_sha,
            seed_schedule=tuple(args.seeds),
            policy_restarts=4,
            max_online_episodes=1200,
            eval_every=80,
            learning_rate=0.05,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_confidence=0.95,
            thresholds=DialogueSteeringThresholds(),
            artifact_fit_prerequisite_passed=fit_passed,
        )
        _write_immutable(
            output / "report.json",
            (report.to_json() + "\n").encode("utf-8"),
        )
        manifest = {
            "schema_version": "dialogue-steering-formal-manifest.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed": True,
            "formal_claim_allowed": True,
            "admitted": report.admission.admitted,
            "exit_reason": report.admission.exit_reason,
            "preregistration_path": str(args.preregistration.resolve()),
            "preregistration_sha256": prereg_sha,
            "a1_n_plus_one_formal": a1,
            "run_configuration": configuration,
            "bundle_sha256": _sha256(bundle_path),
            "trace_sha256": _sha256(output / "dialogue_steering_trace.json.gz"),
            "report_sha256": _sha256(output / "report.json"),
            "raw_text_retained": False,
            "evaluation_writeback_allowed": False,
        }
        _write_json(output / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "completed": True,
                "admitted": report.admission.admitted,
                "failed_conditions": report.admission.failed_conditions,
                "exit_reason": report.admission.exit_reason,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


def _preregister(args: argparse.Namespace) -> int:
    configuration = _formal_configuration(args)
    a1 = _a1_attestation(args.seven_day_formal_report.resolve())
    target = args.preregistration.resolve()
    if target.exists():
        _load_preregistration(
            target,
            expected_configuration=configuration,
            expected_a1=a1,
        )
        print(
            json.dumps(
                {"preregistration": str(target), "sha256": _sha256(target)},
                indent=2,
            )
        )
        return 0
    payload = {
        "schema_version": PREREG_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim": (
            "On real MSC SHADOW turns, a gate trained only from terminal N+1 "
            "representation PE credit and owner-published PE proxies beats "
            "matched noop, always-on, and random-gate controls."
        ),
        "a1_n_plus_one_formal": a1,
        "run_configuration": configuration,
        "training_selection": "best restart by train loss only",
        "heldout_unit": "official validation split; dyad-clustered bootstrap",
        "exit_conditions": (
            "insensitive-signal",
            "no-convergence",
            "control-not-beaten",
            "structural-integrity-failure",
        ),
        "forbidden_substitutions": (
            "evaluation-readout-as-credit",
            "judge-score-as-credit",
            "threshold-lowering-after-observation",
            "reader-or-executor-dialogue-update",
        ),
    }
    _write_json(target, payload)
    print(
        json.dumps(
            {
                "preregistration": str(target),
                "sha256": _sha256(target),
            },
            indent=2,
        )
    )
    return 0


def _preflight(args: argparse.Namespace) -> int:
    configuration = _formal_configuration(args)
    a1 = _a1_attestation(args.seven_day_formal_report.resolve())
    counts = {}
    for split, limit in (
        ("train", args.train_dyads),
        ("validation", args.validation_dyads),
    ):
        dyads, audit = load_msc_split(args.msc_root, split=split, strict=True)
        selected = _stable_subset(dyads, limit)
        counts[split] = {
            "dyads": len(selected),
            "turns": len(build_msc_next_turn_examples(selected)),
            "official_verified": audit.verified,
        }
    passed = (
        counts["validation"]["turns"]
        >= DialogueSteeringThresholds().min_real_trace_turns
    )
    result = {
        "schema_version": "dialogue-steering-preflight.v1",
        "passed": passed,
        "a1": a1,
        "counts": counts,
        "run_configuration": configuration,
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


def _status(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    payload = {
        "preregistration_exists": args.preregistration.resolve().is_file(),
        "bundle_exists": (output / "steering_artifact_bundle.json").is_file(),
        "trace_exists": (output / "dialogue_steering_trace.json.gz").is_file(),
        "report_exists": (output / "report.json").is_file(),
        "manifest_exists": (output / "artifact_manifest.json").is_file(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("status", "preflight", "preregister", "formal"))
    parser.add_argument(
        "--msc-root",
        type=Path,
        default=Path("data/external/msc/v0.1/extracted"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--seven-day-formal-report",
        type=Path,
        required=True,
        help=(
            "Path to an audited A1 ablation_results.json or its output root; "
            "the manifest, promotion verdict, and independent audit are required."
        ),
    )
    parser.add_argument(
        "--model-source",
        type=Path,
        default=Path("artifacts/eta_stage2_merged_v2_20260803"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--substrate-device", default="mps")
    parser.add_argument(
        "--substrate-model-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float32",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=C3_DEFAULT_SUBSTRATE_MAX_LENGTH,
    )
    parser.add_argument("--runtime-max-new-tokens", type=int, default=16)
    parser.add_argument("--runtime-startup-timeout", type=float, default=600.0)
    parser.add_argument("--train-dyads", type=int, default=24)
    parser.add_argument("--validation-dyads", type=int, default=24)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument(
        "--accept-noncommercial-license",
        action="store_true",
        required=True,
        help="Acknowledge that the MSC corpus is restricted to research use.",
    )
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "status":
        return _status(args)
    if args.stage == "preflight":
        return _preflight(args)
    if args.stage == "preregister":
        return _preregister(args)
    return _run_formal(args)


if __name__ == "__main__":
    raise SystemExit(main())
