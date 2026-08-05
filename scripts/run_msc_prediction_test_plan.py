#!/usr/bin/env python3
"""Run the preregistered MSC N+1 prediction test plan on Apple MPS.

R3 uses the target-owning frozen substrate for zero-truncation contexts, R4
collects the Volvence arm through the complete service/runtime path, and R5
varies only temporal_n_z while fixing the PE head. Formal dispatch requires a
passed smoke artifact and an immutable preregistration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from companion_test_plan_common import (
    exclusive_mps_lock,
    execution_environment,
    mps_payload,
    print_json,
    require_mps,
    run_plan_command,
)
from freeze_msc_execution_root import MSC_FROZEN_EXECUTION_ROOT_SCHEMA
from freeze_seven_day_execution_root import (
    EXCLUDED_PATTERNS,
    _collect_files,
    _tree_sha256,
)


PLAN_ID = "msc-n-plus-one-prediction-mps.v1"
FORMAL_BLOCKED_EXIT = 3
FORMAL_BLOCKERS: tuple[str, ...] = ()
STAGES = ("status", "preflight", "smoke", "preregister", "formal")
MSC_EXECUTION_SOURCE_ROOTS = (
    "packages/*/src",
    "packages/*/pyproject.toml",
    "pyproject.toml",
    "scripts/companion_test_plan_common.py",
    "scripts/freeze_seven_day_execution_root.py",
    "scripts/freeze_msc_execution_root.py",
    "scripts/msc_prediction_checkpoint.py",
    "scripts/run_msc_prediction_research.py",
    "scripts/run_msc_prediction_test_plan.py",
)
SMOKE_FORMAL_LINEAGE_FIELDS = (
    "msc_root",
    "corpus_provenance_sha256",
    "encoder",
    "context_encoder_mode",
    "volvence_context_mode",
    "device",
    "substrate_model",
    "substrate_device",
    "substrate_activation_width",
    "substrate_layer_indices",
    "substrate_context_limit",
    "substrate_weights_sha256",
    "runtime_max_new_tokens",
    "runtime_startup_timeout",
    "temporal_capacity_n_z",
    "temporal_capacity_fixed_forward_head_n_z",
    "max_seq_length",
    "encoder_batch_size",
    "head_batch_size",
    "learning_rate",
    "retrieval_count",
    "seeds",
    "source_sha256",
)


def _prepend_workspace_sources(execution_root: Path) -> None:
    for source_root in reversed(
        tuple(
            path.resolve()
            for path in sorted((execution_root / "packages").glob("*/src"))
            if path.is_dir()
        )
    ):
        value = str(source_root)
        if value not in sys.path:
            sys.path.insert(0, value)


def _write_immutable_json(path: Path, payload: object) -> None:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    target = path.resolve()
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f"existing preflight report differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _execution_source_snapshot(root: Path) -> dict[str, object]:
    resolved = root.resolve()
    files = _collect_files(resolved, MSC_EXECUTION_SOURCE_ROOTS)
    return {
        "roots": list(MSC_EXECUTION_SOURCE_ROOTS),
        "excluded": list(EXCLUDED_PATTERNS),
        "file_count": len(files),
        "tree_sha256": _tree_sha256(resolved, files),
    }


def _lineage_projection(
    configuration: dict[str, object],
    *,
    label: str,
) -> dict[str, object]:
    missing = tuple(
        field for field in SMOKE_FORMAL_LINEAGE_FIELDS if field not in configuration
    )
    if missing:
        raise ValueError(f"{label} lacks frozen lineage fields: {missing!r}")
    return {
        field: configuration[field] for field in SMOKE_FORMAL_LINEAGE_FIELDS
    }


def _validate_smoke_formal_lineage(
    *,
    smoke: dict[str, object],
    formal_configuration: dict[str, object],
) -> None:
    raw_smoke_configuration = smoke.get("runner_run_configuration")
    if not isinstance(raw_smoke_configuration, dict):
        raise ValueError("MSC smoke lacks runner_run_configuration lineage")
    smoke_lineage = _lineage_projection(
        raw_smoke_configuration,
        label="MSC smoke run configuration",
    )
    formal_lineage = _lineage_projection(
        formal_configuration,
        label="MSC formal run configuration",
    )
    if smoke_lineage != formal_lineage:
        drifted = tuple(
            field
            for field in SMOKE_FORMAL_LINEAGE_FIELDS
            if smoke_lineage[field] != formal_lineage[field]
        )
        raise ValueError(
            "MSC smoke/formal mechanism lineage drift: " f"{drifted!r}"
        )


def _validate_frozen_execution_root(
    *,
    execution_root: Path,
    preregistration: Path,
) -> dict[str, object]:
    root = execution_root.resolve()
    preregistration_path = preregistration.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MSC frozen execution root is missing: {root}")
    if not preregistration_path.is_file():
        raise FileNotFoundError(
            f"MSC formal preregistration is missing: {preregistration_path}"
        )
    preregistration_payload = _load_json_object(
        preregistration_path,
        label="MSC formal preregistration",
    )
    expected_snapshot = preregistration_payload.get("execution_source_snapshot")
    if not isinstance(expected_snapshot, dict):
        raise ValueError("MSC preregistration lacks execution_source_snapshot")
    if expected_snapshot.get("roots") != list(MSC_EXECUTION_SOURCE_ROOTS):
        raise ValueError("MSC preregistered execution source roots drift")
    if expected_snapshot.get("excluded") != list(EXCLUDED_PATTERNS):
        raise ValueError("MSC preregistered execution exclusions drift")

    manifest_path = root / "frozen_execution_root_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(
            f"MSC frozen execution manifest is missing: {manifest_path}"
        )
    manifest = _load_json_object(
        manifest_path,
        label="MSC frozen execution manifest",
    )
    actual_snapshot = _execution_source_snapshot(root)
    expected_preregistration_sha256 = _sha256_file(preregistration_path)
    expected_manifest_fields = {
        "schema_version": MSC_FROZEN_EXECUTION_ROOT_SCHEMA,
        "preregistration_sha256": expected_preregistration_sha256,
        "source_tree_sha256": expected_snapshot.get("tree_sha256"),
        "file_count": expected_snapshot.get("file_count"),
        "excluded": list(EXCLUDED_PATTERNS),
        "read_only": True,
    }
    drifted_manifest_fields = tuple(
        field
        for field, expected in expected_manifest_fields.items()
        if manifest.get(field) != expected
    )
    if drifted_manifest_fields:
        raise ValueError(
            "MSC frozen execution manifest drift: "
            f"{drifted_manifest_fields!r}"
        )
    if actual_snapshot != expected_snapshot:
        raise ValueError("MSC frozen execution source tree differs from preregistration")

    source_files = _collect_files(root, MSC_EXECUTION_SOURCE_ROOTS)
    expected_entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in source_files
    ]
    if manifest.get("files") != expected_entries:
        raise ValueError("MSC frozen execution file manifest differs from source tree")
    permitted_files = {
        *(entry["path"] for entry in expected_entries),
        "frozen_execution_root_manifest.json",
    }
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"MSC frozen execution root contains symlink: {path}")
        if path.stat().st_mode & 0o222:
            raise ValueError(f"MSC frozen execution root is writable: {path}")
        if path.is_file() and path.relative_to(root).as_posix() not in permitted_files:
            raise ValueError(f"MSC frozen execution root contains extra file: {path}")
    return manifest


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _write_smoke_manifest(output_dir: Path) -> dict[str, object]:
    output = output_dir.resolve()
    artifact_path = output / "artifact_manifest.json"
    manifest_path = output / "manifest.json"
    temporal_path = output / "temporal_capacity_ladder.json"
    prediction_path = output / "prediction_verdict.json"
    artifact = _load_json_object(artifact_path, label="MSC smoke artifact manifest")
    manifest = _load_json_object(manifest_path, label="MSC smoke manifest")
    temporal = _load_json_object(
        temporal_path,
        label="MSC smoke temporal capacity ladder",
    )
    prediction = _load_json_object(
        prediction_path,
        label="MSC smoke prediction verdict",
    )
    same_substrate = manifest.get("same_substrate_context_attestation")
    full_runtime = manifest.get("full_runtime_context_attestation")
    run_configuration = manifest.get("run_configuration")
    runtime_attestations = manifest.get("temporal_capacity_runtime_attestations")
    observations = temporal.get("observations")
    if not isinstance(runtime_attestations, dict) or not isinstance(
        observations, list
    ):
        raise ValueError("MSC smoke lacks R5 runtime/observation evidence")
    if not isinstance(run_configuration, dict):
        raise ValueError("MSC smoke lacks runner run_configuration")
    temporal_values = {
        row.get("temporal_n_z")
        for row in observations
        if isinstance(row, dict)
    }
    fixed_head_values = {
        row.get("forward_head_n_z")
        for row in observations
        if isinstance(row, dict)
    }
    checks = {
        "runner_completed_pilot": (
            artifact.get("evidence_level") == "pilot"
            and artifact.get("formal_experiment_executed") is False
        ),
        "r3_same_substrate_zero_truncation": (
            isinstance(same_substrate, dict)
            and same_substrate.get("passed") is True
            and same_substrate.get("truncated_token_count") == 0
        ),
        "r4_full_runtime_privacy": (
            isinstance(full_runtime, dict)
            and full_runtime.get("volvence_full_stack") is True
            and full_runtime.get("raw_text_retained") is False
            and full_runtime.get("evaluation_writeback_allowed") is False
        ),
        "r5_all_temporal_capacities": temporal_values == {3, 16, 64, 256},
        "r5_fixed_pe_head": fixed_head_values == {3},
        "r5_zero_norm_count_exposed": isinstance(
            temporal.get("zero_norm_prediction_count"), int
        ),
        "r3_r4_r5_executed_in_order": manifest.get(
            "convergence_stage_order"
        )
        == ["R3", "R4", "R5"],
        "formal_remains_fail_closed_in_smoke": (
            prediction.get("evidence_level") == "pilot"
        ),
    }
    latency_values = []
    token_values = []
    sample_values = []
    for raw in runtime_attestations.values():
        if not isinstance(raw, dict):
            raise ValueError("MSC smoke runtime attestation is not an object")
        latency_values.append(float(raw["total_interval_latency_ms"]))
        token_values.append(int(raw["total_interval_token_count"]))
        sample_values.append(int(raw["sample_count"]))
    payload: dict[str, object] = {
        "schema_version": "msc-r5-smoke-manifest.v1",
        "plan_id": PLAN_ID,
        "passed": bool(checks) and all(checks.values()),
        "formal_claim_permitted": False,
        "checks": checks,
        "runner_artifact_manifest": {
            "path": str(artifact_path),
            "sha256": _sha256_file(artifact_path),
        },
        "runner_run_configuration": run_configuration,
        "measurement": {
            "temporal_capacity_count": len(runtime_attestations),
            "runtime_sample_count": sum(sample_values),
            "total_interval_token_count": sum(token_values),
            "total_interval_latency_ms": sum(latency_values),
        },
    }
    _write_immutable_json(output / "smoke_manifest.json", payload)
    if payload["passed"] is not True:
        raise ValueError("MSC R5 smoke checks did not all pass")
    return payload


def _run_progress(output_dir: Path | None) -> dict[str, object] | None:
    if output_dir is None:
        return None
    output = output_dir.resolve()
    state_path = output / "run_state.json"
    if not output.exists():
        return {
            "output_dir": os.fspath(output),
            "status": "not-started",
            "completed_unit_count": 0,
            "analysis_allowed": False,
            "formal_claim_allowed": False,
        }
    if not state_path.is_file():
        raise FileNotFoundError(
            f"prediction output exists without run_state.json: {output}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("prediction run_state root must be a JSON object")
    if state.get("schema_version") != "msc-prediction-run-state.v1":
        raise ValueError("prediction run_state schema_version is unsupported")
    units = state.get("completed_units")
    if not isinstance(units, dict):
        raise ValueError("prediction run_state completed_units must be an object")
    status = state.get("status")
    analysis_allowed = state.get("analysis_allowed")
    formal_claim_allowed = state.get("formal_claim_allowed")
    raw_corpus_text_retained = state.get("raw_corpus_text_retained")
    if status not in {"running", "complete"}:
        raise ValueError("prediction run_state status is invalid")
    if not isinstance(analysis_allowed, bool):
        raise ValueError("prediction run_state analysis_allowed must be boolean")
    if not isinstance(formal_claim_allowed, bool):
        raise ValueError("prediction run_state formal_claim_allowed must be boolean")
    if raw_corpus_text_retained is not False:
        raise ValueError("prediction checkpoint must not retain raw corpus text")
    return {
        "output_dir": os.fspath(output),
        "status": status,
        "completed_unit_count": len(units),
        "last_completed_unit": state.get("last_completed_unit"),
        "configuration_fingerprint": state.get("configuration_fingerprint"),
        "analysis_allowed": analysis_allowed,
        "formal_claim_allowed": formal_claim_allowed,
        "raw_corpus_text_retained": raw_corpus_text_retained,
    }


def _prediction_status(
    *,
    output_dir: Path | None = None,
    preregistration: Path | None = None,
) -> dict[str, object]:
    preregistered = bool(
        preregistration is not None and preregistration.resolve().is_file()
    )
    status: dict[str, object] = {
        "plan_id": PLAN_ID,
        "formal_eligible": True,
        "formal_blocked_exit": FORMAL_BLOCKED_EXIT,
        "completed_prerequisites": (
            "official-msc-v0.1-corpus",
            "substrate-owned-n-plus-one-target",
            "same-substrate-zero-truncation-context",
            "complete-volvence-runtime-arm",
            "forward-head-capacity-fields-separated-from-temporal-capacity",
            "temporal-controller-capacity-ladder",
        ),
        "formal_blockers": FORMAL_BLOCKERS,
        "permitted_now": ("preflight", "smoke", "preregister", "formal"),
        "formal_preregistered": preregistered,
        "formal_claim_permitted_now": preregistered,
    }
    progress = _run_progress(output_dir)
    if progress is not None:
        status["run_progress"] = progress
    return status


def _preflight(
    *,
    execution_root: Path,
    msc_root: Path,
    substrate_model: str,
    mps: object,
) -> dict[str, object]:
    _prepend_workspace_sources(execution_root)
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "prediction preflight requires huggingface-hub and transformers"
        ) from exc
    from companion_bench.msc_corpus import load_msc_split
    from companion_bench.prediction_research import (
        build_msc_next_turn_examples,
        render_long_context,
    )
    from volvence_zero.substrate import fingerprint_model_weight_files

    corpus = msc_root.resolve()
    provenance_path = corpus.parent / "DOWNLOAD_PROVENANCE.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"MSC DOWNLOAD_PROVENANCE.json is missing next to {corpus}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("MSC provenance root must be a JSON object")
    if provenance.get("schema_version") != "msc-download-provenance.v1":
        raise ValueError("MSC provenance schema_version is unsupported")

    snapshot = Path(snapshot_download(substrate_model, local_files_only=True))
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    context_limit = int(config.max_position_embeddings)
    split_audits: dict[str, object] = {}
    observed_max = 0
    for split in ("train", "validation", "heldout"):
        dyads, audit = load_msc_split(corpus, split=split, strict=True)
        examples = build_msc_next_turn_examples(dyads)
        token_counts = tuple(
            len(
                tokenizer.encode(
                    render_long_context(example),
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for example in examples
        )
        if not token_counts:
            raise ValueError(f"MSC {split} produced no N+1 examples")
        maximum = max(token_counts)
        observed_max = max(observed_max, maximum)
        split_audits[split] = {
            "dyad_count": len(dyads),
            "example_count": len(examples),
            "max_full_history_tokens": maximum,
            "mean_full_history_tokens": sum(token_counts) / len(token_counts),
            "over_model_context_limit": sum(
                count > context_limit for count in token_counts
            ),
            "file_sha256": audit.file_sha256,
            "sorted_id_sha256": audit.sorted_id_sha256,
        }
    full_history_fits = observed_max <= context_limit
    if not full_history_fits:
        raise ValueError(
            "MSC full history exceeds the frozen substrate context limit; "
            "a recency-truncation preregistration is required"
        )
    return {
        "schema_version": "msc-n-plus-one-mps-preflight.v1",
        "plan": _prediction_status(),
        "mps": mps_payload(mps),
        "corpus": {
            "root": os.fspath(corpus),
            "license_policy": "noncommercial-research-only",
            "provenance_schema_version": provenance["schema_version"],
            "splits": split_audits,
        },
        "substrate": {
            "model_id": substrate_model,
            "snapshot_revision": snapshot.name,
            "weights_sha256": fingerprint_model_weight_files(snapshot),
            "declared_context_limit": context_limit,
            "observed_max_full_history_tokens": observed_max,
            "full_history_fits_without_truncation": full_history_fits,
            "128k_arm_adds_distinct_msc_exposure": observed_max > context_limit,
        },
        "claim_boundary": (
            "Preflight proves local data/model/device readiness and the R3 full-history "
            "fit condition only. Passed R5 smoke, a new preregistration, and the "
            "complete R4/R5 formal execution remain required for adjudication."
        ),
    }


def _smoke_command(
    *,
    python: Path,
    execution_root: Path,
    msc_root: Path,
    output_dir: Path,
    substrate_model: str,
    resume: bool,
) -> tuple[str, ...]:
    runner = execution_root / "scripts/run_msc_prediction_research.py"
    if not runner.is_file():
        raise FileNotFoundError(f"MSC mechanism runner does not exist: {runner}")
    argv = (
        str(python),
        str(runner),
        "--msc-root",
        str(msc_root),
        "--output",
        str(output_dir),
        "--accept-noncommercial-license",
        "--device",
        "mps",
        "--substrate-model",
        substrate_model,
        "--substrate-device",
        "mps",
        "--volvence-context-mode",
        "full-runtime",
        "--substrate-layer-indices",
        "11",
        "12",
        "13",
        "--runtime-max-new-tokens",
        "16",
        "--train-dyads",
        "2",
        "--validation-dyads",
        "1",
        "--heldout-dyads",
        "1",
        "--epochs",
        "2",
        "--seeds",
        "0",
        "1",
        "2",
    )
    return (*argv, *(("--resume",) if resume else ()))


def _formal_command(
    *,
    python: Path,
    execution_root: Path,
    msc_root: Path,
    output_dir: Path,
    substrate_model: str,
    preregistration: Path,
    resume: bool,
    emit_run_configuration: Path | None = None,
) -> tuple[str, ...]:
    runner = execution_root / "scripts/run_msc_prediction_research.py"
    if not runner.is_file():
        raise FileNotFoundError(f"MSC formal runner does not exist: {runner}")
    argv = (
        str(python),
        str(runner),
        "--msc-root",
        str(msc_root),
        "--output",
        str(output_dir),
        "--accept-noncommercial-license",
        "--device",
        "mps",
        "--substrate-model",
        substrate_model,
        "--substrate-device",
        "mps",
        "--context-encoder-mode",
        "substrate",
        "--volvence-context-mode",
        "full-runtime",
        "--substrate-layer-indices",
        "11",
        "12",
        "13",
        "--runtime-max-new-tokens",
        "16",
        "--train-dyads",
        "1001",
        "--validation-dyads",
        "500",
        "--heldout-dyads",
        "501",
        "--epochs",
        "8",
        "--seeds",
        "0",
        "1",
        "2",
        *(("--preregistration", str(preregistration)) if emit_run_configuration is None else ()),
        *(
            ("--emit-run-configuration", str(emit_run_configuration))
            if emit_run_configuration is not None
            else ()
        ),
        *(("--resume",) if resume else ()),
    )
    return argv


def _write_formal_preregistration(
    *,
    python: Path,
    execution_root: Path,
    environment: dict[str, str],
    msc_root: Path,
    formal_output_dir: Path,
    smoke_output_dir: Path,
    substrate_model: str,
    preregistration: Path,
) -> dict[str, object]:
    smoke_path = smoke_output_dir.resolve() / "smoke_manifest.json"
    smoke = _load_json_object(smoke_path, label="MSC R5 smoke manifest")
    if (
        smoke.get("schema_version") != "msc-r5-smoke-manifest.v1"
        or smoke.get("passed") is not True
        or smoke.get("formal_claim_permitted") is not False
    ):
        raise ValueError("MSC formal preregistration requires a passed smoke")
    target = preregistration.resolve()
    config_path = target.with_name(f"{target.stem}.run_configuration.json")
    with tempfile.TemporaryDirectory(
        prefix="volvence-msc-prereg-configuration-"
    ) as temporary_directory:
        captured_config_path = (
            Path(temporary_directory) / "run_configuration.json"
        )
        return_code = run_plan_command(
            _formal_command(
                python=python,
                execution_root=execution_root,
                msc_root=msc_root,
                output_dir=formal_output_dir,
                substrate_model=substrate_model,
                preregistration=target,
                resume=False,
                emit_run_configuration=captured_config_path,
            ),
            execution_root=execution_root,
            environment=environment,
        )
        if return_code != 0:
            raise RuntimeError("MSC formal run-configuration capture failed")
        run_configuration = _load_json_object(
            captured_config_path,
            label="MSC formal run configuration",
        )
    _validate_smoke_formal_lineage(
        smoke=smoke,
        formal_configuration=run_configuration,
    )
    _write_immutable_json(config_path, run_configuration)
    frozen_payload: dict[str, object] = {
        "schema_version": "msc-n-plus-one-formal-prereg.v1",
        "claim_scope": "msc-n-plus-one-pe-eta-load-bearing",
        "license_policy": "noncommercial-research-only",
        "run_configuration": run_configuration,
        "execution_source_snapshot": _execution_source_snapshot(execution_root),
        "formal_matrix": {
            "train_dyads": 1001,
            "validation_dyads": 500,
            "heldout_dyads": 501,
            "seeds": [0, 1, 2],
        },
        "temporal_capacity_intervention": {
            "temporal_n_z": [3, 16, 64, 256],
            "fixed_forward_head_n_z": 3,
            "all_other_factors_fixed": True,
            "execution_order": ["R3", "R4", "R5"],
        },
        "primary_test": {
            "comparison": "volvence-minus-long_context",
            "session_index": 5,
            "quality_min_cosine_advantage": 0.02,
            "quality_ci_lower_bound_strictly_positive": True,
            "quality_advantage_slope_strictly_positive": True,
            "scaling_min_cosine_equivalence": -0.01,
            "scaling_max_token_ratio": 0.10,
            "scaling_max_latency_ratio": 0.50,
            "stateless_and_summary_are_eligibility_only": True,
        },
        "passed_smoke_artifact": {
            "path": str(smoke_path),
            "sha256": _sha256_file(smoke_path),
            "measurement": smoke.get("measurement"),
        },
        "exit_conditions": {
            "quality_or_scaling_gate_failure": "REJECT_AND_SIMPLIFY",
            "zero_norm_prediction": "FAIL_ZERO_NORM_PREDICTIONS",
            "no_post_hoc_threshold_changes": True,
        },
    }
    if target.exists():
        existing = _load_json_object(
            target,
            label="MSC formal preregistration",
        )
        if (
            not isinstance(existing.get("created_at_utc"), str)
            or any(existing.get(key) != value for key, value in frozen_payload.items())
            or set(existing) != {*frozen_payload, "created_at_utc"}
        ):
            raise ValueError(f"MSC formal preregistration is immutable: {target}")
        return existing
    payload = {
        **frozen_payload,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_immutable_json(target, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--msc-root",
        type=Path,
        default=Path("data/external/msc/v0.1/extracted"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoke-output-dir",
        type=Path,
        help="Completed smoke root used to freeze the formal preregistration.",
    )
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument(
        "--substrate-model", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "status":
        print_json(
            _prediction_status(
                output_dir=args.output_dir,
                preregistration=args.preregistration,
            )
        )
        return 0

    execution_root = args.execution_root.resolve()
    msc_root = (
        args.msc_root.resolve()
        if args.msc_root.is_absolute()
        else (execution_root / args.msc_root).resolve()
    )
    environment = execution_environment(execution_root)
    if args.stage == "formal":
        if args.preregistration is None:
            raise ValueError("prediction formal requires --preregistration")
        _validate_frozen_execution_root(
            execution_root=execution_root,
            preregistration=args.preregistration.resolve(),
        )
    if args.stage == "preregister":
        if (
            args.output_dir is None
            or args.smoke_output_dir is None
            or args.preregistration is None
        ):
            raise ValueError(
                "prediction preregister requires --output-dir, "
                "--smoke-output-dir, and --preregistration"
            )
        payload = _write_formal_preregistration(
            python=args.python.resolve(),
            execution_root=execution_root,
            environment=environment,
            msc_root=msc_root,
            formal_output_dir=args.output_dir.resolve(),
            smoke_output_dir=args.smoke_output_dir.resolve(),
            substrate_model=args.substrate_model,
            preregistration=args.preregistration.resolve(),
        )
        print_json(
            {
                "plan_id": PLAN_ID,
                "stage": "preregister",
                "preregistration": str(args.preregistration.resolve()),
                "schema_version": payload["schema_version"],
                "sha256": _sha256_file(args.preregistration.resolve()),
                "formal_claim_permitted": False,
            }
        )
        return 0

    with exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID):
        mps = require_mps()
        environment["VZ_COMPANION_MPS_LOCK_HELD"] = "1"
        environment["VZ_COMPANION_MPS_LOCK_PATH"] = str(
            args.mps_lock.resolve()
        )
        if args.stage == "preflight":
            if args.preflight_report is None:
                raise ValueError("prediction preflight requires --preflight-report")
            report = _preflight(
                execution_root=execution_root,
                msc_root=msc_root,
                substrate_model=args.substrate_model,
                mps=mps,
            )
            _write_immutable_json(args.preflight_report, report)
            print_json(report)
            return 0
        if args.output_dir is None:
            raise ValueError(f"prediction {args.stage} requires --output-dir")
        if args.stage == "smoke":
            command = _smoke_command(
                python=args.python.resolve(),
                execution_root=execution_root,
                msc_root=msc_root,
                output_dir=args.output_dir.resolve(),
                substrate_model=args.substrate_model,
                resume=args.resume,
            )
        else:
            if args.preregistration is None:
                raise ValueError(
                    "prediction formal requires --preregistration"
                )
            command = _formal_command(
                python=args.python.resolve(),
                execution_root=execution_root,
                msc_root=msc_root,
                output_dir=args.output_dir.resolve(),
                substrate_model=args.substrate_model,
                preregistration=args.preregistration.resolve(),
                resume=args.resume,
            )
        return_code = run_plan_command(
            command,
            execution_root=execution_root,
            environment=environment,
        )
        if return_code != 0:
            return return_code
        smoke_manifest = None
        if args.stage == "smoke":
            smoke_manifest = _write_smoke_manifest(args.output_dir.resolve())
        else:
            progress = _run_progress(args.output_dir.resolve())
            if (
                progress is None
                or progress.get("status") != "complete"
                or progress.get("formal_claim_allowed") is not True
            ):
                raise ValueError(
                    "MSC formal runner completed without formal claim authorization"
                )
    print_json(
        {
            "plan_id": PLAN_ID,
            "stage": args.stage,
            "evidence_level": (
                "mechanism-only-pilot" if args.stage == "smoke" else "formal"
            ),
            "formal_claim_permitted": args.stage == "formal",
            "smoke_passed": (
                smoke_manifest.get("passed")
                if smoke_manifest is not None
                else None
            ),
            "mps": mps_payload(mps),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
