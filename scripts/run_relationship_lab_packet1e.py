#!/usr/bin/env python3
"""Run the frozen Relationship Lab P1e v2 consumer qualification."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _relative in (
    "packages/lifeform-domain-emogpt/src",
    "packages/lifeform-evolution/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _relative))

from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

from lifeform_domain_emogpt.lab import relationship_transfer_package_dir  # noqa: E402
from lifeform_evolution.relationship_lab_gate0 import (  # noqa: E402
    Gate0CalibrationConfig,
    load_frozen_baseline_attestation,
    run_relationship_gate0_calibration,
)
from lifeform_evolution.relationship_lab_packet1 import (  # noqa: E402
    RelationshipP1GateConfig,
)
from lifeform_evolution.relationship_lab_packet1b import (  # noqa: E402
    load_relationship_packet1b_report,
)
from lifeform_evolution.relationship_lab_packet1e import (  # noqa: E402
    RelationshipP1eVerdict,
    assess_relationship_packet1e,
    load_relationship_p1e_consumer_protocol,
    load_relationship_packet1e_report,
    relationship_p1e_consumer_protocol_path,
    validate_relationship_p1e_local_lineage,
    write_relationship_packet1e_report,
)


_CHECKPOINT_SCHEMA_VERSION = "relationship-p1e-checkpoint.v1"
_CHECKPOINT_STAGES = {
    "initialized",
    "gate0_running",
    "gate0_complete",
    "p1b_running",
    "p1b_complete",
    "complete",
}
_CHECKPOINT_ARTIFACT_KEYS = {
    "active_gate0_dir",
    "active_p1b_dir",
    "baseline_attestation",
    "gate0_report",
    "p1b_report",
    "p1e_report",
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(relationship_p1e_consumer_protocol_path()),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            _REPO_ROOT
            / "artifacts"
            / "relationship_lab"
            / f"qwen25_3b_packet1e_v2_{int(time.time())}"
        ),
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow materializing missing frozen snapshots from Hugging Face.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen lineage, disk, and local snapshot availability.",
    )
    return parser.parse_args(argv)


def _snapshot_size(path: pathlib.Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _materialize_snapshot(
    *,
    repo_id: str,
    revision: str | None,
    allow_download: bool,
    minimum_free_bytes_before_download: int,
) -> tuple[pathlib.Path | None, bool]:
    try:
        cached = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        if not allow_download:
            return None, False
        free_bytes = shutil.disk_usage(_REPO_ROOT).free
        if free_bytes < minimum_free_bytes_before_download:
            raise OSError(
                "P1e refuses snapshot download: "
                f"free_bytes={free_bytes} < required={minimum_free_bytes_before_download}"
            ) from None
        downloaded = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=False,
        )
        return pathlib.Path(downloaded), True
    return pathlib.Path(cached), False


def _preflight(protocol, *, allow_download: bool) -> tuple[dict[str, object], bool]:
    free_before = shutil.disk_usage(_REPO_ROOT).free
    candidate_path, candidate_downloaded = _materialize_snapshot(
        repo_id=protocol.model_source,
        revision=protocol.model_revision,
        allow_download=allow_download,
        minimum_free_bytes_before_download=(
            protocol.minimum_free_bytes_before_download
        ),
    )
    rag_path, rag_downloaded = _materialize_snapshot(
        repo_id=protocol.rag_model_source,
        revision=None,
        allow_download=allow_download,
        minimum_free_bytes_before_download=(
            protocol.minimum_free_bytes_before_download
        ),
    )
    candidate_size = (
        _snapshot_size(candidate_path) if candidate_path is not None else None
    )
    if (
        candidate_size is not None
        and candidate_size > protocol.maximum_candidate_snapshot_bytes
    ):
        raise OSError(
            "P1e candidate snapshot exceeds frozen guard: "
            f"size={candidate_size} > maximum={protocol.maximum_candidate_snapshot_bytes}"
        )
    ready = candidate_path is not None and rag_path is not None
    return (
        {
            "protocol_id": protocol.protocol_id,
            "package_name": protocol.package_name,
            "candidate": {
                "repo_id": protocol.model_source,
                "available": candidate_path is not None,
                "downloaded": candidate_downloaded,
                "snapshot_path": (
                    str(candidate_path) if candidate_path is not None else None
                ),
                "snapshot_bytes": candidate_size,
                "maximum_snapshot_bytes": (
                    protocol.maximum_candidate_snapshot_bytes
                ),
            },
            "rag": {
                "repo_id": protocol.rag_model_source,
                "available": rag_path is not None,
                "downloaded": rag_downloaded,
                "snapshot_path": str(rag_path) if rag_path is not None else None,
                "snapshot_bytes": (
                    _snapshot_size(rag_path) if rag_path is not None else None
                ),
                "top_k": protocol.rag_top_k,
                "candidate_surface": protocol.rag_candidate_surface,
            },
            "free_bytes_before": free_before,
            "free_bytes_after": shutil.disk_usage(_REPO_ROOT).free,
            "ready": ready,
        },
        ready,
    )


def _checkpoint_path(root: pathlib.Path) -> pathlib.Path:
    return root / "checkpoint.json"


def _atomic_write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=".p1e-checkpoint-",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = pathlib.Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except OSError:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def _write_checkpoint(
    root: pathlib.Path,
    *,
    protocol_id: str,
    stage: str,
    artifacts: dict[str, str],
) -> None:
    if stage not in _CHECKPOINT_STAGES:
        raise ValueError(f"unsupported P1e checkpoint stage: {stage}")
    if not set(artifacts).issubset(_CHECKPOINT_ARTIFACT_KEYS):
        raise ValueError("P1e checkpoint contains an unknown artifact key")
    _atomic_write_json(
        _checkpoint_path(root),
        {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "protocol_id": protocol_id,
            "stage": stage,
            "artifacts": dict(sorted(artifacts.items())),
            "updated_at_iso": datetime.now(timezone.utc).isoformat(),
        },
    )


def _load_or_initialize_checkpoint(
    root: pathlib.Path,
    *,
    protocol_id: str,
) -> tuple[str, dict[str, str]]:
    checkpoint_path = _checkpoint_path(root)
    if not root.exists():
        root.mkdir(parents=True)
        _write_checkpoint(
            root,
            protocol_id=protocol_id,
            stage="initialized",
            artifacts={},
        )
        return "initialized", {}
    if not checkpoint_path.is_file():
        raise FileExistsError(f"P1e output directory has no checkpoint: {root}")
    raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "artifacts",
        "protocol_id",
        "schema_version",
        "stage",
        "updated_at_iso",
    }:
        raise ValueError("P1e checkpoint fields do not match schema")
    if raw["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("P1e checkpoint schema_version mismatch")
    if raw["protocol_id"] != protocol_id:
        raise ValueError("P1e checkpoint belongs to another protocol")
    if raw["stage"] not in _CHECKPOINT_STAGES:
        raise ValueError("P1e checkpoint stage is invalid")
    artifacts = raw["artifacts"]
    if (
        not isinstance(artifacts, dict)
        or not set(artifacts).issubset(_CHECKPOINT_ARTIFACT_KEYS)
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not value
            for key, value in artifacts.items()
        )
    ):
        raise ValueError("P1e checkpoint artifacts are invalid")
    return raw["stage"], dict(artifacts)


def _relative_path(root: pathlib.Path, path: pathlib.Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _artifact_path(
    root: pathlib.Path,
    artifacts: dict[str, str],
    key: str,
) -> pathlib.Path | None:
    relative = artifacts.get(key)
    if relative is None:
        return None
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"P1e checkpoint artifact escapes output root: {key}")
    return candidate


def _next_attempt_dir(root: pathlib.Path, prefix: str) -> pathlib.Path:
    first = root / prefix
    if not first.exists():
        return first
    attempt = 2
    while True:
        candidate = root / f"{prefix}_attempt_{attempt}"
        if not candidate.exists():
            return candidate
        attempt += 1


def _run_child(command: list[str], *, accepted_return_codes: set[int]) -> int:
    completed = subprocess.run(command, cwd=str(_REPO_ROOT), check=False)
    if completed.returncode not in accepted_return_codes:
        raise RuntimeError(
            f"P1e child failed with exit code {completed.returncode}: {command}"
        )
    return completed.returncode


def _load_or_run_gate0(
    *,
    root: pathlib.Path,
    protocol,
    artifacts: dict[str, str],
):
    attestation_path = _artifact_path(root, artifacts, "baseline_attestation")
    if attestation_path is None:
        active_dir = _artifact_path(root, artifacts, "active_gate0_dir")
        recovered = (
            active_dir / "baseline_attestation.json"
            if active_dir is not None
            else None
        )
        if recovered is not None and recovered.is_file():
            attestation_path = recovered
            artifacts["baseline_attestation"] = _relative_path(root, recovered)
    if attestation_path is None or not attestation_path.is_file():
        attempt_dir = _next_attempt_dir(root, "gate0_candidate")
        artifacts["active_gate0_dir"] = _relative_path(root, attempt_dir)
        _write_checkpoint(
            root,
            protocol_id=protocol.protocol_id,
            stage="gate0_running",
            artifacts=artifacts,
        )
        command = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_relationship_lab_stateless_baseline.py"),
            "--package-name",
            protocol.package_name,
            "--model-source",
            protocol.model_source,
            "--model-id",
            protocol.model_id,
            "--device",
            protocol.device,
            "--torch-dtype",
            protocol.torch_dtype,
            "--temperature",
            str(protocol.temperature),
            "--top-p",
            str(protocol.top_p),
            "--max-new-tokens",
            str(protocol.max_new_tokens),
            "--seeds",
            ",".join(str(item) for item in protocol.baseline_seed_schedule),
            "--output-dir",
            str(attempt_dir),
        ]
        _run_child(command, accepted_return_codes={0, 2})
        attestation_path = attempt_dir / "baseline_attestation.json"
        if not attestation_path.is_file():
            raise FileNotFoundError("P1e Gate 0 child published no attestation")
        artifacts["baseline_attestation"] = _relative_path(
            root, attestation_path
        )
    baseline = load_frozen_baseline_attestation(attestation_path)
    gate0_report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(),
        baseline=baseline,
        package_root=relationship_transfer_package_dir(protocol.package_name),
        created_at_iso=baseline.frozen_at_iso,
    )
    gate0_path = root / "candidate_gate0_report.json"
    encoded = gate0_report.to_json()
    if gate0_path.exists():
        if gate0_path.read_text(encoding="utf-8") != encoded:
            raise ValueError("P1e candidate Gate 0 report was modified")
    else:
        gate0_path.write_text(encoded, encoding="utf-8")
    artifacts["gate0_report"] = _relative_path(root, gate0_path)
    _write_checkpoint(
        root,
        protocol_id=protocol.protocol_id,
        stage="gate0_complete",
        artifacts=artifacts,
    )
    return baseline, gate0_report, attestation_path


def _load_or_run_p1b(
    *,
    root: pathlib.Path,
    protocol,
    baseline_attestation_path: pathlib.Path,
    artifacts: dict[str, str],
):
    report_path = _artifact_path(root, artifacts, "p1b_report")
    if report_path is None:
        active_dir = _artifact_path(root, artifacts, "active_p1b_dir")
        recovered = (
            active_dir / "packet1b_report.json"
            if active_dir is not None
            else None
        )
        if recovered is not None and recovered.is_file():
            report_path = recovered
            artifacts["p1b_report"] = _relative_path(root, recovered)
    if report_path is not None and report_path.is_file():
        report = load_relationship_packet1b_report(report_path)
        _write_checkpoint(
            root,
            protocol_id=protocol.protocol_id,
            stage="p1b_complete",
            artifacts=artifacts,
        )
        return report
    attempt_dir = _next_attempt_dir(root, "p1b_candidate")
    artifacts["active_p1b_dir"] = _relative_path(root, attempt_dir)
    _write_checkpoint(
        root,
        protocol_id=protocol.protocol_id,
        stage="p1b_running",
        artifacts=artifacts,
    )
    gate = RelationshipP1GateConfig()
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "run_relationship_lab_packet1.py"),
        "run-p1b",
        "--package-name",
        protocol.package_name,
        "--readout-profile",
        protocol.readout_profile,
        "--model-source",
        protocol.model_source,
        "--model-id",
        protocol.model_id,
        "--device",
        protocol.device,
        "--torch-dtype",
        protocol.torch_dtype,
        "--temperature",
        str(protocol.temperature),
        "--top-p",
        str(protocol.top_p),
        "--max-new-tokens",
        str(protocol.max_new_tokens),
        "--seeds",
        ",".join(str(item) for item in protocol.p1b_seed_schedule),
        "--background-depths",
        ",".join(str(item) for item in protocol.background_depths),
        "--rag-embedder",
        protocol.rag_embedder,
        "--rag-model-source",
        protocol.rag_model_source,
        "--rag-device",
        "cpu",
        "--rag-top-k",
        str(protocol.rag_top_k),
        "--rag-candidate-surface",
        protocol.rag_candidate_surface,
        "--gate0-baseline-attestation",
        str(baseline_attestation_path),
        "--minimum-decisions-per-arm",
        str(gate.minimum_decisions_per_arm),
        "--minimum-steelman-accuracy",
        str(gate.minimum_steelman_accuracy),
        "--maximum-steelman-accuracy",
        str(gate.maximum_steelman_accuracy),
        "--minimum-steelman-pair-flip-rate",
        str(gate.minimum_steelman_pair_flip_rate),
        "--minimum-structured-state-pair-flip-rate",
        str(gate.minimum_structured_state_pair_flip_rate),
        "--maximum-rag-to-full-history-token-ratio",
        str(gate.maximum_rag_to_full_history_token_ratio),
        "--maximum-structured-to-full-history-token-ratio",
        str(gate.maximum_structured_to_full_history_token_ratio),
        "--output-dir",
        str(attempt_dir),
    ]
    _run_child(command, accepted_return_codes={0, 2})
    report_path = attempt_dir / "packet1b_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("P1e P1b child published no report")
    report = load_relationship_packet1b_report(report_path)
    artifacts["p1b_report"] = _relative_path(root, report_path)
    _write_checkpoint(
        root,
        protocol_id=protocol.protocol_id,
        stage="p1b_complete",
        artifacts=artifacts,
    )
    return report


def _print_final(report, *, report_path: pathlib.Path) -> None:
    print(
        json.dumps(
            {
                "report": str(report_path),
                "artifact_id": report.artifact_id,
                "consumer_protocol_id": report.consumer_protocol_id,
                "gate0_passed": report.gate0_passed,
                "verdict": report.verdict.value,
                "next_action": report.next_action,
            },
            ensure_ascii=False,
        )
    )


def _final_return_code(report) -> int:
    return (
        0
        if report.verdict
        is RelationshipP1eVerdict.FORMAL_PREREG_FREEZE_CANDIDATE
        else 2
    )


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    protocol = load_relationship_p1e_consumer_protocol(pathlib.Path(args.protocol))
    validate_relationship_p1e_local_lineage(protocol)
    preflight, ready = _preflight(protocol, allow_download=args.allow_download)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if ready else 3
    if not ready:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 3

    root = pathlib.Path(args.output_dir)
    stage, artifacts = _load_or_initialize_checkpoint(
        root,
        protocol_id=protocol.protocol_id,
    )
    final_path = _artifact_path(root, artifacts, "p1e_report")
    if stage == "complete" and final_path is not None:
        report = load_relationship_packet1e_report(final_path)
        _print_final(report, report_path=final_path)
        return _final_return_code(report)

    baseline, gate0_report, attestation_path = _load_or_run_gate0(
        root=root,
        protocol=protocol,
        artifacts=artifacts,
    )
    p1b_report = None
    if gate0_report.gate0_passed:
        p1b_report = _load_or_run_p1b(
            root=root,
            protocol=protocol,
            baseline_attestation_path=attestation_path,
            artifacts=artifacts,
        )
    report = assess_relationship_packet1e(
        protocol=protocol,
        baseline=baseline,
        gate0_report=gate0_report,
        p1b_report=p1b_report,
    )
    json_path, _markdown_path = write_relationship_packet1e_report(
        report,
        output_dir=root,
    )
    artifacts["p1e_report"] = _relative_path(root, json_path)
    _write_checkpoint(
        root,
        protocol_id=protocol.protocol_id,
        stage="complete",
        artifacts=artifacts,
    )
    _print_final(report, report_path=json_path)
    return _final_return_code(report)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
