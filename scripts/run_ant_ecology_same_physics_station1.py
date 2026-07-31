"""Run the preregistered station1-v4 same-physics matched control."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from volvence_ant.evidence.provenance import (
    atomic_write_json,
    collect_ant_provenance,
    ensure_artifact_writable,
    require_ant_artifact_envelope,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)
from volvence_ant.experiments.ecology_p1 import (
    EcologyP1ProgressPaused,
    ecology_p1_progress_writer_lock,
)
from volvence_ant.experiments.ecology_same_physics_baseline import (
    validate_ecology_same_physics_baseline_packet,
)
from volvence_ant.experiments.ecology_same_physics_run import (
    run_ecology_same_physics_station1,
)


_ROOT = Path(__file__).resolve().parents[1]
_RESULT_DIR = Path(
    "research/ant/results/ecology_recovery/same_physics_baseline"
)
_PROGRESS_SCHEMA = "digital-ant-ecology-same-physics-run-binding.v1"


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve(path: Path) -> Path:
    resolved = path if path.is_absolute() else _ROOT / path
    resolved.relative_to(_ROOT)
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_preregistration(path: Path) -> tuple[dict, str, Path]:
    manifest = path.with_suffix(".manifest.json")
    verify_ant_artifact_manifest(
        manifest_path=manifest,
        repo_root=_ROOT,
    )
    envelope = json.loads(path.read_text(encoding="utf-8"))
    require_ant_artifact_envelope(envelope, path=path)
    packet = dict(envelope)
    packet.pop("provenance")
    packet.pop("evidence_envelope_schema_version")
    validate_ecology_same_physics_baseline_packet(
        packet,
        repo_root=_ROOT,
    )
    return packet, _sha256(path), manifest


def _bind_progress(
    *,
    progress_dir: Path,
    preregistration_path: Path,
    preregistration_sha256: str,
) -> None:
    binding_path = progress_dir / "same-physics-binding.json"
    expected = {
        "schema_version": _PROGRESS_SCHEMA,
        "preregistration_path": str(
            preregistration_path.relative_to(_ROOT)
        ),
        "preregistration_sha256": preregistration_sha256,
    }
    if binding_path.exists():
        actual = json.loads(binding_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "same-physics progress is bound to a different "
                "preregistration"
            )
        return
    unexpected = tuple(
        path.name
        for path in progress_dir.iterdir()
        if path.name != ".writer.lock"
    )
    if unexpected:
        raise ValueError(
            "same-physics run requires a new empty progress directory; "
            f"found {unexpected!r}"
        )
    atomic_write_json(binding_path, expected, overwrite=False)


async def _run(args: argparse.Namespace) -> int:
    preregistration = _resolve(args.preregistration)
    packet, preregistration_sha256, preregistration_manifest = (
        _load_preregistration(preregistration)
    )
    progress_dir = _resolve(args.progress_dir)
    report = _resolve(
        args.report
        if args.report is not None
        else _RESULT_DIR
        / f"ecology_same_physics_station1.seed0.{args.run_id}.json"
    )
    ensure_artifact_writable(report, overwrite=False)
    progress_dir.mkdir(parents=True, exist_ok=True)
    with ecology_p1_progress_writer_lock(progress_dir):
        _bind_progress(
            progress_dir=progress_dir,
            preregistration_path=preregistration,
            preregistration_sha256=preregistration_sha256,
        )
        try:
            result = await run_ecology_same_physics_station1(
                packet=packet,
                preregistration_sha256=preregistration_sha256,
                progress_dir=progress_dir,
                max_new_work_items=args.max_new_work_items,
            )
        except EcologyP1ProgressPaused as paused:
            print(str(paused))
            print(f"progress: {progress_dir.relative_to(_ROOT)}")
            return 0
    manifest = write_ant_artifact_bundle(
        artifact_path=report,
        payload=result,
        provenance=collect_ant_provenance(
            repo_root=_ROOT,
            seeds=(int(packet["formal_config"]["seed"]),),
            config={
                "preregistration_sha256": preregistration_sha256,
                "formal_config": packet["formal_config"],
                "station1_thresholds": packet["thresholds"]["station1"],
            },
            model_fingerprint=preregistration_sha256,
            device="cpu",
            training_seeds=(int(packet["formal_config"]["seed"]),),
        ),
        input_paths=(preregistration, preregistration_manifest),
        repo_root=_ROOT,
        overwrite=False,
    )
    print(f"verdict: {result['verdict']}")
    print(f"report: {report.relative_to(_ROOT)}")
    print(f"manifest: {manifest.relative_to(_ROOT)}")
    return 0 if result["verdict"] == "GO" else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered same-physics station1-v4"
    )
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--max-new-work-items", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = _default_run_id()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
