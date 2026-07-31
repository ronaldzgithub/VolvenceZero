"""Run the preregistered five-episode alignment review and one re-probe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from volvence_ant.evidence.provenance import (
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
from volvence_ant.experiments.ecology_same_physics_review import (
    ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_REPORT_SCHEMA_VERSION,
    validate_ecology_same_physics_alignment_review_packet,
)
from volvence_ant.experiments.ecology_same_physics_run import (
    run_ecology_same_physics_alignment_review,
)


_ROOT = Path(__file__).resolve().parents[1]
_RESULT_DIR = Path(
    "research/ant/results/ecology_recovery/same_physics_baseline"
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve(path: Path) -> Path:
    resolved = path if path.is_absolute() else _ROOT / path
    resolved.relative_to(_ROOT)
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_envelope(path: Path) -> tuple[dict, Path]:
    manifest = path.with_suffix(".manifest.json")
    verify_ant_artifact_manifest(
        manifest_path=manifest,
        repo_root=_ROOT,
    )
    envelope = json.loads(path.read_text(encoding="utf-8"))
    require_ant_artifact_envelope(envelope, path=path)
    payload = dict(envelope)
    payload.pop("provenance")
    payload.pop("evidence_envelope_schema_version")
    return payload, manifest


async def _run(args: argparse.Namespace) -> int:
    station1_preregistration_path = _resolve(
        args.station1_preregistration
    )
    station1_packet, station1_preregistration_manifest = _load_envelope(
        station1_preregistration_path
    )
    validate_ecology_same_physics_baseline_packet(
        station1_packet,
        repo_root=_ROOT,
        check_source_bindings=False,
    )
    station1_preregistration_sha256 = _sha256(
        station1_preregistration_path
    )
    review_preregistration_path = _resolve(
        args.review_preregistration
    )
    review_packet, review_preregistration_manifest = _load_envelope(
        review_preregistration_path
    )
    validate_ecology_same_physics_alignment_review_packet(
        review_packet,
        repo_root=_ROOT,
        station1_packet=station1_packet,
        station1_preregistration_sha256=(
            station1_preregistration_sha256
        ),
    )
    review_preregistration_sha256 = _sha256(
        review_preregistration_path
    )
    station1_report_path = _resolve(args.station1_report)
    station1_report, station1_report_manifest = _load_envelope(
        station1_report_path
    )
    if (
        station1_report.get("schema_version")
        != ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_REPORT_SCHEMA_VERSION
    ):
        raise ValueError(
            "alignment review requires a station1 v2 report"
        )
    if (
        station1_report.get("preregistration_sha256")
        != station1_preregistration_sha256
    ):
        raise ValueError(
            "station1 report was produced from a different "
            "preregistration"
        )
    progress_dir = _resolve(args.progress_dir)
    station1_progress_dir = _resolve(args.station1_progress_dir)
    report = _resolve(
        args.report
        if args.report is not None
        else _RESULT_DIR
        / (
            "ecology_same_physics_alignment_review."
            f"seed0.{args.run_id}.json"
        )
    )
    ensure_artifact_writable(report, overwrite=False)
    progress_dir.mkdir(parents=True, exist_ok=True)
    with ecology_p1_progress_writer_lock(progress_dir):
        try:
            result = await run_ecology_same_physics_alignment_review(
                packet=station1_packet,
                preregistration_sha256=(
                    station1_preregistration_sha256
                ),
                review_preregistration_sha256=(
                    review_preregistration_sha256
                ),
                station1_evaluation=station1_report,
                station1_progress_dir=station1_progress_dir,
                review_progress_dir=progress_dir,
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
            seeds=(int(station1_packet["formal_config"]["seed"]),),
            config={
                "review_preregistration_sha256": (
                    review_preregistration_sha256
                ),
                "station1_preregistration_sha256": (
                    station1_preregistration_sha256
                ),
                "station1_report_sha256": _sha256(
                    station1_report_path
                ),
                "review_schedule_sha256": result[
                    "review_schedule_sha256"
                ],
            },
            model_fingerprint=result["station1_checkpoint_sha256"],
            device="cpu",
            training_seeds=(
                int(station1_packet["formal_config"]["seed"]),
            ),
        ),
        input_paths=(
            station1_preregistration_path,
            station1_preregistration_manifest,
            review_preregistration_path,
            review_preregistration_manifest,
            station1_report_path,
            station1_report_manifest,
        ),
        repo_root=_ROOT,
        overwrite=False,
    )
    print(f"verdict: {result['verdict']}")
    print(f"report: {report.relative_to(_ROOT)}")
    print(f"manifest: {manifest.relative_to(_ROOT)}")
    return 0 if result["verdict"] == "GO" else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered same-physics alignment review"
        )
    )
    parser.add_argument(
        "--station1-preregistration",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--review-preregistration",
        type=Path,
        required=True,
    )
    parser.add_argument("--station1-report", type=Path, required=True)
    parser.add_argument(
        "--station1-progress-dir",
        type=Path,
        required=True,
    )
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
