"""Write the source-bound alignment-review preregistration."""

from __future__ import annotations

import argparse
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
from volvence_ant.experiments.ecology_same_physics_baseline import (
    validate_ecology_same_physics_baseline_packet,
)
from volvence_ant.experiments.ecology_same_physics_review import (
    build_ecology_same_physics_alignment_review_packet,
    validate_ecology_same_physics_alignment_review_packet,
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


def _load_station1_preregistration(path: Path) -> tuple[dict, Path]:
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
        check_source_bindings=False,
    )
    return packet, manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preregister the bounded same-physics food-alignment review"
        )
    )
    parser.add_argument(
        "--station1-preregistration",
        type=Path,
        required=True,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_id = args.run_id or _default_run_id()
    station1_path = _resolve(args.station1_preregistration)
    station1_packet, station1_manifest = (
        _load_station1_preregistration(station1_path)
    )
    station1_sha256 = _sha256(station1_path)
    output = _resolve(
        args.output
        if args.output is not None
        else _RESULT_DIR
        / (
            "ecology_same_physics_alignment_review_prereg."
            f"seed0.{run_id}.json"
        )
    )
    ensure_artifact_writable(output, overwrite=False)
    packet = build_ecology_same_physics_alignment_review_packet(
        repo_root=_ROOT,
        station1_packet=station1_packet,
        station1_preregistration_sha256=station1_sha256,
    )
    validate_ecology_same_physics_alignment_review_packet(
        packet,
        repo_root=_ROOT,
        station1_packet=station1_packet,
        station1_preregistration_sha256=station1_sha256,
    )
    manifest = write_ant_artifact_bundle(
        artifact_path=output,
        payload=packet,
        provenance=collect_ant_provenance(
            repo_root=_ROOT,
            seeds=(int(packet["formal_config"]["seed"]),),
            config={
                "schema_version": packet["schema_version"],
                "station1_preregistration_sha256": station1_sha256,
                "review_schedule": packet["review_schedule"],
                "probe": packet["probe"],
                "authorization": packet["authorization"],
            },
            device="cpu",
            training_seeds=(int(packet["formal_config"]["seed"]),),
        ),
        input_paths=(station1_path, station1_manifest),
        repo_root=_ROOT,
        overwrite=False,
    )
    print("status: PREREGISTERED")
    print(f"packet: {output.relative_to(_ROOT)}")
    print(f"manifest: {manifest.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
