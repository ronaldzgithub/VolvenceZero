"""Write the station1-v4 same-physics preregistration bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from volvence_ant.evidence.provenance import (
    collect_ant_provenance,
    ensure_artifact_writable,
    write_ant_artifact_bundle,
)
from volvence_ant.experiments.ecology_same_physics_baseline import (
    build_ecology_same_physics_baseline_packet,
    validate_ecology_same_physics_baseline_packet,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preregister the current-physics typed-milestone matched control"
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_id = args.run_id or _default_run_id()
    output = _resolve(
        args.output
        if args.output is not None
        else _RESULT_DIR
        / f"ecology_same_physics_prereg.seed{args.seed}.{run_id}.json"
    )
    ensure_artifact_writable(output, overwrite=False)
    packet = build_ecology_same_physics_baseline_packet(
        repo_root=_ROOT,
        seed=args.seed,
    )
    validate_ecology_same_physics_baseline_packet(
        packet,
        repo_root=_ROOT,
    )
    manifest = write_ant_artifact_bundle(
        artifact_path=output,
        payload=packet,
        provenance=collect_ant_provenance(
            repo_root=_ROOT,
            seeds=(args.seed,),
            config={
                "schema_version": packet["schema_version"],
                "formal_config": packet["formal_config"],
                "thresholds": packet["thresholds"],
                "schedule_sha256": packet["schedule"]["full_sha256"],
                "matched_fields_sha256": (
                    packet["arms"]["matched_fields_sha256"]
                ),
            },
            device="cpu",
            training_seeds=(args.seed,),
        ),
        repo_root=_ROOT,
        overwrite=False,
    )
    print("status: PREREGISTERED")
    print(f"packet: {output.relative_to(_ROOT)}")
    print(f"manifest: {manifest.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
