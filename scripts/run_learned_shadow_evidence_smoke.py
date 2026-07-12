"""One-command learned-shadow evidence smoke (CP-LSS-01, P0 wiring tier).

Runs a short session under the frozen learned-shadow operator profile
(n_z=16, all four torch autograd backends SHADOW) and writes a unified JSON
artifact proving every owner produced SHADOW evidence with zero write-back:

    python scripts/run_learned_shadow_evidence_smoke.py

This is the synthetic / CPU lane only. It validates wiring and SHADOW
no-side-effect semantics; it is NOT ACTIVE-promotion evidence (that requires
the Linux CUDA real-trace lane per the AGI-uplift plan and
docs/specs/evidence_program.md).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from volvence_zero.agent.learned_shadow_evidence import (
    LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
    build_learned_shadow_rollout_config,
    collect_learned_shadow_evidence,
)
from volvence_zero.agent.session import AgentSessionRunner

_MANIFEST_SCHEMA_VERSION = "learned-shadow-evidence-manifest.v1"

_DEFAULT_TURNS: tuple[str, ...] = (
    "Walk me through the harbor plan for tomorrow.",
    "The tide tables changed; adjust the schedule.",
    "Now summarize what we committed to.",
    "One more check: anything still open?",
)


def _git_output(args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + args, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _collect_provenance() -> dict[str, object]:
    status = _git_output(("status", "--porcelain"))
    return {
        "git_sha": _git_output(("rev-parse", "HEAD")),
        "git_branch": _git_output(("branch", "--show-current")),
        "working_tree_dirty": status not in {"", "unknown"},
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _verify_written_manifest(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload["schema_version"] != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version mismatch: {payload['schema_version']!r}")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest.artifacts must be a non-empty list")
    for item in artifacts:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact missing: {path}")
        actual = _file_record(path)
        if actual["sha256"] != item["sha256"]:
            raise ValueError(f"manifest sha256 mismatch for {path}")


async def main(*, output_dir: Path, turn_count: int) -> int:
    if turn_count < 3:
        raise ValueError(
            "turn_count must be >= 3 so the joint-loop schedule runs a full "
            f"cycle (internal RL evidence); got {turn_count}."
        )
    turns = tuple(
        _DEFAULT_TURNS[index % len(_DEFAULT_TURNS)] for index in range(turn_count)
    )
    runner = AgentSessionRunner(
        config=build_learned_shadow_rollout_config(),
        temporal_latent_dim=LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
        rare_heavy_enabled=False,
    )
    print(f"[learned-shadow] profile: n_z={LEARNED_SHADOW_TEMPORAL_LATENT_DIM}, four backends SHADOW")
    for index, text in enumerate(turns, start=1):
        await runner.run_turn(text)
        print(f"[learned-shadow] turn {index}/{turn_count} complete")

    # Fails loudly (LearnedShadowEvidenceError) if any owner is missing
    # evidence or wrote back under SHADOW.
    payload = collect_learned_shadow_evidence(runner)
    payload["turn_count"] = turn_count
    payload["provenance"] = _collect_provenance()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "learned_shadow_evidence_smoke.json"
    artifact_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path = output_dir / "learned_shadow_evidence_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": _MANIFEST_SCHEMA_VERSION,
                "artifact_kind": "learned_shadow_evidence_manifest",
                "source_schema_version": payload["schema_version"],
                "artifacts": [_file_record(artifact_path)],
                "provenance": payload["provenance"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _verify_written_manifest(manifest_path)

    world = payload["temporal_runtime"]["world"]
    print(
        "[learned-shadow] runtime parity: "
        f"within_tolerance={world['within_tolerance']} promotable={world['promotable']}"
    )
    print(
        "[learned-shadow] ssl torch: "
        f"loss={payload['temporal_ssl']['torch_prediction_loss']:.4f} "
        f"wrote_back={payload['temporal_ssl']['torch_wrote_back']}"
    )
    print(f"[learned-shadow] internal_rl kind={payload['internal_rl']['kind']}")
    print(f"[learned-shadow] cms backend={payload['cms']['backend']}")
    print(f"[learned-shadow] evidence written to {artifact_path}")
    print(f"[learned-shadow] manifest written to {manifest_path} (verified)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/learned_shadow_evidence_smoke")
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=4,
        help="Number of smoke turns (>= 3 so a full joint-loop cycle fires).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(output_dir=args.output_dir, turn_count=args.turns)))
