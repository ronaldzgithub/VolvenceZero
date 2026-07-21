"""Train and gate the learned three-object digital-ant checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from volvence_ant.evidence.ecology_checkpoint import (
    write_ecology_checkpoint_bundle,
)
from volvence_ant.experiments.ecology_curriculum import (
    EcologyCurriculumConfig,
    train_and_evaluate_ecology_checkpoint,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ARCHIVE = Path("research/ant/results/ecology_checkpoint.v4.vzac")
_DEFAULT_REPORT = Path("research/ant/results/ecology_checkpoint.v4.json")


def _repo_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else _REPO_ROOT / path
    resolved.relative_to(_REPO_ROOT)
    return resolved


async def _run(args: argparse.Namespace) -> int:
    config = EcologyCurriculumConfig(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        stage_rounds=args.stage_rounds,
        stage_episodes=args.stage_episodes,
        mastery_min_episodes=args.mastery_min_episodes,
        mastery_min_pickups=args.mastery_min_pickups,
        mastery_min_deliveries=args.mastery_min_deliveries,
        mastery_min_heat_events=args.mastery_min_heat_events,
        interleave_every=args.interleave_every,
        validation_rounds=args.validation_rounds,
        validation_seeds=tuple(args.validation_seeds),
        heldout_rounds=args.heldout_rounds,
        heldout_seeds=tuple(args.heldout_seeds),
        seed=args.seed,
    )
    candidate = await train_and_evaluate_ecology_checkpoint(config)
    archive_path = _repo_path(args.archive)
    report_path = _repo_path(args.report)
    manifest_path = write_ecology_checkpoint_bundle(
        candidate=candidate,
        archive_path=archive_path,
        report_path=report_path,
        repo_root=_REPO_ROOT,
    )
    print(candidate.report.description)
    print(f"checkpoint: {archive_path.relative_to(_REPO_ROOT)}")
    print(f"report: {report_path.relative_to(_REPO_ROOT)}")
    print(f"manifest: {manifest_path.relative_to(_REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Train a real AntSession ecology controller and freeze held-out gates")
    )
    parser.add_argument("--n-ants", type=int, default=8)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--stage-rounds", type=int, default=80)
    parser.add_argument("--stage-episodes", type=int, default=4)
    parser.add_argument("--mastery-min-episodes", type=int, default=3)
    parser.add_argument("--mastery-min-pickups", type=int, default=2)
    parser.add_argument("--mastery-min-deliveries", type=int, default=1)
    parser.add_argument(
        "--mastery-min-heat-events",
        type=int,
        default=2,
    )
    parser.add_argument("--interleave-every", type=int, default=2)
    parser.add_argument("--validation-rounds", type=int, default=80)
    parser.add_argument(
        "--validation-seeds",
        type=int,
        nargs="+",
        default=[43, 59],
    )
    parser.add_argument("--heldout-rounds", type=int, default=120)
    parser.add_argument(
        "--heldout-seeds",
        type=int,
        nargs="+",
        default=[101, 211, 307, 401, 503],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Tensor runtime device. CUDA enables the ant temporal runtime on GPU.",
    )
    parser.add_argument("--archive", type=Path, default=_DEFAULT_ARCHIVE)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    args = parser.parse_args()
    if args.device == "cuda":
        os.environ["VZ_TENSOR_DEVICE"] = "cuda"
    else:
        os.environ.pop("VZ_TENSOR_DEVICE", None)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
