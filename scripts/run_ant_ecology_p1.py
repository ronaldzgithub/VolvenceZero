"""Run one frozen-schedule P1 ecology development repetition.

Every report is written through the evidence bundle writer, so the artifact
carries its own provenance (git SHA, dirty flag, config digest, dependency
versions, device, training seed, layout seeds) and a sidecar
``*.manifest.json`` with the exact bytes -- plan sections 2.1 and 2.3.

The default report path is run-id suffixed and the writer refuses to replace an
existing artifact, so re-running the driver can no longer destroy a previous
``BLOCK`` report in place. ``--overwrite`` is the explicit, logged escape hatch.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from volvence_ant.evidence.provenance import (
    collect_ant_provenance,
    ensure_artifact_writable,
    write_ant_artifact_bundle,
)
from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS,
    ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS,
    EcologyP1Config,
    EcologyP1ProgressPaused,
    ecology_p1_progress_writer_lock,
    run_ecology_p1,
    run_ecology_p1_diagnostics,
)


_ROOT = Path(__file__).resolve().parents[1]
_RESULT_DIR = Path("research/ant/results/ecology_recovery/p1")


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_report(*, diagnostics_only: bool, seed: int, run_id: str) -> Path:
    kind = "diagnostics." if diagnostics_only else ""
    return _RESULT_DIR / f"ecology_p1.{kind}seed{seed}.{run_id}.json"


def _resolve(path: Path) -> Path:
    resolved = path if path.is_absolute() else _ROOT / path
    resolved.relative_to(_ROOT)
    return resolved


async def _run(args: argparse.Namespace) -> int:
    config = EcologyP1Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        training_rounds=args.training_rounds,
        evaluation_rounds=args.evaluation_rounds,
        layouts_per_tier=args.layouts_per_tier,
        seed=args.seed,
    )
    progress_dir = None
    if args.progress_dir is not None:
        progress_dir = _resolve(args.progress_dir)
    repeat_reference_report = None
    if args.repeat_reference_report is not None:
        # Validate containment, then hand the loader a REPO-RELATIVE path:
        # the reference is recorded verbatim in the report, and an absolute
        # path would pin a versioned artifact to one machine's checkout.
        repeat_reference_report = _resolve(
            args.repeat_reference_report
        ).relative_to(_ROOT)
    output = _resolve(
        args.report
        if args.report is not None
        else _default_report(
            diagnostics_only=args.diagnostics_only,
            seed=config.seed,
            run_id=args.run_id,
        )
    )
    # Refuse a colliding artifact before spending the run's budget, not after.
    ensure_artifact_writable(output, overwrite=args.overwrite)
    with ecology_p1_progress_writer_lock(progress_dir):
        try:
            report = (
                run_ecology_p1_diagnostics(config)
                if args.diagnostics_only
                else await run_ecology_p1(
                    config,
                    progress_dir=progress_dir,
                    max_new_work_items=args.max_new_work_items,
                    repeat_reference_report=repeat_reference_report,
                    repo_root=_ROOT,
                )
            )
        except EcologyP1ProgressPaused as paused:
            print(str(paused))
            print(f"progress: {progress_dir.relative_to(_ROOT)}")
            return 0
    payload = report.to_dict()
    rows = (
        report.results
        if args.diagnostics_only
        else report.layout_results
    )
    manifest = write_ant_artifact_bundle(
        artifact_path=output,
        payload=payload,
        provenance=collect_ant_provenance(
            repo_root=_ROOT,
            seeds=(config.seed,),
            config=asdict(config),
            training_seeds=(config.seed,),
            layout_seeds=tuple(sorted({item.seed for item in rows})),
        ),
        repo_root=_ROOT,
        overwrite=args.overwrite,
    )
    if args.diagnostics_only:
        print(f"diagnostics passed={report.passed}")
        verdict_ok = report.passed
    else:
        print(report.description)
        verdict_ok = report.verdict == "PASS"
    print(f"report: {output.relative_to(_ROOT)}")
    print(f"manifest: {manifest.relative_to(_ROOT)}")
    # A BLOCK verdict must be visible to a shell pipeline exactly like the P2
    # driver reports one; returning 0 lets an orchestrator treat a blocked P1
    # as a successful stage and start spending P2 budget.
    return 0 if verdict_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ant ecology P1 matrix")
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument(
        "--training-rounds",
        type=int,
        default=ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS,
    )
    # The held-out budget is a frozen threshold, not a driver preference: the
    # ``formal_configuration`` gate refuses any run below
    # ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS, so a driver default of 40 would
    # have made every CLI-launched P1 run BLOCK on its own argparse default.
    parser.add_argument(
        "--evaluation-rounds",
        type=int,
        default=ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS,
    )
    parser.add_argument("--layouts-per-tier", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument(
        "--progress-dir",
        type=Path,
        default=None,
        help=(
            "Optional resumable progress directory. Training checkpoints are "
            "journaled after every episode and evaluation after every layout."
        ),
    )
    parser.add_argument(
        "--max-new-work-items",
        type=int,
        default=None,
        help=(
            "Stop cleanly after this many newly committed training episodes "
            "or evaluation layouts; requires --progress-dir."
        ),
    )
    parser.add_argument(
        "--repeat-reference-report",
        type=Path,
        default=None,
        help=(
            "A previous P1 report, produced with a DIFFERENT training seed at "
            "the same budget, used as plan section 4.7's independent "
            "repetition. Without it the repeat_run_same_direction gate FAILS: "
            "a single run cannot rule out training accident."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run identifier used in the default report filename; defaults to "
            "a UTC timestamp so no run overwrites another."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Explicit report path. Defaults to a run-id suffixed file under "
            f"{_RESULT_DIR}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Destroy an existing report and manifest at the target path. "
            "Without this flag an existing artifact is never replaced."
        ),
    )
    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = _default_run_id()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
