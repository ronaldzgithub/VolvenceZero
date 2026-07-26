"""Run the P2 formal confirmatory ecology matrix.

Three sub-commands mirror the three batches of the plan:

* ``preflight``  -- P2-A: one training seed, full stack, timing/size/determinism.
* ``shard``      -- P2-B/P2-C: one ``(training_seed, arm)`` cell, resumable.
* ``aggregate``  -- P2.4: fold complete shards into the promotion verdict.

Every sub-command demands a P1 report whose verdict is ``PASS``; without it the
run exits non-zero before spending any budget. P1 reports are run-id suffixed,
so ``--p1-report`` has no constant default: it resolves to the newest report in
the P1 result directory and prints the file it chose.

Every artifact this driver writes goes through the evidence bundle writer, so
each preflight report, each shard report and the confirmatory report carries
its own provenance and a sidecar manifest. A shard's provenance is collected in
the process that ran that shard -- the ``provenance_clean`` gate must certify
the tree each shard was produced from, not the tree the aggregator happens to
sit on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from volvence_ant.evidence.provenance import (
    AntRunProvenance,
    collect_ant_provenance,
    ensure_artifact_writable,
    require_ant_artifact_envelope,
    stable_json_digest,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)
from volvence_ant.experiments.ecology_p2 import (
    ECOLOGY_P2_ABLATION_ARM_NAMES,
    ECOLOGY_P2_ARM_NAMES,
    ECOLOGY_P2_CORE_ARM_NAMES,
    EcologyP2Config,
    EcologyP2PrerequisiteError,
    EcologyP2ProgressPaused,
    aggregate_ecology_p2_shards,
    heldout_layout_seeds,
    preregistration_digest,
    run_ecology_p2_preflight,
    run_ecology_p2_shard,
    shard_report_from_dict,
)


_ROOT = Path(__file__).resolve().parents[1]
_P1_RESULT_DIR = Path("research/ant/results/ecology_recovery/p1")
#: ``run_ant_ecology_p1.py`` writes ``ecology_p1.seed<N>.<run-id>.json``; its
#: diagnostics reports carry a ``diagnostics.`` infix and therefore do not
#: match, which is deliberate -- a diagnostics run never unlocks P2.
_P1_REPORT_GLOB = "ecology_p1.seed*.json"
_DEFAULT_OUTPUT_DIR = Path("research/ant/results/ecology_recovery/p2")
_MANIFEST_SUFFIX = ".manifest.json"
_P1_REPORT_HELP = (
    "P1 report that unlocks P2. Defaults to the most recently modified "
    f"{_P1_REPORT_GLOB!r} under {_P1_RESULT_DIR}; the resolved path is printed "
    "before any budget is spent."
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve_p1_report(explicit: Path | None) -> Path:
    """The P1 artifact that unlocks P2 budget.

    There is no fixed default filename to fall back on: P1 reports are run-id
    suffixed, so a constant default would name a path that can never exist and
    would fail with "P1 report not found" no matter how many P1 runs had
    succeeded. The default is therefore resolved against the directory, and the
    chosen file is printed so an operator can see which run is being trusted.
    """

    if explicit is not None:
        return _resolve(explicit)
    directory = _ROOT / _P1_RESULT_DIR
    candidates = (
        tuple(
            path
            for path in sorted(directory.glob(_P1_REPORT_GLOB))
            if not path.name.endswith(_MANIFEST_SUFFIX)
        )
        if directory.is_dir()
        else ()
    )
    if not candidates:
        raise SystemExit(
            f"no P1 report found under {_P1_RESULT_DIR} matching "
            f"{_P1_REPORT_GLOB!r}; run scripts/run_ant_ecology_p1.py first, or "
            "pass --p1-report explicitly. P2 must not spend budget without a "
            "PASS P1 artifact."
        )
    newest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    print(
        f"p1-report: {newest.relative_to(_ROOT)} "
        f"(newest of {len(candidates)} candidate(s) in {_P1_RESULT_DIR})"
    )
    return newest


def _config(args: argparse.Namespace) -> EcologyP2Config:
    return EcologyP2Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        training_rounds=args.training_rounds,
        validation_rounds=args.validation_rounds,
        heldout_rounds=args.heldout_rounds,
        layouts_per_tier=args.layouts_per_tier,
        training_seeds=tuple(sorted(args.training_seeds)),
        device=args.device,
    )


def _resolve(path: Path) -> Path:
    resolved = path if path.is_absolute() else _ROOT / path
    resolved.relative_to(_ROOT)
    return resolved


def _write_bundle(
    *,
    path: Path,
    payload: dict[str, Any],
    provenance: AntRunProvenance,
    overwrite: bool,
) -> tuple[Path, Path]:
    output = _resolve(path)
    manifest = write_ant_artifact_bundle(
        artifact_path=output,
        payload=payload,
        provenance=provenance,
        repo_root=_ROOT,
        overwrite=overwrite,
    )
    return output, manifest


async def _run_preflight(args: argparse.Namespace) -> int:
    config = _config(args)
    output_path = (
        args.report
        or _DEFAULT_OUTPUT_DIR / f"ecology_p2.preflight.{args.run_id}.json"
    )
    # Refuse a colliding artifact before spending the rehearsal budget.
    ensure_artifact_writable(_resolve(output_path), overwrite=args.overwrite)
    report = await run_ecology_p2_preflight(
        config,
        training_seed=args.training_seed,
        p1_report_path=_resolve_p1_report(args.p1_report),
        repo_root=_ROOT,
        progress_dir=(
            _resolve(args.progress_dir) if args.progress_dir else None
        ),
        arms=tuple(args.arms) if args.arms else ECOLOGY_P2_CORE_ARM_NAMES,
    )
    output, manifest = _write_bundle(
        path=output_path,
        payload=report.to_dict(),
        provenance=collect_ant_provenance(
            repo_root=_ROOT,
            seeds=(report.training_seed,),
            config=asdict(config),
            model_fingerprint=report.preregistration_digest,
            device=config.device,
            training_seeds=(report.training_seed,),
            layout_seeds=heldout_layout_seeds(config),
        ),
        overwrite=args.overwrite,
    )
    print(report.description)
    print(f"report: {output.relative_to(_ROOT)}")
    print(f"manifest: {manifest.relative_to(_ROOT)}")
    return 0 if report.passed else 1


async def _run_shard(args: argparse.Namespace) -> int:
    config = _config(args)
    output_path = (
        args.report
        or _DEFAULT_OUTPUT_DIR
        / "shards"
        / (
            f"ecology_p2.seed{args.training_seed}.{args.arm}"
            f".{args.run_id}.json"
        )
    )
    # Refuse a colliding artifact before spending the shard's budget.
    ensure_artifact_writable(_resolve(output_path), overwrite=args.overwrite)
    try:
        report = await run_ecology_p2_shard(
            config,
            training_seed=args.training_seed,
            arm=args.arm,
            p1_report_path=_resolve_p1_report(args.p1_report),
            repo_root=_ROOT,
            progress_dir=(
                _resolve(args.progress_dir) if args.progress_dir else None
            ),
            max_new_work_items=args.max_new_work_items,
        )
    except EcologyP2ProgressPaused as paused:
        print(str(paused))
        return 0
    output, manifest = _write_bundle(
        path=output_path,
        payload=report.to_dict(),
        # Collected here, in the process that actually ran the shard: this is
        # the only place where the git SHA and dirty flag describe the tree the
        # shard's numbers came from.
        provenance=collect_ant_provenance(
            repo_root=_ROOT,
            seeds=(args.training_seed,),
            config=asdict(config),
            model_fingerprint=report.policy_digest,
            device=config.device,
            training_seeds=(args.training_seed,),
            layout_seeds=tuple(
                sorted({item.seed for item in report.layout_results})
            ),
        ),
        overwrite=args.overwrite,
    )
    print(report.description)
    print(f"shard: {output.relative_to(_ROOT)}")
    print(f"manifest: {manifest.relative_to(_ROOT)}")
    return 0


def _shard_provenance(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """The provenance the shard process recorded for itself."""

    raw = payload.get("provenance")
    if not isinstance(raw, dict):
        raise SystemExit(
            f"P2 shard has no provenance block: {path}. Shards written before "
            "the bundle writer cannot certify the tree they ran on; rerun "
            "them with this driver."
        )
    for field in ("git_sha", "working_tree_dirty"):
        if field not in raw:
            raise SystemExit(
                f"P2 shard provenance is missing {field!r}: {path}"
            )
    return raw


def _run_aggregate(args: argparse.Namespace) -> int:
    config = _config(args)
    output_path = (
        args.report
        or _DEFAULT_OUTPUT_DIR / f"ecology_p2.confirmatory.{args.run_id}.json"
    )
    ensure_artifact_writable(_resolve(output_path), overwrite=args.overwrite)
    shard_dir = _resolve(args.shard_dir)
    paths = tuple(
        path
        for path in sorted(shard_dir.glob("*.json"))
        if not path.name.endswith(_MANIFEST_SUFFIX)
    )
    if not paths:
        raise SystemExit(f"no P2 shard reports under {shard_dir}")
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        verify_ant_artifact_manifest(
            manifest_path=path.with_suffix(_MANIFEST_SUFFIX),
            repo_root=_ROOT,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit(f"P2 shard report must be a JSON object: {path}")
        # A shard written against an older envelope cannot report the device it
        # ran on or its seed namespaces, so it cannot be folded into a formal
        # verdict.
        require_ant_artifact_envelope(payload, path=path)
        payloads.append((path, payload))
    shards = tuple(shard_report_from_dict(payload) for _, payload in payloads)

    shard_provenance = tuple(
        (path, _shard_provenance(path, payload)) for path, payload in payloads
    )
    dirty_shards = tuple(
        str(path.relative_to(_ROOT))
        for path, item in shard_provenance
        if bool(item["working_tree_dirty"])
    )
    shard_shas = {str(item["git_sha"]) for _, item in shard_provenance}
    if len(shard_shas) != 1:
        # Plan section 5.4: any code change invalidates the whole batch. A
        # mixed-SHA aggregate would silently average two implementations.
        raise SystemExit(
            "P2 shards were produced from different commits "
            f"({sorted(shard_shas)}); the confirmatory batch is invalid and "
            "must be rerun in full"
        )
    provenance = collect_ant_provenance(
        repo_root=_ROOT,
        seeds=config.training_seeds,
        config=asdict(config),
        model_fingerprint=preregistration_digest(config),
        device=config.device,
        training_seeds=config.training_seeds,
        layout_seeds=heldout_layout_seeds(config),
    )
    if dirty_shards:
        print(f"dirty shard worktrees: {list(dirty_shards)}")
    report = aggregate_ecology_p2_shards(
        shards,
        # Certifies every tree that produced a number in this report, plus the
        # aggregating tree itself.
        worktree_clean=(
            not provenance.working_tree_dirty and not dirty_shards
        ),
        config=config,
    )
    output, manifest = _write_bundle(
        path=output_path,
        payload=report.to_dict(),
        provenance=collect_ant_provenance(
            repo_root=_ROOT,
            seeds=config.training_seeds,
            config=asdict(config),
            model_fingerprint=stable_json_digest(report.shard_digests),
            device=config.device,
            training_seeds=config.training_seeds,
            layout_seeds=heldout_layout_seeds(config),
        ),
        overwrite=args.overwrite,
    )
    print(report.description)
    for endpoint in report.primary_endpoints:
        status = "PASS" if endpoint.passed else "BLOCK"
        print(f"  [{status}] {endpoint.name}")
    print(f"report: {output.relative_to(_ROOT)}")
    print(f"manifest: {manifest.relative_to(_ROOT)}")
    return 0 if report.verdict == "PASS" else 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-ants", type=int, default=8)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--training-rounds", type=int, default=80)
    parser.add_argument("--validation-rounds", type=int, default=80)
    parser.add_argument("--heldout-rounds", type=int, default=120)
    parser.add_argument("--layouts-per-tier", type=int, default=5)
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Explicit artifact path. Defaults to a run-id suffixed file under "
            f"{_DEFAULT_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Run identifier used in the default artifact filename; defaults "
            "to a UTC timestamp so no run overwrites another."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Destroy an existing artifact and manifest at the target path. "
            "Without this flag an existing artifact is never replaced."
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the ant ecology P2 confirmatory matrix"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="P2-A full-stack rehearsal")
    _add_common(preflight)
    preflight.add_argument("--training-seed", type=int, default=None)
    preflight.add_argument("--p1-report", type=Path, default=None, help=_P1_REPORT_HELP)
    preflight.add_argument("--progress-dir", type=Path, default=None)
    preflight.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Arms to rehearse; defaults to the P2-B core matrix "
            f"({', '.join(ECOLOGY_P2_CORE_ARM_NAMES)})"
        ),
    )

    shard = sub.add_parser("shard", help="one (training_seed, arm) shard")
    _add_common(shard)
    shard.add_argument("--training-seed", type=int, required=True)
    shard.add_argument(
        "--arm",
        type=str,
        required=True,
        choices=ECOLOGY_P2_ARM_NAMES,
        help=(
            "core matrix: "
            f"{', '.join(ECOLOGY_P2_CORE_ARM_NAMES)}; ablations: "
            f"{', '.join(ECOLOGY_P2_ABLATION_ARM_NAMES)}"
        ),
    )
    shard.add_argument("--p1-report", type=Path, default=None, help=_P1_REPORT_HELP)
    shard.add_argument("--progress-dir", type=Path, default=None)
    shard.add_argument(
        "--max-new-work-items",
        type=int,
        default=None,
        help=(
            "Stop cleanly after this many newly committed training episodes "
            "or held-out layouts; requires --progress-dir."
        ),
    )

    aggregate = sub.add_parser("aggregate", help="fold shards into a verdict")
    _add_common(aggregate)
    aggregate.add_argument(
        "--shard-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR / "shards",
    )

    args = parser.parse_args()
    if args.run_id is None:
        args.run_id = _default_run_id()
    try:
        if args.command == "preflight":
            return asyncio.run(_run_preflight(args))
        if args.command == "shard":
            return asyncio.run(_run_shard(args))
        if args.command == "aggregate":
            return _run_aggregate(args)
    except EcologyP2PrerequisiteError as error:
        print(f"P2 blocked by the serial constraint: {error}")
        return 2
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
