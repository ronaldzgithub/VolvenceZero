#!/usr/bin/env python3
"""Resume-safe learned-backend ACTIVE evidence orchestrator.

This script is an operator wrapper around the existing evidence tools. It does
not flip runtime defaults. It resumes at artifact boundaries by recording a
stage marker after every successful command and by re-validating the expected
artifact before skipping a stage.

Promotion-level evidence still requires a continuous real-substrate soak
artifact. The optional chunked lane is cross-platform stability evidence only;
it is deliberately not fed into the ACTIVE gate.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "learned-active-evidence-runner.v1"

STAGE_SMOKE = "shadow-smoke"
STAGE_CHUNKED_SOAK = "platform-chunked-soak"
STAGE_REAL_SOAK = "real-soak"
STAGE_CAPACITY = "capacity-ladder"
STAGE_ABLATION = "same-substrate-ablation"
STAGE_BUILD = "build-promotion-evidence"
STAGE_EVALUATE = "evaluate-promotion"

ORDERED_STAGES = (
    STAGE_SMOKE,
    STAGE_CHUNKED_SOAK,
    STAGE_REAL_SOAK,
    STAGE_CAPACITY,
    STAGE_ABLATION,
    STAGE_BUILD,
    STAGE_EVALUATE,
)


def _json_load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stage_marker(output_dir: Path, stage: str) -> Path:
    return output_dir / ".runner" / f"{stage}.json"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _platform_default_device() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "mps"
    if system == "windows":
        return "cuda"
    return "cuda"


def _run_command(
    command: Sequence[str],
    *,
    dry_run: bool,
    allow_failure: bool = False,
) -> int:
    printable = " ".join(command)
    print(f"[learned-active] $ {printable}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    if completed.returncode != 0 and not allow_failure:
        raise SystemExit(f"stage command failed ({completed.returncode}): {printable}")
    return completed.returncode


def _mark_stage(
    *,
    output_dir: Path,
    stage: str,
    artifact: Path | None,
    command: Sequence[str] | None,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "completed_at": _now(),
        "artifact": str(artifact) if artifact is not None else "",
        "command": list(command or ()),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    if extra:
        payload.update(extra)
    _json_write(_stage_marker(output_dir, stage), payload)


def _stage_done(output_dir: Path, stage: str, artifact: Path | None) -> bool:
    marker = _stage_marker(output_dir, stage)
    if not marker.is_file():
        return False
    if artifact is not None and not artifact.is_file():
        return False
    if artifact is not None:
        _json_load(artifact)
    return True


def _skip_or_run(
    *,
    args: argparse.Namespace,
    stage: str,
    artifact: Path | None,
    command: Sequence[str],
    allow_failure: bool = False,
    extra: dict[str, object] | None = None,
) -> int:
    if args.resume and not args.force_stage and _stage_done(args.output_dir, stage, artifact):
        print(f"[learned-active] resume: skipping {stage} (artifact present)", flush=True)
        return 0
    rc = _run_command(command, dry_run=args.dry_run, allow_failure=allow_failure)
    if rc == 0 and not args.dry_run:
        if artifact is not None:
            _json_load(artifact)
        _mark_stage(
            output_dir=args.output_dir,
            stage=stage,
            artifact=artifact,
            command=command,
            extra=extra,
        )
    return rc


def _selected_stages(args: argparse.Namespace) -> tuple[str, ...]:
    if args.only:
        requested = tuple(part.strip() for part in args.only.split(",") if part.strip())
        unknown = sorted(set(requested) - set(ORDERED_STAGES))
        if unknown:
            raise SystemExit(f"unknown --only stage(s): {', '.join(unknown)}")
        return requested
    if args.from_stage:
        if args.from_stage not in ORDERED_STAGES:
            raise SystemExit(f"unknown --from-stage: {args.from_stage}")
        return ORDERED_STAGES[ORDERED_STAGES.index(args.from_stage) :]
    return ORDERED_STAGES


def _smoke(args: argparse.Namespace) -> Path:
    stage_dir = args.output_dir / "shadow_smoke"
    artifact = stage_dir / "learned_shadow_evidence_smoke.json"
    command = [
        sys.executable,
        "scripts/run_learned_shadow_evidence_smoke.py",
        "--turns",
        str(args.smoke_turns),
        "--output-dir",
        str(stage_dir),
    ]
    _skip_or_run(args=args, stage=STAGE_SMOKE, artifact=artifact, command=command)
    return artifact


def _chunked_soak(args: argparse.Namespace) -> Path:
    stage_dir = args.output_dir / "platform_chunked_soak"
    artifact = stage_dir / "learned_shadow_soak_chunked.json"
    command = [
        sys.executable,
        "-u",
        "scripts/run_learned_shadow_soak_chunks.py",
        "--chunks",
        str(args.chunks),
        "--turns-per-chunk",
        str(args.turns_per_chunk),
        "--output-dir",
        str(stage_dir),
        "--substrate-mode",
        args.substrate_mode,
        "--substrate-device",
        args.substrate_device,
    ]
    _skip_or_run(
        args=args,
        stage=STAGE_CHUNKED_SOAK,
        artifact=artifact,
        command=command,
        extra={"promotion_evidence": False},
    )
    return artifact


def _real_soak(args: argparse.Namespace) -> Path:
    stage_dir = args.output_dir / "real_soak"
    artifact = stage_dir / "learned_shadow_soak.json"
    command = [
        sys.executable,
        "-u",
        "scripts/run_learned_shadow_soak.py",
        "--turns",
        str(args.turns),
        "--sample-every",
        str(args.sample_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--temporal-latent-dim",
        str(args.temporal_latent_dim),
        "--backend-combo",
        "runtime+ssl+internal-rl+cms-torch",
        "--substrate-mode",
        args.substrate_mode,
        "--substrate-model-id",
        args.substrate_model_id,
        "--substrate-device",
        args.substrate_device,
        "--output-dir",
        str(stage_dir),
    ]
    if args.substrate_allow_download:
        command.append("--substrate-allow-download")
    _skip_or_run(args=args, stage=STAGE_REAL_SOAK, artifact=artifact, command=command)
    return artifact


def _capacity_ladder(args: argparse.Namespace) -> Path:
    stage_dir = args.output_dir / "capacity_ladder"
    artifact = stage_dir / "capacity_ladder_manifest.json"
    command = [
        sys.executable,
        "scripts/run_capacity_ladder.py",
        "--n-z",
        args.capacity_n_z,
        "--turns",
        args.capacity_turns,
        "--backend-combos",
        "runtime+ssl+internal-rl+cms-torch",
        "--substrates",
        args.capacity_substrates,
        "--seeds",
        args.capacity_seeds,
        "--output-dir",
        str(stage_dir),
    ]
    if args.execute_capacity:
        command.append("--execute")
    _skip_or_run(args=args, stage=STAGE_CAPACITY, artifact=artifact, command=command)
    return artifact


def _ablation(args: argparse.Namespace) -> Path | None:
    if args.ablation_verdict:
        artifact = args.ablation_verdict
        if not artifact.is_file():
            raise SystemExit(f"--ablation-verdict does not exist: {artifact}")
        _json_load(artifact)
        _mark_stage(
            output_dir=args.output_dir,
            stage=STAGE_ABLATION,
            artifact=artifact,
            command=("external-artifact", str(artifact)),
            extra={"external": True},
        )
        return artifact
    if args.skip_ablation:
        _mark_stage(
            output_dir=args.output_dir,
            stage=STAGE_ABLATION,
            artifact=None,
            command=("skip",),
            extra={"skipped": True, "promotion_gate_bits": "pe_off/eta_off remain false"},
        )
        return None

    stage_dir = args.output_dir / "same_substrate_ablation"
    artifact = stage_dir / f"verdict_{args.ablation_phase}.json"
    command = [
        sys.executable,
        "scripts/companion_bench/run_same_substrate_ablation.py",
        "--phase",
        args.ablation_phase,
        "--output-dir",
        str(stage_dir),
    ]
    if args.resume:
        command.append("--resume")
    if args.ablation_family:
        command += ["--family", args.ablation_family]
    if args.ablation_dry_run:
        command.append("--dry-run")
    _skip_or_run(args=args, stage=STAGE_ABLATION, artifact=artifact, command=command)
    return artifact


def _build_promotion(args: argparse.Namespace, soak_artifact: Path, ablation: Path | None) -> Path:
    artifact = args.output_dir / "promotion" / "promotion_evidence.json"
    command = [
        sys.executable,
        "scripts/build_learned_promotion_evidence.py",
        "--soak-artifact",
        str(soak_artifact),
        "--output",
        str(artifact),
    ]
    if ablation is not None:
        command += ["--ablation-verdict", str(ablation)]
    if args.skip_cms_ab_test:
        command.append("--skip-cms-ab-test")
    if args.prior_runtime_active:
        command.append("--prior-runtime-active")
    if args.prior_ssl_active:
        command.append("--prior-ssl-active")
    _skip_or_run(args=args, stage=STAGE_BUILD, artifact=artifact, command=command)
    return artifact


def _evaluate(args: argparse.Namespace, promotion_artifact: Path) -> Path:
    artifact = args.output_dir / "promotion" / "promotion_report.json"
    command = [
        sys.executable,
        "scripts/evaluate_learned_backend_promotion.py",
        "--artifact",
        str(promotion_artifact),
        "--output",
        str(artifact),
    ]
    _skip_or_run(args=args, stage=STAGE_EVALUATE, artifact=artifact, command=command)
    if not args.dry_run:
        report = _json_load(artifact)
        print(
            "[learned-active] promotion all_eligible={eligible}".format(
                eligible=report.get("all_eligible")
            ),
            flush=True,
        )
    return artifact


def _write_summary(
    args: argparse.Namespace,
    *,
    artifacts: dict[str, str],
    stages: Sequence[str],
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "selected_stages": list(stages),
        "artifacts": artifacts,
        "notes": (
            "Chunked soak is stability evidence only. ACTIVE promotion requires "
            "the continuous real-soak artifact plus component controls and CMS A/B."
        ),
    }
    _json_write(args.output_dir / "run_summary.json", payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/learned_active_evidence"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-stage", choices=ORDERED_STAGES)
    parser.add_argument("--only", help="Comma-separated stage list.")

    parser.add_argument("--smoke-turns", type=int, default=4)
    parser.add_argument("--run-platform-chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunks", type=int, default=25)
    parser.add_argument("--turns-per-chunk", type=int, default=20)

    parser.add_argument("--turns", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--substrate-mode", choices=("synthetic", "hf"), default="hf")
    parser.add_argument("--substrate-model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--substrate-device", default=_platform_default_device())
    parser.add_argument("--substrate-allow-download", action="store_true")

    parser.add_argument("--capacity-n-z", default="16,64,256")
    parser.add_argument("--capacity-turns", default="500")
    parser.add_argument("--capacity-substrates", default="qwen-1.5b-main")
    parser.add_argument("--capacity-seeds", default="0")
    parser.add_argument("--execute-capacity", action="store_true")

    parser.add_argument("--ablation-verdict", type=Path)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-phase", choices=("p1", "p2"), default="p1")
    parser.add_argument("--ablation-family")
    parser.add_argument("--ablation-dry-run", action="store_true")

    parser.add_argument("--skip-cms-ab-test", action="store_true")
    parser.add_argument("--prior-runtime-active", action="store_true")
    parser.add_argument("--prior-ssl-active", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    selected = _selected_stages(args)
    artifacts: dict[str, str] = {}
    soak_artifact = args.output_dir / "real_soak" / "learned_shadow_soak.json"
    ablation_artifact = args.ablation_verdict
    promotion_artifact = args.output_dir / "promotion" / "promotion_evidence.json"

    if STAGE_SMOKE in selected:
        artifacts[STAGE_SMOKE] = str(_smoke(args))
    if STAGE_CHUNKED_SOAK in selected and args.run_platform_chunks:
        artifacts[STAGE_CHUNKED_SOAK] = str(_chunked_soak(args))
    elif STAGE_CHUNKED_SOAK in selected:
        _mark_stage(
            output_dir=args.output_dir,
            stage=STAGE_CHUNKED_SOAK,
            artifact=None,
            command=("skip",),
            extra={"skipped": True},
        )
    if STAGE_REAL_SOAK in selected:
        soak_artifact = _real_soak(args)
        artifacts[STAGE_REAL_SOAK] = str(soak_artifact)
    if STAGE_CAPACITY in selected:
        artifacts[STAGE_CAPACITY] = str(_capacity_ladder(args))
    if STAGE_ABLATION in selected:
        ablation_artifact = _ablation(args)
        if ablation_artifact is not None:
            artifacts[STAGE_ABLATION] = str(ablation_artifact)
    if STAGE_BUILD in selected:
        promotion_artifact = _build_promotion(args, soak_artifact, ablation_artifact)
        artifacts[STAGE_BUILD] = str(promotion_artifact)
    if STAGE_EVALUATE in selected:
        artifacts[STAGE_EVALUATE] = str(_evaluate(args, promotion_artifact))

    if not args.dry_run:
        _write_summary(args, artifacts=artifacts, stages=selected)
        print(f"[learned-active] summary written: {args.output_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
