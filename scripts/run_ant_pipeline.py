"""One-command digital-ant live demo and formal evidence pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import webbrowser

from volvence_ant.evidence import (
    atomic_write_json,
    collect_ant_provenance,
    stable_json_digest,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)
from volvence_ant.env import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.runtime import AntSession, AntSessionConfig
from volvence_ant.viz import LiveAntDashboard, write_replay_dashboard
from volvence_ant.viz.render import save_trajectory_animation

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _ROOT / "research/ant/results"
_FIGURES = _ROOT / "research/ant/figures"
_RUNNER_STATE = _RESULTS / ".runner"
_RUNNER_SCHEMA = "digital-ant-pipeline-stage.v1"

_OUTPUTS = {
    "phase0": ("phase0_homing.json", "phase0_route_learning.json"),
    "matched": ("matched_control.json",),
    "motor": ("motor_calibration.v1.json",),
    "colony": ("phase1_colony.json",),
    "caste": ("phase2_caste.json",),
    "g1": ("dual_substrate.json",),
    "demos": (
        "g2_perturbation.json",
        "g3_bio_overlay.json",
        "g4_safety_reflex.json",
    ),
    "active": ("digital-ant-evidence-bundle.v2.json",),
}


def _commands(
    *,
    profile: str,
    model_id: str | None,
    model_source: str | None,
    workers: int | None,
) -> dict[str, list[str] | None]:
    py = sys.executable
    if profile == "demo":
        seeds, train_ticks, ticks, trace_turns = "0", "20", "20", "20"
        phase0 = ["--exposures", "3", "--route-length", "3", "--n-trials", "4"]
        colony = ["--n-ants", "4", "--rounds", "20", "--seeds", seeds]
        caste = ["--n-individuals", "4", "--rounds", "20"]
    else:
        seeds, train_ticks, ticks, trace_turns = "0,1,2,3,4", "500", "200", "500"
        phase0 = ["--exposures", "10", "--route-length", "5", "--n-trials", "24"]
        colony = ["--n-ants", "20", "--rounds", "700", "--seeds", seeds]
        caste = ["--n-individuals", "16", "--rounds", "500"]
    matched_workers = (
        workers
        if workers is not None
        else (1 if profile == "demo" else min(5, os.cpu_count() or 1))
    )
    g1 = None
    if model_id:
        g1 = [
            py,
            "scripts/run_ant_dual_substrate.py",
            "--model-id",
            model_id,
            "--turns",
            "4" if profile == "formal" else "2",
        ]
        if model_source:
            g1.extend(("--model-source", model_source))
    return {
        "phase0": [py, "scripts/run_ant_phase0.py", *phase0],
        "matched": [
            py,
            "scripts/run_ant_matched_control.py",
            "--train-ticks",
            train_ticks,
            "--ticks",
            ticks,
            "--seeds",
            seeds,
            "--with-latent",
            "--workers",
            str(matched_workers),
        ],
        "motor": [
            py,
            "scripts/run_ant_motor_calibration.py",
            "--seeds",
            seeds,
            "--ticks",
            "60",
            "--switch-tick",
            "30",
        ],
        "colony": [py, "scripts/run_ant_colony.py", *colony],
        "caste": [py, "scripts/run_ant_caste.py", *caste],
        "g1": g1,
        "demos": [py, "scripts/run_ant_demos.py"],
        "active": [
            py,
            "scripts/run_ant_active_evidence.py",
            "--trace-turns",
            trace_turns,
            "--train-ticks",
            train_ticks,
            "--ticks",
            ticks,
            "--seeds",
            seeds,
        ],
    }


def _semantic_command(command: list[str] | None) -> list[str] | None:
    if command is None:
        return None
    semantic: list[str] = []
    skip_next = False
    for token in command[1:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--resume":
            continue
        if token == "--workers":
            skip_next = True
            continue
        semantic.append(token)
    return semantic


def _stage_fingerprint(
    *,
    profile: str,
    stage: str,
    command: list[str] | None,
    outputs: tuple[Path, ...],
) -> str:
    return stable_json_digest(
        {
            "schema_version": _RUNNER_SCHEMA,
            "profile": profile,
            "stage": stage,
            "command": _semantic_command(command),
            "outputs": [str(path.relative_to(_ROOT)) for path in outputs],
        }
    )


def _stage_manifests(outputs: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(path.with_suffix(".manifest.json") for path in outputs)


def _resume_stage(
    *,
    stage: str,
    fingerprint: str,
    outputs: tuple[Path, ...],
) -> bool:
    marker_path = _RUNNER_STATE / f"{stage}.json"
    if not marker_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != _RUNNER_SCHEMA:
        raise ValueError(f"unsupported pipeline stage marker: {marker_path}")
    if marker.get("stage") != stage or marker.get("fingerprint") != fingerprint:
        raise ValueError(f"pipeline stage marker configuration mismatch: {marker_path}")
    manifests = _stage_manifests(outputs)
    expected = [str(path.relative_to(_ROOT)) for path in manifests]
    if marker.get("manifests") != expected:
        raise ValueError(f"pipeline stage marker manifest mismatch: {marker_path}")
    for manifest in manifests:
        verify_ant_artifact_manifest(manifest_path=manifest, repo_root=_ROOT)
    return True


def _commit_stage(
    *,
    stage: str,
    fingerprint: str,
    outputs: tuple[Path, ...],
) -> None:
    manifests = _stage_manifests(outputs)
    for manifest in manifests:
        verify_ant_artifact_manifest(manifest_path=manifest, repo_root=_ROOT)
    atomic_write_json(
        _RUNNER_STATE / f"{stage}.json",
        {
            "schema_version": _RUNNER_SCHEMA,
            "stage": stage,
            "fingerprint": fingerprint,
            "manifests": [str(path.relative_to(_ROOT)) for path in manifests],
        },
    )


def _artifact_verdict(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "overall_verdict" in payload:
        return str(payload["overall_verdict"])
    if "verdict" in payload:
        return str(payload["verdict"])
    if "eligible" in payload:
        return "PASS" if payload["eligible"] else "BLOCK"
    passed = payload.get("passes_antbot_scale", payload.get("familiarity_improved"))
    return "PASS" if passed else "BLOCK"


async def _live_dashboard(*, ticks: int, open_browser: bool) -> tuple[Path, Path | None]:
    dashboard = LiveAntDashboard()
    dashboard.start()
    if open_browser:
        webbrowser.open(dashboard.url)
    print(f"[dashboard] {dashboard.url}")
    tracks = {}
    try:
        for label, writeback in (("learned", True), ("no-optimize", False)):
            world = AntWorld(
                config=AntWorldConfig(seed=0),
                food_sources=(FoodSource(x=6.0, y=0.0),),
            )
            session = AntSession(
                world,
                config=AntSessionConfig(
                    temporal_latent_dim=16,
                    seed=0,
                    session_id=f"pipeline-dashboard:{label}",
                    joint_apply_writeback=writeback,
                ),
            )
            for _ in range(ticks):
                dashboard.publish(label, await session.step())
            tracks[label] = tuple(session.trajectory)
        dashboard.finish()
        replay_path = dashboard.export_replay(_RESULTS / "dashboard_replay.json")
        write_replay_dashboard(
            tracks=tracks, out_path=_FIGURES / "digital_ant_dashboard.html"
        )
        animation_path = save_trajectory_animation(
            tracks=[
                {
                    "label": label,
                    "xs": [record.x for record in records],
                    "ys": [record.y for record in records],
                }
                for label, records in tracks.items()
            ],
            nest=(0.0, 0.0),
            out_path=_FIGURES / "digital_ant_replay.mp4",
        )
        await asyncio.sleep(1.0)
        return replay_path, animation_path
    finally:
        dashboard.close()


async def main(args: argparse.Namespace) -> int:
    _RESULTS.mkdir(parents=True, exist_ok=True)
    _FIGURES.mkdir(parents=True, exist_ok=True)
    commands = _commands(
        profile=args.profile,
        model_id=args.model_id,
        model_source=args.model_source,
        workers=args.workers,
    )
    selected = tuple(commands) if args.stage == "all" else (args.stage,)
    stages: list[dict] = []
    for stage in selected:
        outputs = tuple(_RESULTS / name for name in _OUTPUTS[stage])
        command = commands[stage]
        fingerprint = _stage_fingerprint(
            profile=args.profile,
            stage=stage,
            command=command,
            outputs=outputs,
        )
        if args.resume and _resume_stage(
            stage=stage,
            fingerprint=fingerprint,
            outputs=outputs,
        ):
            status = (
                "PASS"
                if all(_artifact_verdict(path) == "PASS" for path in outputs)
                else "BLOCK"
            )
            executed = False
            resumed = True
        elif command is None:
            status = "BLOCK"
            executed = False
            resumed = False
        else:
            print(f"[pipeline] stage={stage}")
            run_command = list(command)
            if args.resume and stage == "matched":
                run_command.append("--resume")
            subprocess.run(run_command, cwd=_ROOT, check=True)
            _commit_stage(
                stage=stage,
                fingerprint=fingerprint,
                outputs=outputs,
            )
            executed = True
            resumed = False
            status = (
                "PASS"
                if outputs and all(_artifact_verdict(path) == "PASS" for path in outputs)
                else "BLOCK"
            )
        stages.append(
            {
                "stage": stage,
                "status": status,
                "executed": executed,
                "resumed": resumed,
                "outputs": [str(path.relative_to(_ROOT)) for path in outputs],
            }
        )

    replay_inputs: list[Path] = []
    if args.dashboard:
        replay, animation = await _live_dashboard(
            ticks=40 if args.profile == "demo" else 120,
            open_browser=not args.no_open,
        )
        replay_inputs.append(replay)
        replay_inputs.append(_FIGURES / "digital_ant_dashboard.html")
        if animation is not None:
            replay_inputs.append(animation)

    manifests = [
        path
        for output_names in _OUTPUTS.values()
        for name in output_names
        if (path := (_RESULTS / name).with_suffix(".manifest.json")).is_file()
    ]
    for manifest in manifests:
        verify_ant_artifact_manifest(manifest_path=manifest, repo_root=_ROOT)
    artifact_inputs = [
        _ROOT / output
        for stage in stages
        for output in stage["outputs"]
        if (stage["executed"] or stage["resumed"])
        and (_ROOT / output).is_file()
    ]
    supporting_assets = [
        *manifests,
        *sorted(_FIGURES.glob("*")),
        *sorted((_ROOT / "research/ant/reference_data").glob("*")),
    ]
    all_inputs = tuple(
        dict.fromkeys(
            path
            for path in (*artifact_inputs, *replay_inputs, *supporting_assets)
            if path.is_file()
        )
    )
    overall = (
        "PASS"
        if stages and all(stage["status"] == "PASS" for stage in stages)
        else "BLOCK"
    )
    summary = {
        "artifact_kind": "digital-ant-pipeline-summary.v1",
        "profile": args.profile,
        "stages": stages,
        "overall_verdict": overall,
        "verified_manifests": [str(path.relative_to(_ROOT)) for path in manifests],
        "legacy_v1_evidence_deprecated": True,
        "dashboard": args.dashboard,
    }
    manifest = write_ant_artifact_bundle(
        artifact_path=_RESULTS / "pipeline_summary.json",
        payload=summary,
        provenance=collect_ant_provenance(
            repo_root=_ROOT,
            seeds=(0,) if args.profile == "demo" else (0, 1, 2, 3, 4),
            config={
                "profile": args.profile,
                "stage": args.stage,
                "dashboard": args.dashboard,
            },
        ),
        input_paths=all_inputs,
        repo_root=_ROOT,
    )
    print(f"[pipeline] overall={overall}; manifest={manifest}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("demo", "formal"), default="demo")
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "phase0",
            "matched",
            "motor",
            "colony",
            "caste",
            "g1",
            "demos",
            "active",
        ),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--model-source")
    parser.add_argument(
        "--workers",
        type=int,
        help="matched-control seed workers; formal defaults to min(5, CPU count)",
    )
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
