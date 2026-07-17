"""Build the latent-semantic grounding report (experiment 1 of
``docs/specs/semantic-grounding-evidence.md``).

Two modes:

1. Offline (preferred for real evidence): analyze a previously captured
   turn-evidence artifact::

       python scripts/build_semantic_grounding_report.py \
           --turns-artifact artifacts/semantic_grounding/turns.json

2. Capture + analyze in one go (smoke / synthetic lane)::

       python -u scripts/build_semantic_grounding_report.py \
           --run-capture --turns-per-case 12

   Runs scripted cases through an ``AgentSessionRunner``, extracts the
   per-turn grounding evidence, writes BOTH the turns artifact and the
   report. With ``--substrate-mode synthetic`` (default) this is a
   wiring smoke only; real evidence requires the hf substrate lane and
   the LLM semantic proposal channel (see the spec's preconditions).

Outputs (in ``--output-dir``):

* ``semantic_grounding_turns.json``  (capture mode only)
* ``semantic_grounding_report.json``
* ``semantic_grounding_manifest.json``  (sha256 sidecar)

The report is a non-gating reference artifact (R12 readout-only).
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

from volvence_zero.agent.semantic_grounding import (
    GroundingThresholds,
    SemanticGroundingTurnCapture,
    build_semantic_grounding_report,
    turn_evidence_from_payload,
    turn_evidence_to_payload,
)

_MANIFEST_SCHEMA_VERSION = "semantic-grounding-manifest.v1"

# Scripted capture cases. Cases sharing a scenario family use different
# surface wording on purpose (D3 transfer needs the same deep structure
# to appear in more than one case).
_CAPTURE_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "commitment-arc-a",
        (
            "Can you make sure the fuel order goes out today?",
            "I'm counting on that fuel order - it has to happen.",
            "Did the fuel order go out like we agreed?",
            "Good. Next: promise me you'll brief the crew before dawn.",
            "Actually the briefing slipped, didn't it? What happened?",
            "Ok, let's get it back on track for tomorrow.",
        ),
    ),
    (
        "commitment-arc-b",
        (
            "Please lock in the venue booking by tonight.",
            "The venue matters a lot to me - don't let it slip.",
            "Is the venue actually booked now, as promised?",
            "Great. One more commitment: send the invites this week.",
            "The invites still haven't gone out. That's a miss.",
            "Let's recover: what's the new plan for the invites?",
        ),
    ),
    (
        "repair-arc-a",
        (
            "That last suggestion completely missed my point.",
            "I'm honestly frustrated - you didn't listen.",
            "Let me explain again what I actually need.",
            "Ok, that response feels closer to what I meant.",
            "Thanks for adjusting. I feel better about this now.",
            "Let's move forward with the corrected plan.",
        ),
    ),
    (
        "repair-arc-b",
        (
            "You got the schedule wrong and it cost us the morning.",
            "I need you to acknowledge that mistake first.",
            "Here is what the schedule should have looked like.",
            "Yes - that revision matches what I wanted.",
            "Alright, trust partially restored. Keep it careful.",
            "Now finish the rest of the week's plan properly.",
        ),
    ),
    (
        "planning-arc-a",
        (
            "Walk me through the harbor plan for tomorrow.",
            "The tide tables changed; adjust the schedule.",
            "What's still open on the checklist?",
            "The client moved the deadline up two days; replan.",
            "Summarize what we decided this week.",
            "Keep the backup pilot on standby as a contingency.",
        ),
    ),
    (
        "planning-arc-b",
        (
            "Lay out the launch plan for next Monday.",
            "Marketing pushed the date; rework the timeline.",
            "Which launch tasks are still unresolved?",
            "Leadership wants it two days earlier - adapt the plan.",
            "Recap the launch decisions we've made so far.",
            "Add a rollback plan in case the release fails.",
        ),
    ),
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


async def _capture_turns(
    *,
    turns_per_case: int,
    temporal_latent_dim: int,
    substrate_mode: str,
    substrate_model_id: str,
    substrate_device: str,
    substrate_local_files_only: bool,
):
    from volvence_zero.agent.session import AgentSessionRunner

    if substrate_mode == "hf":
        from volvence_zero.substrate import build_transformers_runtime_with_fallback

        default_runtime = build_transformers_runtime_with_fallback(
            model_id=substrate_model_id,
            device=substrate_device,
            local_files_only=substrate_local_files_only,
            allow_live_substrate_mutation=False,
            fallback_to_builtin=False,
        )
    elif substrate_mode == "synthetic":
        from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime

        default_runtime = SyntheticOpenWeightResidualRuntime(
            model_id="semantic-grounding-capture-synthetic",
        )
    else:
        raise ValueError(f"unsupported substrate_mode {substrate_mode!r}")

    all_turns = []
    global_turn_index = 0
    for case_id, pool in _CAPTURE_CASES:
        runner = AgentSessionRunner(
            session_id=f"semantic-grounding-{case_id}",
            temporal_latent_dim=temporal_latent_dim,
            default_residual_runtime=default_runtime,
            rare_heavy_enabled=False,
        )
        capture = SemanticGroundingTurnCapture()
        for index in range(turns_per_case):
            text = pool[index % len(pool)]
            result = await runner.run_turn(text)
            global_turn_index += 1
            capture.observe_turn(
                turn_index=global_turn_index,
                active_snapshots=result.active_snapshots,
                case_id=case_id,
            )
        all_turns.extend(capture.turns)
        print(
            f"[grounding] case {case_id}: {turns_per_case} turns captured",
            flush=True,
        )
    return tuple(all_turns)


async def main(
    *,
    output_dir: Path,
    turns_artifact: Path | None,
    run_capture: bool,
    turns_per_case: int,
    temporal_latent_dim: int,
    substrate_mode: str,
    substrate_model_id: str,
    substrate_device: str,
    substrate_local_files_only: bool,
    min_closed_segments: int,
    min_reused_families: int,
    min_nonzero_delta_ratio: float,
) -> int:
    if turns_artifact is None and not run_capture:
        raise SystemExit(
            "Provide --turns-artifact <path> (offline analysis) or "
            "--run-capture (capture + analyze)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    provenance = _collect_provenance()

    if turns_artifact is not None:
        payload = json.loads(turns_artifact.read_text(encoding="utf-8"))
        turns = turn_evidence_from_payload(payload)
        print(
            f"[grounding] loaded {len(turns)} captured turns from "
            f"{turns_artifact}",
            flush=True,
        )
    else:
        turns = await _capture_turns(
            turns_per_case=turns_per_case,
            temporal_latent_dim=temporal_latent_dim,
            substrate_mode=substrate_mode,
            substrate_model_id=substrate_model_id,
            substrate_device=substrate_device,
            substrate_local_files_only=substrate_local_files_only,
        )
        turns_payload = turn_evidence_to_payload(turns)
        turns_payload["provenance"] = provenance
        turns_payload["substrate_mode"] = substrate_mode
        turns_path = output_dir / "semantic_grounding_turns.json"
        turns_path.write_text(
            json.dumps(turns_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifact_paths.append(turns_path)
        print(f"[grounding] turns artifact written to {turns_path}")

    thresholds = GroundingThresholds(
        min_closed_segments=min_closed_segments,
        min_reused_families=min_reused_families,
        min_nonzero_delta_ratio=min_nonzero_delta_ratio,
    )
    report = build_semantic_grounding_report(turns, thresholds=thresholds)
    report_payload = report.to_payload()
    report_payload["provenance"] = provenance
    if run_capture:
        report_payload["substrate_mode"] = substrate_mode
        if substrate_mode == "synthetic":
            report_payload["evidence_tier"] = (
                "synthetic-smoke: wiring evidence only; real grounding "
                "evidence requires the hf substrate lane."
            )

    report_path = output_dir / "semantic_grounding_report.json"
    report_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifact_paths.append(report_path)

    manifest_path = output_dir / "semantic_grounding_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": _MANIFEST_SCHEMA_VERSION,
                "artifact_kind": "semantic_grounding_manifest",
                "artifacts": [_file_record(path) for path in artifact_paths],
                "provenance": provenance,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "[grounding] verdict: "
        f"{report.verdict} (D1={report.d1_discrimination.passed}, "
        f"D2={report.d2_lead.passed}, D3={report.d3_transfer.passed}, "
        f"coverage={report.coverage.meets_thresholds})"
    )
    print(f"[grounding] report written to {report_path}")
    print(f"[grounding] manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/semantic_grounding"),
    )
    parser.add_argument(
        "--turns-artifact",
        type=Path,
        default=None,
        help="Previously captured semantic_grounding_turns.json to analyze.",
    )
    parser.add_argument(
        "--run-capture",
        action="store_true",
        help="Run scripted cases and capture turn evidence before analysis.",
    )
    parser.add_argument("--turns-per-case", type=int, default=12)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument(
        "--substrate-mode",
        default="synthetic",
        choices=["synthetic", "hf"],
    )
    parser.add_argument(
        "--substrate-model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--substrate-device", default="cpu")
    parser.add_argument("--substrate-allow-download", action="store_true")
    parser.add_argument("--min-closed-segments", type=int, default=50)
    parser.add_argument("--min-reused-families", type=int, default=3)
    parser.add_argument(
        "--min-nonzero-delta-ratio", type=float, default=0.3
    )
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            main(
                output_dir=args.output_dir,
                turns_artifact=args.turns_artifact,
                run_capture=args.run_capture,
                turns_per_case=args.turns_per_case,
                temporal_latent_dim=args.temporal_latent_dim,
                substrate_mode=args.substrate_mode,
                substrate_model_id=args.substrate_model_id,
                substrate_device=args.substrate_device,
                substrate_local_files_only=not args.substrate_allow_download,
                min_closed_segments=args.min_closed_segments,
                min_reused_families=args.min_reused_families,
                min_nonzero_delta_ratio=args.min_nonzero_delta_ratio,
            )
        )
    )
