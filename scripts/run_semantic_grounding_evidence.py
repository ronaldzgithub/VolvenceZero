"""Semantic-grounding evidence pipeline orchestrator.

Runs the two experiments of ``docs/specs/semantic-grounding-evidence.md``
as staged lanes and assembles a single summary artifact:

* ``unit``  — pytest acceptance tests for both harnesses (seconds;
  validates the harness itself, produces no evidence).
* ``smoke`` — synthetic lane for both experiments (seconds; wiring +
  differential-design check, output is explicitly non-citable).
* ``hf``    — real-substrate lane (minutes on CPU; this is the run that
  actually feeds ``claim_latent_abstraction_semantically_grounded`` and
  ``claim_semantic_tracking_not_llm_dependent``).

Every invocation writes to a fresh timestamped directory under
``--output-root`` and ends with ``summary.json`` recording per-stage
status, the extracted verdicts and provenance. Exit code is non-zero if
any requested stage fails, so the pipeline is CI-friendly.

Usage::

    # default: unit + smoke (safe to run anytime)
    python scripts/run_semantic_grounding_evidence.py

    # milestone evidence run (real substrate, both experiments)
    python scripts/run_semantic_grounding_evidence.py --lane hf \
        --substrate-device mps --hf-turns-per-case 20

    # everything
    python scripts/run_semantic_grounding_evidence.py --lane all
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SUMMARY_SCHEMA_VERSION = "semantic-grounding-evidence-summary.v1"

_UNIT_TEST_PATHS = (
    "packages/vz-runtime/tests/test_semantic_grounding.py",
    "tests/lifeform_e2e/test_semantic_proposal_ablation.py",
    "packages/lifeform-service/tests/test_semantic_proposal_channel_switch.py",
)

_LANES = ("unit", "smoke", "hf")


def _git_output(args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + args,
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
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


class _StageRecord:
    def __init__(self, stage_id: str, command: tuple[str, ...]) -> None:
        self.stage_id = stage_id
        self.command = command
        self.status = "pending"
        self.elapsed_seconds = 0.0
        self.notes: dict[str, object] = {}

    def to_payload(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "command": list(self.command),
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "notes": self.notes,
        }


def _run_stage(record: _StageRecord, *, log_path: Path) -> bool:
    """Run one subprocess stage, teeing output to console and a log file."""

    print(f"\n=== [{record.stage_id}] {' '.join(record.command)}", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            record.command,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        returncode = process.wait()
    record.elapsed_seconds = time.monotonic() - started
    record.status = "passed" if returncode == 0 else "failed"
    record.notes["returncode"] = returncode
    record.notes["log"] = str(log_path)
    print(
        f"=== [{record.stage_id}] {record.status} "
        f"({record.elapsed_seconds:.1f}s)",
        flush=True,
    )
    return returncode == 0


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_verdicts(lane_dir: Path) -> dict[str, object]:
    """Pull the citable readouts out of the lane's artifacts."""

    verdicts: dict[str, object] = {}
    ablation = _read_json(lane_dir / "ablation" / "semantic_proposal_ablation_report.json")
    if ablation is not None:
        verdicts["semantic_proposal_ablation"] = {
            "verdict": ablation["verdict"],
            "overall_drop": ablation["overall_drop"],
            "proposal_channel_drop": ablation["proposal_channel_drop"],
            "typed_event_drop": ablation["typed_event_drop"],
            "on_arm_grounding": ablation["on_arm"]["grounding_verdict"],
            "off_arm_grounding": ablation["off_arm"]["grounding_verdict"],
            "evidence_tier": ablation.get("evidence_tier", "hf"),
        }
    grounding = _read_json(lane_dir / "grounding" / "semantic_grounding_report.json")
    if grounding is not None:
        verdicts["semantic_grounding"] = {
            "verdict": grounding["verdict"],
            "d1_passed": grounding["d1_discrimination"]["passed"],
            "d2_passed": grounding["d2_lead"]["passed"],
            "d3_passed": grounding["d3_transfer"]["passed"],
            "coverage_met": grounding["coverage"]["meets_thresholds"],
            "evidence_tier": grounding.get("evidence_tier", "hf"),
        }
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--lane",
        action="append",
        choices=[*_LANES, "all"],
        help=(
            "Lane(s) to run; repeatable. Default: unit + smoke. "
            "'hf' is the real-evidence lane; 'all' runs everything."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/semantic_grounding_evidence"),
        help="Root directory; each run gets a fresh UTC-timestamped subdir.",
    )
    parser.add_argument("--substrate-model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--substrate-device",
        default="cpu",
        help="hf lane device (cpu / mps / cuda).",
    )
    parser.add_argument(
        "--substrate-allow-download",
        action="store_true",
        help="Allow downloading model weights on first hf-lane run.",
    )
    parser.add_argument(
        "--hf-turns-per-case",
        type=int,
        default=20,
        help=(
            "Experiment-1 capture depth in the hf lane. The grounding "
            "coverage gate needs >=50 closed segments; raise this if the "
            "report says insufficient-coverage."
        ),
    )
    args = parser.parse_args()

    requested = args.lane or ["unit", "smoke"]
    lanes = list(_LANES) if "all" in requested else [
        lane for lane in _LANES if lane in requested
    ]

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (_REPO_ROOT / args.output_root / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    print(f"[evidence] lanes: {', '.join(lanes)}")
    print(f"[evidence] output: {run_dir}")

    stages: list[_StageRecord] = []
    lane_verdicts: dict[str, object] = {}

    def _stage(stage_id: str, command: tuple[str, ...]) -> bool:
        record = _StageRecord(stage_id, command)
        stages.append(record)
        return _run_stage(record, log_path=run_dir / f"{stage_id}.log")

    all_ok = True

    if "unit" in lanes:
        all_ok &= _stage(
            "unit-pytest",
            (python, "-m", "pytest", *_UNIT_TEST_PATHS, "-q"),
        )

    if "smoke" in lanes and all_ok:
        smoke_dir = run_dir / "smoke"
        all_ok &= _stage(
            "smoke-ablation",
            (
                python,
                "-u",
                "scripts/build_semantic_proposal_ablation_report.py",
                "--output-dir",
                str(smoke_dir / "ablation"),
            ),
        )
        all_ok &= _stage(
            "smoke-grounding",
            (
                python,
                "-u",
                "scripts/build_semantic_grounding_report.py",
                "--run-capture",
                "--output-dir",
                str(smoke_dir / "grounding"),
            ),
        )
        lane_verdicts["smoke"] = _extract_verdicts(smoke_dir)

    if "hf" in lanes and all_ok:
        hf_dir = run_dir / "hf"
        download_flag = (
            ("--substrate-allow-download",) if args.substrate_allow_download else ()
        )
        all_ok &= _stage(
            "hf-ablation",
            (
                python,
                "-u",
                "scripts/build_semantic_proposal_ablation_report.py",
                "--substrate-mode",
                "hf",
                "--arm-runtime",
                "hf-llm",
                "--substrate-model-id",
                args.substrate_model_id,
                "--substrate-device",
                args.substrate_device,
                *download_flag,
                "--output-dir",
                str(hf_dir / "ablation"),
            ),
        )
        all_ok &= _stage(
            "hf-grounding",
            (
                python,
                "-u",
                "scripts/build_semantic_grounding_report.py",
                "--run-capture",
                "--substrate-mode",
                "hf",
                "--substrate-model-id",
                args.substrate_model_id,
                "--substrate-device",
                args.substrate_device,
                *download_flag,
                "--turns-per-case",
                str(args.hf_turns_per_case),
                "--output-dir",
                str(hf_dir / "grounding"),
            ),
        )
        lane_verdicts["hf"] = _extract_verdicts(hf_dir)

    summary = {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "artifact_kind": "semantic_grounding_evidence_summary",
        "non_gating": True,
        "lanes_requested": lanes,
        "overall_status": "passed" if all_ok else "failed",
        "stages": [record.to_payload() for record in stages],
        "verdicts": lane_verdicts,
        "provenance": _collect_provenance(),
        "notes": (
            "smoke-lane verdicts are wiring/differential-design evidence "
            "only and are not citable for the semantic grounding claims; "
            "claim evidence requires the hf lane (see "
            "docs/specs/semantic-grounding-evidence.md and the claims in "
            "docs/specs/evidence_program.md)."
        ),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== semantic-grounding evidence summary ===")
    for record in stages:
        print(f"  [{record.status:>6}] {record.stage_id} ({record.elapsed_seconds:.1f}s)")
    for lane, verdicts in lane_verdicts.items():
        for experiment, detail in verdicts.items():
            print(f"  [{lane}] {experiment}: {detail['verdict']}")
    print(f"  summary written to {summary_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
