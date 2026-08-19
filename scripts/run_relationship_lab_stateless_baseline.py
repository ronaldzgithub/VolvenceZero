#!/usr/bin/env python3
"""Run and freeze the local-Qwen stateless Gate 0 baseline."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(
    0,
    str(_REPO_ROOT / "packages" / "lifeform-domain-emogpt" / "src"),
)
sys.path.insert(
    0,
    str(_REPO_ROOT / "packages" / "lifeform-evolution" / "src"),
)

from lifeform_evolution.relationship_lab_baseline import (  # noqa: E402
    DEFAULT_STATELESS_MODEL_ID,
    DEFAULT_STATELESS_MODEL_SOURCE,
    HFStatelessRelationshipActionPolicy,
    freeze_stateless_baseline_attestation,
    run_stateless_baseline,
    write_stateless_baseline_run,
)
from lifeform_evolution.relationship_lab_gate0 import (  # noqa: E402
    Gate0CalibrationConfig,
    run_relationship_gate0_calibration,
    write_relationship_gate0_report,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", default=DEFAULT_STATELESS_MODEL_SOURCE)
    parser.add_argument("--model-id", default=DEFAULT_STATELESS_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--seeds",
        default="101,211,307",
        help="Comma-separated matched generation seeds.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face network access; default is local cache only.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "artifacts" / "relationship_lab" / f"stateless_baseline_{int(time.time())}"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    policy = HFStatelessRelationshipActionPolicy(
        model_source=args.model_source,
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=not args.allow_download,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    run = run_stateless_baseline(policy, seed_schedule=seeds)
    frozen_at = datetime.now(timezone.utc).isoformat()
    output_dir = pathlib.Path(args.output_dir)
    _ledger, _summary, attestation_path = write_stateless_baseline_run(
        run,
        output_dir=output_dir,
        frozen_at_iso=frozen_at,
    )
    attestation = freeze_stateless_baseline_attestation(
        run,
        frozen_at_iso=frozen_at,
    )
    gate_report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(),
        baseline=attestation,
    )
    gate_json, _gate_markdown = write_relationship_gate0_report(
        gate_report,
        output_dir / "gate0",
    )
    print(
        json.dumps(
            {
                "baseline_attestation": str(attestation_path),
                "gate0_report": str(gate_json),
                "valid_decisions": run.valid_decisions,
                "evaluated_decisions": len(run.decisions),
                "accuracy": run.correct_decisions / len(run.decisions),
                "machinery_ready": gate_report.machinery_ready,
                "gate0_passed": gate_report.gate0_passed,
            },
            ensure_ascii=False,
        )
    )
    return 0 if gate_report.gate0_passed else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
