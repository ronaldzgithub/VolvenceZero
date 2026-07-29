#!/usr/bin/env python3
"""Freeze and summarize the State KV P6 due-diligence evidence bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_state_kv_identification import (  # noqa: E402
    P2_PERSONA_PAIRS,
    P2_PROBE_SENTENCES,
)
from volvence_zero.agent.profile_registry import resolve_profile  # noqa: E402
from volvence_zero.state_kv_due_diligence import (  # noqa: E402
    build_due_diligence_report,
    build_freeze_manifest,
    freeze_manifest_from_json,
)

PREFIX_MANIFEST = (
    "artifacts/state_kv/projectors/"
    "qwen2.5-0.5b-state-strategy-routed-prefix.manifest.json"
)
EVIDENCE_PATHS = {
    "bank_gain": "artifacts/state_kv/verdict_bank_gain.json",
    "carrier_diagnostics": (
        "artifacts/state_kv/p4-diagnostics/"
        "verdict_carrier_diagnostics.json"
    ),
    "control_dim": (
        "artifacts/state_kv/verdict_control_dim_diagnostic.json"
    ),
    "credit_longitudinal": (
        "artifacts/state_kv/verdict_credit_longitudinal.json"
    ),
    "cost": "artifacts/state_kv/cost-gate/verdict_cost_gate.json",
    "deployment": (
        "artifacts/state_kv/deployment-state-strategy-routed-cpu/"
        "verdict_deployment_gate.json"
    ),
    "generation_seed": (
        "artifacts/state_kv/"
        "p2-state-strategy-routed-rollout-seeds-1701-1703-"
        "full-max16-generation-seed-gate/"
        "verdict_generation_seed_gate.json"
    ),
    "identification": (
        "artifacts/state_kv/"
        "p2-state-strategy-routed-repair-vs-execute-"
        "rollout-seed-1701-full-max16/"
        "verdict_identification.json"
    ),
    "judge_court": (
        "artifacts/state_kv/"
        "p2-state-strategy-routed-rollout-seed-1701-full-max16-"
        "bge-m3e-judge-court/verdict_judge_court.json"
    ),
    "quality_noninferiority": (
        "artifacts/state_kv/quality-noninferiority/"
        "verdict_quality_noninferiority.json"
    ),
    "retention": (
        "artifacts/state_kv/"
        "p2-state-strategy-routed-rollout-seed-1701-full-max16-"
        "retention/verdict_retention_gate.json"
    ),
    "temporal_causal": (
        "artifacts/state_kv/temporal-causal-state-strategy-routed-cpu/"
        "verdict_temporal_causal.json"
    ),
}
PROFILE_LABELS = (
    "state-kv-arm-a",
    "state-kv-arm-a-pure",
    "state-kv-arm-bprime",
    "state-kv-arm-e",
    "state-kv-arm-e-pure",
    "state-kv-arm-g-prefix-pure",
    "state-kv-bank-none",
    "state-kv-bank-personal-only",
    "state-kv-bank-relationship-only",
    "state-kv-bank-dual",
    "state-kv-bank-dual-router-active",
    "state-kv-bank-dual-credit-active",
    "dynamic-residual-off",
    "conditioning-credit-feedback-active",
)
SCENARIO_SETS = (
    "boundary-vs-commit",
    "repair-vs-execute",
    "carrier-diagnostics",
    "deployment-safety-controls",
    "bank-gain-four-arm",
    "control-dimension-three-arm",
    "credit-feedback-long-session",
)
METRIC_DEFINITIONS = (
    "blind-identification-bootstrap-ci",
    "prompt-byte-identity",
    "output-divergence",
    "conditioning-attention-slot-cost",
    "latency-per-generated-token",
    "slot-attention-differentiation",
    "temporal-code-causal-distance",
    "revocation-baseline-equivalence",
    "paired-bank-match-gain-bootstrap-ci",
    "full-minus-rank3-matched-outcome",
    "pe-closed-loop-longitudinal-increment",
)
JUDGE_PANEL = ("BAAI/bge-m3", "moka-ai/m3e-base")


def _resolved_experiment_config() -> dict[str, object]:
    profiles: dict[str, object] = {}
    for label in PROFILE_LABELS:
        resolved = resolve_profile(label)
        profiles[label] = {
            "base_profile": resolved.base_profile,
            "capabilities": [
                capability.name for capability in resolved.capabilities
            ],
            "flag_overrides": {
                key: (
                    value.value
                    if isinstance(value, Enum)
                    else value
                )
                for key, value in sorted(
                    resolved.merged_flag_overrides.items()
                )
            },
            "wiring_overrides": {
                owner: {
                    slot: level.value
                    for slot, level in sorted(overrides.items())
                }
                for owner, overrides in sorted(
                    resolved.merged_wiring_overrides.items()
                )
            },
        }
    scenario_material = {
        pair_id: [
            {
                "user_id": user_id,
                "state_vector": list(state_vector),
                "continuity": continuity,
                "ordering_driver": ordering_driver,
            }
            for user_id, state_vector, continuity, ordering_driver in personas
        ]
        for pair_id, personas in sorted(P2_PERSONA_PAIRS.items())
    }
    return {
        "schema_version": "state-kv-experiment-config.v1",
        "resolved_profiles": profiles,
        "generation": {
            "max_new_tokens": 16,
            "bank_gain_max_new_tokens": 4,
            "temperature": 0.2,
            "sampling_seeds": [1701, 1702, 1703],
            "probe_limit": 0,
        },
        "scenario_material": {
            "p2_persona_pairs": scenario_material,
            "p2_probe_sentences": [
                {"probe_id": probe_id, "text": text}
                for probe_id, text in P2_PROBE_SENTENCES
            ],
        },
        "metric_thresholds": {
            "bank_gain_minimum_samples": 8,
            "bank_irrelevant_router_score_ceiling": 0.2,
            "control_dim_minimum_samples": 8,
            "control_dim_minimum_outcome_delta": 0.02,
            "cost_latency_tolerance": 0.1,
            "quality_noninferiority_margin": 0.0,
        },
        "judge_panel": [
            {
                "model_id": judge_id,
                "kind": "embedding",
                "scoring_method": "embedding-cosine-mean-pool-v1",
                "material_kind": "rendered-state-statement",
            }
            for judge_id in JUDGE_PANEL
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("evidence", "freeze", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/state_kv/due_diligence",
    )
    parser.add_argument("--model-source", default="")
    parser.add_argument("--judge-source", default="")
    parser.add_argument(
        "--force-evidence",
        action="store_true",
        help="rerun completed evidence lanes instead of reusing their artifacts",
    )
    return parser


def _artifact_is_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload.get("gate_state")
    return state in {
        "pass",
        "fail",
        "mechanism_supported",
    } or payload.get("court_state") in {
        "pass",
        "fail",
    }


def _run_evidence_lanes(args: argparse.Namespace) -> None:
    commands: tuple[tuple[str, list[str], Path], ...] = (
        (
            "quality_noninferiority",
            [
                sys.executable,
                "scripts/run_state_kv_quality_noninferiority.py",
            ],
            REPO_ROOT / EVIDENCE_PATHS["quality_noninferiority"],
        ),
        (
            "bank_gain",
            [
                sys.executable,
                "scripts/run_state_kv_bank_gain_gate.py",
                "--max-new-tokens",
                "4",
                *(
                    ["--model-source", args.model_source]
                    if args.model_source
                    else []
                ),
                *(
                    ["--judge-source", args.judge_source]
                    if args.judge_source
                    else []
                ),
            ],
            REPO_ROOT / EVIDENCE_PATHS["bank_gain"],
        ),
        (
            "control_dim",
            [
                sys.executable,
                "scripts/run_state_kv_control_dim_diagnostic.py",
                *(
                    ["--model-source", args.model_source]
                    if args.model_source
                    else []
                ),
            ],
            REPO_ROOT / EVIDENCE_PATHS["control_dim"],
        ),
        (
            "credit_longitudinal",
            [
                sys.executable,
                "scripts/run_state_kv_credit_longitudinal.py",
            ],
            REPO_ROOT / EVIDENCE_PATHS["credit_longitudinal"],
        ),
    )
    for lane_id, command, output in commands:
        if not args.force_evidence and _artifact_is_complete(output):
            print(f"evidence[{lane_id}] = reused:{output.relative_to(REPO_ROOT)}")
            continue
        print(f"evidence[{lane_id}] = running")
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if not _artifact_is_complete(output):
            raise RuntimeError(
                f"evidence lane {lane_id!r} did not produce a terminal verdict"
            )
    # Older lanes are expensive frozen inputs. They are not silently skipped:
    # every one must exist and parse before the new bundle can be frozen.
    for evidence_id, relative_path in EVIDENCE_PATHS.items():
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"required evidence lane {evidence_id!r} is missing: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload.get("schema_version"):
            raise ValueError(
                f"required evidence lane {evidence_id!r} is not a verdict artifact"
            )
        print(f"evidence[{evidence_id}] = verified:{relative_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "freeze_manifest.json"
    report_path = output_dir / "verdict_due_diligence.json"

    if args.mode in ("evidence", "all"):
        _run_evidence_lanes(args)
        if args.mode == "evidence":
            return 0

    if args.mode in ("freeze", "all"):
        manifest = build_freeze_manifest(
            repo_root=REPO_ROOT,
            prefix_manifest_path=PREFIX_MANIFEST,
            evidence_paths=EVIDENCE_PATHS,
            profile_labels=PROFILE_LABELS,
            generation_seeds=(1701, 1702, 1703),
            scenario_sets=SCENARIO_SETS,
            metric_definitions=METRIC_DEFINITIONS,
            judge_panel=JUDGE_PANEL,
            experiment_config=_resolved_experiment_config(),
        )
        manifest_path.write_text(
            manifest.to_json() + "\n",
            encoding="utf-8",
        )
        print(f"freeze_id = {manifest.freeze_id}")
        print(f"manifest = {manifest_path}")
    else:
        manifest = freeze_manifest_from_json(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )

    if args.mode in ("report", "all"):
        report = build_due_diligence_report(
            repo_root=REPO_ROOT,
            manifest=manifest,
        )
        report_path.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"gate_state = {report.gate_state}")
        print(f"report = {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
