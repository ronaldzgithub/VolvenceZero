"""Phase 2 SHADOW evidence smoke for SYS-1 / COG-1 / COG-2 / COG-3.

Runs the canonical ``pe-eta`` baseline next to the four explicit Phase 2
candidate profiles on a small scripted case subset and writes a compact JSON
artifact. This is an opt-in evidence command; default paper-suite matrices do
not include these profiles.

Example:

    python scripts/run_phase2_shadow_evidence_smoke.py --case-limit 1

Fast local / CI smoke without loading a real substrate:

    python scripts/run_phase2_shadow_evidence_smoke.py --synthetic-runner

Include Phase 3 combination profiles:

    python scripts/run_phase2_shadow_evidence_smoke.py --include-phase3-combos
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

from volvence_zero.agent import (
    DEFAULT_DIALOGUE_PROOF_CASES,
    ScriptedDialogueCase,
    build_standard_dialogue_runner,
    default_phase2_shadow_evidence_profiles,
    default_phase3_combination_shadow_profiles,
    run_phase2_shadow_evidence_smoke,
)
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.evaluation import (
    build_cross_generation_aggregate_snapshot,
    build_deterministic_head_to_head_snapshot,
)
from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime


_FOCUS_METRICS: tuple[str, ...] = (
    "mean_persona_geometry_drift",
    "mean_persona_regime_geometry_alignment",
    "cpd_beta_switch_recommended_count",
    "mean_cpd_pe_spike_score",
    "mean_cpd_reward_shift_score",
    "mean_least_control_score",
    "mean_least_control_effort",
    "counterfactual_readout_turn_count",
    "tom_distinct_interlocutor_max",
    "tom_record_total_max",
)

_ARTIFACT_SCHEMA_VERSION = "phase2-shadow-evidence-smoke.v1"
_MANIFEST_SCHEMA_VERSION = "phase2-shadow-evidence-manifest.v1"


def _focus_values(metric_means: dict[str, float]) -> dict[str, float]:
    return {key: metric_means[key] for key in _FOCUS_METRICS if key in metric_means}


def _git_output(args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + args,
            check=True,
            capture_output=True,
            text=True,
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
        raise ValueError(
            f"manifest schema_version mismatch: {payload['schema_version']!r}"
        )
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
        if actual["size_bytes"] != item["size_bytes"]:
            raise ValueError(f"manifest size mismatch for {path}")


def _markdown_metric_row(profile_label: str, values: dict[str, float]) -> str:
    if not values:
        return f"| `{profile_label}` | no focus metrics |"
    formatted = "<br>".join(f"`{key}` = {value:.4f}" for key, value in values.items())
    return f"| `{profile_label}` | {formatted} |"


def _build_markdown_brief(
    *,
    payload: dict[str, object],
    artifact_path: Path,
) -> str:
    focus_metric_means = payload["focus_metric_means"]
    if not isinstance(focus_metric_means, dict):
        raise TypeError("payload['focus_metric_means'] must be a dict")
    head_to_head_results = payload["head_to_head_results"]
    if not isinstance(head_to_head_results, list):
        raise TypeError("payload['head_to_head_results'] must be a list")
    gate = payload["cross_generation_gate_evidence"]
    if not isinstance(gate, dict):
        raise TypeError("payload['cross_generation_gate_evidence'] must be a dict")

    metric_rows = "\n".join(
        _markdown_metric_row(str(profile_label), values)
        for profile_label, values in focus_metric_means.items()
        if isinstance(values, dict)
    )
    head_to_head_lines = "\n".join(
        (
            f"- `{item['profile_a']}` vs `{item['profile_b']}`: "
            f"winrate `{item['winrate_a_vs_b']:.4f}` over `{item['case_count']}` metrics "
            f"(`{item['judge_kind']}`)"
        )
        for item in head_to_head_results
        if isinstance(item, dict)
    )
    if not head_to_head_lines:
        head_to_head_lines = "- No head-to-head rows produced."

    return f"""# Phase 2/3 Shadow Evidence Smoke

> Generated by `scripts/run_phase2_shadow_evidence_smoke.py`.

## Run Shape

- Baseline: `{payload['baseline_label']}`
- Profiles: `{', '.join(str(label) for label in payload['profile_labels'])}`
- Cases: `{', '.join(str(case_id) for case_id in payload['case_ids'])}`
- Runner kind: `{payload['runner_kind']}`
- Includes Phase 3 combos: `{payload['include_phase3_combos']}`
- JSON artifact: `{artifact_path}`
- Git SHA: `{payload['provenance']['git_sha']}`
- Working tree dirty: `{payload['provenance']['working_tree_dirty']}`

## Focus Metric Means

| Profile | Focus metrics |
|---|---|
{metric_rows}

## Deterministic Head-To-Head

{head_to_head_lines}

## Cross-Generation Gate Evidence

- Evidence id: `{gate['evidence_id']}`
- Validation score: `{gate['validation_score']:.4f}`
- Head-to-head aggregate winrate: `{gate['head_to_head_aggregate_winrate']:.4f}`
- Rollback evidence present: `{gate['rollback_evidence_present']}`
- Capacity within cap: `{gate['capacity_within_cap']}`
- Audit evidence id: `{gate['audit_evidence_id']}`

## Notes

This is a SHADOW evidence artifact. It does not make an ACTIVE decision and
does not alter the default paper-suite profile matrix.
"""


def _synthetic_runner(profile_label: str, case: ScriptedDialogueCase) -> AgentSessionRunner:
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id=f"phase2-shadow:{profile_label}:{case.case_id}"
    )
    runtime.runtime_origin = f"synthetic-{profile_label}"
    return build_standard_dialogue_runner(
        profile_label=profile_label,
        case=case,
        residual_runtime=runtime,
    )


async def main(
    *,
    output_dir: Path,
    case_limit: int,
    synthetic_runner: bool,
    include_phase3_combos: bool = False,
) -> int:
    if case_limit <= 0:
        raise ValueError(f"case_limit must be > 0, got {case_limit!r}")
    cases = DEFAULT_DIALOGUE_PROOF_CASES[:case_limit]
    profile_labels = ("pe-eta",) + default_phase2_shadow_evidence_profiles()
    if include_phase3_combos:
        profile_labels = profile_labels + default_phase3_combination_shadow_profiles()
    print(f"[phase2-shadow] profiles={profile_labels}")
    print(f"[phase2-shadow] cases={[case.case_id for case in cases]}")
    print(f"[phase2-shadow] runner={'synthetic' if synthetic_runner else 'default'}")

    if include_phase3_combos:
        from volvence_zero.agent import run_dialogue_pe_eta_ablation_benchmark

        report = await run_dialogue_pe_eta_ablation_benchmark(
            cases=cases,
            profile_labels=profile_labels,
            baseline_label="pe-eta",
            runner_factory=_synthetic_runner if synthetic_runner else None,
        )
    else:
        report = await run_phase2_shadow_evidence_smoke(
            cases=cases,
            runner_factory=_synthetic_runner if synthetic_runner else None,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    per_path_metric_means = {
        path.path_label: dict(path.benchmark_report.metric_means)
        for path in report.path_reports
    }
    metric_deltas_from_baseline = {
        path_label: dict(items)
        for path_label, items in report.metric_deltas_from_baseline
    }
    expensive_snapshot = build_deterministic_head_to_head_snapshot(
        generation_id="phase2-shadow-smoke",
        baseline_label=report.baseline_label,
        per_profile_metric_means=per_path_metric_means,
        metric_names=_FOCUS_METRICS,
        lower_is_better=(
            "mean_persona_geometry_drift",
            "mean_least_control_effort",
        ),
    )
    cross_generation_snapshot = build_cross_generation_aggregate_snapshot(
        expensive_snapshot=expensive_snapshot,
    )
    payload: dict[str, object] = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "phase2_shadow_evidence_smoke",
        "runner_kind": "synthetic" if synthetic_runner else "default",
        "include_phase3_combos": include_phase3_combos,
        "provenance": _collect_provenance(),
        "baseline_label": report.baseline_label,
        "profile_labels": list(profile_labels),
        "case_ids": [case.case_id for case in cases],
        "focus_metrics": list(_FOCUS_METRICS),
        "focus_metric_means": {
            label: _focus_values(means)
            for label, means in per_path_metric_means.items()
        },
        "focus_metric_deltas_from_baseline": {
            label: _focus_values(deltas)
            for label, deltas in metric_deltas_from_baseline.items()
        },
        "per_path_metric_means": per_path_metric_means,
        "metric_deltas_from_baseline": metric_deltas_from_baseline,
        "head_to_head_results": [
            {
                "profile_a": item.profile_a,
                "profile_b": item.profile_b,
                "case_count": item.case_count,
                "winrate_a_vs_b": item.winrate_a_vs_b,
                "judge_kind": item.judge_kind,
                "notes": item.notes,
            }
            for item in expensive_snapshot.head_to_head_results
        ],
        "cross_generation_gate_evidence": {
            "evidence_id": cross_generation_snapshot.modification_gate_evidence.evidence_id,
            "validation_score": cross_generation_snapshot.modification_gate_evidence.validation_score,
            "head_to_head_aggregate_winrate": (
                cross_generation_snapshot.modification_gate_evidence.head_to_head_aggregate_winrate
            ),
            "rollback_evidence_present": (
                cross_generation_snapshot.modification_gate_evidence.rollback_evidence_present
            ),
            "capacity_within_cap": cross_generation_snapshot.modification_gate_evidence.capacity_within_cap,
            "audit_evidence_id": cross_generation_snapshot.modification_gate_evidence.audit_evidence_id,
            "notes": list(cross_generation_snapshot.modification_gate_evidence.notes),
        },
        "description": report.description,
    }

    artifact_path = output_dir / "phase2_shadow_evidence_smoke.json"
    artifact_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    brief_path = output_dir / "phase2_shadow_evidence_smoke.md"
    brief_path.write_text(
        _build_markdown_brief(payload=payload, artifact_path=artifact_path),
        encoding="utf-8",
    )
    manifest_path = output_dir / "phase2_shadow_evidence_manifest.json"
    manifest_payload: dict[str, object] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "phase2_shadow_evidence_manifest",
        "source_schema_version": _ARTIFACT_SCHEMA_VERSION,
        "artifacts": (
            _file_record(artifact_path),
            _file_record(brief_path),
        ),
        "provenance": payload["provenance"],
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _verify_written_manifest(manifest_path)

    print("[phase2-shadow] focus metric means:")
    for label, values in payload["focus_metric_means"].items():  # type: ignore[union-attr]
        if values:
            formatted = ", ".join(f"{key}={value:.4f}" for key, value in values.items())
        else:
            formatted = "(no focus metrics)"
        print(f"  {label:>32}: {formatted}")
    print(
        "[phase2-shadow] cross-generation aggregate winrate="
        f"{cross_generation_snapshot.modification_gate_evidence.head_to_head_aggregate_winrate:.4f}"
    )
    print(f"[phase2-shadow] evidence written to {artifact_path}")
    print(f"[phase2-shadow] brief written to {brief_path}")
    print(f"[phase2-shadow] manifest written to {manifest_path} (verified)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/phase2_shadow_evidence_smoke"),
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=1,
        help="Number of scripted proof cases to run per profile (default: 1).",
    )
    parser.add_argument(
        "--synthetic-runner",
        action="store_true",
        help="Use SyntheticOpenWeightResidualRuntime for a fast no-model smoke.",
    )
    parser.add_argument(
        "--include-phase3-combos",
        action="store_true",
        help="Also include Phase 3 combination SHADOW profiles.",
    )
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            main(
                output_dir=args.output_dir,
                case_limit=args.case_limit,
                synthetic_runner=args.synthetic_runner,
                include_phase3_combos=args.include_phase3_combos,
            )
        )
    )
