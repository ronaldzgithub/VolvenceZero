"""Multi-seed wrapper for Phase 2/3 shadow evidence smoke.

Runs the existing Phase 2 shadow evidence smoke multiple times and aggregates
focus metrics into mean / std / stderr summaries. The current scripted runner
is mostly deterministic, but this wrapper fixes the evidence artifact shape
needed by Phase D decision reports.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from volvence_zero.evaluation import (
    build_cross_generation_aggregate_snapshot,
    build_deterministic_head_to_head_snapshot,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_phase2_shadow_evidence_smoke import (
    _FOCUS_METRICS,
    _collect_provenance,
    _file_record,
    _synthetic_runner,
)

from volvence_zero.agent import (
    DEFAULT_DIALOGUE_PROOF_CASES,
    default_phase2_shadow_evidence_profiles,
    default_phase3_combination_shadow_profiles,
    run_dialogue_pe_eta_ablation_benchmark,
    run_phase2_shadow_evidence_smoke,
)


_MULTISEED_SCHEMA_VERSION = "phase2-shadow-evidence-multiseed.v1"
_MULTISEED_MANIFEST_SCHEMA_VERSION = "phase2-shadow-evidence-multiseed-manifest.v1"


def _summary(values: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "stderr": 0.0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        std = 0.0
    else:
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    stderr = std / math.sqrt(len(values)) if values else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "stderr": round(stderr, 4),
    }


async def _run_once(
    *,
    seed: int,
    case_limit: int,
    synthetic_runner: bool,
    include_phase3_combos: bool,
) -> dict[str, Any]:
    del seed  # reserved for future stochastic runners
    cases = DEFAULT_DIALOGUE_PROOF_CASES[:case_limit]
    if include_phase3_combos:
        profile_labels = ("pe-eta",) + default_phase2_shadow_evidence_profiles() + default_phase3_combination_shadow_profiles()
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
    return {
        "baseline_label": report.baseline_label,
        "case_ids": [case.case_id for case in cases],
        "per_path_metric_means": {
            path.path_label: dict(path.benchmark_report.metric_means)
            for path in report.path_reports
        },
        "description": report.description,
    }


def _aggregate_runs(*, runs: tuple[dict[str, Any], ...]) -> dict[str, dict[str, dict[str, float]]]:
    profiles = sorted({profile for run in runs for profile in run["per_path_metric_means"]})
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for profile in profiles:
        profile_summary: dict[str, dict[str, float]] = {}
        for metric in _FOCUS_METRICS:
            values = tuple(
                float(run["per_path_metric_means"].get(profile, {}).get(metric))
                for run in runs
                if metric in run["per_path_metric_means"].get(profile, {})
            )
            if values:
                profile_summary[metric] = _summary(values)
        aggregate[profile] = profile_summary
    return aggregate


def _mean_metric_means(aggregate: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        profile: {
            metric: summary["mean"]
            for metric, summary in metrics.items()
        }
        for profile, metrics in aggregate.items()
    }


def _write_manifest(*, output_dir: Path, artifact_path: Path, brief_path: Path) -> Path:
    manifest_path = output_dir / "phase2_shadow_evidence_multiseed_manifest.json"
    manifest = {
        "schema_version": _MULTISEED_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "phase2_shadow_evidence_multiseed_manifest",
        "source_schema_version": _MULTISEED_SCHEMA_VERSION,
        "artifacts": (
            _file_record(artifact_path),
            _file_record(brief_path),
        ),
        "provenance": _collect_provenance(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # Reuse single-smoke verifier semantics by checking records locally.
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in payload["artifacts"]:
        path = Path(item["path"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"manifest sha256 mismatch for {path}")
        if len(data) != item["size_bytes"]:
            raise ValueError(f"manifest size mismatch for {path}")
    return manifest_path


def _build_brief(*, payload: dict[str, Any], artifact_path: Path) -> str:
    rows = []
    for profile, metrics in payload["profile_metric_summaries"].items():
        if metrics:
            compact = "<br>".join(
                f"`{metric}` mean={summary['mean']:.4f} std={summary['std']:.4f}"
                for metric, summary in metrics.items()
            )
        else:
            compact = "no focus metrics"
        rows.append(f"| `{profile}` | {compact} |")
    return f"""# Phase 2/3 Shadow Evidence Multi-Seed

> Generated by `scripts/run_phase2_shadow_evidence_multiseed.py`.

## Run Shape

- Schema: `{payload['schema_version']}`
- Baseline: `{payload['baseline_label']}`
- Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`
- Cases: `{', '.join(payload['case_ids'])}`
- Runner kind: `{payload['runner_kind']}`
- Includes Phase 3 combos: `{payload['include_phase3_combos']}`
- JSON artifact: `{artifact_path}`

## Focus Metric Summaries

| Profile | Focus metric summaries |
|---|---|
{chr(10).join(rows)}

## Cross-Generation Gate Evidence

- Validation score: `{payload['cross_generation_gate_evidence']['validation_score']:.4f}`
- Head-to-head aggregate winrate: `{payload['cross_generation_gate_evidence']['head_to_head_aggregate_winrate']:.4f}`

This is SHADOW evidence. It does not make an ACTIVE decision.
"""


async def main(
    *,
    output_dir: Path,
    seeds: tuple[int, ...],
    case_limit: int,
    synthetic_runner: bool,
    include_phase3_combos: bool,
) -> int:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if case_limit <= 0:
        raise ValueError(f"case_limit must be > 0, got {case_limit!r}")
    runs = tuple(
        [
            await _run_once(
                seed=seed,
                case_limit=case_limit,
                synthetic_runner=synthetic_runner,
                include_phase3_combos=include_phase3_combos,
            )
            for seed in seeds
        ]
    )
    aggregate = _aggregate_runs(runs=runs)
    mean_metric_means = _mean_metric_means(aggregate)
    expensive = build_deterministic_head_to_head_snapshot(
        generation_id="phase2-shadow-multiseed",
        baseline_label="pe-eta",
        per_profile_metric_means=mean_metric_means,
        metric_names=_FOCUS_METRICS,
        lower_is_better=("mean_persona_geometry_drift", "mean_least_control_effort"),
    )
    cross_generation = build_cross_generation_aggregate_snapshot(expensive_snapshot=expensive)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": _MULTISEED_SCHEMA_VERSION,
        "artifact_kind": "phase2_shadow_evidence_multiseed",
        "runner_kind": "synthetic" if synthetic_runner else "default",
        "include_phase3_combos": include_phase3_combos,
        "baseline_label": "pe-eta",
        "seeds": list(seeds),
        "case_ids": runs[0]["case_ids"],
        "profile_metric_summaries": aggregate,
        "mean_metric_means": mean_metric_means,
        "head_to_head_results": [
            {
                "profile_a": item.profile_a,
                "profile_b": item.profile_b,
                "case_count": item.case_count,
                "winrate_a_vs_b": item.winrate_a_vs_b,
                "judge_kind": item.judge_kind,
                "notes": item.notes,
            }
            for item in expensive.head_to_head_results
        ],
        "cross_generation_gate_evidence": {
            "evidence_id": cross_generation.modification_gate_evidence.evidence_id,
            "validation_score": cross_generation.modification_gate_evidence.validation_score,
            "head_to_head_aggregate_winrate": cross_generation.modification_gate_evidence.head_to_head_aggregate_winrate,
            "audit_evidence_id": cross_generation.modification_gate_evidence.audit_evidence_id,
            "notes": list(cross_generation.modification_gate_evidence.notes),
        },
        "provenance": _collect_provenance(),
    }
    artifact_path = output_dir / "phase2_shadow_evidence_multiseed.json"
    artifact_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    brief_path = output_dir / "phase2_shadow_evidence_multiseed.md"
    brief_path.write_text(_build_brief(payload=payload, artifact_path=artifact_path), encoding="utf-8")
    manifest_path = _write_manifest(output_dir=output_dir, artifact_path=artifact_path, brief_path=brief_path)
    print(f"[phase2-multiseed] wrote {artifact_path}")
    print(f"[phase2-multiseed] wrote {brief_path}")
    print(f"[phase2-multiseed] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2_shadow_evidence_multiseed"))
    parser.add_argument("--case-limit", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--synthetic-runner", action="store_true")
    parser.add_argument("--include-phase3-combos", action="store_true")
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            main(
                output_dir=args.output_dir,
                seeds=tuple(args.seeds),
                case_limit=args.case_limit,
                synthetic_runner=args.synthetic_runner,
                include_phase3_combos=args.include_phase3_combos,
            )
        )
    )
