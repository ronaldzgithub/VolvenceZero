"""SHADOW smoke for the CMS ATLAS / Titans uplift.

Runs the canonical ``pe-eta`` baseline next to the new
``atlas-titans-cms-uplift`` profile on a small subset of scripted dialogue
proof cases, and reports per-case + aggregate metric deltas. Acceptance
is evidence-only at this stage (see docs/specs/cms-atlas-titans-uplift.md
§7-§8); this script is the lightest entry point on the validation
ladder, sized to run in ~tens of seconds rather than minutes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from volvence_zero.agent import (
    DEFAULT_DIALOGUE_PROOF_CASES,
    default_dialogue_atlas_titans_uplift_profiles,
    run_dialogue_pe_eta_ablation_benchmark,
)


def _format_delta_row(label: str, items: tuple[tuple[str, float], ...], focus: tuple[str, ...]) -> str:
    pairs = []
    lookup = dict(items)
    for key in focus:
        value = lookup.get(key)
        if value is None:
            continue
        sign = "+" if value >= 0 else ""
        pairs.append(f"{key}={sign}{value:.4f}")
    return f"{label:>32}: " + ", ".join(pairs) if pairs else f"{label:>32}: (no overlapping metrics)"


async def main(*, output_dir: Path, case_limit: int) -> int:
    cases = DEFAULT_DIALOGUE_PROOF_CASES[:case_limit]
    profile_labels = default_dialogue_atlas_titans_uplift_profiles()
    print(f"[shadow-smoke] profiles={profile_labels} cases={[c.case_id for c in cases]}")
    report = await run_dialogue_pe_eta_ablation_benchmark(
        cases=cases,
        profile_labels=profile_labels,
        baseline_label="pe-eta",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-path summary metrics.
    per_path_metric_means: dict[str, dict[str, float]] = {}
    for path in report.path_reports:
        per_path_metric_means[path.path_label] = dict(path.benchmark_report.metric_means)

    # Stable focus list of "interesting" metrics for the executive summary.
    focus_metrics = (
        "canonical_pass_rate",
        "memory_runtime_quality_mean",
        "memory_fast_alignment_mean",
        "memory_tower_profile_turn_count",
        "pe_drive_share_mean",
        "regime_evidence_rate",
        "delayed_credit_alignment_mean",
        "evaluation_quality_mean",
    )

    print()
    print("[shadow-smoke] per-profile metric means:")
    for path_label, metric_means in per_path_metric_means.items():
        focus_pairs = []
        for key in focus_metrics:
            value = metric_means.get(key)
            if value is None:
                continue
            focus_pairs.append(f"{key}={value:.4f}")
        if focus_pairs:
            print(f"  {path_label:>32}: " + ", ".join(focus_pairs))
        else:
            keys_sample = ", ".join(sorted(metric_means.keys())[:6])
            print(f"  {path_label:>32}: no focus metrics; available keys sample: {keys_sample} ...")

    print()
    print("[shadow-smoke] aggregate deltas vs pe-eta baseline:")
    for path_label, items in report.metric_deltas_from_baseline:
        print("  " + _format_delta_row(path_label, items, focus_metrics))

    # Write a compact JSON evidence dump that downstream evidence reviews can read.
    evidence: dict[str, object] = {
        "baseline_label": report.baseline_label,
        "profile_labels": list(profile_labels),
        "case_ids": [case.case_id for case in cases],
        "per_path_metric_means": {
            path_label: dict(metric_means)
            for path_label, metric_means in per_path_metric_means.items()
        },
        "metric_deltas_from_baseline": [
            {
                "path_label": path_label,
                "deltas": dict(items),
            }
            for path_label, items in report.metric_deltas_from_baseline
        ],
        "case_deltas_from_baseline": [
            {
                "case_id": case_id,
                "paths": {
                    path_label: dict(metric_items)
                    for path_label, metric_items in path_deltas
                },
            }
            for case_id, path_deltas in report.case_deltas_from_baseline
        ],
        "description": report.description,
    }
    artifact_path = output_dir / "atlas_titans_cms_shadow_smoke.json"
    artifact_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[shadow-smoke] evidence written to {artifact_path}")

    # Sanity: was the uplift actually exercised? Per-turn DialogueBenchmarkTurn
    # does not publish memory_snapshot, so we rely on metric divergence as the
    # behavioral fingerprint of the uplift path. If aggregate CMS-related
    # metrics differ between the two profiles, the uplift has changed CMS
    # state evolution; if they are identical, the flag wiring did not reach
    # the runtime path and we should investigate.
    cms_sensitive_metrics = (
        "mean_learned_recall_confidence",
        "mean_memory_tower_alignment",
        "mean_memory_updater_confidence",
        "mean_memory_updater_effective_lr",
        "carryover_credit_turn_count",
    )
    uplift_means = per_path_metric_means.get("atlas-titans-cms-uplift", {})
    canonical_means = per_path_metric_means.get("pe-eta", {})
    if not uplift_means or not canonical_means:
        print("[shadow-smoke] WARNING: profile reports missing.")
        return 1
    diverged = []
    for key in cms_sensitive_metrics:
        if key not in uplift_means or key not in canonical_means:
            continue
        if abs(uplift_means[key] - canonical_means[key]) > 1e-6:
            diverged.append(key)
    print(
        f"[shadow-smoke] CMS-sensitive metrics that diverged between profiles: "
        f"{len(diverged)} / {len(cms_sensitive_metrics)} -> {diverged}"
    )
    if not diverged:
        print(
            "[shadow-smoke] WARNING: uplift profile is numerically identical to "
            "canonical on CMS-sensitive metrics; the new path may not be active."
        )
        return 1
    print("[shadow-smoke] uplift path is active in runtime; SHADOW evidence captured.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cms_atlas_titans_shadow_smoke"),
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=2,
        help="Number of scripted proof cases to run per profile (smoke = 2).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(output_dir=args.output_dir, case_limit=args.case_limit)))
