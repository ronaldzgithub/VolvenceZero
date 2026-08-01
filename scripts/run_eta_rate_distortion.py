"""Run the ETA rate-distortion criterion sweep and export the artifact bundle.

Two arms (frozen substrate / joint-training validity control) x an alpha
grid x seeds. Produces raw sweep points, per-arm aggregate curves, gap
assessments, the retain/kill/inconclusive verdict, and a provenance manifest
with source SHAs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from volvence_zero.agent.eta_proof_benchmark import ETAOpenWeightRuntimeConfig
from volvence_zero.agent.eta_rate_distortion_evidence import (
    RateDistortionEvidenceReport,
    run_eta_rate_distortion_evidence,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

_KEY_SOURCE_FILES = (
    "packages/vz-substrate/src/volvence_zero/substrate/steered_action_scoring.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-temporal/src/volvence_zero/temporal/torch_store_ssl.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_rate_distortion_evidence.py",
    "scripts/run_eta_rate_distortion.py",
)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _maybe_plot(report: RateDistortionEvidenceReport, output_dir: Path) -> str:
    if importlib.util.find_spec("matplotlib") is None:
        return ""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (axis_rd, axis_switch) = plt.subplots(
        1, 2, figsize=(12, 5)
    )
    colors = {"frozen": "tab:blue", "joint": "tab:red"}
    for arm in report.arms:
        rows = sorted(
            (row for row in report.curves if row.arm == arm),
            key=lambda row: row.rate_mean,
        )
        rates = [row.rate_mean for row in rows]
        distortions = [row.distortion_mean for row in rows]
        errors = [row.distortion_std for row in rows]
        axis_rd.errorbar(
            rates,
            distortions,
            yerr=errors,
            marker="o",
            color=colors.get(arm, "tab:gray"),
            label=f"{arm} arm",
        )
        for row in rows:
            axis_rd.annotate(
                f"a={row.alpha:g}",
                (row.rate_mean, row.distortion_mean),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
        axis_switch.plot(
            [row.alpha for row in rows],
            [row.switch_frequency_mean for row in rows],
            marker="s",
            color=colors.get(arm, "tab:gray"),
            label=f"{arm} switch freq",
        )
    points = tuple(p for p in report.points if p.arm == "frozen")
    if points:
        axis_rd.axhline(
            sum(p.baseline_train_distortion for p in points) / len(points),
            linestyle="--",
            color="gray",
            linewidth=1,
            label="unsteered baseline",
        )
    axis_rd.set_xlabel("rate (mean per-dim KL)")
    axis_rd.set_ylabel("distortion (action NLL, train)")
    axis_rd.set_title(
        f"ETA rate-distortion — verdict: {report.verdict}"
    )
    axis_rd.legend(fontsize=8)
    axis_switch.set_xscale("log")
    axis_switch.set_xlabel("alpha")
    axis_switch.set_ylabel("hard switch frequency")
    axis_switch.set_title("switch frequency vs alpha")
    axis_switch.legend(fontsize=8)
    figure.tight_layout()
    plot_path = output_dir / "rate_distortion_curves.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path.name


def _report_markdown(report: RateDistortionEvidenceReport) -> str:
    lines = [
        "# ETA rate-distortion criterion",
        "",
        f"- schema: `{report.schema_version}`",
        f"- model: `{report.model_id}` on `{report.device}` "
        f"(origin `{report.runtime_origin}`, fallback={report.fallback_active})",
        f"- injection layer: {report.injection_layer_index}, "
        f"control norm cap: {report.control_norm_cap:.2f} "
        f"(probe hidden norm {report.probe_hidden_norm:.2f})",
        f"- n_z={report.n_z}, alpha grid={list(report.alpha_grid)}, "
        f"seeds={list(report.seed_schedule)}, "
        f"updates/run={report.updates_per_run}",
        f"- observation protocol: `{report.observation_protocol}`",
        f"- train steps={report.train_step_count}, "
        f"heldout steps={report.heldout_step_count}",
        "",
        f"## Verdict: `{report.verdict}`",
        "",
        report.verdict_reason,
        "",
        f"- arms distinguishable: {report.arms_distinguishable} "
        f"(max separation {report.arm_separation:.4f}, "
        f"threshold {report.arm_separation_threshold:.4f})",
        "",
        "## Gap assessments",
        "",
    ]
    for gap in report.gaps:
        lines.extend(
            [
                f"### {gap.arm} arm",
                "",
                f"- gap detected: **{gap.gap_detected}**",
                f"- distortion span {gap.distortion_span:.4f}, "
                f"rate span {gap.rate_span:.4f}, "
                f"noise scale {gap.noise_scale:.4f}",
                f"- max adjacent drop {gap.max_drop:.4f} "
                f"({gap.max_drop_share:.1%} of span) over "
                f"{gap.max_drop_rate_share:.1%} of the rate span, "
                f"between alpha={gap.gap_low_alpha:g} and "
                f"alpha={gap.gap_high_alpha:g}",
                f"- boundary F1 inside gap {gap.boundary_f1_gap_region:.3f} "
                f"vs outside {gap.boundary_f1_outside_gap:.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Aggregate curves (train distortion)",
            "",
            "| arm | alpha | rate | distortion | ±std | heldout d | "
            "boundary F1 | switch freq |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in sorted(
        report.curves, key=lambda r: (r.arm, r.rate_mean)
    ):
        lines.append(
            f"| {row.arm} | {row.alpha:g} | {row.rate_mean:.4f} | "
            f"{row.distortion_mean:.4f} | {row.distortion_std:.4f} | "
            f"{row.heldout_distortion_mean:.4f} | "
            f"{row.boundary_f1_mean:.3f} | "
            f"{row.switch_frequency_mean:.3f} |"
        )
    baselines = tuple(
        p.baseline_train_distortion
        for p in report.points
        if p.arm == "frozen"
    )
    if baselines:
        lines.extend(
            [
                "",
                f"Unsteered baseline distortion (train): "
                f"{sum(baselines) / len(baselines):.4f}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ETA rate-distortion criterion sweep."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--substrate-learning-rate", type=float, default=1e-4
    )
    parser.add_argument("--switch-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--model-id", default=ETAOpenWeightRuntimeConfig().model_id
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("frozen", "joint"),
        default=("frozen", "joint"),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.updates < 1:
        parser.error("--updates must be at least 1")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = ETAOpenWeightRuntimeConfig(
        model_id=args.model_id,
        device=args.device,
    )
    report = run_eta_rate_distortion_evidence(
        alpha_grid=tuple(args.alphas),
        seed_schedule=tuple(range(args.seeds)),
        n_z=args.n_z,
        updates_per_run=args.updates,
        learning_rate=args.learning_rate,
        substrate_learning_rate=args.substrate_learning_rate,
        switch_threshold=args.switch_threshold,
        open_weight_config=config,
        arms=tuple(args.arms),
    )
    elapsed = time.perf_counter() - started

    _write_json(output_dir / "report.json", asdict(report))
    points_path = output_dir / "points.jsonl"
    with points_path.open("w", encoding="utf-8") as handle:
        for point in report.points:
            handle.write(json.dumps(asdict(point), sort_keys=False) + "\n")
    _write_json(
        output_dir / "curves.json",
        [asdict(row) for row in report.curves],
    )
    _write_json(
        output_dir / "gap_assessments.json",
        [asdict(gap) for gap in report.gaps],
    )
    plot_name = _maybe_plot(report, output_dir)
    (output_dir / "report.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )

    result_files = [
        "report.json",
        "points.jsonl",
        "curves.json",
        "gap_assessments.json",
        "report.md",
    ]
    if plot_name:
        result_files.append(plot_name)
    manifest = {
        "schema_version": "eta-rate-distortion-manifest.v1",
        "experiment_id": "eta-rate-distortion-criterion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "verdict": report.verdict,
        "verdict_reason": report.verdict_reason,
        "arms": list(report.arms),
        "alpha_grid": list(report.alpha_grid),
        "seed_schedule": list(report.seed_schedule),
        "n_z": report.n_z,
        "updates_per_run": report.updates_per_run,
        "learning_rate": report.learning_rate,
        "substrate_learning_rate": report.substrate_learning_rate,
        "switch_threshold": report.switch_threshold,
        "model_id": report.model_id,
        "device": report.device,
        "runtime_origin": report.runtime_origin,
        "fallback_active": report.fallback_active,
        "injection_layer_index": report.injection_layer_index,
        "control_norm_cap": report.control_norm_cap,
        "observation_protocol": report.observation_protocol,
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "matplotlib_plot": plot_name or "unavailable",
        "result_files": {
            name: _sha256(output_dir / name) for name in result_files
        },
        "source_files": {
            name: _sha256(_REPO_ROOT / name) for name in _KEY_SOURCE_FILES
        },
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)

    print(
        json.dumps(
            {
                "verdict": report.verdict,
                "verdict_reason": report.verdict_reason,
                "arms_distinguishable": report.arms_distinguishable,
                "arm_separation": round(report.arm_separation, 4),
                "gaps": {
                    gap.arm: {
                        "detected": gap.gap_detected,
                        "max_drop_share": round(gap.max_drop_share, 3),
                        "max_drop_rate_share": round(
                            gap.max_drop_rate_share, 3
                        ),
                    }
                    for gap in report.gaps
                },
                "elapsed_seconds": round(elapsed, 1),
                "output_dir": str(output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
