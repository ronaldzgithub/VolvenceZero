"""Run the ETA rate-distortion criterion sweep and export the artifact bundle.

Two arms (frozen substrate / joint-training validity control) x an alpha
grid x seeds. Produces raw sweep points, per-arm aggregate curves, gap
assessments, the retain/kill/inconclusive verdict, and a provenance manifest
with source SHAs.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import inspect
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

from companion_test_plan_common import (
    MPSAvailability,
    exclusive_mps_lock,
    mps_payload,
    require_mps,
)
from msc_prediction_checkpoint import PredictionRunCheckpointStore
from preregister_eta_rate_distortion import (
    PREREGISTRATION_SCHEMA_VERSION,
    VERDICT_SET,
)
from volvence_zero.agent.eta_proof_benchmark import ETAOpenWeightRuntimeConfig
from volvence_zero.agent.eta_rate_distortion_evidence import (
    RateDistortionEvidenceReport,
    RateDistortionPoint,
    assess_gap,
    run_eta_rate_distortion_evidence,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

PLAN_ID = "eta-rate-distortion-mps.v1"
CHECKPOINT_SCHEMA_NAMESPACE = "eta-rate-distortion"


class PreregistrationMismatch(RuntimeError):
    """Raised when the frozen protocol disagrees with what would be run."""


class RateDistortionCheckpointCache:
    """Per-cell resume journal for the sweep.

    Each ``(arm, alpha, seed)`` cell is a separate immutable unit, so an
    interrupted sweep resumes at cell granularity instead of discarding hours
    of accelerator time.
    """

    def __init__(self, store: PredictionRunCheckpointStore) -> None:
        self._store = store

    @staticmethod
    def _unit(*, arm: str, alpha: float, seed: int) -> str:
        return f"points/{arm}/alpha-{alpha:g}/seed-{seed}"

    def _relative_path(self, *, arm: str, alpha: float, seed: int) -> str:
        return f"checkpoints/{self._unit(arm=arm, alpha=alpha, seed=seed)}.json"

    def load_point(
        self, *, arm: str, alpha: float, seed: int
    ) -> RateDistortionPoint | None:
        payload = self._store.load_json(
            unit=self._unit(arm=arm, alpha=alpha, seed=seed),
            relative_path=self._relative_path(arm=arm, alpha=alpha, seed=seed),
        )
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError(
                f"rate-distortion checkpoint for {arm}/{alpha:g}/{seed} is not "
                "an object"
            )
        return RateDistortionPoint(**payload)

    def store_point(self, point: RateDistortionPoint) -> None:
        self._store.save_json(
            unit=self._unit(
                arm=point.arm, alpha=point.alpha, seed=point.seed
            ),
            relative_path=self._relative_path(
                arm=point.arm, alpha=point.alpha, seed=point.seed
            ),
            payload=asdict(point),
        )


def _validate_preregistration(
    payload: object, *, args: argparse.Namespace
) -> dict[str, object]:
    """Fail closed unless the frozen protocol matches this exact execution."""

    if not isinstance(payload, dict):
        raise PreregistrationMismatch("preregistration root must be an object")
    if payload.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        raise PreregistrationMismatch(
            "preregistration schema_version must be "
            f"{PREREGISTRATION_SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    sweep = payload.get("sweep")
    if not isinstance(sweep, dict):
        raise PreregistrationMismatch("preregistration sweep must be an object")
    requested = {
        "alpha_grid": list(args.alphas),
        "seed_schedule": list(range(args.seeds)),
        "n_z": args.n_z,
        "updates_per_run": args.updates,
        "learning_rate": args.learning_rate,
        "substrate_learning_rate": args.substrate_learning_rate,
        "switch_threshold": args.switch_threshold,
        "arms": list(args.arms),
        "model_id": args.model_id,
        "device": args.device,
    }
    for key, value in requested.items():
        if sweep.get(key) != value:
            raise PreregistrationMismatch(
                f"preregistered {key}={sweep.get(key)!r} does not match the "
                f"requested {key}={value!r}"
            )

    thresholds = payload.get("gap_thresholds")
    if not isinstance(thresholds, dict):
        raise PreregistrationMismatch(
            "preregistration gap_thresholds must be an object"
        )
    defaults = inspect.signature(assess_gap).parameters
    for name in (
        "drop_share_threshold",
        "rate_share_threshold",
        "noise_multiple",
    ):
        if thresholds.get(name) != defaults[name].default:
            raise PreregistrationMismatch(
                f"preregistered {name}={thresholds.get(name)!r} does not match "
                f"the executing default {defaults[name].default!r}"
            )

    frozen_sources = payload.get("frozen_source_files")
    if not isinstance(frozen_sources, dict) or not frozen_sources:
        raise PreregistrationMismatch(
            "preregistration must freeze at least one source file"
        )
    drifted = {
        name: {"preregistered": expected, "current": _sha256(_REPO_ROOT / name)}
        for name, expected in frozen_sources.items()
        if _sha256(_REPO_ROOT / name) != expected
    }
    if drifted:
        raise PreregistrationMismatch(
            "source drift since preregistration; a mid-campaign source change "
            f"invalidates the run: {json.dumps(drifted, sort_keys=True)}"
        )
    return payload

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


def _report_markdown(
    report: RateDistortionEvidenceReport, *, preregistered: bool
) -> str:
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
        (
            "This verdict is authoritative under the frozen protocol."
            if preregistered
            else "**Not preregistered — mechanism-only smoke.** This verdict "
            "is not authoritative and must not be cited as evidence."
        ),
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
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
        help=(
            "Shared lock preventing this sweep from running while another "
            "companion evidence plan already owns the MPS device."
        ),
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=None,
        help=(
            "Frozen protocol from preregister_eta_rate_distortion.py. Without "
            "it the run is stamped mechanism-only-smoke and its verdict is "
            "not authoritative."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse an existing output directory's per-cell checkpoints. "
            "Without it an existing output directory is refused."
        ),
    )
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
    preregistration: dict[str, object] | None = None
    preregistration_sha256 = ""
    if args.preregistration is not None:
        preregistration = _validate_preregistration(
            json.loads(args.preregistration.read_text(encoding="utf-8")),
            args=args,
        )
        preregistration_sha256 = _sha256(args.preregistration)
    config = ETAOpenWeightRuntimeConfig(
        model_id=args.model_id,
        device=args.device,
    )
    uses_mps = args.device.startswith("mps")
    with ExitStack() as stack:
        mps: MPSAvailability | None = None
        if uses_mps:
            # Two MPS evidence stages must never share the device: the memory
            # contention silently distorts every wall-clock readout.
            stack.enter_context(
                exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID)
            )
            mps = require_mps()
        # Created only once the device is actually ours, so a failed preflight
        # leaves no half-made journal that a later run would have to --resume.
        checkpoint_store = PredictionRunCheckpointStore(
            output_dir=output_dir,
            configuration={
                "experiment_id": "eta-rate-distortion-criterion",
                "alpha_grid": list(args.alphas),
                "seed_schedule": list(range(args.seeds)),
                "n_z": args.n_z,
                "updates_per_run": args.updates,
                "learning_rate": args.learning_rate,
                "substrate_learning_rate": args.substrate_learning_rate,
                "switch_threshold": args.switch_threshold,
                "arms": list(args.arms),
                "model_id": args.model_id,
                "device": args.device,
                "preregistration_sha256": preregistration_sha256,
                "source_sha256": {
                    name: _sha256(_REPO_ROOT / name)
                    for name in _KEY_SOURCE_FILES
                },
            },
            resume=args.resume,
            schema_namespace=CHECKPOINT_SCHEMA_NAMESPACE,
        )
        started = time.perf_counter()
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
            point_cache=RateDistortionCheckpointCache(checkpoint_store),
        )
        elapsed = time.perf_counter() - started
    checkpoint_store.mark_complete()
    if report.verdict not in VERDICT_SET:
        raise RuntimeError(
            f"verdict {report.verdict!r} is outside the frozen verdict set "
            f"{VERDICT_SET}"
        )

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
        _report_markdown(report, preregistered=preregistration is not None),
        encoding="utf-8",
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
        "preregistered": preregistration is not None,
        "preregistration_path": (
            str(args.preregistration) if args.preregistration else ""
        ),
        "preregistration_sha256": preregistration_sha256,
        "verdict_authoritative": preregistration is not None,
        "claim_scope": (
            "eta-temporal-abstraction-criterion-only"
            if preregistration is not None
            else "mechanism-only-smoke"
        ),
        "resumed": bool(args.resume),
        "checkpoint_units": checkpoint_store.immutable_file_manifest(),
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
        "mps_exclusive_lock": str(args.mps_lock) if uses_mps else "not-required",
        "mps_attestation": mps_payload(mps) if mps is not None else "not-required",
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
