"""Run the preregistered Branch-B faithful ETA directional screen."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers

from companion_test_plan_common import (
    MPSAvailability,
    exclusive_mps_lock,
    mps_payload,
    require_mps,
)
from preregister_eta_faithful_rewrite_screen import (
    FROZEN_SOURCE_FILES,
    MODEL_SOURCE,
    PREREGISTRATION_SCHEMA_VERSION,
    build_preregistration,
)
from volvence_zero.agent.eta_faithful_rewrite_screen import (
    FaithfulETAScreenPoint,
    FaithfulETAScreenReport,
    FaithfulETAScreenThresholds,
    run_eta_faithful_rewrite_screen,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V4,
    eta_stage2_probe_rows,
)
from volvence_zero.substrate import (
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-faithful-rewrite-directional-screen-mps.v1"


class PreregistrationMismatch(RuntimeError):
    """Frozen Branch-B protocol disagrees with the requested execution."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _expected_preregistration(args: argparse.Namespace) -> dict[str, object]:
    return build_preregistration(
        device=args.device,
        max_length=args.max_length,
        alpha_grid=tuple(args.alpha_grid),
        primary_alpha=args.primary_alpha,
        seed_schedule=tuple(args.seed_schedule),
        updates_per_run=args.updates_per_run,
        learning_rate=args.learning_rate,
        switch_threshold=args.switch_threshold,
        n_z=args.n_z,
        injection_layer_index=args.injection_layer_index,
        residual_width=args.residual_width,
        steering_rank=args.steering_rank,
        control_norm_ratio=args.control_norm_ratio,
        screen_train_routes=args.screen_train_routes,
        screen_heldout_routes=args.screen_heldout_routes,
        corpus_seed=args.corpus_seed,
        objective_count=args.objective_count,
        corridor_count=args.corridor_count,
        extra_edge_probability=args.extra_edge_probability,
        train_routes=args.train_routes,
        heldout_routes=args.heldout_routes,
        train_lengths=tuple(args.train_lengths),
        heldout_lengths=tuple(args.heldout_lengths),
    )


def _validate_preregistration(
    payload: object,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise PreregistrationMismatch("preregistration root must be an object")
    if payload.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        raise PreregistrationMismatch("preregistration schema_version mismatch")
    expected = _expected_preregistration(args)
    for key in (
        "experiment_id",
        "claim_scope",
        "source_lineage",
        "configuration",
        "thresholds",
        "decision_rules",
        "prohibited_after_execution",
        "frozen_source_files",
    ):
        if payload.get(key) != expected.get(key):
            raise PreregistrationMismatch(
                f"preregistered faithful ETA field drifted: {key}"
            )
    return payload


class _PointCache:
    def __init__(
        self,
        *,
        root: Path,
        preregistration_sha256: str,
        source_files: dict[str, str],
    ) -> None:
        self._root = root
        self._preregistration_sha256 = preregistration_sha256
        self._source_files = source_files

    @staticmethod
    def _alpha_key(alpha: float) -> str:
        return format(alpha, ".12g").replace("-", "m").replace(".", "p")

    def _path(self, *, alpha: float, seed: int) -> Path:
        return (
            self._root
            / f"alpha-{self._alpha_key(alpha)}"
            / f"seed-{seed}.json"
        )

    def load_point(
        self, *, alpha: float, seed: int
    ) -> FaithfulETAScreenPoint | None:
        path = self._path(alpha=alpha, seed=seed)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("preregistration_sha256") != self._preregistration_sha256:
            raise RuntimeError(f"Faithful ETA checkpoint prereg drift: {path}")
        if payload.get("source_files") != self._source_files:
            raise RuntimeError(f"Faithful ETA checkpoint source drift: {path}")
        point = FaithfulETAScreenPoint(**payload["point"])
        if point.alpha != alpha or point.seed != seed:
            raise RuntimeError(f"Faithful ETA checkpoint identity drift: {path}")
        return point

    def store_point(self, point: FaithfulETAScreenPoint) -> None:
        path = self._path(alpha=point.alpha, seed=point.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "eta-faithful-rewrite-screen-checkpoint.v1",
            "preregistration_sha256": self._preregistration_sha256,
            "source_files": self._source_files,
            "point": asdict(point),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise FileExistsError(
                    f"Faithful ETA checkpoint already differs: {path}"
                )
            return
        temporary = path.with_suffix(".tmp")
        _write_json(temporary, payload)
        temporary.replace(path)


def _report_markdown(report: FaithfulETAScreenReport) -> str:
    admission = report.admission
    lines = [
        "# ETA 忠实重实现 directional screen",
        "",
        "> 新 claim：不改写已封存的 Stage-3 `kill-eta`；通过只准入另立预注册的权威扫。",
        "",
        "## 结论",
        "",
        "- authoritative-sweep admission："
        f"`{'PASS' if admission.admitted_for_authoritative_sweep else 'FAIL'}`",
        f"- failed conditions：`{admission.failed_conditions}`",
        f"- alpha-rate Spearman：`{report.alpha_rate_spearman:.4f}`",
        f"- rate span：`{report.rate_span:.4f}`",
        "",
        "## 预注册曲线与因果对照",
        "",
        "| alpha | rate | heldout NLL | zero-z penalty | permuted-z penalty | "
        "oracle F1 | boundary p contrast |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.aggregates:
        lines.append(
            f"| {row.alpha:g} | {row.train_rate_mean:.4f} | "
            f"{row.heldout_distortion_mean:.4f} | "
            f"{row.zero_z_penalty_mean:.4f} | "
            f"{row.permuted_z_penalty_mean:.4f} | "
            f"{row.oracle_boundary_f1_mean:.4f} | "
            f"{row.boundary_probability_contrast_mean:.4f} |"
        )
    lines.extend(
        (
            "",
            "## 忠实度 attestation",
            "",
            f"- input：layer {report.injection_layer_index}, full width "
            f"{report.residual_width} → learned projection → z{report.n_z}",
            f"- steering：rank-{report.steering_rank} "
            "`A·diag(tanh(Cz))·Bᵀ·e`，free bias=false",
            f"- token audit：max {report.max_observed_source_tokens} / "
            f"{report.scorer_max_length}，truncated={report.truncated_row_count}",
            "- active_subgoal boundary 只用于 evaluation；substrate frozen；"
            "production wiring / evaluation feedback 均未改变。",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered faithful ETA rewrite screen."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    parser.add_argument("--model-source", default=MODEL_SOURCE)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--alpha-grid", type=float, nargs="+", default=(0.03, 0.30, 3.00)
    )
    parser.add_argument("--primary-alpha", type=float, default=0.30)
    parser.add_argument("--seed-schedule", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--updates-per-run", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--switch-threshold", type=float, default=0.55)
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--injection-layer-index", type=int, default=20)
    parser.add_argument("--residual-width", type=int, default=896)
    parser.add_argument("--steering-rank", type=int, default=8)
    parser.add_argument("--control-norm-ratio", type=float, default=0.25)
    parser.add_argument("--screen-train-routes", type=int, default=16)
    parser.add_argument("--screen-heldout-routes", type=int, default=8)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    args = parser.parse_args(argv)

    preregistration = _validate_preregistration(
        json.loads(args.preregistration.read_text(encoding="utf-8")),
        args=args,
    )
    preregistration_sha256 = _sha256(args.preregistration)
    source_files = {
        name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
    }
    model_root = (_REPO_ROOT / args.model_source).resolve()
    model_weights_sha256 = fingerprint_model_weight_files(model_root)
    expected_weights = preregistration["configuration"]["model_weights_sha256"]
    if model_weights_sha256 != expected_weights:
        raise RuntimeError("Faithful ETA model weights drifted from preregistration.")

    output_dir: Path = args.output_dir
    protected_names = (
        "points.jsonl",
        "report.json",
        "report.md",
        "artifact_manifest.json",
    )
    existing = tuple(name for name in protected_names if (output_dir / name).exists())
    if existing:
        raise FileExistsError(
            "Faithful ETA output directory already contains final results: "
            f"{existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = generate_eta_proof_corpus(
        seed=args.corpus_seed,
        objective_count=args.objective_count,
        corridor_count=args.corridor_count,
        extra_edge_probability=args.extra_edge_probability,
        train_route_count=args.train_routes,
        heldout_route_count=args.heldout_routes,
        train_lengths=tuple(args.train_lengths),
        heldout_lengths=tuple(args.heldout_lengths),
    )
    train_probe_rows, train_classes = eta_stage2_probe_rows(
        corpus.train_cases[: args.screen_train_routes],
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    heldout_probe_rows, heldout_classes = eta_stage2_probe_rows(
        corpus.heldout_cases[: args.screen_heldout_routes],
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    if train_classes != heldout_classes:
        raise RuntimeError("Faithful ETA token audit saw class vocabulary drift.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_root),
        local_files_only=True,
    )
    audited_texts = tuple(
        row.observation_text + "\nNext move:"
        for row in (*train_probe_rows, *heldout_probe_rows)
    )
    token_lengths = tuple(
        len(
            tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        for text in audited_texts
    )
    max_observed_source_tokens = max(token_lengths)
    over_budget = sum(length > args.max_length for length in token_lengths)
    if over_budget:
        raise ValueError(
            f"Faithful ETA token audit found {over_budget} rows above "
            f"max_length={args.max_length}; max={max_observed_source_tokens}."
        )

    point_cache = _PointCache(
        root=output_dir / "checkpoints" / "points",
        preregistration_sha256=preregistration_sha256,
        source_files=source_files,
    )
    uses_mps = args.device.startswith("mps")
    started = time.perf_counter()
    with ExitStack() as stack:
        mps: MPSAvailability | None = None
        if uses_mps:
            stack.enter_context(exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID))
            mps = require_mps()
        runtime = TransformersOpenWeightResidualRuntime(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            pretrained_source=str(model_root),
            device=args.device,
            max_length=args.max_length,
            fail_on_truncation=True,
            layer_indices=(args.injection_layer_index,),
            activation_width=args.residual_width,
            local_files_only=True,
            runtime_origin="hf-local",
            allow_live_substrate_mutation=False,
            allow_offline_substrate_training=False,
            model_dtype="float32",
        )
        report = run_eta_faithful_rewrite_screen(
            corpus=corpus,
            runtime=runtime,
            model_source=args.model_source,
            device=args.device,
            screen_train_route_count=args.screen_train_routes,
            screen_heldout_route_count=args.screen_heldout_routes,
            alpha_grid=tuple(args.alpha_grid),
            primary_alpha=args.primary_alpha,
            seed_schedule=tuple(args.seed_schedule),
            updates_per_run=args.updates_per_run,
            learning_rate=args.learning_rate,
            switch_threshold=args.switch_threshold,
            n_z=args.n_z,
            injection_layer_index=args.injection_layer_index,
            residual_width=args.residual_width,
            steering_rank=args.steering_rank,
            control_norm_ratio=args.control_norm_ratio,
            scorer_max_length=args.max_length,
            max_observed_source_tokens=max_observed_source_tokens,
            point_cache=point_cache,
            thresholds=FaithfulETAScreenThresholds(),
            progress=lambda message: print(message, flush=True),
        )
        elapsed = time.perf_counter() - started

    _write_json(output_dir / "report.json", asdict(report))
    with (output_dir / "points.jsonl").open("w", encoding="utf-8") as handle:
        for point in report.points:
            handle.write(json.dumps(asdict(point), ensure_ascii=False) + "\n")
    (output_dir / "report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    result_files = ("report.json", "points.jsonl", "report.md")
    manifest = {
        "schema_version": "eta-faithful-rewrite-screen-manifest.v1",
        "experiment_id": "eta-faithful-rewrite-directional-screen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report.claim_scope,
        "source_lineage": preregistration["source_lineage"],
        "preregistered": True,
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": preregistration_sha256,
        "screen_adjudication_authoritative": True,
        "authoritative_sweep_admitted": (
            report.admission.admitted_for_authoritative_sweep
        ),
        "sealed_stage3_verdict_changed": False,
        "production_promotion_authorized": False,
        "artifact_installed": False,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
        "free_bias_present": report.free_bias_present,
        "substrate_trainable_parameter_count": (
            report.substrate_trainable_parameter_count
        ),
        "max_observed_source_tokens": max_observed_source_tokens,
        "truncated_row_count": report.truncated_row_count,
        "completed_cell_count": len(report.points),
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_exclusive_lock": str(args.mps_lock) if uses_mps else "not-required",
        "mps_attestation": mps_payload(mps) if mps is not None else "not-required",
        "model_weights_sha256": model_weights_sha256,
        "source_files": source_files,
        "result_files": {
            name: _sha256(output_dir / name) for name in result_files
        },
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    primary = next(
        row for row in report.aggregates if row.alpha == report.primary_alpha
    )
    print(
        json.dumps(
            {
                "authoritative_sweep_admitted": (
                    report.admission.admitted_for_authoritative_sweep
                ),
                "failed_conditions": report.admission.failed_conditions,
                "alpha_rate_spearman": report.alpha_rate_spearman,
                "rate_span": report.rate_span,
                "primary_alpha": report.primary_alpha,
                "primary_zero_z_penalty": primary.zero_z_penalty_mean,
                "primary_permuted_z_penalty": (
                    primary.permuted_z_penalty_mean
                ),
                "primary_oracle_boundary_f1": (
                    primary.oracle_boundary_f1_mean
                ),
                "primary_boundary_probability_contrast": (
                    primary.boundary_probability_contrast_mean
                ),
                "elapsed_seconds": round(elapsed, 1),
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
