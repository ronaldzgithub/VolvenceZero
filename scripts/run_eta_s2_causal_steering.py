"""Run the preregistered ETA S2 no-bias causal-steering experiment."""

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
from preregister_eta_s2_causal_steering import (
    FROZEN_SOURCE_FILES,
    PREREGISTRATION_SCHEMA_VERSION,
    SOURCE_S1_ARTIFACT,
    SOURCE_S1_MANIFEST,
    SOURCE_S1_REPORT,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V4,
    eta_stage2_probe_rows,
)
from volvence_zero.agent.eta_s2_causal_steering import (
    S2CausalSteeringReport,
    S2CausalSteeringThresholds,
    run_eta_s2_causal_steering,
)
from volvence_zero.substrate import (
    FrozenResidualReadoutArtifact,
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-s2-no-bias-causal-steering-mps.v1"


class PreregistrationMismatch(RuntimeError):
    """Frozen S2 protocol disagrees with the requested execution."""


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


def _requested_configuration(
    args: argparse.Namespace,
    *,
    artifact: FrozenResidualReadoutArtifact,
) -> dict[str, object]:
    return {
        "model_id": artifact.model_fingerprint.model_id,
        "model_source": args.model_source,
        "model_version": artifact.model_fingerprint.version,
        "model_weights_sha256": artifact.model_fingerprint.weights_sha256,
        "device": args.device,
        "model_dtype": "float32",
        "max_length": args.max_length,
        "token_budget_policy": "fail-loud-preflight.v1",
        "injection_layer_index": 20,
        "hidden_size": 896,
        "control_norm_ratio": args.control_norm_ratio,
        "scale_fractions": list(args.scale_fractions),
        "primary_scale_fraction": args.primary_scale_fraction,
        "shuffled_class_shifts": list(args.shuffled_class_shifts),
        "probe_train_rows": args.probe_train_rows,
        "batch_size": args.batch_size,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_confidence": args.bootstrap_confidence,
        "corpus": {
            "corpus_seed": args.corpus_seed,
            "objective_count": args.objective_count,
            "corridor_count": args.corridor_count,
            "extra_edge_probability": args.extra_edge_probability,
            "train_routes": args.train_routes,
            "heldout_routes": args.heldout_routes,
            "train_lengths": list(args.train_lengths),
            "heldout_lengths": list(args.heldout_lengths),
        },
        "observation_protocol": "partially-observable-staged-plan.v4",
        "evaluation_split": "heldout-only",
        "controls": [
            "target-plus-axis",
            "target-minus-axis",
            "noop",
            "mean-of-three-cyclic-shuffled-class-axes",
        ],
        "aggregation": "mean-within-route-then-bootstrap-routes",
    }


def _validate_preregistration(
    payload: object,
    *,
    args: argparse.Namespace,
    artifact: FrozenResidualReadoutArtifact,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise PreregistrationMismatch("preregistration root must be an object")
    if payload.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        raise PreregistrationMismatch("preregistration schema_version mismatch")
    if payload.get("claim_scope") != "s2-heldout-axis-causal-steering":
        raise PreregistrationMismatch("preregistration claim_scope mismatch")
    if payload.get("configuration") != _requested_configuration(
        args,
        artifact=artifact,
    ):
        raise PreregistrationMismatch(
            "preregistered configuration does not match requested execution"
        )
    if payload.get("thresholds") != asdict(S2CausalSteeringThresholds()):
        raise PreregistrationMismatch("preregistered S2 thresholds drifted")

    source = payload.get("source_s1")
    if not isinstance(source, dict):
        raise PreregistrationMismatch("source S1 lineage is missing")
    expected_source = {
        "report_path": SOURCE_S1_REPORT,
        "report_sha256": _sha256(_REPO_ROOT / SOURCE_S1_REPORT),
        "manifest_path": SOURCE_S1_MANIFEST,
        "manifest_sha256": _sha256(_REPO_ROOT / SOURCE_S1_MANIFEST),
        "artifact_path": SOURCE_S1_ARTIFACT,
        "artifact_file_sha256": _sha256(_REPO_ROOT / SOURCE_S1_ARTIFACT),
        "artifact_id": artifact.artifact_id,
        "admitted": True,
    }
    if source != expected_source:
        raise PreregistrationMismatch("source S1 artifact lineage drifted")
    report = json.loads((_REPO_ROOT / SOURCE_S1_REPORT).read_text(encoding="utf-8"))
    manifest = json.loads(
        (_REPO_ROOT / SOURCE_S1_MANIFEST).read_text(encoding="utf-8")
    )
    if report.get("admission", {}).get("admitted") is not True or (
        manifest.get("s1_admitted_for_s2") is not True
    ):
        raise PreregistrationMismatch("source S1 no longer admits S2 evidence")

    frozen = payload.get("frozen_source_files")
    if not isinstance(frozen, dict):
        raise PreregistrationMismatch("frozen_source_files is missing")
    for name in FROZEN_SOURCE_FILES:
        actual = _sha256(_REPO_ROOT / name)
        if frozen.get(name) != actual:
            raise PreregistrationMismatch(
                f"frozen source hash drift for {name}: "
                f"preregistered={frozen.get(name)!r}, actual={actual!r}"
            )
    return payload


def _report_markdown(report: S2CausalSteeringReport) -> str:
    admission = report.primary_admission
    lines = [
        "# ETA S2 无 bias 因果残差 steering",
        "",
        "> 范围：heldout cumulative-prefix 上的 matched causal evidence；"
        "不训练参数、不安装 artifact、不改 production WiringLevel。",
        "",
        "## 结论",
        "",
        f"- S2 primary admission：`{'PASS' if admission.admitted else 'FAIL'}`",
        f"- primary scale：`{report.primary_scale_fraction:.2f} × cap`",
        f"- failed conditions：`{admission.failed_conditions}`",
        "",
        "## 剂量与对照",
        "",
        "| scale | +vs noop (95% CI) | +vs minus (95% CI) | "
        "+vs shuffled (95% CI) | route wins noop/minus/shuffle |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report.scale_metrics:
        lines.append(
            f"| {row.scale_fraction:.2f} | {row.plus_vs_noop.mean:.4f} "
            f"[{row.plus_vs_noop.ci_lower:.4f}, {row.plus_vs_noop.ci_upper:.4f}] | "
            f"{row.plus_vs_minus.mean:.4f} "
            f"[{row.plus_vs_minus.ci_lower:.4f}, {row.plus_vs_minus.ci_upper:.4f}] | "
            f"{row.plus_vs_shuffled.mean:.4f} "
            f"[{row.plus_vs_shuffled.ci_lower:.4f}, "
            f"{row.plus_vs_shuffled.ci_upper:.4f}] | "
            f"{row.plus_vs_noop_route_win_rate:.3f} / "
            f"{row.plus_vs_minus_route_win_rate:.3f} / "
            f"{row.plus_vs_shuffled_route_win_rate:.3f} |"
        )
    lines.extend(
        (
            "",
            "## 守门边界",
            "",
            "- 主判只读 0.50×cap；0.25/1.00 仅 dose diagnostic，不能救主判。",
            "- target-plus 必须同时优于 noop、sign reversal、shuffled axes，"
            "且 route bootstrap 下界为正。",
            "- 本结果不自动授权 S3 或 production promotion；后续仍需独立契约。",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered ETA S2 no-bias causal steering."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    parser.add_argument(
        "--model-source",
        default="artifacts/eta_stage2_merged_v2_20260803",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--scale-fractions", type=float, nargs="+", default=(0.25, 0.50, 1.00)
    )
    parser.add_argument("--primary-scale-fraction", type=float, default=0.50)
    parser.add_argument(
        "--shuffled-class-shifts", type=int, nargs="+", default=(1, 3, 5)
    )
    parser.add_argument("--control-norm-ratio", type=float, default=0.25)
    parser.add_argument("--probe-train-rows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    args = parser.parse_args(argv)

    artifact = FrozenResidualReadoutArtifact.from_json(
        (_REPO_ROOT / SOURCE_S1_ARTIFACT).read_text(encoding="utf-8")
    )
    preregistration = _validate_preregistration(
        json.loads(args.preregistration.read_text(encoding="utf-8")),
        args=args,
        artifact=artifact,
    )
    preregistration_sha256 = _sha256(args.preregistration)
    source_sha256 = {
        name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
    }
    model_root = (_REPO_ROOT / args.model_source).resolve()
    actual_weights_sha256 = fingerprint_model_weight_files(model_root)
    if actual_weights_sha256 != artifact.model_fingerprint.weights_sha256:
        raise RuntimeError("S2 model weights differ from the frozen S1 artifact")

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
            f"S2 output directory already contains protected results: {existing}"
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
        corpus.train_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    heldout_probe_rows, heldout_classes = eta_stage2_probe_rows(
        corpus.heldout_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    if train_classes != heldout_classes:
        raise RuntimeError("S2 token-budget audit saw split vocabulary drift")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_root),
        local_files_only=True,
    )
    audited_texts = tuple(
        row.observation_text + "\nNext move:"
        for row in (
            *train_probe_rows[: args.probe_train_rows],
            *heldout_probe_rows,
        )
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
            f"S2 token-budget audit found {over_budget} sources above "
            f"max_length={args.max_length}; max={max_observed_source_tokens}"
        )
    uses_mps = args.device.startswith("mps")
    started = time.perf_counter()
    with ExitStack() as stack:
        mps: MPSAvailability | None = None
        if uses_mps:
            stack.enter_context(exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID))
            mps = require_mps()
        runtime = TransformersOpenWeightResidualRuntime(
            model_id=artifact.model_fingerprint.model_id,
            pretrained_source=str(model_root),
            device=args.device,
            max_length=args.max_length,
            fail_on_truncation=True,
            layer_indices=(20,),
            activation_width=896,
            local_files_only=True,
            runtime_origin="hf-local",
            allow_live_substrate_mutation=False,
            allow_offline_substrate_training=False,
            model_dtype="float32",
        )
        report, points = run_eta_s2_causal_steering(
            corpus=corpus,
            runtime=runtime,
            artifact=artifact,
            model_source=args.model_source,
            device=args.device,
            scale_fractions=tuple(args.scale_fractions),
            primary_scale_fraction=args.primary_scale_fraction,
            shuffled_class_shifts=tuple(args.shuffled_class_shifts),
            control_norm_ratio=args.control_norm_ratio,
            scorer_max_length=args.max_length,
            max_observed_source_tokens=max_observed_source_tokens,
            probe_train_row_count=args.probe_train_rows,
            batch_size=args.batch_size,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_confidence=args.bootstrap_confidence,
            thresholds=S2CausalSteeringThresholds(),
            progress=lambda message: print(message, flush=True),
        )
        elapsed = time.perf_counter() - started

    _write_json(output_dir / "report.json", asdict(report))
    with (output_dir / "points.jsonl").open("w", encoding="utf-8") as handle:
        for point in points:
            handle.write(json.dumps(asdict(point), ensure_ascii=False) + "\n")
    (output_dir / "report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    result_files = ("report.json", "points.jsonl", "report.md")
    manifest = {
        "schema_version": "eta-s2-causal-steering-manifest.v1",
        "experiment_id": "eta-s2-no-bias-causal-residual-steering",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report.claim_scope,
        "source_s1": preregistration["source_s1"],
        "preregistered": True,
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": preregistration_sha256,
        "causal_admission_authoritative": True,
        "s2_causal_supported": report.primary_admission.admitted,
        "production_promotion_authorized": False,
        "artifact_installed": False,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
        "max_observed_source_tokens": max_observed_source_tokens,
        "truncated_row_count": report.truncated_row_count,
        "trainable_parameter_count": report.trainable_parameter_count,
        "free_bias_present": report.free_bias_present,
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_exclusive_lock": str(args.mps_lock) if uses_mps else "not-required",
        "mps_attestation": mps_payload(mps) if mps is not None else "not-required",
        "model_weights_sha256": actual_weights_sha256,
        "source_files": source_sha256,
        "result_files": {
            name: _sha256(output_dir / name) for name in result_files
        },
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    primary = next(
        row
        for row in report.scale_metrics
        if row.scale_fraction == report.primary_scale_fraction
    )
    print(
        json.dumps(
            {
                "s2_causal_supported": report.primary_admission.admitted,
                "primary_scale_fraction": report.primary_scale_fraction,
                "plus_vs_noop": asdict(primary.plus_vs_noop),
                "plus_vs_minus": asdict(primary.plus_vs_minus),
                "plus_vs_shuffled": asdict(primary.plus_vs_shuffled),
                "route_win_rates": {
                    "noop": primary.plus_vs_noop_route_win_rate,
                    "minus": primary.plus_vs_minus_route_win_rate,
                    "shuffled": primary.plus_vs_shuffled_route_win_rate,
                },
                "failed_conditions": report.primary_admission.failed_conditions,
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
