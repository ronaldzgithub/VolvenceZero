"""Run the preregistered ETA S1 full-width frozen residual readout."""

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
from preregister_eta_s1_residual_readout import (
    FROZEN_SOURCE_FILES,
    PREREGISTRATION_SCHEMA_VERSION,
    SOURCE_P1_REPORT,
)
from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    generate_eta_proof_corpus,
)
from volvence_zero.agent.eta_s1_residual_readout import (
    S1ResidualReadoutEvidenceReport,
    S1ResidualReadoutThresholds,
    run_eta_s1_residual_readout,
)
from volvence_zero.substrate import (
    SubstrateFingerprint,
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-s1-full-width-residual-readout-mps.v1"


class PreregistrationMismatch(RuntimeError):
    """Frozen S1 protocol disagrees with the requested execution."""


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
    model_weights_sha256: str,
) -> dict[str, object]:
    return {
        "model_id": args.model_id,
        "model_source": args.model_source,
        "model_version": args.model_version,
        "model_weights_sha256": model_weights_sha256,
        "device": args.device,
        "model_dtype": "float32",
        "max_length": args.max_length,
        "fail_on_truncation": True,
        "layer_indices": [20],
        "activation_widths": [896],
        "ridge_alpha": args.ridge_alpha,
        "training_mode": "closed-form-standardized-ridge-one-hot.v1",
        "class_axis": "class-weight-minus-other-class-mean-l2.v1",
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
        "probe_surface": "cumulative-trajectory-prefix",
    }


def _validate_preregistration(
    payload: object,
    *,
    args: argparse.Namespace,
    model_weights_sha256: str,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise PreregistrationMismatch("preregistration root must be an object")
    if payload.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        raise PreregistrationMismatch("preregistration schema_version mismatch")
    if payload.get("claim_scope") != "s1-readout-admission-no-causal-claim":
        raise PreregistrationMismatch("preregistration claim_scope mismatch")
    if payload.get("configuration") != _requested_configuration(
        args,
        model_weights_sha256=model_weights_sha256,
    ):
        raise PreregistrationMismatch(
            "preregistered configuration does not match requested execution"
        )
    if payload.get("thresholds") != asdict(S1ResidualReadoutThresholds()):
        raise PreregistrationMismatch("preregistered S1 thresholds drifted")

    source = payload.get("source_p1_artifact")
    if not isinstance(source, dict):
        raise PreregistrationMismatch("source P1 artifact is missing")
    source_path = _REPO_ROOT / SOURCE_P1_REPORT
    if source.get("path") != SOURCE_P1_REPORT or source.get("sha256") != _sha256(
        source_path
    ):
        raise PreregistrationMismatch("source P1 report lineage drifted")
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    attribution = source_report.get("attribution")
    if not isinstance(attribution, dict) or (
        attribution.get("dominant_attribution")
        != "incentive-bypass-via-free-bias"
    ):
        raise PreregistrationMismatch("source P1 attribution drifted")

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


def _report_markdown(report: S1ResidualReadoutEvidenceReport) -> str:
    train = report.train_metrics
    heldout = report.heldout_metrics
    admission = report.admission
    return "\n".join(
        (
            "# ETA S1 冻结残差读出",
            "",
            "> 范围：只验证 full-width residual 的线性可读性；不构成 causal "
            "steering 证据，不安装 artifact，不改 production WiringLevel。",
            "",
            "## 结论",
            "",
            f"- S1 admission：`{'PASS' if admission.admitted else 'FAIL'}`",
            f"- heldout accuracy：`{heldout.accuracy:.4f}` "
            f"（chance `{heldout.chance_accuracy:.4f}`，majority "
            f"`{heldout.majority_accuracy:.4f}`）",
            f"- early / late：`{heldout.early_accuracy:.4f}` / "
            f"`{heldout.late_accuracy:.4f}`",
            f"- train / heldout gap：`{train.accuracy - heldout.accuracy:.4f}`",
            f"- mean / min score margin：`{heldout.mean_score_margin:.4f}` / "
            f"`{heldout.min_score_margin:.4f}`",
            f"- artifact：`{report.artifact_id}`",
            "",
            "## 固定几何",
            "",
            f"- layer：`{report.layer_indices}`；width："
            f"`{report.activation_widths}`",
            f"- classes：`{len(report.class_ids)}`；train / heldout rows："
            f"`{report.train_row_count}` / `{report.heldout_row_count}`",
            "- axis：每类 effective weight 减其他类均值后 L2 normalize；"
            "不含 bias。",
            "",
            "## 下一门",
            "",
            (
                "只有 S1 PASS 才允许用本 artifact 另行预注册 S2 的 "
                "+axis / −axis / noop / shuffled-axis matched controls。"
            ),
            "",
        )
    )


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered ETA S1 frozen residual readout."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    parser.add_argument(
        "--model-id", default=ETAOpenWeightRuntimeConfig().model_id
    )
    parser.add_argument(
        "--model-source",
        default="artifacts/eta_stage2_merged_v2_20260803",
    )
    parser.add_argument("--model-version", default="eta-stage2-merged-v2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    args = parser.parse_args(argv)
    if args.max_length < 64 or args.ridge_alpha <= 0.0:
        parser.error("--max-length must be >= 64 and --ridge-alpha must be positive")

    model_root = (_REPO_ROOT / args.model_source).resolve()
    model_weights_sha256 = fingerprint_model_weight_files(model_root)
    preregistration = _validate_preregistration(
        json.loads(args.preregistration.read_text(encoding="utf-8")),
        args=args,
        model_weights_sha256=model_weights_sha256,
    )
    preregistration_sha256 = _sha256(args.preregistration)
    source_sha256 = {
        name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
    }

    output_dir: Path = args.output_dir
    protected_names = (
        "readout_artifact.json",
        "report.json",
        "report.md",
        "artifact_manifest.json",
    )
    existing = tuple(name for name in protected_names if (output_dir / name).exists())
    if existing:
        raise FileExistsError(
            f"S1 output directory already contains protected results: {existing}"
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
    model_fingerprint = SubstrateFingerprint(
        model_id=args.model_id,
        version=args.model_version,
        weights_sha256=model_weights_sha256,
    )
    uses_mps = args.device.startswith("mps")
    started = time.perf_counter()
    with ExitStack() as stack:
        mps: MPSAvailability | None = None
        if uses_mps:
            stack.enter_context(exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID))
            mps = require_mps()
        runtime = TransformersOpenWeightResidualRuntime(
            model_id=args.model_id,
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
        report, artifact = run_eta_s1_residual_readout(
            corpus=corpus,
            runtime=runtime,
            model_fingerprint=model_fingerprint,
            model_source=args.model_source,
            device=args.device,
            expected_layer_indices=(20,),
            expected_activation_widths=(896,),
            ridge_alpha=args.ridge_alpha,
            thresholds=S1ResidualReadoutThresholds(),
            progress=lambda message: print(message, flush=True),
        )
        elapsed = time.perf_counter() - started

    (output_dir / "readout_artifact.json").write_text(
        artifact.to_json() + "\n",
        encoding="utf-8",
    )
    _write_json(output_dir / "report.json", asdict(report))
    (output_dir / "report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    result_files = ("readout_artifact.json", "report.json", "report.md")
    manifest = {
        "schema_version": "eta-s1-residual-readout-manifest.v1",
        "experiment_id": "eta-s1-full-width-frozen-residual-readout",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report.claim_scope,
        "source_p1_artifact": preregistration["source_p1_artifact"],
        "preregistered": True,
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": preregistration_sha256,
        "readout_admission_authoritative": True,
        "s1_admitted_for_s2": report.admission.admitted,
        "formal_causal_claim_allowed": False,
        "artifact_installation_authorized": False,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_exclusive_lock": str(args.mps_lock) if uses_mps else "not-required",
        "mps_attestation": mps_payload(mps) if mps is not None else "not-required",
        "model_weights_sha256": model_weights_sha256,
        "source_files": source_sha256,
        "result_files": {
            name: _sha256(output_dir / name) for name in result_files
        },
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "s1_admitted_for_s2": report.admission.admitted,
                "heldout_accuracy": round(report.heldout_metrics.accuracy, 4),
                "late_accuracy": round(report.heldout_metrics.late_accuracy, 4),
                "train_heldout_gap": round(
                    report.train_metrics.accuracy
                    - report.heldout_metrics.accuracy,
                    4,
                ),
                "artifact_id": report.artifact_id,
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
