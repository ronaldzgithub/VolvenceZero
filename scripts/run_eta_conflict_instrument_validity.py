"""P2c · C1 — goal-ambiguous junction instrument validity (read-only).

Structural headroom is computed with no model. When ``--device`` is an MPS
device and ``--measure-base-uncertainty`` is set, the frozen merged S1 model
scores expert-action NLL on goal-stripped vs subgoal-revealed prompts to prove
the base model is genuinely uncertain when the goal is hidden. No controller is
fit, no bias is added, no parameter is trained, no production wiring changes.
"""

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

from volvence_zero.agent.eta_conflict_instrument import (
    ETA_CONFLICT_INSTRUMENT_SCHEMA_VERSION,
    GOAL_AMBIGUOUS_JUNCTION_PROTOCOL,
    ConflictInstrumentReport,
    ConflictInstrumentThresholds,
    assess_conflict_instrument,
    build_conflict_junction_rows,
    compute_conflict_headroom,
    measure_base_uncertainty,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_rate_distortion_evidence import _action_options
from volvence_zero.substrate import (
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-conflict-instrument-validity-mps.v1"
SOURCE_FILES = (
    "packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "scripts/run_eta_conflict_instrument_validity.py",
)


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


def _report_markdown(report: ConflictInstrumentReport) -> str:
    headroom = report.headroom
    admission = report.admission
    lines = [
        "# ETA 冲突映射仪器有效性（P2c · C1，只读）",
        "",
        "> 范围：目标剥离路口仪器的结构余量 + 基底不确定性；不训练参数、"
        "不加 bias、不改 production。",
        "",
        "## 结论",
        "",
        f"- 仪器有效性：`{'VALID' if admission.valid else 'INVALID'}`",
        f"- 观测协议：`{report.observation_protocol}`",
        f"- failed conditions：`{admission.failed_conditions}`",
        "",
        "## 结构余量（heldout，无模型）",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| conflict row fraction | {headroom.conflict_row_fraction:.3f} |",
        f"| constant-operator error | {headroom.constant_operator_error_rate:.3f} |",
        f"| oracle (view,subgoal) error | {headroom.oracle_conditional_error_rate:.3f} |",
        f"| view-subgoal residual ambiguity | {headroom.view_subgoal_residual_ambiguity} |",
        f"| unique local views | {headroom.unique_local_views} |",
        f"| rows / mean out-edges | {headroom.row_count} / {headroom.mean_available_targets:.2f} |",
        "",
    ]
    if report.base_uncertainty is not None:
        base = report.base_uncertainty
        lines.extend(
            (
                "## 基底不确定性（heldout，frozen merged 模型）",
                "",
                "| 指标 | 值 |",
                "|---|---:|",
                f"| goal-stripped expert NLL (mean/median) | "
                f"{base.goal_stripped_mean_expert_nll:.4f} / "
                f"{base.goal_stripped_median_expert_nll:.4f} |",
                f"| subgoal-revealed expert NLL (mean) | "
                f"{base.subgoal_revealed_mean_expert_nll:.4f} |",
                f"| steerable headroom (stripped − revealed) | "
                f"{base.steerable_headroom_nll:.4f} |",
                f"| fraction base uncertain (NLL>{base.uncertain_nll_threshold}) | "
                f"{base.fraction_base_uncertain:.3f} |",
                "",
            )
        )
    lines.extend(
        (
            "## 守门边界",
            "",
            "- 恒定算子错误率证明无条件映射不足；(view,subgoal) 残余歧义为 0 "
            "证明 subgoal 是唯一缺失比特。",
            "- 本结果只判仪器是否值得跑 C2 条件化学习式 steering screen；"
            "不改写任何已封存 verdict，不授权 production。",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the read-only ETA conflict-instrument validity check."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    parser.add_argument(
        "--model-source",
        default="artifacts/eta_stage2_merged_v2_20260803",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--measure-base-uncertainty", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--uncertain-nll-threshold", type=float, default=0.10)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    protected_names = ("report.json", "report.md", "artifact_manifest.json")
    existing = tuple(
        name for name in protected_names if (output_dir / name).exists()
    )
    if existing:
        raise FileExistsError(
            f"conflict-instrument output already has protected results: {existing}"
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
    heldout_rows = build_conflict_junction_rows(corpus, split="heldout")
    train_rows = build_conflict_junction_rows(corpus, split="train")
    heldout_headroom = compute_conflict_headroom(heldout_rows, split="heldout")
    train_headroom = compute_conflict_headroom(train_rows, split="train")

    thresholds = ConflictInstrumentThresholds()
    base_uncertainty = None
    model_weights_sha256 = "not-loaded"
    started = time.perf_counter()
    uses_mps = args.device.startswith("mps")
    with ExitStack() as stack:
        mps: MPSAvailability | None = None
        if args.measure_base_uncertainty:
            model_root = (_REPO_ROOT / args.model_source).resolve()
            model_weights_sha256 = fingerprint_model_weight_files(model_root)
            if uses_mps:
                stack.enter_context(
                    exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID)
                )
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
            probe_texts = tuple(row.observation_text for row in train_rows[:16])
            scorer = runtime.build_steered_action_scorer(
                action_options=_action_options(corpus.environment),
                injection_layer_index=20,
                max_length=args.max_length,
                control_norm_ratio=0.25,
                probe_texts=probe_texts,
                joint_training=False,
                prefix_cache=True,
            )
            if scorer.trainable_parameters():
                raise RuntimeError(
                    "conflict-instrument scorer must be fully frozen"
                )
            base_uncertainty = measure_base_uncertainty(
                heldout_rows,
                scorer=scorer,
                split="heldout",
                uncertain_nll_threshold=args.uncertain_nll_threshold,
                batch_size=args.batch_size,
            )
    elapsed = time.perf_counter() - started

    admission = assess_conflict_instrument(
        headroom=heldout_headroom,
        thresholds=thresholds,
        base_uncertainty=base_uncertainty,
    )
    report = ConflictInstrumentReport(
        schema_version=ETA_CONFLICT_INSTRUMENT_SCHEMA_VERSION,
        claim_scope="conflict-junction-instrument-validity",
        observation_protocol=GOAL_AMBIGUOUS_JUNCTION_PROTOCOL,
        corpus_seed=corpus.seed,
        objective_count=corpus.objective_count,
        model_id=args.model_id if args.measure_base_uncertainty else "not-loaded",
        model_source=(
            args.model_source if args.measure_base_uncertainty else "not-loaded"
        ),
        device=args.device if args.measure_base_uncertainty else "cpu-structural",
        thresholds=thresholds,
        headroom=heldout_headroom,
        base_uncertainty=base_uncertainty,
        admission=admission,
        trainable_parameter_count=0,
        free_bias_present=False,
        production_wiring_changed=False,
        feedback_to_learning=False,
        description=(
            "Read-only conflict-junction instrument validity: structural "
            f"headroom + optional base uncertainty. valid={admission.valid}."
        ),
    )

    _write_json(output_dir / "report.json", asdict(report))
    _write_json(
        output_dir / "train_headroom.json",
        asdict(train_headroom),
    )
    (output_dir / "report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    result_files = ("report.json", "report.md", "train_headroom.json")
    manifest = {
        "schema_version": "eta-conflict-instrument-manifest.v1",
        "experiment_id": "eta-conflict-junction-instrument-validity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report.claim_scope,
        "read_only_diagnostic": True,
        "instrument_valid": admission.valid,
        "base_uncertainty_evaluated": base_uncertainty is not None,
        "production_promotion_authorized": False,
        "artifact_installed": False,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_exclusive_lock": (
            str(args.mps_lock)
            if (uses_mps and args.measure_base_uncertainty)
            else "not-required"
        ),
        "mps_attestation": mps_payload(mps) if mps is not None else "not-required",
        "model_weights_sha256": model_weights_sha256,
        "source_files": {
            name: _sha256(_REPO_ROOT / name) for name in SOURCE_FILES
        },
        "result_files": {
            name: _sha256(output_dir / name) for name in result_files
        },
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)

    print(
        json.dumps(
            {
                "instrument_valid": admission.valid,
                "failed_conditions": admission.failed_conditions,
                "conflict_row_fraction": heldout_headroom.conflict_row_fraction,
                "constant_operator_error_rate": (
                    heldout_headroom.constant_operator_error_rate
                ),
                "oracle_conditional_error_rate": (
                    heldout_headroom.oracle_conditional_error_rate
                ),
                "view_subgoal_residual_ambiguity": (
                    heldout_headroom.view_subgoal_residual_ambiguity
                ),
                "base_uncertainty": (
                    asdict(base_uncertainty)
                    if base_uncertainty is not None
                    else None
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
