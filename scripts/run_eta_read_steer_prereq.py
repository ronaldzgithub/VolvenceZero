"""P2c · S3 prerequisite — run the read->steer loop with a non-oracle sensor.

Refits a frozen linear condition reader on the C2 surface (context-carrying
residual), trains the C2 conditional executor, and evaluates whether the
read (not oracle) condition still closes the goal-stripped gap and beats the
equal-budget unconditional operator, with route-level bootstrap CIs.

Screen/prerequisite only: no controller installed, no production wiring
change, no evaluation feedback into learning.
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

from volvence_zero.agent.eta_conditional_steering_screen import (
    ACTION_PROMPT_SUFFIX,
)
from volvence_zero.agent.eta_conflict_instrument import (
    build_conflict_junction_rows,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_rate_distortion_evidence import _action_options
from volvence_zero.agent.eta_read_steer_prereq import (
    ReadSteerReport,
    ReadSteerThresholds,
    run_eta_read_steer_prereq,
)
from volvence_zero.substrate import (
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-read-steer-prereq-mps.v1"
SOURCE_FILES = (
    "packages/vz-runtime/src/volvence_zero/agent/eta_read_steer_prereq.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "scripts/run_eta_read_steer_prereq.py",
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


def _report_markdown(report: ReadSteerReport) -> str:
    agg = report.aggregate
    admission = report.admission
    lines = [
        "# ETA 读→扳闭环（P2c · S3 前置）",
        "",
        "> 用 refit 冻结线性 sensor 从上下文残差在线读出 subgoal（非 oracle），"
        "再用 C2 执行器扳目标剥离动作；route-level bootstrap CI 守门。"
        "不安装控制器、不改 production、不回灌 evaluation。",
        "",
        "## 结论",
        "",
        f"- S3 前置 admission：`{'PASS' if admission.admitted else 'FAIL'}`",
        f"- failed conditions：`{admission.failed_conditions}`",
        f"- reader heldout top-1：`{agg.reader_heldout_accuracy_mean:.3f}`"
        f"（chance {1.0 / report.subgoal_class_count:.3f}）",
        f"- seeds：`{report.seed_schedule}`，rank={report.steering_rank}，"
        f"updates={report.updates_per_run}，bootstrap={report.bootstrap_resamples}",
        "",
        "## heldout NLL（seed 平均）",
        "",
        "| arm | expert NLL |",
        "|---|---:|",
        f"| noop | {agg.noop_nll_mean:.4f} |",
        f"| subgoal-revealed 天花板 | {agg.subgoal_revealed_ceiling_nll_mean:.4f} |",
        f"| conditional-oracle | {agg.conditional_oracle_nll_mean:.4f} |",
        f"| **conditional-online（读→扳）** | {agg.conditional_online_nll_mean:.4f} |",
        f"| unconditional（等预算恒定） | {agg.unconditional_nll_mean:.4f} |",
        f"| random-condition | {agg.random_condition_nll_mean:.4f} |",
        "",
        "## 判定量（seed 平均；bootstrap CI 下界取最差 seed）",
        "",
        "| 门 | 值 | 阈值 |",
        "|---|---:|---:|",
        f"| online gap closed (noop−online) | {agg.online_gap_closed_nll_mean:.4f} | "
        f"≥{report.thresholds.min_online_gap_closed_nll} |",
        f"| online conditional advantage (uncond−online) | "
        f"{agg.online_conditional_advantage_nll_mean:.4f} | "
        f"≥{report.thresholds.min_online_conditional_advantage_nll} |",
        f"| online−noop CI 下界(min) | {agg.online_vs_noop_ci_lower_min:.4f} | >0 |",
        f"| online−uncond CI 下界(min) | "
        f"{agg.online_vs_unconditional_ci_lower_min:.4f} | >0 |",
        f"| reader heldout acc | {agg.reader_heldout_accuracy_mean:.3f} | "
        f"≥{report.thresholds.min_reader_heldout_accuracy} |",
        "",
        "## 守门边界",
        "",
        "- sensor 是 refit 冻结线性读出（无 LM 训练、steer 时冻结）；executor "
        "为冻结基底上的 rank-8 乘性写入（no free bias、zero-code strict no-op）。",
        "- condition 读自**上下文携带目标**的残差（agent 从记忆/上下文知道 subgoal），"
        "应用于**目标剥离**路口动作前向——这才是可部署的读→扳环。",
        "- PASS 只表示可准入 S3 Internal RL 的正式预注册；不授权 production。",
        "",
    ]
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ETA read->steer S3 prerequisite."
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
    parser.add_argument("--steering-rank", type=int, default=8)
    parser.add_argument("--screen-train-route-count", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--control-norm-ratio", type=float, default=0.25)
    parser.add_argument("--reader-ridge-lambda", type=float, default=10.0)
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

    output_dir: Path = args.output_dir
    protected_names = ("report.json", "report.md", "artifact_manifest.json")
    existing = tuple(
        name for name in protected_names if (output_dir / name).exists()
    )
    if existing:
        raise FileExistsError(
            f"S3-prereq output already has protected results: {existing}"
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
    model_root = (_REPO_ROOT / args.model_source).resolve()
    model_weights_sha256 = fingerprint_model_weight_files(model_root)

    probe_rows = build_conflict_junction_rows(corpus, split="train")
    probe_texts = tuple(
        row.observation_text + ACTION_PROMPT_SUFFIX for row in probe_rows[:16]
    )

    started = time.perf_counter()
    uses_mps = args.device.startswith("mps")
    with ExitStack() as stack:
        mps: MPSAvailability | None = None
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
        scorer = runtime.build_steered_action_scorer(
            action_options=_action_options(corpus.environment),
            injection_layer_index=20,
            prompt_suffix="",
            max_length=args.max_length,
            control_norm_ratio=args.control_norm_ratio,
            probe_texts=probe_texts,
            joint_training=False,
            prefix_cache=True,
        )
        report = run_eta_read_steer_prereq(
            corpus=corpus,
            runtime=runtime,
            scorer=scorer,
            model_source=args.model_source,
            device=args.device,
            injection_layer_index=20,
            residual_width=896,
            steering_rank=args.steering_rank,
            screen_train_route_count=args.screen_train_route_count,
            seed_schedule=tuple(args.seeds),
            updates_per_run=args.updates,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            reader_ridge_lambda=args.reader_ridge_lambda,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_confidence=args.bootstrap_confidence,
            thresholds=ReadSteerThresholds(),
            progress=lambda message: print(message, flush=True),
        )
    elapsed = time.perf_counter() - started

    _write_json(output_dir / "report.json", asdict(report))
    (output_dir / "report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    result_files = ("report.json", "report.md")
    manifest = {
        "schema_version": "eta-read-steer-prereq-manifest.v1",
        "experiment_id": "eta-conditional-steering-read-loop-s3-prereq",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report.claim_scope,
        "s3_prerequisite": True,
        "prerequisite_met": report.admission.admitted,
        "production_promotion_authorized": False,
        "artifact_installed": False,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
        "free_bias_present": report.free_bias_present,
        "zero_code_strict_noop": report.zero_code_strict_noop,
        "substrate_trainable_parameter_count": (
            report.substrate_trainable_parameter_count
        ),
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_exclusive_lock": str(args.mps_lock) if uses_mps else "not-required",
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

    agg = report.aggregate
    print(
        json.dumps(
            {
                "prerequisite_met": report.admission.admitted,
                "failed_conditions": report.admission.failed_conditions,
                "reader_heldout_accuracy": agg.reader_heldout_accuracy_mean,
                "noop_nll": agg.noop_nll_mean,
                "conditional_oracle_nll": agg.conditional_oracle_nll_mean,
                "conditional_online_nll": agg.conditional_online_nll_mean,
                "unconditional_nll": agg.unconditional_nll_mean,
                "online_gap_closed_nll": agg.online_gap_closed_nll_mean,
                "online_conditional_advantage_nll": (
                    agg.online_conditional_advantage_nll_mean
                ),
                "online_vs_noop_ci_lower_min": agg.online_vs_noop_ci_lower_min,
                "online_vs_unconditional_ci_lower_min": (
                    agg.online_vs_unconditional_ci_lower_min
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
