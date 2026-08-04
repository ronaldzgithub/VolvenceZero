"""P2c · C2 — run the conditional learned steering directional screen.

Trains a matched-budget conditional vs unconditional rank-r multiplicative
write on the frozen merged S1 model over the goal-ambiguous junction
instrument, then judges whether conditioning on the subgoal closes the
goal-stripped NLL gap and beats the equal-budget unconditional operator.

Screen only: no controller is installed, no production wiring changes, no
evaluation feeds back into learning.
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
    ConditionalSteeringReport,
    ConditionalSteeringThresholds,
    run_eta_conditional_steering_screen,
)
from volvence_zero.agent.eta_conflict_instrument import (
    build_conflict_junction_rows,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_rate_distortion_evidence import _action_options
from volvence_zero.substrate import (
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-conditional-steering-screen-mps.v1"
SOURCE_FILES = (
    "packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "scripts/run_eta_conditional_steering_screen.py",
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


def _report_markdown(report: ConditionalSteeringReport) -> str:
    agg = report.aggregate
    admission = report.admission
    lines = [
        "# ETA 条件化学习式 steering screen（P2c · C2）",
        "",
        "> 范围：goal-ambiguous junction 仪器上，matched-budget 条件 vs 无条件"
        "学习式低秩乘性写入；不安装控制器、不改 production、不回灌 evaluation。",
        "",
        "## 结论",
        "",
        f"- screen admission：`{'PASS' if admission.admitted else 'FAIL'}`",
        f"- failed conditions：`{admission.failed_conditions}`",
        f"- seeds：`{report.seed_schedule}`，rank={report.steering_rank}，"
        f"updates={report.updates_per_run}",
        "",
        "## heldout NLL（seed 平均）",
        "",
        "| arm | expert NLL |",
        "|---|---:|",
        f"| noop（目标隐藏基线） | {agg.heldout_noop_nll_mean:.4f} |",
        f"| subgoal-revealed 天花板 | {agg.subgoal_revealed_ceiling_nll_mean:.4f} |",
        f"| conditional（本方案） | {agg.heldout_conditional_nll_mean:.4f} |",
        f"| unconditional（等预算恒定） | {agg.heldout_unconditional_nll_mean:.4f} |",
        f"| random-condition（错条件） | {agg.heldout_random_condition_nll_mean:.4f} |",
        "",
        "## 判定量（seed 平均 / 最差）",
        "",
        "| 门 | 值 | 阈值 |",
        "|---|---:|---:|",
        f"| gap closed (noop−cond) | {agg.gap_closed_nll_mean:.4f} "
        f"(min {agg.gap_closed_nll_min:.4f}) | ≥{report.thresholds.min_gap_closed_nll} |",
        f"| conditional advantage (uncond−cond) | {agg.conditional_advantage_nll_mean:.4f} "
        f"(min {agg.conditional_advantage_nll_min:.4f}) | "
        f"≥{report.thresholds.min_conditional_advantage_nll} |",
        f"| condition specificity (rand−cond) | {agg.condition_specificity_nll_mean:.4f} | "
        f"≥{report.thresholds.min_condition_specificity_nll} |",
        f"| gap closed fraction | {agg.gap_closed_fraction_mean:.4f} | "
        f"≥{report.thresholds.min_gap_closed_fraction} |",
        "",
        "## 守门边界",
        "",
        "- 距离天花板与恒定算子皆为 matched-budget（同 norm cap）对照；"
        "no free bias、zero-code strict no-op、substrate 冻结。",
        "- screen 只决定是否准入独立权威 sweep；不改写任何已封存 verdict，"
        "不授权 production。",
        "",
    ]
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ETA conditional learned steering screen."
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
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--control-norm-ratio", type=float, default=0.25)
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
            f"C2 output already has protected results: {existing}"
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
        report = run_eta_conditional_steering_screen(
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
            thresholds=ConditionalSteeringThresholds(),
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
        "schema_version": "eta-conditional-steering-manifest.v1",
        "experiment_id": "eta-conditional-learned-steering-screen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report.claim_scope,
        "directional_screen": True,
        "screen_admitted": report.admission.admitted,
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
                "screen_admitted": report.admission.admitted,
                "failed_conditions": report.admission.failed_conditions,
                "heldout_noop_nll": agg.heldout_noop_nll_mean,
                "heldout_conditional_nll": agg.heldout_conditional_nll_mean,
                "heldout_unconditional_nll": agg.heldout_unconditional_nll_mean,
                "heldout_random_condition_nll": (
                    agg.heldout_random_condition_nll_mean
                ),
                "subgoal_revealed_ceiling_nll": (
                    agg.subgoal_revealed_ceiling_nll_mean
                ),
                "gap_closed_nll": agg.gap_closed_nll_mean,
                "conditional_advantage_nll": agg.conditional_advantage_nll_mean,
                "condition_specificity_nll": agg.condition_specificity_nll_mean,
                "gap_closed_fraction": agg.gap_closed_fraction_mean,
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
