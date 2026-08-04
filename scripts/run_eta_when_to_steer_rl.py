"""S3 · run the when-to-steer Internal RL gate against the frozen prereg.

Trains the bounded gate policy by self-written REINFORCE over PE proxies and
evaluates whether it learns WHEN to steer from sparse outcome credit: beating
noop / always-on-belief / random-gate with route-level bootstrap CIs and
concentrating steering where the belief is fresh.

Evidence lane only: no controller installed, no production wiring change, no
evaluation feedback into learning, no substrate/reader/executor training. The
frozen prereg thresholds are asserted to match the module defaults.
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
from volvence_zero.agent.eta_when_to_steer_rl import (
    WhenToSteerReport,
    WhenToSteerThresholds,
    run_eta_when_to_steer_rl,
)
from volvence_zero.substrate import (
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-when-to-steer-rl-mps.v1"
PREREG_PATH = "artifacts/eta_s3_internal_rl_prereg_20260805.json"
SOURCE_FILES = (
    "packages/vz-runtime/src/volvence_zero/agent/eta_when_to_steer_rl.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_read_steer_prereq.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "scripts/run_eta_when_to_steer_rl.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True, cwd=_REPO_ROOT
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _assert_prereg_consistency(
    prereg: dict, thresholds: WhenToSteerThresholds
) -> None:
    """Fail loudly if module thresholds drift from the frozen prereg."""

    rules = prereg["decision_rules"]
    checks = {
        "min_convergence_improvement_nll": (
            thresholds.min_convergence_improvement_nll,
            rules["convergence"]["min_improvement_nll"],
        ),
        "min_gain_vs_noop_nll": (
            thresholds.min_gain_vs_noop_nll,
            rules["gain_vs_noop"]["min_nll"],
        ),
        "min_gain_vs_always_on_nll": (
            thresholds.min_gain_vs_always_on_nll,
            rules["gain_vs_always_on_belief"]["min_nll"],
        ),
        "min_gain_vs_random_gate_nll": (
            thresholds.min_gain_vs_random_gate_nll,
            rules["gain_vs_random_gate"]["min_nll"],
        ),
        "min_gate_selectivity": (
            thresholds.min_gate_selectivity,
            rules["gate_selectivity"]["min"],
        ),
    }
    mismatched = {
        name: pair for name, pair in checks.items() if pair[0] != pair[1]
    }
    if mismatched:
        raise ValueError(
            f"module thresholds drift from frozen prereg: {mismatched}"
        )
    if tuple(prereg["seed_schedule"]) == ():
        raise ValueError("prereg seed_schedule must be non-empty")


def _report_markdown(report: WhenToSteerReport) -> str:
    agg = report.aggregate
    adm = report.admission
    lines = [
        "# ETA 学何时扳（S3 · Internal RL）",
        "",
        "> 冻结 sensor+executor，唯一在线更新门控策略；只给稀疏终局信用、只观测 PE 代理，"
        "从不给每步对错标签。REINFORCE+baseline 学「何时扳」。"
        "不安装控制器、不改 production、不回灌 evaluation、不训基底/reader/executor。",
        "",
        "## 结论",
        "",
        f"- S3 admission：`{'PASS' if adm.admitted else 'FAIL'}`",
        f"- failed：`{adm.failed_conditions}`",
        f"- seeds `{report.seed_schedule}`，episodes {report.max_online_episodes}，"
        f"post-switch 占比 {report.post_switch_fraction:.3f}",
        "",
        "## heldout NLL（seed 平均）",
        "",
        "| arm | NLL |",
        "|---|---:|",
        f"| noop | {agg.noop_nll_mean:.4f} |",
        f"| always_on_belief | {agg.always_on_belief_nll_mean:.4f} |",
        f"| random_gate | {agg.random_gate_nll_mean:.4f} |",
        f"| **pe_gated_online（学到的门控）** | {agg.pe_gated_online_nll_mean:.4f} |",
        f"| oracle_gate（上界诊断） | {agg.oracle_gate_ceiling_nll_mean:.4f} |",
        f"| pe_hard_gate（硬规则上界诊断） | {agg.pe_hard_gate_ceiling_nll_mean:.4f} |",
        f"| fresh_ceiling | {agg.fresh_ceiling_nll_mean:.4f} |",
        "",
        "## 判定门（seed 平均；bootstrap CI 下界取最差 seed）",
        "",
        "| 门 | 值 | 阈值 |",
        "|---|---:|---:|",
        f"| 收敛改善（初始→最终） | {agg.convergence_improvement_nll_mean:.4f} | "
        f"≥{report.thresholds.min_convergence_improvement_nll} |",
        f"| gain vs noop | {agg.noop_nll_mean - agg.pe_gated_online_nll_mean:.4f}"
        f"（CI下界 {agg.gain_vs_noop_ci_lower_min:.4f}） | "
        f"≥{report.thresholds.min_gain_vs_noop_nll}, CI>0 |",
        f"| gain vs always-on | "
        f"{agg.always_on_belief_nll_mean - agg.pe_gated_online_nll_mean:.4f}"
        f"（CI下界 {agg.gain_vs_always_on_ci_lower_min:.4f}） | "
        f"≥{report.thresholds.min_gain_vs_always_on_nll}, CI>0 |",
        f"| gain vs random-gate | "
        f"{agg.random_gate_nll_mean - agg.pe_gated_online_nll_mean:.4f}"
        f"（CI下界 {agg.gain_vs_random_gate_ci_lower_min:.4f}） | "
        f"≥{report.thresholds.min_gain_vs_random_gate_nll}, CI>0 |",
        f"| 门控选择性 steer(非切换)−steer(切换) | {agg.gate_selectivity_mean:.4f}"
        f" | ≥{report.thresholds.min_gate_selectivity} |",
        "",
        "## 边界",
        "",
        "- 存在硬规则上界（belief==fresh 才出手）；本 claim = **RL 从稀疏终局信用学到逼近该上界**，"
        "对应 companion 无免费每步标签的真实约束。",
        "- `substrate_trainable=0`、reader/executor 冻结（未变）、no free bias、zero-code no-op；"
        "仅策略参数在线更新。PASS 只准入独立权威 sweep，不授权 production。",
        "- 不改写任何封存 verdict（kill-eta / S2 / B screen / C2 / 08 / S3-A）。",
    ]
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ETA when-to-steer Internal RL (S3)."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    parser.add_argument(
        "--model-source", default="artifacts/eta_stage2_merged_v2_20260803"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--steering-rank", type=int, default=8)
    parser.add_argument("--executor-updates", type=int, default=80)
    parser.add_argument("--executor-learning-rate", type=float, default=0.01)
    parser.add_argument("--reader-ridge-lambda", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--policy-learning-rate", type=float, default=0.1)
    parser.add_argument("--policy-batch-cases", type=int, default=8)
    parser.add_argument("--entropy-coef", type=float, default=0.1)
    parser.add_argument("--init-noop-bias", type=float, default=0.0)
    parser.add_argument("--policy-restarts", type=int, default=4)
    parser.add_argument("--max-online-episodes", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=80)
    parser.add_argument("--convergence-window", type=int, default=3)
    parser.add_argument("--baseline-beta", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--control-norm-ratio", type=float, default=0.25)
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
    protected = ("report.json", "report.md", "artifact_manifest.json")
    existing = tuple(name for name in protected if (output_dir / name).exists())
    if existing:
        raise FileExistsError(f"S3 output already has results: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    prereg_path = _REPO_ROOT / PREREG_PATH
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    thresholds = WhenToSteerThresholds()
    _assert_prereg_consistency(prereg, thresholds)

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
        report = run_eta_when_to_steer_rl(
            corpus=corpus,
            runtime=runtime,
            scorer=scorer,
            model_source=args.model_source,
            device=args.device,
            injection_layer_index=20,
            residual_width=896,
            steering_rank=args.steering_rank,
            executor_updates=args.executor_updates,
            executor_learning_rate=args.executor_learning_rate,
            reader_ridge_lambda=args.reader_ridge_lambda,
            batch_size=args.batch_size,
            policy_learning_rate=args.policy_learning_rate,
            policy_batch_cases=args.policy_batch_cases,
            entropy_coef=args.entropy_coef,
            init_noop_bias=args.init_noop_bias,
            max_online_episodes=args.max_online_episodes,
            eval_every=args.eval_every,
            convergence_window=args.convergence_window,
            baseline_beta=args.baseline_beta,
            seed_schedule=tuple(args.seeds),
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_confidence=args.bootstrap_confidence,
            thresholds=thresholds,
            progress=lambda message: print(message, flush=True),
        )
    elapsed = time.perf_counter() - started

    _write_json(output_dir / "report.json", asdict(report))
    (output_dir / "report.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    result_files = ("report.json", "report.md")
    manifest = {
        "schema_version": "eta-when-to-steer-rl-manifest.v1",
        "experiment_id": "eta-conditional-steering-internal-rl-when-to-steer",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report.claim_scope,
        "prereg_path": PREREG_PATH,
        "prereg_sha256": _sha256(prereg_path),
        "s3_prerequisite_met": report.admission.admitted,
        "production_promotion_authorized": False,
        "controller_installed": False,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
        "substrate_trainable_parameter_count": (
            report.substrate_trainable_parameter_count
        ),
        "reader_parameters_changed": report.reader_parameters_changed,
        "executor_parameters_changed": report.executor_parameters_changed,
        "free_bias_present": report.free_bias_present,
        "zero_code_strict_noop": report.zero_code_strict_noop,
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_exclusive_lock": str(args.mps_lock) if uses_mps else "not-required",
        "mps_attestation": mps_payload(mps) if mps is not None else "not-required",
        "model_weights_sha256": model_weights_sha256,
        "source_files": {name: _sha256(_REPO_ROOT / name) for name in SOURCE_FILES},
        "result_files": {
            name: _sha256(output_dir / name) for name in result_files
        },
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)

    agg = report.aggregate
    print(
        json.dumps(
            {
                "s3_admitted": report.admission.admitted,
                "failed_conditions": report.admission.failed_conditions,
                "noop_nll": agg.noop_nll_mean,
                "always_on_belief_nll": agg.always_on_belief_nll_mean,
                "random_gate_nll": agg.random_gate_nll_mean,
                "pe_gated_online_nll": agg.pe_gated_online_nll_mean,
                "oracle_gate_ceiling_nll": agg.oracle_gate_ceiling_nll_mean,
                "convergence_improvement_nll": (
                    agg.convergence_improvement_nll_mean
                ),
                "gate_selectivity": agg.gate_selectivity_mean,
                "gain_vs_noop_ci_lower_min": agg.gain_vs_noop_ci_lower_min,
                "gain_vs_always_on_ci_lower_min": (
                    agg.gain_vs_always_on_ci_lower_min
                ),
                "gain_vs_random_gate_ci_lower_min": (
                    agg.gain_vs_random_gate_ci_lower_min
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
