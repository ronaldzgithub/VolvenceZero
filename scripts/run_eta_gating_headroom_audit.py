"""S3-A · gating headroom audit (read-only diagnostic).

Before spending Internal-RL compute on "learn WHEN to steer", prove the gating
problem has measurable headroom on this instrument -- otherwise S3 repeats the
P2a mistake (a task with no room for the studied intervention).

The natural source of headroom is that a WRONG condition is catastrophic
(C2/08: random-condition NLL 7.62 vs noop 2.81). We model a realistic non-oracle
failure: a *stale belief*. Within a route the active subgoal changes when an
objective is passed; an agent that reads its subgoal from the PREVIOUS step's
context (memory lag) holds a stale belief exactly at these "post-switch"
junctions. Steering on the stale belief there is wrong; noop is merely mediocre.

We measure, per heldout junction, the expert-action NLL under:

* ``noop``                 -- do nothing (goal hidden baseline)
* ``always_on_belief``     -- always steer with the stale belief condition
* ``oracle_gate_belief``   -- steer only where the belief is actually correct
* ``pe_gate_belief``       -- steer only where belief agrees with a fresh read
                              (an OBSERVABLE proxy, no oracle)
* ``fresh_ceiling``        -- steer with the fresh (current) read (08 online ref)

If oracle-gating beats always-on and beats noop by a margin, gating headroom
exists; if the observable pe-gate captures most of that saving, S3-C has a
signal to learn from. Read-only: one operator seed trained (reusing the C2
executor), no controller installed, no production wiring, no evaluation
feedback, no substrate training.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
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
    _ConditionalOperator,
    _subgoal_vocabulary,
    _train_operator,
    _zero_code_max_abs,
)
from volvence_zero.agent.eta_conflict_instrument import (
    build_conflict_junction_rows,
)
from volvence_zero.agent.eta_proof_benchmark import generate_eta_proof_corpus
from volvence_zero.agent.eta_rate_distortion_evidence import _action_options
from volvence_zero.agent.eta_read_steer_prereq import (
    _capture_examples,
    _labelled_rows,
    _per_row_baseline_nll,
    _per_row_controlled_nll,
    _stack_residuals_action,
    fit_condition_reader,
)
from volvence_zero.substrate import (
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-gating-headroom-audit-mps.v1"
SOURCE_FILES = (
    "packages/vz-runtime/src/volvence_zero/agent/eta_read_steer_prereq.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "scripts/run_eta_gating_headroom_audit.py",
)

MIN_POST_SWITCH_FRACTION = 0.10
MIN_GATING_HEADROOM_VS_ALWAYSON = 0.30
MIN_GATING_GAIN_VS_NOOP = 0.30
MIN_STALENESS_DETECTABILITY = 0.50


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


def _reader_scores(reader, residuals: np.ndarray) -> np.ndarray:
    standardized = (residuals - reader.feature_mean) / reader.feature_scale
    return standardized @ reader.weights


def _gated(
    steered: list[float], noop: list[float], gate_open: list[bool]
) -> list[float]:
    return [
        steered[index] if gate_open[index] else noop[index]
        for index in range(len(noop))
    ]


def _audit_markdown(report: dict) -> str:
    a = report["arms_mean"]
    g = report["gating"]
    obs = report["observability"]
    adm = report["admission"]
    lines = [
        "# ETA 门控余量审计（S3-A，只读诊断）",
        "",
        "> 证明「何时扳」在本仪器上有可测且可观测的门控余量，再花 Internal RL 算力。"
        "过期 belief 在切换路口错条件（灾难），noop 平庸，择时可赢。"
        "不安装控制器、不改 production、不回灌 evaluation、不训基底。",
        "",
        "## 结论",
        "",
        f"- 门控余量 admission：`{'PASS' if adm['admitted'] else 'FAIL'}`",
        f"- failed：`{adm['failed_conditions']}`",
        f"- post-switch 行占比：`{report['post_switch_fraction']:.3f}`"
        f"（阈值 ≥{MIN_POST_SWITCH_FRACTION}）",
        "",
        "## 各臂 heldout expert NLL（全体行均值）",
        "",
        "| arm | NLL |",
        "|---|---:|",
        f"| noop（目标隐藏基线） | {a['noop']:.4f} |",
        f"| always_on_belief（过期条件恒定出手） | {a['always_on_belief']:.4f} |",
        f"| **oracle_gate_belief（择时上界）** | {a['oracle_gate_belief']:.4f} |",
        f"| pe_gate_belief（可观测一致性门） | {a['pe_gate_belief']:.4f} |",
        f"| fresh_ceiling（08 online 参考） | {a['fresh_ceiling']:.4f} |",
        "",
        "## post-switch 子集（belief 过期处）",
        "",
        "| arm | NLL |",
        "|---|---:|",
        f"| noop | {report['post_switch_arms_mean']['noop']:.4f} |",
        f"| always_on_belief | {report['post_switch_arms_mean']['always_on_belief']:.4f} |",
        f"| fresh_ceiling | {report['post_switch_arms_mean']['fresh_ceiling']:.4f} |",
        "",
        "## 门控余量与可观测性",
        "",
        "| 量 | 值 | 阈值 |",
        "|---|---:|---:|",
        f"| 余量 = always_on − oracle_gate | {g['headroom_vs_alwayson']:.4f} | "
        f"≥{MIN_GATING_HEADROOM_VS_ALWAYSON} |",
        f"| 增益 = noop − oracle_gate | {g['gain_vs_noop']:.4f} | "
        f"≥{MIN_GATING_GAIN_VS_NOOP} |",
        f"| 可观测门捕获 = always_on − pe_gate | {g['pe_gate_recovers']:.4f} | — |",
        f"| staleness 可检测性 P(belief≠fresh \\| post-switch) | "
        f"{obs['staleness_detectability']:.3f} | ≥{MIN_STALENESS_DETECTABILITY} |",
        f"| 误报 P(belief≠fresh \\| 非 post-switch) | "
        f"{obs['staleness_false_alarm']:.3f} | — |",
        f"| reader margin（belief 上下文）均值 | {obs['belief_margin_mean']:.3f} | — |",
        "",
        "## 含义",
        "",
        "- oracle_gate 明显优于 always_on ⇒ **择时有价值**（错条件出手是净损）。",
        "- oracle_gate 明显优于 noop ⇒ 择时仍胜「什么都不做」。",
        "- pe_gate（belief 与 fresh 读一致才出手）已捕获大部分余量 ⇒ **存在可观测信号**，"
        "S3-C 的策略有东西可学（RL 从稀疏结局信用学阈值/组合，而非硬编码规则）。",
        "- staleness 可检测性高 ⇒ 门控信号存在；这是 S3 从 PE 代理学「何时扳」的前提。",
        "",
        "PASS ⇒ 准入 S3-C（Internal RL 学何时扳）；不改写任何封存 verdict。",
    ]
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ETA gating-headroom audit (S3-A)."
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
    parser.add_argument("--operator-seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--control-norm-ratio", type=float, default=0.25)
    parser.add_argument("--reader-ridge-lambda", type=float, default=10.0)
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
        raise FileExistsError(f"audit output already has results: {existing}")
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
    subgoal_vocabulary = _subgoal_vocabulary(corpus)
    subgoal_index = {name: index for index, name in enumerate(subgoal_vocabulary)}
    class_count = len(subgoal_vocabulary)

    model_root = (_REPO_ROOT / args.model_source).resolve()
    model_weights_sha256 = fingerprint_model_weight_files(model_root)

    train_rows = _labelled_rows(build_conflict_junction_rows(corpus, split="train"))
    heldout_rows = _labelled_rows(
        build_conflict_junction_rows(corpus, split="heldout")
    )
    probe_texts = tuple(
        row.observation_text + ACTION_PROMPT_SUFFIX for row in train_rows[:16]
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
        if scorer.trainable_parameters():
            raise RuntimeError("audit requires a frozen substrate scorer")

        train_examples = _capture_examples(
            train_rows,
            runtime=runtime,
            scorer=scorer,
            subgoal_index=subgoal_index,
            injection_layer_index=20,
            residual_width=896,
            progress=lambda message: print(message, flush=True),
            split_label="train",
        )
        heldout_examples = _capture_examples(
            heldout_rows,
            runtime=runtime,
            scorer=scorer,
            subgoal_index=subgoal_index,
            injection_layer_index=20,
            residual_width=896,
            progress=lambda message: print(message, flush=True),
            split_label="heldout",
        )
        reader = fit_condition_reader(
            train_examples,
            class_count=class_count,
            ridge_lambda=args.reader_ridge_lambda,
        )

        train_residuals = _stack_residuals_action(torch, train_examples)
        train_subgoals = torch.tensor(
            [example.subgoal_index for example in train_examples],
            dtype=torch.long,
        )
        operator = _ConditionalOperator(
            torch=torch,
            width=896,
            rank=args.steering_rank,
            class_count=class_count,
            conditional=True,
            seed=args.operator_seed,
        )
        _train_operator(
            torch=torch,
            operator=operator,
            residuals=train_residuals,
            subgoal_indices=train_subgoals,
            action_indices=tuple(
                example.action_index for example in train_examples
            ),
            texts=tuple(example.observation_text for example in train_examples),
            scorer=scorer,
            updates=args.updates,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.operator_seed,
            progress=lambda message: print(message, flush=True),
            label="audit-operator",
        )

        heldout_residuals = _stack_residuals_action(torch, heldout_examples)
        context_residuals = np.asarray(
            [example.context_residual for example in heldout_examples],
            dtype=np.float64,
        )
        scores = _reader_scores(reader, context_residuals)
        fresh_pred = scores.argmax(axis=1)
        sorted_scores = np.sort(scores, axis=1)
        fresh_margin = sorted_scores[:, -1] - sorted_scores[:, -2]
        true_subgoal = np.asarray(
            [example.subgoal_index for example in heldout_examples],
            dtype=np.int64,
        )

        # Stale belief: the fresh read carried over from the previous labelled
        # junction in the same route (memory lag). First-in-route -> own read.
        last_in_case: dict[str, int] = {}
        belief_source = np.arange(len(heldout_examples))
        post_switch = np.zeros(len(heldout_examples), dtype=bool)
        for index, example in enumerate(heldout_examples):
            previous = last_in_case.get(example.case_id)
            if previous is not None:
                belief_source[index] = previous
                post_switch[index] = (
                    true_subgoal[index] != true_subgoal[previous]
                )
            last_in_case[example.case_id] = index
        belief_pred = fresh_pred[belief_source]

        action_indices = tuple(
            example.action_index for example in heldout_examples
        )
        texts = tuple(example.observation_text for example in heldout_examples)
        with torch.no_grad():
            belief_deltas = operator.deltas(
                residuals=heldout_residuals,
                subgoal_indices=torch.tensor(belief_pred, dtype=torch.long),
            )
            fresh_deltas = operator.deltas(
                residuals=heldout_residuals,
                subgoal_indices=torch.tensor(fresh_pred, dtype=torch.long),
            )
            zero_code_max_abs = _zero_code_max_abs(
                torch=torch,
                operator=operator,
                residuals=heldout_residuals,
                subgoal_indices=torch.tensor(true_subgoal, dtype=torch.long),
            )
        noop_rows = _per_row_baseline_nll(
            texts=texts,
            action_indices=action_indices,
            scorer=scorer,
            batch_size=args.batch_size,
        )
        always_on_belief_rows = _per_row_controlled_nll(
            deltas=belief_deltas,
            texts=texts,
            action_indices=action_indices,
            scorer=scorer,
            batch_size=args.batch_size,
        )
        fresh_ceiling_rows = _per_row_controlled_nll(
            deltas=fresh_deltas,
            texts=texts,
            action_indices=action_indices,
            scorer=scorer,
            batch_size=args.batch_size,
        )
    elapsed = time.perf_counter() - started

    belief_correct = (belief_pred == true_subgoal).tolist()
    belief_agrees_fresh = (belief_pred == fresh_pred).tolist()
    oracle_gate_rows = _gated(always_on_belief_rows, noop_rows, belief_correct)
    pe_gate_rows = _gated(always_on_belief_rows, noop_rows, belief_agrees_fresh)

    def mean(values: list[float]) -> float:
        return statistics.fmean(values)

    post_switch_indices = [
        index for index, flag in enumerate(post_switch.tolist()) if flag
    ]
    non_switch_indices = [
        index
        for index in range(len(heldout_examples))
        if index not in set(post_switch_indices)
    ]

    def subset_mean(values: list[float], indices: list[int]) -> float:
        return statistics.fmean(values[i] for i in indices) if indices else 0.0

    detect = (
        statistics.fmean(
            not belief_agrees_fresh[i] for i in post_switch_indices
        )
        if post_switch_indices
        else 0.0
    )
    false_alarm = (
        statistics.fmean(
            not belief_agrees_fresh[i] for i in non_switch_indices
        )
        if non_switch_indices
        else 0.0
    )

    arms_mean = {
        "noop": mean(noop_rows),
        "always_on_belief": mean(always_on_belief_rows),
        "oracle_gate_belief": mean(oracle_gate_rows),
        "pe_gate_belief": mean(pe_gate_rows),
        "fresh_ceiling": mean(fresh_ceiling_rows),
    }
    gating = {
        "headroom_vs_alwayson": (
            arms_mean["always_on_belief"] - arms_mean["oracle_gate_belief"]
        ),
        "gain_vs_noop": arms_mean["noop"] - arms_mean["oracle_gate_belief"],
        "pe_gate_recovers": (
            arms_mean["always_on_belief"] - arms_mean["pe_gate_belief"]
        ),
    }
    observability = {
        "staleness_detectability": detect,
        "staleness_false_alarm": false_alarm,
        "belief_margin_mean": float(fresh_margin[belief_source].mean()),
        "fresh_margin_mean": float(fresh_margin.mean()),
        "belief_correct_fraction": float(np.mean(belief_pred == true_subgoal)),
    }
    conditions = {
        "post-switch-fraction": (
            float(post_switch.mean()) >= MIN_POST_SWITCH_FRACTION
        ),
        "gating-headroom-vs-alwayson": (
            gating["headroom_vs_alwayson"] >= MIN_GATING_HEADROOM_VS_ALWAYSON
        ),
        "gating-gain-vs-noop": (
            gating["gain_vs_noop"] >= MIN_GATING_GAIN_VS_NOOP
        ),
        "staleness-detectable": (
            observability["staleness_detectability"]
            >= MIN_STALENESS_DETECTABILITY
        ),
    }
    failed = tuple(name for name, ok in conditions.items() if not ok)
    admission = {
        "admitted": not failed,
        "conditions": conditions,
        "failed_conditions": failed,
    }

    report = {
        "schema_version": "eta-gating-headroom-audit.v1",
        "claim_scope": "s3-gating-headroom-audit",
        "observation_protocol": "goal-ambiguous-junction.v5",
        "model_id": args.model_id,
        "model_source": args.model_source,
        "device": args.device,
        "corpus_seed": corpus.seed,
        "injection_layer_index": 20,
        "residual_width": 896,
        "steering_rank": args.steering_rank,
        "operator_seed": args.operator_seed,
        "updates": args.updates,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "reader_ridge_lambda": args.reader_ridge_lambda,
        "train_row_count": len(train_examples),
        "heldout_row_count": len(heldout_examples),
        "control_norm_cap": float(scorer.control_norm_cap),
        "post_switch_row_count": int(post_switch.sum()),
        "post_switch_fraction": float(post_switch.mean()),
        "arms_mean": arms_mean,
        "post_switch_arms_mean": {
            "noop": subset_mean(noop_rows, post_switch_indices),
            "always_on_belief": subset_mean(
                always_on_belief_rows, post_switch_indices
            ),
            "fresh_ceiling": subset_mean(
                fresh_ceiling_rows, post_switch_indices
            ),
        },
        "gating": gating,
        "observability": observability,
        "admission": admission,
        "thresholds": {
            "min_post_switch_fraction": MIN_POST_SWITCH_FRACTION,
            "min_gating_headroom_vs_alwayson": MIN_GATING_HEADROOM_VS_ALWAYSON,
            "min_gating_gain_vs_noop": MIN_GATING_GAIN_VS_NOOP,
            "min_staleness_detectability": MIN_STALENESS_DETECTABILITY,
        },
        "free_bias_present": False,
        "zero_code_strict_noop": zero_code_max_abs == 0.0,
        "zero_code_max_abs": zero_code_max_abs,
        "substrate_trainable_parameter_count": 0,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
    }

    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(
        _audit_markdown(report), encoding="utf-8"
    )
    result_files = ("report.json", "report.md")
    manifest = {
        "schema_version": "eta-gating-headroom-audit-manifest.v1",
        "experiment_id": "eta-s3a-gating-headroom-audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report["claim_scope"],
        "read_only_diagnostic": True,
        "controller_installed": False,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
        "substrate_trainable_parameter_count": 0,
        "gating_headroom_admitted": admission["admitted"],
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

    print(
        json.dumps(
            {
                "gating_headroom_admitted": admission["admitted"],
                "failed_conditions": failed,
                "post_switch_fraction": report["post_switch_fraction"],
                "arms_mean": arms_mean,
                "gating": gating,
                "observability": observability,
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
