"""Packet 3：S3-E when-to-steer RL 的编程域复刻。

结构映射（对齐 `eta_when_to_steer_rl` 的 goal-ambiguous junction 协议）：

- **route / case** = 一个通过的 episode（终局信用源：route 上 expert
  动作 NLL 均值——只有通过 episode 的动作序列有 expert 资格）；
- **junction** = episode 内一个决策点；**expert 动作** = 该点实际动作；
- **潜在条件（subgoal 等价物）** = 该点的 expert 下一步动作类
  （investigate / edit / test / submit）。观测文本 goal-stripped
  （无任务描述），revealed 文本携带任务——reader 从 revealed 残差读
  出条件，gate 学"何时用（滞后的）belief 去 steer"；相位切换
  （如 edit→test）提供真实 post_switch 结构；
- **几何**：Qwen2.5-Coder-1.5B fp32、注入层 13、宽 1536、rank 8
  （前置 a 判词）——artifact 几何绑定，必须对本基底重 fit。

算法链**整体复用** vz-runtime 的 S3-E 实现（`_capture_examples` →
reader/executor fit → `_precompute_records` → `_run_seed`(REINFORCE ×
multi-restart) → `_aggregate` → `assess_when_to_steer` 六门）——计划
勘探确认这些函数对 row 只做鸭子字段访问，本模块提供同构 row 而不
复制平行实现（vz-runtime 是该算法的 SSOT）。
"""

from __future__ import annotations

import hashlib
import statistics

from dataclasses import dataclass
from typing import Any

from volvence_zero.agent.eta_conditional_steering_screen import (
    _ConditionalOperator,
    _train_operator,
    _zero_code_max_abs,
)
from volvence_zero.agent.eta_read_steer_prereq import (
    _capture_examples,
    _stack_residuals_action,
    fit_condition_reader,
)
from volvence_zero.agent.eta_when_to_steer_rl import (
    WhenToSteerReport,
    WhenToSteerThresholds,
    _aggregate,
    _feature_matrix,
    _fresh_ceiling_nll,
    _GATE_FEATURE_NAMES,
    _precompute_records,
    _run_seed,
    _standardize_columns,
    assess_when_to_steer,
)

from lifeform_domain_coding.lab.junctions import (
    JUNCTION_ACTIONS,
    collect_junctions,
)

CODING_WHEN_TO_STEER_SCHEMA_VERSION = "coding-lab-when-to-steer-rl.v1"
CODING_OBSERVATION_PROTOCOL = "coding-goal-stripped-junction.v1"

#: Scorer option ids follow the ETA "move:{target}" convention so the
#: reused `_base_action_entropy` (which builds ids from
#: ``row.available_targets``) resolves without adaptation. Surfaces are
#: bare identifiers (ETA precedent): the restricted action softmax
#: disambiguates on the FIRST token, so no shared leading space.
ACTION_SURFACES: dict[str, str] = {
    "investigate": "investigate",
    "edit": "edit",
    "test": "test",
    "submit": "submit",
}


@dataclass(frozen=True)
class CodingJunctionRow:
    """Duck-type twin of ``ConflictJunctionRow`` for the coding domain."""

    case_id: str
    split: str
    step_index: int
    current_location_id: str
    available_targets: tuple[str, ...]
    observation_text: str
    subgoal_revealed_text: str
    active_subgoal: str
    expert_action_id: str
    local_view_id: str


def build_coding_junction_rows(
    trajectory_paths: tuple[Any, ...],
    *,
    heldout_fraction: float = 0.3,
) -> tuple[tuple[CodingJunctionRow, ...], tuple[CodingJunctionRow, ...]]:
    """Rows from PASSING episodes only; case-level deterministic split.

    Split hashes the episode identity (trajectory sha) so a route never
    straddles train/heldout — the ETA corpus's case-disjoint discipline.
    """

    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    records = collect_junctions(tuple(trajectory_paths))
    threshold = int(heldout_fraction * 2**32)
    train: list[CodingJunctionRow] = []
    heldout: list[CodingJunctionRow] = []
    step_by_case: dict[str, int] = {}
    for record in records:
        if not record.episode_passed:
            continue
        if record.action_taken not in JUNCTION_ACTIONS:
            continue
        case_id = record.trajectory_sha256
        digest = hashlib.sha256(case_id.encode("utf-8")).digest()
        split = "heldout" if int.from_bytes(digest[:4], "big") < threshold else "train"
        step = step_by_case.get(case_id, 0)
        step_by_case[case_id] = step + 1
        row = CodingJunctionRow(
            case_id=case_id,
            split=split,
            step_index=step,
            current_location_id=record.state_key,
            available_targets=JUNCTION_ACTIONS,
            observation_text=record.observation_text,
            subgoal_revealed_text=record.revealed_text,
            active_subgoal=record.action_taken,
            expert_action_id=f"move:{record.action_taken}",
            local_view_id=record.state_key,
        )
        (heldout if split == "heldout" else train).append(row)
    return tuple(train), tuple(heldout)


def rows_manifest(
    train_rows: tuple[CodingJunctionRow, ...],
    heldout_rows: tuple[CodingJunctionRow, ...],
) -> dict:
    def cases(rows: tuple[CodingJunctionRow, ...]) -> int:
        return len({row.case_id for row in rows})

    switches = 0
    previous: dict[str, str] = {}
    for row in (*train_rows, *heldout_rows):
        last = previous.get(row.case_id)
        if last is not None and last != row.active_subgoal:
            switches += 1
        previous[row.case_id] = row.active_subgoal
    return {
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "train_cases": cases(train_rows),
        "heldout_cases": cases(heldout_rows),
        "phase_switches": switches,
    }


def run_coding_when_to_steer_rl(
    *,
    train_rows: tuple[CodingJunctionRow, ...],
    heldout_rows: tuple[CodingJunctionRow, ...],
    runtime: Any,
    scorer: Any,
    model_source: str,
    device: str,
    corpus_fingerprint: int,
    injection_layer_index: int = 13,
    residual_width: int = 1536,
    steering_rank: int = 8,
    executor_updates: int = 80,
    executor_learning_rate: float = 0.01,
    reader_ridge_lambda: float = 10.0,
    batch_size: int = 16,
    policy_learning_rate: float = 0.1,
    policy_batch_cases: int = 8,
    entropy_coef: float = 0.1,
    init_noop_bias: float = 0.0,
    policy_restarts: int = 4,
    max_online_episodes: int = 1200,
    eval_every: int = 80,
    convergence_window: int = 3,
    baseline_beta: float = 0.9,
    seed_schedule: tuple[int, ...] = (0, 1, 2, 3, 4),
    bootstrap_resamples: int = 5000,
    bootstrap_confidence: float = 0.95,
    thresholds: WhenToSteerThresholds | None = None,
    progress: Any | None = None,
) -> WhenToSteerReport:
    """Mirror of ``run_eta_when_to_steer_rl`` over coding junction rows."""

    import torch  # noqa: PLC0415 — heavyweight optional dependency

    if steering_rank < 1 or steering_rank > residual_width:
        raise ValueError("steering_rank must be in [1, residual_width]")
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("seed_schedule must be non-empty and unique")
    if max_online_episodes < eval_every or eval_every < 1:
        raise ValueError("episodes/eval_every misconfigured")
    if scorer.trainable_parameters():
        raise RuntimeError("Packet 3 requires a frozen substrate scorer")
    if not train_rows or not heldout_rows:
        raise ValueError("both splits must be non-empty")

    subgoal_index = {name: index for index, name in enumerate(JUNCTION_ACTIONS)}
    class_count = len(JUNCTION_ACTIONS)

    train_examples = _capture_examples(
        train_rows,
        runtime=runtime,
        scorer=scorer,
        subgoal_index=subgoal_index,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        progress=progress,
        split_label="train",
    )
    heldout_examples = _capture_examples(
        heldout_rows,
        runtime=runtime,
        scorer=scorer,
        subgoal_index=subgoal_index,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        progress=progress,
        split_label="heldout",
    )
    reader = fit_condition_reader(
        train_examples, class_count=class_count, ridge_lambda=reader_ridge_lambda
    )

    train_residuals = _stack_residuals_action(torch, train_examples)
    train_subgoals = torch.tensor(
        [example.subgoal_index for example in train_examples], dtype=torch.long
    )
    operator = _ConditionalOperator(
        torch=torch,
        width=residual_width,
        rank=steering_rank,
        class_count=class_count,
        conditional=True,
        seed=0,
    )
    _train_operator(
        torch=torch,
        operator=operator,
        residuals=train_residuals,
        subgoal_indices=train_subgoals,
        action_indices=tuple(example.action_index for example in train_examples),
        texts=tuple(example.observation_text for example in train_examples),
        scorer=scorer,
        updates=executor_updates,
        learning_rate=executor_learning_rate,
        batch_size=batch_size,
        seed=0,
        progress=progress,
        label="executor",
    )
    executor_after_setup = tuple(
        param.detach().clone() for param in operator.parameters()
    )

    train_records = _precompute_records(
        torch=torch,
        examples=train_examples,
        rows=train_rows,
        reader=reader,
        operator=operator,
        scorer=scorer,
        batch_size=batch_size,
    )
    heldout_records = _precompute_records(
        torch=torch,
        examples=heldout_examples,
        rows=heldout_rows,
        reader=reader,
        operator=operator,
        scorer=scorer,
        batch_size=batch_size,
    )
    heldout_fresh_ceiling = _fresh_ceiling_nll(
        torch=torch,
        examples=heldout_examples,
        reader=reader,
        operator=operator,
        scorer=scorer,
        batch_size=batch_size,
    )
    with torch.no_grad():
        zero_code_max_abs = _zero_code_max_abs(
            torch=torch,
            operator=operator,
            residuals=_stack_residuals_action(torch, heldout_examples),
            subgoal_indices=torch.tensor(
                [example.subgoal_index for example in heldout_examples],
                dtype=torch.long,
            ),
        )
    executor_changed = any(
        float((a - b.detach()).abs().max()) > 1e-8
        for a, b in zip(executor_after_setup, operator.parameters(), strict=True)
    )

    train_matrix = _feature_matrix(train_records)
    heldout_matrix = _feature_matrix(heldout_records)
    mean, scale = _standardize_columns(train_matrix, continuous_columns=(0, 1, 3))
    train_features = torch.tensor((train_matrix - mean) / scale, dtype=torch.float32)
    heldout_features = torch.tensor(
        (heldout_matrix - mean) / scale, dtype=torch.float32
    )

    seed_points = tuple(
        _run_seed(
            torch=torch,
            seed=seed,
            train_records=train_records,
            train_features=train_features,
            heldout_records=heldout_records,
            heldout_features=heldout_features,
            heldout_fresh_ceiling=heldout_fresh_ceiling,
            max_online_episodes=max_online_episodes,
            policy_learning_rate=policy_learning_rate,
            policy_batch_cases=policy_batch_cases,
            entropy_coef=entropy_coef,
            init_noop_bias=init_noop_bias,
            policy_restarts=policy_restarts,
            eval_every=eval_every,
            convergence_window=convergence_window,
            baseline_beta=baseline_beta,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
            progress=progress,
        )
        for seed in seed_schedule
    )
    aggregate = _aggregate(seed_points)
    active_thresholds = thresholds or WhenToSteerThresholds()
    admission = assess_when_to_steer(
        aggregate=aggregate,
        thresholds=active_thresholds,
        free_bias_present=False,
        zero_code_strict_noop=zero_code_max_abs == 0.0,
        substrate_trainable_parameter_count=0,
        reader_parameters_changed=False,
        executor_parameters_changed=executor_changed,
        policy_parameters_changed=all(
            point.policy_parameters_changed for point in seed_points
        ),
    )
    post_switch_fraction = statistics.fmean(
        1.0 if record.post_switch else 0.0 for record in heldout_records
    )
    return WhenToSteerReport(
        schema_version=CODING_WHEN_TO_STEER_SCHEMA_VERSION,
        claim_scope="coding-junction-internal-rl-when-to-steer",
        observation_protocol=CODING_OBSERVATION_PROTOCOL,
        model_id=runtime.model_id,
        model_source=model_source,
        device=device,
        corpus_seed=corpus_fingerprint,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        steering_rank=steering_rank,
        subgoal_class_count=class_count,
        gate_feature_names=_GATE_FEATURE_NAMES,
        train_row_count=len(train_records),
        heldout_row_count=len(heldout_records),
        post_switch_fraction=post_switch_fraction,
        control_norm_cap=float(scorer.control_norm_cap),
        executor_updates=executor_updates,
        executor_learning_rate=executor_learning_rate,
        policy_learning_rate=policy_learning_rate,
        policy_batch_cases=policy_batch_cases,
        entropy_coef=entropy_coef,
        init_noop_bias=init_noop_bias,
        policy_restarts=policy_restarts,
        max_online_episodes=max_online_episodes,
        eval_every=eval_every,
        convergence_window=convergence_window,
        seed_schedule=tuple(seed_schedule),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        thresholds=active_thresholds,
        seed_points=seed_points,
        aggregate=aggregate,
        admission=admission,
        free_bias_present=False,
        zero_code_strict_noop=zero_code_max_abs == 0.0,
        substrate_trainable_parameter_count=0,
        reader_parameters_changed=False,
        executor_parameters_changed=executor_changed,
        production_wiring_changed=False,
        feedback_to_learning=False,
        description=(
            "Coding-domain internal RL gate over PE proxies; "
            f"admitted={admission.admitted}."
        ),
    )


__all__ = [
    "ACTION_SURFACES",
    "CODING_OBSERVATION_PROTOCOL",
    "CODING_WHEN_TO_STEER_SCHEMA_VERSION",
    "CodingJunctionRow",
    "build_coding_junction_rows",
    "rows_manifest",
    "run_coding_when_to_steer_rl",
]
