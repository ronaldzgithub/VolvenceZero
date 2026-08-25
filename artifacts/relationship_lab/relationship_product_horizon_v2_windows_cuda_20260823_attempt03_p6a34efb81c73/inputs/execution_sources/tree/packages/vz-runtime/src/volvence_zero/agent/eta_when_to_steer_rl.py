"""S3 · Internal RL that learns WHEN to steer (from sparse outcome credit).

S3-prereq (08) proved read + steer; S3-A (gating headroom audit) proved that a
stale belief makes always-on steering catastrophic at post-switch junctions
(post-switch always-on 4.16 > noop 2.53), leaving a large, fully observable
gating headroom (oracle/pe-hard gate 1.09 vs always-on 1.79). This module tests
the third layer: can a bounded gate policy LEARN when to steer from sparse
outcome credit alone -- never given per-step correctness?

Design (frozen prereg: ``artifacts/eta_s3_internal_rl_prereg_20260805.json``):

* Sensor (frozen): the 08 linear reader gives a fresh read and a lagged belief
  read per junction; the executor is driven by the (non-oracle) belief.
* Executor (frozen): the C2 rank-8 multiplicative write; no free bias; strict
  zero-code no-op.
* Because the executor and belief are deterministic per row, the two possible
  outcomes ``noop_nll`` / ``steer_belief_nll`` are PRECOMPUTED once, so the RL
  loop is a fast table lookup -- the only GPU work is capture + executor train
  + a few NLL sweeps.
* Policy (the only online-updated component): a tiny softmax over PE proxies
  ``(belief_margin, fresh_margin, belief_disagrees_fresh, base_action_entropy)``
  choosing ``{noop, steer(belief)}``. Trained by a self-written REINFORCE with a
  running baseline. Credit is a single TERMINAL scalar per route episode,
  ``R = -mean(route chosen NLL)`` -- sparse in delivery, covering every in-route
  gate decision; true PE/outcome enters only credit, never the observation.

There is a known hard-rule ceiling (steer iff belief==fresh) reaching the
oracle gate; the claim is NOT that only RL reaches it, but that RL LEARNS to
approach it from sparse outcome credit within a bounded episode budget -- the
product-relevant question, since the companion task has no free per-step label.

Evidence lane only: no controller installed, no production wiring change, no
evaluation feedback into learning, no substrate/reader/executor training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Any

import numpy as np

from volvence_zero.agent.eta_conditional_steering_screen import (
    ACTION_PROMPT_SUFFIX,
    _ConditionalOperator,
    _subgoal_vocabulary,
    _train_operator,
    _zero_code_max_abs,
)
from volvence_zero.agent.eta_conflict_instrument import (
    ConflictJunctionRow,
    build_conflict_junction_rows,
)
from volvence_zero.agent.eta_proof_benchmark import ETAProofCorpus
from volvence_zero.agent.eta_read_steer_prereq import (
    _capture_examples,
    _labelled_rows,
    _per_row_baseline_nll,
    _per_row_controlled_nll,
    _stack_residuals_action,
    fit_condition_reader,
)
from volvence_zero.substrate import OpenWeightResidualRuntime


ETA_WHEN_TO_STEER_RL_SCHEMA_VERSION = "eta-when-to-steer-rl.v1"
_GATE_FEATURE_NAMES = (
    "belief_margin",
    "fresh_margin",
    "belief_disagrees_fresh",
    "base_action_entropy",
)


@dataclass(frozen=True)
class WhenToSteerThresholds:
    min_convergence_improvement_nll: float = 0.20
    min_gain_vs_noop_nll: float = 0.30
    min_gain_vs_always_on_nll: float = 0.20
    min_gain_vs_random_gate_nll: float = 0.20
    min_gate_selectivity: float = 0.30
    require_bootstrap_lower_positive: bool = True


@dataclass(frozen=True)
class SteerGateArmNLL:
    noop: float
    pe_gated_online: float
    always_on_belief: float
    random_gate: float
    oracle_gate_ceiling: float
    pe_hard_gate_ceiling: float
    fresh_ceiling: float


@dataclass(frozen=True)
class WhenToSteerEffect:
    mean: float
    ci_lower: float
    ci_upper: float
    route_count: int


@dataclass(frozen=True)
class WhenToSteerSeedPoint:
    seed: int
    arms: SteerGateArmNLL
    early_window_nll: float
    final_window_nll: float
    convergence_improvement_nll: float
    steer_rate_overall: float
    steer_rate_post_switch: float
    steer_rate_non_switch: float
    gate_selectivity: float
    gain_vs_noop: WhenToSteerEffect
    gain_vs_always_on: WhenToSteerEffect
    gain_vs_random_gate: WhenToSteerEffect
    selected_restart: int
    selection_train_nll: float
    policy_parameters_changed: bool


@dataclass(frozen=True)
class WhenToSteerAggregate:
    seed_count: int
    noop_nll_mean: float
    pe_gated_online_nll_mean: float
    always_on_belief_nll_mean: float
    random_gate_nll_mean: float
    oracle_gate_ceiling_nll_mean: float
    pe_hard_gate_ceiling_nll_mean: float
    fresh_ceiling_nll_mean: float
    convergence_improvement_nll_mean: float
    gate_selectivity_mean: float
    gain_vs_noop_ci_lower_min: float
    gain_vs_always_on_ci_lower_min: float
    gain_vs_random_gate_ci_lower_min: float


@dataclass(frozen=True)
class WhenToSteerAdmission:
    admitted: bool
    condition_convergence: bool
    condition_gain_vs_noop: bool
    condition_gain_vs_always_on: bool
    condition_gain_vs_random_gate: bool
    condition_gate_selectivity: bool
    condition_structural: bool
    failed_conditions: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class WhenToSteerReport:
    schema_version: str
    claim_scope: str
    observation_protocol: str
    model_id: str
    model_source: str
    device: str
    corpus_seed: int
    injection_layer_index: int
    residual_width: int
    steering_rank: int
    subgoal_class_count: int
    gate_feature_names: tuple[str, ...]
    train_row_count: int
    heldout_row_count: int
    post_switch_fraction: float
    control_norm_cap: float
    executor_updates: int
    executor_learning_rate: float
    policy_learning_rate: float
    policy_batch_cases: int
    entropy_coef: float
    init_noop_bias: float
    policy_restarts: int
    max_online_episodes: int
    eval_every: int
    convergence_window: int
    seed_schedule: tuple[int, ...]
    bootstrap_resamples: int
    bootstrap_confidence: float
    thresholds: WhenToSteerThresholds
    seed_points: tuple[WhenToSteerSeedPoint, ...]
    aggregate: WhenToSteerAggregate
    admission: WhenToSteerAdmission
    free_bias_present: bool
    zero_code_strict_noop: bool
    substrate_trainable_parameter_count: int
    reader_parameters_changed: bool
    executor_parameters_changed: bool
    production_wiring_changed: bool
    feedback_to_learning: bool
    description: str = ""


@dataclass(frozen=True)
class _RowRecord:
    case_id: str
    order_in_case: int
    post_switch: bool
    belief_correct: bool
    belief_agrees_fresh: bool
    belief_margin: float
    fresh_margin: float
    base_action_entropy: float
    noop_nll: float
    steer_belief_nll: float


class _GatePolicy:
    """Tiny softmax gate over PE proxies -> {noop, steer}. Only online update."""

    def __init__(
        self,
        *,
        torch: Any,
        feature_dim: int,
        seed: int,
        init_noop_bias: float = 0.0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self._torch = torch
        self._W = (
            torch.randn(feature_dim, 2, generator=generator) * 0.01
        ).requires_grad_(True)
        # Conservative prior: start biased toward noop (index 0) so the gate
        # must EARN each steer from reward, avoiding the always-steer attractor
        # that fresh-row credit otherwise locks in.
        self._b = torch.tensor([init_noop_bias, 0.0]).requires_grad_(True)

    def parameters(self) -> tuple[Any, ...]:
        return (self._W, self._b)

    def logits(self, features: Any) -> Any:
        return features @ self._W + self._b

    def log_probs(self, features: Any) -> Any:
        return self._torch.log_softmax(self.logits(features), dim=-1)

    def steer_decisions(self, features: Any) -> Any:
        return self.logits(features).argmax(dim=-1)


def _reader_scores(reader, residuals: np.ndarray) -> np.ndarray:
    standardized = (residuals - reader.feature_mean) / reader.feature_scale
    return standardized @ reader.weights


def _base_action_entropy(
    rows: tuple[ConflictJunctionRow, ...],
    *,
    scorer: Any,
    batch_size: int,
) -> list[float]:
    """Entropy of the base model over each junction's available next-moves.

    A per-row PE proxy of base uncertainty. Restricted to the row's available
    transitions (the legitimate option set at that junction).
    """

    flat_texts: list[str] = []
    flat_actions: list[int] = []
    spans: list[tuple[int, int]] = []
    for row in rows:
        text = row.observation_text + ACTION_PROMPT_SUFFIX
        start = len(flat_texts)
        for target in row.available_targets:
            flat_texts.append(text)
            flat_actions.append(scorer.action_index(f"move:{target}"))
        spans.append((start, len(flat_texts)))
    nlls = _per_row_baseline_nll(
        texts=tuple(flat_texts),
        action_indices=tuple(flat_actions),
        scorer=scorer,
        batch_size=batch_size,
    )
    entropies: list[float] = []
    for start, stop in spans:
        option_nlls = nlls[start:stop]
        weights = [math.exp(-value) for value in option_nlls]
        total = sum(weights)
        if total <= 0.0:
            entropies.append(0.0)
            continue
        probabilities = [weight / total for weight in weights]
        entropies.append(
            -sum(p * math.log(p) for p in probabilities if p > 0.0)
        )
    return entropies


def _standardize_columns(
    matrix: np.ndarray, continuous_columns: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(matrix.shape[1])
    scale = np.ones(matrix.shape[1])
    for column in continuous_columns:
        mean[column] = matrix[:, column].mean()
        scale[column] = matrix[:, column].std() + 1e-6
    return mean, scale


def _feature_matrix(records: tuple[_RowRecord, ...]) -> np.ndarray:
    return np.asarray(
        [
            [
                record.belief_margin,
                record.fresh_margin,
                1.0 if record.belief_agrees_fresh is False else 0.0,
                record.base_action_entropy,
            ]
            for record in records
        ],
        dtype=np.float64,
    )


def _route_bootstrap(
    *,
    case_ids: tuple[str, ...],
    per_row_effect: list[float],
    seed: int,
    resamples: int,
    confidence: float,
) -> WhenToSteerEffect:
    grouped: dict[str, list[float]] = {}
    for case_id, value in zip(case_ids, per_row_effect, strict=True):
        grouped.setdefault(case_id, []).append(value)
    route_effects = [
        statistics.fmean(values) for _, values in sorted(grouped.items())
    ]
    rng = random.Random(seed)
    count = len(route_effects)
    draws = sorted(
        statistics.fmean(route_effects[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = draws[min(max(int(tail * resamples), 0), resamples - 1)]
    upper = draws[min(max(int((1.0 - tail) * resamples) - 1, 0), resamples - 1)]
    return WhenToSteerEffect(
        mean=statistics.fmean(route_effects),
        ci_lower=lower,
        ci_upper=upper,
        route_count=count,
    )


def _build_records(
    *,
    examples: tuple[Any, ...],
    reader,
    noop_nll: list[float],
    steer_belief_nll: list[float],
    base_entropy: list[float],
) -> tuple[_RowRecord, ...]:
    context = np.asarray(
        [example.context_residual for example in examples], dtype=np.float64
    )
    scores = _reader_scores(reader, context)
    fresh_pred = scores.argmax(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    true_subgoal = np.asarray(
        [example.subgoal_index for example in examples], dtype=np.int64
    )
    last_in_case: dict[str, int] = {}
    order_in_case: dict[str, int] = {}
    records: list[_RowRecord] = []
    for index, example in enumerate(examples):
        previous = last_in_case.get(example.case_id)
        if previous is None:
            belief_index = index
            post_switch = False
        else:
            belief_index = previous
            post_switch = bool(true_subgoal[index] != true_subgoal[previous])
        belief_pred = int(fresh_pred[belief_index])
        order = order_in_case.get(example.case_id, 0)
        order_in_case[example.case_id] = order + 1
        last_in_case[example.case_id] = index
        records.append(
            _RowRecord(
                case_id=example.case_id,
                order_in_case=order,
                post_switch=post_switch,
                belief_correct=belief_pred == int(true_subgoal[index]),
                belief_agrees_fresh=belief_pred == int(fresh_pred[index]),
                belief_margin=float(margin[belief_index]),
                fresh_margin=float(margin[index]),
                base_action_entropy=base_entropy[index],
                noop_nll=noop_nll[index],
                steer_belief_nll=steer_belief_nll[index],
            )
        )
    return tuple(records)


def _evaluate_policy(
    *,
    torch: Any,
    policy: _GatePolicy,
    features: Any,
    records: tuple[_RowRecord, ...],
) -> tuple[list[float], list[bool]]:
    with torch.no_grad():
        decisions = policy.steer_decisions(features).tolist()
    chosen: list[float] = []
    steer_flags: list[bool] = []
    for record, decision in zip(records, decisions, strict=True):
        steer = bool(decision == 1)
        steer_flags.append(steer)
        chosen.append(record.steer_belief_nll if steer else record.noop_nll)
    return chosen, steer_flags


def _train_gate_policy(
    *,
    torch: Any,
    seed: int,
    train_records: tuple[_RowRecord, ...],
    train_features: Any,
    heldout_records: tuple[_RowRecord, ...],
    heldout_features: Any,
    max_online_episodes: int,
    policy_learning_rate: float,
    policy_batch_cases: int,
    entropy_coef: float,
    init_noop_bias: float,
    eval_every: int,
    baseline_beta: float,
    progress: Any | None,
) -> tuple[_GatePolicy, list[float], tuple[Any, ...]]:
    policy = _GatePolicy(
        torch=torch,
        feature_dim=train_features.shape[1],
        seed=seed,
        init_noop_bias=init_noop_bias,
    )
    initial = tuple(param.detach().clone() for param in policy.parameters())
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy_learning_rate)
    action_generator = torch.Generator().manual_seed(seed * 2 + 1)
    episode_rng = random.Random(seed * 3 + 7)

    case_rows: dict[str, list[int]] = {}
    for index, record in enumerate(train_records):
        case_rows.setdefault(record.case_id, []).append(index)
    case_ids = sorted(case_rows)
    noop_tensor = torch.tensor(
        [record.noop_nll for record in train_records], dtype=torch.float32
    )
    steer_tensor = torch.tensor(
        [record.steer_belief_nll for record in train_records],
        dtype=torch.float32,
    )

    baseline = 0.0
    reward_var = 1.0
    # Trajectory starts at the random-init policy so convergence improvement is
    # measured from the untrained baseline (robust to fast convergence).
    initial_chosen, _ = _evaluate_policy(
        torch=torch,
        policy=policy,
        features=heldout_features,
        records=heldout_records,
    )
    heldout_trajectory: list[float] = [statistics.fmean(initial_chosen)]
    episodes_used = 0
    next_eval = eval_every
    # Minibatch REINFORCE: each update samples ``policy_batch_cases`` route
    # episodes and averages their advantage-weighted score, cutting the
    # terminal-credit variance that otherwise blends fresh/stale credit.
    while episodes_used < max_online_episodes:
        batch = min(policy_batch_cases, max_online_episodes - episodes_used)
        batch_rewards: list[float] = []
        step_log_probs = []
        step_actions = []
        entropies = []
        for _ in range(batch):
            case = case_ids[episode_rng.randrange(len(case_ids))]
            indices = torch.tensor(case_rows[case], dtype=torch.long)
            log_probs = policy.log_probs(train_features[indices])
            actions = torch.multinomial(
                log_probs.detach().exp(),
                num_samples=1,
                generator=action_generator,
            ).squeeze(1)
            chosen = torch.where(
                actions == 1, steer_tensor[indices], noop_tensor[indices]
            )
            batch_rewards.append(float(-chosen.mean().item()))
            step_log_probs.append(log_probs)
            step_actions.append(actions)
            entropies.append(-(log_probs.exp() * log_probs).sum(dim=-1).mean())
        mean_reward = statistics.fmean(batch_rewards)
        baseline = baseline_beta * baseline + (1.0 - baseline_beta) * mean_reward
        batch_var = statistics.pvariance(batch_rewards) if batch > 1 else 0.0
        reward_var = baseline_beta * reward_var + (1.0 - baseline_beta) * batch_var
        reward_std = math.sqrt(reward_var) + 1e-6
        # Advantage normalisation keeps the gradient alive near the always-steer
        # equilibrium (where raw advantages collapse to ~0); entropy keeps the
        # gate exploring noop on post-switch rows instead of collapsing.
        losses = []
        for reward, log_probs, actions in zip(
            batch_rewards, step_log_probs, step_actions, strict=True
        ):
            advantage = (reward - baseline) / reward_std
            selected = log_probs[torch.arange(len(actions)), actions].sum()
            losses.append(-advantage * selected)
        loss = torch.stack(losses).mean() - entropy_coef * torch.stack(
            entropies
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episodes_used += batch
        if episodes_used >= next_eval or episodes_used >= max_online_episodes:
            next_eval += eval_every
            chosen_nll, _ = _evaluate_policy(
                torch=torch,
                policy=policy,
                features=heldout_features,
                records=heldout_records,
            )
            heldout_trajectory.append(statistics.fmean(chosen_nll))
            if progress is not None:
                progress(
                    f"S3 gate seed={seed} ep={episodes_used}/"
                    f"{max_online_episodes} heldout_nll="
                    f"{heldout_trajectory[-1]:.4f}"
                )
    return policy, heldout_trajectory, initial


def _run_seed(
    *,
    torch: Any,
    seed: int,
    train_records: tuple[_RowRecord, ...],
    train_features: Any,
    heldout_records: tuple[_RowRecord, ...],
    heldout_features: Any,
    heldout_fresh_ceiling: list[float],
    max_online_episodes: int,
    policy_learning_rate: float,
    policy_batch_cases: int,
    entropy_coef: float,
    init_noop_bias: float,
    policy_restarts: int,
    eval_every: int,
    convergence_window: int,
    baseline_beta: float,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    progress: Any | None,
) -> WhenToSteerSeedPoint:
    # Robustification: run several restarts and keep the one with the best
    # TRAIN chosen-NLL. Selection uses only train rows (never the heldout test
    # metric of the verdict), so it is honest model selection -- it rescues
    # the occasional init that collapses to always-steer without touching any
    # decision threshold.
    best: dict[str, Any] | None = None
    for restart in range(policy_restarts):
        restart_seed = seed * 1000 + restart
        candidate, trajectory, initial = _train_gate_policy(
            torch=torch,
            seed=restart_seed,
            train_records=train_records,
            train_features=train_features,
            heldout_records=heldout_records,
            heldout_features=heldout_features,
            max_online_episodes=max_online_episodes,
            policy_learning_rate=policy_learning_rate,
            policy_batch_cases=policy_batch_cases,
            entropy_coef=entropy_coef,
            init_noop_bias=init_noop_bias,
            eval_every=eval_every,
            baseline_beta=baseline_beta,
            progress=progress,
        )
        train_chosen, _ = _evaluate_policy(
            torch=torch,
            policy=candidate,
            features=train_features,
            records=train_records,
        )
        train_nll = statistics.fmean(train_chosen)
        if best is None or train_nll < best["train_nll"]:
            best = {
                "policy": candidate,
                "trajectory": trajectory,
                "initial": initial,
                "train_nll": train_nll,
                "restart": restart,
            }
    policy = best["policy"]
    trajectory = best["trajectory"]
    initial = best["initial"]
    selected_restart = best["restart"]
    selection_train_nll = best["train_nll"]
    window = min(convergence_window, len(trajectory))
    # trajectory[0] is the untrained-policy heldout NLL (recorded before any
    # update); improvement is measured against it, robust to fast convergence.
    early_window = trajectory[0]
    final_window = statistics.fmean(trajectory[-window:])

    chosen_nll, steer_flags = _evaluate_policy(
        torch=torch,
        policy=policy,
        features=heldout_features,
        records=heldout_records,
    )
    case_ids = tuple(record.case_id for record in heldout_records)
    noop_rows = [record.noop_nll for record in heldout_records]
    always_on_rows = [record.steer_belief_nll for record in heldout_records]
    steer_rate = statistics.fmean(1.0 if flag else 0.0 for flag in steer_flags)
    random_gate_rows = [
        steer_rate * always_on_rows[index]
        + (1.0 - steer_rate) * noop_rows[index]
        for index in range(len(noop_rows))
    ]
    oracle_rows = [
        record.steer_belief_nll if record.belief_correct else record.noop_nll
        for record in heldout_records
    ]
    pe_hard_rows = [
        record.steer_belief_nll if record.belief_agrees_fresh else record.noop_nll
        for record in heldout_records
    ]

    post_switch_flags = [record.post_switch for record in heldout_records]
    post_steer = [
        steer_flags[index]
        for index in range(len(steer_flags))
        if post_switch_flags[index]
    ]
    non_steer = [
        steer_flags[index]
        for index in range(len(steer_flags))
        if not post_switch_flags[index]
    ]
    steer_rate_post = (
        statistics.fmean(1.0 if f else 0.0 for f in post_steer)
        if post_steer
        else 0.0
    )
    steer_rate_non = (
        statistics.fmean(1.0 if f else 0.0 for f in non_steer)
        if non_steer
        else 0.0
    )

    gain_vs_noop = _route_bootstrap(
        case_ids=case_ids,
        per_row_effect=[
            noop_rows[i] - chosen_nll[i] for i in range(len(chosen_nll))
        ],
        seed=seed * 101 + 3,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
    )
    gain_vs_always_on = _route_bootstrap(
        case_ids=case_ids,
        per_row_effect=[
            always_on_rows[i] - chosen_nll[i] for i in range(len(chosen_nll))
        ],
        seed=seed * 101 + 5,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
    )
    gain_vs_random = _route_bootstrap(
        case_ids=case_ids,
        per_row_effect=[
            random_gate_rows[i] - chosen_nll[i] for i in range(len(chosen_nll))
        ],
        seed=seed * 101 + 7,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
    )
    changed = any(
        float((a - b.detach()).abs().max()) > 1e-8
        for a, b in zip(initial, policy.parameters(), strict=True)
    )
    return WhenToSteerSeedPoint(
        seed=seed,
        arms=SteerGateArmNLL(
            noop=statistics.fmean(noop_rows),
            pe_gated_online=statistics.fmean(chosen_nll),
            always_on_belief=statistics.fmean(always_on_rows),
            random_gate=statistics.fmean(random_gate_rows),
            oracle_gate_ceiling=statistics.fmean(oracle_rows),
            pe_hard_gate_ceiling=statistics.fmean(pe_hard_rows),
            fresh_ceiling=statistics.fmean(heldout_fresh_ceiling),
        ),
        early_window_nll=early_window,
        final_window_nll=final_window,
        convergence_improvement_nll=early_window - final_window,
        steer_rate_overall=steer_rate,
        steer_rate_post_switch=steer_rate_post,
        steer_rate_non_switch=steer_rate_non,
        gate_selectivity=steer_rate_non - steer_rate_post,
        gain_vs_noop=gain_vs_noop,
        gain_vs_always_on=gain_vs_always_on,
        gain_vs_random_gate=gain_vs_random,
        selected_restart=selected_restart,
        selection_train_nll=selection_train_nll,
        policy_parameters_changed=changed,
    )


def assess_when_to_steer(
    *,
    aggregate: WhenToSteerAggregate,
    thresholds: WhenToSteerThresholds,
    free_bias_present: bool,
    zero_code_strict_noop: bool,
    substrate_trainable_parameter_count: int,
    reader_parameters_changed: bool,
    executor_parameters_changed: bool,
    policy_parameters_changed: bool,
) -> WhenToSteerAdmission:
    lower_ok = not thresholds.require_bootstrap_lower_positive
    conditions = {
        "convergence": (
            aggregate.convergence_improvement_nll_mean
            >= thresholds.min_convergence_improvement_nll
        ),
        "gain-vs-noop": (
            aggregate.pe_gated_online_nll_mean
            <= aggregate.noop_nll_mean - thresholds.min_gain_vs_noop_nll
            and (lower_ok or aggregate.gain_vs_noop_ci_lower_min > 0.0)
        ),
        "gain-vs-always-on": (
            aggregate.pe_gated_online_nll_mean
            <= aggregate.always_on_belief_nll_mean
            - thresholds.min_gain_vs_always_on_nll
            and (lower_ok or aggregate.gain_vs_always_on_ci_lower_min > 0.0)
        ),
        "gain-vs-random-gate": (
            aggregate.pe_gated_online_nll_mean
            <= aggregate.random_gate_nll_mean
            - thresholds.min_gain_vs_random_gate_nll
            and (lower_ok or aggregate.gain_vs_random_gate_ci_lower_min > 0.0)
        ),
        "gate-selectivity": (
            aggregate.gate_selectivity_mean >= thresholds.min_gate_selectivity
        ),
        "structural-integrity": (
            not free_bias_present
            and zero_code_strict_noop
            and substrate_trainable_parameter_count == 0
            and not reader_parameters_changed
            and not executor_parameters_changed
            and policy_parameters_changed
        ),
    }
    failed = tuple(name for name, ok in conditions.items() if not ok)
    return WhenToSteerAdmission(
        admitted=not failed,
        condition_convergence=conditions["convergence"],
        condition_gain_vs_noop=conditions["gain-vs-noop"],
        condition_gain_vs_always_on=conditions["gain-vs-always-on"],
        condition_gain_vs_random_gate=conditions["gain-vs-random-gate"],
        condition_gate_selectivity=conditions["gate-selectivity"],
        condition_structural=conditions["structural-integrity"],
        failed_conditions=failed,
        description=(
            "Gate policy learned WHEN to steer from sparse outcome credit: "
            "beats noop / always-on / random-gate and concentrates steering "
            "where the belief is fresh."
            if not failed
            else "When-to-steer prerequisite blocked: " + ", ".join(failed)
        ),
    )


def _aggregate(points: tuple[WhenToSteerSeedPoint, ...]) -> WhenToSteerAggregate:
    def mean(selector: Any) -> float:
        return statistics.fmean(selector(point) for point in points)

    return WhenToSteerAggregate(
        seed_count=len(points),
        noop_nll_mean=mean(lambda p: p.arms.noop),
        pe_gated_online_nll_mean=mean(lambda p: p.arms.pe_gated_online),
        always_on_belief_nll_mean=mean(lambda p: p.arms.always_on_belief),
        random_gate_nll_mean=mean(lambda p: p.arms.random_gate),
        oracle_gate_ceiling_nll_mean=mean(lambda p: p.arms.oracle_gate_ceiling),
        pe_hard_gate_ceiling_nll_mean=mean(
            lambda p: p.arms.pe_hard_gate_ceiling
        ),
        fresh_ceiling_nll_mean=mean(lambda p: p.arms.fresh_ceiling),
        convergence_improvement_nll_mean=mean(
            lambda p: p.convergence_improvement_nll
        ),
        gate_selectivity_mean=mean(lambda p: p.gate_selectivity),
        gain_vs_noop_ci_lower_min=min(
            point.gain_vs_noop.ci_lower for point in points
        ),
        gain_vs_always_on_ci_lower_min=min(
            point.gain_vs_always_on.ci_lower for point in points
        ),
        gain_vs_random_gate_ci_lower_min=min(
            point.gain_vs_random_gate.ci_lower for point in points
        ),
    )


def run_eta_when_to_steer_rl(
    *,
    corpus: ETAProofCorpus,
    runtime: OpenWeightResidualRuntime,
    scorer: Any,
    model_source: str,
    device: str,
    injection_layer_index: int = 20,
    residual_width: int = 896,
    steering_rank: int = 8,
    executor_updates: int = 80,
    executor_learning_rate: float = 0.01,
    reader_ridge_lambda: float = 10.0,
    batch_size: int = 32,
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
    import torch

    if steering_rank < 1 or steering_rank > residual_width:
        raise ValueError("steering_rank must be in [1, residual_width]")
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("seed_schedule must be non-empty and unique")
    if max_online_episodes < eval_every or eval_every < 1:
        raise ValueError("episodes/eval_every misconfigured")
    if scorer.trainable_parameters():
        raise RuntimeError("S3 requires a frozen substrate scorer")

    subgoal_vocabulary = _subgoal_vocabulary(corpus)
    subgoal_index = {name: index for index, name in enumerate(subgoal_vocabulary)}
    class_count = len(subgoal_vocabulary)

    train_rows = _labelled_rows(build_conflict_junction_rows(corpus, split="train"))
    heldout_rows = _labelled_rows(
        build_conflict_junction_rows(corpus, split="heldout")
    )
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
        action_indices=tuple(
            example.action_index for example in train_examples
        ),
        texts=tuple(example.observation_text for example in train_examples),
        scorer=scorer,
        updates=executor_updates,
        learning_rate=executor_learning_rate,
        batch_size=batch_size,
        seed=0,
        progress=progress,
        label="executor",
    )
    # Executor is trained once at setup then FROZEN for the online RL phase.
    # The structural invariant is "unchanged during RL", so snapshot here and
    # compare after the policy loop (which never touches the operator).
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
    train_features = torch.tensor(
        (train_matrix - mean) / scale, dtype=torch.float32
    )
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
        schema_version=ETA_WHEN_TO_STEER_RL_SCHEMA_VERSION,
        claim_scope="conditional-steering-internal-rl-when-to-steer",
        observation_protocol="goal-ambiguous-junction.v5",
        model_id=runtime.model_id,
        model_source=model_source,
        device=device,
        corpus_seed=corpus.seed,
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
            "Internal RL gate over PE proxies; "
            f"admitted={admission.admitted}."
        ),
    )


def _precompute_records(
    *,
    torch: Any,
    examples: tuple[Any, ...],
    rows: tuple[ConflictJunctionRow, ...],
    reader,
    operator: _ConditionalOperator,
    scorer: Any,
    batch_size: int,
) -> tuple[_RowRecord, ...]:
    residuals = _stack_residuals_action(torch, examples)
    belief_pred = _belief_predictions(reader, examples)
    texts = tuple(example.observation_text for example in examples)
    action_indices = tuple(example.action_index for example in examples)
    noop_nll = _per_row_baseline_nll(
        texts=texts,
        action_indices=action_indices,
        scorer=scorer,
        batch_size=batch_size,
    )
    with torch.no_grad():
        belief_deltas = operator.deltas(
            residuals=residuals,
            subgoal_indices=torch.tensor(belief_pred, dtype=torch.long),
        )
    steer_belief_nll = _per_row_controlled_nll(
        deltas=belief_deltas,
        texts=texts,
        action_indices=action_indices,
        scorer=scorer,
        batch_size=batch_size,
    )
    base_entropy = _base_action_entropy(
        rows, scorer=scorer, batch_size=batch_size
    )
    return _build_records(
        examples=examples,
        reader=reader,
        noop_nll=noop_nll,
        steer_belief_nll=steer_belief_nll,
        base_entropy=base_entropy,
    )


def _belief_predictions(reader, examples: tuple[Any, ...]) -> list[int]:
    context = np.asarray(
        [example.context_residual for example in examples], dtype=np.float64
    )
    fresh_pred = _reader_scores(reader, context).argmax(axis=1)
    last_in_case: dict[str, int] = {}
    belief: list[int] = []
    for index, example in enumerate(examples):
        previous = last_in_case.get(example.case_id, index)
        belief.append(int(fresh_pred[previous]))
        last_in_case[example.case_id] = index
    return belief


def _fresh_ceiling_nll(
    *,
    torch: Any,
    examples: tuple[Any, ...],
    reader,
    operator: _ConditionalOperator,
    scorer: Any,
    batch_size: int,
) -> list[float]:
    residuals = _stack_residuals_action(torch, examples)
    context = np.asarray(
        [example.context_residual for example in examples], dtype=np.float64
    )
    fresh_pred = _reader_scores(reader, context).argmax(axis=1)
    with torch.no_grad():
        deltas = operator.deltas(
            residuals=residuals,
            subgoal_indices=torch.tensor(fresh_pred, dtype=torch.long),
        )
    return _per_row_controlled_nll(
        deltas=deltas,
        texts=tuple(example.observation_text for example in examples),
        action_indices=tuple(example.action_index for example in examples),
        scorer=scorer,
        batch_size=batch_size,
    )


__all__ = [
    "ETA_WHEN_TO_STEER_RL_SCHEMA_VERSION",
    "SteerGateArmNLL",
    "WhenToSteerAdmission",
    "WhenToSteerAggregate",
    "WhenToSteerEffect",
    "WhenToSteerReport",
    "WhenToSteerSeedPoint",
    "WhenToSteerThresholds",
    "assess_when_to_steer",
    "run_eta_when_to_steer_rl",
]
