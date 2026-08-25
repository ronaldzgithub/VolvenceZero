"""P2c · C2 — conditional learned steering screen on the conflict instrument.

C1 validated the goal-ambiguous junction instrument: under goal stripping a
constant operator misclassifies ~46% of junctions, the base model is uncertain
(expert NLL 2.81), and the active subgoal is the single bit that closes the gap
(revealed NLL 0.22). This screen asks the decisive question P2b framed:

    Can a learned, subgoal-CONDITIONAL low-rank intervention on the frozen
    residual close that gap, and does conditioning beat an equal-budget
    UNCONDITIONAL operator?

Executor (ReFT / B-screen lineage; no free bias, strict zero-code no-op):

    delta_i = U @ ( tanh(Z[k_i]) ⊙ (Vᵀ h_i) )

where ``h_i`` is the layer-20 residual of the goal-stripped prefix, ``U, V ∈
R[width×rank]`` are shared learned factors, and ``Z[k]`` is a per-subgoal code.
The write is purely multiplicative on ``h_i`` (delta→0 as h→0: no additive
bias) and vanishes exactly when the code is zero (strict no-op). Conditioning
lives entirely in the per-subgoal code, so the ``unconditional`` control (a
single shared code, same rank and budget) is a faithful constant operator.

Distortion is the expert-action NLL through the frozen steered model
(``scorer.action_nll`` / ``controlled_action_nll``), the repository's Eq.3
distortion. No substrate parameter is trained, no reward is reinvented, no
production wiring changes. The screen only decides whether the conditional
learned intervention earns an authoritative sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import statistics
from typing import Any

from volvence_zero.agent.eta_conflict_instrument import (
    ConflictJunctionRow,
    build_conflict_junction_rows,
)
from volvence_zero.agent.eta_proof_benchmark import ETAProofCorpus
from volvence_zero.substrate import OpenWeightResidualRuntime


ETA_CONDITIONAL_STEERING_SCHEMA_VERSION = "eta-conditional-steering-screen.v1"
ACTION_PROMPT_SUFFIX = "\nNext move:"


@dataclass(frozen=True)
class ConditionalSteeringThresholds:
    min_gap_closed_nll: float = 0.30
    min_conditional_advantage_nll: float = 0.15
    min_condition_specificity_nll: float = 0.15
    min_gap_closed_fraction: float = 0.30


@dataclass(frozen=True)
class ConditionalSteeringArmNLL:
    noop: float
    conditional: float
    unconditional: float
    random_condition: float
    subgoal_revealed_ceiling: float


@dataclass(frozen=True)
class ConditionalSteeringSeedPoint:
    seed: int
    heldout: ConditionalSteeringArmNLL
    train_conditional_nll: float
    gap_closed_nll: float
    conditional_advantage_nll: float
    condition_specificity_nll: float
    gap_closed_fraction: float
    conditional_low_rank_parameters_changed: bool
    unconditional_low_rank_parameters_changed: bool
    zero_code_strict_noop_max_abs: float
    final_train_loss: float


@dataclass(frozen=True)
class ConditionalSteeringAggregate:
    seed_count: int
    heldout_noop_nll_mean: float
    heldout_conditional_nll_mean: float
    heldout_unconditional_nll_mean: float
    heldout_random_condition_nll_mean: float
    subgoal_revealed_ceiling_nll_mean: float
    gap_closed_nll_mean: float
    conditional_advantage_nll_mean: float
    condition_specificity_nll_mean: float
    gap_closed_fraction_mean: float
    gap_closed_nll_min: float
    conditional_advantage_nll_min: float


@dataclass(frozen=True)
class ConditionalSteeringAdmission:
    admitted: bool
    condition_gap_closed: bool
    condition_conditional_advantage: bool
    condition_specificity: bool
    condition_gap_closed_fraction: bool
    condition_structural: bool
    failed_conditions: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ConditionalSteeringReport:
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
    train_row_count: int
    heldout_row_count: int
    control_norm_cap: float
    updates_per_run: int
    learning_rate: float
    batch_size: int
    seed_schedule: tuple[int, ...]
    thresholds: ConditionalSteeringThresholds
    seed_points: tuple[ConditionalSteeringSeedPoint, ...]
    aggregate: ConditionalSteeringAggregate
    admission: ConditionalSteeringAdmission
    free_bias_present: bool
    zero_code_strict_noop: bool
    substrate_trainable_parameter_count: int
    production_wiring_changed: bool
    feedback_to_learning: bool
    description: str = ""


@dataclass(frozen=True)
class _JunctionExample:
    observation_text: str
    subgoal_revealed_text: str
    subgoal_index: int
    action_index: int
    residual: tuple[float, ...]


def _labelled_rows(
    rows: tuple[ConflictJunctionRow, ...],
) -> tuple[ConflictJunctionRow, ...]:
    return tuple(row for row in rows if row.active_subgoal is not None)


def _subgoal_vocabulary(
    corpus: ETAProofCorpus,
) -> tuple[str, ...]:
    return tuple(
        location.location_id
        for location in corpus.environment.objective_locations()
    )


def _capture_examples(
    rows: tuple[ConflictJunctionRow, ...],
    *,
    runtime: OpenWeightResidualRuntime,
    scorer: Any,
    subgoal_index: dict[str, int],
    injection_layer_index: int,
    residual_width: int,
    progress: Any | None,
    split_label: str,
) -> tuple[_JunctionExample, ...]:
    examples: list[_JunctionExample] = []
    total = len(rows)
    for index, row in enumerate(rows):
        if row.active_subgoal is None:
            raise ValueError("capture requires subgoal-labelled rows")
        scored_prefix = row.observation_text + ACTION_PROMPT_SUFFIX
        capture = runtime.capture(source_text=scored_prefix)
        activations = capture.residual_activations
        if (
            len(activations) != 1
            or activations[0].layer_index != injection_layer_index
            or len(activations[0].activation) != residual_width
        ):
            raise RuntimeError(
                "conditional steering requires one full-width residual at "
                f"layer {injection_layer_index}; row {row.case_id!r}."
            )
        examples.append(
            _JunctionExample(
                observation_text=scored_prefix,
                subgoal_revealed_text=(
                    row.subgoal_revealed_text + ACTION_PROMPT_SUFFIX
                ),
                subgoal_index=subgoal_index[row.active_subgoal],
                action_index=scorer.action_index(row.expert_action_id),
                residual=tuple(float(value) for value in activations[0].activation),
            )
        )
        if progress is not None and (
            index + 1 == total or (index + 1) % 32 == 0
        ):
            progress(f"C2 capture {split_label}: {index + 1}/{total}")
    if not examples:
        raise RuntimeError(f"conditional steering captured no {split_label} rows")
    return tuple(examples)


class _ConditionalOperator:
    """Learned rank-r multiplicative write, optionally subgoal-conditioned."""

    def __init__(
        self,
        *,
        torch: Any,
        width: int,
        rank: int,
        class_count: int,
        conditional: bool,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        scale = 1.0 / (width**0.5)
        self._torch = torch
        self._conditional = conditional
        self._U = (
            torch.randn(width, rank, generator=generator) * scale
        ).requires_grad_(True)
        self._V = (
            torch.randn(width, rank, generator=generator) * scale
        ).requires_grad_(True)
        code_rows = class_count if conditional else 1
        self._Z = (
            torch.randn(code_rows, rank, generator=generator) * scale
        ).requires_grad_(True)

    def parameters(self) -> tuple[Any, ...]:
        return (self._U, self._V, self._Z)

    def deltas(self, *, residuals: Any, subgoal_indices: Any) -> Any:
        torch = self._torch
        # Vᵀ h  ->  [B, rank]
        projected = residuals @ self._V
        if self._conditional:
            gate = torch.tanh(self._Z[subgoal_indices])
        else:
            gate = torch.tanh(self._Z[0]).unsqueeze(0)
        gated = gate * projected
        # U @ (gated)  ->  [B, width]
        return gated @ self._U.t()


def _zero_code_max_abs(
    *,
    torch: Any,
    operator: _ConditionalOperator,
    residuals: Any,
    subgoal_indices: Any,
) -> float:
    """Max |delta| when the learned code is forced to zero (must be 0.0)."""

    saved = operator._Z.detach().clone()
    with torch.no_grad():
        operator._Z.zero_()
        delta = operator.deltas(
            residuals=residuals, subgoal_indices=subgoal_indices
        )
        max_abs = float(delta.abs().max())
        operator._Z.copy_(saved)
    return max_abs


def _stack_residuals(torch: Any, examples: tuple[_JunctionExample, ...]) -> Any:
    return torch.tensor(
        [example.residual for example in examples], dtype=torch.float32
    )


def _train_operator(
    *,
    torch: Any,
    operator: _ConditionalOperator,
    residuals: Any,
    subgoal_indices: Any,
    action_indices: tuple[int, ...],
    texts: tuple[str, ...],
    scorer: Any,
    updates: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    progress: Any | None,
    label: str,
) -> float:
    optimizer = torch.optim.Adam(operator.parameters(), lr=learning_rate)
    rng = random.Random(seed * 1_000_003 + 7)
    row_count = residuals.shape[0]
    order = list(range(row_count))
    final_loss = 0.0
    for update_index in range(updates):
        rng.shuffle(order)
        batch = order[: min(batch_size, row_count)]
        batch_residuals = residuals[batch]
        batch_subgoals = subgoal_indices[batch]
        deltas = operator.deltas(
            residuals=batch_residuals, subgoal_indices=batch_subgoals
        )
        nll = scorer.action_nll(
            source_texts=tuple(texts[i] for i in batch),
            control_deltas=deltas,
            action_indices=tuple(action_indices[i] for i in batch),
        )
        loss = nll.mean()
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"C2 non-finite loss ({label}) seed={seed} update={update_index}"
            )
        optimizer.step()
        final_loss = float(loss.detach())
        if progress is not None and (
            update_index + 1 == updates or (update_index + 1) % 10 == 0
        ):
            progress(
                f"C2 train {label} seed={seed} "
                f"update={update_index + 1}/{updates} loss={final_loss:.4f}"
            )
    return final_loss


def _eval_arm_nll(
    *,
    torch: Any,
    deltas: Any,
    texts: tuple[str, ...],
    action_indices: tuple[int, ...],
    scorer: Any,
    batch_size: int,
) -> float:
    values: list[float] = []
    row_count = len(texts)
    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        values.extend(
            scorer.controlled_action_nll(
                source_texts=texts[start:stop],
                control_deltas=deltas[start:stop],
                action_indices=action_indices[start:stop],
            )
        )
    return statistics.fmean(values)


def _baseline_nll(
    *,
    texts: tuple[str, ...],
    action_indices: tuple[int, ...],
    scorer: Any,
    batch_size: int,
) -> float:
    values: list[float] = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        values.extend(
            scorer.baseline_action_nll(
                source_texts=texts[start:stop],
                action_indices=action_indices[start:stop],
            )
        )
    return statistics.fmean(values)


def _run_seed(
    *,
    torch: Any,
    seed: int,
    width: int,
    rank: int,
    class_count: int,
    scorer: Any,
    train_examples: tuple[_JunctionExample, ...],
    heldout_examples: tuple[_JunctionExample, ...],
    updates: int,
    learning_rate: float,
    batch_size: int,
    progress: Any | None,
) -> ConditionalSteeringSeedPoint:
    train_residuals = _stack_residuals(torch, train_examples)
    train_subgoals = torch.tensor(
        [example.subgoal_index for example in train_examples], dtype=torch.long
    )
    train_actions = tuple(example.action_index for example in train_examples)
    train_texts = tuple(example.observation_text for example in train_examples)

    heldout_residuals = _stack_residuals(torch, heldout_examples)
    heldout_subgoals = torch.tensor(
        [example.subgoal_index for example in heldout_examples],
        dtype=torch.long,
    )
    heldout_actions = tuple(
        example.action_index for example in heldout_examples
    )
    heldout_texts = tuple(
        example.observation_text for example in heldout_examples
    )
    heldout_revealed_texts = tuple(
        example.subgoal_revealed_text for example in heldout_examples
    )

    conditional = _ConditionalOperator(
        torch=torch,
        width=width,
        rank=rank,
        class_count=class_count,
        conditional=True,
        seed=seed,
    )
    unconditional = _ConditionalOperator(
        torch=torch,
        width=width,
        rank=rank,
        class_count=class_count,
        conditional=False,
        seed=seed + 1,
    )
    conditional_init = tuple(
        parameter.detach().clone() for parameter in conditional.parameters()
    )
    unconditional_init = tuple(
        parameter.detach().clone() for parameter in unconditional.parameters()
    )

    final_train_loss = _train_operator(
        torch=torch,
        operator=conditional,
        residuals=train_residuals,
        subgoal_indices=train_subgoals,
        action_indices=train_actions,
        texts=train_texts,
        scorer=scorer,
        updates=updates,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        progress=progress,
        label="conditional",
    )
    _train_operator(
        torch=torch,
        operator=unconditional,
        residuals=train_residuals,
        subgoal_indices=train_subgoals,
        action_indices=train_actions,
        texts=train_texts,
        scorer=scorer,
        updates=updates,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed + 1,
        progress=progress,
        label="unconditional",
    )

    with torch.no_grad():
        conditional_deltas = conditional.deltas(
            residuals=heldout_residuals, subgoal_indices=heldout_subgoals
        )
        unconditional_deltas = unconditional.deltas(
            residuals=heldout_residuals, subgoal_indices=heldout_subgoals
        )
        rng = random.Random(seed * 7919 + 13)
        random_subgoals = torch.tensor(
            [
                _derange_index(rng, int(value), class_count)
                for value in heldout_subgoals.tolist()
            ],
            dtype=torch.long,
        )
        random_deltas = conditional.deltas(
            residuals=heldout_residuals, subgoal_indices=random_subgoals
        )
        # A zeroed code must produce an exact no-op: tanh(0)=0 makes the
        # multiplicative write vanish for every residual.
        zero_code_strict_noop_max_abs = _zero_code_max_abs(
            torch=torch,
            operator=conditional,
            residuals=heldout_residuals,
            subgoal_indices=heldout_subgoals,
        )

    noop_nll = _baseline_nll(
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    revealed_nll = _baseline_nll(
        texts=heldout_revealed_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    conditional_nll = _eval_arm_nll(
        torch=torch,
        deltas=conditional_deltas,
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    unconditional_nll = _eval_arm_nll(
        torch=torch,
        deltas=unconditional_deltas,
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    random_nll = _eval_arm_nll(
        torch=torch,
        deltas=random_deltas,
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )

    gap = noop_nll - revealed_nll
    return ConditionalSteeringSeedPoint(
        seed=seed,
        heldout=ConditionalSteeringArmNLL(
            noop=noop_nll,
            conditional=conditional_nll,
            unconditional=unconditional_nll,
            random_condition=random_nll,
            subgoal_revealed_ceiling=revealed_nll,
        ),
        train_conditional_nll=final_train_loss,
        gap_closed_nll=noop_nll - conditional_nll,
        conditional_advantage_nll=unconditional_nll - conditional_nll,
        condition_specificity_nll=random_nll - conditional_nll,
        gap_closed_fraction=(
            (noop_nll - conditional_nll) / gap if gap > 1e-9 else 0.0
        ),
        conditional_low_rank_parameters_changed=_parameters_changed(
            torch, conditional_init, conditional.parameters()
        ),
        unconditional_low_rank_parameters_changed=_parameters_changed(
            torch, unconditional_init, unconditional.parameters()
        ),
        zero_code_strict_noop_max_abs=zero_code_strict_noop_max_abs,
        final_train_loss=final_train_loss,
    )


def _derange_index(rng: random.Random, value: int, class_count: int) -> int:
    if class_count < 2:
        return value
    choice = rng.randrange(class_count - 1)
    return choice if choice < value else choice + 1


def _parameters_changed(
    torch: Any, init: tuple[Any, ...], current: tuple[Any, ...]
) -> bool:
    return any(
        float((a - b.detach()).abs().max()) > 1e-8
        for a, b in zip(init, current, strict=True)
    )


def assess_conditional_steering(
    *,
    aggregate: ConditionalSteeringAggregate,
    thresholds: ConditionalSteeringThresholds,
    free_bias_present: bool,
    zero_code_strict_noop: bool,
    substrate_trainable_parameter_count: int,
    conditional_parameters_changed: bool,
) -> ConditionalSteeringAdmission:
    conditions = {
        "gap-closed": (
            aggregate.gap_closed_nll_mean >= thresholds.min_gap_closed_nll
        ),
        "conditional-advantage": (
            aggregate.conditional_advantage_nll_mean
            >= thresholds.min_conditional_advantage_nll
        ),
        "condition-specificity": (
            aggregate.condition_specificity_nll_mean
            >= thresholds.min_condition_specificity_nll
        ),
        "gap-closed-fraction": (
            aggregate.gap_closed_fraction_mean
            >= thresholds.min_gap_closed_fraction
        ),
        "structural-integrity": (
            not free_bias_present
            and zero_code_strict_noop
            and substrate_trainable_parameter_count == 0
            and conditional_parameters_changed
        ),
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    return ConditionalSteeringAdmission(
        admitted=not failed,
        condition_gap_closed=conditions["gap-closed"],
        condition_conditional_advantage=conditions["conditional-advantage"],
        condition_specificity=conditions["condition-specificity"],
        condition_gap_closed_fraction=conditions["gap-closed-fraction"],
        condition_structural=conditions["structural-integrity"],
        failed_conditions=failed,
        description=(
            "Conditional learned steering closes the goal-stripped gap and "
            "beats the equal-budget unconditional operator."
            if not failed
            else "Conditional steering screen blocked: " + ", ".join(failed)
        ),
    )


def _aggregate_seed_points(
    points: tuple[ConditionalSteeringSeedPoint, ...],
) -> ConditionalSteeringAggregate:
    if not points:
        raise ValueError("aggregate requires seed points")

    def mean(selector: Any) -> float:
        return statistics.fmean(selector(point) for point in points)

    return ConditionalSteeringAggregate(
        seed_count=len(points),
        heldout_noop_nll_mean=mean(lambda p: p.heldout.noop),
        heldout_conditional_nll_mean=mean(lambda p: p.heldout.conditional),
        heldout_unconditional_nll_mean=mean(lambda p: p.heldout.unconditional),
        heldout_random_condition_nll_mean=mean(
            lambda p: p.heldout.random_condition
        ),
        subgoal_revealed_ceiling_nll_mean=mean(
            lambda p: p.heldout.subgoal_revealed_ceiling
        ),
        gap_closed_nll_mean=mean(lambda p: p.gap_closed_nll),
        conditional_advantage_nll_mean=mean(
            lambda p: p.conditional_advantage_nll
        ),
        condition_specificity_nll_mean=mean(
            lambda p: p.condition_specificity_nll
        ),
        gap_closed_fraction_mean=mean(lambda p: p.gap_closed_fraction),
        gap_closed_nll_min=min(point.gap_closed_nll for point in points),
        conditional_advantage_nll_min=min(
            point.conditional_advantage_nll for point in points
        ),
    )


def run_eta_conditional_steering_screen(
    *,
    corpus: ETAProofCorpus,
    runtime: OpenWeightResidualRuntime,
    scorer: Any,
    model_source: str,
    device: str,
    injection_layer_index: int = 20,
    residual_width: int = 896,
    steering_rank: int = 8,
    screen_train_route_count: int = 48,
    seed_schedule: tuple[int, ...] = (0, 1, 2),
    updates_per_run: int = 80,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    thresholds: ConditionalSteeringThresholds | None = None,
    progress: Any | None = None,
) -> ConditionalSteeringReport:
    import torch

    if steering_rank < 1 or steering_rank > residual_width:
        raise ValueError("steering_rank must be in [1, residual_width]")
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("seed_schedule must be non-empty and unique")
    if updates_per_run < 1 or learning_rate <= 0.0 or batch_size < 1:
        raise ValueError("updates, learning rate and batch size must be positive")
    if scorer.trainable_parameters():
        raise RuntimeError("C2 requires a frozen substrate scorer")

    subgoal_vocabulary = _subgoal_vocabulary(corpus)
    subgoal_index = {name: index for index, name in enumerate(subgoal_vocabulary)}
    class_count = len(subgoal_vocabulary)

    train_rows = _labelled_rows(
        build_conflict_junction_rows(corpus, split="train")
    )[: screen_train_route_count * 8]
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

    seed_points = tuple(
        _run_seed(
            torch=torch,
            seed=seed,
            width=residual_width,
            rank=steering_rank,
            class_count=class_count,
            scorer=scorer,
            train_examples=train_examples,
            heldout_examples=heldout_examples,
            updates=updates_per_run,
            learning_rate=learning_rate,
            batch_size=batch_size,
            progress=progress,
        )
        for seed in seed_schedule
    )
    aggregate = _aggregate_seed_points(seed_points)
    zero_code_strict_noop = all(
        point.zero_code_strict_noop_max_abs == 0.0 for point in seed_points
    )
    conditional_changed = all(
        point.conditional_low_rank_parameters_changed for point in seed_points
    )
    active_thresholds = thresholds or ConditionalSteeringThresholds()
    admission = assess_conditional_steering(
        aggregate=aggregate,
        thresholds=active_thresholds,
        free_bias_present=False,
        zero_code_strict_noop=zero_code_strict_noop,
        substrate_trainable_parameter_count=0,
        conditional_parameters_changed=conditional_changed,
    )
    return ConditionalSteeringReport(
        schema_version=ETA_CONDITIONAL_STEERING_SCHEMA_VERSION,
        claim_scope="conditional-learned-steering-directional-screen",
        observation_protocol="goal-ambiguous-junction.v5",
        model_id=runtime.model_id,
        model_source=model_source,
        device=device,
        corpus_seed=corpus.seed,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        steering_rank=steering_rank,
        subgoal_class_count=class_count,
        train_row_count=len(train_examples),
        heldout_row_count=len(heldout_examples),
        control_norm_cap=float(scorer.control_norm_cap),
        updates_per_run=updates_per_run,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed_schedule=tuple(seed_schedule),
        thresholds=active_thresholds,
        seed_points=seed_points,
        aggregate=aggregate,
        admission=admission,
        free_bias_present=False,
        zero_code_strict_noop=zero_code_strict_noop,
        substrate_trainable_parameter_count=0,
        production_wiring_changed=False,
        feedback_to_learning=False,
        description=(
            "Matched-budget conditional vs unconditional learned steering on "
            f"the goal-ambiguous junction instrument; admitted={admission.admitted}."
        ),
    )


__all__ = [
    "ACTION_PROMPT_SUFFIX",
    "ETA_CONDITIONAL_STEERING_SCHEMA_VERSION",
    "ConditionalSteeringAdmission",
    "ConditionalSteeringAggregate",
    "ConditionalSteeringArmNLL",
    "ConditionalSteeringReport",
    "ConditionalSteeringSeedPoint",
    "ConditionalSteeringThresholds",
    "assess_conditional_steering",
    "run_eta_conditional_steering_screen",
]
