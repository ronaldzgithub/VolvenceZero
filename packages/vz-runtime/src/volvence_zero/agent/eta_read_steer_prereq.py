"""P2c · S3 prerequisite — close the read->steer loop with a non-oracle sensor.

C2 proved the executor: given the true subgoal, a learned rank-8 conditional
write closes the goal-stripped action-NLL gap and beats an equal-budget
unconditional operator. C2's condition was the *oracle* subgoal, so it does not
by itself certify a deployable loop. This packet asks the S3 prerequisite:

    Can a frozen (non-oracle) linear sensor supply the condition online, so the
    read->steer loop closes without ever being told the answer?

Two audited facts scope the design:

* The existing S1 v2 probe does NOT transfer to this instrument's surface
  (top-1 ~= chance on both revealed and stripped prompts): a probe fit on the
  V4 staged-plan surface cannot be reused verbatim as the C2-surface reader.
* On the goal-STRIPPED junction the subgoal is unreadable by anyone (top-1 ~=
  chance): by construction the goal is absent from that prompt. The condition
  must therefore come from CONTEXT that carries the goal -- in the companion
  product this is the relational context; in the maze surrogate it is the
  goal-revealed prefix (the agent knowing its own subgoal from memory).

So the loop is: a fresh frozen linear reader, refit on the C2 surface, reads the
subgoal from the context-carrying residual, and the C2 executor steers the
ambiguous-junction action from that read (not oracle) condition. The reader is
a linear sensor (no LM training, frozen at steer time); the executor is the
frozen-substrate rank-8 write. Route-level bootstrap CIs gate promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import statistics
from typing import Any

import numpy as np

from volvence_zero.agent.eta_conditional_steering_screen import (
    ACTION_PROMPT_SUFFIX,
    _ConditionalOperator,
    _derange_index,
    _subgoal_vocabulary,
    _train_operator,
    _zero_code_max_abs,
)
from volvence_zero.agent.eta_conflict_instrument import (
    ConflictJunctionRow,
    build_conflict_junction_rows,
)
from volvence_zero.agent.eta_proof_benchmark import ETAProofCorpus
from volvence_zero.substrate import OpenWeightResidualRuntime


ETA_READ_STEER_PREREQ_SCHEMA_VERSION = "eta-read-steer-prereq.v1"


@dataclass(frozen=True)
class ReadSteerThresholds:
    min_reader_heldout_accuracy: float = 0.80
    min_online_gap_closed_nll: float = 0.30
    min_online_conditional_advantage_nll: float = 0.15
    require_bootstrap_lower_positive: bool = True


@dataclass(frozen=True)
class ReadSteerEffect:
    mean: float
    ci_lower: float
    ci_upper: float
    route_count: int


@dataclass(frozen=True)
class ReadSteerArmNLL:
    noop: float
    conditional_oracle: float
    conditional_online: float
    unconditional: float
    random_condition: float
    subgoal_revealed_ceiling: float


@dataclass(frozen=True)
class ReadSteerSeedPoint:
    seed: int
    reader_train_accuracy: float
    reader_heldout_accuracy: float
    heldout: ReadSteerArmNLL
    online_gap_closed_nll: float
    online_conditional_advantage_nll: float
    online_vs_noop: ReadSteerEffect
    online_vs_unconditional: ReadSteerEffect
    zero_code_strict_noop_max_abs: float
    conditional_parameters_changed: bool


@dataclass(frozen=True)
class ReadSteerAggregate:
    seed_count: int
    reader_heldout_accuracy_mean: float
    noop_nll_mean: float
    conditional_oracle_nll_mean: float
    conditional_online_nll_mean: float
    unconditional_nll_mean: float
    random_condition_nll_mean: float
    subgoal_revealed_ceiling_nll_mean: float
    online_gap_closed_nll_mean: float
    online_conditional_advantage_nll_mean: float
    online_vs_noop_ci_lower_min: float
    online_vs_unconditional_ci_lower_min: float


@dataclass(frozen=True)
class ReadSteerAdmission:
    admitted: bool
    condition_reader_accuracy: bool
    condition_online_gap_closed: bool
    condition_online_conditional_advantage: bool
    condition_bootstrap: bool
    condition_structural: bool
    failed_conditions: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ReadSteerReport:
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
    reader_ridge_lambda: float
    train_row_count: int
    heldout_row_count: int
    control_norm_cap: float
    updates_per_run: int
    learning_rate: float
    batch_size: int
    seed_schedule: tuple[int, ...]
    bootstrap_resamples: int
    bootstrap_confidence: float
    thresholds: ReadSteerThresholds
    seed_points: tuple[ReadSteerSeedPoint, ...]
    aggregate: ReadSteerAggregate
    admission: ReadSteerAdmission
    free_bias_present: bool
    zero_code_strict_noop: bool
    substrate_trainable_parameter_count: int
    production_wiring_changed: bool
    feedback_to_learning: bool
    description: str = ""


@dataclass(frozen=True)
class _ReadSteerExample:
    case_id: str
    observation_text: str
    subgoal_revealed_text: str
    subgoal_index: int
    action_index: int
    action_residual: tuple[float, ...]
    context_residual: tuple[float, ...]


def _labelled_rows(
    rows: tuple[ConflictJunctionRow, ...],
) -> tuple[ConflictJunctionRow, ...]:
    return tuple(row for row in rows if row.active_subgoal is not None)


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
) -> tuple[_ReadSteerExample, ...]:
    examples: list[_ReadSteerExample] = []
    total = len(rows)
    for index, row in enumerate(rows):
        if row.active_subgoal is None:
            raise ValueError("read-steer capture requires labelled rows")
        action_text = row.observation_text + ACTION_PROMPT_SUFFIX
        context_text = row.subgoal_revealed_text + ACTION_PROMPT_SUFFIX
        action_residual = _capture_one(
            runtime, action_text, injection_layer_index, residual_width
        )
        context_residual = _capture_one(
            runtime, context_text, injection_layer_index, residual_width
        )
        examples.append(
            _ReadSteerExample(
                case_id=row.case_id,
                observation_text=action_text,
                subgoal_revealed_text=context_text,
                subgoal_index=subgoal_index[row.active_subgoal],
                action_index=scorer.action_index(row.expert_action_id),
                action_residual=action_residual,
                context_residual=context_residual,
            )
        )
        if progress is not None and (
            index + 1 == total or (index + 1) % 32 == 0
        ):
            progress(f"S3-prereq capture {split_label}: {index + 1}/{total}")
    if not examples:
        raise RuntimeError(f"read-steer captured no {split_label} rows")
    return tuple(examples)


def _capture_one(
    runtime: OpenWeightResidualRuntime,
    text: str,
    injection_layer_index: int,
    residual_width: int,
) -> tuple[float, ...]:
    capture = runtime.capture(source_text=text)
    activations = capture.residual_activations
    if (
        len(activations) != 1
        or activations[0].layer_index != injection_layer_index
        or len(activations[0].activation) != residual_width
    ):
        raise RuntimeError(
            "read-steer requires one full-width residual at layer "
            f"{injection_layer_index}"
        )
    return tuple(float(value) for value in activations[0].activation)


@dataclass(frozen=True)
class _FrozenLinearReader:
    weights: Any
    feature_mean: Any
    feature_scale: Any

    def predict(self, residuals: Any) -> Any:
        standardized = (residuals - self.feature_mean) / self.feature_scale
        return (standardized @ self.weights).argmax(axis=1)


def fit_condition_reader(
    examples: tuple[_ReadSteerExample, ...],
    *,
    class_count: int,
    ridge_lambda: float,
) -> _FrozenLinearReader:
    """Closed-form ridge multiclass reader on context-carrying residuals.

    A frozen linear sensor (no LM training). It maps the context residual --
    which carries the subgoal -- to the subgoal index, so the executor can be
    driven by a *read* (non-oracle) condition.
    """

    features = np.asarray(
        [example.context_residual for example in examples], dtype=np.float64
    )
    labels = np.asarray(
        [example.subgoal_index for example in examples], dtype=np.int64
    )
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0) + 1e-6
    standardized = (features - feature_mean) / feature_scale
    one_hot = np.eye(class_count)[labels]
    gram = standardized.T @ standardized + ridge_lambda * np.eye(
        standardized.shape[1]
    )
    weights = np.linalg.solve(gram, standardized.T @ one_hot)
    return _FrozenLinearReader(
        weights=weights,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def _reader_accuracy(
    reader: _FrozenLinearReader, examples: tuple[_ReadSteerExample, ...]
) -> float:
    residuals = np.asarray(
        [example.context_residual for example in examples], dtype=np.float64
    )
    truths = np.asarray(
        [example.subgoal_index for example in examples], dtype=np.int64
    )
    predictions = reader.predict(residuals)
    return float((predictions == truths).mean())


def _per_row_controlled_nll(
    *,
    deltas: Any,
    texts: tuple[str, ...],
    action_indices: tuple[int, ...],
    scorer: Any,
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        values.extend(
            scorer.controlled_action_nll(
                source_texts=texts[start:stop],
                control_deltas=deltas[start:stop],
                action_indices=action_indices[start:stop],
            )
        )
    return values


def _per_row_baseline_nll(
    *,
    texts: tuple[str, ...],
    action_indices: tuple[int, ...],
    scorer: Any,
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        values.extend(
            scorer.baseline_action_nll(
                source_texts=texts[start:stop],
                action_indices=action_indices[start:stop],
            )
        )
    return values


def _route_bootstrap(
    *,
    case_ids: tuple[str, ...],
    per_row_effect: list[float],
    seed: int,
    resamples: int,
    confidence: float,
) -> ReadSteerEffect:
    grouped: dict[str, list[float]] = {}
    for case_id, value in zip(case_ids, per_row_effect, strict=True):
        grouped.setdefault(case_id, []).append(value)
    route_effects = [
        statistics.fmean(values) for _, values in sorted(grouped.items())
    ]
    if not route_effects:
        raise ValueError("route bootstrap requires effects")
    rng = random.Random(seed)
    count = len(route_effects)
    draws = sorted(
        statistics.fmean(route_effects[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = draws[min(max(int(tail * resamples), 0), resamples - 1)]
    upper = draws[
        min(max(int((1.0 - tail) * resamples) - 1, 0), resamples - 1)
    ]
    return ReadSteerEffect(
        mean=statistics.fmean(route_effects),
        ci_lower=lower,
        ci_upper=upper,
        route_count=count,
    )


def _run_seed(
    *,
    torch: Any,
    seed: int,
    width: int,
    rank: int,
    class_count: int,
    scorer: Any,
    train_examples: tuple[_ReadSteerExample, ...],
    heldout_examples: tuple[_ReadSteerExample, ...],
    reader: _FrozenLinearReader,
    updates: int,
    learning_rate: float,
    batch_size: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    progress: Any | None,
) -> ReadSteerSeedPoint:
    train_residuals = _stack_residuals_action(torch, train_examples)
    train_subgoals = torch.tensor(
        [example.subgoal_index for example in train_examples], dtype=torch.long
    )
    train_actions = tuple(example.action_index for example in train_examples)
    train_texts = tuple(example.observation_text for example in train_examples)

    heldout_residuals = _stack_residuals_action(torch, heldout_examples)
    heldout_subgoals = torch.tensor(
        [example.subgoal_index for example in heldout_examples],
        dtype=torch.long,
    )
    heldout_actions = tuple(example.action_index for example in heldout_examples)
    heldout_texts = tuple(example.observation_text for example in heldout_examples)
    heldout_revealed = tuple(
        example.subgoal_revealed_text for example in heldout_examples
    )
    heldout_cases = tuple(example.case_id for example in heldout_examples)

    online_indices = torch.tensor(
        reader.predict(
            np.asarray(
                [example.context_residual for example in heldout_examples],
                dtype=np.float64,
            )
        ),
        dtype=torch.long,
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
    _train_operator(
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
        oracle_deltas = conditional.deltas(
            residuals=heldout_residuals, subgoal_indices=heldout_subgoals
        )
        online_deltas = conditional.deltas(
            residuals=heldout_residuals, subgoal_indices=online_indices
        )
        unconditional_deltas = unconditional.deltas(
            residuals=heldout_residuals, subgoal_indices=heldout_subgoals
        )
        rng = random.Random(seed * 7919 + 13)
        random_indices = torch.tensor(
            [
                _derange_index(rng, int(value), class_count)
                for value in heldout_subgoals.tolist()
            ],
            dtype=torch.long,
        )
        random_deltas = conditional.deltas(
            residuals=heldout_residuals, subgoal_indices=random_indices
        )
        zero_code_max_abs = _zero_code_max_abs(
            torch=torch,
            operator=conditional,
            residuals=heldout_residuals,
            subgoal_indices=heldout_subgoals,
        )

    noop_rows = _per_row_baseline_nll(
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    revealed_rows = _per_row_baseline_nll(
        texts=heldout_revealed,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    oracle_rows = _per_row_controlled_nll(
        deltas=oracle_deltas,
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    online_rows = _per_row_controlled_nll(
        deltas=online_deltas,
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    unconditional_rows = _per_row_controlled_nll(
        deltas=unconditional_deltas,
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )
    random_rows = _per_row_controlled_nll(
        deltas=random_deltas,
        texts=heldout_texts,
        action_indices=heldout_actions,
        scorer=scorer,
        batch_size=batch_size,
    )

    noop_mean = statistics.fmean(noop_rows)
    online_mean = statistics.fmean(online_rows)
    unconditional_mean = statistics.fmean(unconditional_rows)
    online_vs_noop = _route_bootstrap(
        case_ids=heldout_cases,
        per_row_effect=[n - o for n, o in zip(noop_rows, online_rows, strict=True)],
        seed=seed * 101 + 3,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
    )
    online_vs_unconditional = _route_bootstrap(
        case_ids=heldout_cases,
        per_row_effect=[
            u - o
            for u, o in zip(unconditional_rows, online_rows, strict=True)
        ],
        seed=seed * 101 + 5,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
    )
    return ReadSteerSeedPoint(
        seed=seed,
        reader_train_accuracy=_reader_accuracy(reader, train_examples),
        reader_heldout_accuracy=_reader_accuracy(reader, heldout_examples),
        heldout=ReadSteerArmNLL(
            noop=noop_mean,
            conditional_oracle=statistics.fmean(oracle_rows),
            conditional_online=online_mean,
            unconditional=unconditional_mean,
            random_condition=statistics.fmean(random_rows),
            subgoal_revealed_ceiling=statistics.fmean(revealed_rows),
        ),
        online_gap_closed_nll=noop_mean - online_mean,
        online_conditional_advantage_nll=unconditional_mean - online_mean,
        online_vs_noop=online_vs_noop,
        online_vs_unconditional=online_vs_unconditional,
        zero_code_strict_noop_max_abs=zero_code_max_abs,
        conditional_parameters_changed=any(
            float((a - b.detach()).abs().max()) > 1e-8
            for a, b in zip(
                conditional_init, conditional.parameters(), strict=True
            )
        ),
    )


def _stack_residuals_action(
    torch: Any, examples: tuple[_ReadSteerExample, ...]
) -> Any:
    return torch.tensor(
        [example.action_residual for example in examples], dtype=torch.float32
    )


def assess_read_steer(
    *,
    aggregate: ReadSteerAggregate,
    thresholds: ReadSteerThresholds,
    free_bias_present: bool,
    zero_code_strict_noop: bool,
    substrate_trainable_parameter_count: int,
    conditional_parameters_changed: bool,
) -> ReadSteerAdmission:
    conditions = {
        "reader-accuracy": (
            aggregate.reader_heldout_accuracy_mean
            >= thresholds.min_reader_heldout_accuracy
        ),
        "online-gap-closed": (
            aggregate.online_gap_closed_nll_mean
            >= thresholds.min_online_gap_closed_nll
        ),
        "online-conditional-advantage": (
            aggregate.online_conditional_advantage_nll_mean
            >= thresholds.min_online_conditional_advantage_nll
        ),
        "bootstrap-lower-positive": (
            not thresholds.require_bootstrap_lower_positive
            or (
                aggregate.online_vs_noop_ci_lower_min > 0.0
                and aggregate.online_vs_unconditional_ci_lower_min > 0.0
            )
        ),
        "structural-integrity": (
            not free_bias_present
            and zero_code_strict_noop
            and substrate_trainable_parameter_count == 0
            and conditional_parameters_changed
        ),
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    return ReadSteerAdmission(
        admitted=not failed,
        condition_reader_accuracy=conditions["reader-accuracy"],
        condition_online_gap_closed=conditions["online-gap-closed"],
        condition_online_conditional_advantage=conditions[
            "online-conditional-advantage"
        ],
        condition_bootstrap=conditions["bootstrap-lower-positive"],
        condition_structural=conditions["structural-integrity"],
        failed_conditions=failed,
        description=(
            "Read->steer loop closes: a frozen linear sensor supplies the "
            "condition online and conditional steering still beats noop and "
            "the equal-budget unconditional operator."
            if not failed
            else "Read->steer prerequisite blocked: " + ", ".join(failed)
        ),
    )


def _aggregate(
    points: tuple[ReadSteerSeedPoint, ...],
) -> ReadSteerAggregate:
    def mean(selector: Any) -> float:
        return statistics.fmean(selector(point) for point in points)

    return ReadSteerAggregate(
        seed_count=len(points),
        reader_heldout_accuracy_mean=mean(lambda p: p.reader_heldout_accuracy),
        noop_nll_mean=mean(lambda p: p.heldout.noop),
        conditional_oracle_nll_mean=mean(lambda p: p.heldout.conditional_oracle),
        conditional_online_nll_mean=mean(lambda p: p.heldout.conditional_online),
        unconditional_nll_mean=mean(lambda p: p.heldout.unconditional),
        random_condition_nll_mean=mean(lambda p: p.heldout.random_condition),
        subgoal_revealed_ceiling_nll_mean=mean(
            lambda p: p.heldout.subgoal_revealed_ceiling
        ),
        online_gap_closed_nll_mean=mean(lambda p: p.online_gap_closed_nll),
        online_conditional_advantage_nll_mean=mean(
            lambda p: p.online_conditional_advantage_nll
        ),
        online_vs_noop_ci_lower_min=min(
            point.online_vs_noop.ci_lower for point in points
        ),
        online_vs_unconditional_ci_lower_min=min(
            point.online_vs_unconditional.ci_lower for point in points
        ),
    )


def run_eta_read_steer_prereq(
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
    seed_schedule: tuple[int, ...] = (0, 1, 2, 3, 4),
    updates_per_run: int = 80,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    reader_ridge_lambda: float = 10.0,
    bootstrap_resamples: int = 5000,
    bootstrap_confidence: float = 0.95,
    thresholds: ReadSteerThresholds | None = None,
    progress: Any | None = None,
) -> ReadSteerReport:
    import torch

    if steering_rank < 1 or steering_rank > residual_width:
        raise ValueError("steering_rank must be in [1, residual_width]")
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("seed_schedule must be non-empty and unique")
    if updates_per_run < 1 or learning_rate <= 0.0 or batch_size < 1:
        raise ValueError("updates, learning rate and batch size must be positive")
    if scorer.trainable_parameters():
        raise RuntimeError("S3-prereq requires a frozen substrate scorer")

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
    reader = fit_condition_reader(
        train_examples,
        class_count=class_count,
        ridge_lambda=reader_ridge_lambda,
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
            reader=reader,
            updates=updates_per_run,
            learning_rate=learning_rate,
            batch_size=batch_size,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
            progress=progress,
        )
        for seed in seed_schedule
    )
    aggregate = _aggregate(seed_points)
    zero_code_strict_noop = all(
        point.zero_code_strict_noop_max_abs == 0.0 for point in seed_points
    )
    conditional_changed = all(
        point.conditional_parameters_changed for point in seed_points
    )
    active_thresholds = thresholds or ReadSteerThresholds()
    admission = assess_read_steer(
        aggregate=aggregate,
        thresholds=active_thresholds,
        free_bias_present=False,
        zero_code_strict_noop=zero_code_strict_noop,
        substrate_trainable_parameter_count=0,
        conditional_parameters_changed=conditional_changed,
    )
    return ReadSteerReport(
        schema_version=ETA_READ_STEER_PREREQ_SCHEMA_VERSION,
        claim_scope="conditional-learned-steering-read-loop-s3-prereq",
        observation_protocol="goal-ambiguous-junction.v5",
        model_id=runtime.model_id,
        model_source=model_source,
        device=device,
        corpus_seed=corpus.seed,
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        steering_rank=steering_rank,
        subgoal_class_count=class_count,
        reader_ridge_lambda=reader_ridge_lambda,
        train_row_count=len(train_examples),
        heldout_row_count=len(heldout_examples),
        control_norm_cap=float(scorer.control_norm_cap),
        updates_per_run=updates_per_run,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed_schedule=tuple(seed_schedule),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
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
            "Read->steer loop with a refit frozen linear condition reader; "
            f"admitted={admission.admitted}."
        ),
    )


__all__ = [
    "ETA_READ_STEER_PREREQ_SCHEMA_VERSION",
    "ReadSteerAdmission",
    "ReadSteerAggregate",
    "ReadSteerArmNLL",
    "ReadSteerEffect",
    "ReadSteerReport",
    "ReadSteerSeedPoint",
    "ReadSteerThresholds",
    "assess_read_steer",
    "fit_condition_reader",
    "run_eta_read_steer_prereq",
]
