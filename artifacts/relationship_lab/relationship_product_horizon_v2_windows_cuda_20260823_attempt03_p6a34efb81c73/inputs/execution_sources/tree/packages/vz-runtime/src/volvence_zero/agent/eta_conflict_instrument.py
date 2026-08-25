"""Goal-ambiguous junction instrument (P2c · C1): read-only instrument validity.

P2a proved the V4 staged-plan rate-distortion instrument has no headroom for a
*conditional* intervention: a constant low-rank operator already saturates it
(`permuted_z_penalty=0`, static steering gain `+2e-4`). The reason is that the
V4 observation prefix announces the goal, so the expert action is a function of
the prefix and a constant map suffices.

This module redesigns the observation surface to **strip the goal**. The same
local view (current location + available transitions) then recurs under
different active subgoals demanding different expert actions -- a *conflicting
mapping* that no constant operator can saturate. The active subgoal becomes the
single missing bit that disambiguates the action, which is exactly what a
subgoal-conditional steering policy would have to supply.

Everything here is read-only: it replays the frozen corpus, classifies conflict
junctions structurally (no model), and -- when a runtime is supplied -- measures
the base model's action uncertainty on goal-stripped vs subgoal-revealed
prompts. It fits no controller, adds no bias, trains no parameter, and changes
no production wiring. It only decides whether the redesigned instrument is valid
enough to justify the conditional-learned-steering screen (C2).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import statistics
from typing import Any

from volvence_zero.agent.eta_proof_benchmark import ETAProofCase, ETAProofCorpus
from volvence_zero.internal_rl import HierarchicalRouteSpec


ETA_CONFLICT_INSTRUMENT_SCHEMA_VERSION = "eta-conflict-junction-instrument.v1"
GOAL_AMBIGUOUS_JUNCTION_PROTOCOL = "goal-ambiguous-junction.v5"


def _goal_stripped_text(
    current_location_id: str, available_targets: tuple[str, ...]
) -> str:
    """Render the goal-ambiguous view: local node + out-edges only.

    No task context, no route plan, no next-objective announcement, no
    completed-objective ledger -- nothing that reveals which subgoal is active.
    """

    return (
        f"Current location: {current_location_id}. "
        f"Available transitions: {', '.join(available_targets)}."
    )


def _subgoal_revealed_text(
    active_subgoal: str,
    current_location_id: str,
    available_targets: tuple[str, ...],
) -> str:
    """The goal-stripped view plus the single disambiguating bit.

    Used only as a paired control: the base model's expert-action NLL gap
    between this and the goal-stripped view is the causal value of knowing the
    subgoal -- i.e. the headroom a steering policy would have to recover.
    """

    return (
        f"Objective: {active_subgoal}. "
        f"Current location: {current_location_id}. "
        f"Available transitions: {', '.join(available_targets)}."
    )


@dataclass(frozen=True)
class ConflictJunctionRow:
    case_id: str
    split: str
    step_index: int
    current_location_id: str
    available_targets: tuple[str, ...]
    observation_text: str
    subgoal_revealed_text: str
    active_subgoal: str | None
    expert_action_id: str
    local_view_id: str


@dataclass(frozen=True)
class ConflictHeadroomMetrics:
    split: str
    row_count: int
    unique_local_views: int
    conflict_view_count: int
    conflict_row_count: int
    conflict_row_fraction: float
    constant_operator_error_rate: float
    oracle_conditional_error_rate: float
    view_subgoal_residual_ambiguity: int
    subgoal_labelled_row_count: int
    mean_available_targets: float


@dataclass(frozen=True)
class ConflictBaseUncertaintyMetrics:
    split: str
    scored_row_count: int
    goal_stripped_mean_expert_nll: float
    goal_stripped_median_expert_nll: float
    subgoal_revealed_mean_expert_nll: float
    steerable_headroom_nll: float
    fraction_base_uncertain: float
    uncertain_nll_threshold: float


@dataclass(frozen=True)
class ConflictInstrumentThresholds:
    min_conflict_row_fraction: float = 0.50
    min_constant_operator_error: float = 0.20
    max_view_subgoal_residual_ambiguity: int = 0
    min_steerable_headroom_nll: float = 0.05
    min_fraction_base_uncertain: float = 0.30


@dataclass(frozen=True)
class ConflictInstrumentAdmission:
    valid: bool
    condition_conflict_fraction: bool
    condition_constant_operator_error: bool
    condition_view_subgoal_determinism: bool
    condition_base_headroom: bool
    condition_base_uncertain_fraction: bool
    base_uncertainty_evaluated: bool
    failed_conditions: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ConflictInstrumentReport:
    schema_version: str
    claim_scope: str
    observation_protocol: str
    corpus_seed: int
    objective_count: int
    model_id: str
    model_source: str
    device: str
    thresholds: ConflictInstrumentThresholds
    headroom: ConflictHeadroomMetrics
    base_uncertainty: ConflictBaseUncertaintyMetrics | None
    admission: ConflictInstrumentAdmission
    trainable_parameter_count: int
    free_bias_present: bool
    production_wiring_changed: bool
    feedback_to_learning: bool
    description: str = ""


def _route_from_case(case: ETAProofCase) -> HierarchicalRouteSpec:
    return HierarchicalRouteSpec(
        case_id=case.case_id,
        split=case.split,
        source_text=case.source_text,
        waypoints=case.route_signature,
        split_detail=case.split_detail,
        description=case.description,
    )


def build_conflict_junction_rows(
    corpus: ETAProofCorpus,
    *,
    split: str,
) -> tuple[ConflictJunctionRow, ...]:
    """Replay one split into goal-stripped junction rows.

    Each row is one expert controller-action step rendered with the goal-ambiguous
    view. The active subgoal (next objective ahead) is recorded as evaluation
    ground truth only -- it never appears in ``observation_text``.
    """

    if split == "train":
        cases = corpus.train_cases
    elif split == "heldout":
        cases = corpus.heldout_cases
    else:
        raise ValueError(f"split must be 'train' or 'heldout', got {split!r}")

    environment = corpus.environment
    objective_ids = {
        location.location_id for location in environment.objective_locations()
    }
    rows: list[ConflictJunctionRow] = []
    for case in cases:
        route = _route_from_case(case)
        waypoints = route.waypoints
        state = environment.reset(route)
        for step_index, target_id in enumerate(waypoints[1:]):
            observation = environment.observe(state)
            available = tuple(observation.available_targets)
            next_objective = next(
                (
                    node
                    for node in waypoints[step_index + 1 :]
                    if node in objective_ids
                ),
                None,
            )
            local_view_id = (
                f"{observation.current_location_id}|"
                + ">".join(sorted(available))
            )
            observation_text = _goal_stripped_text(
                observation.current_location_id, available
            )
            revealed_text = (
                _subgoal_revealed_text(
                    next_objective, observation.current_location_id, available
                )
                if next_objective is not None
                else observation_text
            )
            rows.append(
                ConflictJunctionRow(
                    case_id=case.case_id,
                    split=split,
                    step_index=step_index,
                    current_location_id=observation.current_location_id,
                    available_targets=available,
                    observation_text=observation_text,
                    subgoal_revealed_text=revealed_text,
                    active_subgoal=next_objective,
                    expert_action_id=f"move:{target_id}",
                    local_view_id=local_view_id,
                )
            )
            state = environment.step(state, target_id=target_id).next_state
    if not rows:
        raise RuntimeError(
            f"conflict-junction replay produced no rows for split {split!r}"
        )
    return tuple(rows)


def compute_conflict_headroom(
    rows: tuple[ConflictJunctionRow, ...],
    *,
    split: str,
) -> ConflictHeadroomMetrics:
    """Structural headroom: can a constant map saturate the goal-stripped view?

    A *conflict view* is a local view that recurs with >=2 distinct expert
    actions. The constant-operator error rate is the misclassification of the
    best single action per view (over conflict rows). The oracle-conditional
    error rate does the same after also conditioning on the active subgoal; a
    valid instrument drives it toward zero, proving the subgoal is the missing
    bit a conditional policy must supply.
    """

    if not rows:
        raise ValueError("conflict headroom requires rows")
    view_actions: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        view_actions[row.local_view_id].append(row.expert_action_id)
    conflict_views = {
        view for view, actions in view_actions.items() if len(set(actions)) >= 2
    }
    conflict_rows = tuple(
        row for row in rows if row.local_view_id in conflict_views
    )

    constant_errors = 0
    for view in conflict_views:
        actions = view_actions[view]
        majority = Counter(actions).most_common(1)[0][1]
        constant_errors += len(actions) - majority

    view_subgoal_actions: dict[tuple[str, str | None], list[str]] = defaultdict(
        list
    )
    for row in conflict_rows:
        view_subgoal_actions[(row.local_view_id, row.active_subgoal)].append(
            row.expert_action_id
        )
    oracle_errors = 0
    for actions in view_subgoal_actions.values():
        majority = Counter(actions).most_common(1)[0][1]
        oracle_errors += len(actions) - majority
    residual_ambiguity = sum(
        1 for actions in view_subgoal_actions.values() if len(set(actions)) >= 2
    )

    conflict_row_count = len(conflict_rows)
    return ConflictHeadroomMetrics(
        split=split,
        row_count=len(rows),
        unique_local_views=len(view_actions),
        conflict_view_count=len(conflict_views),
        conflict_row_count=conflict_row_count,
        conflict_row_fraction=conflict_row_count / len(rows),
        constant_operator_error_rate=(
            constant_errors / conflict_row_count if conflict_row_count else 0.0
        ),
        oracle_conditional_error_rate=(
            oracle_errors / conflict_row_count if conflict_row_count else 0.0
        ),
        view_subgoal_residual_ambiguity=residual_ambiguity,
        subgoal_labelled_row_count=sum(
            1 for row in rows if row.active_subgoal is not None
        ),
        mean_available_targets=statistics.fmean(
            len(row.available_targets) for row in rows
        ),
    )


def measure_base_uncertainty(
    rows: tuple[ConflictJunctionRow, ...],
    *,
    scorer: Any,
    split: str,
    uncertain_nll_threshold: float = 0.10,
    batch_size: int = 32,
) -> ConflictBaseUncertaintyMetrics:
    """Base-model expert-action NLL on goal-stripped vs subgoal-revealed views.

    Only rows with a defined active subgoal are scored (the revealed control is
    ill-defined otherwise). The mean NLL gap is the causal value of the subgoal
    bit; a large gap with a high goal-stripped NLL is the steerable headroom the
    conditional screen would exploit.
    """

    labelled = tuple(row for row in rows if row.active_subgoal is not None)
    if not labelled:
        raise ValueError("base uncertainty requires subgoal-labelled rows")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    stripped_nll: list[float] = []
    revealed_nll: list[float] = []
    for start in range(0, len(labelled), batch_size):
        batch = labelled[start : start + batch_size]
        action_indices = tuple(
            scorer.action_index(row.expert_action_id) for row in batch
        )
        stripped_nll.extend(
            scorer.baseline_action_nll(
                source_texts=tuple(row.observation_text for row in batch),
                action_indices=action_indices,
            )
        )
        revealed_nll.extend(
            scorer.baseline_action_nll(
                source_texts=tuple(row.subgoal_revealed_text for row in batch),
                action_indices=action_indices,
            )
        )
    stripped_mean = statistics.fmean(stripped_nll)
    revealed_mean = statistics.fmean(revealed_nll)
    return ConflictBaseUncertaintyMetrics(
        split=split,
        scored_row_count=len(labelled),
        goal_stripped_mean_expert_nll=stripped_mean,
        goal_stripped_median_expert_nll=statistics.median(stripped_nll),
        subgoal_revealed_mean_expert_nll=revealed_mean,
        steerable_headroom_nll=stripped_mean - revealed_mean,
        fraction_base_uncertain=(
            sum(value > uncertain_nll_threshold for value in stripped_nll)
            / len(stripped_nll)
        ),
        uncertain_nll_threshold=uncertain_nll_threshold,
    )


def assess_conflict_instrument(
    *,
    headroom: ConflictHeadroomMetrics,
    thresholds: ConflictInstrumentThresholds,
    base_uncertainty: ConflictBaseUncertaintyMetrics | None,
) -> ConflictInstrumentAdmission:
    conditions = {
        "conflict-row-fraction": (
            headroom.conflict_row_fraction >= thresholds.min_conflict_row_fraction
        ),
        "constant-operator-error": (
            headroom.constant_operator_error_rate
            >= thresholds.min_constant_operator_error
        ),
        "view-subgoal-determinism": (
            headroom.view_subgoal_residual_ambiguity
            <= thresholds.max_view_subgoal_residual_ambiguity
        ),
    }
    base_evaluated = base_uncertainty is not None
    if base_evaluated:
        conditions["base-steerable-headroom"] = (
            base_uncertainty.steerable_headroom_nll
            >= thresholds.min_steerable_headroom_nll
        )
        conditions["base-uncertain-fraction"] = (
            base_uncertainty.fraction_base_uncertain
            >= thresholds.min_fraction_base_uncertain
        )
    failed = tuple(name for name, passed in conditions.items() if not passed)
    return ConflictInstrumentAdmission(
        valid=not failed,
        condition_conflict_fraction=conditions["conflict-row-fraction"],
        condition_constant_operator_error=conditions["constant-operator-error"],
        condition_view_subgoal_determinism=conditions["view-subgoal-determinism"],
        condition_base_headroom=conditions.get("base-steerable-headroom", False),
        condition_base_uncertain_fraction=conditions.get(
            "base-uncertain-fraction", False
        ),
        base_uncertainty_evaluated=base_evaluated,
        failed_conditions=failed,
        description=(
            "Goal-ambiguous junction instrument is valid: conflicting mapping "
            "leaves headroom no constant operator can saturate."
            if not failed
            else "Conflict instrument invalid: " + ", ".join(failed)
        ),
    )


__all__ = [
    "ETA_CONFLICT_INSTRUMENT_SCHEMA_VERSION",
    "GOAL_AMBIGUOUS_JUNCTION_PROTOCOL",
    "ConflictBaseUncertaintyMetrics",
    "ConflictHeadroomMetrics",
    "ConflictInstrumentAdmission",
    "ConflictInstrumentReport",
    "ConflictInstrumentThresholds",
    "ConflictJunctionRow",
    "assess_conflict_instrument",
    "build_conflict_junction_rows",
    "compute_conflict_headroom",
    "measure_base_uncertainty",
]
