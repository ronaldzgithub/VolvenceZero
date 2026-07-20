"""Evidence gate for learned backend ACTIVE candidacy (CP-15 / CP-23).

This module does NOT flip runtime defaults. It codifies the plan's promotion
rules so operators can evaluate a frozen evidence artifact before changing one
`FinalRolloutConfig` field at a time:

runtime -> SSL -> Internal RL -> CMS torch.

Validation-delta gate versions (threshold pre-registration discipline,
`docs/specs/credit-and-self-modification.md` §可证伪发布门纪律 rule 2):

- **v1** (historical): absolute trailing-window MAE improvement
  ``validation_delta >= 0.02``. Kept as the criterion for all observation
  windows opened before 2026-07-20; artifacts from those windows are
  re-evaluated under v1 only.
- **v2** (pre-registered 2026-07-20, window id
  ``cp11-validation-v2-2026-07-20``): per-axis **relative** improvement of
  the learned head over the lag-1 persistence baseline, computed on the
  filled trailing window, excluding near-constant axes
  (``window_target_std < 0.01``). Promotion requires every informative axis
  to improve by ``>= 15%`` and at least 2 informative axes. Motivation: the
  510-turn re-measurement (`research/validation-delta-rca-2026-07-20.md`
  §7) showed axis target stds of 0.006-0.032, making the absolute 0.02
  threshold structurally unreachable even for heads that beat persistence
  by 36%. v2 only applies to observation windows opened AFTER this
  pre-registration; re-reading old windows under v2 is a what-if, never a
  promotion verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

VALIDATION_GATE_V2_WINDOW_ID = "cp11-validation-v2-2026-07-20"
VALIDATION_DELTA_V2_MIN_RELATIVE_IMPROVEMENT = 0.15
VALIDATION_DELTA_V2_INFORMATIVE_STD_FLOOR = 0.01
VALIDATION_DELTA_V2_MIN_INFORMATIVE_AXES = 2


class LearnedBackendComponent(str, Enum):
    TEMPORAL_RUNTIME = "temporal_runtime_backend"
    TEMPORAL_SSL = "temporal_ssl_backend"
    INTERNAL_RL = "internal_rl_backend"
    CMS_TORCH = "cms_torch_backend"


@dataclass(frozen=True)
class ValidationDeltaV2Readout:
    """Pre-registered v2 validation readout (relative-to-persistence).

    ``min_relative_improvement`` is the minimum over informative axes of
    ``(persistence_mae - learned_mae) / persistence_mae``. Axes whose
    ``target_std`` sits below the informative floor are excluded (a
    near-constant target makes constant prediction optimal; forcing a
    learned head to beat it measures noise-fitting, not learning).
    """

    window_filled: bool
    informative_axes: tuple[str, ...]
    excluded_axes: tuple[str, ...]
    per_axis_relative_improvement: tuple[tuple[str, float], ...]
    min_relative_improvement: float
    blocking_reasons: tuple[str, ...] = field(default=())

    @property
    def passed(self) -> bool:
        return not self.blocking_reasons


def compute_validation_delta_v2(
    *,
    window_filled: bool,
    axis_learned_maes: Mapping[str, float],
    axis_persistence_maes: Mapping[str, float],
    axis_target_stds: Mapping[str, float],
) -> ValidationDeltaV2Readout:
    """Compute the pre-registered v2 readout from CP-11 window fields.

    Pure and fail-loud: the three mappings must cover the same axis set;
    a missing axis is a contract violation, not a silent skip.
    """

    axes = tuple(sorted(axis_learned_maes))
    if tuple(sorted(axis_persistence_maes)) != axes or tuple(sorted(axis_target_stds)) != axes:
        raise ValueError(
            "validation_delta_v2 axis sets diverge: "
            f"learned={axes} persistence={tuple(sorted(axis_persistence_maes))} "
            f"stds={tuple(sorted(axis_target_stds))}"
        )
    informative: list[str] = []
    excluded: list[str] = []
    per_axis: list[tuple[str, float]] = []
    blocking: list[str] = []
    for axis in axes:
        if axis_target_stds[axis] < VALIDATION_DELTA_V2_INFORMATIVE_STD_FLOOR:
            excluded.append(axis)
            continue
        informative.append(axis)
        persistence = axis_persistence_maes[axis]
        if persistence <= 0.0:
            # A zero-error persistence baseline on an informative axis is
            # unimprovable; count it as a hard block rather than dividing
            # by zero or silently excluding.
            per_axis.append((axis, 0.0))
            blocking.append(f"axis {axis}: persistence_mae<=0 (unimprovable)")
            continue
        relative = (persistence - axis_learned_maes[axis]) / persistence
        per_axis.append((axis, round(relative, 4)))
    minimum = min((value for _, value in per_axis), default=0.0)
    if not window_filled:
        blocking.append("window_unfilled")
    if len(informative) < VALIDATION_DELTA_V2_MIN_INFORMATIVE_AXES:
        blocking.append(
            f"informative_axes {len(informative)}<{VALIDATION_DELTA_V2_MIN_INFORMATIVE_AXES}"
        )
    for axis, value in per_axis:
        if value < VALIDATION_DELTA_V2_MIN_RELATIVE_IMPROVEMENT:
            blocking.append(
                f"axis {axis}: relative_improvement {value:.4f}"
                f"<{VALIDATION_DELTA_V2_MIN_RELATIVE_IMPROVEMENT:.2f}"
            )
    return ValidationDeltaV2Readout(
        window_filled=window_filled,
        informative_axes=tuple(informative),
        excluded_axes=tuple(excluded),
        per_axis_relative_improvement=tuple(per_axis),
        min_relative_improvement=round(minimum, 4),
        blocking_reasons=tuple(blocking),
    )


@dataclass(frozen=True)
class LearnedActiveEvidence:
    component: LearnedBackendComponent
    real_trace_turns: int
    validation_delta: float
    strict_eta_gate_passed: bool
    pe_off_control_direction_correct: bool
    eta_off_control_direction_correct: bool
    rollback_drill_passed: bool
    latency_slo_ok: bool
    safety_gate_ok: bool
    prior_runtime_active: bool = False
    prior_ssl_active: bool = False
    internal_rl_no_reward_leakage: bool = True
    cms_retention_non_degrading: bool = True
    cms_absorption_improved: bool = True
    # v2 window evidence (None for v1-window artifacts).
    validation_delta_v2: ValidationDeltaV2Readout | None = None

    def __post_init__(self) -> None:
        if self.real_trace_turns < 0:
            raise ValueError("real_trace_turns must be non-negative")


@dataclass(frozen=True)
class LearnedActiveGateVerdict:
    component: LearnedBackendComponent
    eligible: bool
    required_turns: int
    required_validation_delta: float
    missing_gates: tuple[str, ...]
    description: str
    validation_gate_version: str = "v1"


def evaluate_learned_active_candidate(
    evidence: LearnedActiveEvidence,
    *,
    validation_gate_version: str = "v1",
) -> LearnedActiveGateVerdict:
    """Return whether one component may be promoted to ACTIVE candidate.

    The evaluator is intentionally conservative and component-local. It never
    treats parity as capability evidence; every component requires real traces,
    rollback, controls, latency and safety.

    ``validation_gate_version`` selects the pre-registered validation
    criterion for the artifact's observation window (see module docstring):
    ``"v1"`` = absolute delta >= 0.02 (windows opened before 2026-07-20);
    ``"v2"`` = per-axis relative-to-persistence readout (windows opened
    after the 2026-07-20 pre-registration; requires
    ``evidence.validation_delta_v2``). Anything else fails loudly.
    """

    if validation_gate_version not in {"v1", "v2"}:
        raise ValueError(
            f"unknown validation_gate_version: {validation_gate_version!r}"
        )
    required_turns = 500
    required_delta = 0.02
    missing: list[str] = []
    if evidence.real_trace_turns < required_turns:
        missing.append(f"real_trace_turns<{required_turns}")
    if validation_gate_version == "v1":
        if evidence.validation_delta < required_delta:
            missing.append(f"validation_delta<{required_delta:.2f}")
    else:
        required_delta = VALIDATION_DELTA_V2_MIN_RELATIVE_IMPROVEMENT
        readout = evidence.validation_delta_v2
        if readout is None:
            missing.append("validation_delta_v2_readout_missing")
        elif not readout.passed:
            missing.extend(
                f"validation_delta_v2:{reason}" for reason in readout.blocking_reasons
            )
    if not evidence.strict_eta_gate_passed:
        missing.append("strict_eta_gate")
    if not evidence.pe_off_control_direction_correct:
        missing.append("pe_off_control")
    if not evidence.eta_off_control_direction_correct:
        missing.append("eta_off_control")
    if not evidence.rollback_drill_passed:
        missing.append("rollback_drill")
    if not evidence.latency_slo_ok:
        missing.append("latency_slo")
    if not evidence.safety_gate_ok:
        missing.append("safety_gate")

    if evidence.component is LearnedBackendComponent.TEMPORAL_SSL and not evidence.prior_runtime_active:
        missing.append("runtime_active_first")
    if evidence.component is LearnedBackendComponent.INTERNAL_RL:
        if not evidence.prior_runtime_active:
            missing.append("runtime_active_first")
        if not evidence.prior_ssl_active:
            missing.append("ssl_active_first")
        if not evidence.internal_rl_no_reward_leakage:
            missing.append("reward_leakage")
    if evidence.component is LearnedBackendComponent.CMS_TORCH:
        if not evidence.cms_retention_non_degrading:
            missing.append("cms_retention")
        if not evidence.cms_absorption_improved:
            missing.append("cms_absorption")

    missing_tuple = tuple(dict.fromkeys(missing))
    eligible = not missing_tuple
    return LearnedActiveGateVerdict(
        component=evidence.component,
        eligible=eligible,
        required_turns=required_turns,
        required_validation_delta=required_delta,
        missing_gates=missing_tuple,
        description=(
            f"{evidence.component.value} ACTIVE candidate "
            f"{'eligible' if eligible else 'blocked'}; "
            f"validation_gate={validation_gate_version}; "
            f"missing={missing_tuple or ('none',)}"
        ),
        validation_gate_version=validation_gate_version,
    )


__all__ = [
    "LearnedActiveEvidence",
    "LearnedActiveGateVerdict",
    "LearnedBackendComponent",
    "VALIDATION_DELTA_V2_INFORMATIVE_STD_FLOOR",
    "VALIDATION_DELTA_V2_MIN_INFORMATIVE_AXES",
    "VALIDATION_DELTA_V2_MIN_RELATIVE_IMPROVEMENT",
    "VALIDATION_GATE_V2_WINDOW_ID",
    "ValidationDeltaV2Readout",
    "compute_validation_delta_v2",
    "evaluate_learned_active_candidate",
]
