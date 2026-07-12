"""Evidence gate for learned backend ACTIVE candidacy (CP-15 / CP-23).

This module does NOT flip runtime defaults. It codifies the plan's promotion
rules so operators can evaluate a frozen evidence artifact before changing one
`FinalRolloutConfig` field at a time:

runtime -> SSL -> Internal RL -> CMS torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LearnedBackendComponent(str, Enum):
    TEMPORAL_RUNTIME = "temporal_runtime_backend"
    TEMPORAL_SSL = "temporal_ssl_backend"
    INTERNAL_RL = "internal_rl_backend"
    CMS_TORCH = "cms_torch_backend"


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


def evaluate_learned_active_candidate(
    evidence: LearnedActiveEvidence,
) -> LearnedActiveGateVerdict:
    """Return whether one component may be promoted to ACTIVE candidate.

    The evaluator is intentionally conservative and component-local. It never
    treats parity as capability evidence; every component requires real traces,
    rollback, controls, latency and safety.
    """

    required_turns = 500
    required_delta = 0.02
    missing: list[str] = []
    if evidence.real_trace_turns < required_turns:
        missing.append(f"real_trace_turns<{required_turns}")
    if evidence.validation_delta < required_delta:
        missing.append(f"validation_delta<{required_delta:.2f}")
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
            f"missing={missing_tuple or ('none',)}"
        ),
    )


__all__ = [
    "LearnedActiveEvidence",
    "LearnedActiveGateVerdict",
    "LearnedBackendComponent",
    "evaluate_learned_active_candidate",
]
