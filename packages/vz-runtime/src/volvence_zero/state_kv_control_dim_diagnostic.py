"""Read-only evidence gate for the State KV P5-d control-dimension question."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from math import sqrt
from typing import Sequence

CONTROL_DIM_SCHEMA_VERSION = "state-kv-control-dim-diagnostic.v2"
LEGACY_CONTROL_RANK = 3


@dataclass(frozen=True)
class ControlDimSample:
    """One matched full-code versus rank-3 outcome observation."""

    sample_id: str
    full_code: tuple[float, ...]
    full_outcome: float | None
    rank3_outcome: float | None
    dynamic_off_outcome: float | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("control-dimension sample requires sample_id")
        if len(self.full_code) <= LEGACY_CONTROL_RANK:
            raise ValueError(
                "control-dimension sample requires a full code wider than 3"
            )
        outcomes = (
            self.full_outcome,
            self.rank3_outcome,
            self.dynamic_off_outcome,
        )
        if any(value is None for value in outcomes) and any(
            value is not None for value in outcomes
        ):
            raise ValueError(
                "control-dimension sample must provide all three matched "
                "outcomes or none"
            )

    @property
    def tail_energy_ratio(self) -> float:
        total = sqrt(sum(value * value for value in self.full_code))
        tail = sqrt(
            sum(
                value * value
                for value in self.full_code[LEGACY_CONTROL_RANK:]
            )
        )
        return tail / total if total > 0.0 else 0.0


@dataclass(frozen=True)
class ControlDimClaim:
    claim: str
    state: str
    detail: str


@dataclass(frozen=True)
class ControlDimVerdict:
    artifact_id: str
    source_artifacts: tuple[str, ...]
    minimum_samples: int
    minimum_outcome_delta: float
    sample_count: int
    matched_outcome_count: int
    mean_tail_energy_ratio: float
    mean_full_minus_rank3_outcome: float | None
    outcome_delta_ci: tuple[float, float] | None
    mean_rank3_minus_off_outcome: float | None
    rank3_minus_off_ci: tuple[float, float] | None
    claims: tuple[ControlDimClaim, ...]

    @property
    def bottleneck_proven(self) -> bool:
        return bool(self.claims) and all(
            claim.state == "pass" for claim in self.claims
        )

    @property
    def gate_state(self) -> str:
        states = {claim.state for claim in self.claims}
        if self.bottleneck_proven:
            return "pass"
        if "fail" in states:
            return "fail"
        return "insufficient_data"

    @property
    def p5d_decision(self) -> str:
        if self.bottleneck_proven:
            return "proceed-to-full-dimension-artifact"
        if self.gate_state == "fail":
            return "retain-rank3-stop-d1-d2"
        return "await-matched-full-dimension-evidence"

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONTROL_DIM_SCHEMA_VERSION,
            "gate_state": self.gate_state,
            "artifact_id": self.artifact_id,
            "source_artifacts": list(self.source_artifacts),
            "legacy_control_rank": LEGACY_CONTROL_RANK,
            "minimum_samples": self.minimum_samples,
            "minimum_outcome_delta": self.minimum_outcome_delta,
            "sample_count": self.sample_count,
            "matched_outcome_count": self.matched_outcome_count,
            "mean_tail_energy_ratio": self.mean_tail_energy_ratio,
            "mean_full_minus_rank3_outcome": (
                self.mean_full_minus_rank3_outcome
            ),
            "outcome_delta_ci": (
                list(self.outcome_delta_ci)
                if self.outcome_delta_ci is not None
                else None
            ),
            "mean_rank3_minus_off_outcome": self.mean_rank3_minus_off_outcome,
            "rank3_minus_off_ci": (
                list(self.rank3_minus_off_ci)
                if self.rank3_minus_off_ci is not None
                else None
            ),
            "claims": [
                {
                    "claim": claim.claim,
                    "state": claim.state,
                    "detail": claim.detail,
                }
                for claim in self.claims
            ],
            "bottleneck_proven": self.bottleneck_proven,
            "p5d_decision": self.p5d_decision,
            "rollback": (
                "No substrate change was admitted; rank-3 remains the "
                "deployed behavior."
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def _bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 4000,
) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return (
        means[int(0.025 * (samples - 1))],
        means[int(0.975 * (samples - 1))],
    )


def build_control_dim_verdict(
    *,
    samples: Sequence[ControlDimSample],
    artifact_id: str,
    source_artifacts: Sequence[str],
    minimum_samples: int = 8,
    minimum_outcome_delta: float = 0.02,
    bootstrap_seed: int = 7401,
) -> ControlDimVerdict:
    if not artifact_id:
        raise ValueError("control-dimension verdict requires artifact_id")
    if minimum_samples < 2:
        raise ValueError("minimum_samples must be >= 2")
    if minimum_outcome_delta <= 0.0:
        raise ValueError("minimum_outcome_delta must be positive")

    matched = [
        sample for sample in samples if sample.full_outcome is not None
    ]
    deltas = [
        float(sample.full_outcome) - float(sample.rank3_outcome)
        for sample in matched
    ]
    rank3_off_deltas = [
        float(sample.rank3_outcome) - float(sample.dynamic_off_outcome)
        for sample in matched
    ]
    tail_mean = (
        sum(sample.tail_energy_ratio for sample in samples) / len(samples)
        if samples
        else 0.0
    )
    delta_mean = sum(deltas) / len(deltas) if deltas else None
    delta_ci = (
        _bootstrap_ci(deltas, seed=bootstrap_seed) if deltas else None
    )
    rank3_off_mean = (
        sum(rank3_off_deltas) / len(rank3_off_deltas)
        if rank3_off_deltas
        else None
    )
    rank3_off_ci = (
        _bootstrap_ci(rank3_off_deltas, seed=bootstrap_seed + 1)
        if rank3_off_deltas
        else None
    )

    if len(samples) < minimum_samples:
        width_state = "insufficient_data"
    elif tail_mean > 0.0:
        width_state = "pass"
    else:
        width_state = "fail"
    if len(matched) < minimum_samples or delta_ci is None:
        outcome_state = "insufficient_data"
    elif delta_ci[0] >= minimum_outcome_delta:
        outcome_state = "pass"
    else:
        outcome_state = "fail"
    if len(matched) < minimum_samples or rank3_off_ci is None:
        causal_state = "insufficient_data"
    elif rank3_off_ci[0] > 0.0:
        causal_state = "pass"
    else:
        causal_state = "fail"
    claims = (
        ControlDimClaim(
            claim="claim_full_code_contains_tail_signal",
            state=width_state,
            detail=(
                f"n={len(samples)}, mean tail/total L2="
                f"{tail_mean:.6f}"
            ),
        ),
        ControlDimClaim(
            claim="claim_rank3_control_beats_dynamic_residual_off",
            state=causal_state,
            detail=(
                f"matched_n={len(matched)}, mean_delta={rank3_off_mean!r}, "
                f"ci={rank3_off_ci!r}, required_ci_low>0"
            ),
        ),
        ControlDimClaim(
            claim="claim_full_dimension_improves_matched_outcome",
            state=outcome_state,
            detail=(
                f"matched_n={len(matched)}, mean_delta={delta_mean!r}, "
                f"ci={delta_ci!r}, required_ci_low="
                f"{minimum_outcome_delta:.6f}"
            ),
        ),
    )
    return ControlDimVerdict(
        artifact_id=artifact_id,
        source_artifacts=tuple(source_artifacts),
        minimum_samples=minimum_samples,
        minimum_outcome_delta=minimum_outcome_delta,
        sample_count=len(samples),
        matched_outcome_count=len(matched),
        mean_tail_energy_ratio=tail_mean,
        mean_full_minus_rank3_outcome=delta_mean,
        outcome_delta_ci=delta_ci,
        mean_rank3_minus_off_outcome=rank3_off_mean,
        rank3_minus_off_ci=rank3_off_ci,
        claims=claims,
    )


__all__ = [
    "CONTROL_DIM_SCHEMA_VERSION",
    "LEGACY_CONTROL_RANK",
    "ControlDimClaim",
    "ControlDimSample",
    "ControlDimVerdict",
    "build_control_dim_verdict",
]
