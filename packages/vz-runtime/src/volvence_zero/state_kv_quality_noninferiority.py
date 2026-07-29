"""Direct G-vs-B-prime quality non-inferiority evidence gate."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Sequence

QUALITY_NONINFERIORITY_SCHEMA_VERSION = (
    "state-kv-quality-noninferiority.v1"
)


@dataclass(frozen=True)
class QualityPair:
    experiment_id: str
    scenario_id: str
    sampling_seed: int
    judge_model_id: str
    substrate_fingerprint: str
    candidate_accuracy: float
    bprime_accuracy: float

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.scenario_id:
            raise ValueError("quality pair requires experiment and scenario ids")
        if self.sampling_seed < 0:
            raise ValueError("quality pair sampling_seed must be non-negative")
        if not self.judge_model_id or not self.substrate_fingerprint:
            raise ValueError(
                "quality pair requires judge and substrate fingerprints"
            )
        for name, value in (
            ("candidate_accuracy", self.candidate_accuracy),
            ("bprime_accuracy", self.bprime_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def delta(self) -> float:
        return self.candidate_accuracy - self.bprime_accuracy


@dataclass(frozen=True)
class QualityNoninferiorityVerdict:
    pairs: tuple[QualityPair, ...]
    noninferiority_margin: float
    minimum_pairs: int
    minimum_scenarios: int
    minimum_seeds: int
    minimum_judges: int
    bootstrap_seed: int
    mean_delta: float | None
    delta_ci: tuple[float, float] | None
    gate_state: str
    detail: str

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": QUALITY_NONINFERIORITY_SCHEMA_VERSION,
            "gate_state": self.gate_state,
            "candidate_arm": "state-kv-arm-g-prefix-pure",
            "baseline_arm": "state-kv-arm-bprime",
            "noninferiority_margin": self.noninferiority_margin,
            "minimum_pairs": self.minimum_pairs,
            "minimum_scenarios": self.minimum_scenarios,
            "minimum_seeds": self.minimum_seeds,
            "minimum_judges": self.minimum_judges,
            "bootstrap_seed": self.bootstrap_seed,
            "mean_delta": self.mean_delta,
            "delta_ci": list(self.delta_ci) if self.delta_ci else None,
            "scenario_ids": sorted({pair.scenario_id for pair in self.pairs}),
            "sampling_seeds": sorted({pair.sampling_seed for pair in self.pairs}),
            "judge_model_ids": sorted(
                {pair.judge_model_id for pair in self.pairs}
            ),
            "substrate_fingerprints": sorted(
                {pair.substrate_fingerprint for pair in self.pairs}
            ),
            "pairs": [
                {
                    "experiment_id": pair.experiment_id,
                    "scenario_id": pair.scenario_id,
                    "sampling_seed": pair.sampling_seed,
                    "judge_model_id": pair.judge_model_id,
                    "substrate_fingerprint": pair.substrate_fingerprint,
                    "candidate_accuracy": pair.candidate_accuracy,
                    "bprime_accuracy": pair.bprime_accuracy,
                    "delta": pair.delta,
                }
                for pair in self.pairs
            ],
            "claims": [
                {
                    "claim": "claim_quality_noninferior_to_bprime",
                    "state": self.gate_state,
                    "detail": self.detail,
                }
            ],
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
    if not values:
        raise ValueError("quality bootstrap requires at least one value")
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


def build_quality_noninferiority_verdict(
    *,
    pairs: Sequence[QualityPair],
    noninferiority_margin: float = 0.0,
    minimum_pairs: int = 8,
    minimum_scenarios: int = 2,
    minimum_seeds: int = 3,
    minimum_judges: int = 2,
    bootstrap_seed: int = 7501,
) -> QualityNoninferiorityVerdict:
    if noninferiority_margin < 0.0:
        raise ValueError("noninferiority_margin must be non-negative")
    requirements = (
        minimum_pairs,
        minimum_scenarios,
        minimum_seeds,
        minimum_judges,
    )
    if any(value < 1 for value in requirements):
        raise ValueError("quality coverage requirements must be positive")
    substrate_fingerprints = {pair.substrate_fingerprint for pair in pairs}
    if len(substrate_fingerprints) > 1:
        raise ValueError(
            "quality pairs must share one frozen substrate fingerprint"
        )
    deltas = [pair.delta for pair in pairs]
    mean_delta = sum(deltas) / len(deltas) if deltas else None
    delta_ci = _bootstrap_ci(deltas, seed=bootstrap_seed) if deltas else None
    coverage = (
        len(pairs) >= minimum_pairs
        and len({pair.scenario_id for pair in pairs}) >= minimum_scenarios
        and len({pair.sampling_seed for pair in pairs}) >= minimum_seeds
        and len({pair.judge_model_id for pair in pairs}) >= minimum_judges
        and len(substrate_fingerprints) == 1
    )
    if not coverage or delta_ci is None:
        gate_state = "insufficient_data"
    elif delta_ci[0] >= -noninferiority_margin:
        gate_state = "pass"
    else:
        gate_state = "fail"
    detail = (
        f"pairs={len(pairs)}, scenarios="
        f"{len({pair.scenario_id for pair in pairs})}, seeds="
        f"{len({pair.sampling_seed for pair in pairs})}, judges="
        f"{len({pair.judge_model_id for pair in pairs})}, "
        f"mean_delta={mean_delta!r}, ci={delta_ci!r}, "
        f"margin={noninferiority_margin:.4f}"
    )
    return QualityNoninferiorityVerdict(
        pairs=tuple(pairs),
        noninferiority_margin=noninferiority_margin,
        minimum_pairs=minimum_pairs,
        minimum_scenarios=minimum_scenarios,
        minimum_seeds=minimum_seeds,
        minimum_judges=minimum_judges,
        bootstrap_seed=bootstrap_seed,
        mean_delta=mean_delta,
        delta_ci=delta_ci,
        gate_state=gate_state,
        detail=detail,
    )


__all__ = [
    "QUALITY_NONINFERIORITY_SCHEMA_VERSION",
    "QualityNoninferiorityVerdict",
    "QualityPair",
    "build_quality_noninferiority_verdict",
]
