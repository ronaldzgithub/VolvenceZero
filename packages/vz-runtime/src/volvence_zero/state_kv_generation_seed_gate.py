"""Cross-generation-seed gate for State-KV retention evidence.

The gate is read-only. It consumes one published retention report per
generation seed and verifies that the same frozen material remains retained
across distinct stochastic rollouts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from volvence_zero.state_kv_identification import (
    CONTROL_ARM_LABEL,
    PREFIX_ARM_LABEL,
    bootstrap_matching_ci,
)
from volvence_zero.state_kv_retention_gate import RETENTION_GATE_SCHEMA_VERSION

GENERATION_SEED_GATE_SCHEMA_VERSION = "state-kv-generation-seed-gate.v1"
DEFAULT_BOOTSTRAP_SEEDS: tuple[int, ...] = (
    20260726,
    20260727,
    20260728,
    1701,
    31337,
)


class SeedGateClaimState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class GenerationSeedGateState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class SeedGateClaim:
    name: str
    state: SeedGateClaimState
    detail: str

    def as_json_dict(self) -> dict[str, object]:
        return {
            "claim": self.name,
            "state": self.state.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SeedArmReadout:
    arm_label: str
    correct: int
    total: int
    accuracy: float
    ci_low: float
    ci_high: float

    def as_json_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm_label,
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
        }


@dataclass(frozen=True)
class GenerationSeedPanel:
    report_path: Path
    gate_state: str
    sampling_seed: int
    judge_model_id: str
    prefix_artifact_id: str
    substrate_fingerprint: str
    candidate_arm: str
    required_p2_pairs: tuple[str, ...]
    rollout_material: tuple[tuple[object, ...], ...]
    stochastic_generation_rollout_covered: bool
    candidate: SeedArmReadout
    control: SeedArmReadout

    def as_json_dict(self) -> dict[str, object]:
        return {
            "report_path": str(self.report_path),
            "gate_state": self.gate_state,
            "sampling_seed": self.sampling_seed,
            "judge_model_id": self.judge_model_id,
            "prefix_artifact_id": self.prefix_artifact_id,
            "substrate_fingerprint": self.substrate_fingerprint,
            "candidate_arm": self.candidate_arm,
            "required_p2_pairs": list(self.required_p2_pairs),
            "rollout_material": [list(item) for item in self.rollout_material],
            "stochastic_generation_rollout_covered": (
                self.stochastic_generation_rollout_covered
            ),
            "candidate": self.candidate.as_json_dict(),
            "control": self.control.as_json_dict(),
        }


@dataclass(frozen=True)
class GenerationSeedGateReport:
    schema_version: str
    gate_state: GenerationSeedGateState
    claims: tuple[SeedGateClaim, ...]
    panels: tuple[GenerationSeedPanel, ...]
    min_generation_seeds: int
    generation_seeds: tuple[int, ...]
    bootstrap_seeds: tuple[int, ...]
    aggregate_candidate: SeedArmReadout
    aggregate_control: SeedArmReadout
    notes: tuple[str, ...]

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate_state": self.gate_state.value,
            "claims": [claim.as_json_dict() for claim in self.claims],
            "panels": [panel.as_json_dict() for panel in self.panels],
            "min_generation_seeds": self.min_generation_seeds,
            "generation_seeds": list(self.generation_seeds),
            "bootstrap_seeds": list(self.bootstrap_seeds),
            "aggregate_candidate": self.aggregate_candidate.as_json_dict(),
            "aggregate_control": self.aggregate_control.as_json_dict(),
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_json_dict(), ensure_ascii=False, indent=2)


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _str(payload: Mapping[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} requires non-empty string field {key!r}")
    return value


def _int(payload: Mapping[str, Any], key: str, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} requires integer field {key!r}")
    return value


def _number(payload: Mapping[str, Any], key: str, path: Path) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{path} requires numeric field {key!r}")
    return float(value)


def _bool(payload: Mapping[str, Any], key: str, path: Path) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path} requires boolean field {key!r}")
    return value


def _arm(
    aggregates: Sequence[object], arm_label: str, path: Path
) -> SeedArmReadout:
    matches = [
        item
        for item in aggregates
        if isinstance(item, dict) and item.get("arm") == arm_label
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path} requires exactly one aggregate for {arm_label!r}"
        )
    item = matches[0]
    correct = _int(item, "correct", path)
    total = _int(item, "total", path)
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError(
            f"{path} has invalid {arm_label!r} counts {correct}/{total}"
        )
    return SeedArmReadout(
        arm_label=arm_label,
        correct=correct,
        total=total,
        accuracy=_number(item, "accuracy", path),
        ci_low=_number(item, "ci_low_min", path),
        ci_high=_number(item, "ci_high_max", path),
    )


def load_generation_seed_panel(
    report_path: Path | str,
) -> GenerationSeedPanel:
    """Load one retention report and freeze its generation-seed material."""

    path = Path(report_path).expanduser().resolve()
    payload = _read_object(path)
    if payload.get("schema_version") != RETENTION_GATE_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema {payload.get('schema_version')!r}; "
            f"expected {RETENTION_GATE_SCHEMA_VERSION!r}"
        )
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError(f"{path} requires non-empty inputs")
    if not all(isinstance(item, dict) for item in raw_inputs):
        raise ValueError(f"{path} inputs must be objects")
    inputs: list[Mapping[str, Any]] = list(raw_inputs)

    seeds = {_int(item, "sampling_seed", path) for item in inputs}
    judges = {_str(item, "judge_model_id", path) for item in inputs}
    artifacts = {_str(item, "prefix_artifact_id", path) for item in inputs}
    substrates = {_str(item, "substrate_fingerprint", path) for item in inputs}
    candidates = {_str(item, "candidate_arm", path) for item in inputs}
    if not all(len(values) == 1 for values in (seeds, judges, artifacts, substrates, candidates)):
        raise ValueError(
            f"{path} must contain one seed/judge/artifact/substrate/candidate"
        )

    rollout_material = tuple(
        sorted(
            (
                _str(item, "lane", path),
                _str(item, "p2_pair", path),
                _int(item, "probe_limit", path),
                _int(item, "probe_count", path),
                _int(item, "case_count", path),
                _int(item, "max_new_tokens", path),
                round(_number(item, "temperature", path), 6),
                _bool(item, "stochastic_generation_rollout", path),
            )
            for item in inputs
        )
    )
    if not all(item[-1] for item in rollout_material):
        raise ValueError(f"{path} must contain only stochastic rollout inputs")
    raw_pairs = payload.get("required_p2_pairs")
    if not isinstance(raw_pairs, list) or not all(
        isinstance(item, str) and item for item in raw_pairs
    ):
        raise ValueError(f"{path} requires string required_p2_pairs")
    raw_aggregates = payload.get("aggregates")
    if not isinstance(raw_aggregates, list):
        raise ValueError(f"{path} requires aggregates")

    return GenerationSeedPanel(
        report_path=path,
        gate_state=_str(payload, "gate_state", path),
        sampling_seed=next(iter(seeds)),
        judge_model_id=next(iter(judges)),
        prefix_artifact_id=next(iter(artifacts)),
        substrate_fingerprint=next(iter(substrates)),
        candidate_arm=next(iter(candidates)),
        required_p2_pairs=tuple(sorted(raw_pairs)),
        rollout_material=rollout_material,
        stochastic_generation_rollout_covered=_bool(
            payload, "stochastic_generation_rollout_covered", path
        ),
        candidate=_arm(raw_aggregates, PREFIX_ARM_LABEL, path),
        control=_arm(raw_aggregates, CONTROL_ARM_LABEL, path),
    )


def _aggregate(
    panels: Sequence[GenerationSeedPanel],
    *,
    candidate: bool,
    bootstrap_seeds: Sequence[int],
) -> SeedArmReadout:
    readouts = [
        panel.candidate if candidate else panel.control for panel in panels
    ]
    correct = sum(item.correct for item in readouts)
    total = sum(item.total for item in readouts)
    votes = [True] * correct + [False] * (total - correct)
    intervals = [
        bootstrap_matching_ci(votes, seed=seed)
        for seed in bootstrap_seeds
    ]
    return SeedArmReadout(
        arm_label=PREFIX_ARM_LABEL if candidate else CONTROL_ARM_LABEL,
        correct=correct,
        total=total,
        accuracy=correct / total,
        ci_low=min(item[1] for item in intervals),
        ci_high=max(item[2] for item in intervals),
    )


def _overall_state(
    claims: Sequence[SeedGateClaim],
) -> GenerationSeedGateState:
    if any(claim.state is SeedGateClaimState.FAIL for claim in claims):
        return GenerationSeedGateState.FAIL
    if any(
        claim.state is SeedGateClaimState.INSUFFICIENT_DATA for claim in claims
    ):
        return GenerationSeedGateState.INSUFFICIENT_DATA
    return GenerationSeedGateState.PASS


def build_generation_seed_gate_report(
    *,
    panels: Sequence[GenerationSeedPanel],
    min_generation_seeds: int = 3,
    bootstrap_seeds: Sequence[int] = DEFAULT_BOOTSTRAP_SEEDS,
) -> GenerationSeedGateReport:
    """Build a gate over distinct stochastic generation seeds."""

    if not panels:
        raise ValueError("generation-seed gate requires at least one panel")
    if min_generation_seeds < 2:
        raise ValueError("min_generation_seeds must be >= 2")
    if not bootstrap_seeds:
        raise ValueError("bootstrap_seeds must not be empty")

    claims: list[SeedGateClaim] = []
    material = {
        (
            panel.judge_model_id,
            panel.prefix_artifact_id,
            panel.substrate_fingerprint,
            panel.candidate_arm,
            panel.required_p2_pairs,
            panel.rollout_material,
        )
        for panel in panels
    }
    claims.append(
        SeedGateClaim(
            "claim_consistent_material",
            (
                SeedGateClaimState.PASS
                if len(material) == 1
                else SeedGateClaimState.FAIL
            ),
            (
                "all seed panels share one judge/artifact/substrate/pair/rollout matrix"
                if len(material) == 1
                else f"seed panels contain {len(material)} material matrices"
            ),
        )
    )

    generation_seeds = tuple(sorted({panel.sampling_seed for panel in panels}))
    duplicate_seeds = len(generation_seeds) != len(panels)
    enough = len(generation_seeds) >= min_generation_seeds
    claims.append(
        SeedGateClaim(
            "claim_generation_seed_coverage",
            (
                SeedGateClaimState.FAIL
                if duplicate_seeds
                else (
                    SeedGateClaimState.PASS
                    if enough
                    else SeedGateClaimState.INSUFFICIENT_DATA
                )
            ),
            (
                "duplicate generation-seed panels are not allowed: "
                f"{generation_seeds}"
                if duplicate_seeds
                else (
                    f"{len(generation_seeds)} distinct generation seeds: "
                    f"{generation_seeds}; required={min_generation_seeds}"
                )
            ),
        )
    )

    retained = all(
        panel.gate_state == "pass"
        and panel.stochastic_generation_rollout_covered
        for panel in panels
    )
    claims.append(
        SeedGateClaim(
            "claim_each_seed_retained",
            SeedGateClaimState.PASS if retained else SeedGateClaimState.FAIL,
            (
                "every generation seed passed stochastic retention"
                if retained
                else "at least one generation seed failed stochastic retention"
            ),
        )
    )

    per_seed_identification = all(
        panel.candidate.ci_low > 0.5 for panel in panels
    )
    per_seed_causality = all(
        panel.control.ci_low <= 0.5 <= panel.control.ci_high
        and panel.candidate.ci_low > 0.5
        for panel in panels
    )
    aggregate_candidate = _aggregate(
        panels, candidate=True, bootstrap_seeds=bootstrap_seeds
    )
    aggregate_control = _aggregate(
        panels, candidate=False, bootstrap_seeds=bootstrap_seeds
    )
    identification = (
        per_seed_identification and aggregate_candidate.ci_low > 0.5
    )
    causality = (
        per_seed_causality
        and aggregate_control.ci_low <= 0.5 <= aggregate_control.ci_high
        and aggregate_candidate.ci_low > 0.5
    )
    claims.append(
        SeedGateClaim(
            "claim_cross_seed_identification",
            (
                SeedGateClaimState.PASS
                if identification
                else SeedGateClaimState.FAIL
            ),
            (
                f"per-seed candidate CIs clear chance; aggregate "
                f"{aggregate_candidate.correct}/{aggregate_candidate.total}, "
                f"CI=({aggregate_candidate.ci_low:.3f}, "
                f"{aggregate_candidate.ci_high:.3f})"
                if identification
                else "candidate failed chance clearance for a seed or aggregate"
            ),
        )
    )
    claims.append(
        SeedGateClaim(
            "claim_cross_seed_carrier_causality",
            SeedGateClaimState.PASS if causality else SeedGateClaimState.FAIL,
            (
                f"all controls cover chance; aggregate control "
                f"{aggregate_control.correct}/{aggregate_control.total}, "
                f"CI=({aggregate_control.ci_low:.3f}, "
                f"{aggregate_control.ci_high:.3f})"
                if causality
                else "control/candidate causality failed for a seed or aggregate"
            ),
        )
    )

    state = _overall_state(claims)
    return GenerationSeedGateReport(
        schema_version=GENERATION_SEED_GATE_SCHEMA_VERSION,
        gate_state=state,
        claims=tuple(claims),
        panels=tuple(panels),
        min_generation_seeds=min_generation_seeds,
        generation_seeds=generation_seeds,
        bootstrap_seeds=tuple(bootstrap_seeds),
        aggregate_candidate=aggregate_candidate,
        aggregate_control=aggregate_control,
        notes=(
            "This read-only gate separates generation-seed replication from "
            "the bootstrap seeds used only to estimate confidence intervals.",
        ),
    )
