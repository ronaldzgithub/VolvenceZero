"""Aggregate retention gate for State-KV identification verdicts.

This module is deliberately read-only: it consumes already written
``verdict_identification.json`` artifacts and their adjacent
``substrate_fingerprint.json`` files, then checks whether the retained result
survives aggregation across held-out material and bootstrap seeds. It does not
call a model, judge, trainer, or owner API, so it cannot feed evaluation back
into learning.
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
    IDENTIFICATION_SCHEMA_VERSION,
    PREFIX_ARM_LABEL,
    bootstrap_matching_ci,
)

RETENTION_GATE_SCHEMA_VERSION = "state-kv-retention-gate.v1"

_CLAIM_CONSISTENT_ARTIFACT = "claim_consistent_artifact"
_CLAIM_INDIVIDUAL_RETAINED = "claim_individual_retained"
_CLAIM_HELDOUT_PAIR_COVERAGE = "claim_heldout_pair_coverage"
_CLAIM_AGGREGATE_IDENTIFICATION = "claim_aggregate_identification"
_CLAIM_AGGREGATE_CARRIER_CAUSALITY = "claim_aggregate_carrier_causality"
_CLAIM_BOOTSTRAP_SEED_STABILITY = "claim_bootstrap_seed_stability"

CLAIM_NAMES: tuple[str, ...] = (
    _CLAIM_CONSISTENT_ARTIFACT,
    _CLAIM_INDIVIDUAL_RETAINED,
    _CLAIM_HELDOUT_PAIR_COVERAGE,
    _CLAIM_AGGREGATE_IDENTIFICATION,
    _CLAIM_AGGREGATE_CARRIER_CAUSALITY,
    _CLAIM_BOOTSTRAP_SEED_STABILITY,
)


class GateClaimState(str, Enum):
    """Per-claim retention gate state."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class RetentionGateState(str, Enum):
    """Overall retention gate state."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class GateClaim:
    """One computed claim in the retention gate."""

    name: str
    state: GateClaimState
    detail: str

    def as_json_dict(self) -> dict[str, object]:
        return {
            "claim": self.name,
            "state": self.state.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ArmAggregate:
    """Summed blind-matching evidence for one arm."""

    arm_label: str
    correct: int
    total: int
    accuracy: float
    ci_low_min: float
    ci_high_max: float
    bootstrap_seeds: tuple[int, ...]

    def as_json_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm_label,
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 6),
            "ci_low_min": round(self.ci_low_min, 6),
            "ci_high_max": round(self.ci_high_max, 6),
            "bootstrap_seeds": list(self.bootstrap_seeds),
        }


@dataclass(frozen=True)
class RetentionGateEvidence:
    """Validated input verdict plus the fingerprint material needed for gating."""

    verdict_path: Path
    fingerprint_path: Path
    lane: str
    p2_pair: str
    candidate_arm: str
    verdict_state: str
    c5_grade: str
    substrate_fingerprint: str
    judge_model_id: str
    prefix_artifact_id: str
    candidate_correct: int
    candidate_total: int
    candidate_ci_low: float
    candidate_ci_high: float
    control_correct: int
    control_total: int
    control_ci_low: float
    control_ci_high: float

    def as_json_dict(self) -> dict[str, object]:
        return {
            "verdict_path": str(self.verdict_path),
            "fingerprint_path": str(self.fingerprint_path),
            "lane": self.lane,
            "p2_pair": self.p2_pair,
            "candidate_arm": self.candidate_arm,
            "verdict_state": self.verdict_state,
            "c5_grade": self.c5_grade,
            "substrate_fingerprint": self.substrate_fingerprint,
            "judge_model_id": self.judge_model_id,
            "prefix_artifact_id": self.prefix_artifact_id,
            "candidate": {
                "correct": self.candidate_correct,
                "total": self.candidate_total,
                "ci_low": round(self.candidate_ci_low, 6),
                "ci_high": round(self.candidate_ci_high, 6),
            },
            "control": {
                "correct": self.control_correct,
                "total": self.control_total,
                "ci_low": round(self.control_ci_low, 6),
                "ci_high": round(self.control_ci_high, 6),
            },
        }


@dataclass(frozen=True)
class RetentionGateReport:
    """The published aggregate retention verdict."""

    schema_version: str
    gate_state: RetentionGateState
    claims: tuple[GateClaim, ...]
    inputs: tuple[RetentionGateEvidence, ...]
    aggregates: tuple[ArmAggregate, ...]
    required_p2_pairs: tuple[str, ...]
    bootstrap_seeds: tuple[int, ...]
    stochastic_generation_rollout_covered: bool
    notes: tuple[str, ...]

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate_state": self.gate_state.value,
            "claims": [claim.as_json_dict() for claim in self.claims],
            "inputs": [item.as_json_dict() for item in self.inputs],
            "aggregates": [item.as_json_dict() for item in self.aggregates],
            "required_p2_pairs": list(self.required_p2_pairs),
            "bootstrap_seeds": list(self.bootstrap_seeds),
            "stochastic_generation_rollout_covered": (
                self.stochastic_generation_rollout_covered
            ),
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(), ensure_ascii=False, indent=2, sort_keys=False
        )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _str_field(payload: Mapping[str, Any], key: str, *, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} requires non-empty string field {key!r}")
    return value


def _optional_str_field(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"field {key!r} must be a string when present")
    return value


def _matching_readout(
    verdict: Mapping[str, Any],
    *,
    arm_label: str,
    path: Path,
) -> Mapping[str, Any]:
    raw_matching = verdict.get("matching")
    if not isinstance(raw_matching, list):
        raise ValueError(f"{path} requires a list matching field")
    matches = [
        item
        for item in raw_matching
        if isinstance(item, dict) and item.get("arm") == arm_label
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path} requires exactly one matching readout for {arm_label!r}"
        )
    return matches[0]


def _int_field(payload: Mapping[str, Any], key: str, *, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{path} requires integer field {key!r}")
    return value


def _float_field(payload: Mapping[str, Any], key: str, *, path: Path) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{path} requires numeric field {key!r}")
    return float(value)


def load_retention_evidence(verdict_path: Path | str) -> RetentionGateEvidence:
    """Load and validate one identification verdict plus its fingerprint."""

    resolved = Path(verdict_path).expanduser().resolve()
    verdict = _read_json_object(resolved)
    if verdict.get("schema_version") != IDENTIFICATION_SCHEMA_VERSION:
        raise ValueError(
            f"{resolved} has schema {verdict.get('schema_version')!r}; "
            f"expected {IDENTIFICATION_SCHEMA_VERSION!r}"
        )
    candidate_arm = _str_field(verdict, "candidate_arm", path=resolved)
    candidate = _matching_readout(verdict, arm_label=candidate_arm, path=resolved)
    control = _matching_readout(
        verdict, arm_label=CONTROL_ARM_LABEL, path=resolved
    )

    fingerprint_path = resolved.with_name("substrate_fingerprint.json")
    fingerprint = _read_json_object(fingerprint_path)
    material = fingerprint.get("identification_material")
    if not isinstance(material, dict):
        raise ValueError(
            f"{fingerprint_path} requires identification_material object"
        )
    return RetentionGateEvidence(
        verdict_path=resolved,
        fingerprint_path=fingerprint_path,
        lane=_str_field(material, "lane", path=fingerprint_path),
        p2_pair=_optional_str_field(material, "p2_pair"),
        candidate_arm=candidate_arm,
        verdict_state=_str_field(verdict, "verdict_state", path=resolved),
        c5_grade=_str_field(verdict, "c5_grade", path=resolved),
        substrate_fingerprint=_str_field(
            verdict, "substrate_fingerprint", path=resolved
        ),
        judge_model_id=_str_field(verdict, "judge_model_id", path=resolved),
        prefix_artifact_id=_str_field(
            fingerprint,
            "personal_conditioning_prefix_id",
            path=fingerprint_path,
        ),
        candidate_correct=_int_field(candidate, "correct", path=resolved),
        candidate_total=_int_field(candidate, "total", path=resolved),
        candidate_ci_low=_float_field(candidate, "ci_low", path=resolved),
        candidate_ci_high=_float_field(candidate, "ci_high", path=resolved),
        control_correct=_int_field(control, "correct", path=resolved),
        control_total=_int_field(control, "total", path=resolved),
        control_ci_low=_float_field(control, "ci_low", path=resolved),
        control_ci_high=_float_field(control, "ci_high", path=resolved),
    )


def _state_for_claims(claims: Sequence[GateClaim]) -> RetentionGateState:
    if any(claim.state is GateClaimState.FAIL for claim in claims):
        return RetentionGateState.FAIL
    if any(claim.state is GateClaimState.INSUFFICIENT_DATA for claim in claims):
        return RetentionGateState.INSUFFICIENT_DATA
    return RetentionGateState.PASS


def _aggregate_arm(
    *,
    arm_label: str,
    correct: int,
    total: int,
    bootstrap_seeds: Sequence[int],
) -> ArmAggregate:
    if total <= 0:
        raise ValueError(f"{arm_label} aggregate requires positive total")
    if correct < 0 or correct > total:
        raise ValueError(
            f"{arm_label} aggregate correct count {correct} is outside total {total}"
        )
    votes = tuple([True] * correct + [False] * (total - correct))
    intervals = [
        bootstrap_matching_ci(votes, seed=seed)
        for seed in bootstrap_seeds
    ]
    return ArmAggregate(
        arm_label=arm_label,
        correct=correct,
        total=total,
        accuracy=correct / total,
        ci_low_min=min(item[1] for item in intervals),
        ci_high_max=max(item[2] for item in intervals),
        bootstrap_seeds=tuple(bootstrap_seeds),
    )


def build_retention_gate_report(
    *,
    evidences: Sequence[RetentionGateEvidence],
    required_p2_pairs: Sequence[str],
    bootstrap_seeds: Sequence[int] = (20260726, 20260727, 20260728),
) -> RetentionGateReport:
    """Build a retained/pending verdict from existing State-KV evidence."""

    if not evidences:
        raise ValueError("retention gate requires at least one input verdict")
    if not required_p2_pairs:
        raise ValueError("retention gate requires at least one required P2 pair")
    if not bootstrap_seeds:
        raise ValueError("retention gate requires at least one bootstrap seed")
    if len(set(bootstrap_seeds)) != len(tuple(bootstrap_seeds)):
        raise ValueError("bootstrap seeds must be unique")

    claims: list[GateClaim] = []
    prefix_ids = {item.prefix_artifact_id for item in evidences}
    substrates = {item.substrate_fingerprint for item in evidences}
    judges = {item.judge_model_id for item in evidences}
    candidates = {item.candidate_arm for item in evidences}
    if len(prefix_ids) == len(substrates) == len(judges) == len(candidates) == 1:
        claims.append(
            GateClaim(
                name=_CLAIM_CONSISTENT_ARTIFACT,
                state=GateClaimState.PASS,
                detail=(
                    f"{len(evidences)} verdicts share artifact "
                    f"{next(iter(prefix_ids))}, substrate {next(iter(substrates))}, "
                    f"judge {next(iter(judges))}, candidate {next(iter(candidates))}"
                ),
            )
        )
    else:
        claims.append(
            GateClaim(
                name=_CLAIM_CONSISTENT_ARTIFACT,
                state=GateClaimState.FAIL,
                detail=(
                    "input verdicts do not share one artifact/substrate/judge/"
                    f"candidate: artifacts={sorted(prefix_ids)}, "
                    f"substrates={sorted(substrates)}, judges={sorted(judges)}, "
                    f"candidates={sorted(candidates)}"
                ),
            )
        )

    bad_verdicts = [
        item.verdict_path.name
        for item in evidences
        if item.verdict_state != "retain-strict" or item.c5_grade != "decode-matched"
    ]
    if bad_verdicts:
        claims.append(
            GateClaim(
                name=_CLAIM_INDIVIDUAL_RETAINED,
                state=GateClaimState.FAIL,
                detail=(
                    "all input verdicts must be retain-strict/decode-matched; "
                    f"failed inputs: {', '.join(bad_verdicts)}"
                ),
            )
        )
    else:
        claims.append(
            GateClaim(
                name=_CLAIM_INDIVIDUAL_RETAINED,
                state=GateClaimState.PASS,
                detail=f"{len(evidences)} input verdicts are retain-strict",
            )
        )

    observed_pairs = {
        item.p2_pair for item in evidences if item.lane == "p2" and item.p2_pair
    }
    missing_pairs = sorted(set(required_p2_pairs) - observed_pairs)
    if missing_pairs:
        claims.append(
            GateClaim(
                name=_CLAIM_HELDOUT_PAIR_COVERAGE,
                state=GateClaimState.INSUFFICIENT_DATA,
                detail=(
                    "missing required P2 held-out pairs: "
                    f"{', '.join(missing_pairs)}"
                ),
            )
        )
    else:
        claims.append(
            GateClaim(
                name=_CLAIM_HELDOUT_PAIR_COVERAGE,
                state=GateClaimState.PASS,
                detail=(
                    f"covered required P2 held-out pairs: "
                    f"{', '.join(sorted(required_p2_pairs))}"
                ),
            )
        )

    candidate = _aggregate_arm(
        arm_label=PREFIX_ARM_LABEL,
        correct=sum(item.candidate_correct for item in evidences),
        total=sum(item.candidate_total for item in evidences),
        bootstrap_seeds=bootstrap_seeds,
    )
    control = _aggregate_arm(
        arm_label=CONTROL_ARM_LABEL,
        correct=sum(item.control_correct for item in evidences),
        total=sum(item.control_total for item in evidences),
        bootstrap_seeds=bootstrap_seeds,
    )
    if candidate.ci_low_min > 0.5:
        claims.append(
            GateClaim(
                name=_CLAIM_AGGREGATE_IDENTIFICATION,
                state=GateClaimState.PASS,
                detail=(
                    f"{candidate.correct}/{candidate.total} accuracy="
                    f"{candidate.accuracy:.3f}; min CI low across bootstrap "
                    f"seeds is {candidate.ci_low_min:.3f}"
                ),
            )
        )
    else:
        claims.append(
            GateClaim(
                name=_CLAIM_AGGREGATE_IDENTIFICATION,
                state=GateClaimState.FAIL,
                detail=(
                    f"aggregate candidate CI lower bound "
                    f"{candidate.ci_low_min:.3f} does not clear chance"
                ),
            )
        )

    control_at_chance = control.ci_low_min <= 0.5 <= control.ci_high_max
    if control_at_chance and candidate.ci_low_min > 0.5:
        claims.append(
            GateClaim(
                name=_CLAIM_AGGREGATE_CARRIER_CAUSALITY,
                state=GateClaimState.PASS,
                detail=(
                    f"{CONTROL_ARM_LABEL} aggregate CI "
                    f"({control.ci_low_min:.3f}, {control.ci_high_max:.3f}) "
                    f"covers chance; {PREFIX_ARM_LABEL} clears it"
                ),
            )
        )
    else:
        claims.append(
            GateClaim(
                name=_CLAIM_AGGREGATE_CARRIER_CAUSALITY,
                state=GateClaimState.FAIL,
                detail=(
                    f"control aggregate CI ({control.ci_low_min:.3f}, "
                    f"{control.ci_high_max:.3f}); candidate CI low "
                    f"{candidate.ci_low_min:.3f}"
                ),
            )
        )

    seed_floor = min(item.candidate_ci_low for item in evidences)
    if seed_floor > 0.5 and candidate.ci_low_min > 0.5:
        claims.append(
            GateClaim(
                name=_CLAIM_BOOTSTRAP_SEED_STABILITY,
                state=GateClaimState.PASS,
                detail=(
                    "all published per-verdict CI lows clear chance, and "
                    "aggregate CI lows clear chance across bootstrap seeds "
                    f"{tuple(bootstrap_seeds)}"
                ),
            )
        )
    else:
        claims.append(
            GateClaim(
                name=_CLAIM_BOOTSTRAP_SEED_STABILITY,
                state=GateClaimState.FAIL,
                detail=(
                    f"per-verdict CI low floor={seed_floor:.3f}; aggregate "
                    f"CI low floor={candidate.ci_low_min:.3f}"
                ),
            )
        )

    notes = (
        "This gate covers held-out pair aggregation and bootstrap-seed "
        "robustness only. Prefix-KV generation remains greedy-only, so true "
        "stochastic generation rollout stability is not covered here.",
    )
    return RetentionGateReport(
        schema_version=RETENTION_GATE_SCHEMA_VERSION,
        gate_state=_state_for_claims(claims),
        claims=tuple(claims),
        inputs=tuple(evidences),
        aggregates=(control, candidate),
        required_p2_pairs=tuple(required_p2_pairs),
        bootstrap_seeds=tuple(bootstrap_seeds),
        stochastic_generation_rollout_covered=False,
        notes=notes,
    )
