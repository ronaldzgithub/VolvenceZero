"""Multi-judge court gate for State-KV retention evidence.

This module is deliberately read-only. It consumes already published
``state-kv-retention-gate.v1`` reports, one per judge, and checks whether the
same State-KV artifact survives a small court of distinct judges under the
same held-out material and rollout configuration.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from volvence_zero.state_kv_identification import CONTROL_ARM_LABEL, PREFIX_ARM_LABEL
from volvence_zero.state_kv_retention_gate import RETENTION_GATE_SCHEMA_VERSION

JUDGE_COURT_SCHEMA_VERSION = "state-kv-judge-court.v1"

_CLAIM_CONSISTENT_MATERIAL = "claim_consistent_material"
_CLAIM_MULTI_JUDGE_COVERAGE = "claim_multi_judge_coverage"
_CLAIM_PANEL_RETAINED = "claim_panel_retained"
_CLAIM_COURT_IDENTIFICATION = "claim_court_identification"
_CLAIM_COURT_CARRIER_CAUSALITY = "claim_court_carrier_causality"

CLAIM_NAMES: tuple[str, ...] = (
    _CLAIM_CONSISTENT_MATERIAL,
    _CLAIM_MULTI_JUDGE_COVERAGE,
    _CLAIM_PANEL_RETAINED,
    _CLAIM_COURT_IDENTIFICATION,
    _CLAIM_COURT_CARRIER_CAUSALITY,
)


class CourtClaimState(str, Enum):
    """Per-claim court state."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class JudgeCourtState(str, Enum):
    """Overall court gate state."""

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class CourtClaim:
    """One computed claim in the judge court."""

    name: str
    state: CourtClaimState
    detail: str

    def as_json_dict(self) -> dict[str, object]:
        return {
            "claim": self.name,
            "state": self.state.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CourtArmReadout:
    """One arm aggregate from a retention report."""

    arm_label: str
    correct: int
    total: int
    accuracy: float
    ci_low_min: float
    ci_high_max: float

    def as_json_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm_label,
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 6),
            "ci_low_min": round(self.ci_low_min, 6),
            "ci_high_max": round(self.ci_high_max, 6),
        }


@dataclass(frozen=True)
class JudgePanel:
    """One retention report normalized for court comparison."""

    report_path: Path
    gate_state: str
    judge_model_id: str
    prefix_artifact_id: str
    substrate_fingerprint: str
    candidate_arm: str
    required_p2_pairs: tuple[str, ...]
    rollout_matrix: tuple[tuple[object, ...], ...]
    stochastic_generation_rollout_covered: bool
    candidate: CourtArmReadout
    control: CourtArmReadout

    def as_json_dict(self) -> dict[str, object]:
        return {
            "report_path": str(self.report_path),
            "gate_state": self.gate_state,
            "judge_model_id": self.judge_model_id,
            "prefix_artifact_id": self.prefix_artifact_id,
            "substrate_fingerprint": self.substrate_fingerprint,
            "candidate_arm": self.candidate_arm,
            "required_p2_pairs": list(self.required_p2_pairs),
            "rollout_matrix": [list(item) for item in self.rollout_matrix],
            "stochastic_generation_rollout_covered": (
                self.stochastic_generation_rollout_covered
            ),
            "candidate": self.candidate.as_json_dict(),
            "control": self.control.as_json_dict(),
        }


@dataclass(frozen=True)
class JudgeCourtReport:
    """The published multi-judge court verdict."""

    schema_version: str
    court_state: JudgeCourtState
    claims: tuple[CourtClaim, ...]
    panels: tuple[JudgePanel, ...]
    min_judges: int
    judge_model_ids: tuple[str, ...]
    notes: tuple[str, ...]

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "court_state": self.court_state.value,
            "claims": [claim.as_json_dict() for claim in self.claims],
            "panels": [panel.as_json_dict() for panel in self.panels],
            "min_judges": self.min_judges,
            "judge_model_ids": list(self.judge_model_ids),
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


def _bool_field(payload: Mapping[str, Any], key: str, *, path: Path) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{path} requires boolean field {key!r}")
    return value


def _number_field(payload: Mapping[str, Any], key: str, *, path: Path) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{path} requires numeric field {key!r}")
    return float(value)


def _int_field(payload: Mapping[str, Any], key: str, *, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} requires integer field {key!r}")
    return value


def _input_rollout_key(item: Mapping[str, Any], *, path: Path) -> tuple[object, ...]:
    return (
        _str_field(item, "lane", path=path),
        _str_field(item, "p2_pair", path=path),
        _int_field(item, "probe_limit", path=path),
        _int_field(item, "probe_count", path=path),
        _int_field(item, "case_count", path=path),
        _int_field(item, "max_new_tokens", path=path),
        round(_number_field(item, "temperature", path=path), 6),
        item.get("sampling_seed"),
        _bool_field(item, "stochastic_generation_rollout", path=path),
    )


def _arm_readout(
    aggregates: Sequence[object],
    *,
    arm_label: str,
    path: Path,
) -> CourtArmReadout:
    matches = [
        item
        for item in aggregates
        if isinstance(item, dict) and item.get("arm") == arm_label
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{path} requires exactly one aggregate readout for {arm_label!r}"
        )
    item = matches[0]
    return CourtArmReadout(
        arm_label=arm_label,
        correct=_int_field(item, "correct", path=path),
        total=_int_field(item, "total", path=path),
        accuracy=_number_field(item, "accuracy", path=path),
        ci_low_min=_number_field(item, "ci_low_min", path=path),
        ci_high_max=_number_field(item, "ci_high_max", path=path),
    )


def load_judge_panel(report_path: Path | str) -> JudgePanel:
    """Load and validate one retention report as a court panel."""

    resolved = Path(report_path).expanduser().resolve()
    payload = _read_json_object(resolved)
    if payload.get("schema_version") != RETENTION_GATE_SCHEMA_VERSION:
        raise ValueError(
            f"{resolved} has schema {payload.get('schema_version')!r}; "
            f"expected {RETENTION_GATE_SCHEMA_VERSION!r}"
        )
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError(f"{resolved} requires non-empty inputs list")
    inputs: list[Mapping[str, Any]] = []
    for item in raw_inputs:
        if not isinstance(item, dict):
            raise ValueError(f"{resolved} inputs must be objects")
        inputs.append(item)

    judges = {_str_field(item, "judge_model_id", path=resolved) for item in inputs}
    prefix_ids = {
        _str_field(item, "prefix_artifact_id", path=resolved) for item in inputs
    }
    substrates = {
        _str_field(item, "substrate_fingerprint", path=resolved) for item in inputs
    }
    candidates = {
        _str_field(item, "candidate_arm", path=resolved) for item in inputs
    }
    if len(judges) != 1:
        raise ValueError(f"{resolved} must contain exactly one judge, got {judges}")
    if len(prefix_ids) != 1:
        raise ValueError(
            f"{resolved} must contain exactly one prefix artifact, got {prefix_ids}"
        )
    if len(substrates) != 1:
        raise ValueError(
            f"{resolved} must contain exactly one substrate, got {substrates}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"{resolved} must contain exactly one candidate arm, got {candidates}"
        )
    raw_required = payload.get("required_p2_pairs")
    if not isinstance(raw_required, list) or not all(
        isinstance(item, str) and item for item in raw_required
    ):
        raise ValueError(f"{resolved} requires string required_p2_pairs list")
    raw_aggregates = payload.get("aggregates")
    if not isinstance(raw_aggregates, list):
        raise ValueError(f"{resolved} requires aggregates list")

    return JudgePanel(
        report_path=resolved,
        gate_state=_str_field(payload, "gate_state", path=resolved),
        judge_model_id=next(iter(judges)),
        prefix_artifact_id=next(iter(prefix_ids)),
        substrate_fingerprint=next(iter(substrates)),
        candidate_arm=next(iter(candidates)),
        required_p2_pairs=tuple(sorted(raw_required)),
        rollout_matrix=tuple(
            sorted(_input_rollout_key(item, path=resolved) for item in inputs)
        ),
        stochastic_generation_rollout_covered=_bool_field(
            payload, "stochastic_generation_rollout_covered", path=resolved
        ),
        candidate=_arm_readout(
            raw_aggregates, arm_label=PREFIX_ARM_LABEL, path=resolved
        ),
        control=_arm_readout(
            raw_aggregates, arm_label=CONTROL_ARM_LABEL, path=resolved
        ),
    )


def _state_for_claims(claims: Sequence[CourtClaim]) -> JudgeCourtState:
    if any(claim.state is CourtClaimState.FAIL for claim in claims):
        return JudgeCourtState.FAIL
    if any(claim.state is CourtClaimState.INSUFFICIENT_DATA for claim in claims):
        return JudgeCourtState.INSUFFICIENT_DATA
    return JudgeCourtState.PASS


def build_judge_court_report(
    *,
    panels: Sequence[JudgePanel],
    min_judges: int = 2,
) -> JudgeCourtReport:
    """Build a court verdict from per-judge retention panels."""

    if not panels:
        raise ValueError("judge court requires at least one retention panel")
    if min_judges < 2:
        raise ValueError("judge court requires min_judges >= 2")

    claims: list[CourtClaim] = []
    prefix_ids = {panel.prefix_artifact_id for panel in panels}
    substrates = {panel.substrate_fingerprint for panel in panels}
    candidates = {panel.candidate_arm for panel in panels}
    required_pairs = {panel.required_p2_pairs for panel in panels}
    rollout_matrices = {panel.rollout_matrix for panel in panels}
    stochastic_flags = {
        panel.stochastic_generation_rollout_covered for panel in panels
    }
    if (
        len(prefix_ids)
        == len(substrates)
        == len(candidates)
        == len(required_pairs)
        == len(rollout_matrices)
        == len(stochastic_flags)
        == 1
    ):
        claims.append(
            CourtClaim(
                name=_CLAIM_CONSISTENT_MATERIAL,
                state=CourtClaimState.PASS,
                detail=(
                    f"{len(panels)} panels share artifact "
                    f"{next(iter(prefix_ids))}, substrate "
                    f"{next(iter(substrates))}, candidate "
                    f"{next(iter(candidates))}, required pairs "
                    f"{next(iter(required_pairs))}, and rollout matrix"
                ),
            )
        )
    else:
        claims.append(
            CourtClaim(
                name=_CLAIM_CONSISTENT_MATERIAL,
                state=CourtClaimState.FAIL,
                detail=(
                    "panels do not share one artifact/substrate/candidate/"
                    "required-pair/rollout matrix: "
                    f"artifacts={sorted(prefix_ids)}, "
                    f"substrates={sorted(substrates)}, "
                    f"candidates={sorted(candidates)}, "
                    f"required_pairs={sorted(required_pairs)}, "
                    f"rollout_matrix_count={len(rollout_matrices)}, "
                    f"stochastic_flags={sorted(stochastic_flags)}"
                ),
            )
        )

    judge_ids = tuple(sorted({panel.judge_model_id for panel in panels}))
    if len(judge_ids) >= min_judges:
        claims.append(
            CourtClaim(
                name=_CLAIM_MULTI_JUDGE_COVERAGE,
                state=CourtClaimState.PASS,
                detail=(
                    f"{len(judge_ids)} distinct judges observed: "
                    f"{', '.join(judge_ids)}"
                ),
            )
        )
    else:
        claims.append(
            CourtClaim(
                name=_CLAIM_MULTI_JUDGE_COVERAGE,
                state=CourtClaimState.INSUFFICIENT_DATA,
                detail=(
                    f"need at least {min_judges} distinct judges, observed "
                    f"{len(judge_ids)}: {', '.join(judge_ids)}"
                ),
            )
        )

    failed_panels = [
        panel.report_path.name
        for panel in panels
        if panel.gate_state != "pass"
    ]
    if failed_panels:
        claims.append(
            CourtClaim(
                name=_CLAIM_PANEL_RETAINED,
                state=CourtClaimState.FAIL,
                detail=(
                    "all judge panels must pass their retention gate; failed: "
                    f"{', '.join(failed_panels)}"
                ),
            )
        )
    else:
        claims.append(
            CourtClaim(
                name=_CLAIM_PANEL_RETAINED,
                state=CourtClaimState.PASS,
                detail=f"{len(panels)} panels passed their retention gates",
            )
        )

    weak_identification = [
        f"{panel.judge_model_id}:{panel.candidate.ci_low_min:.3f}"
        for panel in panels
        if panel.candidate.ci_low_min <= 0.5
    ]
    if weak_identification:
        claims.append(
            CourtClaim(
                name=_CLAIM_COURT_IDENTIFICATION,
                state=CourtClaimState.FAIL,
                detail=(
                    "each judge must clear chance on the candidate arm; "
                    f"weak panels: {', '.join(weak_identification)}"
                ),
            )
        )
    else:
        floor = min(panel.candidate.ci_low_min for panel in panels)
        claims.append(
            CourtClaim(
                name=_CLAIM_COURT_IDENTIFICATION,
                state=CourtClaimState.PASS,
                detail=f"all candidate CI lows clear chance; court floor={floor:.3f}",
            )
        )

    causality_failures = [
        (
            f"{panel.judge_model_id}:control=({panel.control.ci_low_min:.3f},"
            f"{panel.control.ci_high_max:.3f})"
        )
        for panel in panels
        if not (
            panel.control.ci_low_min <= 0.5 <= panel.control.ci_high_max
            and panel.candidate.ci_low_min > 0.5
        )
    ]
    if causality_failures:
        claims.append(
            CourtClaim(
                name=_CLAIM_COURT_CARRIER_CAUSALITY,
                state=CourtClaimState.FAIL,
                detail=(
                    "each judge must keep the A-pure control at chance while "
                    f"the candidate clears it; failures: "
                    f"{', '.join(causality_failures)}"
                ),
            )
        )
    else:
        claims.append(
            CourtClaim(
                name=_CLAIM_COURT_CARRIER_CAUSALITY,
                state=CourtClaimState.PASS,
                detail="all panels keep A-pure at chance while G-prefix clears chance",
            )
        )

    state = _state_for_claims(claims)
    notes = (
        (
            "Court pass means the same retained State-KV result survived "
            "distinct judges on identical published material. It does not "
            "train or call any owner."
        )
        if state is JudgeCourtState.PASS
        else (
            "Court did not pass; keep claims scoped to the individual judge "
            "retention reports that passed."
        ),
    )
    return JudgeCourtReport(
        schema_version=JUDGE_COURT_SCHEMA_VERSION,
        court_state=state,
        claims=tuple(claims),
        panels=tuple(panels),
        min_judges=min_judges,
        judge_model_ids=judge_ids,
        notes=notes,
    )
