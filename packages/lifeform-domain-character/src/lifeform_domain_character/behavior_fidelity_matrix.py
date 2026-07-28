"""Frozen, read-only evaluation matrices for character behavior fidelity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from lifeform_domain_character.behavior_fidelity import (
    BehaviorFidelityReference,
    BehaviorFidelityStimulus,
)


BEHAVIOR_FIDELITY_MATRIX_SCHEMA_VERSION = (
    "character-behavior-fidelity-matrix.v1"
)
_RESOURCE_ROOT = Path(__file__).resolve().parent


class BehaviorFidelityCaseKind(str, Enum):
    POSITIVE = "positive"
    NEAR_NEGATIVE = "near_negative"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    COMPETING_BEHAVIOR = "competing_behavior"


class PromotionExpectation(str, Enum):
    REQUIRED = "required"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class BehaviorFidelityMatrixThresholds:
    required_case_counts: tuple[
        tuple[BehaviorFidelityCaseKind, int], ...
    ]
    minimum_positive_promotion_hits: int
    maximum_non_positive_promotion_hits: int
    minimum_case_fidelity_score: float
    minimum_positive_mean_baked_cold_delta: float
    require_source_digest_verified: bool
    require_no_feedback: bool
    require_competing_family_match: bool

    def __post_init__(self) -> None:
        count_map = dict(self.required_case_counts)
        if len(count_map) != len(BehaviorFidelityCaseKind):
            raise ValueError(
                "required_case_counts must cover every case kind exactly once"
            )
        if set(count_map) != set(BehaviorFidelityCaseKind):
            raise ValueError(
                "required_case_counts contains an unknown or missing case kind"
            )
        if any(count <= 0 for count in count_map.values()):
            raise ValueError("required case counts must be positive")
        positive_count = count_map[BehaviorFidelityCaseKind.POSITIVE]
        non_positive_count = sum(count_map.values()) - positive_count
        if not (
            0
            <= self.minimum_positive_promotion_hits
            <= positive_count
        ):
            raise ValueError(
                "minimum_positive_promotion_hits exceeds positive cases"
            )
        if not (
            0
            <= self.maximum_non_positive_promotion_hits
            <= non_positive_count
        ):
            raise ValueError(
                "maximum_non_positive_promotion_hits exceeds "
                "non-positive cases"
            )
        for name, value in (
            (
                "minimum_case_fidelity_score",
                self.minimum_case_fidelity_score,
            ),
            (
                "minimum_positive_mean_baked_cold_delta",
                self.minimum_positive_mean_baked_cold_delta,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class BehaviorFidelityMatrixCase:
    kind: BehaviorFidelityCaseKind
    promotion_expectation: PromotionExpectation
    expected_behavior_family: str
    stimulus: BehaviorFidelityStimulus
    reference: BehaviorFidelityReference
    reviewed_rationale: str

    def __post_init__(self) -> None:
        if self.stimulus.case_id != self.reference.case_id:
            raise ValueError("matrix case stimulus/reference case_id mismatch")
        if self.stimulus.scene_id != self.reference.scene_id:
            raise ValueError("matrix case stimulus/reference scene_id mismatch")
        if (
            self.stimulus.evidence_locator
            != self.reference.evidence_locator
        ):
            raise ValueError(
                "matrix case stimulus/reference evidence locator mismatch"
            )
        if not self.expected_behavior_family.strip():
            raise ValueError("expected_behavior_family must be non-empty")
        if not self.reviewed_rationale.strip():
            raise ValueError("reviewed_rationale must be non-empty")
        expected_promotion = (
            PromotionExpectation.REQUIRED
            if self.kind is BehaviorFidelityCaseKind.POSITIVE
            else PromotionExpectation.FORBIDDEN
        )
        if self.promotion_expectation is not expected_promotion:
            raise ValueError(
                f"{self.kind.value} case requires promotion expectation "
                f"{expected_promotion.value!r}"
            )


@dataclass(frozen=True)
class BehaviorFidelityMatrix:
    schema_version: str
    suite_id: str
    character_id: str
    target_schema_id: str
    source_chapter_ids: tuple[str, ...]
    reviewed_by: str
    description: str
    thresholds: BehaviorFidelityMatrixThresholds
    cases: tuple[BehaviorFidelityMatrixCase, ...]

    def __post_init__(self) -> None:
        if self.schema_version != BEHAVIOR_FIDELITY_MATRIX_SCHEMA_VERSION:
            raise ValueError(
                "behavior fidelity matrix schema mismatch: "
                f"{self.schema_version!r}"
            )
        for name, value in (
            ("suite_id", self.suite_id),
            ("character_id", self.character_id),
            ("target_schema_id", self.target_schema_id),
            ("reviewed_by", self.reviewed_by),
            ("description", self.description),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if (
            len(self.source_chapter_ids) < 2
            or any(not item.strip() for item in self.source_chapter_ids)
            or len(set(self.source_chapter_ids))
            != len(self.source_chapter_ids)
        ):
            raise ValueError(
                "source_chapter_ids must contain at least two unique values"
            )
        if not self.cases:
            raise ValueError("behavior fidelity matrix must contain cases")
        case_ids = tuple(case.stimulus.case_id for case in self.cases)
        scene_ids = tuple(case.stimulus.scene_id for case in self.cases)
        evidence_locators = tuple(
            case.stimulus.evidence_locator for case in self.cases
        )
        for name, values in (
            ("case_id", case_ids),
            ("scene_id", scene_ids),
            ("evidence_locator", evidence_locators),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"matrix {name} values must be unique")
        actual_counts = {
            kind: sum(case.kind is kind for case in self.cases)
            for kind in BehaviorFidelityCaseKind
        }
        if actual_counts != dict(self.thresholds.required_case_counts):
            raise ValueError(
                "matrix case counts do not match frozen thresholds"
            )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def load_behavior_fidelity_matrix(
    path: str | Path,
) -> BehaviorFidelityMatrix:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("behavior fidelity matrix root must be an object")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "suite_id",
            "character_id",
            "target_schema_id",
            "source_chapter_ids",
            "reviewed_by",
            "description",
            "thresholds",
            "cases",
        },
        context="matrix",
    )
    thresholds_raw = _require_dict(raw["thresholds"], "thresholds")
    _require_exact_keys(
        thresholds_raw,
        {
            "required_case_counts",
            "minimum_positive_promotion_hits",
            "maximum_non_positive_promotion_hits",
            "minimum_case_fidelity_score",
            "minimum_positive_mean_baked_cold_delta",
            "require_source_digest_verified",
            "require_no_feedback",
            "require_competing_family_match",
        },
        context="thresholds",
    )
    counts_raw = _require_dict(
        thresholds_raw["required_case_counts"],
        "required_case_counts",
    )
    _require_exact_keys(
        counts_raw,
        {kind.value for kind in BehaviorFidelityCaseKind},
        context="required_case_counts",
    )
    thresholds = BehaviorFidelityMatrixThresholds(
        required_case_counts=tuple(
            (
                kind,
                _require_int(counts_raw[kind.value], kind.value),
            )
            for kind in BehaviorFidelityCaseKind
        ),
        minimum_positive_promotion_hits=_require_int(
            thresholds_raw["minimum_positive_promotion_hits"],
            "minimum_positive_promotion_hits",
        ),
        maximum_non_positive_promotion_hits=_require_int(
            thresholds_raw["maximum_non_positive_promotion_hits"],
            "maximum_non_positive_promotion_hits",
        ),
        minimum_case_fidelity_score=_require_float(
            thresholds_raw["minimum_case_fidelity_score"],
            "minimum_case_fidelity_score",
        ),
        minimum_positive_mean_baked_cold_delta=_require_float(
            thresholds_raw["minimum_positive_mean_baked_cold_delta"],
            "minimum_positive_mean_baked_cold_delta",
        ),
        require_source_digest_verified=_require_bool(
            thresholds_raw["require_source_digest_verified"],
            "require_source_digest_verified",
        ),
        require_no_feedback=_require_bool(
            thresholds_raw["require_no_feedback"],
            "require_no_feedback",
        ),
        require_competing_family_match=_require_bool(
            thresholds_raw["require_competing_family_match"],
            "require_competing_family_match",
        ),
    )
    cases_raw = raw["cases"]
    if not isinstance(cases_raw, list):
        raise TypeError("cases must be an array")
    return BehaviorFidelityMatrix(
        schema_version=_require_str(
            raw["schema_version"],
            "schema_version",
        ),
        suite_id=_require_str(raw["suite_id"], "suite_id"),
        character_id=_require_str(raw["character_id"], "character_id"),
        target_schema_id=_require_str(
            raw["target_schema_id"],
            "target_schema_id",
        ),
        source_chapter_ids=_require_str_tuple(
            raw["source_chapter_ids"],
            "source_chapter_ids",
        ),
        reviewed_by=_require_str(raw["reviewed_by"], "reviewed_by"),
        description=_require_str(raw["description"], "description"),
        thresholds=thresholds,
        cases=tuple(_parse_case(item) for item in cases_raw),
    )


def load_zhang_wuji_action_applicability_matrix(
) -> BehaviorFidelityMatrix:
    return load_behavior_fidelity_matrix(
        _RESOURCE_ROOT
        / "evaluation"
        / "zhang_wuji_action_applicability_v1.json"
    )


def _parse_case(raw: object) -> BehaviorFidelityMatrixCase:
    case_raw = _require_dict(raw, "case")
    _require_exact_keys(
        case_raw,
        {
            "kind",
            "promotion_expectation",
            "expected_behavior_family",
            "stimulus",
            "reference",
            "reviewed_rationale",
        },
        context="case",
    )
    stimulus_raw = _require_dict(case_raw["stimulus"], "stimulus")
    reference_raw = _require_dict(case_raw["reference"], "reference")
    _require_exact_keys(
        stimulus_raw,
        {
            "case_id",
            "character_id",
            "scene_id",
            "phase_label",
            "setting",
            "decision_point",
            "evidence_locator",
        },
        context="stimulus",
    )
    _require_exact_keys(
        reference_raw,
        {
            "case_id",
            "scene_id",
            "canonical_action",
            "canonical_outcome",
            "evidence_locator",
            "reviewed_by",
        },
        context="reference",
    )
    return BehaviorFidelityMatrixCase(
        kind=BehaviorFidelityCaseKind(
            _require_str(case_raw["kind"], "case.kind")
        ),
        promotion_expectation=PromotionExpectation(
            _require_str(
                case_raw["promotion_expectation"],
                "case.promotion_expectation",
            )
        ),
        expected_behavior_family=_require_str(
            case_raw["expected_behavior_family"],
            "case.expected_behavior_family",
        ),
        stimulus=BehaviorFidelityStimulus(
            case_id=_require_str(
                stimulus_raw["case_id"],
                "stimulus.case_id",
            ),
            character_id=_require_str(
                stimulus_raw["character_id"],
                "stimulus.character_id",
            ),
            scene_id=_require_str(
                stimulus_raw["scene_id"],
                "stimulus.scene_id",
            ),
            phase_label=_require_str(
                stimulus_raw["phase_label"],
                "stimulus.phase_label",
            ),
            setting=_require_str(
                stimulus_raw["setting"],
                "stimulus.setting",
            ),
            decision_point=_require_str(
                stimulus_raw["decision_point"],
                "stimulus.decision_point",
            ),
            evidence_locator=_require_str(
                stimulus_raw["evidence_locator"],
                "stimulus.evidence_locator",
            ),
        ),
        reference=BehaviorFidelityReference(
            case_id=_require_str(
                reference_raw["case_id"],
                "reference.case_id",
            ),
            scene_id=_require_str(
                reference_raw["scene_id"],
                "reference.scene_id",
            ),
            canonical_action=_require_str(
                reference_raw["canonical_action"],
                "reference.canonical_action",
            ),
            canonical_outcome=_require_str(
                reference_raw["canonical_outcome"],
                "reference.canonical_outcome",
            ),
            evidence_locator=_require_str(
                reference_raw["evidence_locator"],
                "reference.evidence_locator",
            ),
            reviewed_by=_require_str(
                reference_raw["reviewed_by"],
                "reference.reviewed_by",
            ),
        ),
        reviewed_rationale=_require_str(
            case_raw["reviewed_rationale"],
            "case.reviewed_rationale",
        ),
    )


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{context} keys mismatch: expected={sorted(expected)!r}, "
            f"actual={sorted(payload)!r}"
        )


def _require_dict(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def _require_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _require_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    return float(value)


def _require_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a boolean")
    return value


def _require_str(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _require_str_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return tuple(
        _require_str(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )


__all__ = [
    "BEHAVIOR_FIDELITY_MATRIX_SCHEMA_VERSION",
    "BehaviorFidelityCaseKind",
    "BehaviorFidelityMatrix",
    "BehaviorFidelityMatrixCase",
    "BehaviorFidelityMatrixThresholds",
    "PromotionExpectation",
    "load_behavior_fidelity_matrix",
    "load_zhang_wuji_action_applicability_matrix",
]
