"""Read-only, reviewed behavioral-fidelity evaluation for character scenes.

The evaluator is deliberately separate from chapter bake:

* capture sees only a reviewed setting and decision point;
* canonical action/outcome are held in a separate reference object;
* semantic judgement is a reviewed artifact, never keyword routing;
* no outcome, reward, credit, or scene-end writeback is submitted;
* the evaluated lifeform is a disposable sandbox.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from lifeform_core import Lifeform, TurnTriggerKind

from lifeform_domain_character.narrative import NarrativeScene


BEHAVIOR_FIDELITY_SCHEMA_VERSION = "character-behavior-fidelity.v2"
_BEHAVIOR_FIDELITY_SCHEMA_V1 = "character-behavior-fidelity.v1"
BEHAVIOR_FIDELITY_DIMENSIONS: tuple[str, ...] = (
    "action_choice_alignment",
    "protective_intent_alignment",
    "risk_posture_alignment",
    "situation_model_alignment",
    "character_motivation_alignment",
)


class BehaviorFidelityEvidenceSource(str, Enum):
    SYSTEM_SELF_EVAL = "system_self_eval"
    LLM_JUDGE = "llm_judge"
    EXTERNAL_VALIDATED = "external_validated"


@dataclass(frozen=True)
class BehaviorFidelityStimulus:
    case_id: str
    character_id: str
    scene_id: str
    phase_label: str
    setting: str
    decision_point: str
    evidence_locator: str

    def __post_init__(self) -> None:
        _require_non_empty(
            case_id=self.case_id,
            character_id=self.character_id,
            scene_id=self.scene_id,
            phase_label=self.phase_label,
            setting=self.setting,
            decision_point=self.decision_point,
            evidence_locator=self.evidence_locator,
        )

    @property
    def setting_prompt(self) -> str:
        return (
            f"你正置身于这个局势：{self.setting}。"
            "只确认你此刻看到的局势，不要假设后来会发生什么。"
        )

    @property
    def decision_prompt(self) -> str:
        return (
            f"此刻你必须决定：{self.decision_point}。"
            "请用第一人称说出你现在会采取的具体行动和直接理由；"
            "不要讨论评估，也不要假设尚未发生的结果。"
        )

    @property
    def digest(self) -> str:
        return _canonical_sha256(
            {
                "case_id": self.case_id,
                "character_id": self.character_id,
                "scene_id": self.scene_id,
                "phase_label": self.phase_label,
                "setting_prompt": self.setting_prompt,
                "decision_prompt": self.decision_prompt,
                "evidence_locator": self.evidence_locator,
            }
        )


@dataclass(frozen=True)
class BehaviorFidelityReference:
    case_id: str
    scene_id: str
    canonical_action: str
    canonical_outcome: str
    evidence_locator: str
    reviewed_by: str

    def __post_init__(self) -> None:
        _require_non_empty(
            case_id=self.case_id,
            scene_id=self.scene_id,
            canonical_action=self.canonical_action,
            canonical_outcome=self.canonical_outcome,
            evidence_locator=self.evidence_locator,
            reviewed_by=self.reviewed_by,
        )

    @property
    def digest(self) -> str:
        return _canonical_sha256(
            {
                "case_id": self.case_id,
                "scene_id": self.scene_id,
                "canonical_action": self.canonical_action,
                "canonical_outcome": self.canonical_outcome,
                "evidence_locator": self.evidence_locator,
                "reviewed_by": self.reviewed_by,
            }
        )


@dataclass(frozen=True)
class BehaviorFidelityCapture:
    schema_version: str
    case_id: str
    scene_id: str
    arm_id: str
    stimulus_digest: str
    source_state_sha256_before: str
    source_state_sha256_after: str
    source_state_unchanged: bool
    source_state_digest_verified: bool
    candidate_response: str
    candidate_response_sha256: str
    active_regime: str
    active_abstract_action: str
    world_z_t: tuple[float, ...]
    self_z_t: tuple[float, ...]
    action_grounding_source_case_id: str | None
    action_grounding_action_labels: tuple[str, ...]
    sandbox_learning_fingerprint_before: str
    sandbox_learning_fingerprint_after: str
    outcome_feedback_submitted: bool
    evaluation_feedback_submitted: bool
    sandbox_discarded: bool

    def __post_init__(self) -> None:
        if self.schema_version != BEHAVIOR_FIDELITY_SCHEMA_VERSION:
            raise ValueError(
                "behavior fidelity capture schema mismatch: "
                f"{self.schema_version!r}"
            )
        _require_non_empty(
            case_id=self.case_id,
            scene_id=self.scene_id,
            arm_id=self.arm_id,
            stimulus_digest=self.stimulus_digest,
            source_state_sha256_before=self.source_state_sha256_before,
            source_state_sha256_after=self.source_state_sha256_after,
            candidate_response=self.candidate_response,
            candidate_response_sha256=self.candidate_response_sha256,
            sandbox_learning_fingerprint_before=(
                self.sandbox_learning_fingerprint_before
            ),
            sandbox_learning_fingerprint_after=(
                self.sandbox_learning_fingerprint_after
            ),
        )
        if self.candidate_response_sha256 != _text_sha256(
            self.candidate_response
        ):
            raise ValueError("candidate response digest mismatch")
        if self.source_state_unchanged != (
            self.source_state_sha256_before
            == self.source_state_sha256_after
        ):
            raise ValueError("source_state_unchanged disagrees with digests")
        if self.outcome_feedback_submitted:
            raise ValueError("behavior evaluation must not submit outcome feedback")
        if self.evaluation_feedback_submitted:
            raise ValueError(
                "behavior evaluation must not submit evaluation feedback"
            )
        if not self.sandbox_discarded:
            raise ValueError("behavior evaluation sandbox must be disposable")
        if self.action_grounding_source_case_id is None:
            if self.action_grounding_action_labels:
                raise ValueError(
                    "action grounding labels require a source case id"
                )
        elif (
            not self.action_grounding_source_case_id.strip()
            or not self.action_grounding_action_labels
            or any(
                not label.strip()
                for label in self.action_grounding_action_labels
            )
        ):
            raise ValueError(
                "action grounding lineage requires a non-empty source case "
                "id and action labels"
            )


@dataclass(frozen=True)
class ReviewedBehaviorFidelityAssessment:
    schema_version: str
    case_id: str
    arm_id: str
    stimulus_digest: str
    reference_digest: str
    candidate_response_sha256: str
    reviewer: str
    evidence_source: BehaviorFidelityEvidenceSource
    dimension_scores: tuple[tuple[str, float], ...]
    dimension_rationales: tuple[tuple[str, str], ...]
    future_knowledge_leakage: bool
    future_knowledge_rationale: str

    def __post_init__(self) -> None:
        if self.schema_version != BEHAVIOR_FIDELITY_SCHEMA_VERSION:
            raise ValueError(
                "behavior fidelity assessment schema mismatch: "
                f"{self.schema_version!r}"
            )
        _require_non_empty(
            case_id=self.case_id,
            arm_id=self.arm_id,
            stimulus_digest=self.stimulus_digest,
            reference_digest=self.reference_digest,
            candidate_response_sha256=self.candidate_response_sha256,
            reviewer=self.reviewer,
            future_knowledge_rationale=self.future_knowledge_rationale,
        )
        scores = dict(self.dimension_scores)
        rationales = dict(self.dimension_rationales)
        if tuple(sorted(scores)) != tuple(
            sorted(BEHAVIOR_FIDELITY_DIMENSIONS)
        ):
            raise ValueError(
                "assessment must score exactly the behavior fidelity "
                f"dimensions {BEHAVIOR_FIDELITY_DIMENSIONS!r}"
            )
        if tuple(sorted(rationales)) != tuple(sorted(scores)):
            raise ValueError(
                "assessment rationales must cover exactly the scored dimensions"
            )
        for dimension, score in self.dimension_scores:
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(
                    f"score for {dimension!r} must be in [0, 1]"
                )
            if not rationales[dimension].strip():
                raise ValueError(
                    f"rationale for {dimension!r} must be non-empty"
                )


@dataclass(frozen=True)
class BehaviorFidelityReport:
    schema_version: str
    case_id: str
    scene_id: str
    arm_id: str
    evidence_source: str
    reviewer: str
    candidate_response: str
    reference_action: str
    dimension_scores: tuple[tuple[str, float], ...]
    overall_score: float
    proof_gates: tuple[tuple[str, str], ...]
    behavior_fidelity_passed: bool
    claim_status: str
    verification_failures: tuple[str, ...]


@dataclass(frozen=True)
class BehaviorFidelityComparisonReport:
    schema_version: str
    case_id: str
    baked_arm_id: str
    cold_arm_id: str
    baked_score: float
    cold_score: float
    baked_minus_cold: float
    baked_passed: bool
    cold_passed: bool
    profile_answer_holdout_passed: bool
    learned_behavior_advantage: bool
    claim_status: str
    description: str


def build_scene_behavior_fidelity_inputs(
    *,
    character_id: str,
    scene: NarrativeScene,
    reviewed_by: str,
) -> tuple[BehaviorFidelityStimulus, BehaviorFidelityReference]:
    """Split one reviewed scene into oracle-free stimulus and reference."""

    case_id = f"behavior-fidelity:{character_id}:{scene.scene_id}"
    return (
        BehaviorFidelityStimulus(
            case_id=case_id,
            character_id=character_id,
            scene_id=scene.scene_id,
            phase_label=scene.phase_label,
            setting=scene.setting,
            decision_point=scene.decision_point,
            evidence_locator=scene.evidence_locator,
        ),
        BehaviorFidelityReference(
            case_id=case_id,
            scene_id=scene.scene_id,
            canonical_action=scene.canonical_action,
            canonical_outcome=scene.canonical_outcome,
            evidence_locator=scene.evidence_locator,
            reviewed_by=reviewed_by,
        ),
    )


async def capture_behavior_fidelity_async(
    *,
    stimulus: BehaviorFidelityStimulus,
    lifeform: Lifeform,
    arm_id: str,
    source_state_sha256_before: str,
    source_state_sha256_after: str,
    session_id: str | None = None,
    source_state_digest_reader: Callable[[], str] | None = None,
) -> BehaviorFidelityCapture:
    """Capture one autonomous decision in a disposable lifeform sandbox."""

    _require_non_empty(
        arm_id=arm_id,
        source_state_sha256_before=source_state_sha256_before,
        source_state_sha256_after=source_state_sha256_after,
    )
    measured_source_before = (
        source_state_digest_reader()
        if source_state_digest_reader is not None
        else source_state_sha256_before
    )
    if measured_source_before != source_state_sha256_before:
        raise ValueError(
            "measured source-state digest disagrees with declared before "
            "digest"
        )
    session = lifeform.create_session(
        session_id=session_id
        or f"behavior-fidelity:{arm_id}:{stimulus.scene_id}"
    )
    before = session.brain_session.runner.export_learning_checkpoint(
        checkpoint_id=f"{stimulus.case_id}:{arm_id}:before",
        include_runtime_replay=False,
    )
    await session.run_turn(
        stimulus.setting_prompt,
        trigger_kind=TurnTriggerKind.APPRENTICE,
    )
    decision = await session.run_turn(
        stimulus.decision_prompt,
        trigger_kind=TurnTriggerKind.USER_INPUT,
    )
    after = session.brain_session.runner.export_learning_checkpoint(
        checkpoint_id=f"{stimulus.case_id}:{arm_id}:after",
        include_runtime_replay=False,
    )
    z_codes = dict(decision.track_z_t_codes)
    action_grounding = decision.active_snapshots[
        "case_memory"
    ].value.action_grounding
    measured_source_after = (
        source_state_digest_reader()
        if source_state_digest_reader is not None
        else source_state_sha256_after
    )
    if measured_source_after != source_state_sha256_after:
        raise ValueError(
            "measured source-state digest disagrees with declared after "
            "digest"
        )
    response = decision.response.text.strip()
    return BehaviorFidelityCapture(
        schema_version=BEHAVIOR_FIDELITY_SCHEMA_VERSION,
        case_id=stimulus.case_id,
        scene_id=stimulus.scene_id,
        arm_id=arm_id,
        stimulus_digest=stimulus.digest,
        source_state_sha256_before=measured_source_before,
        source_state_sha256_after=measured_source_after,
        source_state_unchanged=(
            measured_source_before == measured_source_after
        ),
        source_state_digest_verified=(
            source_state_digest_reader is not None
        ),
        candidate_response=response,
        candidate_response_sha256=_text_sha256(response),
        active_regime=decision.active_regime or "",
        active_abstract_action=decision.active_abstract_action or "",
        world_z_t=tuple(z_codes.get("world", ())),
        self_z_t=tuple(z_codes.get("self", ())),
        action_grounding_source_case_id=(
            action_grounding.source_case_id
            if action_grounding is not None
            else None
        ),
        action_grounding_action_labels=(
            action_grounding.action_labels
            if action_grounding is not None
            else ()
        ),
        sandbox_learning_fingerprint_before=before.fingerprint,
        sandbox_learning_fingerprint_after=after.fingerprint,
        outcome_feedback_submitted=False,
        evaluation_feedback_submitted=False,
        sandbox_discarded=True,
    )


def review_behavior_fidelity(
    *,
    capture: BehaviorFidelityCapture,
    reference: BehaviorFidelityReference,
    assessment: ReviewedBehaviorFidelityAssessment,
    minimum_overall_score: float = 0.7,
    minimum_action_choice_score: float = 0.65,
    minimum_situation_model_score: float = 0.6,
) -> BehaviorFidelityReport:
    """Validate a reviewed semantic assessment and publish a read-only report."""

    for name, value in (
        ("minimum_overall_score", minimum_overall_score),
        ("minimum_action_choice_score", minimum_action_choice_score),
        ("minimum_situation_model_score", minimum_situation_model_score),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    mismatches = tuple(
        name
        for name, actual, expected in (
            ("case_id", assessment.case_id, capture.case_id),
            ("arm_id", assessment.arm_id, capture.arm_id),
            (
                "stimulus_digest",
                assessment.stimulus_digest,
                capture.stimulus_digest,
            ),
            (
                "candidate_response_sha256",
                assessment.candidate_response_sha256,
                capture.candidate_response_sha256,
            ),
            ("reference_digest", assessment.reference_digest, reference.digest),
            ("reference_case_id", reference.case_id, capture.case_id),
            ("reference_scene_id", reference.scene_id, capture.scene_id),
        )
        if actual != expected
    )
    if mismatches:
        raise ValueError(
            "behavior fidelity assessment binding mismatch: "
            + ", ".join(mismatches)
        )
    scores = dict(assessment.dimension_scores)
    overall_score = round(
        sum(scores.values()) / len(BEHAVIOR_FIDELITY_DIMENSIONS),
        6,
    )
    proof_gates = (
        (
            "source_state_unchanged",
            (
                "pass"
                if capture.source_state_unchanged
                and capture.source_state_digest_verified
                else "fail"
            ),
        ),
        (
            "no_outcome_or_evaluation_feedback",
            (
                "pass"
                if not capture.outcome_feedback_submitted
                and not capture.evaluation_feedback_submitted
                else "fail"
            ),
        ),
        (
            "disposable_sandbox",
            "pass" if capture.sandbox_discarded else "fail",
        ),
        (
            "no_future_knowledge_leakage",
            "pass" if not assessment.future_knowledge_leakage else "fail",
        ),
        (
            "concrete_action_alignment",
            (
                "pass"
                if scores["action_choice_alignment"]
                >= minimum_action_choice_score
                else "fail"
            ),
        ),
        (
            "situation_model_alignment",
            (
                "pass"
                if scores["situation_model_alignment"]
                >= minimum_situation_model_score
                else "fail"
            ),
        ),
        (
            "overall_behavior_fidelity",
            "pass" if overall_score >= minimum_overall_score else "fail",
        ),
    )
    failures = tuple(
        gate for gate, status in proof_gates if status != "pass"
    )
    passed = not failures
    evidence_source = assessment.evidence_source.value
    claim_status = (
        "external-validated-pass"
        if passed
        and assessment.evidence_source
        is BehaviorFidelityEvidenceSource.EXTERNAL_VALIDATED
        else (
            f"{evidence_source}-diagnostic-pass"
            if passed
            else f"{evidence_source}-diagnostic-fail"
        )
    )
    return BehaviorFidelityReport(
        schema_version=BEHAVIOR_FIDELITY_SCHEMA_VERSION,
        case_id=capture.case_id,
        scene_id=capture.scene_id,
        arm_id=capture.arm_id,
        evidence_source=evidence_source,
        reviewer=assessment.reviewer,
        candidate_response=capture.candidate_response,
        reference_action=reference.canonical_action,
        dimension_scores=assessment.dimension_scores,
        overall_score=overall_score,
        proof_gates=proof_gates,
        behavior_fidelity_passed=passed,
        claim_status=claim_status,
        verification_failures=failures,
    )


def compare_behavior_fidelity_reports(
    *,
    baked: BehaviorFidelityReport,
    cold: BehaviorFidelityReport,
    minimum_advantage: float = 0.05,
    profile_answer_holdout_passed: bool = False,
) -> BehaviorFidelityComparisonReport:
    """Compare a baked arm with a matched cold control."""

    if baked.case_id != cold.case_id:
        raise ValueError("behavior fidelity comparison requires one case_id")
    if baked.arm_id == cold.arm_id:
        raise ValueError("behavior fidelity comparison arm ids must differ")
    if not 0.0 <= minimum_advantage <= 1.0:
        raise ValueError("minimum_advantage must be in [0, 1]")
    delta = round(baked.overall_score - cold.overall_score, 6)
    advantage = (
        baked.behavior_fidelity_passed
        and delta >= minimum_advantage
        and profile_answer_holdout_passed
    )
    return BehaviorFidelityComparisonReport(
        schema_version=BEHAVIOR_FIDELITY_SCHEMA_VERSION,
        case_id=baked.case_id,
        baked_arm_id=baked.arm_id,
        cold_arm_id=cold.arm_id,
        baked_score=baked.overall_score,
        cold_score=cold.overall_score,
        baked_minus_cold=delta,
        baked_passed=baked.behavior_fidelity_passed,
        cold_passed=cold.behavior_fidelity_passed,
        profile_answer_holdout_passed=profile_answer_holdout_passed,
        learned_behavior_advantage=advantage,
        claim_status=(
            "diagnostic-pass" if advantage else "diagnostic-fail"
        ),
        description=(
            "Read-only matched-control comparison; reviewed scores never "
            "enter PE, credit, regime, memory, or Internal RL. Learned "
            "advantage additionally requires a profile-answer holdout."
        ),
    )


def behavior_fidelity_capture_from_dict(
    payload: dict[str, Any],
) -> BehaviorFidelityCapture:
    source_schema_version = str(payload["schema_version"])
    is_v1 = source_schema_version == _BEHAVIOR_FIDELITY_SCHEMA_V1
    if is_v1:
        action_grounding_source_case_id = None
        action_grounding_action_labels: tuple[str, ...] = ()
    else:
        raw_source_case_id = payload["action_grounding_source_case_id"]
        action_grounding_source_case_id = (
            str(raw_source_case_id)
            if raw_source_case_id is not None
            else None
        )
        action_grounding_action_labels = tuple(
            str(value)
            for value in payload["action_grounding_action_labels"]
        )
    return BehaviorFidelityCapture(
        schema_version=(
            BEHAVIOR_FIDELITY_SCHEMA_VERSION
            if is_v1
            else source_schema_version
        ),
        case_id=str(payload["case_id"]),
        scene_id=str(payload["scene_id"]),
        arm_id=str(payload["arm_id"]),
        stimulus_digest=str(payload["stimulus_digest"]),
        source_state_sha256_before=str(
            payload["source_state_sha256_before"]
        ),
        source_state_sha256_after=str(
            payload["source_state_sha256_after"]
        ),
        source_state_unchanged=bool(payload["source_state_unchanged"]),
        source_state_digest_verified=(
            False
            if is_v1
            else bool(payload["source_state_digest_verified"])
        ),
        candidate_response=str(payload["candidate_response"]),
        candidate_response_sha256=str(payload["candidate_response_sha256"]),
        active_regime=str(payload["active_regime"]),
        active_abstract_action=str(payload["active_abstract_action"]),
        world_z_t=tuple(float(value) for value in payload["world_z_t"]),
        self_z_t=tuple(float(value) for value in payload["self_z_t"]),
        action_grounding_source_case_id=action_grounding_source_case_id,
        action_grounding_action_labels=action_grounding_action_labels,
        sandbox_learning_fingerprint_before=str(
            payload["sandbox_learning_fingerprint_before"]
        ),
        sandbox_learning_fingerprint_after=str(
            payload["sandbox_learning_fingerprint_after"]
        ),
        outcome_feedback_submitted=bool(
            payload["outcome_feedback_submitted"]
        ),
        evaluation_feedback_submitted=bool(
            payload["evaluation_feedback_submitted"]
        ),
        sandbox_discarded=bool(payload["sandbox_discarded"]),
    )


def reviewed_behavior_fidelity_assessment_from_dict(
    payload: dict[str, Any],
) -> ReviewedBehaviorFidelityAssessment:
    source_schema_version = str(payload["schema_version"])
    return ReviewedBehaviorFidelityAssessment(
        schema_version=(
            BEHAVIOR_FIDELITY_SCHEMA_VERSION
            if source_schema_version == _BEHAVIOR_FIDELITY_SCHEMA_V1
            else source_schema_version
        ),
        case_id=str(payload["case_id"]),
        arm_id=str(payload["arm_id"]),
        stimulus_digest=str(payload["stimulus_digest"]),
        reference_digest=str(payload["reference_digest"]),
        candidate_response_sha256=str(
            payload["candidate_response_sha256"]
        ),
        reviewer=str(payload["reviewer"]),
        evidence_source=BehaviorFidelityEvidenceSource(
            str(payload["evidence_source"])
        ),
        dimension_scores=tuple(
            (str(name), float(score))
            for name, score in payload["dimension_scores"]
        ),
        dimension_rationales=tuple(
            (str(name), str(rationale))
            for name, rationale in payload["dimension_rationales"]
        ),
        future_knowledge_leakage=bool(
            payload["future_knowledge_leakage"]
        ),
        future_knowledge_rationale=str(
            payload["future_knowledge_rationale"]
        ),
    )


def behavior_fidelity_report_from_dict(
    payload: dict[str, Any],
) -> BehaviorFidelityReport:
    source_schema_version = str(payload["schema_version"])
    return BehaviorFidelityReport(
        schema_version=(
            BEHAVIOR_FIDELITY_SCHEMA_VERSION
            if source_schema_version == _BEHAVIOR_FIDELITY_SCHEMA_V1
            else source_schema_version
        ),
        case_id=str(payload["case_id"]),
        scene_id=str(payload["scene_id"]),
        arm_id=str(payload["arm_id"]),
        evidence_source=str(payload["evidence_source"]),
        reviewer=str(payload["reviewer"]),
        candidate_response=str(payload["candidate_response"]),
        reference_action=str(payload["reference_action"]),
        dimension_scores=tuple(
            (str(name), float(score))
            for name, score in payload["dimension_scores"]
        ),
        overall_score=float(payload["overall_score"]),
        proof_gates=tuple(
            (str(name), str(status))
            for name, status in payload["proof_gates"]
        ),
        behavior_fidelity_passed=bool(
            payload["behavior_fidelity_passed"]
        ),
        claim_status=str(payload["claim_status"]),
        verification_failures=tuple(
            str(failure) for failure in payload["verification_failures"]
        ),
    )


def _require_non_empty(**values: str) -> None:
    for name, value in values.items():
        if not str(value).strip():
            raise ValueError(f"{name} must be non-empty")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "BEHAVIOR_FIDELITY_DIMENSIONS",
    "BEHAVIOR_FIDELITY_SCHEMA_VERSION",
    "BehaviorFidelityCapture",
    "BehaviorFidelityComparisonReport",
    "BehaviorFidelityEvidenceSource",
    "BehaviorFidelityReference",
    "BehaviorFidelityReport",
    "BehaviorFidelityStimulus",
    "ReviewedBehaviorFidelityAssessment",
    "behavior_fidelity_capture_from_dict",
    "behavior_fidelity_report_from_dict",
    "build_scene_behavior_fidelity_inputs",
    "capture_behavior_fidelity_async",
    "compare_behavior_fidelity_reports",
    "review_behavior_fidelity",
    "reviewed_behavior_fidelity_assessment_from_dict",
]
