from __future__ import annotations

import pytest

from volvence_zero.agent.session_observation import (
    _require_cognition_task_readouts,
)
from volvence_zero.application.action_abstraction import (
    NoOpActionApplicabilityEvaluator,
)
from volvence_zero.application.modules import case_memory as case_memory_module
from volvence_zero.application.storage import CaseMemoryRecord
from volvence_zero.cognition_task import (
    CognitionRequiredReadout,
    CognitionTaskContract,
    CognitionTaskKind,
    MissingRequiredCognitionReadoutError,
)


def _reviewed_action_case() -> CaseMemoryRecord:
    return CaseMemoryRecord(
        case_id="case:gate-water-mark",
        domain="parallel-world",
        problem_pattern="门口出现异常水痕，同行者仍在危险位置",
        user_state_pattern="角色必须依据现场观察采取行动",
        risk_markers=("immediate-risk",),
        track_tags=("world",),
        regime_tags=(),
        intervention_ordering=("挡在门前", "示意阿兰后退"),
        outcome_label="stable",
        delayed_signal_count=1,
        escalation_observed=False,
        repair_observed=False,
        confidence=0.9,
        relevance_score=0.9,
        description="reviewed observable action case",
    )


def test_choose_observable_action_requires_owner_readout() -> None:
    contract = CognitionTaskContract(
        kind=CognitionTaskKind.CHOOSE_OBSERVABLE_ACTION
    )

    assert contract.required_readouts == (
        CognitionRequiredReadout.RESPONSE_ACTION_REALIZATION,
    )
    with pytest.raises(
        MissingRequiredCognitionReadoutError,
        match="before expression",
    ):
        _require_cognition_task_readouts(
            cognition_task_contract=contract,
            response_assembly=None,
        )


def test_choose_observable_action_uses_perceived_input_as_case_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perceived_event = "门外的水痕正在延伸，阿兰站在没有遮挡的位置。"

    def similarity(left: str, right: str) -> float:
        if right in case_memory_module._ACTION_REQUEST_PROTOTYPES:
            return 0.0
        if right in case_memory_module._REFLECTIVE_OPINION_PROTOTYPES:
            return 1.0
        assert left == perceived_event
        assert "水痕" in right
        return 0.95

    monkeypatch.setattr(
        case_memory_module,
        "semantic_topic_similarity",
        similarity,
    )
    monkeypatch.setattr(
        case_memory_module,
        "semantic_embedding_backend_status",
        lambda: ("stub", "deterministic-test", False),
    )

    legacy = case_memory_module._select_action_grounding(
        records=(_reviewed_action_case(),),
        entries=(),
        user_input=perceived_event,
        cognition_task_contract=None,
        abstract_action="discovered_family_0",
        action_applicability_evaluator=NoOpActionApplicabilityEvaluator(),
    )
    typed = case_memory_module._select_action_grounding(
        records=(_reviewed_action_case(),),
        entries=(),
        user_input=perceived_event,
        cognition_task_contract=CognitionTaskContract(
            kind=CognitionTaskKind.CHOOSE_OBSERVABLE_ACTION
        ),
        abstract_action="discovered_family_0",
        action_applicability_evaluator=NoOpActionApplicabilityEvaluator(),
    )

    assert legacy is None
    assert typed is not None
    assert typed.source_case_id == "case:gate-water-mark"
    assert typed.action_statement == "I will 挡在门前, then 示意阿兰后退."
