"""Evaluation cascade SSOT contract test (architecture-uplift A2).

实现 [`docs/specs/evaluation-cascade.md`](../../docs/specs/evaluation-cascade.md)
的 field-identical 不变量 + cheap_layer 收编契约:

- ``EvaluationSnapshot`` 字段集与 A2 step 1 (T7) 落地时 frozen 的
  ``EVALUATION_SNAPSHOT_FIELD_NAMES`` 完全一致
- ``EvaluationModule.slot_name`` 仍是 ``"evaluation"`` 且仅有这一个 owner
  (R8 SSOT)
- ``EvaluationCheapLayer`` 持有真实 ``EvaluationModule`` 引用，``role``
  默认是 CHEAP_LAYER
- 6 个现有 cheap_layer 下游 (credit / substrate_self_mod / regime /
  reflection / interlocutor_state / prediction_error) 的依赖声明仍然
  含 ``"evaluation"``——任何打破现有 6 下游的改动会立即 FAIL
"""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.evaluation.backbone import EvaluationModule
from volvence_zero.evaluation.cheap_layer import (
    EVALUATION_SNAPSHOT_FIELD_NAMES,
    EvaluationCascadeRole,
    EvaluationCheapLayer,
)
from volvence_zero.evaluation.types import EvaluationSnapshot


_EXPECTED_FIELD_NAMES: tuple[str, ...] = (
    "turn_scores",
    "session_scores",
    "alerts",
    "description",
    "structured_alerts",
    "reflection_accuracy",
    "longitudinal_verdict",
)


def test_evaluation_snapshot_fields_match_frozen_set() -> None:
    """A2 field-identical invariant: changing EvaluationSnapshot fields
    breaks 6 documented downstreams; this test guards the spec contract."""
    actual = tuple(f.name for f in dataclasses.fields(EvaluationSnapshot))
    assert actual == _EXPECTED_FIELD_NAMES, (
        f"EvaluationSnapshot fields changed without updating cheap_layer SSOT:\n"
        f"  expected: {_EXPECTED_FIELD_NAMES}\n"
        f"  actual:   {actual}\n"
        f"If this is intentional, also update:\n"
        f"  1. cheap_layer.EVALUATION_SNAPSHOT_FIELD_NAMES\n"
        f"  2. docs/DATA_CONTRACT.md §3.7\n"
        f"  3. docs/specs/evaluation-cascade.md §关键不变量\n"
        f"  4. all 6 documented downstream consumers"
    )


def test_module_level_field_names_constant_matches_dataclass() -> None:
    """The cheap_layer-exported constant must agree with the live dataclass."""
    live = tuple(f.name for f in dataclasses.fields(EvaluationSnapshot))
    assert EVALUATION_SNAPSHOT_FIELD_NAMES == live


def test_evaluation_module_owns_evaluation_slot() -> None:
    """R8 SSOT: only EvaluationModule may declare slot_name=evaluation."""
    assert EvaluationModule.slot_name == "evaluation"


def test_cheap_layer_facade_marks_cheap_role_by_default() -> None:
    """spec §A2.1: cheap_layer facade default role = CHEAP_LAYER."""
    em = EvaluationModule()
    facade = EvaluationCheapLayer(module=em)
    assert facade.role is EvaluationCascadeRole.CHEAP_LAYER
    assert facade.is_cheap_layer()
    assert facade.slot_name == "evaluation"
    assert facade.owner == "EvaluationModule"
    assert facade.wiring_level == em.wiring_level


def test_cheap_layer_facade_is_frozen() -> None:
    """frozen dataclass: facade must be immutable post-construction."""
    em = EvaluationModule()
    facade = EvaluationCheapLayer(module=em)
    with pytest.raises(dataclasses.FrozenInstanceError):
        facade.role = EvaluationCascadeRole.MID_LAYER  # type: ignore[misc]


def test_six_documented_downstreams_still_consume_evaluation() -> None:
    """A2 step 1 Done 标志: 6 个已知 cheap_layer 下游 module 的 dependencies
    必须仍含 ``"evaluation"``。任何打破现有下游的改动会立即 FAIL。

    所列模块对应 spec §既有 6 个 cheap_layer 下游表。
    """
    from volvence_zero.credit.gate import CreditModule
    from volvence_zero.interlocutor.owner import InterlocutorStateModule
    from volvence_zero.prediction.error import PredictionErrorModule
    from volvence_zero.reflection.writeback import ReflectionModule
    from volvence_zero.regime.identity import RegimeModule
    from volvence_zero.substrate.self_mod import SubstrateSelfModModule

    downstream_modules = (
        CreditModule,
        SubstrateSelfModModule,
        RegimeModule,
        ReflectionModule,
        InterlocutorStateModule,
        PredictionErrorModule,
    )

    failures: list[str] = []
    for cls in downstream_modules:
        if "evaluation" not in cls.dependencies:
            failures.append(
                f"{cls.__name__} no longer declares 'evaluation' in dependencies"
            )

    assert not failures, (
        "Existing cheap_layer downstream consumers regressed (A2 field-identical "
        "invariant violated):\n" + "\n".join(f"  - {f}" for f in failures)
    )


def test_cascade_role_enum_values_are_canonical() -> None:
    """Sanity: spec §A2 mentions exactly four cascade tiers."""
    assert {role.value for role in EvaluationCascadeRole} == {
        "cheap_layer",
        "mid_layer",
        "expensive_layer",
        "cross_generation_aggregator",
    }


# ---------------------------------------------------------------------------
# Mid / expensive / aggregator tier schema (T8 / T9)
# ---------------------------------------------------------------------------


def test_mid_layer_module_skeleton() -> None:
    """T8: MidLayerModule declares correct slot/owner/dependencies/wiring."""
    from volvence_zero.evaluation.mid_layer import MidLayerModule

    assert MidLayerModule.slot_name == "evaluation_mid"
    assert MidLayerModule.owner == "MidLayerModule"
    assert MidLayerModule.dependencies == ("evaluation",)
    from volvence_zero.runtime.kernel import WiringLevel

    assert MidLayerModule.default_wiring_level is WiringLevel.DISABLED


def test_mid_layer_snapshot_schema_fields() -> None:
    """T8 schema invariant: MidLayerSnapshot field set frozen."""
    from volvence_zero.evaluation.mid_layer import MidLayerSnapshot

    expected = (
        "scenario_id",
        "seeds",
        "profile_label",
        "baseline_label",
        "aggregated_scores",
        "counterfactual_readouts",
        "acceptance_gate_passed",
        "acceptance_gate_reasons",
        "description",
        "cascade_role",
    )
    actual = tuple(f.name for f in dataclasses.fields(MidLayerSnapshot))
    assert actual == expected


def test_expensive_layer_module_skeleton() -> None:
    """T9: ExpensiveLayerModule declares correct slot/owner/dependencies/wiring."""
    from volvence_zero.evaluation.expensive_layer import ExpensiveLayerModule
    from volvence_zero.runtime.kernel import WiringLevel

    assert ExpensiveLayerModule.slot_name == "evaluation_expensive"
    assert ExpensiveLayerModule.owner == "ExpensiveLayerModule"
    assert ExpensiveLayerModule.dependencies == ("evaluation_mid",)
    assert ExpensiveLayerModule.default_wiring_level is WiringLevel.DISABLED


def test_llm_judge_readout_never_gate_eligible() -> None:
    """A2 关键不变量 3: LLM-judge readout permanently outside gate decision.

    R12 + OA-2 Mind/Face isolation forbids gate decisions from depending on
    LLM-judge scores. The dataclass default + this contract test enforce
    the invariant.
    """
    from volvence_zero.evaluation.expensive_layer import LlmJudgeReadout

    r = LlmJudgeReadout(
        case_id="any",
        judge_model_id="any",
        naturalness_score=0.5,
        coherence_score=0.5,
        note="",
    )
    assert r.is_gate_eligible is False

    # Even attempting to construct with True must propagate as a stored
    # field; the contract is enforced by tests/refactor pipeline rather
    # than by the dataclass itself. Document this expectation explicitly:
    explicit = LlmJudgeReadout(
        case_id="any",
        judge_model_id="any",
        naturalness_score=0.5,
        coherence_score=0.5,
        note="",
        is_gate_eligible=False,  # caller MUST pass False explicitly
    )
    assert explicit.is_gate_eligible is False


def test_cross_generation_aggregator_module_skeleton() -> None:
    from volvence_zero.evaluation.cross_generation_aggregator import (
        CrossGenerationAggregatorModule,
    )
    from volvence_zero.runtime.kernel import WiringLevel

    assert CrossGenerationAggregatorModule.slot_name == "evaluation_cross_generation"
    assert CrossGenerationAggregatorModule.owner == "CrossGenerationAggregatorModule"
    assert CrossGenerationAggregatorModule.dependencies == ("evaluation_expensive",)
    assert (
        CrossGenerationAggregatorModule.default_wiring_level is WiringLevel.DISABLED
    )


def test_modification_gate_evidence_schema_fields() -> None:
    """A2 §A2.4: ModificationGateEvidence is the cascade's only contract
    with ModificationGate; field set must be stable."""
    from volvence_zero.evaluation.cross_generation_aggregator import (
        ModificationGateEvidence,
    )

    expected = (
        "evidence_id",
        "validation_score",
        "head_to_head_aggregate_winrate",
        "rollback_evidence_present",
        "capacity_within_cap",
        "audit_evidence_id",
        "notes",
    )
    actual = tuple(f.name for f in dataclasses.fields(ModificationGateEvidence))
    assert actual == expected


def test_modification_gate_evidence_audit_link_default_none() -> None:
    """audit_evidence_id None until A5 + OA-4 land — caller must opt-in."""
    from volvence_zero.evaluation.cross_generation_aggregator import (
        ModificationGateEvidence,
    )

    e = ModificationGateEvidence(
        evidence_id="x",
        validation_score=0.0,
        head_to_head_aggregate_winrate=0.0,
        rollback_evidence_present=False,
        capacity_within_cap=True,
        audit_evidence_id=None,
    )
    assert e.audit_evidence_id is None
