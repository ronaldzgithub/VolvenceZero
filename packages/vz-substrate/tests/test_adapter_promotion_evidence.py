from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from volvence_zero.credit import GateDecision
from volvence_zero.substrate import ContinuationScore

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "adapter_promotion_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "adapter_promotion_evidence",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load adapter evidence helpers from {SCRIPT_PATH}")
EVIDENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVIDENCE
SPEC.loader.exec_module(EVIDENCE)


def _state(marker: float) -> tuple[float, ...]:
    return (marker,) * 16


def _case(index: int, *, expectation: str):
    return EVIDENCE.AdapterHeldOutCase(
        case_id=f"case-{index}",
        cohort="relationship" if expectation == "improve" else "safety",
        expectation=expectation,
        source_text=f"source {index}",
        continuation_text=f"continuation {index}",
        conditioning_state=_state(0.6),
        counterfactual_conditioning_state=_state(0.4),
        applied_control=(0.2,),
    )


def _score(case, nll: float) -> ContinuationScore:
    return ContinuationScore(
        source_text=case.source_text,
        continuation_text=case.continuation_text,
        token_count=2,
        mean_negative_log_likelihood=nll,
        geometric_mean_probability=0.5,
        applied_control=case.applied_control,
        backend_name="test",
        description="test score",
    )


def test_held_out_loader_is_strict_and_builds_typed_conditioning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "held-out.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "adapter-held-out-case.v1",
                "case_id": "held-out-1",
                "cohort": "relationship",
                "expectation": "improve",
                "source_text": "source",
                "continuation_text": "continuation",
                "conditioning_state": list(_state(0.6)),
                "counterfactual_conditioning_state": list(_state(0.4)),
                "applied_control": [0.1],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = EVIDENCE.load_held_out_cases(path)
    snapshot = EVIDENCE.conditioning_snapshot(cases[0])

    assert snapshot.state_vector == _state(0.6)
    assert snapshot.confidence == 1.0
    assert snapshot.source_fingerprint


def test_offline_gate_allows_complete_held_out_evidence() -> None:
    cases = tuple(
        _case(index, expectation="improve" if index < 6 else "preserve")
        for index in range(8)
    )
    observations = EVIDENCE.collect_observations(
        cases=cases,
        baseline_scorer=lambda case: _score(case, 1.0),
        candidate_scorer=lambda case, counterfactual: _score(
            case,
            0.9 if counterfactual else (0.8 if case.expectation == "improve" else 0.99),
        ),
    )
    summary = EVIDENCE.summarize_observations(
        observations=observations,
        thresholds=EVIDENCE.AdapterPromotionThresholds(),
    )
    decision, reasons, _ = EVIDENCE.decide_offline_promotion(
        target="substrate.common_adapter_bundle",
        old_value_hash="a" * 64,
        new_value_hash="b" * 64,
        summary=summary,
        capacity_cost=0.1,
        rollback_evidence="restore frozen base",
    )

    assert summary["evidence_integrity"] is True
    assert summary["validation_delta"] == pytest.approx(0.2)
    assert decision is GateDecision.ALLOW
    assert reasons == ()


def test_offline_gate_blocks_incomplete_or_regressing_evidence() -> None:
    cases = tuple(
        EVIDENCE.AdapterHeldOutCase(
            case_id=f"case-{index}",
            cohort="relationship",
            expectation="improve",
            source_text="source",
            continuation_text="continuation",
            conditioning_state=_state(0.6),
            counterfactual_conditioning_state=None,
            applied_control=(0.0,),
        )
        for index in range(2)
    )
    observations = EVIDENCE.collect_observations(
        cases=cases,
        baseline_scorer=lambda case: _score(case, 1.0),
        candidate_scorer=lambda case, counterfactual: _score(case, 1.2),
    )
    summary = EVIDENCE.summarize_observations(
        observations=observations,
        thresholds=EVIDENCE.AdapterPromotionThresholds(),
    )
    decision, reasons, _ = EVIDENCE.decide_offline_promotion(
        target="substrate.character_package.demo",
        old_value_hash="a" * 64,
        new_value_hash="b" * 64,
        summary=summary,
        capacity_cost=0.1,
        rollback_evidence="disable character package",
    )

    assert summary["evidence_integrity"] is False
    assert decision is GateDecision.BLOCK
    assert any("validation_delta" in reason for reason in reasons)
    assert any("contract_integrity" in reason for reason in reasons)
