"""C4: evaluation mid / expensive / cross-generation tiers are real.

All tiers stay DISABLED by default and read-only (R12): they re-emit
owner-published readouts, never recompute owner state, and LLM-judge
readouts remain permanently gate-ineligible.
"""

from __future__ import annotations

import asyncio

from volvence_zero.evaluation.cross_generation_aggregator import (
    CrossGenerationAggregatorModule,
    build_cross_generation_window_snapshot,
)
from volvence_zero.evaluation.expensive_layer import (
    ExpensiveLayerModule,
    ExpensiveLayerSnapshot,
    HeadToHeadResult,
    LlmJudgeCase,
    build_substrate_geometry_scores,
    run_llm_judge_readouts,
)
from volvence_zero.evaluation.mid_layer import MidLayerModule, MidLayerSnapshot
from volvence_zero.evaluation.types import EvaluationSnapshot
from volvence_zero.runtime.kernel import Snapshot
from volvence_zero.substrate import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
)


def _snapshot(slot: str, value: object, *, owner: str = "test") -> Snapshot:
    return Snapshot(slot_name=slot, owner=owner, version=1, timestamp_ms=1, value=value)


def _evaluation_value() -> EvaluationSnapshot:
    return EvaluationSnapshot((), (), (), "base")


def _substrate_value() -> SubstrateSnapshot:
    steps = tuple(
        ResidualSequenceStep(
            step=index,
            token=f"t{index}",
            feature_surface=(),
            residual_activations=(
                ResidualActivation(
                    layer_index=0,
                    activation=(0.1 * index, 0.2, -0.1 * index),
                    step=index,
                ),
            ),
            description="",
        )
        for index in range(4)
    )
    return SubstrateSnapshot(
        model_id="test-model",
        is_frozen=True,
        surface_kind=SurfaceKind.FEATURE_SURFACE,
        token_logits=(),
        feature_surface=(FeatureSignal(name="warmth", values=(0.5,), source="test"),),
        residual_activations=(),
        residual_sequence=steps,
        unavailable_fields=(),
        description="test substrate",
    )


# ---------------------------------------------------------------------------
# Mid tier: PE / regime readout aggregation
# ---------------------------------------------------------------------------


def test_mid_layer_aggregates_pe_and_regime_readouts() -> None:
    from volvence_zero.credit import CreditLedger
    from volvence_zero.prediction.error import PredictionErrorModule
    from volvence_zero.regime import RegimeModule

    pe_snapshot = asyncio.run(PredictionErrorModule().process_standalone(turn_index=0))
    regime_module = RegimeModule()
    regime_snapshot = asyncio.run(regime_module.process_standalone())

    mid = asyncio.run(
        MidLayerModule().process(
            {
                "evaluation": _snapshot("evaluation", _evaluation_value()),
                "credit": _snapshot("credit", CreditLedger().snapshot()),
                "prediction_error": pe_snapshot,
                "regime": regime_snapshot,
            }
        )
    )
    metric_names = {score.metric_name for score in mid.value.aggregated_scores}
    # Bootstrap PE turn publishes no PE scores; regime scores must appear.
    assert "regime_persistence" in metric_names


def test_mid_layer_still_works_without_pe_and_regime() -> None:
    from volvence_zero.credit import CreditLedger

    mid = asyncio.run(
        MidLayerModule().process(
            {
                "evaluation": _snapshot("evaluation", _evaluation_value()),
                "credit": _snapshot("credit", CreditLedger().snapshot()),
            }
        )
    )
    assert isinstance(mid.value, MidLayerSnapshot)


# ---------------------------------------------------------------------------
# Expensive tier: substrate geometry + LLM judge seam
# ---------------------------------------------------------------------------


def test_substrate_geometry_scores_shape() -> None:
    scores = build_substrate_geometry_scores(_substrate_value())
    names = [score.metric_name for score in scores]
    assert names == ["persona_geometry_norm", "persona_geometry_drift"]
    assert scores[0].value > 0.0
    assert 0.0 <= scores[1].value <= 1.0


def test_expensive_layer_reemits_mid_scores_and_geometry() -> None:
    mid_value = MidLayerSnapshot(
        scenario_id="",
        seeds=(),
        profile_label="",
        baseline_label="",
        aggregated_scores=(),
        counterfactual_readouts=(),
        acceptance_gate_passed=True,
        acceptance_gate_reasons=(),
        description="mid",
    )
    snapshot = asyncio.run(
        ExpensiveLayerModule().process(
            {
                "evaluation_mid": _snapshot("evaluation_mid", mid_value),
                "substrate": _snapshot("substrate", _substrate_value()),
            }
        )
    )
    metric_names = {score.metric_name for score in snapshot.value.aggregated_scores}
    assert "persona_geometry_norm" in metric_names
    # Zero-LLM invariant: no backend injected -> no judge readouts.
    assert snapshot.value.llm_judge_readouts == ()


class _FakeJudgeBackend:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert "naturalness" in system_prompt
        assert "Assistant response" in user_prompt
        self.calls += 1
        return self.reply


def test_llm_judge_seam_produces_gate_ineligible_readouts() -> None:
    backend = _FakeJudgeBackend("naturalness=0.8 coherence=0.9 note=fluent")
    readouts = run_llm_judge_readouts(
        backend=backend,
        judge_model_id="fake-judge",
        cases=(LlmJudgeCase(case_id="c1", dialogue_context="ctx", response_text="hi"),),
    )
    assert backend.calls == 1
    assert readouts[0].naturalness_score == 0.8
    assert readouts[0].coherence_score == 0.9
    assert readouts[0].is_gate_eligible is False


def test_llm_judge_malformed_reply_fails_loudly() -> None:
    backend = _FakeJudgeBackend("I think it was pretty good!")
    try:
        run_llm_judge_readouts(
            backend=backend,
            judge_model_id="fake-judge",
            cases=(LlmJudgeCase(case_id="c1", dialogue_context="", response_text=""),),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("malformed judge reply must raise")


def test_submit_judge_cases_requires_backend() -> None:
    module = ExpensiveLayerModule()
    try:
        module.submit_judge_cases(
            (LlmJudgeCase(case_id="c1", dialogue_context="", response_text=""),)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("submitting cases without a backend must raise")


# ---------------------------------------------------------------------------
# Cross-generation tier: bounded multi-generation window
# ---------------------------------------------------------------------------


def _expensive_generation(generation_id: str, winrate: float) -> ExpensiveLayerSnapshot:
    return ExpensiveLayerSnapshot(
        generation_id=generation_id,
        head_to_head_results=(
            HeadToHeadResult(
                profile_a="candidate",
                profile_b="baseline",
                case_count=10,
                winrate_a_vs_b=winrate,
                confidence_interval_low=winrate,
                confidence_interval_high=winrate,
                judge_kind="deterministic",
                notes="",
            ),
        ),
        llm_judge_readouts=(),
        aggregated_scores=(),
        description="gen",
    )


def test_window_snapshot_aggregates_across_generations() -> None:
    snapshot = build_cross_generation_window_snapshot(
        window=(
            _expensive_generation("gen-1", 0.6),
            _expensive_generation("gen-2", 0.8),
        ),
        timestamp_ms=10,
    )
    assert snapshot.generation_id_window == ("gen-1", "gen-2")
    assert (
        snapshot.modification_gate_evidence.head_to_head_aggregate_winrate == 0.7
    )


def test_aggregator_module_keeps_bounded_window() -> None:
    module = CrossGenerationAggregatorModule()
    latest = None
    for index in range(7):
        latest = asyncio.run(
            module.process(
                {
                    "evaluation_expensive": _snapshot(
                        "evaluation_expensive",
                        _expensive_generation(f"gen-{index}", 0.5),
                    ),
                }
            )
        )
    assert latest is not None
    # Window bounded at 5: oldest generations evicted.
    window = latest.value.generation_id_window
    assert len(window) == 5
    assert "gen-0" not in window
    assert "gen-6" in window
