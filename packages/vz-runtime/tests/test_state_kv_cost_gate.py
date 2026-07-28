"""Cost/latency gate acceptance (State KV P3 cost half).

Pins the three properties that make ``verdict_cost_gate.json`` publishable:

1. The payload claim is computed from the exact chat payload the synthesizer
   sent, with arm G charged for its prefix slots -- strict per-probe
   dominance, not an average.
2. The latency claim can never pass on a fake substrate: timing a trace-only
   runtime says nothing about production latency, so it tops out at
   ``insufficient_data`` (the same ceiling the identification verdict uses).
3. The measuring proxy fails loudly on unattributable measurements -- a
   generate call outside a declared turn, a turn with no generate call, or a
   turn with more than one.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

from volvence_zero.agent.response import LLMResponseSynthesizer, ResponseContext
from volvence_zero.application.runtime import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.state_kv_cost_gate import (
    COST_GATE_SCHEMA_VERSION,
    DEFAULT_COST_ARM_LABELS,
    TEXT_CARRIER_ARM_LABEL,
    MeasuredTurnCost,
    MeasuringRuntimeProxy,
    build_cost_gate_verdict,
    evaluate_latency_claim,
    evaluate_payload_claim,
    run_cost_gate,
)
from volvence_zero.state_kv_identification import (
    PREFIX_ARM_LABEL,
    ClaimState,
    ProbeCase,
    SubstrateEvidenceKind,
)


def _turn(
    *,
    arm_label: str = PREFIX_ARM_LABEL,
    user_id: str = "alice",
    probe_id: str = "p0",
    prompt_tokens: int = 100,
    prefix_slots: int = 0,
    latency_ms: float = 50.0,
) -> MeasuredTurnCost:
    return MeasuredTurnCost(
        arm_label=arm_label,
        user_id=user_id,
        probe_id=probe_id,
        prompt_tokens=prompt_tokens,
        prompt_chars=prompt_tokens * 4,
        prefix_slots=prefix_slots,
        generated_tokens=32,
        latency_ms=latency_ms,
    )


def _arm_turns(
    *,
    arm_label: str,
    prompt_tokens: int,
    prefix_slots: int = 0,
    latency_ms: float = 50.0,
    probe_ids: tuple[str, ...] = ("p0", "p1", "p2", "p3"),
) -> tuple[MeasuredTurnCost, ...]:
    return tuple(
        _turn(
            arm_label=arm_label,
            probe_id=probe_id,
            prompt_tokens=prompt_tokens,
            prefix_slots=prefix_slots,
            latency_ms=latency_ms,
        )
        for probe_id in probe_ids
    )


# ---------------------------------------------------------------------------
# Payload claim
# ---------------------------------------------------------------------------


def test_payload_claim_charges_prefix_slots_to_arm_g() -> None:
    # G prompt is shorter (no statement) but the slots close most of the gap:
    # 80 + 16 = 96 < 120 -- still cheaper, claim passes.
    candidate = _arm_turns(
        arm_label=PREFIX_ARM_LABEL, prompt_tokens=80, prefix_slots=16
    )
    baseline = _arm_turns(
        arm_label=TEXT_CARRIER_ARM_LABEL, prompt_tokens=120
    )

    claim = evaluate_payload_claim(
        candidate_turns=candidate, baseline_turns=baseline
    )

    assert claim.state is ClaimState.PASS
    assert "cheaper on all 4" in claim.detail


def test_payload_claim_fails_when_slots_erase_the_advantage() -> None:
    # 80 + 48 = 128 >= 120: a big prefix must not masquerade as free.
    candidate = _arm_turns(
        arm_label=PREFIX_ARM_LABEL, prompt_tokens=80, prefix_slots=48
    )
    baseline = _arm_turns(
        arm_label=TEXT_CARRIER_ARM_LABEL, prompt_tokens=120
    )

    claim = evaluate_payload_claim(
        candidate_turns=candidate, baseline_turns=baseline
    )

    assert claim.state is ClaimState.FAIL
    assert "128 >= 120" in claim.detail


def test_payload_claim_requires_matching_probe_sets() -> None:
    candidate = _arm_turns(
        arm_label=PREFIX_ARM_LABEL, prompt_tokens=80, probe_ids=("p0",)
    )
    baseline = _arm_turns(
        arm_label=TEXT_CARRIER_ARM_LABEL, prompt_tokens=120, probe_ids=("p1",)
    )

    claim = evaluate_payload_claim(
        candidate_turns=candidate, baseline_turns=baseline
    )

    assert claim.state is ClaimState.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# Latency claim
# ---------------------------------------------------------------------------


def test_latency_claim_never_passes_on_a_fake_substrate() -> None:
    candidate = _arm_turns(
        arm_label=PREFIX_ARM_LABEL, prompt_tokens=80, latency_ms=10.0
    )
    baseline = _arm_turns(
        arm_label=TEXT_CARRIER_ARM_LABEL, prompt_tokens=120, latency_ms=100.0
    )

    claim = evaluate_latency_claim(
        candidate_turns=candidate,
        baseline_turns=baseline,
        substrate_kind=SubstrateEvidenceKind.TRACE_ONLY,
        tolerance=0.10,
    )

    assert claim.state is ClaimState.INSUFFICIENT_DATA
    assert "trace-only" in claim.detail


def test_latency_claim_median_respects_the_tolerance_budget() -> None:
    baseline = _arm_turns(
        arm_label=TEXT_CARRIER_ARM_LABEL, prompt_tokens=120, latency_ms=100.0
    )
    within = evaluate_latency_claim(
        candidate_turns=_arm_turns(
            arm_label=PREFIX_ARM_LABEL, prompt_tokens=80, latency_ms=109.0
        ),
        baseline_turns=baseline,
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        tolerance=0.10,
    )
    over = evaluate_latency_claim(
        candidate_turns=_arm_turns(
            arm_label=PREFIX_ARM_LABEL, prompt_tokens=80, latency_ms=111.0
        ),
        baseline_turns=baseline,
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        tolerance=0.10,
    )

    assert within.state is ClaimState.PASS
    assert over.state is ClaimState.FAIL


def test_latency_claim_needs_enough_turns_for_a_stable_median() -> None:
    claim = evaluate_latency_claim(
        candidate_turns=_arm_turns(
            arm_label=PREFIX_ARM_LABEL, prompt_tokens=80, probe_ids=("p0",)
        ),
        baseline_turns=_arm_turns(
            arm_label=TEXT_CARRIER_ARM_LABEL, prompt_tokens=120, probe_ids=("p0",)
        ),
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        tolerance=0.10,
    )

    assert claim.state is ClaimState.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------


def _verdict(
    *,
    substrate_kind: SubstrateEvidenceKind,
    candidate_latency: float = 50.0,
    candidate_prompt_tokens: int = 80,
):
    turns = (
        *_arm_turns(
            arm_label=PREFIX_ARM_LABEL,
            prompt_tokens=candidate_prompt_tokens,
            prefix_slots=16,
            latency_ms=candidate_latency,
        ),
        *_arm_turns(
            arm_label=TEXT_CARRIER_ARM_LABEL,
            prompt_tokens=120,
            latency_ms=100.0,
        ),
    )
    return build_cost_gate_verdict(
        turns=turns,
        substrate_kind=substrate_kind,
        substrate_fingerprint="model@fp",
    )


def test_gate_passes_only_when_both_claims_pass() -> None:
    verdict = _verdict(substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS)

    assert verdict.schema_version == COST_GATE_SCHEMA_VERSION
    assert verdict.gate_state == "pass"
    assert {claim.state for claim in verdict.claims} == {ClaimState.PASS}
    assert verdict.arm_summaries[0].arm_label == PREFIX_ARM_LABEL
    assert verdict.arm_summaries[0].prefix_slots == 16


def test_fake_substrate_caps_the_gate_at_insufficient_data() -> None:
    verdict = _verdict(substrate_kind=SubstrateEvidenceKind.TRACE_ONLY)

    assert verdict.gate_state == "insufficient_data"
    assert verdict.claim(
        "claim_conditioning_payload_smaller"
    ).state is ClaimState.PASS
    assert verdict.claim(
        "claim_latency_not_worse"
    ).state is ClaimState.INSUFFICIENT_DATA


def test_any_failed_claim_fails_the_gate() -> None:
    verdict = _verdict(
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        candidate_latency=200.0,
    )

    assert verdict.gate_state == "fail"


def test_verdict_rejects_missing_fingerprint_and_unknown_arms() -> None:
    turns = _arm_turns(arm_label=PREFIX_ARM_LABEL, prompt_tokens=80)
    with pytest.raises(ValueError, match="fingerprint"):
        build_cost_gate_verdict(
            turns=turns,
            substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
            substrate_fingerprint="",
        )
    with pytest.raises(ValueError, match="unexpected arms"):
        build_cost_gate_verdict(
            turns=_arm_turns(arm_label="state-kv-arm-e", prompt_tokens=80),
            substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
            substrate_fingerprint="model@fp",
        )


# ---------------------------------------------------------------------------
# Measuring proxy protocol
# ---------------------------------------------------------------------------


class _FakeRuntime:
    """Trace-only fake mirroring the identification tests' substrate."""

    model_id = "fake-frozen-substrate"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        parts = [str(kwargs.get("system_context", "")), str(kwargs.get("prompt", ""))]
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8]
        return SimpleNamespace(
            text=f"reply-{digest}",
            token_count=7,
            personal_conditioning_applied=False,
        )


def _whitespace_counter(chat_messages) -> int:
    return sum(len(content.split()) for _, content in chat_messages)


def test_proxy_rejects_unattributable_and_missing_generate_calls() -> None:
    proxy = MeasuringRuntimeProxy(
        _FakeRuntime(), token_counter=_whitespace_counter
    )

    with pytest.raises(RuntimeError, match="outside a declared turn"):
        proxy.generate(chat_messages=(("system", "s"), ("user", "u")))

    proxy.begin_turn(
        arm_label=PREFIX_ARM_LABEL, user_id="alice", probe_id="p0", prefix_slots=16
    )
    with pytest.raises(RuntimeError, match="no generate call"):
        proxy.finish_turn()


def test_proxy_measures_the_exact_sent_payload() -> None:
    proxy = MeasuringRuntimeProxy(
        _FakeRuntime(), token_counter=_whitespace_counter
    )
    proxy.begin_turn(
        arm_label=PREFIX_ARM_LABEL, user_id="alice", probe_id="p0", prefix_slots=16
    )
    proxy.generate(
        chat_messages=(("system", "one two three"), ("user", "four five")),
        prompt="four five",
        system_context="one two three",
    )
    turn = proxy.finish_turn()

    assert turn.prompt_tokens == 5
    assert turn.prompt_chars == len("one two three") + len("four five")
    assert turn.prefix_slots == 16
    assert turn.generated_tokens == 7
    assert turn.latency_ms >= 0.0
    assert turn.conditioning_cost_tokens == 21


# ---------------------------------------------------------------------------
# End-to-end on the production synthesizer (fake substrate)
# ---------------------------------------------------------------------------


def _assembly() -> ResponseAssemblySnapshot:
    return ResponseAssemblySnapshot(
        regime_id="steady",
        regime_name="Steady",
        abstract_action=None,
        response_mode=ResponseMode.SUPPORT,
        answer_depth_limit="high-level-only",
        citation_mode="none",
        clarification_required=False,
        refer_out_required=False,
        ordering_plan=(),
        knowledge_briefs=(),
        case_briefs=(),
        playbook_ordering=(),
        required_disclaimers=(),
        required_disclaimer_phrases=(),
        control_code=(),
        control_scale=0.0,
        max_questions=0,
        prompt_residue_summary="",
        prompt_residue_ratio=0.0,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="cost gate probe assembly",
    )


def _conditioning(*, user: str) -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(0.6 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 2),),
        source_fingerprint=f"conditioning-{user}",
        confidence=0.7,
        is_cold_start=False,
        description=f"{user} probe state",
        rendered_statement=(
            f"State readout for {user}: trust moderate, overwhelm moderate, "
            "decision readiness moderate, boundary compliance moderate."
        ),
    )


def _cases() -> tuple[ProbeCase, ...]:
    return tuple(
        ProbeCase(
            user_id="alice",
            probe_id=probe_id,
            user_input=text,
            conditioning=_conditioning(user="alice"),
            assembly=_assembly(),
        )
        for probe_id, text in (
            ("p0", "我又搞砸了"),
            ("p1", "今天还是没睡好"),
            ("p2", "你觉得我该继续吗"),
            ("p3", "我现在有点撑不住了"),
        )
    )


def _base_context() -> ResponseContext:
    return ResponseContext(
        regime_id="steady",
        regime_name="Steady",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="none",
    )


def test_run_cost_gate_measures_both_arms_through_the_production_prompt_path() -> None:
    runtime = _FakeRuntime()
    proxy = MeasuringRuntimeProxy(runtime, token_counter=_whitespace_counter)
    synthesizer = LLMResponseSynthesizer(
        runtime=proxy, max_new_tokens=32, temperature=0.0
    )

    verdict = run_cost_gate(
        cases=_cases(),
        synthesizer=synthesizer,
        measuring_runtime=proxy,
        base_context=_base_context(),
        substrate_kind=SubstrateEvidenceKind.TRACE_ONLY,
        substrate_fingerprint="fake@fp",
        prefix_slots=16,
    )

    assert verdict.gate_state in ("insufficient_data", "fail")
    by_arm = {summary.arm_label: summary for summary in verdict.arm_summaries}
    assert set(by_arm) == set(DEFAULT_COST_ARM_LABELS)
    assert by_arm[PREFIX_ARM_LABEL].prefix_slots == 16
    assert by_arm[TEXT_CARRIER_ARM_LABEL].prefix_slots == 0
    # The text arm's prompt must be strictly larger: it carries the rendered
    # statement the prefix arm delivers latently.
    assert (
        by_arm[TEXT_CARRIER_ARM_LABEL].median_prompt_tokens
        > by_arm[PREFIX_ARM_LABEL].median_prompt_tokens
    )
    # 8 turns, one generate call each, all through the real synthesizer.
    assert len(verdict.turn_table) == 8
    assert len(runtime.calls) == 8
    # Latency on a fake substrate must not have produced a positive claim.
    assert verdict.claim(
        "claim_latency_not_worse"
    ).state is ClaimState.INSUFFICIENT_DATA
