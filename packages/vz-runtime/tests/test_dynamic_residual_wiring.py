"""State KV P5-a: independent switch for the generation-time z_t residual.

Before this switch the temporal controller's ``control_code`` reached the
substrate whenever it was non-empty, regardless of any personal-conditioning
wiring -- so an "ablation" that closed State KV still leaked state through a
second latent carrier. These tests pin the three wiring levels at the
substrate boundary (the ``runtime.generate`` kwargs) and the independence of
the two channels.
"""

from __future__ import annotations

from dataclasses import replace
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


class _RecordingRuntime:
    model_id = "test-runtime"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            text="hello",
            token_count=1,
            personal_conditioning_applied=(
                kwargs.get("personal_conditioning") is not None
            ),
        )


def _context(**overrides: Any) -> ResponseContext:
    base = ResponseContext(
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
        user_input="hi",
    )
    return replace(base, **overrides) if overrides else base


def _assembly(
    *,
    control_code: tuple[float, ...] = (0.2, -0.1, 0.05),
    control_scale: float = 0.12,
) -> ResponseAssemblySnapshot:
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
        control_code=control_code,
        control_scale=control_scale,
        max_questions=0,
        prompt_residue_summary="",
        prompt_residue_ratio=0.0,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="test assembly",
    )


def _conditioning() -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(0.5 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 2),),
        source_fingerprint="conditioning-test",
        confidence=0.7,
        is_cold_start=False,
        description="test",
    )


def test_active_default_forwards_control_code_and_attests_scale() -> None:
    """Default wiring is 'active' and byte-for-byte today's behaviour."""

    runtime = _RecordingRuntime()
    synthesizer = LLMResponseSynthesizer(runtime=runtime)

    response = synthesizer.synthesize(context=_context(), assembly=_assembly())

    call = runtime.calls[0]
    assert call["control_parameters"] == (0.2, -0.1, 0.05)
    assert call["control_scale"] > 0.0
    assert (
        f"dynamic_residual=active:{call['control_scale']:.3f}"
        in response.rationale_tags
    )


def test_disabled_is_byte_equivalent_to_a_no_control_run() -> None:
    """DISABLED must reach the substrate exactly like a run whose temporal
    controller produced no code at all -- same kwargs, not merely scale 0."""

    disabled_runtime = _RecordingRuntime()
    LLMResponseSynthesizer(runtime=disabled_runtime).synthesize(
        context=_context(dynamic_residual_wiring="disabled"),
        assembly=_assembly(),
    )
    baseline_runtime = _RecordingRuntime()
    LLMResponseSynthesizer(runtime=baseline_runtime).synthesize(
        context=_context(),
        assembly=_assembly(control_code=(), control_scale=0.0),
    )

    disabled_call = disabled_runtime.calls[0]
    baseline_call = baseline_runtime.calls[0]
    assert disabled_call["control_parameters"] == ()
    assert disabled_call["control_scale"] == 0.0
    assert disabled_call == baseline_call


def test_disabled_attests_in_rationale_tags() -> None:
    runtime = _RecordingRuntime()
    response = LLMResponseSynthesizer(runtime=runtime).synthesize(
        context=_context(dynamic_residual_wiring="disabled"),
        assembly=_assembly(),
    )

    assert "dynamic_residual=disabled" in response.rationale_tags
    assert not any(
        tag.startswith("dynamic_residual=active")
        for tag in response.rationale_tags
    )


def test_shadow_withholds_injection_but_audits_would_be_scale() -> None:
    """SHADOW records what it would have injected without injecting it."""

    runtime = _RecordingRuntime()
    response = LLMResponseSynthesizer(runtime=runtime).synthesize(
        context=_context(dynamic_residual_wiring="shadow"),
        assembly=_assembly(),
    )

    call = runtime.calls[0]
    assert call["control_parameters"] == ()
    assert call["control_scale"] == 0.0
    shadow_tags = [
        tag
        for tag in response.rationale_tags
        if tag.startswith("dynamic_residual=shadow:would_be:")
    ]
    assert len(shadow_tags) == 1
    would_be = float(shadow_tags[0].rsplit(":", 1)[1])
    assert would_be > 0.0


def test_invalid_wiring_value_fails_loudly() -> None:
    with pytest.raises(ValueError, match="dynamic_residual_wiring"):
        _context(dynamic_residual_wiring="on")


def test_channels_are_independent_across_all_four_combinations() -> None:
    """Closing one latent channel must not open, close, or alter the other."""

    conditioning = _conditioning()
    for wiring in ("active", "disabled"):
        for snapshot in (conditioning, None):
            runtime = _RecordingRuntime()
            LLMResponseSynthesizer(runtime=runtime).synthesize(
                context=_context(
                    dynamic_residual_wiring=wiring,
                    personal_conditioning=snapshot,
                ),
                assembly=_assembly(),
            )
            call = runtime.calls[0]
            expected_control = (0.2, -0.1, 0.05) if wiring == "active" else ()
            assert call["control_parameters"] == expected_control, (
                f"wiring={wiring} conditioning={snapshot is not None}"
            )
            assert call["personal_conditioning"] is snapshot, (
                f"wiring={wiring} conditioning={snapshot is not None}"
            )


def test_final_rollout_config_defaults_to_active() -> None:
    from volvence_zero.integration.final_wiring import FinalRolloutConfig
    from volvence_zero.runtime import WiringLevel

    assert (
        FinalRolloutConfig().generation_dynamic_residual is WiringLevel.ACTIVE
    )


def test_dynamic_residual_off_profile_resolves_with_capability() -> None:
    from volvence_zero.agent.profile_registry import resolve_profile

    resolved = resolve_profile("dynamic-residual-off")
    assert tuple(c.name for c in resolved.capabilities) == (
        "dynamic-residual-off",
    )
    assert resolved.merged_flag_overrides["generation_dynamic_residual"] == (
        "WiringLevel.DISABLED"
    )


def test_dynamic_residual_off_runner_builds_disabled_config() -> None:
    from volvence_zero.agent.dialogue import (
        DEFAULT_DIALOGUE_PROOF_CASES,
        build_standard_dialogue_runner,
    )
    from volvence_zero.runtime import WiringLevel

    runner = build_standard_dialogue_runner(
        profile_label="dynamic-residual-off",
        case=DEFAULT_DIALOGUE_PROOF_CASES[0],
    )
    assert (
        runner._config.generation_dynamic_residual is WiringLevel.DISABLED
    )
    # The other latent channel keeps its default wiring: independence.
    assert (
        runner._config.personal_conditioning
        is type(runner._config)().personal_conditioning
    )
