"""Prompt-carrier closure and per-turn carrier attestation.

Implements the P0-contract acceptance of
``docs/specs/state-kv-identification-evidence.md``: the claim "relationship
state did not reach the model through the prompt" must be checkable by
comparing fingerprints, not asserted in prose.

Two facts these tests pin down, in this order:

1. The **default** path carries memory in the prompt — ``prompt_residue_summary``
   embeds retrieved memory content verbatim. Any prompt-free claim about the
   default arm is false, and ``test_default_delivery_puts_memory_text_in_prompt``
   is the standing counter-example.
2. ``prompt_state_delivery="suppressed"`` closes that carrier: two users with
   divergent state produce a byte-identical prompt, so a behavioural difference
   downstream cannot be attributed to prompt text.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from volvence_zero.agent.carrier_attestation import (
    AUDIT_FINGERPRINT_LENGTH,
    decode_fingerprint,
    prompt_fingerprint,
)
from volvence_zero.agent.prompts import (
    build_chat_messages,
    build_system_prompt,
    state_prompt_section_count,
)
from volvence_zero.agent.response import LLMResponseSynthesizer, ResponseContext
from volvence_zero.application.runtime import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)

# The retrieved memory content that runtime_helpers._prompt_residue_summary
# splices into the residue section on the default path.
_ALICE_MEMORY = (
    "Carry forward continuity from prior context: her cat died last week."
)
_BOB_MEMORY = (
    "Carry forward continuity from prior context: he starts the new job Monday."
)


class _RecordingRuntime:
    model_id = "test-runtime"

    def __init__(self, *, applies_personal_conditioning: bool = True) -> None:
        self.calls: list[dict[str, Any]] = []
        self._applies = applies_personal_conditioning

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            text="hello",
            token_count=1,
            personal_conditioning_applied=(
                self._applies and kwargs.get("personal_conditioning") is not None
            ),
        )


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
        description="test assembly",
    )


def _alice_assembly() -> ResponseAssemblySnapshot:
    return replace(
        _assembly(),
        regime_name="Steady Support",
        prompt_residue_summary=_ALICE_MEMORY,
        required_disclaimers=("grief-sensitive",),
        ordering_driver="continuum-support-first",
    )


def _bob_assembly() -> ResponseAssemblySnapshot:
    return replace(
        _assembly(),
        regime_name="Task Focus",
        prompt_residue_summary=_BOB_MEMORY,
        clarification_required=True,
        ordering_driver="continuum-structure-first",
    )


def _context(
    *,
    prompt_state_delivery: str = "text",
    conditioning: PersonalConditioningSnapshot | None = None,
    sampling_seed: int | None = None,
    character_grounding_statement: str = "",
    character_grounding_ref: str = "",
) -> ResponseContext:
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
        user_input="我又搞砸了",
        prompt_state_delivery=prompt_state_delivery,
        personal_conditioning=conditioning,
        sampling_seed=sampling_seed,
        character_grounding_statement=character_grounding_statement,
        character_grounding_ref=character_grounding_ref,
    )


def _conditioning(*, fill: float) -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(fill for _ in PERSONAL_CONDITIONING_VECTOR_LABELS),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 2),),
        source_fingerprint=f"conditioning-{fill}",
        confidence=0.7,
        is_cold_start=False,
        description="test",
    )


def _tag_value(response: Any, key: str) -> str:
    prefix = f"{key}="
    matches = [t for t in response.rationale_tags if t.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {key} tag, got {matches}"
    return matches[0][len(prefix) :]


# ---------------------------------------------------------------------------
# 1. The default path carries memory in the prompt
# ---------------------------------------------------------------------------


def test_default_delivery_puts_memory_text_in_prompt() -> None:
    """Standing counter-example to any prompt-free claim about the default arm."""

    prompt = build_system_prompt(
        assembly=_alice_assembly(), context=_context()
    )

    assert "her cat died last week" in prompt
    assert state_prompt_section_count(
        assembly=_alice_assembly(), context=_context()
    ) > 0


def test_default_delivery_prompt_diverges_between_users() -> None:
    alice = build_system_prompt(assembly=_alice_assembly(), context=_context())
    bob = build_system_prompt(assembly=_bob_assembly(), context=_context())

    assert alice != bob


def test_character_grounding_reaches_default_prompt_and_attestation() -> None:
    context = _context(
        character_grounding_statement=(
            "我是张无忌，要从光明顶之后的亲历位置回答。"
        ),
        character_grounding_ref="character:zhang-wuji:test",
    )

    prompt = build_system_prompt(assembly=_alice_assembly(), context=context)
    response = LLMResponseSynthesizer(runtime=_RecordingRuntime()).synthesize(
        context=context, assembly=_alice_assembly()
    )

    assert "Character grounding for this turn" in prompt
    assert "not content to summarize" in prompt
    assert "Use only the parts directly relevant" in prompt
    assert "我是张无忌" in prompt
    assert _tag_value(response, "character_grounding") == "character:zhang-wuji:test"


# ---------------------------------------------------------------------------
# 2. Suppression closes carrier C1
# ---------------------------------------------------------------------------


def test_suppressed_prompt_is_byte_identical_across_divergent_state() -> None:
    suppressed = _context(prompt_state_delivery="suppressed")

    alice = build_system_prompt(assembly=_alice_assembly(), context=suppressed)
    bob = build_system_prompt(assembly=_bob_assembly(), context=suppressed)

    assert alice == bob
    assert "her cat died last week" not in alice
    assert "he starts the new job Monday" not in alice
    assert "Steady Support" not in alice
    assert "grief-sensitive" not in alice
    assert state_prompt_section_count(
        assembly=_alice_assembly(), context=suppressed
    ) == 0


def test_suppressed_chat_messages_share_one_fingerprint() -> None:
    suppressed = _context(prompt_state_delivery="suppressed")

    alice = build_chat_messages(assembly=_alice_assembly(), context=suppressed)
    bob = build_chat_messages(assembly=_bob_assembly(), context=suppressed)

    assert prompt_fingerprint(messages=alice) == prompt_fingerprint(messages=bob)
    assert len(prompt_fingerprint(messages=alice)) == AUDIT_FINGERPRINT_LENGTH


def test_text_and_suppressed_fingerprints_differ() -> None:
    """Suppression must be observable, not a cosmetic flag."""

    assembly = _alice_assembly()
    text_fp = prompt_fingerprint(
        messages=build_chat_messages(assembly=assembly, context=_context())
    )
    suppressed_fp = prompt_fingerprint(
        messages=build_chat_messages(
            assembly=assembly,
            context=_context(prompt_state_delivery="suppressed"),
        )
    )

    assert text_fp != suppressed_fp


# ---------------------------------------------------------------------------
# 3. Per-turn attestation on every LLM turn
# ---------------------------------------------------------------------------


def test_attestation_tags_are_emitted_unconditionally() -> None:
    synthesizer = LLMResponseSynthesizer(runtime=_RecordingRuntime())

    response = synthesizer.synthesize(
        context=_context(), assembly=_alice_assembly()
    )

    assert len(_tag_value(response, "prompt_fp")) == AUDIT_FINGERPRINT_LENGTH
    assert len(_tag_value(response, "decode_fp")) == AUDIT_FINGERPRINT_LENGTH
    assert int(_tag_value(response, "prompt_state_sections")) > 0


def test_pure_arms_agree_on_prompt_while_only_residual_carries_state() -> None:
    """The core carrier-identification setup, in one turn per arm.

    arm A-pure: no conditioning. arm E-pure: conditioning through the
    residual channel. Same prompt fingerprint, same decode fingerprint,
    zero state-derived prompt sections — the only open channel is C3.
    """

    suppressed = "suppressed"
    arm_a_runtime = _RecordingRuntime()
    arm_e_runtime = _RecordingRuntime()

    arm_a = LLMResponseSynthesizer(runtime=arm_a_runtime).synthesize(
        context=_context(prompt_state_delivery=suppressed),
        assembly=_alice_assembly(),
    )
    arm_e = LLMResponseSynthesizer(runtime=arm_e_runtime).synthesize(
        context=_context(
            prompt_state_delivery=suppressed,
            conditioning=_conditioning(fill=0.5),
        ),
        assembly=_alice_assembly(),
    )

    assert _tag_value(arm_a, "prompt_fp") == _tag_value(arm_e, "prompt_fp")
    assert _tag_value(arm_a, "decode_fp") == _tag_value(arm_e, "decode_fp")
    assert _tag_value(arm_a, "prompt_state_sections") == "0"
    assert _tag_value(arm_e, "prompt_state_sections") == "0"

    # C3 open on E only, and audited from the runtime's own report.
    assert arm_a_runtime.calls[0]["personal_conditioning"] is None
    assert arm_e_runtime.calls[0]["personal_conditioning"] is not None
    assert any(t.startswith("personal_conditioning=") for t in arm_e.rationale_tags)


def test_decode_fingerprint_separates_sampling_config_from_state() -> None:
    """C5 must be reportable: differing decode config yields a differing fp."""

    base = _context(prompt_state_delivery="suppressed")
    alice = LLMResponseSynthesizer(runtime=_RecordingRuntime()).synthesize(
        context=base, assembly=_alice_assembly()
    )
    bob = LLMResponseSynthesizer(runtime=_RecordingRuntime()).synthesize(
        context=base, assembly=_bob_assembly()
    )

    # Prompt carrier closed, sampling carrier still open: exactly the
    # situation the spec grades as retain-prompt-closed rather than
    # retain-strict.
    assert _tag_value(alice, "prompt_fp") == _tag_value(bob, "prompt_fp")
    assert _tag_value(alice, "decode_fp") != _tag_value(bob, "decode_fp")


def test_decode_fingerprint_ignores_none_constraints_consistently() -> None:
    assert decode_fingerprint(
        constraints=None, temperature=0.7, max_new_tokens=512
    ) == decode_fingerprint(constraints=None, temperature=0.7, max_new_tokens=512)
    assert decode_fingerprint(
        constraints=None, temperature=0.7, max_new_tokens=512
    ) != decode_fingerprint(constraints=None, temperature=0.2, max_new_tokens=512)


def test_decode_fingerprint_includes_sampling_seed() -> None:
    assert decode_fingerprint(
        constraints=None,
        temperature=0.7,
        max_new_tokens=512,
        sampling_seed=7,
    ) == decode_fingerprint(
        constraints=None,
        temperature=0.7,
        max_new_tokens=512,
        sampling_seed=7,
    )
    assert decode_fingerprint(
        constraints=None,
        temperature=0.7,
        max_new_tokens=512,
        sampling_seed=7,
    ) != decode_fingerprint(
        constraints=None,
        temperature=0.7,
        max_new_tokens=512,
        sampling_seed=8,
    )


def test_sampling_seed_is_forwarded_and_audited() -> None:
    runtime = _RecordingRuntime()

    response = LLMResponseSynthesizer(runtime=runtime).synthesize(
        context=_context(sampling_seed=1234), assembly=_alice_assembly()
    )

    assert runtime.calls[0]["sampling_seed"] == 1234
    assert _tag_value(response, "sampling_seed") == "1234"


# ---------------------------------------------------------------------------
# 4. Fail-loud wiring guards
# ---------------------------------------------------------------------------


def test_response_context_rejects_unknown_prompt_state_delivery() -> None:
    with pytest.raises(ValueError, match="prompt_state_delivery"):
        _context(prompt_state_delivery="latent")


def test_response_context_rejects_statement_under_suppression() -> None:
    with pytest.raises(ValueError, match="suppressed"):
        replace(
            _context(prompt_state_delivery="suppressed"),
            personal_conditioning_statement="Stable, high continuity.",
            personal_conditioning_statement_ref="v1:0.70:abcdef",
        )


def test_response_context_rejects_character_grounding_without_ref() -> None:
    with pytest.raises(ValueError, match="character grounding"):
        _context(character_grounding_statement="我是张无忌。")


def test_response_context_rejects_character_grounding_under_suppression() -> None:
    with pytest.raises(ValueError, match="suppressed"):
        _context(
            prompt_state_delivery="suppressed",
            character_grounding_statement="我是张无忌。",
            character_grounding_ref="character:zhang-wuji:test",
        )


def test_final_rollout_config_rejects_unknown_prompt_state_delivery() -> None:
    with pytest.raises(ValueError, match="prompt_state_delivery"):
        FinalRolloutConfig(prompt_state_delivery="latent")


def test_final_rollout_config_accepts_prefix_kv_mode() -> None:
    config = FinalRolloutConfig(personal_conditioning_mode="prefix_kv")

    assert config.personal_conditioning_mode == "prefix_kv"


def test_final_rollout_config_rejects_unknown_personal_conditioning_mode() -> None:
    with pytest.raises(ValueError, match="personal_conditioning_mode"):
        FinalRolloutConfig(personal_conditioning_mode="soft_prompt")


def test_final_rollout_config_rejects_text_mode_under_suppression() -> None:
    with pytest.raises(ValueError, match="prompt_state_delivery='text'"):
        FinalRolloutConfig(
            personal_conditioning_mode="text",
            prompt_state_delivery="suppressed",
        )


def test_default_config_keeps_the_production_prompt_path() -> None:
    assert FinalRolloutConfig().prompt_state_delivery == "text"
