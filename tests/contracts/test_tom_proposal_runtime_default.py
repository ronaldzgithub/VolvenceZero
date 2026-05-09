"""Phase 1 W1.C contract test: ToM proposal runtime fail-closed default.

When ``build_final_runtime_modules`` is called WITHOUT an explicit
``tom_proposal_runtime`` parameter, two paths must hold:

1. **NoOp / scripted semantic runtime**: ``tom_proposal_runtime``
   stays ``None`` and the four ToM owners (``belief_about_other`` /
   ``intent_about_other`` / ``feeling_about_other`` /
   ``preference_about_other``) publish empty-records snapshots.
   This preserves the back-compat surface for unit tests and offline
   harnesses that never wire an LLM.
2. **LLM-backed semantic runtime**: ``build_final_runtime_modules``
   default-constructs an ``LLMToMProposalRuntime`` sharing the same
   text provider and the four ToM owners can publish typed records.
   This is the fail-closed default opt-in: the kernel does not
   "find" an LLM out of nowhere, but if one is already wired upstream
   for ``semantic_proposal_runtime`` it gets re-used here.

The test pins the typed accessor that lets the kernel share the
provider (``LLMSemanticProposalRuntime.text_provider``) and the
isinstance check that decides default construction.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    NoOpSemanticProposalRuntime,
    SemanticProposalRuntime,
)
from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    FeelingAboutOtherSnapshot,
    IntentAboutOtherSnapshot,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.substrate import FeatureSignal, FeatureSurfaceSubstrateAdapter


class _FakeTextProvider:
    """Deterministic text provider that emits a small ToM JSON payload.

    The payload lists one ToM record per slot so we can verify that
    the default-constructed ``LLMToMProposalRuntime`` does propagate
    typed records through to all four owners.
    """

    _PAYLOAD = (
        "["
        '{"target_slot": "belief_about_other", '
        '"summary": "user thinks meeting is tomorrow", '
        '"detail": "user mentioned meeting time", '
        '"evidence": "meeting tomorrow", '
        '"confidence": 0.8, '
        '"control_signal": 0.3},'
        '{"target_slot": "intent_about_other", '
        '"summary": "user wants brief reply", '
        '"detail": "user asked to be quick", '
        '"evidence": "be quick", '
        '"confidence": 0.7, '
        '"control_signal": 0.25},'
        '{"target_slot": "feeling_about_other", '
        '"summary": "user feels tired", '
        '"detail": "user mentioned long day", '
        '"evidence": "long day", '
        '"confidence": 0.75, '
        '"control_signal": 0.4},'
        '{"target_slot": "preference_about_other", '
        '"summary": "user prefers short summaries", '
        '"detail": "user prefers terse replies", '
        '"evidence": "short summaries", '
        '"confidence": 0.7, '
        '"control_signal": 0.3}'
        "]"
    )

    def generate(
        self, *, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0
    ) -> str:
        del prompt, max_new_tokens, temperature
        return self._PAYLOAD


def _substrate() -> FeatureSurfaceSubstrateAdapter:
    return FeatureSurfaceSubstrateAdapter(
        model_id="tom-default-runtime-model",
        feature_surface=(
            FeatureSignal(
                name="tom_default_context",
                values=(0.5,),
                source="adapter",
            ),
        ),
    )


def _run_with_active_tom(
    *,
    semantic_proposal_runtime,
    user_input: str,
) -> dict[str, object]:
    config = FinalRolloutConfig(
        belief_about_other=WiringLevel.ACTIVE,
        intent_about_other=WiringLevel.ACTIVE,
        feeling_about_other=WiringLevel.ACTIVE,
        preference_about_other=WiringLevel.ACTIVE,
    )
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=_substrate(),
            user_input=user_input,
            semantic_proposal_runtime=semantic_proposal_runtime,
            session_id="tom-default-runtime",
            wave_id="wave-tom-default",
            turn_index=1,
        )
    )
    return {
        "belief": result.active_snapshots["belief_about_other"].value,
        "intent": result.active_snapshots["intent_about_other"].value,
        "feeling": result.active_snapshots["feeling_about_other"].value,
        "preference": result.active_snapshots["preference_about_other"].value,
    }


def test_text_provider_accessor_is_public_and_typed() -> None:
    """The kernel relies on this accessor to share providers across
    LLM-driven runtimes; this test pins the public surface.
    """
    provider = _FakeTextProvider()
    runtime = LLMSemanticProposalRuntime(provider=provider)
    assert runtime.text_provider is provider


def test_noop_semantic_runtime_keeps_tom_records_empty() -> None:
    """Fail-closed back-compat: NoOp semantic runtime -> no default
    ToM runtime -> ToM owners publish empty-records snapshots.
    """
    snaps = _run_with_active_tom(
        semantic_proposal_runtime=NoOpSemanticProposalRuntime(),
        user_input="alice prefers short summaries; meeting tomorrow",
    )
    for slot, value in snaps.items():
        assert value.records == (), (
            f"NoOp semantic runtime must NOT default-wire a ToM "
            f"runtime; {slot} should have empty records but got "
            f"{len(value.records)}"
        )


def test_llm_semantic_runtime_default_wires_tom_runtime() -> None:
    """Fail-closed opt-in: LLM-backed semantic runtime exposes a
    ``text_provider`` so ``build_final_runtime_modules`` can share
    it for a default ``LLMToMProposalRuntime``. The four ToM owners
    must each publish at least one typed record on this turn.
    """
    semantic_runtime = LLMSemanticProposalRuntime(provider=_FakeTextProvider())
    snaps = _run_with_active_tom(
        semantic_proposal_runtime=semantic_runtime,
        user_input="alice prefers short summaries; meeting tomorrow",
    )

    assert isinstance(snaps["belief"], BeliefAboutOtherSnapshot)
    assert isinstance(snaps["intent"], IntentAboutOtherSnapshot)
    assert isinstance(snaps["feeling"], FeelingAboutOtherSnapshot)
    assert isinstance(snaps["preference"], PreferenceAboutOtherSnapshot)

    record_counts = {slot: len(value.records) for slot, value in snaps.items()}
    for slot, count in record_counts.items():
        assert count >= 1, (
            f"LLM-backed semantic runtime should default-wire a ToM "
            f"runtime that produces records for every slot; "
            f"{slot} got {count} records (counts={record_counts})"
        )


class _LLMRuntimeWrapperLookalike(SemanticProposalRuntime):
    """A wrapper that exposes ``text_provider`` but is NOT an
    ``LLMSemanticProposalRuntime`` instance.

    Wave E1 (debt #10B item 3 hypothesis 2): if a user-supplied
    decorator wraps the LLM runtime to add e.g. tracing or guardrail
    checks, the strict ``isinstance`` check in
    ``build_final_runtime_modules`` will silently skip auto-wiring
    the ToM / common-ground proposal runtimes, producing 0 records
    even though a real LLM is plugged in. This stub reproduces that
    pattern so we can pin the diagnostic warning.
    """

    runtime_id = "wrapper-lookalike"

    def __init__(self, *, inner: LLMSemanticProposalRuntime) -> None:
        self._inner = inner

    @property
    def text_provider(self):  # type: ignore[no-untyped-def]
        return self._inner.text_provider

    def propose(self, **kwargs):  # type: ignore[no-untyped-def]
        return self._inner.propose(**kwargs)


def test_wrapper_around_llm_runtime_emits_warning_and_keeps_fail_closed() -> None:
    """A wrapper that hides the LLM runtime under ``isinstance`` check
    must surface a one-shot warning so the silent 0-records situation
    is diagnosable. The wiring stays fail-closed (no default ToM /
    common-ground runtime auto-wired).
    """
    inner = LLMSemanticProposalRuntime(provider=_FakeTextProvider())
    wrapper = _LLMRuntimeWrapperLookalike(inner=inner)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        snaps = _run_with_active_tom(
            semantic_proposal_runtime=wrapper,
            user_input="alice prefers short summaries; meeting tomorrow",
        )
    # Records stay empty (fail-closed default preserved).
    for slot, value in snaps.items():
        assert value.records == (), (
            f"Wrapper around LLM runtime must keep ToM owners empty by "
            f"default (no auto-wire); {slot} got {len(value.records)} "
            "records."
        )
    messages = [
        str(record.message)
        for record in caught
        if issubclass(record.category, UserWarning)
    ]
    tom_warnings = [m for m in messages if "ToM proposal runtime" in m]
    cg_warnings = [m for m in messages if "common-ground proposal runtime" in m]
    assert tom_warnings, (
        "Wrapper case should emit a UserWarning about ToM auto-wire "
        f"FAIL-CLOSED (got messages={messages})"
    )
    assert cg_warnings, (
        "Wrapper case should emit a UserWarning about common-ground "
        f"auto-wire FAIL-CLOSED (got messages={messages})"
    )
