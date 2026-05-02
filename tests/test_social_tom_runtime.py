"""Unit tests for structured Theory-of-Mind proposal runtime."""

from __future__ import annotations

import pytest

from volvence_zero.semantic_state import (
    NoOpSemanticProposalRuntime,
    SemanticProposalOperation,
)
from volvence_zero.social import LLMToMProposalRuntime


class _ScriptedProvider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, object]] = []

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(
            {"max_new_tokens": max_new_tokens, "temperature": temperature}
        )
        if not self._responses:
            raise RuntimeError("scripted provider exhausted")
        return self._responses.pop(0)


class _ExplodingProvider:
    def generate(self, *, prompt: str, **_: object) -> str:
        raise RuntimeError("tom provider failed")


def _propose(
    runtime: LLMToMProposalRuntime,
    *,
    target_slot: str = "belief_about_other",
    user_input: str = "Alice believes the meeting is tomorrow.",
    turn_index: int = 3,
):
    return runtime.propose(
        target_slot=target_slot,
        user_input=user_input,
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=turn_index,
    )


def test_structured_tom_runtime_emits_only_requested_owner_records() -> None:
    provider = _ScriptedProvider(
        [
            """
            [
              {
                "target_slot": "belief_about_other",
                "summary": "Alice believes the meeting is tomorrow.",
                "detail": "Alice explicitly says the meeting is tomorrow.",
                "evidence": "meeting is tomorrow",
                "confidence": 0.86,
                "control_signal": 0.41
              },
              {
                "target_slot": "preference_about_other",
                "summary": "Alice prefers slow planning.",
                "detail": "Alice asks for slow planning.",
                "evidence": "slow planning",
                "confidence": 0.82,
                "control_signal": 0.31
              }
            ]
            """
        ]
    )
    runtime = LLMToMProposalRuntime(provider=provider)

    belief_batch = _propose(runtime, target_slot="belief_about_other")
    preference_batch = _propose(runtime, target_slot="preference_about_other")

    assert len(provider.prompts) == 1
    assert len(belief_batch.proposals) == 1
    assert belief_batch.proposals[0].target_slot == "belief_about_other"
    assert belief_batch.proposals[0].operation is SemanticProposalOperation.OBSERVE
    assert belief_batch.proposals[0].confidence == 0.86
    assert belief_batch.proposals[0].evidence == "meeting is tomorrow"
    assert len(preference_batch.proposals) == 1
    assert preference_batch.proposals[0].target_slot == "preference_about_other"


def test_structured_tom_runtime_filters_low_confidence_records() -> None:
    provider = _ScriptedProvider(
        [
            """
            [
              {
                "target_slot": "belief_about_other",
                "summary": "weak claim",
                "detail": "weak detail",
                "evidence": "weak evidence",
                "confidence": 0.34,
                "control_signal": 0.20
              }
            ]
            """
        ]
    )
    runtime = LLMToMProposalRuntime(provider=provider)

    batch = _propose(runtime)

    assert batch.proposals == ()


def test_structured_tom_runtime_malformed_json_falls_back() -> None:
    provider = _ScriptedProvider(["not json"])
    runtime = LLMToMProposalRuntime(provider=provider)

    batch = _propose(runtime)

    assert batch.runtime_id == LLMToMProposalRuntime.runtime_id
    assert batch.proposals == ()


def test_structured_tom_runtime_non_tom_target_delegates_to_base() -> None:
    provider = _ScriptedProvider([])
    runtime = LLMToMProposalRuntime(provider=provider)

    batch = _propose(runtime, target_slot="commitment")

    assert provider.prompts == []
    assert batch.runtime_id == NoOpSemanticProposalRuntime.runtime_id


def test_structured_tom_runtime_provider_errors_propagate() -> None:
    runtime = LLMToMProposalRuntime(provider=_ExplodingProvider())

    with pytest.raises(RuntimeError, match="tom provider failed"):
        _propose(runtime)
