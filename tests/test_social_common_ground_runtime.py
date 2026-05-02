"""Unit tests for structured common-ground proposal runtime."""

from __future__ import annotations

import pytest

from volvence_zero.social_cognition import SocialScopeKind
from volvence_zero.social_common_ground_runtime import LLMCommonGroundProposalRuntime


class _ScriptedProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        del max_new_tokens, temperature
        self.prompts.append(prompt)
        return self.response


class _ExplodingProvider:
    def generate(self, *, prompt: str, **_: object) -> str:
        raise RuntimeError("common-ground provider failed")


def test_common_ground_runtime_parses_dyad_and_group_atoms() -> None:
    runtime = LLMCommonGroundProposalRuntime(
        provider=_ScriptedProvider(
            """
            [
              {
                "scope_kind": "dyad",
                "scope_id": "self:alice",
                "summary": "We both know Alice wants a slower pace.",
                "accepted_by_ids": ["self", "alice"],
                "evidence": "let's slow down like we agreed",
                "confidence": 0.82,
                "recursion_depth": 2,
                "control_signal": 0.40
              },
              {
                "scope_kind": "group",
                "scope_id": "team:launch",
                "summary": "The launch team accepted the new deadline.",
                "accepted_by_ids": ["alice", "bob", "carol"],
                "evidence": ["all three confirmed"],
                "confidence": 0.78,
                "recursion_depth": 1,
                "control_signal": 0.35
              }
            ]
            """
        )
    )

    batch = runtime.propose(user_input="As we agreed, slow down.", turn_index=1)

    assert len(batch.proposals) == 2
    assert batch.proposals[0].scope_kind is SocialScopeKind.DYAD
    assert batch.proposals[0].accepted_by_ids == ("self", "alice")
    assert batch.proposals[0].evidence == ("let's slow down like we agreed",)
    assert batch.proposals[1].scope_kind is SocialScopeKind.GROUP


def test_common_ground_runtime_drops_low_confidence_and_invalid_scope() -> None:
    runtime = LLMCommonGroundProposalRuntime(
        provider=_ScriptedProvider(
            """
            [
              {
                "scope_kind": "dyad",
                "scope_id": "self:alice",
                "summary": "weak shared claim",
                "accepted_by_ids": ["self", "alice"],
                "evidence": "weak evidence",
                "confidence": 0.30,
                "recursion_depth": 2
              },
              {
                "scope_kind": "global",
                "scope_id": "all",
                "summary": "invalid global claim",
                "accepted_by_ids": ["self"],
                "evidence": "invalid",
                "confidence": 0.80,
                "recursion_depth": 1
              }
            ]
            """
        )
    )

    batch = runtime.propose(user_input="weak", turn_index=2)

    assert batch.proposals == ()


def test_common_ground_runtime_malformed_json_returns_empty_batch() -> None:
    runtime = LLMCommonGroundProposalRuntime(provider=_ScriptedProvider("not json"))

    batch = runtime.propose(user_input="hello", turn_index=3)

    assert batch.proposals == ()


def test_common_ground_runtime_empty_input_does_not_call_provider() -> None:
    provider = _ScriptedProvider("[]")
    runtime = LLMCommonGroundProposalRuntime(provider=provider)

    batch = runtime.propose(user_input="", turn_index=4)

    assert batch.proposals == ()
    assert provider.prompts == []


def test_common_ground_runtime_provider_errors_propagate() -> None:
    runtime = LLMCommonGroundProposalRuntime(provider=_ExplodingProvider())

    with pytest.raises(RuntimeError, match="common-ground provider failed"):
        runtime.propose(user_input="hello", turn_index=5)
