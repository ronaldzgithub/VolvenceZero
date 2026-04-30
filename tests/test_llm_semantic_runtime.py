"""Unit tests for ``LLMSemanticProposalRuntime``.

We exercise the runtime against a deterministic in-memory provider
so the behaviour we verify is the runtime's own logic (slot routing,
label parsing, fallback, payload shape), not transformers / Qwen
inference quality. Real-substrate verification lives in the
end-to-end demo + verify scripts.
"""

from __future__ import annotations

import pytest

from volvence_zero.semantic_state import (
    NoOpSemanticProposalRuntime,
    SemanticProposalBatch,
    SemanticProposalOperation,
)
from volvence_zero.semantic_state.llm_runtime import (
    LLMSemanticProposalRuntime,
    _parse_commitment_label,
)


class _ScriptedProvider:
    """Returns successive scripted responses; records every prompt."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def generate(
        self, *, prompt: str, max_new_tokens: int = 8, temperature: float = 0.0
    ) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(
            {"max_new_tokens": max_new_tokens, "temperature": temperature}
        )
        if not self._responses:
            raise RuntimeError("Scripted provider exhausted")
        return self._responses.pop(0)


class _ExplodingProvider:
    def generate(self, *, prompt: str, **_: object) -> str:
        raise RuntimeError("simulated LLM outage")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("create", SemanticProposalOperation.CREATE),
        ("Create", SemanticProposalOperation.CREATE),
        ("CREATE.", SemanticProposalOperation.CREATE),
        ("complete\n", SemanticProposalOperation.COMPLETE),
        ("  block  ", SemanticProposalOperation.BLOCK),
        ("defer because of work", SemanticProposalOperation.DEFER),
        ("observe", SemanticProposalOperation.OBSERVE),
    ],
)
def test_parse_commitment_label_accepts_variants(raw, expected) -> None:
    assert _parse_commitment_label(raw) is expected


@pytest.mark.parametrize("raw", ["", "  ", "I'm not sure", "?", "completed"])
def test_parse_commitment_label_rejects_invalid(raw) -> None:
    """Anything other than the five labels falls through.

    Note: ``completed`` (past tense) is intentionally rejected. The
    LLM was asked for ``complete``; if it improvises we drop to
    fallback rather than guessing what it meant. That preserves the
    "no keyword inference over text" red line.
    """
    assert _parse_commitment_label(raw) is None


def _propose(runtime, *, target_slot="commitment", user_input="text", turn_index=0):
    return runtime.propose(
        target_slot=target_slot,
        user_input=user_input,
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=turn_index,
    )


def test_runtime_emits_create_for_create_label() -> None:
    provider = _ScriptedProvider(["create"])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(runtime, user_input="I will study for an hour tomorrow.")

    assert isinstance(batch, SemanticProposalBatch)
    assert batch.runtime_id == "semantic-llm-commitment"
    assert len(batch.proposals) == 1
    proposal = batch.proposals[0]
    assert proposal.operation is SemanticProposalOperation.CREATE
    assert proposal.target_slot == "commitment"
    assert proposal.confidence > 0.0
    assert proposal.evidence.startswith("I will")
    assert "study" in proposal.evidence
    assert provider.kwargs[0]["temperature"] == 0.0


def test_runtime_routes_non_commitment_to_base() -> None:
    provider = _ScriptedProvider([])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(runtime, target_slot="open_loop", user_input="hi")

    assert provider.prompts == [], "non-commitment slots must NOT call LLM"
    assert batch.runtime_id == NoOpSemanticProposalRuntime.runtime_id
    assert all(
        p.operation is SemanticProposalOperation.OBSERVE for p in batch.proposals
    )


def test_runtime_skips_llm_on_empty_input() -> None:
    provider = _ScriptedProvider([])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(runtime, user_input="")

    assert provider.prompts == []
    assert batch.runtime_id == NoOpSemanticProposalRuntime.runtime_id


def test_runtime_falls_back_on_unparseable_label() -> None:
    provider = _ScriptedProvider(["I think the user is excited."])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(runtime, user_input="hello there")

    assert provider.prompts, "fallback path still consumes one LLM call"
    assert batch.runtime_id == "semantic-llm-commitment"
    assert "fell back to base" in batch.description
    assert all(
        p.operation is SemanticProposalOperation.OBSERVE for p in batch.proposals
    )


def test_runtime_propagates_provider_exceptions() -> None:
    """LLM provider errors should bubble up.

    The kernel-level orchestrator is responsible for converting an
    exception into a per-turn fault entry. Swallowing the error here
    would create a silent "second owner" of failure handling \u2014
    explicitly forbidden by the SSOT rules.
    """
    runtime = LLMSemanticProposalRuntime(provider=_ExplodingProvider())

    with pytest.raises(RuntimeError, match="simulated LLM outage"):
        _propose(runtime, user_input="I will exercise tomorrow.")


def test_runtime_classifies_each_label_correctly() -> None:
    provider = _ScriptedProvider(["complete", "block", "defer", "observe"])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch_complete = _propose(
        runtime, user_input="I finished the chapter today.", turn_index=0
    )
    batch_block = _propose(
        runtime, user_input="I can't keep going right now.", turn_index=1
    )
    batch_defer = _propose(
        runtime, user_input="Let's move it to next Monday.", turn_index=2
    )
    batch_observe = _propose(
        runtime, user_input="Tell me about the weather.", turn_index=3
    )

    assert (
        batch_complete.proposals[0].operation is SemanticProposalOperation.COMPLETE
    )
    assert batch_block.proposals[0].operation is SemanticProposalOperation.BLOCK
    assert batch_defer.proposals[0].operation is SemanticProposalOperation.DEFER
    assert batch_observe.proposals[0].operation is SemanticProposalOperation.OBSERVE
    assert {p.proposal_id for batch in (
        batch_complete, batch_block, batch_defer, batch_observe
    ) for p in batch.proposals} == {
        "commitment:llm-complete:0",
        "commitment:llm-block:1",
        "commitment:llm-defer:2",
        "commitment:llm-observe:3",
    }


def test_runtime_clamps_user_input_to_safe_length() -> None:
    """Ensure runaway long inputs don't blow up the prompt size.

    We include a 5k-char input and assert the rendered prompt is
    bounded. The clamp is a defence against an attacker pasting
    massive content; semantic event extraction does NOT need full
    document context.
    """
    provider = _ScriptedProvider(["observe"])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    long_input = "X" * 5000
    _propose(runtime, user_input=long_input)

    prompt = provider.prompts[0]
    assert prompt.count("X") <= 600
