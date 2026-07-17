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


def test_runtime_emits_structured_alignment_evidence() -> None:
    provider = _ScriptedProvider([
        (
            '{"operation": "block", '
            '"alignment_evidence": "I cannot keep this promise now.", '
            '"confidence": 0.82}'
        )
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = runtime.propose(
        target_slot="commitment",
        user_input="I cannot keep this promise now.",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=_previous_with_active(1),
        turn_index=7,
    )

    proposal = batch.proposals[0]
    assert proposal.operation is SemanticProposalOperation.BLOCK
    assert proposal.evidence == "I cannot keep this promise now."
    assert proposal.detail == "I cannot keep this promise now."
    assert proposal.confidence == 0.82
    assert proposal.proposal_id == "commitment:llm-block:7"


def test_runtime_low_confidence_structured_output_becomes_observe() -> None:
    provider = _ScriptedProvider([
        (
            '{"operation": "block", '
            '"alignment_evidence": "ambiguous response", '
            '"confidence": 0.31}'
        )
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = runtime.propose(
        target_slot="commitment",
        user_input="Maybe not sure.",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=_previous_with_active(1),
        turn_index=8,
    )

    proposal = batch.proposals[0]
    assert proposal.operation is SemanticProposalOperation.OBSERVE
    assert proposal.confidence == 0.25
    assert proposal.evidence == "ambiguous response"


def test_runtime_structured_malformed_output_falls_back() -> None:
    provider = _ScriptedProvider(['{"operation": "block", "confidence": 0.8}'])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(runtime, user_input="I cannot commit to this.")

    assert "fell back to base" in batch.description
    assert all(
        proposal.operation is SemanticProposalOperation.OBSERVE
        for proposal in batch.proposals
    )


def test_runtime_routes_unknown_slot_to_base() -> None:
    """Slots outside the nine semantic owners never invoke the LLM.

    G2 widened the generic allowlist to all nine semantic slots, so the
    fail-closed guard now lives at the boundary of the semantic owner
    registry itself: an unknown slot falls through to base untouched.
    """
    provider = _ScriptedProvider([])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(runtime, target_slot="not_a_semantic_slot", user_input="hi")

    assert provider.prompts == [], "unknown slots must NOT call LLM"
    assert batch.runtime_id == NoOpSemanticProposalRuntime.runtime_id
    assert all(
        p.operation is SemanticProposalOperation.OBSERVE for p in batch.proposals
    )


def test_runtime_emits_boundary_consent_typed_proposal_from_schema_payload() -> None:
    provider = _ScriptedProvider([
        (
            '{"runtime_id": "test", "schema_version": 1, "description": "boundary", '
            '"proposals": [{"proposal_id": "ignored", "target_slot": "boundary_consent", '
            '"operation": "block", "summary": "external action denied", '
            '"detail": "User denied external action.", "confidence": 0.77, '
            '"evidence": "Do not act externally", "control_signal": 0.66, '
            '"requires_confirmation": true}]}'
        )
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(
        runtime,
        target_slot="boundary_consent",
        user_input="Do not act externally without asking.",
        turn_index=4,
    )

    proposal = batch.proposals[0]
    assert proposal.target_slot == "boundary_consent"
    assert proposal.operation is SemanticProposalOperation.BLOCK
    assert proposal.proposal_id == "boundary_consent:llm-block:4:0"
    assert proposal.requires_confirmation is True
    assert proposal.control_signal == 0.66
    assert "proposal.schema" in provider.prompts[0] or '"proposals"' in provider.prompts[0]


def test_runtime_emits_goal_value_typed_proposal_from_schema_payload() -> None:
    provider = _ScriptedProvider([
        (
            '{"runtime_id": "test", "schema_version": 1, "description": "goal", '
            '"proposals": [{"proposal_id": "ignored", "target_slot": "goal_value", '
            '"operation": "defer", "summary": "value tradeoff", '
            '"detail": "Goal needs value clarification.", "confidence": 0.71, '
            '"evidence": "I need to think about the tradeoff", "control_signal": 0.52}]}'
        )
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(
        runtime,
        target_slot="goal_value",
        user_input="I need to think about the tradeoff.",
        turn_index=5,
    )

    proposal = batch.proposals[0]
    assert proposal.target_slot == "goal_value"
    assert proposal.operation is SemanticProposalOperation.DEFER
    assert proposal.proposal_id == "goal_value:llm-defer:5:0"
    assert proposal.confidence == 0.71


def test_runtime_emits_relationship_state_typed_proposal_from_schema_payload() -> None:
    """W2-B: relationship_state is now part of the generic LLM slot set.

    Verifies the runtime classifies a relational signal through the
    same JSON schema path that ``boundary_consent`` / ``goal_value``
    use, and that the typed proposal carries the LTV-relevant
    ``observe`` operation with non-zero confidence.
    """
    provider = _ScriptedProvider([
        (
            '{"runtime_id": "test", "schema_version": 1, "description": "rapport", '
            '"proposals": [{"proposal_id": "ignored", "target_slot": "relationship_state", '
            '"operation": "observe", "summary": "user shared parental fatigue", '
            '"detail": "User mentioned exhausting day balancing work and child.", '
            '"confidence": 0.68, '
            '"evidence": "today work and the kid wore me out", "control_signal": 0.32}]}'
        )
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(
        runtime,
        target_slot="relationship_state",
        user_input="Today work and the kid wore me out.",
        turn_index=7,
    )

    assert batch.proposals, "relationship_state generic path must emit a proposal"
    proposal = batch.proposals[0]
    assert proposal.target_slot == "relationship_state"
    assert proposal.operation is SemanticProposalOperation.OBSERVE
    assert proposal.proposal_id == "relationship_state:llm-observe:7:0"
    assert proposal.confidence == 0.68


def test_runtime_emits_user_model_typed_proposal_from_schema_payload() -> None:
    """W2-B: user_model is now part of the generic LLM slot set.

    Verifies a ``create`` operation flows through the schema path so a
    vertical can drive durable user-model updates (preferences /
    archetype tags) from typed LLM extraction without the kernel
    growing a vertical-specific schema.
    """
    provider = _ScriptedProvider([
        (
            '{"runtime_id": "test", "schema_version": 1, "description": "user-model", '
            '"proposals": [{"proposal_id": "ignored", "target_slot": "user_model", '
            '"operation": "create", "summary": "user prefers concise category-level advice", '
            '"detail": "User asked for direct answers without deep clinical detail.", '
            '"confidence": 0.74, '
            '"evidence": "just tell me what category to look at", "control_signal": 0.40}]}'
        )
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(
        runtime,
        target_slot="user_model",
        user_input="Just tell me what category to look at.",
        turn_index=8,
    )

    assert batch.proposals, "user_model generic path must emit a proposal"
    proposal = batch.proposals[0]
    assert proposal.target_slot == "user_model"
    assert proposal.operation is SemanticProposalOperation.CREATE
    assert proposal.proposal_id == "user_model:llm-create:8:0"
    assert proposal.confidence == 0.74


@pytest.mark.parametrize(
    ("slot", "operation", "expected_op"),
    [
        ("plan_intent", "create", SemanticProposalOperation.CREATE),
        ("open_loop", "observe", SemanticProposalOperation.OBSERVE),
        ("execution_result", "complete", SemanticProposalOperation.COMPLETE),
        ("belief_assumption", "create", SemanticProposalOperation.CREATE),
    ],
)
def test_runtime_emits_typed_proposals_for_remaining_four_slots(
    slot: str, operation: str, expected_op: SemanticProposalOperation
) -> None:
    """G2 (P1-3): the last four semantic owners join the generic LLM path.

    Every one of the nine semantic slots now has a typed LLM proposal
    source when an LLM runtime is wired; owners keep single-writer
    semantics and their own confidence filters.
    """
    provider = _ScriptedProvider([
        (
            '{"runtime_id": "test", "schema_version": 1, "description": "g2", '
            f'"proposals": [{{"proposal_id": "ignored", "target_slot": "{slot}", '
            f'"operation": "{operation}", "summary": "typed {slot} signal", '
            f'"detail": "Typed proposal for {slot}.", "confidence": 0.7, '
            '"evidence": "typed evidence", "control_signal": 0.4}]}'
        )
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(
        runtime,
        target_slot=slot,
        user_input="I finished planning what to do next week.",
        turn_index=9,
    )

    assert batch.proposals, f"{slot} generic path must emit a proposal"
    proposal = batch.proposals[0]
    assert proposal.target_slot == slot
    assert proposal.operation is expected_op
    assert proposal.proposal_id == f"{slot}:llm-{operation}:9:0"
    assert f"Target owner slot: {slot}" in provider.prompts[0]
    assert "Slot semantics:" in provider.prompts[0]


def test_generic_prompt_for_original_slots_has_no_hint_line() -> None:
    """The original four generic slots keep their prompt byte-shape."""
    provider = _ScriptedProvider([
        '{"runtime_id": "test", "schema_version": 1, "description": "d", "proposals": []}'
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    _propose(
        runtime,
        target_slot="goal_value",
        user_input="thinking about goals",
        turn_index=1,
    )

    assert "Slot semantics:" not in provider.prompts[0]


def test_runtime_generic_payload_target_mismatch_falls_back() -> None:
    provider = _ScriptedProvider([
        (
            '{"runtime_id": "test", "schema_version": 1, "description": "bad", '
            '"proposals": [{"proposal_id": "ignored", "target_slot": "commitment", '
            '"operation": "create", "summary": "wrong target", '
            '"detail": "Wrong target slot.", "confidence": 0.90, '
            '"evidence": "wrong target"}]}'
        )
    ])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = _propose(
        runtime,
        target_slot="goal_value",
        user_input="I have a goal.",
        turn_index=6,
    )

    assert "fell back to base" in batch.description
    assert all(
        proposal.operation is SemanticProposalOperation.OBSERVE
        for proposal in batch.proposals
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
    """Full label coverage with an active commitment present.

    The lifecycle-precondition guard (BLOCK / COMPLETE / DEFER need
    an active commitment to act on) is exercised by separate tests;
    here we want to confirm every LLM-emitted label maps to the
    right operation given the precondition holds.
    """
    provider = _ScriptedProvider(["complete", "block", "defer", "observe"])
    runtime = LLMSemanticProposalRuntime(provider=provider)
    active = _previous_with_active(1)

    def _with_active(text: str, turn_index: int):
        return runtime.propose(
            target_slot="commitment",
            user_input=text,
            substrate_snapshot=None,
            memory_snapshot=None,
            previous_snapshot=active,
            turn_index=turn_index,
        )

    batch_complete = _with_active("I finished the chapter today.", 0)
    batch_block = _with_active("I can't keep going right now.", 1)
    batch_defer = _with_active("Let's move it to next Monday.", 2)
    batch_observe = _with_active("Tell me about the weather.", 3)

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


def _previous_with_active(count: int):
    """A real ``CommitmentSnapshot`` with ``count`` active commitments.

    ``_has_active_commitment`` dispatches via ``isinstance`` (R8 typed
    dispatch), so a duck-typed stand-in no longer satisfies the guard.
    """
    from volvence_zero.semantic_state import CommitmentSnapshot, SemanticRecord

    records = tuple(
        SemanticRecord(
            record_id=f"commit-{index}",
            summary=f"active commitment {index}",
            detail="test fixture",
            confidence=0.8,
            status="active",
            source_turn=0,
            evidence="fixture",
        )
        for index in range(count)
    )
    return CommitmentSnapshot(
        active_commitments=records,
        honored_commitment_refs=(),
        at_risk_commitments=(),
        trust_obligation_count=0,
        continuity_score=0.5,
        control_signal=0.0,
        description="fixture commitment snapshot",
    )


@pytest.mark.parametrize(
    "label", ["block", "complete", "defer"]
)
def test_runtime_routes_non_create_to_observe_when_no_active_commitment(
    label: str,
) -> None:
    """Structural guard: BLOCK / COMPLETE / DEFER without a target -> OBSERVE.

    The 0.5B Qwen probe showed the model over-classifies neutral
    turns as ``block``; the guard refuses to apply lifecycle
    operations to a commitment list that's empty, so the proposal
    falls back to a confidence-0.25 OBSERVE that the
    ``CommitmentModule.min_proposal_confidence=0.40`` filter then
    drops at the owner layer. Defence in depth.
    """
    provider = _ScriptedProvider([label])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = runtime.propose(
        target_slot="commitment",
        user_input="something the LLM mistakes for a block",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=0,
    )

    assert len(batch.proposals) == 1
    assert batch.proposals[0].operation is SemanticProposalOperation.OBSERVE


@pytest.mark.parametrize("label", ["block", "complete", "defer"])
def test_runtime_passes_through_non_create_when_active_commitment_exists(
    label: str,
) -> None:
    """With an active commitment, the LLM's lifecycle classification stands."""
    provider = _ScriptedProvider([label])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = runtime.propose(
        target_slot="commitment",
        user_input="follow-up to a real commitment",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=_previous_with_active(1),
        turn_index=0,
    )

    expected = {
        "block": SemanticProposalOperation.BLOCK,
        "complete": SemanticProposalOperation.COMPLETE,
        "defer": SemanticProposalOperation.DEFER,
    }[label]
    assert batch.proposals[0].operation is expected


def test_runtime_create_always_passes_through_regardless_of_state() -> None:
    """CREATE has no precondition: making a new commitment doesn't need an existing one."""
    provider = _ScriptedProvider(["create"])
    runtime = LLMSemanticProposalRuntime(provider=provider)

    batch = runtime.propose(
        target_slot="commitment",
        user_input="I will start a new habit.",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=0,
    )

    assert batch.proposals[0].operation is SemanticProposalOperation.CREATE


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
