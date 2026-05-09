"""Contract test: LLM proposal runtimes persist across all turns of a session.

Option B / debt #10B item 3 follow-up: ``LLMToMProposalRuntime`` and
``LLMCommonGroundProposalRuntime`` were re-constructed every turn
inside ``build_final_runtime_modules`` when the caller passed ``None``.
That reset their typed ``LLMProposalAttemptAccumulator`` every turn, so
``per_round_*_proposal_attempts_total`` capped at ``[1]`` even after a
5-turn session and the cumulative ``parsed_ok`` counter never reflected
session-level activation.

This test pins the post-fix invariant: ``AgentSessionRunner.__init__``
constructs both runtimes ONCE from the unwrapped semantic runtime when
that runtime is an ``LLMSemanticProposalRuntime`` instance, exposes
them as ``_tom_proposal_runtime`` / ``_common_ground_proposal_runtime``,
and threads them through every ``run_final_wiring_turn`` call so the
counters accumulate.

We don't drive the full ``run_turn`` (heavy + needs substrate model);
we assert the session-construction invariants and verify counter
accumulation by simulating multi-turn ``propose()`` calls directly on
the persisted runtime instances. Counter accumulation across multiple
``propose()`` calls is the exact behaviour the per-turn-rebuild bug
was breaking.
"""

from __future__ import annotations

from volvence_zero.agent import AgentSessionRunner
from volvence_zero.semantic_state import NoOpSemanticProposalRuntime
from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
from volvence_zero.social import (
    LLMCommonGroundProposalRuntime,
    LLMToMProposalRuntime,
)
from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime


_VALID_TOM_PAYLOAD = (
    "["
    '{"target_slot": "feeling_about_other", '
    '"summary": "user is tired", '
    '"detail": "expressed exhaustion in turn", '
    '"evidence": "I am so tired", '
    '"confidence": 0.7, '
    '"control_signal": 0.3}'
    "]"
)

_VALID_CG_PAYLOAD = (
    "["
    "{"
    '"scope_kind": "dyad", '
    '"scope_id": "user_and_ai", '
    '"summary": "shared topic acknowledged", '
    '"accepted_by_ids": ["user", "ai"], '
    '"evidence": "yes that works", '
    '"confidence": 0.8, '
    '"recursion_depth": 1, '
    '"control_signal": 0.4'
    "}"
    "]"
)


class _CountingProvider:
    """Text provider that counts every ``generate`` call.

    Returns a different valid payload depending on the prompt content
    so both ToM and common-ground parsers see a parseable response.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self, *, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0
    ) -> str:
        del max_new_tokens, temperature
        self.call_count += 1
        # Both prompts share the "Return a JSON array" wording but
        # only the common-ground prompt mentions "common-ground" /
        # "scope_kind" — route on a stable substring.
        if "scope_kind" in prompt:
            return _VALID_CG_PAYLOAD
        return _VALID_TOM_PAYLOAD


def _build_runner(
    *, semantic_proposal_runtime: object | None
) -> AgentSessionRunner:
    """Build a synthetic-substrate runner for fast init in tests.

    The synthetic residual runtime sidesteps the ~30s distilgpt2 load
    (the default ``substrate_model_id``); session construction is
    otherwise unchanged.
    """
    return AgentSessionRunner(
        session_id="persistence-test",
        semantic_proposal_runtime=semantic_proposal_runtime,  # type: ignore[arg-type]
        default_residual_runtime=SyntheticOpenWeightResidualRuntime(
            model_id="persistence-test-runtime"
        ),
    )


def test_session_runner_constructs_persistent_tom_and_cg_runtimes_for_llm_semantic() -> None:
    """When wired with an ``LLMSemanticProposalRuntime``, the runner
    creates exactly one ``LLMToMProposalRuntime`` and one
    ``LLMCommonGroundProposalRuntime`` and keeps them as instance
    attributes so they survive every per-turn ``run_final_wiring_turn``.
    """
    provider = _CountingProvider()
    semantic_runtime = LLMSemanticProposalRuntime(provider=provider)
    runner = _build_runner(semantic_proposal_runtime=semantic_runtime)

    assert isinstance(runner._tom_proposal_runtime, LLMToMProposalRuntime), (
        f"runner should auto-derive an LLMToMProposalRuntime from "
        f"LLMSemanticProposalRuntime; got {type(runner._tom_proposal_runtime).__name__}"
    )
    assert isinstance(
        runner._common_ground_proposal_runtime, LLMCommonGroundProposalRuntime
    ), (
        f"runner should auto-derive an LLMCommonGroundProposalRuntime "
        f"from LLMSemanticProposalRuntime; got "
        f"{type(runner._common_ground_proposal_runtime).__name__}"
    )

    # Identity is preserved across attribute reads (the session keeps
    # ONE instance per runtime kind; per-turn callers re-bind to the
    # same object so the LLMProposalAttemptAccumulator accumulates).
    assert runner._tom_proposal_runtime is runner._tom_proposal_runtime
    assert (
        runner._common_ground_proposal_runtime is runner._common_ground_proposal_runtime
    )


def test_session_runner_keeps_both_runtimes_none_for_noop_semantic() -> None:
    """Fail-closed back-compat: NoOp semantic runtime must NOT
    auto-wire LLM-backed ToM / common-ground runtimes. Existing unit
    tests and offline harnesses depend on this stay-empty default.
    """
    runner = _build_runner(semantic_proposal_runtime=NoOpSemanticProposalRuntime())

    assert runner._tom_proposal_runtime is None
    assert runner._common_ground_proposal_runtime is None


def test_persisted_tom_runtime_accumulates_counters_across_simulated_turns() -> None:
    """Three ``propose()`` calls on the SAME persisted runtime instance
    must produce ``proposals_received_total == 3``. Combined with the
    identity-preservation assertion above, this proves that when
    ``run_final_wiring_turn`` reuses ``runner._tom_proposal_runtime``
    each turn, the counter naturally accumulates over the whole
    session (instead of resetting every turn under the old
    per-turn-rebuild path).
    """
    provider = _CountingProvider()
    semantic_runtime = LLMSemanticProposalRuntime(provider=provider)
    runner = _build_runner(semantic_proposal_runtime=semantic_runtime)
    tom_runtime = runner._tom_proposal_runtime
    assert tom_runtime is not None  # narrow for type-checker / fail-loud

    initial = tom_runtime.attempt_counters
    assert initial.proposals_received_total == 0
    assert initial.last_parse_status == "no_call"

    for turn_index in (1, 2, 3):
        tom_runtime.propose(
            target_slot="feeling_about_other",
            user_input=f"turn {turn_index} about feeling tired",
            substrate_snapshot=None,
            memory_snapshot=None,
            previous_snapshot=None,
            turn_index=turn_index,
        )

    counters = tom_runtime.attempt_counters
    assert counters.proposals_received_total == 3, (
        f"persisted runtime should accumulate 3 calls across simulated "
        f"turns; got {counters.proposals_received_total}"
    )
    assert counters.proposals_parsed_ok == 3
    assert counters.proposals_emitted_total == 3
    assert counters.last_parse_status == "ok"


def test_persisted_common_ground_runtime_accumulates_counters_across_simulated_turns() -> None:
    """Same shape against the common-ground LLM runtime: identity is
    preserved, counter accumulates over multiple ``propose()`` calls.
    """
    provider = _CountingProvider()
    semantic_runtime = LLMSemanticProposalRuntime(provider=provider)
    runner = _build_runner(semantic_proposal_runtime=semantic_runtime)
    cg_runtime = runner._common_ground_proposal_runtime
    assert cg_runtime is not None

    for turn_index in (1, 2, 3, 4):
        cg_runtime.propose(
            user_input=f"turn {turn_index} we agreed to slow down",
            turn_index=turn_index,
        )

    counters = cg_runtime.attempt_counters
    assert counters.proposals_received_total == 4, (
        f"persisted common-ground runtime should accumulate 4 calls "
        f"across simulated turns; got {counters.proposals_received_total}"
    )
    assert counters.proposals_parsed_ok == 4
    assert counters.proposals_emitted_total == 4
    assert counters.last_parse_status == "ok"
