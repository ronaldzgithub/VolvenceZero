"""Contract tests for LLM proposal runtime diagnostic counters.

Wave E1 (debt #10B item 3) wired typed counters into the three
LLM-backed proposal runtimes so a 0-records evidence run can be
diagnosed without enabling the env-gated JSONL sink. These tests
verify:

* The shared :class:`LLMProposalAttemptCounters` contract enforces
  its invariants (non-negative, terminal-status sum bounded by
  receive total, emit-without-parse-success guard).
* Each LLM-backed runtime advances counters correctly through the
  three terminal parse statuses (``ok`` / ``parse_error`` /
  ``empty_or_rejected``) using a fake provider.
* The counters surface on the four ToM owner snapshots and on the
  ``CommonGroundSnapshot`` via the new ``proposal_diagnostics``
  field, so a longitudinal probe can read evidence directly off
  the published snapshots.

We deliberately use a fake provider here so the contract is
verifiable without a real LLM. The Qwen 1.5B real-trace probe is
a separate evidence run (see
``examples/run_cross_session_probe_llm.py``) and is not pulled in
by these contract tests.
"""

from __future__ import annotations

import asyncio
import pytest

from volvence_zero.llm_proposal_diagnostics import LLMProposalAttemptCounters
from volvence_zero.runtime import (
    Snapshot,
    propagate,
    SlotRegistry,
    EventRecorder,
    WiringLevel,
)
from volvence_zero.semantic_state._llm_proposal_counters import (
    LLMProposalAttemptAccumulator,
)
from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
from volvence_zero.social.common_ground import (
    CommonGroundModule,
    LLMCommonGroundProposalRuntime,
)
from volvence_zero.social.tom import (
    BeliefAboutOtherModule,
    FeelingAboutOtherModule,
    IntentAboutOtherModule,
    LLMToMProposalRuntime,
    PreferenceAboutOtherModule,
)


class _FakeProvider:
    """Provider stub returning a fixed payload per call.

    A list of payloads supports a single test orchestrating multiple
    LLM calls (for example: first call returns malformed JSON, second
    returns valid output) so we can assert counter transitions without
    a real model.
    """

    def __init__(self, payloads: tuple[str, ...]) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
    ) -> str:
        del prompt, max_new_tokens, temperature
        if not self._payloads:
            raise RuntimeError("FakeProvider: no payloads left.")
        payload = self._payloads.pop(0)
        self.calls += 1
        return payload


# ---------------------------------------------------------------------
# Pure dataclass contract
# ---------------------------------------------------------------------


def test_empty_counters_have_no_call_status() -> None:
    counters = LLMProposalAttemptCounters.empty()
    assert counters.proposals_received_total == 0
    assert counters.proposals_parsed_ok == 0
    assert counters.proposals_emitted_total == 0
    assert counters.last_parse_status == "no_call"
    assert counters.last_parse_error == ""


def test_negative_counter_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        LLMProposalAttemptCounters(
            proposals_received_total=-1,
            proposals_parsed_ok=0,
            proposals_rejected_malformed_json=0,
            proposals_rejected_schema_mismatch=0,
            proposals_emitted_total=0,
            last_parse_status="no_call",
            last_parse_error="",
        )


def test_terminal_sum_exceeding_received_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds proposals_received_total"):
        LLMProposalAttemptCounters(
            proposals_received_total=1,
            proposals_parsed_ok=2,
            proposals_rejected_malformed_json=0,
            proposals_rejected_schema_mismatch=0,
            proposals_emitted_total=0,
            last_parse_status="ok",
            last_parse_error="",
        )


def test_emitted_without_parsed_ok_rejected() -> None:
    with pytest.raises(ValueError, match="cannot emit typed"):
        LLMProposalAttemptCounters(
            proposals_received_total=1,
            proposals_parsed_ok=0,
            proposals_rejected_malformed_json=1,
            proposals_rejected_schema_mismatch=0,
            proposals_emitted_total=3,
            last_parse_status="parse_error",
            last_parse_error="",
        )


def test_unknown_parse_status_rejected() -> None:
    with pytest.raises(ValueError, match="not in"):
        LLMProposalAttemptCounters(
            proposals_received_total=0,
            proposals_parsed_ok=0,
            proposals_rejected_malformed_json=0,
            proposals_rejected_schema_mismatch=0,
            proposals_emitted_total=0,
            last_parse_status="success",  # not the documented enum
            last_parse_error="",
        )


def test_emitted_total_can_exceed_parsed_ok_for_multi_item_call() -> None:
    counters = LLMProposalAttemptCounters(
        proposals_received_total=1,
        proposals_parsed_ok=1,
        proposals_rejected_malformed_json=0,
        proposals_rejected_schema_mismatch=0,
        proposals_emitted_total=4,
        last_parse_status="ok",
        last_parse_error="",
    )
    assert counters.proposals_emitted_total == 4
    assert counters.proposals_parsed_ok == 1


# ---------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------


def test_accumulator_advance_through_all_terminal_statuses() -> None:
    acc = LLMProposalAttemptAccumulator()
    assert acc.snapshot().last_parse_status == "no_call"

    acc.record_attempt(
        parse_status="ok",
        parse_error=None,
        parsed_count=2,
        emitted_count=2,
    )
    snap = acc.snapshot()
    assert snap.proposals_received_total == 1
    assert snap.proposals_parsed_ok == 1
    assert snap.proposals_emitted_total == 2
    assert snap.last_parse_status == "ok"

    acc.record_attempt(
        parse_status="parse_error",
        parse_error="Expecting value: line 1 column 1 (char 0)",
        parsed_count=0,
        emitted_count=0,
    )
    snap = acc.snapshot()
    assert snap.proposals_received_total == 2
    assert snap.proposals_rejected_malformed_json == 1
    assert "Expecting value" in snap.last_parse_error
    assert snap.last_parse_status == "parse_error"

    acc.record_attempt(
        parse_status="empty_or_rejected",
        parse_error=None,
        parsed_count=0,
        emitted_count=0,
    )
    snap = acc.snapshot()
    assert snap.proposals_rejected_schema_mismatch == 1
    assert snap.last_parse_status == "empty_or_rejected"
    assert snap.proposals_received_total == 3


def test_accumulator_rejects_unknown_status() -> None:
    acc = LLMProposalAttemptAccumulator()
    with pytest.raises(ValueError, match="not in"):
        acc.record_attempt(
            parse_status="success",
            parse_error=None,
            parsed_count=1,
            emitted_count=1,
        )


def test_accumulator_rejects_emitted_exceeding_parsed_in_single_call() -> None:
    acc = LLMProposalAttemptAccumulator()
    with pytest.raises(ValueError, match="cannot exceed parsed_count"):
        acc.record_attempt(
            parse_status="ok",
            parse_error=None,
            parsed_count=1,
            emitted_count=2,
        )


# ---------------------------------------------------------------------
# ToM runtime
# ---------------------------------------------------------------------


_VALID_TOM_PAYLOAD = (
    "["
    '{"target_slot": "feeling_about_other", '
    '"summary": "user expresses sadness", '
    '"detail": "user mentions feeling overwhelmed by work", '
    '"evidence": "I am so tired", '
    '"confidence": 0.8, '
    '"control_signal": 0.2}'
    "]"
)


def test_tom_runtime_counters_advance_on_ok_parse() -> None:
    provider = _FakeProvider(payloads=(_VALID_TOM_PAYLOAD,))
    runtime = LLMToMProposalRuntime(provider=provider)
    initial = runtime.attempt_counters
    assert initial.proposals_received_total == 0
    assert initial.last_parse_status == "no_call"

    batch = runtime.propose(
        target_slot="feeling_about_other",
        user_input="I am so tired",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=0,
    )
    counters = runtime.attempt_counters
    assert len(batch.proposals) == 1
    assert counters.proposals_received_total == 1
    assert counters.proposals_parsed_ok == 1
    assert counters.proposals_emitted_total == 1
    assert counters.last_parse_status == "ok"


def test_tom_runtime_counters_advance_on_malformed_json() -> None:
    provider = _FakeProvider(payloads=("not actually json",))
    runtime = LLMToMProposalRuntime(provider=provider)
    runtime.propose(
        target_slot="belief_about_other",
        user_input="hello",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=0,
    )
    counters = runtime.attempt_counters
    assert counters.proposals_received_total == 1
    assert counters.proposals_rejected_malformed_json == 1
    assert counters.proposals_parsed_ok == 0
    assert counters.last_parse_status == "parse_error"
    assert counters.last_parse_error  # non-empty


def test_tom_runtime_counters_advance_on_schema_mismatch() -> None:
    # Valid JSON list, but each item fails the typed schema (missing
    # required fields), so the parser returns ``empty_or_rejected``.
    provider = _FakeProvider(payloads=('[{"unrelated": "field"}]',))
    runtime = LLMToMProposalRuntime(provider=provider)
    runtime.propose(
        target_slot="intent_about_other",
        user_input="hello",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=0,
    )
    counters = runtime.attempt_counters
    assert counters.proposals_received_total == 1
    assert counters.proposals_rejected_schema_mismatch == 1
    assert counters.proposals_parsed_ok == 0
    assert counters.last_parse_status == "empty_or_rejected"


def test_tom_runtime_caches_per_turn_so_counter_advances_once() -> None:
    """Four ToM slots share one runtime per turn; cache short-circuits.

    The runtime caches its parsed decisions per ``(user_input,
    turn_index)`` so the four ToM module calls (one per slot) only
    invoke the provider once. The counter should match: one call,
    one ``ok`` outcome, even though the runtime emitted up to four
    typed proposals across slots.
    """
    multi_slot_payload = (
        "["
        '{"target_slot": "belief_about_other", '
        '"summary": "user model 1", '
        '"detail": "detail 1", '
        '"evidence": "evidence 1", '
        '"confidence": 0.6, "control_signal": 0.1},'
        '{"target_slot": "intent_about_other", '
        '"summary": "user intent 1", '
        '"detail": "detail 2", '
        '"evidence": "evidence 2", '
        '"confidence": 0.6, "control_signal": 0.1}'
        "]"
    )
    provider = _FakeProvider(payloads=(multi_slot_payload,))
    runtime = LLMToMProposalRuntime(provider=provider)
    for slot in (
        "belief_about_other",
        "intent_about_other",
        "feeling_about_other",
        "preference_about_other",
    ):
        runtime.propose(
            target_slot=slot,
            user_input="ambient turn input",
            substrate_snapshot=None,
            memory_snapshot=None,
            previous_snapshot=None,
            turn_index=0,
        )
    counters = runtime.attempt_counters
    assert provider.calls == 1
    assert counters.proposals_received_total == 1
    assert counters.proposals_parsed_ok == 1
    assert counters.proposals_emitted_total == 2


# ---------------------------------------------------------------------
# Common-ground runtime
# ---------------------------------------------------------------------


_VALID_CG_PAYLOAD = (
    "["
    '{"scope_kind": "dyad", '
    '"scope_id": "primary+self", '
    '"summary": "they accept the meeting time", '
    '"accepted_by_ids": ["primary", "self"], '
    '"evidence": "yes that works", '
    '"confidence": 0.7, '
    '"recursion_depth": 1, '
    '"control_signal": 0.2}'
    "]"
)


def test_common_ground_runtime_counters_on_ok_parse() -> None:
    provider = _FakeProvider(payloads=(_VALID_CG_PAYLOAD,))
    runtime = LLMCommonGroundProposalRuntime(provider=provider)
    runtime.propose(user_input="yes that works", turn_index=0)
    counters = runtime.attempt_counters
    assert counters.proposals_received_total == 1
    assert counters.proposals_parsed_ok == 1
    assert counters.proposals_emitted_total == 1
    assert counters.last_parse_status == "ok"


def test_common_ground_runtime_counters_on_malformed_json() -> None:
    provider = _FakeProvider(payloads=("not json",))
    runtime = LLMCommonGroundProposalRuntime(provider=provider)
    runtime.propose(user_input="hi", turn_index=0)
    counters = runtime.attempt_counters
    assert counters.proposals_received_total == 1
    assert counters.proposals_rejected_malformed_json == 1
    assert counters.last_parse_status == "parse_error"


# ---------------------------------------------------------------------
# Owner snapshot surfacing
# ---------------------------------------------------------------------


def _empty_substrate_snapshot() -> Snapshot:
    from volvence_zero.runtime.kernel import RuntimePlaceholderValue

    return Snapshot(
        slot_name="substrate",
        owner="test",
        version=0,
        timestamp_ms=0,
        value=RuntimePlaceholderValue(
            reason="test-stub",
            expected_slot="substrate",
            produced_by="test",
            detail="test stub",
        ),
    )


def _empty_memory_snapshot() -> Snapshot:
    from volvence_zero.runtime.kernel import RuntimePlaceholderValue

    return Snapshot(
        slot_name="memory",
        owner="test",
        version=0,
        timestamp_ms=0,
        value=RuntimePlaceholderValue(
            reason="test-stub",
            expected_slot="memory",
            produced_by="test",
            detail="test stub",
        ),
    )


def _empty_multi_party_snapshot() -> Snapshot:
    from volvence_zero.runtime.kernel import RuntimePlaceholderValue

    return Snapshot(
        slot_name="multi_party_identity",
        owner="test",
        version=0,
        timestamp_ms=0,
        value=RuntimePlaceholderValue(
            reason="test-stub",
            expected_slot="multi_party_identity",
            produced_by="test",
            detail="test stub",
        ),
    )


def test_tom_module_publishes_proposal_diagnostics_on_snapshot() -> None:
    provider = _FakeProvider(payloads=(_VALID_TOM_PAYLOAD,))
    runtime = LLMToMProposalRuntime(provider=provider)
    module = FeelingAboutOtherModule(
        proposal_runtime=runtime,
        user_input="I am so tired",
        turn_index=0,
        wiring_level=WiringLevel.ACTIVE,
    )
    upstream = {
        "substrate": _empty_substrate_snapshot(),
        "memory": _empty_memory_snapshot(),
        "multi_party_identity": _empty_multi_party_snapshot(),
    }
    snapshot = asyncio.run(module.process(upstream))
    diagnostics = snapshot.value.proposal_diagnostics
    assert diagnostics is not None
    assert diagnostics.proposals_received_total == 1
    assert diagnostics.proposals_parsed_ok == 1
    assert diagnostics.last_parse_status == "ok"


def test_tom_module_publishes_none_diagnostics_when_runtime_is_noop() -> None:
    """No LLM runtime wired -> diagnostics is ``None`` (not zeros).

    Zero counters mean "the runtime ran but emitted nothing"; ``None``
    means "no LLM-backed runtime at all". Distinguishing the two is
    the whole point of the diagnostic surface.
    """
    module = BeliefAboutOtherModule(
        proposal_runtime=None,
        user_input="hello",
        turn_index=0,
        wiring_level=WiringLevel.ACTIVE,
    )
    upstream = {
        "substrate": _empty_substrate_snapshot(),
        "memory": _empty_memory_snapshot(),
        "multi_party_identity": _empty_multi_party_snapshot(),
    }
    snapshot = asyncio.run(module.process(upstream))
    assert snapshot.value.proposal_diagnostics is None


def test_common_ground_module_publishes_proposal_diagnostics() -> None:
    provider = _FakeProvider(payloads=(_VALID_CG_PAYLOAD,))
    runtime = LLMCommonGroundProposalRuntime(provider=provider)
    module = CommonGroundModule(
        proposal_runtime=runtime,
        user_input="yes that works",
        turn_index=0,
        wiring_level=WiringLevel.ACTIVE,
    )
    from volvence_zero.runtime.kernel import RuntimePlaceholderValue

    placeholder = lambda slot: Snapshot(
        slot_name=slot,
        owner="test",
        version=0,
        timestamp_ms=0,
        value=RuntimePlaceholderValue(
            reason="test-stub",
            expected_slot=slot,
            produced_by="test",
            detail="test stub",
        ),
    )
    upstream = {
        "multi_party_identity": placeholder("multi_party_identity"),
        "conversational_role": placeholder("conversational_role"),
        "belief_about_other": placeholder("belief_about_other"),
        "memory": placeholder("memory"),
    }
    snapshot = asyncio.run(module.process(upstream))
    diagnostics = snapshot.value.proposal_diagnostics
    assert diagnostics is not None
    assert diagnostics.proposals_received_total == 1
    assert diagnostics.last_parse_status == "ok"


def test_common_ground_module_publishes_none_diagnostics_when_runtime_absent() -> None:
    module = CommonGroundModule(
        proposal_runtime=None,
        user_input=None,
        turn_index=0,
        wiring_level=WiringLevel.ACTIVE,
    )
    from volvence_zero.runtime.kernel import RuntimePlaceholderValue

    placeholder = lambda slot: Snapshot(
        slot_name=slot,
        owner="test",
        version=0,
        timestamp_ms=0,
        value=RuntimePlaceholderValue(
            reason="test-stub",
            expected_slot=slot,
            produced_by="test",
            detail="test stub",
        ),
    )
    upstream = {
        "multi_party_identity": placeholder("multi_party_identity"),
        "conversational_role": placeholder("conversational_role"),
        "belief_about_other": placeholder("belief_about_other"),
        "memory": placeholder("memory"),
    }
    snapshot = asyncio.run(module.process(upstream))
    assert snapshot.value.proposal_diagnostics is None


# ---------------------------------------------------------------------
# Semantic (commitment) runtime
# ---------------------------------------------------------------------


def test_semantic_llm_runtime_counters_on_structured_ok() -> None:
    payload = (
        '{"operation": "create", '
        '"alignment_evidence": "user said: lets meet next week", '
        '"confidence": 0.7}'
    )
    provider = _FakeProvider(payloads=(payload,))
    runtime = LLMSemanticProposalRuntime(provider=provider)
    runtime.propose(
        target_slot="commitment",
        user_input="lets meet next week",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=0,
    )
    counters = runtime.attempt_counters
    assert counters.proposals_received_total == 1
    assert counters.proposals_parsed_ok == 1
    assert counters.proposals_emitted_total == 1
    assert counters.last_parse_status == "ok"


def test_semantic_llm_runtime_counters_on_unparseable_label() -> None:
    provider = _FakeProvider(payloads=("nonsense reply",))
    runtime = LLMSemanticProposalRuntime(provider=provider)
    runtime.propose(
        target_slot="commitment",
        user_input="hello",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=0,
    )
    counters = runtime.attempt_counters
    assert counters.proposals_received_total == 1
    assert counters.proposals_rejected_malformed_json == 1
    assert counters.last_parse_status == "parse_error"
