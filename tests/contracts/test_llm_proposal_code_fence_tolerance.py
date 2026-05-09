"""Contract test: LLM proposal parsers tolerate Markdown code fences.

The 2026-05-09 cross-session probe (debt #10B item 3) diagnosed that
Qwen 2.5 1.5B-Instruct wraps its JSON output in ```json fences even
when the prompt explicitly says "no markdown". The strict pre-fix
parser then rejected 100% of proposals as ``parse_error``.

This test pins the post-fix behaviour: the same fence-wrapped payloads
that previously produced zero typed proposals now produce non-empty
typed batches. It uses a deterministic fake provider that returns
fence-wrapped payloads (the exact shape captured from the live Qwen
run in ``artifacts/eq_uplift/llm_proposal_debug.jsonl``) so the test
does not depend on a downloaded HF model.
"""

from __future__ import annotations

from volvence_zero.semantic_state import SemanticProposalOperation
from volvence_zero.social.common_ground import LLMCommonGroundProposalRuntime
from volvence_zero.social.tom import LLMToMProposalRuntime


class _FencedToMProvider:
    """Returns the exact ``[]`` fence-wrapped payload Qwen 1.5B emits
    on a turn with no clear ToM observation, plus a non-empty case.
    """

    _PAYLOAD_NONEMPTY = (
        "```json\n"
        "[\n"
        '  {"target_slot": "feeling_about_other", '
        '"summary": "user feels stuck", '
        '"detail": "user came back after a tough week", '
        '"evidence": "Coming back after a tough week", '
        '"confidence": 0.7, '
        '"control_signal": 0.4}\n'
        "]\n"
        "```"
    )

    _PAYLOAD_EMPTY = "```json\n[]\n```"

    def __init__(self, *, payload: str) -> None:
        self._payload = payload

    def generate(
        self, *, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0
    ) -> str:
        del prompt, max_new_tokens, temperature
        return self._payload


class _FencedCommonGroundProvider:
    """Returns the exact fence-wrapped payload captured from the live
    Qwen 1.5B run on turn 1 of cross-session-emotional-followup.
    """

    _PAYLOAD = (
        "```json\n"
        "[\n"
        "  {\n"
        '    "scope_kind": "dyad",\n'
        '    "scope_id": "user_and_ai",\n'
        '    "summary": "The user and AI acknowledge they were discussing the same topic.",\n'
        '    "accepted_by_ids": ["user", "ai"],\n'
        '    "evidence": "I wanted to pick up where we left off when I shared about feeling stuck.",\n'
        '    "confidence": 0.95,\n'
        '    "recursion_depth": 2,\n'
        '    "control_signal": 0.85\n'
        "  }\n"
        "]\n"
        "```"
    )

    def generate(
        self, *, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0
    ) -> str:
        del prompt, max_new_tokens, temperature
        return self._PAYLOAD


def test_tom_runtime_accepts_fenced_nonempty_payload() -> None:
    """A fence-wrapped non-empty payload now produces typed proposals.

    Pre-fix this would return zero proposals (``parse_status =
    parse_error``). Post-fix the strict schema validation runs on the
    de-fenced JSON and one ``feeling_about_other`` proposal survives.
    """
    runtime = LLMToMProposalRuntime(
        provider=_FencedToMProvider(payload=_FencedToMProvider._PAYLOAD_NONEMPTY)
    )
    batch = runtime.propose(
        target_slot="feeling_about_other",
        user_input="Coming back after a tough week",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=1,
    )
    assert len(batch.proposals) == 1, (
        f"fence-stripping should let the schema parser see one valid "
        f"proposal; got batch={batch!r}"
    )
    proposal = batch.proposals[0]
    assert proposal.target_slot == "feeling_about_other"
    assert proposal.operation == SemanticProposalOperation.OBSERVE
    assert proposal.confidence == 0.7


def test_tom_runtime_accepts_fenced_empty_payload_as_zero_proposals() -> None:
    """A fence-wrapped empty array means "no observation this turn",
    not "malformed" — the runtime emits zero proposals without
    raising. This is the dominant Qwen pattern on warmth-first turns.
    """
    runtime = LLMToMProposalRuntime(
        provider=_FencedToMProvider(payload=_FencedToMProvider._PAYLOAD_EMPTY)
    )
    batch = runtime.propose(
        target_slot="belief_about_other",
        user_input="just want to be heard today",
        substrate_snapshot=None,
        memory_snapshot=None,
        previous_snapshot=None,
        turn_index=2,
    )
    assert batch.proposals == ()


def test_common_ground_runtime_accepts_fenced_payload() -> None:
    """The exact Qwen-emitted payload that was rejected pre-fix now
    yields one valid dyad atom proposal.
    """
    runtime = LLMCommonGroundProposalRuntime(provider=_FencedCommonGroundProvider())
    batch = runtime.propose(
        user_input="I wanted to pick up where we left off",
        turn_index=1,
    )
    assert len(batch.proposals) == 1, (
        f"fence-stripping should let the schema parser see one valid "
        f"common-ground proposal; got batch={batch!r}"
    )
    proposal = batch.proposals[0]
    assert proposal.scope_kind == "dyad"
    assert proposal.scope_id == "user_and_ai"
    assert proposal.confidence == 0.95


def test_strip_code_fence_leaves_plain_json_untouched() -> None:
    """Plain JSON without a fence flows through unchanged, so the
    fence-stripping pre-pass cannot mask a regression in the LLM's
    no-markdown compliance.
    """
    from volvence_zero.social._llm_parsing import strip_code_fence

    assert strip_code_fence('[]') == '[]'
    assert strip_code_fence('  [{"a": 1}]  ') == '  [{"a": 1}]  '
    assert strip_code_fence('') == ''


def test_strip_code_fence_handles_info_string_variants() -> None:
    """Tolerates ``json``, ``JSON``, and absent info strings."""
    from volvence_zero.social._llm_parsing import strip_code_fence

    assert strip_code_fence("```json\n[]\n```") == "[]"
    assert strip_code_fence("```JSON\n[1, 2]\n```") == "[1, 2]"
    assert strip_code_fence("```\n[]\n```") == "[]"


def test_strip_code_fence_rejects_non_alnum_info_string() -> None:
    """Avoid eating fences whose info string contains punctuation —
    those are likely not the JSON wrapper we want to strip and the
    strict parser should see the raw text and report it.
    """
    from volvence_zero.social._llm_parsing import strip_code_fence

    raw = "```not-json/v1\n[]\n```"
    assert strip_code_fence(raw) == raw
