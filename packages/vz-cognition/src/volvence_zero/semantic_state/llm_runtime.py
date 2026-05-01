"""LLM-driven semantic proposal runtime.

Wraps a text-generation provider to extract richer commitment-life-cycle
events from a user turn than the ``NoOpSemanticProposalRuntime``'s
single OBSERVE emission. Currently we only upgrade the ``commitment``
slot; other owners delegate to a base runtime so this class is
strictly additive.

Why a separate module:

- Keeps ``semantic_state/__init__.py`` import cost flat for callers
  (e.g. evaluation harness) that never need an LLM.
- Pairs naturally with ``volvence_zero.substrate.text_generation``
  which provides the concrete ``HFTextGenerationProvider``; tests
  can pass any duck-typed object with a ``generate`` method.

Failure model (deliberate, fail-loud aware):

- LLM raises during ``generate`` \u2192 we re-raise. The kernel-level
  orchestrator converts that into a per-turn fault that is recorded
  in evaluation, not silently absorbed.
- LLM returns an unparseable label \u2192 fall through to the base
  runtime (typically NoOp). This is a *bounded soft fallback*:
  the kernel still emits an OBSERVE so downstream consumers see
  the turn happened, but no spurious CREATE / COMPLETE is invented.
- Empty / whitespace user input \u2192 delegate to base.

Security note: the LLM only sees the user's free-form text; it
never sees prior tool output or affordance-write content. That
keeps prompt-injection blast radius bounded to "the user can
influence what semantic operation is logged", not "the user can
make the system run a tool".
"""

from __future__ import annotations

from typing import Protocol

from volvence_zero.semantic_state import (
    NoOpSemanticProposalRuntime,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
    SemanticSnapshotValue,
)
from volvence_zero.substrate import SubstrateSnapshot
from volvence_zero.memory import MemorySnapshot


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


_VALID_COMMITMENT_LABELS: dict[str, SemanticProposalOperation] = {
    "create": SemanticProposalOperation.CREATE,
    "complete": SemanticProposalOperation.COMPLETE,
    "block": SemanticProposalOperation.BLOCK,
    "defer": SemanticProposalOperation.DEFER,
    "observe": SemanticProposalOperation.OBSERVE,
}


_COMMITMENT_PROMPT = (
    "You classify a user's message in a multi-turn dialogue.\n"
    "Read this message and decide what it does to commitments / "
    "promises / plans the user has with the assistant.\n"
    "\n"
    "User message:\n"
    '"""\n'
    "{user_input}\n"
    '"""\n'
    "\n"
    "Pick exactly ONE label from the list:\n"
    '- "create": user makes a NEW commitment ("I will...", "I\'ll start...", "I want to commit to...")\n'
    '- "complete": user reports FINISHING a commitment ("I did it", "I finished", "It\'s done")\n'
    '- "block": user REJECTS, withdraws, or signals inability ("I can\'t", "Sorry I didn\'t", "I won\'t")\n'
    '- "defer": user asks to POSTPONE a commitment ("can we move it", "later this week", "not today")\n'
    '- "observe": none of the above; the message is neutral or about something else\n'
    "\n"
    "Respond with ONLY the single lowercase label word, no punctuation, no explanation."
)


def _has_active_commitment(previous_snapshot: SemanticSnapshotValue | None) -> bool:
    """True iff ``previous_snapshot`` exposes >=1 active commitment.

    The runtime ships with structural typing only \u2014 we duck-type
    on ``active_commitments`` so a future replacement of
    ``CommitmentSnapshot`` (e.g. with a richer typed lifecycle
    aggregate) does not require a runtime re-import. Anything that
    isn't a tuple-like with ``len() >= 1`` is treated as "no active
    commitment", which fails closed: BLOCK / COMPLETE / DEFER are
    routed to OBSERVE rather than risk applying to nothing.
    """
    if previous_snapshot is None:
        return False
    active = getattr(previous_snapshot, "active_commitments", None)
    if active is None:
        return False
    try:
        return len(active) > 0
    except TypeError:
        return False


def _parse_commitment_label(text: str) -> SemanticProposalOperation | None:
    """Extract a commitment operation from the LLM's raw response.

    The LLM is the classifier; here we only enforce that its output
    is one of the labels the prompt allowed. We deliberately:

    - lowercase + strip whitespace (LLMs sometimes capitalise)
    - take the first token only (LLMs sometimes produce a sentence)
    - strip simple trailing punctuation

    If the result still isn't in the enum, return ``None`` so the
    caller can fall back. We never *guess* an operation from a
    partial match \u2014 that would re-introduce keyword-matching
    over user-facing text via the back door.
    """
    if not text:
        return None
    cleaned = text.strip().lower()
    if not cleaned:
        return None
    first = cleaned.split()[0]
    word = first.rstrip(".,;:!?\"'`)")
    return _VALID_COMMITMENT_LABELS.get(word)


_OPERATION_SUMMARY: dict[SemanticProposalOperation, str] = {
    SemanticProposalOperation.CREATE: "llm-detected-new-commitment",
    SemanticProposalOperation.COMPLETE: "llm-detected-completion",
    SemanticProposalOperation.BLOCK: "llm-detected-block",
    SemanticProposalOperation.DEFER: "llm-detected-defer",
    SemanticProposalOperation.OBSERVE: "llm-observed-no-commitment-signal",
}


_OPERATION_CONFIDENCE: dict[SemanticProposalOperation, float] = {
    SemanticProposalOperation.CREATE: 0.55,
    SemanticProposalOperation.COMPLETE: 0.60,
    SemanticProposalOperation.BLOCK: 0.60,
    SemanticProposalOperation.DEFER: 0.50,
    SemanticProposalOperation.OBSERVE: 0.25,
}


_OPERATION_CONTROL: dict[SemanticProposalOperation, float] = {
    SemanticProposalOperation.CREATE: 0.10,
    SemanticProposalOperation.COMPLETE: 0.12,
    SemanticProposalOperation.BLOCK: 0.10,
    SemanticProposalOperation.DEFER: 0.06,
    SemanticProposalOperation.OBSERVE: 0.02,
}


class LLMSemanticProposalRuntime(SemanticProposalRuntime):
    """Upgrades the ``commitment`` slot with LLM-classified operations.

    All other slots delegate to ``base_runtime`` (defaults to
    :class:`NoOpSemanticProposalRuntime`). This gives us a focused
    win on the most behaviourally-load-bearing owner without
    forcing us to design 9 prompts on day one.

    Construction example::

        from volvence_zero.substrate import HFTextGenerationProvider
        provider = HFTextGenerationProvider(model=..., tokenizer=...)
        runtime = LLMSemanticProposalRuntime(provider=provider)

    Args:
        provider: Anything matching ``_GenerateProtocol``.
        base_runtime: Where non-commitment slots are routed. Also
            used as the safe fallback when the LLM returns an
            unparseable label.
        commitment_slot_id: Slot name to upgrade. Defaults to
            ``"commitment"``. Exposed for tests / future expansion.
        max_new_tokens: Upper bound on generated tokens per call.
            Eight is enough for a single label; raising it costs
            latency without buying accuracy.
    """

    runtime_id = "semantic-llm-commitment"

    def __init__(
        self,
        *,
        provider: _GenerateProtocol,
        base_runtime: SemanticProposalRuntime | None = None,
        commitment_slot_id: str = "commitment",
        max_new_tokens: int = 8,
    ) -> None:
        self._provider = provider
        self._base = base_runtime or NoOpSemanticProposalRuntime()
        self._commitment_slot_id = commitment_slot_id
        self._max_new_tokens = max_new_tokens

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: SemanticSnapshotValue | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        if target_slot != self._commitment_slot_id or not user_input:
            return self._base.propose(
                target_slot=target_slot,
                user_input=user_input,
                substrate_snapshot=substrate_snapshot,
                memory_snapshot=memory_snapshot,
                previous_snapshot=previous_snapshot,
                turn_index=turn_index,
            )
        prompt = _COMMITMENT_PROMPT.format(user_input=user_input.strip()[:600])
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        operation = _parse_commitment_label(raw)
        # Structural guard: BLOCK / COMPLETE / DEFER only make sense
        # when there's a commitment to act on. Without an active
        # commitment in ``previous_snapshot`` we re-route them to
        # OBSERVE. This catches small-model bias (e.g. Qwen 0.5B
        # over-classifying neutral / emotional turns as "block")
        # without replacing the LLM's decision \u2014 it just refuses
        # to apply lifecycle operations to a non-existent target.
        # CREATE has no precondition (the whole point is to create
        # a new commitment); OBSERVE is always valid.
        if operation in {
            SemanticProposalOperation.BLOCK,
            SemanticProposalOperation.COMPLETE,
            SemanticProposalOperation.DEFER,
        } and not _has_active_commitment(previous_snapshot):
            operation = SemanticProposalOperation.OBSERVE
        if operation is None:
            base_batch = self._base.propose(
                target_slot=target_slot,
                user_input=user_input,
                substrate_snapshot=substrate_snapshot,
                memory_snapshot=memory_snapshot,
                previous_snapshot=previous_snapshot,
                turn_index=turn_index,
            )
            return SemanticProposalBatch(
                proposals=base_batch.proposals,
                runtime_id=self.runtime_id,
                schema_version=base_batch.schema_version,
                description=(
                    f"LLM runtime fell back to base for {target_slot}; "
                    f"unparseable label \"{raw[:32]!r}\"."
                ),
            )
        evidence = user_input.strip()[:240]
        proposal = SemanticProposal(
            proposal_id=(
                f"{target_slot}:llm-{operation.value}:{turn_index}"
            ),
            target_slot=target_slot,
            operation=operation,
            summary=_OPERATION_SUMMARY[operation],
            detail=evidence,
            confidence=_OPERATION_CONFIDENCE[operation],
            evidence=evidence,
            control_signal=_OPERATION_CONTROL[operation],
        )
        return SemanticProposalBatch(
            proposals=(proposal,),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=(
                f"LLM runtime classified turn {turn_index} as "
                f"{operation.value} for {target_slot}."
            ),
        )


__all__ = [
    "LLMSemanticProposalRuntime",
]
