"""LLM-driven semantic proposal runtime.

Wraps a text-generation provider to extract typed semantic proposals
from a user turn. The runtime keeps the original richer commitment
life-cycle path and adds a small schema-bound proposal path for selected
owners; other owners delegate to a base runtime so this class remains
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

from dataclasses import dataclass
import json
from json import JSONDecodeError
from typing import Protocol

from volvence_zero.llm_proposal_diagnostics import LLMProposalAttemptCounters
from volvence_zero.semantic_state import (
    CommitmentSnapshot,
    load_semantic_json_schema,
    load_semantic_prompt_template,
    NoOpSemanticProposalRuntime,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
    SemanticSnapshotValue,
)
from volvence_zero.semantic_state._llm_proposal_counters import (
    LLMProposalAttemptAccumulator,
)
from volvence_zero.substrate import SubstrateSnapshot
from volvence_zero.memory import MemorySnapshot


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


_VALID_COMMITMENT_LABELS: dict[str, SemanticProposalOperation] = {
    "observe": SemanticProposalOperation.OBSERVE,
    "create": SemanticProposalOperation.CREATE,
    "revise": SemanticProposalOperation.REVISE,
    "defer": SemanticProposalOperation.DEFER,
    "activate": SemanticProposalOperation.ACTIVATE,
    "complete": SemanticProposalOperation.COMPLETE,
    "close": SemanticProposalOperation.CLOSE,
    "block": SemanticProposalOperation.BLOCK,
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
    "Return a compact JSON object with exactly these fields:\n"
    "{{\n"
    '  \"operation\": \"observe|create|revise|defer|activate|complete|close|block\",\n'
    '  \"alignment_evidence\": \"short quote or rationale from the user message\",\n'
    '  \"confidence\": 0.0\n'
    "}}\n"
    "\n"
    "Operation meanings:\n"
    '- \"create\": user makes a NEW commitment.\n'
    '- \"activate\": assistant should surface an existing commitment.\n'
    '- \"revise\": user agrees only with changes or conditions.\n'
    '- \"defer\": user asks to postpone or hold the commitment.\n'
    '- \"complete\": user reports finishing a commitment.\n'
    '- \"close\": user wraps up without a stronger completion signal.\n'
    '- \"block\": user rejects, withdraws, or signals inability.\n'
    '- \"observe\": none of the above.\n'
    "\n"
    "Do not include markdown or explanatory text."
)

_MIN_STRUCTURED_COMMITMENT_CONFIDENCE = 0.40
_MIN_GENERIC_PROPOSAL_CONFIDENCE = 0.35
# W2-B: extended to ``relationship_state`` and ``user_model`` so verticals
# that need typed relational signals (LTV / private-domain ops, EQ
# companion under stress) can drive these owners through the same JSON-
# schema generic path that ``boundary_consent`` / ``goal_value`` use.
# The shared schema (``schemas/proposal.schema.json``) is target_slot-
# agnostic so no schema change is required; the prompt template
# (``prompts/extraction.md``) is also slot-generic. Owners that opt into
# this path inherit the same fail-loud parsing + fallback-to-base
# behaviour used by the original two slots.
_GENERIC_LLM_SLOT_IDS = frozenset({
    "boundary_consent",
    "goal_value",
    "relationship_state",
    "user_model",
})


def _has_active_commitment(previous_snapshot: SemanticSnapshotValue | None) -> bool:
    """True iff ``previous_snapshot`` exposes >=1 active commitment.

    Typed dispatch via ``isinstance(CommitmentSnapshot)`` per R8 / SSOT:
    the previous snapshot is a 9-arm union over typed semantic owners
    (see ``contracts.py: SemanticSnapshotValue``); only the
    ``CommitmentSnapshot`` arm carries ``active_commitments``. Other
    arms (PlanIntent / OpenLoop / UserModel / ...) are not commitment
    state owners, so we fail closed: BLOCK / COMPLETE / DEFER route to
    OBSERVE rather than crediting an unrelated owner.
    """
    if not isinstance(previous_snapshot, CommitmentSnapshot):
        return False
    return len(previous_snapshot.active_commitments) > 0


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
    SemanticProposalOperation.OBSERVE: "llm-observed-no-commitment-signal",
    SemanticProposalOperation.CREATE: "llm-detected-new-commitment",
    SemanticProposalOperation.REVISE: "llm-detected-commitment-revision",
    SemanticProposalOperation.DEFER: "llm-detected-defer",
    SemanticProposalOperation.ACTIVATE: "llm-detected-commitment-activation",
    SemanticProposalOperation.COMPLETE: "llm-detected-completion",
    SemanticProposalOperation.CLOSE: "llm-detected-close",
    SemanticProposalOperation.BLOCK: "llm-detected-block",
}


_OPERATION_CONFIDENCE: dict[SemanticProposalOperation, float] = {
    SemanticProposalOperation.OBSERVE: 0.25,
    SemanticProposalOperation.CREATE: 0.55,
    SemanticProposalOperation.REVISE: 0.60,
    SemanticProposalOperation.DEFER: 0.50,
    SemanticProposalOperation.ACTIVATE: 0.55,
    SemanticProposalOperation.COMPLETE: 0.60,
    SemanticProposalOperation.CLOSE: 0.50,
    SemanticProposalOperation.BLOCK: 0.60,
}


_OPERATION_CONTROL: dict[SemanticProposalOperation, float] = {
    SemanticProposalOperation.OBSERVE: 0.02,
    SemanticProposalOperation.CREATE: 0.10,
    SemanticProposalOperation.REVISE: 0.10,
    SemanticProposalOperation.DEFER: 0.06,
    SemanticProposalOperation.ACTIVATE: 0.10,
    SemanticProposalOperation.COMPLETE: 0.12,
    SemanticProposalOperation.CLOSE: 0.08,
    SemanticProposalOperation.BLOCK: 0.10,
}


@dataclass(frozen=True)
class _CommitmentDecision:
    operation: SemanticProposalOperation
    evidence: str
    confidence: float
    structured: bool


@dataclass(frozen=True)
class _ParsedProposal:
    operation: SemanticProposalOperation
    summary: str
    detail: str
    confidence: float
    evidence: str
    control_signal: float
    requires_confirmation: bool


def _parse_commitment_decision(text: str) -> _CommitmentDecision | None:
    """Parse the structured AAC classifier payload.

    JSON is the primary protocol. The legacy single-token label parser remains
    as a compatibility fallback for small local models and old tests.
    """

    structured = _parse_structured_commitment_decision(text)
    if structured is not None:
        return structured
    operation = _parse_commitment_label(text)
    if operation is None:
        return None
    return _CommitmentDecision(
        operation=operation,
        evidence="",
        confidence=_OPERATION_CONFIDENCE[operation],
        structured=False,
    )


def _parse_structured_commitment_decision(text: str) -> _CommitmentDecision | None:
    raw = text.strip()
    if not raw.startswith("{"):
        return None
    try:
        payload = json.loads(raw)
    except JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    operation_raw = payload.get("operation")
    evidence_raw = payload.get("alignment_evidence")
    confidence_raw = payload.get("confidence")
    if not isinstance(operation_raw, str):
        return None
    operation = _VALID_COMMITMENT_LABELS.get(operation_raw.strip().lower())
    if operation is None:
        return None
    if not isinstance(evidence_raw, str) or not evidence_raw.strip():
        return None
    if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
        return None
    confidence = float(confidence_raw)
    if confidence < 0.0 or confidence > 1.0:
        return None
    if confidence < _MIN_STRUCTURED_COMMITMENT_CONFIDENCE:
        return _CommitmentDecision(
            operation=SemanticProposalOperation.OBSERVE,
            evidence=evidence_raw.strip()[:240],
            confidence=min(confidence, _OPERATION_CONFIDENCE[SemanticProposalOperation.OBSERVE]),
            structured=True,
        )
    return _CommitmentDecision(
        operation=operation,
        evidence=evidence_raw.strip()[:240],
        confidence=confidence,
        structured=True,
    )


def _generic_prompt(*, target_slot: str, user_input: str) -> str:
    template = load_semantic_prompt_template()
    schema = load_semantic_json_schema()
    return (
        f"{template}\n\n"
        f"Target owner slot: {target_slot}\n"
        "Return JSON matching this schema. Only emit proposals whose target_slot equals the target owner slot.\n"
        f"Schema:\n{schema}\n\n"
        "User message:\n"
        '"""\n'
        f"{user_input}\n"
        '"""\n'
        "Return JSON only."
    )


def _parse_generic_proposals(text: str, *, target_slot: str) -> tuple[_ParsedProposal, ...] | None:
    raw = text.strip()
    if not raw.startswith("{"):
        return None
    try:
        payload = json.loads(raw)
    except JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    proposals_raw = payload.get("proposals")
    if not isinstance(proposals_raw, list):
        return None
    parsed: list[_ParsedProposal] = []
    for item in proposals_raw:
        if not isinstance(item, dict):
            return None
        if item.get("target_slot") != target_slot:
            return None
        operation_raw = item.get("operation")
        summary_raw = item.get("summary")
        detail_raw = item.get("detail")
        confidence_raw = item.get("confidence")
        evidence_raw = item.get("evidence")
        if not isinstance(operation_raw, str):
            return None
        operation = _VALID_COMMITMENT_LABELS.get(operation_raw.strip().lower())
        if operation is None:
            return None
        if not isinstance(summary_raw, str) or not summary_raw.strip():
            return None
        if not isinstance(detail_raw, str) or not detail_raw.strip():
            return None
        if not isinstance(evidence_raw, str) or not evidence_raw.strip():
            return None
        if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
            return None
        confidence = float(confidence_raw)
        if confidence < 0.0 or confidence > 1.0:
            return None
        if confidence < _MIN_GENERIC_PROPOSAL_CONFIDENCE:
            continue
        control_raw = item.get("control_signal", _OPERATION_CONTROL[operation])
        if isinstance(control_raw, bool) or not isinstance(control_raw, (int, float)):
            return None
        requires_confirmation_raw = item.get("requires_confirmation", False)
        if not isinstance(requires_confirmation_raw, bool):
            return None
        parsed.append(
            _ParsedProposal(
                operation=operation,
                summary=summary_raw.strip()[:160],
                detail=detail_raw.strip()[:320],
                confidence=confidence,
                evidence=evidence_raw.strip()[:320],
                control_signal=max(0.0, min(1.0, float(control_raw))),
                requires_confirmation=requires_confirmation_raw,
            )
        )
    return tuple(parsed)


class LLMSemanticProposalRuntime(SemanticProposalRuntime):
    """Upgrades selected slots with LLM-classified typed proposals.

    ``commitment`` uses the AAC-focused classifier. ``boundary_consent``
    and ``goal_value`` use the shared proposal schema. All other slots
    delegate to ``base_runtime`` (defaults to
    :class:`NoOpSemanticProposalRuntime`). This gives us a focused
    win on the most behaviourally-load-bearing owners without
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
            Defaults to a small JSON-sized budget for structured AAC output.
    """

    runtime_id = "semantic-llm-commitment"

    def __init__(
        self,
        *,
        provider: _GenerateProtocol,
        base_runtime: SemanticProposalRuntime | None = None,
        commitment_slot_id: str = "commitment",
        max_new_tokens: int = 96,
    ) -> None:
        self._provider = provider
        self._base = base_runtime or NoOpSemanticProposalRuntime()
        self._commitment_slot_id = commitment_slot_id
        self._max_new_tokens = max_new_tokens
        # Always-on typed counters (Wave E1). Owners that wire this
        # runtime can read ``attempt_counters`` and surface it on the
        # commitment / boundary_consent / goal_value snapshots so a
        # 0-records evidence run is diagnosable without env vars.
        self._counters = LLMProposalAttemptAccumulator()

    @property
    def attempt_counters(self) -> LLMProposalAttemptCounters:
        """Return an immutable snapshot of cumulative LLM call counters.

        Counters cover the commitment classifier path AND the generic
        proposal path (``boundary_consent`` / ``goal_value``). Owner
        modules that read this should pair each turn's counter delta
        with their own emission count to differentiate
        runtime-side parse failures from owner-side filtering.
        """
        return self._counters.snapshot()

    @property
    def text_provider(self) -> _GenerateProtocol:
        """Public accessor for the underlying text-generation provider.

        Phase 1 W1.C of the EQ-owner uplift: kernels that wire this
        runtime can share the same provider with sibling LLM-driven
        runtimes (e.g. ``LLMToMProposalRuntime``,
        ``LLMCommonGroundProposalRuntime``) without re-loading model
        weights. The accessor is the typed kernel-level handle for
        provider sharing; downstream code MUST NOT reach for the
        private ``_provider`` attribute.
        """
        return self._provider

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
        if not user_input:
            return self._base_propose(
                target_slot=target_slot,
                user_input=user_input,
                substrate_snapshot=substrate_snapshot,
                memory_snapshot=memory_snapshot,
                previous_snapshot=previous_snapshot,
                turn_index=turn_index,
            )
        if target_slot != self._commitment_slot_id:
            if target_slot in _GENERIC_LLM_SLOT_IDS:
                return self._propose_generic_slot(
                    target_slot=target_slot,
                    user_input=user_input,
                    substrate_snapshot=substrate_snapshot,
                    memory_snapshot=memory_snapshot,
                    previous_snapshot=previous_snapshot,
                    turn_index=turn_index,
                )
            return self._base_propose(
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
        decision = _parse_commitment_decision(raw)
        operation = decision.operation if decision is not None else None
        # Diagnostic counter (Wave E1). The commitment classifier
        # returns at most one typed proposal per call; we map the
        # parse outcome onto the shared three-state vocabulary so the
        # counter snapshot is comparable across all three LLM-backed
        # proposal runtimes.
        if operation is None:
            self._counters.record_attempt(
                parse_status="parse_error",
                parse_error=f"unparseable commitment label: {raw[:120]!r}",
                parsed_count=0,
                emitted_count=0,
            )
        else:
            self._counters.record_attempt(
                parse_status="ok",
                parse_error=None,
                parsed_count=1,
                emitted_count=1,
            )
        # Structural guard: BLOCK / COMPLETE / DEFER only make sense
        # when there's a commitment to act on. Without an active
        # commitment in ``previous_snapshot`` we re-route them to
        # OBSERVE. This catches small-model bias (e.g. Qwen 0.5B
        # over-classifying neutral / emotional turns as "block")
        # without replacing the LLM's decision \u2014 it just refuses
        # to apply lifecycle operations to a non-existent target.
        # CREATE has no precondition (the whole point is to create
        # a new commitment); OBSERVE is always valid.
        guarded_to_observe = False
        if operation in {
            SemanticProposalOperation.DEFER,
            SemanticProposalOperation.REVISE,
            SemanticProposalOperation.COMPLETE,
            SemanticProposalOperation.CLOSE,
            SemanticProposalOperation.BLOCK,
        } and not _has_active_commitment(previous_snapshot):
            operation = SemanticProposalOperation.OBSERVE
            guarded_to_observe = True
        if operation is None:
            base_batch = self._base_propose(
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
        evidence = (
            decision.evidence
            if decision is not None and decision.evidence
            else user_input.strip()[:240]
        )
        confidence = (
            decision.confidence
            if decision is not None and decision.structured and not guarded_to_observe
            else _OPERATION_CONFIDENCE[operation]
        )
        proposal = SemanticProposal(
            proposal_id=(
                f"{target_slot}:llm-{operation.value}:{turn_index}"
            ),
            target_slot=target_slot,
            operation=operation,
            summary=_OPERATION_SUMMARY[operation],
            detail=evidence,
            confidence=confidence,
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

    def _base_propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: SemanticSnapshotValue | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        return self._base.propose(
            target_slot=target_slot,
            user_input=user_input,
            substrate_snapshot=substrate_snapshot,
            memory_snapshot=memory_snapshot,
            previous_snapshot=previous_snapshot,
            turn_index=turn_index,
        )

    def _propose_generic_slot(
        self,
        *,
        target_slot: str,
        user_input: str,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: SemanticSnapshotValue | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        prompt = _generic_prompt(
            target_slot=target_slot,
            user_input=user_input.strip()[:600],
        )
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        parsed = _parse_generic_proposals(raw, target_slot=target_slot)
        if parsed is None:
            self._counters.record_attempt(
                parse_status="parse_error",
                parse_error=f"unparseable generic proposal payload: {raw[:120]!r}",
                parsed_count=0,
                emitted_count=0,
            )
            base_batch = self._base_propose(
                target_slot=target_slot,
                user_input=user_input,
                substrate_snapshot=None,
                memory_snapshot=None,
                previous_snapshot=None,
                turn_index=turn_index,
            )
            return SemanticProposalBatch(
                proposals=base_batch.proposals,
                runtime_id=self.runtime_id,
                schema_version=base_batch.schema_version,
                description=(
                    f"LLM runtime fell back to base for {target_slot}; "
                    f"unparseable proposal payload \"{raw[:32]!r}\"."
                ),
            )
        if not parsed:
            self._counters.record_attempt(
                parse_status="empty_or_rejected",
                parse_error=None,
                parsed_count=0,
                emitted_count=0,
            )
        else:
            self._counters.record_attempt(
                parse_status="ok",
                parse_error=None,
                parsed_count=len(parsed),
                emitted_count=len(parsed),
            )
        proposals = tuple(
            SemanticProposal(
                proposal_id=f"{target_slot}:llm-{item.operation.value}:{turn_index}:{index}",
                target_slot=target_slot,
                operation=item.operation,
                summary=item.summary,
                detail=item.detail,
                confidence=item.confidence,
                evidence=item.evidence,
                control_signal=item.control_signal,
                requires_confirmation=item.requires_confirmation,
            )
            for index, item in enumerate(parsed)
        )
        return SemanticProposalBatch(
            proposals=proposals,
            runtime_id=self.runtime_id,
            schema_version=1,
            description=(
                f"LLM runtime emitted {len(proposals)} typed proposal(s) for {target_slot}."
            ),
        )


__all__ = [
    "LLMSemanticProposalRuntime",
]
