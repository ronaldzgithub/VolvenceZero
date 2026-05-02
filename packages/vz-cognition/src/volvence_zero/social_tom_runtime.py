"""Structured LLM proposal runtime for Theory-of-Mind owners (R17).

The runtime is intentionally proposal-only: it turns LLM structured output
into ``SemanticProposal`` records targeted at the existing ToM owners. It
does not own belief / intent / feeling / preference state and it does not
route renderer behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from typing import Protocol

from volvence_zero.memory import MemorySnapshot
from volvence_zero.semantic_state import (
    NoOpSemanticProposalRuntime,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.substrate import SubstrateSnapshot


_TOM_TARGET_SLOTS: frozenset[str] = frozenset(
    {
        "belief_about_other",
        "intent_about_other",
        "feeling_about_other",
        "preference_about_other",
    }
)
_MIN_TOM_CONFIDENCE = 0.50


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


@dataclass(frozen=True)
class _ToMDecision:
    target_slot: str
    summary: str
    detail: str
    evidence: str
    confidence: float
    control_signal: float


_TOM_PROMPT = (
    "You extract Theory-of-Mind observations from one dialogue turn.\n"
    "Return a JSON array. Each item must have exactly these fields:\n"
    "[\n"
    "  {{\n"
    '    \"target_slot\": \"belief_about_other|intent_about_other|feeling_about_other|preference_about_other\",\n'
    '    \"summary\": \"short stable claim\",\n'
    '    \"detail\": \"specific evidence-aware detail\",\n'
    '    \"evidence\": \"short quote or observation from the user message\",\n'
    '    \"confidence\": 0.0,\n'
    '    \"control_signal\": 0.0\n'
    "  }}\n"
    "]\n"
    "\n"
    "Do not infer demographics. Do not output markdown. If there is no "
    "clear Theory-of-Mind observation, return [].\n"
    "\n"
    "User message:\n"
    '\"\"\"\n'
    "{user_input}\n"
    '\"\"\"'
)


class LLMToMProposalRuntime(SemanticProposalRuntime):
    """Structured proposal source for R17 ToM owners."""

    runtime_id = "social-tom-llm-structured"

    def __init__(
        self,
        *,
        provider: _GenerateProtocol,
        base_runtime: SemanticProposalRuntime | None = None,
        max_new_tokens: int = 384,
    ) -> None:
        self._provider = provider
        self._base = base_runtime or NoOpSemanticProposalRuntime()
        self._max_new_tokens = max_new_tokens
        self._cache_key: tuple[str, int] | None = None
        self._cache_decisions: tuple[_ToMDecision, ...] | None = None

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        if target_slot not in _TOM_TARGET_SLOTS or not user_input:
            return self._base.propose(
                target_slot=target_slot,
                user_input=user_input,
                substrate_snapshot=substrate_snapshot,
                memory_snapshot=memory_snapshot,
                previous_snapshot=previous_snapshot,
                turn_index=turn_index,
            )

        decisions = self._decisions_for_turn(user_input=user_input, turn_index=turn_index)
        proposals = tuple(
            SemanticProposal(
                proposal_id=f"{decision.target_slot}:tom-llm:{turn_index}:{index}",
                target_slot=decision.target_slot,
                operation=SemanticProposalOperation.OBSERVE,
                summary=decision.summary,
                detail=decision.detail,
                confidence=decision.confidence,
                evidence=decision.evidence,
                control_signal=decision.control_signal,
            )
            for index, decision in enumerate(decisions)
            if decision.target_slot == target_slot
        )
        return SemanticProposalBatch(
            proposals=proposals,
            runtime_id=self.runtime_id,
            schema_version=1,
            description=(
                f"Structured ToM runtime emitted {len(proposals)} proposal(s) "
                f"for {target_slot} at turn {turn_index}."
            ),
        )

    def _decisions_for_turn(
        self,
        *,
        user_input: str,
        turn_index: int,
    ) -> tuple[_ToMDecision, ...]:
        cache_key = (user_input, turn_index)
        if self._cache_key == cache_key and self._cache_decisions is not None:
            return self._cache_decisions
        prompt = _TOM_PROMPT.format(user_input=user_input.strip()[:800])
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        decisions = _parse_tom_decisions(raw)
        if decisions is None:
            decisions = ()
        self._cache_key = cache_key
        self._cache_decisions = decisions
        return decisions


def _parse_tom_decisions(text: str) -> tuple[_ToMDecision, ...] | None:
    try:
        payload = json.loads(text.strip())
    except JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    decisions: list[_ToMDecision] = []
    for item in payload:
        decision = _parse_tom_decision(item)
        if decision is not None:
            decisions.append(decision)
    return tuple(decisions)


def _parse_tom_decision(item: object) -> _ToMDecision | None:
    if not isinstance(item, dict):
        return None
    target_slot = item.get("target_slot")
    summary = item.get("summary")
    detail = item.get("detail")
    evidence = item.get("evidence")
    confidence = item.get("confidence")
    control_signal = item.get("control_signal", 0.0)
    if target_slot not in _TOM_TARGET_SLOTS:
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(detail, str) or not detail.strip():
        return None
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if isinstance(control_signal, bool) or not isinstance(control_signal, (int, float)):
        return None
    confidence_value = float(confidence)
    control_value = float(control_signal)
    if confidence_value < _MIN_TOM_CONFIDENCE or confidence_value > 1.0:
        return None
    if control_value < 0.0 or control_value > 1.0:
        return None
    return _ToMDecision(
        target_slot=target_slot,
        summary=summary.strip()[:160],
        detail=detail.strip()[:500],
        evidence=evidence.strip()[:240],
        confidence=confidence_value,
        control_signal=control_value,
    )


__all__ = ["LLMToMProposalRuntime"]
