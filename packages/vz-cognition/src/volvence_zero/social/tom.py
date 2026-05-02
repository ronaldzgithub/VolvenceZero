"""Theory-of-Mind owners + structured LLM proposal runtime (R17).

This module bundles the four ToM owners and their structured LLM
proposal runtime. The runtime is a collaborator of the owners — not an
independent owner — so it lives in the same file rather than in a
separate ``_runtime.py`` shard.

Owner contract: each ToM module is the single owner of its own slot
(`belief_about_other` / `intent_about_other` / `feeling_about_other` /
`preference_about_other`). The LLM runtime only emits typed
:class:`SemanticProposal` records targeted at those slots; it does not
own state and it does not route renderer behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from typing import Any, Mapping, Protocol

from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    NoOpSemanticProposalRuntime,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    FeelingAboutOtherSnapshot,
    IntentAboutOtherSnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceAboutOtherSnapshot,
)
from volvence_zero.substrate import SubstrateSnapshot


class _OtherMindOwnerModule(RuntimeModule[Any]):
    record_kind: OtherMindRecordKind
    snapshot_type: type[Any]
    empty_description: str
    dependencies = ("substrate", "memory", "multi_party_identity")
    default_wiring_level = WiringLevel.SHADOW
    min_proposal_confidence = 0.50

    def __init__(
        self,
        *,
        proposal_runtime: SemanticProposalRuntime | None = None,
        user_input: str | None = None,
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._proposal_runtime = proposal_runtime
        self._user_input = user_input
        self._turn_index = turn_index

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[Any]:
        records: tuple[OtherMindRecord, ...] = ()
        control_signal = 0.0
        if self._proposal_runtime is not None:
            substrate_snapshot = upstream.get("substrate")
            memory_snapshot = upstream.get("memory")
            batch = self._proposal_runtime.propose(
                target_slot=self.slot_name,
                user_input=self._user_input,
                substrate_snapshot=(
                    substrate_snapshot.value
                    if substrate_snapshot is not None
                    and isinstance(substrate_snapshot.value, SubstrateSnapshot)
                    else None
                ),
                memory_snapshot=(
                    memory_snapshot.value
                    if memory_snapshot is not None
                    and isinstance(memory_snapshot.value, MemorySnapshot)
                    else None
                ),
                previous_snapshot=None,
                turn_index=self._turn_index,
            )
            proposals = tuple(
                proposal
                for proposal in batch.proposals
                if proposal.target_slot == self.slot_name
                and proposal.confidence >= self.min_proposal_confidence
            )
            records = tuple(
                _record_from_proposal(
                    proposal=proposal,
                    kind=self.record_kind,
                    turn_index=self._turn_index,
                )
                for proposal in proposals
            )
            control_signal = _mean_control_signal(proposals)
        return self.publish(self._snapshot(records=records, control_signal=control_signal))

    def _snapshot(
        self,
        *,
        records: tuple[OtherMindRecord, ...],
        control_signal: float,
    ) -> Any:
        return self.snapshot_type(
            records=records,
            active_predictions=(),
            control_signal=control_signal,
            description=(
                self.empty_description
                if not records
                else f"{self.owner} published explicit records={len(records)}."
            ),
        )


class BeliefAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "belief_about_other"
    owner = "BeliefAboutOtherModule"
    value_type = BeliefAboutOtherSnapshot
    record_kind = OtherMindRecordKind.BELIEF
    snapshot_type = BeliefAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no belief-about-other records yet."


class IntentAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "intent_about_other"
    owner = "IntentAboutOtherModule"
    value_type = IntentAboutOtherSnapshot
    record_kind = OtherMindRecordKind.INTENT
    snapshot_type = IntentAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no intent-about-other records yet."


class FeelingAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "feeling_about_other"
    owner = "FeelingAboutOtherModule"
    value_type = FeelingAboutOtherSnapshot
    record_kind = OtherMindRecordKind.FEELING
    snapshot_type = FeelingAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no feeling-about-other records yet."


class PreferenceAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "preference_about_other"
    owner = "PreferenceAboutOtherModule"
    value_type = PreferenceAboutOtherSnapshot
    record_kind = OtherMindRecordKind.PREFERENCE
    snapshot_type = PreferenceAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no preference-about-other records yet."


def _record_from_proposal(
    *,
    proposal: SemanticProposal,
    kind: OtherMindRecordKind,
    turn_index: int,
) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=proposal.proposal_id,
        interlocutor_id="primary",
        kind=kind,
        summary=proposal.summary,
        detail=proposal.detail,
        confidence=proposal.confidence,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=turn_index,
        prediction_error_refs=(),
        evidence=proposal.evidence,
    )


def _mean_control_signal(proposals: tuple[SemanticProposal, ...]) -> float:
    if not proposals:
        return 0.0
    return sum(proposal.control_signal for proposal in proposals) / len(proposals)


# ---------------------------------------------------------------------------
# Structured LLM proposal runtime (collaborator of the four ToM owners above)
# ---------------------------------------------------------------------------


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


__all__ = [
    "BeliefAboutOtherModule",
    "FeelingAboutOtherModule",
    "IntentAboutOtherModule",
    "LLMToMProposalRuntime",
    "PreferenceAboutOtherModule",
]
