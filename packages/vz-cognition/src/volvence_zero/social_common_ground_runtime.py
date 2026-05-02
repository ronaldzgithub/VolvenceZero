"""Structured proposal runtime for common-ground atoms (R19)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
from typing import Protocol

from volvence_zero.social_cognition import (
    MAX_COMMON_GROUND_RECURSION_DEPTH,
    SocialScopeKind,
)


_MIN_COMMON_GROUND_CONFIDENCE = 0.50
_VALID_SCOPE_KINDS = {SocialScopeKind.DYAD.value, SocialScopeKind.GROUP.value}


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


@dataclass(frozen=True)
class CommonGroundProposal:
    scope_kind: SocialScopeKind
    scope_id: str
    summary: str
    accepted_by_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: float
    recursion_depth: int
    control_signal: float = 0.0


@dataclass(frozen=True)
class CommonGroundProposalBatch:
    proposals: tuple[CommonGroundProposal, ...]
    runtime_id: str
    description: str


_COMMON_GROUND_PROMPT = (
    "Extract common-ground observations from one dialogue turn.\n"
    "Return a JSON array. Each item must have exactly these fields:\n"
    "[\n"
    "  {{\n"
    '    \"scope_kind\": \"dyad|group\",\n'
    '    \"scope_id\": \"stable dyad or group id\",\n'
    '    \"summary\": \"shared fact or mutual acceptance\",\n'
    '    \"accepted_by_ids\": [\"id1\", \"id2\"],\n'
    '    \"evidence\": \"short quote or observation\",\n'
    '    \"confidence\": 0.0,\n'
    '    \"recursion_depth\": 2,\n'
    '    \"control_signal\": 0.0\n'
    "  }}\n"
    "]\n"
    "\n"
    "Only include facts mutually accepted or clearly shared. Do not infer "
    "common ground from keywords alone. If none is clear, return [].\n"
    "\n"
    "User message:\n"
    '\"\"\"\n'
    "{user_input}\n"
    '\"\"\"'
)


class LLMCommonGroundProposalRuntime:
    """Structured source for CommonGroundModule proposals."""

    runtime_id = "common-ground-llm-structured"

    def __init__(
        self,
        *,
        provider: _GenerateProtocol,
        max_new_tokens: int = 384,
    ) -> None:
        self._provider = provider
        self._max_new_tokens = max_new_tokens

    def propose(
        self,
        *,
        user_input: str | None,
        turn_index: int,
    ) -> CommonGroundProposalBatch:
        if not user_input:
            return CommonGroundProposalBatch(
                proposals=(),
                runtime_id=self.runtime_id,
                description="No user input; common-ground runtime emitted no proposals.",
            )
        prompt = _COMMON_GROUND_PROMPT.format(user_input=user_input.strip()[:800])
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        proposals = _parse_common_ground_proposals(raw)
        return CommonGroundProposalBatch(
            proposals=proposals or (),
            runtime_id=self.runtime_id,
            description=(
                f"Structured common-ground runtime emitted "
                f"{len(proposals or ())} proposal(s) at turn {turn_index}."
            ),
        )


def _parse_common_ground_proposals(text: str) -> tuple[CommonGroundProposal, ...] | None:
    try:
        payload = json.loads(text.strip())
    except JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    proposals: list[CommonGroundProposal] = []
    for item in payload:
        proposal = _parse_common_ground_proposal(item)
        if proposal is not None:
            proposals.append(proposal)
    return tuple(proposals)


def _parse_common_ground_proposal(item: object) -> CommonGroundProposal | None:
    if not isinstance(item, dict):
        return None
    scope_kind_raw = item.get("scope_kind")
    scope_id = item.get("scope_id")
    summary = item.get("summary")
    accepted_by_ids = item.get("accepted_by_ids")
    evidence = item.get("evidence")
    confidence = item.get("confidence")
    recursion_depth = item.get("recursion_depth")
    control_signal = item.get("control_signal", 0.0)
    if not isinstance(scope_kind_raw, str) or scope_kind_raw not in _VALID_SCOPE_KINDS:
        return None
    if not isinstance(scope_id, str) or not scope_id.strip():
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(accepted_by_ids, list):
        return None
    accepted = tuple(item for item in accepted_by_ids if isinstance(item, str))
    if len(accepted) != len(accepted_by_ids) or not accepted or len(set(accepted)) != len(accepted):
        return None
    if isinstance(evidence, str):
        evidence_tuple = (evidence.strip(),) if evidence.strip() else ()
    elif isinstance(evidence, list):
        evidence_tuple = tuple(item.strip() for item in evidence if isinstance(item, str) and item.strip())
    else:
        return None
    if not evidence_tuple:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    confidence_value = float(confidence)
    if confidence_value < _MIN_COMMON_GROUND_CONFIDENCE or confidence_value > 1.0:
        return None
    if isinstance(recursion_depth, bool) or not isinstance(recursion_depth, int):
        return None
    if recursion_depth < 0 or recursion_depth > MAX_COMMON_GROUND_RECURSION_DEPTH:
        return None
    if isinstance(control_signal, bool) or not isinstance(control_signal, (int, float)):
        return None
    control_value = float(control_signal)
    if control_value < 0.0 or control_value > 1.0:
        return None
    return CommonGroundProposal(
        scope_kind=SocialScopeKind(scope_kind_raw),
        scope_id=scope_id.strip(),
        summary=summary.strip()[:240],
        accepted_by_ids=accepted,
        evidence=evidence_tuple[:4],
        confidence=confidence_value,
        recursion_depth=recursion_depth,
        control_signal=control_value,
    )


__all__ = [
    "CommonGroundProposal",
    "CommonGroundProposalBatch",
    "LLMCommonGroundProposalRuntime",
]
