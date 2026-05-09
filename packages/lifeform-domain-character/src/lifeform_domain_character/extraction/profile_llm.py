"""LLM-assisted reviewed profile extraction.

Workflow:

1. Caller supplies ``novel_text`` and a generative provider (any
   object exposing ``generate(prompt, max_new_tokens, temperature)``
   — same protocol as ``LLMSemanticProposalRuntime``'s text
   provider).
2. ``extract_profile_candidate`` chunks the text, prompts the LLM
   for structured JSON describing the protagonist, parses each
   chunk's JSON, and aggregates field-level proposals into a
   :class:`ReviewedProfileCandidate`.
3. The reviewer inspects the candidate — particularly fields the
   LLM flagged as low-confidence in ``requires_review`` — and calls
   ``review_profile_candidate`` to convert it into a
   :class:`CharacterSoulProfile`.

What the LLM is asked to produce:

A small JSON object per chunk with the SAME shape as
:class:`CharacterSoulProfile` plus an ``inference_confidence`` and
a ``low_confidence_fields`` list. We do NOT ask the LLM to commit
the result; that is the reviewer's responsibility.

Test posture:

The unit / contract tests use a fake provider that returns
deterministic JSON. The real-LLM smoke is skipped by default
(same pattern as debt #10B's evidence chain test).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, Iterable, Protocol

from lifeform_domain_character.profile import (
    CharacterBoundaryPrior,
    CharacterDrivePrior,
    CharacterKnowledgeSeed,
    CharacterSignatureCase,
    CharacterSoulProfile,
    CharacterStrategyPrior,
)


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


@dataclass(frozen=True)
class ReviewedProfileCandidate:
    """LLM-proposed profile awaiting human review.

    The candidate carries everything a reviewer needs to make a
    decision but is intentionally NOT a usable
    :class:`CharacterSoulProfile`. Pass it to
    :func:`review_profile_candidate` along with a ``reviewer`` id
    and any modifications.

    Fields:

    * Per-section proposed records: same field types as the final
      profile.
    * ``requires_review`` — sorted list of field names the LLM
      flagged as low-confidence; the reviewer must address each.
    * ``aggregate_inference_confidence`` — mean of per-chunk
      confidence values.
    * ``provenance_chunks`` — short text snippets (locator +
      excerpt prefix) the LLM used to support its proposal,
      reviewer-readable.
    """

    profile_id: str
    character_name: str
    source_title: str
    description: str
    knowledge_seeds: tuple[CharacterKnowledgeSeed, ...]
    signature_cases: tuple[CharacterSignatureCase, ...]
    strategy_priors: tuple[CharacterStrategyPrior, ...]
    boundary_priors: tuple[CharacterBoundaryPrior, ...]
    drive_priors: tuple[CharacterDrivePrior, ...]
    requires_review: tuple[str, ...]
    aggregate_inference_confidence: float
    provenance_chunks: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


_PROFILE_EXTRACTION_PROMPT = (
    "You are extracting a reviewed CharacterSoulProfile candidate from a "
    "novel chunk. The character of interest is the named protagonist below. "
    "Return a SINGLE JSON object with these top-level keys (all strings "
    "unless noted):\n"
    "{{\n"
    '  "character_name": "<the protagonist\'s primary name>",\n'
    '  "description": "<one short paragraph paraphrasing who they are>",\n'
    '  "drive_priors": [ {{"name": "...", "target": 0.6, '
    '"homeostatic_band": [0.4, 0.8], "pe_weight": 0.5, '
    '"initial_level": 0.5, "rationale": "..."}} ],\n'
    '  "boundary_priors": [ {{"boundary_id": "...", '
    '"trigger_reasons": ["..."], "answer_depth_limit_hint": "soft|strong|absolute", '
    '"description": "..."}} ],\n'
    '  "signature_cases": [ {{"case_id": "...", "problem_pattern": "...", '
    '"description": "...", "regime_tags": ["..."], "track_tags": ["self"]}} ],\n'
    '  "knowledge_seeds": [ {{"seed_id": "...", "domain": "...", '
    '"title": "...", "summary": "...", "snippet": "..."}} ],\n'
    '  "strategy_priors": [ {{"rule_id": "...", "problem_pattern": "...", '
    '"recommended_regime": "...", "recommended_ordering": ["..."], '
    '"recommended_pacing": "...", "description": "..."}} ],\n'
    '  "low_confidence_fields": ["<top-level field name>", ...],\n'
    '  "inference_confidence": 0.0\n'
    "}}\n"
    "\n"
    "Rules:\n"
    "* Paraphrase. Do not include verbatim novel text.\n"
    "* If you are uncertain about a field, list its top-level name in "
    "low_confidence_fields and still emit your best paraphrase.\n"
    "* Keep arrays small (<=4 items each); reviewer will expand.\n"
    "* No markdown, no explanation outside the JSON.\n"
    "\n"
    "Protagonist: {character_focus}\n"
    "Novel chunk:\n"
    '"""\n'
    "{novel_chunk}\n"
    '"""'
)


def extract_profile_candidate(
    novel_text: str,
    *,
    llm_runtime: _GenerateProtocol,
    character_focus: str,
    profile_id: str,
    source_title: str,
    chunk_size: int = 4000,
    max_new_tokens: int = 1024,
) -> ReviewedProfileCandidate:
    """Run the LLM extraction over a novel and return a candidate.

    The function is best-effort: parse failures or schema-mismatches
    on individual chunks are recorded in ``notes`` rather than
    raising, because the reviewer is the final arbiter. Total parse
    failure (no chunk yielded a valid JSON) raises
    :class:`ValueError`.
    """
    if not novel_text.strip():
        raise ValueError("extract_profile_candidate: novel_text is empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = _split_for_extraction(novel_text, chunk_size=chunk_size)
    knowledge_seeds: list[CharacterKnowledgeSeed] = []
    signature_cases: list[CharacterSignatureCase] = []
    strategy_priors: list[CharacterStrategyPrior] = []
    boundary_priors: list[CharacterBoundaryPrior] = []
    drive_priors: list[CharacterDrivePrior] = []
    notes: list[str] = []
    confidences: list[float] = []
    low_confidence: set[str] = set()
    provenance_chunks: list[str] = []
    descriptions: list[str] = []
    parsed_any = False
    for index, chunk in enumerate(chunks):
        prompt = _PROFILE_EXTRACTION_PROMPT.format(
            character_focus=character_focus,
            novel_chunk=chunk[:chunk_size],
        )
        try:
            raw = llm_runtime.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        except Exception as exc:  # pragma: no cover - provider failures noted
            notes.append(
                f"chunk {index}: provider raised "
                f"{type(exc).__name__}: {exc!s}"
            )
            continue
        parsed = _parse_extraction_response(raw, chunk_index=index)
        if parsed is None:
            notes.append(f"chunk {index}: unparseable response")
            continue
        parsed_any = True
        knowledge_seeds.extend(parsed.knowledge_seeds)
        signature_cases.extend(parsed.signature_cases)
        strategy_priors.extend(parsed.strategy_priors)
        boundary_priors.extend(parsed.boundary_priors)
        drive_priors.extend(parsed.drive_priors)
        confidences.append(parsed.inference_confidence)
        low_confidence.update(parsed.low_confidence_fields)
        descriptions.append(parsed.description)
        provenance_chunks.append(
            f"chunk_{index}:offset={index * chunk_size}:"
            f"{chunk[:80].strip()}..."
        )
    if not parsed_any:
        raise ValueError(
            "extract_profile_candidate: no chunk yielded a valid JSON "
            f"response. notes={notes!r}"
        )
    aggregate_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )
    description = " | ".join(descriptions)[:1024] if descriptions else ""
    knowledge_seeds = _dedupe(knowledge_seeds, key=lambda x: x.seed_id)
    signature_cases = _dedupe(signature_cases, key=lambda x: x.case_id)
    strategy_priors = _dedupe(strategy_priors, key=lambda x: x.rule_id)
    boundary_priors = _dedupe(boundary_priors, key=lambda x: x.boundary_id)
    drive_priors = _dedupe(drive_priors, key=lambda x: x.name)
    return ReviewedProfileCandidate(
        profile_id=profile_id,
        character_name=character_focus,
        source_title=source_title,
        description=description,
        knowledge_seeds=tuple(knowledge_seeds),
        signature_cases=tuple(signature_cases),
        strategy_priors=tuple(strategy_priors),
        boundary_priors=tuple(boundary_priors),
        drive_priors=tuple(drive_priors),
        requires_review=tuple(sorted(low_confidence)),
        aggregate_inference_confidence=aggregate_confidence,
        provenance_chunks=tuple(provenance_chunks),
        notes=tuple(notes),
    )


def review_profile_candidate(
    candidate: ReviewedProfileCandidate,
    *,
    reviewer: str,
    review_locator: str,
    version: str = "0.1.0",
    target_contexts: tuple[str, ...] = (
        "character-companion",
        "fictional-roleplay",
    ),
) -> CharacterSoulProfile:
    """Convert a reviewed candidate into a :class:`CharacterSoulProfile`.

    The reviewer is the accountable party for the resulting profile.
    They MUST:

    * Have addressed every field in ``candidate.requires_review``
      (this function does not enforce that — the audit trail is in
      the reviewer's commit history / review locator string).
    * Provide a non-empty ``reviewer`` id and ``review_locator``
      (e.g. PR id + reviewer name).
    """
    if not reviewer.strip():
        raise ValueError("review_profile_candidate: reviewer must be non-empty")
    if not review_locator.strip():
        raise ValueError(
            "review_profile_candidate: review_locator must be non-empty"
        )
    if not candidate.boundary_priors:
        raise ValueError(
            "review_profile_candidate: candidate has zero boundary_priors; "
            "a profile without boundaries is not safe to ship — the reviewer "
            "must add at least one boundary before accepting."
        )
    return CharacterSoulProfile(
        profile_id=candidate.profile_id,
        character_name=candidate.character_name,
        source_title=candidate.source_title,
        version=version,
        reviewed_by=reviewer,
        source_uri=review_locator,
        description=(
            candidate.description
            or f"Reviewed character profile for {candidate.character_name}."
        ),
        knowledge_seeds=candidate.knowledge_seeds,
        signature_cases=candidate.signature_cases,
        strategy_priors=candidate.strategy_priors,
        boundary_priors=candidate.boundary_priors,
        drive_priors=candidate.drive_priors,
        target_contexts=target_contexts,
    )


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParsedChunk:
    description: str
    knowledge_seeds: tuple[CharacterKnowledgeSeed, ...]
    signature_cases: tuple[CharacterSignatureCase, ...]
    strategy_priors: tuple[CharacterStrategyPrior, ...]
    boundary_priors: tuple[CharacterBoundaryPrior, ...]
    drive_priors: tuple[CharacterDrivePrior, ...]
    low_confidence_fields: tuple[str, ...]
    inference_confidence: float


def _parse_extraction_response(
    text: str, *, chunk_index: int
) -> _ParsedChunk | None:
    raw = text.strip()
    # Strip code fences if any (LLMs sometimes wrap JSON in ```).
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    if not raw.startswith("{"):
        return None
    try:
        payload = json.loads(raw)
    except JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    knowledge = _coerce_knowledge_seeds(
        payload.get("knowledge_seeds") or [], chunk_index=chunk_index
    )
    cases = _coerce_signature_cases(
        payload.get("signature_cases") or [], chunk_index=chunk_index
    )
    strategies = _coerce_strategy_priors(
        payload.get("strategy_priors") or [], chunk_index=chunk_index
    )
    boundaries = _coerce_boundary_priors(
        payload.get("boundary_priors") or [], chunk_index=chunk_index
    )
    drives = _coerce_drive_priors(payload.get("drive_priors") or [])
    low_conf = tuple(
        str(item)
        for item in payload.get("low_confidence_fields") or []
        if isinstance(item, str)
    )
    confidence_raw = payload.get("inference_confidence")
    confidence = (
        float(confidence_raw)
        if isinstance(confidence_raw, (int, float))
        else 0.0
    )
    description = str(payload.get("description") or "")
    return _ParsedChunk(
        description=description,
        knowledge_seeds=knowledge,
        signature_cases=cases,
        strategy_priors=strategies,
        boundary_priors=boundaries,
        drive_priors=drives,
        low_confidence_fields=low_conf,
        inference_confidence=confidence,
    )


def _coerce_knowledge_seeds(
    items: list[Any], *, chunk_index: int
) -> tuple[CharacterKnowledgeSeed, ...]:
    out: list[CharacterKnowledgeSeed] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            out.append(
                CharacterKnowledgeSeed(
                    seed_id=str(
                        item.get("seed_id")
                        or f"llm-knowledge:{chunk_index}:{index}"
                    ),
                    domain=str(item.get("domain") or "extracted"),
                    title=str(item.get("title") or "Untitled extracted seed"),
                    summary=str(item.get("summary") or ""),
                    snippet=str(item.get("snippet") or ""),
                    evidence_locator=f"llm-extraction:chunk={chunk_index}",
                    confidence=float(item.get("confidence", 0.6)),
                    evidence_strength=str(
                        item.get("evidence_strength", "medium")
                    ),
                    topic_tags=tuple(
                        str(t) for t in item.get("topic_tags", ())
                    ),
                )
            )
        except (ValueError, TypeError):
            continue
    return tuple(out)


def _coerce_signature_cases(
    items: list[Any], *, chunk_index: int
) -> tuple[CharacterSignatureCase, ...]:
    out: list[CharacterSignatureCase] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            out.append(
                CharacterSignatureCase(
                    case_id=str(
                        item.get("case_id")
                        or f"llm-case:{chunk_index}:{index}"
                    ),
                    domain=str(item.get("domain") or "extracted"),
                    problem_pattern=str(
                        item.get("problem_pattern") or "extracted-pattern"
                    ),
                    user_state_pattern=str(
                        item.get("user_state_pattern") or "unspecified"
                    ),
                    risk_markers=tuple(
                        str(t) for t in item.get("risk_markers", ("risk-low",))
                    ),
                    track_tags=tuple(
                        str(t) for t in item.get("track_tags", ("self",))
                    ),
                    regime_tags=tuple(
                        str(t)
                        for t in item.get("regime_tags", ("guided_exploration",))
                    ),
                    intervention_ordering=tuple(
                        str(t)
                        for t in item.get("intervention_ordering", ("step_one",))
                    ),
                    outcome_label=str(
                        item.get("outcome_label", "unspecified")
                    ),
                    description=str(
                        item.get("description") or "Extracted case (please review)"
                    ),
                    confidence=float(item.get("confidence", 0.55)),
                )
            )
        except (ValueError, TypeError):
            continue
    return tuple(out)


def _coerce_strategy_priors(
    items: list[Any], *, chunk_index: int
) -> tuple[CharacterStrategyPrior, ...]:
    out: list[CharacterStrategyPrior] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            out.append(
                CharacterStrategyPrior(
                    rule_id=str(
                        item.get("rule_id")
                        or f"llm-rule:{chunk_index}:{index}"
                    ),
                    problem_pattern=str(
                        item.get("problem_pattern") or "extracted-pattern"
                    ),
                    recommended_regime=item.get("recommended_regime"),
                    recommended_ordering=tuple(
                        str(t)
                        for t in item.get("recommended_ordering", ("step_one",))
                    ),
                    recommended_pacing=str(
                        item.get("recommended_pacing", "standard")
                    ),
                    avoid_patterns=tuple(
                        str(t)
                        for t in item.get("avoid_patterns", ("rushed",))
                    ),
                    applicability_scope=tuple(
                        str(t)
                        for t in item.get("applicability_scope", ("risk-low",))
                    ),
                    confidence=float(item.get("confidence", 0.55)),
                    description=str(
                        item.get("description") or "Extracted strategy (review)"
                    ),
                )
            )
        except (ValueError, TypeError):
            continue
    return tuple(out)


def _coerce_boundary_priors(
    items: list[Any], *, chunk_index: int
) -> tuple[CharacterBoundaryPrior, ...]:
    out: list[CharacterBoundaryPrior] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            out.append(
                CharacterBoundaryPrior(
                    boundary_id=str(
                        item.get("boundary_id")
                        or f"llm-boundary:{chunk_index}:{index}"
                    ),
                    regime_id=item.get("regime_id"),
                    trigger_reasons=tuple(
                        str(t)
                        for t in item.get(
                            "trigger_reasons", ("test-trigger",)
                        )
                    ),
                    answer_depth_limit_hint=str(
                        item.get("answer_depth_limit_hint", "standard")
                    ),
                    clarification_required=bool(
                        item.get("clarification_required", False)
                    ),
                    refer_out_required=bool(
                        item.get("refer_out_required", False)
                    ),
                    blocked_topics=tuple(
                        str(t) for t in item.get("blocked_topics", ())
                    ),
                    required_disclaimers=tuple(
                        str(t)
                        for t in item.get("required_disclaimers", ())
                    ),
                    confidence=float(item.get("confidence", 0.6)),
                    description=str(
                        item.get("description")
                        or "Extracted boundary (please review)"
                    ),
                )
            )
        except (ValueError, TypeError):
            continue
    return tuple(out)


def _coerce_drive_priors(items: list[Any]) -> tuple[CharacterDrivePrior, ...]:
    out: list[CharacterDrivePrior] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            band_raw = item.get("homeostatic_band", [0.4, 0.8])
            band = tuple(float(b) for b in band_raw)
            if len(band) != 2:
                continue
            out.append(
                CharacterDrivePrior(
                    name=str(item.get("name") or "extracted_drive"),
                    target=float(item.get("target", 0.5)),
                    homeostatic_band=band,
                    decay_per_tick=float(item.get("decay_per_tick", 0.005)),
                    pe_weight=float(item.get("pe_weight", 0.5)),
                    initial_level=float(item.get("initial_level", 0.5)),
                    recharge_per_turn=float(
                        item.get("recharge_per_turn", 0.0)
                    ),
                    recharge_per_regime=tuple(
                        (str(name), float(value))
                        for name, value in item.get(
                            "recharge_per_regime", []
                        )
                    ),
                )
            )
        except (ValueError, TypeError):
            continue
    return tuple(out)


def _split_for_extraction(text: str, *, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + chunk_size, len(text))
        chunks.append(text[cursor:end])
        cursor = end
    return chunks


def _dedupe(items: Iterable[Any], *, key) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        ident = key(item)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(item)
    return out


__all__ = [
    "ReviewedProfileCandidate",
    "extract_profile_candidate",
    "review_profile_candidate",
]
