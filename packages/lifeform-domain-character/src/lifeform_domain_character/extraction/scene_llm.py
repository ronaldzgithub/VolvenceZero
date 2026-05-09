"""LLM-assisted narrative scene extraction.

Workflow mirror of profile_llm.py:

1. Caller supplies novel text + a profile + LLM provider.
2. ``extract_arc_candidate`` chunks the text and prompts the LLM
   for a JSON array of decision-point scenes per chunk.
3. Aggregated scene proposals + low-confidence flags wrap into a
   :class:`NarrativeArcCandidate`.
4. Reviewer accepts via ``review_arc_candidate(...)`` which returns
   a typed :class:`NarrativeArc`.

Why two stages (segmentation + structure) collapsed into one prompt:

For v0 we ask the LLM to emit a small array of typed scene objects
per chunk. Future iterations can split into a hierarchical
"chapter index → per-chapter scenes" pipeline; the v0 path is
sufficient for the e2e Wave T11 demo.

What the LLM is asked to produce per chunk:

A JSON object with ``scenes: [...]`` where each entry has the
typed :class:`NarrativeScene` field set:

* ``setting`` — first-person paraphrase ("you stand at the bridge")
* ``decision_point`` — explicit question prompt
* ``canonical_action`` — short reviewer-paraphrase of what the
  protagonist actually did
* ``canonical_outcome`` — short reviewer-paraphrase of the result
* ``emotional_register`` from the closed enum (calm / warm / etc.)
* ``phase_label`` from {child / adolescent / mature / elder}
* optional ``risk_markers`` / ``expected_regime``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, Iterable, Protocol

from lifeform_domain_character.narrative import NarrativeArc, NarrativeScene
from lifeform_domain_character.profile import CharacterSoulProfile


class _GenerateProtocol(Protocol):
    def generate(
        self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...
    ) -> str: ...


_VALID_PHASES: frozenset[str] = frozenset(
    {"child", "adolescent", "mature", "elder"}
)
_VALID_EMOTIONAL_REGISTERS: frozenset[str] = frozenset(
    {
        "calm",
        "warm",
        "tense",
        "crisis",
        "grief",
        "joy",
        "shame",
        "resolve",
        "doubt",
        "wonder",
    }
)


@dataclass(frozen=True)
class NarrativeArcCandidate:
    """LLM-proposed arc awaiting human review."""

    arc_id: str
    character_id: str
    scenes: tuple[NarrativeScene, ...]
    life_phase_boundaries: tuple[tuple[int, str], ...]
    low_confidence_scenes: tuple[str, ...]
    aggregate_inference_confidence: float
    provenance_chunks: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


_SCENE_EXTRACTION_PROMPT = (
    "You are extracting a NarrativeArcCandidate from a novel chunk for the "
    "named protagonist. Return a SINGLE JSON object:\n"
    "{{\n"
    '  "scenes": [\n'
    "    {{\n"
    '      "scene_id": "<short stable id>",\n'
    '      "phase_label": "child|adolescent|mature|elder",\n'
    '      "setting": "<first-person paraphrase: \\"you stand at...\\">",\n'
    '      "decision_point": "<the moment of choice as a direct question>",\n'
    '      "canonical_action": "<short paraphrase of what the protagonist did>",\n'
    '      "canonical_outcome": "<short paraphrase of the result>",\n'
    '      "emotional_register": "calm|warm|tense|crisis|grief|joy|shame|resolve|doubt|wonder",\n'
    '      "risk_markers": ["risk-low"],\n'
    '      "expected_regime": null,\n'
    '      "evidence_locator": "<short pointer into the chunk>",\n'
    '      "low_confidence": false\n'
    "    }}\n"
    "  ],\n"
    '  "inference_confidence": 0.0\n'
    "}}\n"
    "\n"
    "Rules:\n"
    "* Paraphrase. Do NOT include verbatim novel text.\n"
    "* Each scene must have a clear decision_point — skip narrative "
    "passages that are description without a choice.\n"
    "* Keep arrays small (<= 4 scenes per chunk); reviewer will "
    "consolidate across chunks.\n"
    "* Use first-person framing in setting (\"you ...\" in English / "
    "second-person \u4f60 in Chinese), not third-person.\n"
    "* No markdown, no explanation outside the JSON.\n"
    "\n"
    "Protagonist: {character_focus}\n"
    "Novel chunk:\n"
    '"""\n'
    "{novel_chunk}\n"
    '"""'
)


def extract_arc_candidate(
    novel_text: str,
    *,
    profile: CharacterSoulProfile,
    llm_runtime: _GenerateProtocol,
    arc_id: str,
    chunk_size: int = 4000,
    max_new_tokens: int = 1536,
) -> NarrativeArcCandidate:
    """Extract a candidate arc by calling the LLM over chunks.

    Best-effort: chunk-level parse failures are recorded in
    ``notes``; total failure (no chunk parsed) raises ValueError.
    """
    if not novel_text.strip():
        raise ValueError("extract_arc_candidate: novel_text is empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = _split_for_extraction(novel_text, chunk_size=chunk_size)
    aggregated_scenes: list[NarrativeScene] = []
    low_conf: list[str] = []
    confidences: list[float] = []
    notes: list[str] = []
    provenance: list[str] = []
    parsed_any = False
    for index, chunk in enumerate(chunks):
        prompt = _SCENE_EXTRACTION_PROMPT.format(
            character_focus=profile.character_name,
            novel_chunk=chunk[:chunk_size],
        )
        try:
            raw = llm_runtime.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        except Exception as exc:  # pragma: no cover - provider failure note
            notes.append(
                f"chunk {index}: provider raised "
                f"{type(exc).__name__}: {exc!s}"
            )
            continue
        result = _parse_scenes_response(raw, chunk_index=index)
        if result is None:
            notes.append(f"chunk {index}: unparseable response")
            continue
        parsed_any = True
        aggregated_scenes.extend(result.scenes)
        low_conf.extend(result.low_confidence_scene_ids)
        confidences.append(result.inference_confidence)
        provenance.append(
            f"chunk_{index}:offset={index * chunk_size}:"
            f"{chunk[:80].strip()}..."
        )
    if not parsed_any:
        raise ValueError(
            "extract_arc_candidate: no chunk yielded a valid JSON response. "
            f"notes={notes!r}"
        )
    aggregated_scenes = _dedupe_scenes(aggregated_scenes)
    life_phases = _infer_phase_boundaries(aggregated_scenes)
    aggregate_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )
    return NarrativeArcCandidate(
        arc_id=arc_id,
        character_id=profile.profile_id,
        scenes=tuple(aggregated_scenes),
        life_phase_boundaries=life_phases,
        low_confidence_scenes=tuple(sorted(set(low_conf))),
        aggregate_inference_confidence=aggregate_confidence,
        provenance_chunks=tuple(provenance),
        notes=tuple(notes),
    )


def review_arc_candidate(
    candidate: NarrativeArcCandidate,
    *,
    reviewer: str,
    review_locator: str,
    accepted_scene_ids: tuple[str, ...] | None = None,
) -> NarrativeArc:
    """Convert a candidate into a typed :class:`NarrativeArc`.

    The reviewer is the accountable party. They MAY restrict the
    accepted scene set via ``accepted_scene_ids`` (None means accept
    all candidate scenes). The minimum 5-scene rule of NarrativeArc
    still applies — if the accepted set drops below 5 the function
    raises.
    """
    if not reviewer.strip():
        raise ValueError("review_arc_candidate: reviewer must be non-empty")
    if not review_locator.strip():
        raise ValueError(
            "review_arc_candidate: review_locator must be non-empty"
        )
    accepted_set = (
        set(accepted_scene_ids) if accepted_scene_ids is not None else None
    )
    accepted_scenes: tuple[NarrativeScene, ...]
    if accepted_set is None:
        accepted_scenes = candidate.scenes
    else:
        accepted_scenes = tuple(
            scene for scene in candidate.scenes if scene.scene_id in accepted_set
        )
    # Re-derive phase boundaries on the accepted subset so the
    # NarrativeArc's typed invariants hold.
    life_phases = _infer_phase_boundaries(list(accepted_scenes))
    return NarrativeArc(
        arc_id=candidate.arc_id,
        character_id=candidate.character_id,
        scenes=accepted_scenes,
        life_phase_boundaries=life_phases,
        reviewed_by=reviewer,
        source_provenance=review_locator,
    )


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParsedSceneChunk:
    scenes: tuple[NarrativeScene, ...]
    low_confidence_scene_ids: tuple[str, ...]
    inference_confidence: float


def _parse_scenes_response(
    text: str, *, chunk_index: int
) -> _ParsedSceneChunk | None:
    raw = text.strip()
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
    scenes_raw = payload.get("scenes")
    if not isinstance(scenes_raw, list):
        return None
    scenes: list[NarrativeScene] = []
    low_conf: list[str] = []
    for index, item in enumerate(scenes_raw):
        if not isinstance(item, dict):
            continue
        scene = _coerce_scene(item, chunk_index=chunk_index, index=index)
        if scene is None:
            continue
        scenes.append(scene)
        if bool(item.get("low_confidence", False)):
            low_conf.append(scene.scene_id)
    confidence_raw = payload.get("inference_confidence")
    confidence = (
        float(confidence_raw)
        if isinstance(confidence_raw, (int, float))
        else 0.0
    )
    return _ParsedSceneChunk(
        scenes=tuple(scenes),
        low_confidence_scene_ids=tuple(low_conf),
        inference_confidence=confidence,
    )


def _coerce_scene(
    item: dict[str, Any], *, chunk_index: int, index: int
) -> NarrativeScene | None:
    phase = item.get("phase_label")
    register = item.get("emotional_register")
    if phase not in _VALID_PHASES:
        return None
    if register not in _VALID_EMOTIONAL_REGISTERS:
        return None
    setting = str(item.get("setting", "")).strip()
    decision = str(item.get("decision_point", "")).strip()
    action = str(item.get("canonical_action", "")).strip()
    outcome = str(item.get("canonical_outcome", "")).strip()
    if not setting or not decision or not action or not outcome:
        return None
    locator = str(
        item.get("evidence_locator")
        or f"llm-extraction:chunk={chunk_index}:scene={index}"
    )
    risk_markers = tuple(
        str(t) for t in item.get("risk_markers", ("risk-low",))
    )
    expected_regime = item.get("expected_regime")
    expected_regime_str: str | None = (
        str(expected_regime) if isinstance(expected_regime, str) else None
    )
    scene_id = str(
        item.get("scene_id") or f"llm-scene:{chunk_index}:{index}"
    )
    try:
        return NarrativeScene(
            scene_id=scene_id,
            phase_label=str(phase),
            setting=setting,
            decision_point=decision,
            canonical_action=action,
            canonical_outcome=outcome,
            emotional_register=str(register),
            risk_markers=risk_markers,
            expected_regime=expected_regime_str,
            evidence_locator=locator,
        )
    except (ValueError, TypeError):
        return None


def _dedupe_scenes(items: Iterable[NarrativeScene]) -> list[NarrativeScene]:
    seen: set[str] = set()
    out: list[NarrativeScene] = []
    for scene in items:
        if scene.scene_id in seen:
            continue
        seen.add(scene.scene_id)
        out.append(scene)
    return out


_PHASE_ORDER: dict[str, int] = {
    "child": 0,
    "adolescent": 1,
    "mature": 2,
    "elder": 3,
}


def _infer_phase_boundaries(
    scenes: list[NarrativeScene],
) -> tuple[tuple[int, str], ...]:
    """Compute non-decreasing ``(scene_index, phase_label)`` boundaries.

    Walk the scene list once; record (index, phase_label) at each
    transition. If the input is non-monotonic (e.g. mature before
    adolescent) we still emit the highest-rank we've seen so the
    NarrativeArc invariants hold; reviewer is responsible for fixing
    truly out-of-order arcs.
    """
    if not scenes:
        return ()
    boundaries: list[tuple[int, str]] = [(0, scenes[0].phase_label)]
    seen_rank = _PHASE_ORDER.get(scenes[0].phase_label, 0)
    last_label = scenes[0].phase_label
    for index, scene in enumerate(scenes[1:], start=1):
        rank = _PHASE_ORDER.get(scene.phase_label, seen_rank)
        if rank > seen_rank:
            boundaries.append((index, scene.phase_label))
            seen_rank = rank
            last_label = scene.phase_label
        elif scene.phase_label != last_label and rank == seen_rank:
            # Same rank, different label — treat as no-op to keep the
            # boundary list non-decreasing.
            continue
    return tuple(boundaries)


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


__all__ = [
    "NarrativeArcCandidate",
    "extract_arc_candidate",
    "review_arc_candidate",
]
