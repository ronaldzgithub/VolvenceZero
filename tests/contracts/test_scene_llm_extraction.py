"""Wave T8 contract tests — LLM-assisted scene extraction.

Same fake-provider pattern as Wave T7. Pin:

* Valid LLM JSON yields typed ``NarrativeScene``s with closed-enum
  phase / emotional_register fields.
* Schema-mismatch on individual chunks recorded in notes.
* Total parse failure raises.
* ``review_arc_candidate`` produces a typed :class:`NarrativeArc`
  whose schema invariants (>= 5 scenes, monotonic phase boundaries,
  unique scene ids) hold automatically — the reviewer cannot
  accidentally produce an invalid arc.
"""

from __future__ import annotations

import pytest

from lifeform_domain_character import (
    NarrativeArc,
    NarrativeArcCandidate,
    build_zhang_wuji_profile,
    extract_arc_candidate,
    review_arc_candidate,
)


def _scene_object(
    scene_id: str,
    *,
    phase: str,
    register: str = "calm",
    setting: str = "你站在山下。",
    decision: str = "你怎么做？",
    action: str = "你停下脚步，倾听。",
    outcome: str = "你听到了风声。",
) -> dict:
    return {
        "scene_id": scene_id,
        "phase_label": phase,
        "setting": setting,
        "decision_point": decision,
        "canonical_action": action,
        "canonical_outcome": outcome,
        "emotional_register": register,
        "risk_markers": ["risk-low"],
        "expected_regime": None,
        "evidence_locator": "test",
        "low_confidence": False,
    }


def _build_payload(scenes: list[dict], *, confidence: float = 0.8) -> str:
    import json

    return json.dumps({"scenes": scenes, "inference_confidence": confidence})


class _FakeProvider:
    def __init__(self, payloads: tuple[str, ...]) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 1536,
        temperature: float = 0.0,
    ) -> str:
        del prompt, max_new_tokens, temperature
        if not self._payloads:
            raise RuntimeError("FakeProvider exhausted")
        self.calls += 1
        return self._payloads.pop(0)


def test_extract_arc_candidate_parses_valid_response() -> None:
    payload = _build_payload(
        [
            _scene_object("s-1", phase="child"),
            _scene_object("s-2", phase="adolescent", register="resolve"),
        ]
    )
    provider = _FakeProvider(payloads=(payload,))
    candidate = extract_arc_candidate(
        novel_text="some short novel text",
        profile=build_zhang_wuji_profile(),
        llm_runtime=provider,
        arc_id="test-arc",
        chunk_size=2000,
    )
    assert isinstance(candidate, NarrativeArcCandidate)
    assert len(candidate.scenes) == 2
    assert {s.scene_id for s in candidate.scenes} == {"s-1", "s-2"}


def test_extract_arc_candidate_skips_invalid_phase_label() -> None:
    payload = _build_payload(
        [
            _scene_object("good-scene", phase="child"),
            _scene_object("bad-scene", phase="middle-aged"),  # invalid
        ]
    )
    provider = _FakeProvider(payloads=(payload,))
    candidate = extract_arc_candidate(
        novel_text="text",
        profile=build_zhang_wuji_profile(),
        llm_runtime=provider,
        arc_id="test-arc",
    )
    ids = {s.scene_id for s in candidate.scenes}
    assert "good-scene" in ids
    assert "bad-scene" not in ids


def test_extract_arc_candidate_skips_missing_required_fields() -> None:
    """A scene missing decision_point is silently dropped; the chunk
    is still considered parsed (top-level JSON is fine), so the
    candidate comes back with empty ``scenes``. The downstream
    ``review_arc_candidate`` will then reject the candidate via the
    NarrativeArc >= 5 scenes invariant."""
    incomplete = {
        "scene_id": "no-decision",
        "phase_label": "child",
        "setting": "你在桥边。",
        # decision_point missing
        "canonical_action": "你停下。",
        "canonical_outcome": "什么也没发生。",
        "emotional_register": "calm",
    }
    import json

    payload = json.dumps({"scenes": [incomplete], "inference_confidence": 0.5})
    provider = _FakeProvider(payloads=(payload,))
    candidate = extract_arc_candidate(
        novel_text="text",
        profile=build_zhang_wuji_profile(),
        llm_runtime=provider,
        arc_id="test-arc",
    )
    # Incomplete scene was dropped during coercion, so the candidate
    # has 0 scenes. The reviewer must reject this candidate.
    assert candidate.scenes == ()
    with pytest.raises(ValueError, match=">= 5 scenes"):
        review_arc_candidate(
            candidate, reviewer="r", review_locator="loc"
        )


def test_extract_arc_candidate_records_unparseable_chunk_in_notes() -> None:
    payload_good = _build_payload(
        [_scene_object(f"s-{i}", phase="child") for i in range(3)]
    )
    provider = _FakeProvider(payloads=("not json", payload_good))
    candidate = extract_arc_candidate(
        novel_text="text " * 600,
        profile=build_zhang_wuji_profile(),
        llm_runtime=provider,
        arc_id="test-arc",
        chunk_size=1500,
    )
    assert "unparseable response" in " ".join(candidate.notes)
    assert len(candidate.scenes) == 3


def test_extract_arc_candidate_raises_on_total_failure() -> None:
    provider = _FakeProvider(payloads=("trash",) * 3)
    with pytest.raises(ValueError, match="no chunk yielded"):
        extract_arc_candidate(
            novel_text="text " * 1000,
            profile=build_zhang_wuji_profile(),
            llm_runtime=provider,
            arc_id="test-arc",
            chunk_size=1500,
        )


def test_review_arc_candidate_produces_typed_arc() -> None:
    scenes = [
        _scene_object("s-1", phase="child"),
        _scene_object("s-2", phase="child"),
        _scene_object("s-3", phase="adolescent"),
        _scene_object("s-4", phase="adolescent"),
        _scene_object("s-5", phase="mature"),
    ]
    provider = _FakeProvider(payloads=(_build_payload(scenes),))
    candidate = extract_arc_candidate(
        novel_text="text",
        profile=build_zhang_wuji_profile(),
        llm_runtime=provider,
        arc_id="test-arc-typed",
    )
    arc = review_arc_candidate(
        candidate, reviewer="reviewer-x", review_locator="pr-007"
    )
    assert isinstance(arc, NarrativeArc)
    assert arc.arc_id == "test-arc-typed"
    assert len(arc.scenes) == 5
    # Phase boundaries are monotonic
    rank = {"child": 0, "adolescent": 1, "mature": 2, "elder": 3}
    last_rank = -1
    for _, label in arc.life_phase_boundaries:
        current = rank[label]
        assert current >= last_rank
        last_rank = current


def test_review_arc_candidate_below_minimum_scenes_raises() -> None:
    """If the reviewer's accepted_scene_ids drops below the
    NarrativeArc minimum (5 scenes), the underlying schema rejects."""
    scenes = [
        _scene_object(f"s-{i}", phase="child") for i in range(5)
    ]
    provider = _FakeProvider(payloads=(_build_payload(scenes),))
    candidate = extract_arc_candidate(
        novel_text="text",
        profile=build_zhang_wuji_profile(),
        llm_runtime=provider,
        arc_id="test-arc-minimum",
    )
    with pytest.raises(ValueError, match=">= 5 scenes"):
        review_arc_candidate(
            candidate,
            reviewer="r",
            review_locator="pr",
            accepted_scene_ids=("s-1", "s-2"),
        )


def test_review_arc_candidate_empty_reviewer_raises() -> None:
    scenes = [_scene_object(f"s-{i}", phase="child") for i in range(5)]
    provider = _FakeProvider(payloads=(_build_payload(scenes),))
    candidate = extract_arc_candidate(
        novel_text="text",
        profile=build_zhang_wuji_profile(),
        llm_runtime=provider,
        arc_id="test-arc-rev",
    )
    with pytest.raises(ValueError, match="reviewer"):
        review_arc_candidate(
            candidate, reviewer="   ", review_locator="loc"
        )
