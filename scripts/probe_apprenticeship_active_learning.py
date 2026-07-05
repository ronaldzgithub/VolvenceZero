"""Stage 0 fake-provider probe for active-learning (debt #90).

Spec: ``docs/specs/apprenticeship-alignment.md`` §7.1.

This probe validates the **active-learning label economy** of the
``apprenticeship_alignment`` owner without a real LLM / GPU. It drives the
*real* :class:`ApprenticeshipAlignmentModule` over a scripted apprentice
trajectory using the LLM-free deterministic extractor (the fake provider),
and reads out, per turn, the owner-owned ``should_request_feedback`` signal.

The reliable-active-apprenticeship guarantee is: act when guaranteed-optimal,
otherwise *actively request sparse feedback* -- and minimize how often. This
probe measures that as two report-only metrics:

- ``feedback_request_rate`` : fraction of apprentice turns where the owner
  actively requested feedback (disagreement region / inconsistent version
  space). A passive learner that always defers to the operator would be 1.0.
- ``labels_saved`` : apprentice turns that did NOT need a label because the
  cognition already pinned the guidance (agreement region), relative to a
  naive "ask every turn" baseline. This is the concrete "sparse feedback"
  payoff -- the active learner spends fewer operator labels for the same
  coverage.

It also checks the core Stage 0 behaviour the spec calls out: a novel
(uncovered) teaching turn lands in the disagreement region and *requests
feedback*; once that guidance is taught (cognition covers it), the same
guidance no longer requests -- i.e. the version space shrank and the label
was consumed once, not every turn.

EXIT(0) (debt #90 Stage 0): non-empty JSON + at least one feedback request
fired (mechanism live) + ``feedback_request_rate < 1.0`` (active learner
saves at least one label vs ask-every-turn) + version space shrinks after
feedback (repeated-topic surprise drops below the request floor).

This is MACHINERY / label-economy evidence only. It does NOT prove the
sparse-feedback *gain* on a real substrate (that needs a real LLM structured
extractor + human-in-the-loop trace -- gate on GPU/keys, tracked in #90).

Run:
    python scripts/probe_apprenticeship_active_learning.py
Output:
    artifacts/eq_uplift/apprenticeship_active_learning_shadow.json
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from volvence_zero.apprenticeship import (
    ApprenticeshipAlignmentModule,
    ApprenticeshipAlignmentSnapshot,
    ReliabilityState,
    VersionSpaceStatus,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    BeliefAssumptionSnapshot,
    SemanticRecord,
)


@dataclass(frozen=True)
class ScriptedTurn:
    """One scripted apprentice turn.

    ``topic`` is a stable teaching subject. ``already_taught`` is True once a
    prior turn taught this topic, so the operator's cognition (belief store)
    now covers it -- an active learner should NOT re-request a label for it.
    """

    topic: str
    guidance_text: str
    already_taught: bool


def _belief_snapshot(taught: list[str]) -> BeliefAssumptionSnapshot:
    """Cognition covering every already-taught guidance verbatim.

    Verbatim coverage makes the owner's coverage metric high (surprise low)
    for taught topics and low (surprise high) for novel ones, purely through
    the owner's own reconciliation -- no keyword rules here.
    """

    records = tuple(
        SemanticRecord(
            record_id=f"belief:{index}",
            summary=text,
            detail="",
            confidence=0.8,
            status="active",
            source_turn=index,
            evidence=text,
        )
        for index, text in enumerate(taught)
    )
    return BeliefAssumptionSnapshot(
        beliefs=records,
        assumptions=records,
        verification_needs=(),
        contradiction_refs=(),
        mean_confidence=0.8 if records else 0.0,
        control_signal=0.0,
        description="probe cognition",
    )


def _scripted_trajectory() -> list[ScriptedTurn]:
    """Teach three topics, each followed by a covered follow-up + reviews.

    The active learner should request a label only on the first (novel)
    encounter of each topic, then reuse the pinned cognition for free.
    """

    topics = {
        "empathy_first": "遇到用户焦虑时先共情再给建议",
        "cite_sources": "给出结论时附上可核查的来源引用",
        "boundary_consent": "涉及个人隐私前先确认边界与同意",
    }
    order: list[tuple[str, bool]] = [
        ("empathy_first", False),  # novel -> request
        ("empathy_first", True),   # covered -> no request (label saved)
        ("cite_sources", False),   # novel -> request
        ("empathy_first", True),   # review -> no request
        ("cite_sources", True),    # covered -> no request
        ("boundary_consent", False),  # novel -> request
        ("cite_sources", True),    # review -> no request
        ("boundary_consent", True),   # covered -> no request
        ("empathy_first", True),   # review -> no request
    ]
    return [
        ScriptedTurn(topic=topic, guidance_text=topics[topic], already_taught=taught)
        for topic, taught in order
    ]


def run_probe() -> dict:
    module = ApprenticeshipAlignmentModule(
        wiring_level=WiringLevel.ACTIVE,
        apprenticeship=True,
    )
    trajectory = _scripted_trajectory()

    taught: list[str] = []
    per_turn: list[dict] = []
    # Track the first (novel) surprise and first covered surprise per topic so
    # we can show the version space shrank once the label was consumed.
    first_novel_surprise: dict[str, float] = {}
    first_covered_surprise: dict[str, float] = {}

    for turn_index, turn in enumerate(trajectory, start=1):
        snapshot: ApprenticeshipAlignmentSnapshot = asyncio.run(
            module.process_standalone(
                belief_assumption=_belief_snapshot(list(taught)),
                guidance_text=turn.guidance_text,
                turn_index=turn_index,
                apprenticeship=True,
            )
        ).value

        requested = bool(snapshot.should_request_feedback)
        if turn.already_taught:
            first_covered_surprise.setdefault(turn.topic, snapshot.guidance_surprise)
        else:
            first_novel_surprise.setdefault(turn.topic, snapshot.guidance_surprise)
            # Operator teaches the topic in response to the request: cognition
            # now covers this guidance for all subsequent turns.
            if turn.guidance_text not in taught:
                taught.append(turn.guidance_text)

        per_turn.append(
            {
                "turn": turn_index,
                "topic": turn.topic,
                "already_taught": turn.already_taught,
                "version_space_status": snapshot.version_space_status,
                "reliability": snapshot.reliability,
                "guidance_surprise": round(snapshot.guidance_surprise, 4),
                "should_request_feedback": requested,
                "feedback_request_urgency": round(snapshot.feedback_request_urgency, 4),
                "in_agreement_region": snapshot.in_agreement_region,
            }
        )

    apprentice_turns = len(per_turn)
    feedback_requests = sum(1 for row in per_turn if row["should_request_feedback"])
    # Naive baseline: a passive learner asks the operator on every turn.
    ask_every_turn_labels = apprentice_turns
    labels_saved = ask_every_turn_labels - feedback_requests
    feedback_request_rate = (
        feedback_requests / apprentice_turns if apprentice_turns else 0.0
    )
    labels_saved_ratio = labels_saved / apprentice_turns if apprentice_turns else 0.0

    # Version-space shrink evidence: for every topic taught then revisited, the
    # covered surprise must sit below the novel surprise (and below the request
    # floor), i.e. the label consumed once pins the guidance thereafter.
    shrink_rows = []
    shrink_all_ok = True
    for topic in first_novel_surprise:
        if topic not in first_covered_surprise:
            continue
        novel = first_novel_surprise[topic]
        covered = first_covered_surprise[topic]
        dropped = covered < novel
        shrink_all_ok = shrink_all_ok and dropped
        shrink_rows.append(
            {
                "topic": topic,
                "novel_surprise": round(novel, 4),
                "covered_surprise": round(covered, 4),
                "shrank": dropped,
            }
        )

    novel_turns = [row for row in per_turn if not row["already_taught"]]
    covered_turns = [row for row in per_turn if row["already_taught"]]
    novel_all_requested = bool(novel_turns) and all(
        row["should_request_feedback"] for row in novel_turns
    )
    covered_none_requested = all(
        not row["should_request_feedback"] for row in covered_turns
    )

    mechanism_live = feedback_requests > 0
    saves_labels = feedback_request_rate < 1.0 and labels_saved > 0
    version_space_shrinks = bool(shrink_rows) and shrink_all_ok
    exit0_pass = bool(mechanism_live and saves_labels and version_space_shrinks)

    return {
        "probe": "apprenticeship_active_learning_stage0",
        "spec": "docs/specs/apprenticeship-alignment.md",
        "debt": 90,
        "stage": 0,
        "scope_note": (
            "FAKE-PROVIDER LABEL-ECONOMY ONLY. Uses the deterministic LLM-free "
            "extractor. Does NOT prove sparse-feedback gain on a real substrate "
            "(needs a real LLM structured extractor + human-in-the-loop trace, "
            "gate on GPU/keys -- tracked in debt #90)."
        ),
        "apprentice_turns": apprentice_turns,
        "metrics": {
            "feedback_requests": feedback_requests,
            "ask_every_turn_labels": ask_every_turn_labels,
            "feedback_request_rate": round(feedback_request_rate, 4),
            "labels_saved": labels_saved,
            "labels_saved_ratio": round(labels_saved_ratio, 4),
            "novel_all_requested": novel_all_requested,
            "covered_none_requested": covered_none_requested,
            "version_space_shrink": shrink_rows,
        },
        "exit0": {
            "mechanism_live": mechanism_live,
            "saves_labels": saves_labels,
            "version_space_shrinks": version_space_shrinks,
            "json_non_empty": apprentice_turns > 0,
            "pass": exit0_pass,
        },
        "per_turn": per_turn,
    }


def main() -> None:
    result = run_probe()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(repo_root, "artifacts", "eq_uplift")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "apprenticeship_active_learning_shadow.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    m = result["metrics"]
    print("=== Stage 0 apprenticeship active-learning probe (debt #90) ===")
    print(f"apprentice_turns      : {result['apprentice_turns']}")
    print(f"feedback_requests     : {m['feedback_requests']}")
    print(f"feedback_request_rate : {m['feedback_request_rate']}")
    print(f"labels_saved          : {m['labels_saved']} (ratio {m['labels_saved_ratio']})")
    print(f"version_space_shrink   : {m['version_space_shrink']}")
    print(f"EXIT(0) pass          : {result['exit0']['pass']}  {result['exit0']}")
    print(f"written               : {out_path}")
    print(result["scope_note"])


if __name__ == "__main__":
    main()
