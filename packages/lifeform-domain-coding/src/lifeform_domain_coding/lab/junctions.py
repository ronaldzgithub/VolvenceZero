"""Junction corpus construction (coding-lab Packet 3 前置).

A *junction* is one hand decision point inside a logged episode: the
state the hand could observe, and the discrete move class it chose
next. The corpus is built strictly from trajectory JSONL files (the
replay substrate — never from reruns) and is labeled **contrastively**:
a state key only yields a training junction when the log contains both
a passing branch and a failing branch whose chosen moves differ, so
labels encode an action *difference*, not survivor noise (plan risk
R-CL4).

Action vocabulary is a protocol enum over the five sandbox tools —
exact tool-name matching here is protocol dispatch, not semantic
keyword routing.
"""

from __future__ import annotations

import hashlib
import pathlib

from collections import Counter, defaultdict
from dataclasses import dataclass

from lifeform_domain_coding.lab.trajectory import read_trajectory

ACTION_INVESTIGATE = "investigate"
ACTION_EDIT = "edit"
ACTION_TEST = "test"
ACTION_SUBMIT = "submit"
#: A tool request outside the sandbox protocol (real API hands do emit
#: these; the episode runner logs UnknownTool and continues). Kept in
#: junction histories because it is a real move the hand made, but
#: excluded from contrastive labels: the steering action space cannot
#: contain "call a nonexistent tool".
ACTION_INVALID = "invalid"

JUNCTION_ACTIONS: tuple[str, ...] = (
    ACTION_INVESTIGATE,
    ACTION_EDIT,
    ACTION_TEST,
    ACTION_SUBMIT,
)

_TOOL_TO_ACTION: dict[str, str] = {
    "read_file": ACTION_INVESTIGATE,
    "list_dir": ACTION_INVESTIGATE,
    "grep": ACTION_INVESTIGATE,
    "write_file": ACTION_EDIT,
    "run_test": ACTION_TEST,
}


@dataclass(frozen=True)
class JunctionRecord:
    """One decision point lifted from one episode log.

    The protocol state is published both as a grouping key
    (``state_key``) and as structured fields, so downstream feature
    builders never parse the key string back apart.
    """

    junction_id: str
    state_key: str
    state_text: str
    action_taken: str
    episode_passed: bool
    decisions_to_end: int
    provenance: str
    trajectory_sha256: str
    category: str
    reads_bucket: int
    has_edited: bool
    test_state: str
    #: Goal-stripped observation (no category, no task description) and
    #: its goal-revealing counterpart. The S3-E replication reads the
    #: latent condition out of ``revealed_text`` residuals while the
    #: gate/scorer only ever see ``observation_text`` — same structure
    #: as the ETA goal-ambiguous junction protocol. Neither carries the
    #: "\nNext move:" suffix; capture appends it.
    observation_text: str = ""
    revealed_text: str = ""


@dataclass(frozen=True)
class ContrastiveJunction:
    """A state key with an evidence-backed action difference.

    ``expert_action`` comes from the shortest passing branch at this
    state; ``non_expert_actions`` are moves that failing branches chose
    at the same state and the passing branch did not.
    """

    state_key: str
    state_text: str
    expert_action: str
    non_expert_actions: tuple[str, ...]
    expert_provenance: str
    non_expert_provenances: tuple[str, ...]
    passing_records: int
    failing_records: int


def _state_key(
    *,
    category: str,
    investigate_count: int,
    has_edited: bool,
    test_state: str,
) -> str:
    reads_bucket = min(investigate_count, 3)
    return f"{category}|reads={reads_bucket}|edited={int(has_edited)}|tests={test_state}"


def _state_text(
    *,
    category: str,
    description: str,
    history: tuple[str, ...],
    test_state: str,
) -> str:
    """Bounded observation text for NLL scoring and residual capture.

    Must comfortably fit the frozen scorer's max_length window, so the
    description is trimmed and only the recent move classes are kept.
    """

    recent = ",".join(history[-6:]) if history else "none"
    return (
        f"[coding junction] category={category} moves_so_far={recent} "
        f"tests={test_state}\ntask: {description[:400]}\nNext move:"
    )


def _observation_pair(
    *,
    category: str,
    description: str,
    history: tuple[str, ...],
    test_state: str,
) -> tuple[str, str]:
    """(goal-stripped observation, goal-revealing context), no suffix."""

    recent = ",".join(history[-6:]) if history else "none"
    stripped = f"[coding junction] moves_so_far={recent} tests={test_state}"
    revealed = (
        f"[coding junction] category={category} moves_so_far={recent} "
        f"tests={test_state}\ntask: {description[:400]}"
    )
    return stripped, revealed


class IncompleteTrajectoryError(ValueError):
    """Trajectory lacks its ``oracle_outcome`` terminal event.

    Raised for logs still being written by a live episode (or truncated
    by a crash). Corpus collection skips these explicitly and reports
    the count — an unfinished log is not evidence, but it is also not a
    contract violation.
    """


def extract_junctions(trajectory_path: pathlib.Path) -> tuple[JunctionRecord, ...]:
    """Lift every decision point out of one episode trajectory."""

    path = pathlib.Path(trajectory_path)
    events = read_trajectory(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    presented = next(e for e in events if e["event_type"] == "task_presented")
    category = str(presented["payload"]["category"])
    description = str(presented["payload"]["description"])
    outcomes = [e for e in events if e["event_type"] == "oracle_outcome"]
    if not outcomes:
        raise IncompleteTrajectoryError(
            f"trajectory has no oracle_outcome (still in flight?): {path!s}"
        )
    passed = bool(outcomes[0]["payload"]["passed"])

    decisions = [e for e in events if e["event_type"] == "hand_decision"]
    tool_results = {
        e["payload"]["step_index"]: e["payload"]
        for e in events
        if e["event_type"] == "tool_result"
    }

    records: list[JunctionRecord] = []
    investigate_count = 0
    has_edited = False
    test_state = "none"
    history: list[str] = []
    total = len(decisions)
    for index, event in enumerate(decisions):
        payload = event["payload"]
        if payload["kind"] == "submit":
            action = ACTION_SUBMIT
        else:
            tool_name = str(payload["tool_name"])
            action = _TOOL_TO_ACTION.get(tool_name, ACTION_INVALID)
        key = _state_key(
            category=category,
            investigate_count=investigate_count,
            has_edited=has_edited,
            test_state=test_state,
        )
        text = _state_text(
            category=category,
            description=description,
            history=tuple(history),
            test_state=test_state,
        )
        stripped, revealed = _observation_pair(
            category=category,
            description=description,
            history=tuple(history),
            test_state=test_state,
        )
        provenance = f"{path.parent.parent.name}/{path.stem}#d{index}"
        records.append(
            JunctionRecord(
                junction_id=f"{sha256[:12]}-d{index:03d}",
                state_key=key,
                state_text=text,
                action_taken=action,
                episode_passed=passed,
                decisions_to_end=total - index,
                provenance=provenance,
                trajectory_sha256=sha256,
                category=category,
                reads_bucket=min(investigate_count, 3),
                has_edited=has_edited,
                test_state=test_state,
                observation_text=stripped,
                revealed_text=revealed,
            )
        )
        # Advance protocol-level state.
        history.append(action)
        if action == ACTION_INVESTIGATE:
            investigate_count += 1
        elif action == ACTION_EDIT:
            has_edited = True
        elif action == ACTION_TEST:
            step_index = payload["step_index"]
            result = tool_results.get(step_index)
            if result is None or not result["succeeded"]:
                test_state = "failed"
            else:
                exit_code = result["result"].get("exit_code")
                test_state = "passed" if exit_code == 0 else "failed"
    return tuple(records)


def collect_junctions(
    trajectory_paths: tuple[pathlib.Path, ...],
) -> tuple[JunctionRecord, ...]:
    """Extract junctions from settled trajectories.

    In-flight / truncated logs (no ``oracle_outcome``) are skipped with
    a printed count — an explicit, observable policy so live runs can
    coexist with corpus builds without masking real corruption.
    """

    records: list[JunctionRecord] = []
    incomplete = 0
    for path in trajectory_paths:
        try:
            records.extend(extract_junctions(path))
        except IncompleteTrajectoryError:
            incomplete += 1
    if incomplete:
        print(f"[junctions] skipped {incomplete} in-flight trajectories")
    return tuple(records)


def build_contrastive_corpus(
    records: tuple[JunctionRecord, ...],
) -> tuple[ContrastiveJunction, ...]:
    """Group by state key and keep only evidence-backed action differences."""

    by_key: dict[str, list[JunctionRecord]] = defaultdict(list)
    for record in records:
        by_key[record.state_key].append(record)

    corpus: list[ContrastiveJunction] = []
    for key in sorted(by_key):
        group = by_key[key]
        passing = [r for r in group if r.episode_passed]
        failing = [r for r in group if not r.episode_passed]
        if not passing or not failing:
            continue
        # Expert = the modal move among passing branches, ties broken by
        # the shortest branch (fewest decisions to episode end). Invalid
        # moves never become labels on either side.
        passing = [r for r in passing if r.action_taken in JUNCTION_ACTIONS]
        failing = [r for r in failing if r.action_taken in JUNCTION_ACTIONS]
        if not passing or not failing:
            continue
        pass_counts = Counter(r.action_taken for r in passing)
        top_count = max(pass_counts.values())
        candidates = {a for a, c in pass_counts.items() if c == top_count}
        best = min(
            (r for r in passing if r.action_taken in candidates),
            key=lambda r: (r.decisions_to_end, r.provenance),
        )
        expert_action = best.action_taken
        non_expert = sorted(
            {r.action_taken for r in failing} - {expert_action}
        )
        if not non_expert:
            continue
        corpus.append(
            ContrastiveJunction(
                state_key=key,
                state_text=best.state_text,
                expert_action=expert_action,
                non_expert_actions=tuple(non_expert),
                expert_provenance=best.provenance,
                non_expert_provenances=tuple(
                    r.provenance for r in failing if r.action_taken in non_expert
                )[:8],
                passing_records=len(passing),
                failing_records=len(failing),
            )
        )
    return tuple(corpus)


def split_corpus(
    corpus: tuple[ContrastiveJunction, ...],
    *,
    eval_fraction: float = 0.3,
) -> tuple[tuple[ContrastiveJunction, ...], tuple[ContrastiveJunction, ...]]:
    """Deterministic content-addressed train/eval split by state key."""

    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    train: list[ContrastiveJunction] = []
    evaluation: list[ContrastiveJunction] = []
    threshold = int(eval_fraction * 2**32)
    for junction in corpus:
        digest = hashlib.sha256(junction.state_key.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big")
        (evaluation if bucket < threshold else train).append(junction)
    return tuple(train), tuple(evaluation)


def corpus_manifest(
    records: tuple[JunctionRecord, ...],
    corpus: tuple[ContrastiveJunction, ...],
) -> dict:
    """Accounting surface for prereg / audit artifacts."""

    return {
        "junction_records": len(records),
        "distinct_state_keys": len({r.state_key for r in records}),
        "contrastive_junctions": len(corpus),
        "action_distribution": dict(Counter(r.action_taken for r in records)),
        "expert_action_distribution": dict(
            Counter(j.expert_action for j in corpus)
        ),
        "source_trajectories": len({r.trajectory_sha256 for r in records}),
    }


__all__ = [
    "ACTION_EDIT",
    "ACTION_INVALID",
    "ACTION_INVESTIGATE",
    "ACTION_SUBMIT",
    "ACTION_TEST",
    "JUNCTION_ACTIONS",
    "ContrastiveJunction",
    "IncompleteTrajectoryError",
    "JunctionRecord",
    "build_contrastive_corpus",
    "collect_junctions",
    "corpus_manifest",
    "extract_junctions",
    "split_corpus",
]
