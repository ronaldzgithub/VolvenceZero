"""Junction corpus mechanics (Packet 3 前置).

Covers extraction from trajectory logs, protocol-state advancement,
contrastive labeling (including the invalid-move exclusion), and the
deterministic split.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from lifeform_domain_coding.lab.junctions import (
    ACTION_EDIT,
    ACTION_INVALID,
    ACTION_INVESTIGATE,
    build_contrastive_corpus,
    collect_junctions,
    corpus_manifest,
    extract_junctions,
    split_corpus,
)


def _write_trajectory(
    path: pathlib.Path,
    *,
    category: str,
    passed: bool,
    moves: tuple[tuple[str, bool], ...],
) -> pathlib.Path:
    """moves = ((tool_name_or_'submit', tool_succeeded), ...)."""

    events: list[dict] = [
        {
            "event_index": 0,
            "event_type": "task_presented",
            "monotonic_seconds": 0.0,
            "payload": {
                "task_id": f"task-{category}",
                "category": category,
                "description": f"do the {category} change",
                "episode_index": 0,
                "hand_id": "fixture",
            },
        }
    ]
    step = 0
    for tool, succeeded in moves:
        if tool == "submit":
            events.append(
                {
                    "event_index": len(events),
                    "event_type": "hand_decision",
                    "monotonic_seconds": 0.0,
                    "payload": {
                        "step_index": step,
                        "kind": "submit",
                        "tool_name": "",
                        "parameters": {},
                        "note": "",
                        "metadata": {},
                    },
                }
            )
            step += 1
            continue
        events.append(
            {
                "event_index": len(events),
                "event_type": "hand_decision",
                "monotonic_seconds": 0.0,
                "payload": {
                    "step_index": step,
                    "kind": "tool",
                    "tool_name": tool,
                    "parameters": {},
                    "note": "",
                    "metadata": {},
                },
            }
        )
        result = {"exit_code": 0} if tool == "run_test" and succeeded else {}
        events.append(
            {
                "event_index": len(events),
                "event_type": "tool_result",
                "monotonic_seconds": 0.0,
                "payload": {
                    "step_index": step,
                    "tool_name": tool,
                    "succeeded": succeeded,
                    "result_keys": sorted(result),
                    "result": result,
                },
            }
        )
        step += 1
    events.append(
        {
            "event_index": len(events),
            "event_type": "oracle_outcome",
            "monotonic_seconds": 0.0,
            "payload": {
                "task_id": f"task-{category}",
                "passed": passed,
                "acceptance_passed": passed,
                "regression_passed": passed,
                "failed_test_ids": [],
                "error_test_ids": [],
                "invariant_violations": [],
                "submitted": True,
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n",
        encoding="utf-8",
    )
    return path


def test_extract_tracks_protocol_state(tmp_path: pathlib.Path) -> None:
    path = _write_trajectory(
        tmp_path / "e0.jsonl",
        category="fix_bug",
        passed=True,
        moves=(
            ("read_file", True),
            ("write_file", True),
            ("run_test", True),
            ("submit", True),
        ),
    )
    records = extract_junctions(path)
    assert [r.action_taken for r in records] == [
        "investigate",
        "edit",
        "test",
        "submit",
    ]
    # State key advances: first decision sees reads=0/edited=0, the
    # submit decision sees reads=1/edited=1/tests=passed.
    assert records[0].state_key == "fix_bug|reads=0|edited=0|tests=none"
    assert records[-1].state_key == "fix_bug|reads=1|edited=1|tests=passed"
    assert all(r.episode_passed for r in records)
    assert records[0].decisions_to_end == 4


def test_unknown_tool_is_invalid_but_recorded(tmp_path: pathlib.Path) -> None:
    path = _write_trajectory(
        tmp_path / "e1.jsonl",
        category="fix_bug",
        passed=False,
        moves=(("glob", False), ("submit", True)),
    )
    records = extract_junctions(path)
    assert records[0].action_taken == ACTION_INVALID


def test_contrastive_labels_take_action_difference(tmp_path: pathlib.Path) -> None:
    # Same starting state: the passing branch investigates first, the
    # failing branch edits blindly.
    paths = (
        _write_trajectory(
            tmp_path / "pass.jsonl",
            category="fix_bug",
            passed=True,
            moves=(("read_file", True), ("write_file", True), ("submit", True)),
        ),
        _write_trajectory(
            tmp_path / "fail.jsonl",
            category="fix_bug",
            passed=False,
            moves=(("write_file", True), ("submit", True)),
        ),
    )
    records = collect_junctions(paths)
    corpus = build_contrastive_corpus(records)
    assert len(corpus) == 1
    junction = corpus[0]
    assert junction.state_key == "fix_bug|reads=0|edited=0|tests=none"
    assert junction.expert_action == ACTION_INVESTIGATE
    assert junction.non_expert_actions == (ACTION_EDIT,)
    assert junction.passing_records == 1 and junction.failing_records == 1


def test_invalid_moves_never_become_labels(tmp_path: pathlib.Path) -> None:
    paths = (
        _write_trajectory(
            tmp_path / "pass.jsonl",
            category="fix_bug",
            passed=True,
            moves=(("read_file", True), ("submit", True)),
        ),
        _write_trajectory(
            tmp_path / "fail.jsonl",
            category="fix_bug",
            passed=False,
            moves=(("glob", False), ("submit", True)),
        ),
    )
    corpus = build_contrastive_corpus(collect_junctions(paths))
    # The invalid move itself never becomes a label — but the failing
    # branch's subsequent in-vocabulary submit (the invalid probe does
    # not advance protocol state) still contrasts against investigate.
    assert len(corpus) == 1
    assert corpus[0].expert_action == ACTION_INVESTIGATE
    assert corpus[0].non_expert_actions == ("submit",)
    assert ACTION_INVALID not in corpus[0].non_expert_actions


def test_identical_actions_yield_no_contrast(tmp_path: pathlib.Path) -> None:
    paths = (
        _write_trajectory(
            tmp_path / "pass.jsonl",
            category="add_helper",
            passed=True,
            moves=(("read_file", True), ("write_file", True), ("submit", True)),
        ),
        _write_trajectory(
            tmp_path / "fail.jsonl",
            category="add_helper",
            passed=False,
            moves=(("read_file", True), ("write_file", True), ("submit", True)),
        ),
    )
    corpus = build_contrastive_corpus(collect_junctions(paths))
    assert corpus == ()


def test_split_is_deterministic_and_partitions(tmp_path: pathlib.Path) -> None:
    paths = []
    for index in range(12):
        paths.append(
            _write_trajectory(
                tmp_path / f"pass-{index}.jsonl",
                category=f"cat{index}",
                passed=True,
                moves=(("read_file", True), ("write_file", True), ("submit", True)),
            )
        )
        paths.append(
            _write_trajectory(
                tmp_path / f"fail-{index}.jsonl",
                category=f"cat{index}",
                passed=False,
                moves=(("write_file", True), ("submit", True)),
            )
        )
    corpus = build_contrastive_corpus(collect_junctions(tuple(paths)))
    assert len(corpus) == 12
    train_a, eval_a = split_corpus(corpus, eval_fraction=0.3)
    train_b, eval_b = split_corpus(corpus, eval_fraction=0.3)
    assert train_a == train_b and eval_a == eval_b
    assert len(train_a) + len(eval_a) == len(corpus)
    assert train_a and eval_a  # both sides populated at n=12

    manifest = corpus_manifest(collect_junctions(tuple(paths)), corpus)
    assert manifest["contrastive_junctions"] == 12
    assert manifest["expert_action_distribution"] == {"investigate": 12}


def test_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        split_corpus((), eval_fraction=1.0)
