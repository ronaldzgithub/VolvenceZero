"""Packet 3 S3-E 复刻的行构建与 split 纪律测试（不加载模型）。"""

from __future__ import annotations

import json
import pathlib

import pytest

from lifeform_evolution.coding_lab_s3e import (
    build_coding_junction_rows,
    rows_manifest,
)


def _write_trajectory(
    path: pathlib.Path,
    *,
    category: str,
    passed: bool,
    tools: tuple[str, ...],
) -> pathlib.Path:
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
    for step, tool in enumerate(tools):
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
        events.append(
            {
                "event_index": len(events),
                "event_type": "tool_result",
                "monotonic_seconds": 0.0,
                "payload": {
                    "step_index": step,
                    "tool_name": tool,
                    "succeeded": True,
                    "result_keys": ["exit_code"],
                    "result": {"exit_code": 0},
                },
            }
        )
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


@pytest.fixture()
def corpus_paths(tmp_path: pathlib.Path) -> tuple[pathlib.Path, ...]:
    paths = []
    for index in range(10):
        paths.append(
            _write_trajectory(
                tmp_path / f"pass-{index}.jsonl",
                category=f"cat{index}",
                passed=True,
                tools=("read_file", "write_file", "run_test", "submit"),
            )
        )
    paths.append(
        _write_trajectory(
            tmp_path / "fail-0.jsonl",
            category="cat0",
            passed=False,
            tools=("write_file", "submit"),
        )
    )
    return tuple(paths)


def test_rows_from_passing_episodes_only(corpus_paths) -> None:
    train, heldout = build_coding_junction_rows(corpus_paths)
    rows = (*train, *heldout)
    # 10 passing episodes x 4 decisions; the failing episode contributes 0.
    assert len(rows) == 40
    assert len({row.case_id for row in rows}) == 10


def test_subgoal_is_expert_move_and_texts_are_split(corpus_paths) -> None:
    train, heldout = build_coding_junction_rows(corpus_paths)
    row = (*train, *heldout)[0]
    assert row.active_subgoal in ("investigate", "edit", "test", "submit")
    assert row.expert_action_id == f"move:{row.active_subgoal}"
    # Goal-stripped observation must not leak the task description or
    # category; the revealed text must carry both.
    assert "task:" not in row.observation_text
    assert "category=" not in row.observation_text
    assert "task:" in row.subgoal_revealed_text
    assert "category=" in row.subgoal_revealed_text
    # Suffix is appended downstream by capture, never pre-baked.
    assert not row.observation_text.endswith("Next move:")


def test_split_is_case_disjoint_and_deterministic(corpus_paths) -> None:
    train_a, heldout_a = build_coding_junction_rows(corpus_paths)
    train_b, heldout_b = build_coding_junction_rows(corpus_paths)
    assert train_a == train_b and heldout_a == heldout_b
    train_cases = {row.case_id for row in train_a}
    heldout_cases = {row.case_id for row in heldout_a}
    assert not train_cases & heldout_cases


def test_manifest_counts_phase_switches(corpus_paths) -> None:
    train, heldout = build_coding_junction_rows(corpus_paths)
    manifest = rows_manifest(train, heldout)
    # investigate→edit→test→submit = 3 switches per passing episode.
    assert manifest["phase_switches"] == 30
    assert manifest["train_rows"] + manifest["heldout_rows"] == 40


def test_bad_heldout_fraction_rejected(corpus_paths) -> None:
    with pytest.raises(ValueError):
        build_coding_junction_rows(corpus_paths, heldout_fraction=0.0)
