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
    episode_index: int = 0,
) -> pathlib.Path:
    # ``episode_index`` keeps sibling episodes byte-distinct: case_id is
    # the trajectory content hash, so identical logs would collapse into
    # one case.
    events: list[dict] = [
        {
            "event_index": 0,
            "event_type": "task_presented",
            "monotonic_seconds": 0.0,
            "payload": {
                "task_id": f"task-{category}-{episode_index:03d}",
                "category": category,
                "description": f"do the {category} change",
                "episode_index": episode_index,
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


def _cohort(
    tmp_path: pathlib.Path,
    *,
    prefix: str,
    tools: tuple[str, ...],
    passes: int,
    total: int,
    index_offset: int,
) -> list[pathlib.Path]:
    return [
        _write_trajectory(
            tmp_path / f"{prefix}-{index}.jsonl",
            category="cat",
            passed=index < passes,
            tools=tools,
            episode_index=index_offset + index,
        )
        for index in range(total)
    ]


@pytest.fixture()
def corpus_paths(tmp_path: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Three strategy cohorts on ONE category, so the corpus owner's
    credit table has enough support to separate moves.

    Credit-backed contrasts this produces:
    ``reads=0|edited=0|tests=none`` investigate (10/20) over edit (1/10),
    and ``reads=1|edited=1|tests=none`` test (8/10) over submit (2/10).
    The two single-action states in these routes stay unlabelled — no
    contrast, no expert target.
    """

    return tuple(
        _cohort(
            tmp_path,
            prefix="thorough",
            tools=("read_file", "write_file", "run_test", "submit"),
            passes=8,
            total=10,
            index_offset=100,
        )
        + _cohort(
            tmp_path,
            prefix="blind-edit",
            tools=("write_file", "submit"),
            passes=1,
            total=10,
            index_offset=200,
        )
        + _cohort(
            tmp_path,
            prefix="untested",
            tools=("read_file", "write_file", "submit"),
            passes=2,
            total=10,
            index_offset=300,
        )
    )


def test_rows_from_passing_episodes_only(corpus_paths) -> None:
    train, heldout = build_coding_junction_rows(corpus_paths)
    rows = (*train, *heldout)
    # 11 passing episodes; only decisions at credit-labelled states are
    # kept (thorough/untested contribute 2 each, blind-edit 1).
    assert len(rows) == 8 * 2 + 2 * 2 + 1
    assert len({row.case_id for row in rows}) == 11
    # Failing episodes never contribute rows, only credit statistics.
    assert all(row.split in ("train", "heldout") for row in rows)


def test_subgoal_is_credit_backed_expert_and_texts_are_split(corpus_paths) -> None:
    train, heldout = build_coding_junction_rows(corpus_paths)
    rows = (*train, *heldout)
    row = rows[0]
    assert row.active_subgoal in ("investigate", "edit", "test", "submit")
    assert row.expert_action_id == f"move:{row.active_subgoal}"
    # Subgoal is the credit-table expert for the state, NOT whatever this
    # (passing) episode happened to do: the blind-edit cohort's passing
    # episode sits at the opening state and is still labelled investigate.
    by_state = {row.current_location_id: row.active_subgoal for row in rows}
    assert by_state["cat|reads=0|edited=0|tests=none"] == "investigate"
    assert by_state["cat|reads=1|edited=1|tests=none"] == "test"
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
    # Routes keeping two labelled states switch subgoal once
    # (investigate → test); the single-row blind-edit case cannot switch.
    assert manifest["phase_switches"] == 10
    assert manifest["train_rows"] + manifest["heldout_rows"] == 21


def test_bad_heldout_fraction_rejected(corpus_paths) -> None:
    with pytest.raises(ValueError):
        build_coding_junction_rows(corpus_paths, heldout_fraction=0.0)
