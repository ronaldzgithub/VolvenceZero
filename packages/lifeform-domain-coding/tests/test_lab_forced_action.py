"""Directed tests for the Packet 3.5 forced-action RCT machinery."""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from lifeform_domain_coding.lab.hands import (
    ConstraintAwareScriptedHand,
    ForcedActionAssignment,
    ForcedActionHand,
    HandAction,
    HandContext,
    HandDecision,
    ScriptedHand,
    TranscriptEntry,
)
from lifeform_domain_coding.lab.junctions import (
    ACTION_EDIT,
    ACTION_INVESTIGATE,
    ACTION_SUBMIT,
    ACTION_TEST,
    FORCED_ASSIGNMENT_METADATA_KEY,
    InterventionalAssignmentRecord,
    build_interventional_action_outcome_table,
    extract_forced_assignment,
    extract_junctions,
    interventional_expert_actions,
    state_key_for,
    transcript_protocol_state,
    wilson_interval,
)
from lifeform_domain_coding.lab.tasks import ChainTask, FileEdit
from lifeform_domain_coding.lab.trajectory import TrajectoryWriter


def _context(
    *,
    transcript: tuple[TranscriptEntry, ...] = (),
    step_index: int = 0,
    preamble: str = "",
) -> HandContext:
    return HandContext(
        task_id="task-0",
        task_description="demo task",
        package_name="demo_pkg",
        step_index=step_index,
        max_steps=24,
        transcript=transcript,
        context_preamble=preamble,
    )


def _entry(tool_name: str, *, succeeded: bool = True, exit_code: int | None = 0) -> TranscriptEntry:
    result: dict[str, object] = {"content": "x"}
    if tool_name == "run_test":
        result = {"exit_code": exit_code}
    return TranscriptEntry(
        tool_name=tool_name, parameters={}, result=result, succeeded=succeeded
    )


class _StubHand:
    """Deterministic inner hand emitting a scripted sequence of actions."""

    def __init__(self, actions: list[HandAction]) -> None:
        self._actions = list(actions)
        self.seen_preambles: list[str] = []

    def hand_id(self) -> str:
        return "stub"

    async def decide(self, context: HandContext) -> HandDecision:
        self.seen_preambles.append(context.context_preamble)
        return HandDecision(action=self._actions.pop(0), metadata={"stub": True})


_TARGET_KEY = "fix_bug|reads=1|edited=0|tests=none"


def _assignment(action: str | None) -> ForcedActionAssignment:
    return ForcedActionAssignment(
        target_state_keys=(_TARGET_KEY,),
        assigned_action=action,
        assignment_id="p35-test",
    )


class TestTranscriptProtocolState:
    def test_matches_extract_junctions_semantics(self) -> None:
        calls = (
            ("read_file", True, None),
            ("write_file", True, None),
            ("run_test", True, 1),
            ("bogus_tool", False, None),
        )
        state = transcript_protocol_state(calls)
        assert state == {
            "investigate_count": 1,
            "reads_bucket": 1,
            "has_edited": True,
            "test_state": "failed",
        }

    def test_failed_test_call_marks_failed(self) -> None:
        state = transcript_protocol_state((("run_test", False, None),))
        assert state["test_state"] == "failed"

    def test_state_key_for_matches_private_builder(self) -> None:
        key = state_key_for(
            category="fix_bug", investigate_count=5, has_edited=False, test_state="none"
        )
        assert key == "fix_bug|reads=3|edited=0|tests=none"


class TestForcedActionHand:
    def test_triggers_once_at_target_state_direct_investigate(self) -> None:
        inner = _StubHand([HandAction(kind="submit")])
        hand = ForcedActionHand(
            inner=inner, category="fix_bug", assignment=_assignment(ACTION_INVESTIGATE)
        )
        transcript = (_entry("read_file"),)
        decision = asyncio.run(hand.decide(_context(transcript=transcript, step_index=1)))
        assert decision.action.tool_name == "list_dir"
        record = decision.metadata[FORCED_ASSIGNMENT_METADATA_KEY]
        assert record["state_key"] == _TARGET_KEY
        assert record["realization"] == "direct"
        assert record["compliant"] is True
        assert hand.triggered
        # Subsequent decisions delegate transparently.
        follow_up = asyncio.run(hand.decide(_context(transcript=transcript, step_index=2)))
        assert follow_up.action.kind == "submit"
        assert FORCED_ASSIGNMENT_METADATA_KEY not in follow_up.metadata

    def test_no_trigger_off_target(self) -> None:
        inner = _StubHand([HandAction(kind="tool", tool_name="read_file", parameters={"path": "a"})])
        hand = ForcedActionHand(
            inner=inner, category="fix_bug", assignment=_assignment(ACTION_SUBMIT)
        )
        decision = asyncio.run(hand.decide(_context(step_index=0)))
        assert decision.action.tool_name == "read_file"
        assert not hand.triggered

    def test_forced_submit_is_direct(self) -> None:
        inner = _StubHand([])
        hand = ForcedActionHand(
            inner=inner, category="fix_bug", assignment=_assignment(ACTION_SUBMIT)
        )
        decision = asyncio.run(
            hand.decide(_context(transcript=(_entry("read_file"),), step_index=1))
        )
        assert decision.action.kind == "submit"
        record = decision.metadata[FORCED_ASSIGNMENT_METADATA_KEY]
        assert record["assigned_action"] == ACTION_SUBMIT
        assert record["decide_attempts"] == 0

    def test_constraint_edit_compliant_on_retry(self) -> None:
        inner = _StubHand(
            [
                HandAction(kind="tool", tool_name="run_test", parameters={"test_path": "t"}),
                HandAction(
                    kind="tool",
                    tool_name="write_file",
                    parameters={"path": "src/a.py", "content": "x", "mode": "append"},
                ),
            ]
        )
        hand = ForcedActionHand(
            inner=inner, category="fix_bug", assignment=_assignment(ACTION_EDIT)
        )
        decision = asyncio.run(
            hand.decide(_context(transcript=(_entry("read_file"),), step_index=1))
        )
        record = decision.metadata[FORCED_ASSIGNMENT_METADATA_KEY]
        assert record["compliant"] is True
        assert record["decide_attempts"] == 2
        assert decision.action.tool_name == "write_file"
        assert all("[FORCED-ACTION DIRECTIVE]" in p for p in inner.seen_preambles)

    def test_constraint_noncompliant_keeps_natural_action_itt(self) -> None:
        inner = _StubHand(
            [
                HandAction(kind="tool", tool_name="read_file", parameters={"path": "a"}),
                HandAction(kind="tool", tool_name="read_file", parameters={"path": "b"}),
            ]
        )
        hand = ForcedActionHand(
            inner=inner, category="fix_bug", assignment=_assignment(ACTION_TEST)
        )
        decision = asyncio.run(
            hand.decide(_context(transcript=(_entry("read_file"),), step_index=1))
        )
        record = decision.metadata[FORCED_ASSIGNMENT_METADATA_KEY]
        assert record["compliant"] is False
        assert record["assigned_action"] == ACTION_TEST
        assert record["realized_action"] == ACTION_INVESTIGATE
        assert decision.action.tool_name == "read_file"

    def test_control_annotates_without_changing_action(self) -> None:
        inner = _StubHand(
            [HandAction(kind="tool", tool_name="run_test", parameters={"test_path": "t"})]
        )
        hand = ForcedActionHand(inner=inner, category="fix_bug", assignment=_assignment(None))
        decision = asyncio.run(
            hand.decide(_context(transcript=(_entry("read_file"),), step_index=1))
        )
        assert decision.action.tool_name == "run_test"
        record = decision.metadata[FORCED_ASSIGNMENT_METADATA_KEY]
        assert record["arm"] == "control"
        assert record["assigned_action"] is None
        assert record["realized_action"] == ACTION_TEST

    def test_rejects_unknown_action(self) -> None:
        with pytest.raises(ValueError):
            _assignment("refactor")


class TestConstraintAwareScriptedHand:
    def _hand(self) -> ConstraintAwareScriptedHand:
        task = ChainTask(
            task_id="task-0",
            category="fix_bug",
            description="demo",
            target_files=("src/config.py",),
            acceptance_test_source="def test_demo():\n    assert True\n",
            reference_edits=(FileEdit(path="src/config.py", old="a", new="b"),),
            invariant_sabotage_edits=(),
            acceptance_sabotage_edits=(),
        )
        return ConstraintAwareScriptedHand(
            tasks_by_id={"task-0": task},
            episode_index_by_task_id={"task-0": 0},
            hand_seed=1,
            invariant_sabotage_rate=0.0,
            acceptance_sabotage_rate=0.0,
        )

    def test_obeys_edit_directive(self) -> None:
        hand = self._hand()
        decision = asyncio.run(
            hand.decide(
                _context(preamble="[FORCED-ACTION DIRECTIVE] ... class edit (write_file) ...")
            )
        )
        assert decision.action.tool_name == "write_file"
        assert decision.action.parameters["mode"] == "append"

    def test_obeys_test_directive(self) -> None:
        hand = self._hand()
        decision = asyncio.run(
            hand.decide(
                _context(preamble="[FORCED-ACTION DIRECTIVE] ... class test (run_test) ...")
            )
        )
        assert decision.action.tool_name == "run_test"

    def test_without_directive_behaves_like_scripted(self) -> None:
        hand = self._hand()
        decision = asyncio.run(hand.decide(_context()))
        base = ScriptedHand(
            tasks_by_id=hand._tasks_by_id,
            episode_index_by_task_id={"task-0": 0},
            hand_seed=1,
            invariant_sabotage_rate=0.0,
            acceptance_sabotage_rate=0.0,
        )
        expected = asyncio.run(base.decide(_context()))
        assert decision.action == expected.action


def _write_marked_trajectory(
    path: pathlib.Path,
    *,
    record: dict[str, object] | None,
    passed: bool = True,
    duplicate: bool = False,
) -> None:
    writer = TrajectoryWriter(path)
    writer.append(
        "task_presented",
        {
            "task_id": "task-0",
            "category": "fix_bug",
            "description": "demo",
            "episode_index": 0,
            "hand_id": "stub",
        },
    )
    metadata: dict[str, object] = {}
    if record is not None:
        metadata = {FORCED_ASSIGNMENT_METADATA_KEY: record}
    writer.append(
        "hand_decision",
        {
            "step_index": 0,
            "kind": "tool",
            "tool_name": "list_dir",
            "parameters": {"path": "."},
            "note": "",
            "metadata": metadata,
        },
    )
    if duplicate:
        writer.append(
            "hand_decision",
            {
                "step_index": 1,
                "kind": "submit",
                "tool_name": "",
                "parameters": {},
                "note": "",
                "metadata": metadata,
            },
        )
    writer.append(
        "oracle_outcome",
        {
            "task_id": "task-0",
            "passed": passed,
            "acceptance_passed": passed,
            "regression_passed": passed,
            "failed_test_ids": [],
            "error_test_ids": [],
            "invariant_violations": [],
            "failure_details": [],
            "submitted": True,
        },
    )
    writer.close()


def _marked_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "coding-lab-forced-action-assignment.v1",
        "assignment_id": "p35-test",
        "state_key": _TARGET_KEY,
        "step_index": 0,
        "arm": "intervention",
        "assigned_action": ACTION_INVESTIGATE,
        "realized_action": ACTION_INVESTIGATE,
        "compliant": True,
        "decide_attempts": 0,
        "realization": "direct",
    }
    record.update(overrides)
    return record


class TestAssignmentExtraction:
    def test_round_trip(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "trajectories" / "episode-000.jsonl"
        path.parent.mkdir(parents=True)
        _write_marked_trajectory(path, record=_marked_record())
        record = extract_forced_assignment(path)
        assert record is not None
        assert record.state_key == _TARGET_KEY
        assert record.arm == "intervention"
        assert record.assigned_action == ACTION_INVESTIGATE
        assert record.episode_passed is True

    def test_unmarked_trajectory_returns_none(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "trajectories" / "episode-001.jsonl"
        path.parent.mkdir(parents=True)
        _write_marked_trajectory(path, record=None)
        assert extract_forced_assignment(path) is None

    def test_duplicate_assignment_fails_loudly(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "trajectories" / "episode-002.jsonl"
        path.parent.mkdir(parents=True)
        _write_marked_trajectory(path, record=_marked_record(), duplicate=True)
        with pytest.raises(ValueError, match="one-shot"):
            extract_forced_assignment(path)

    def test_extraction_coexists_with_junction_extraction(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "trajectories" / "episode-003.jsonl"
        path.parent.mkdir(parents=True)
        _write_marked_trajectory(path, record=_marked_record())
        junctions = extract_junctions(path)
        assert junctions  # the marked decision still yields junction records


def _assignment_record(
    *,
    state_key: str = _TARGET_KEY,
    assigned: str | None = ACTION_TEST,
    passed: bool = True,
    compliant: bool = True,
    arm: str = "intervention",
    sha: str = "0" * 64,
) -> InterventionalAssignmentRecord:
    return InterventionalAssignmentRecord(
        trajectory_sha256=sha,
        provenance="chain-00/episode-000#s0",
        category="fix_bug",
        state_key=state_key,
        arm=arm,
        assigned_action=assigned,
        realized_action=assigned or ACTION_INVESTIGATE,
        compliant=compliant,
        decide_attempts=1,
        step_index=0,
        episode_passed=passed,
    )


class TestInterventionalTable:
    def test_itt_grouping_and_expert_map(self) -> None:
        records = tuple(
            [_assignment_record(assigned=ACTION_TEST, passed=True) for _ in range(6)]
            + [_assignment_record(assigned=ACTION_SUBMIT, passed=False) for _ in range(5)]
            + [_assignment_record(assigned=ACTION_SUBMIT, passed=True)]
            + [_assignment_record(assigned=None, arm="control", passed=True)]
        )
        table = build_interventional_action_outcome_table(records)
        stats = {s.assigned_action: s for s in table[_TARGET_KEY]}
        assert stats[ACTION_TEST].trials == 6
        assert stats[ACTION_TEST].passes == 6
        assert stats[ACTION_SUBMIT].trials == 6
        assert stats[ACTION_SUBMIT].passes == 1
        experts = interventional_expert_actions(records)
        assert experts == {_TARGET_KEY: ACTION_TEST}

    def test_noncompliance_counts_dilute_not_drop(self) -> None:
        records = tuple(
            [_assignment_record(assigned=ACTION_TEST, passed=True, compliant=False)] * 5
        )
        table = build_interventional_action_outcome_table(records)
        stat = table[_TARGET_KEY][0]
        assert stat.trials == 5
        assert stat.compliant_trials == 0

    def test_wilson_interval_bounds(self) -> None:
        low, high = wilson_interval(9, 10)
        assert 0.0 <= low < 0.9 < high <= 1.0
        with pytest.raises(ValueError):
            wilson_interval(1, 0)
