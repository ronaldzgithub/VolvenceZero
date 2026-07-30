from __future__ import annotations

import json

import pytest

from volvence_zero.agent.gate78_shared_trace import (
    GATE78_PARTITION_COUNTS,
    GATE78_TRACE_SEEDS,
    build_gate78_episode_plans,
    export_gate78_shared_trace_bundle,
    load_gate78_partition,
    verify_gate78_shared_trace_bundle,
)


def test_gate78_plans_freeze_multisession_and_partition_contract() -> None:
    plans = build_gate78_episode_plans(701)

    assert len(plans) == sum(count for _partition, count in GATE78_PARTITION_COUNTS)
    assert {
        partition: sum(plan.partition == partition for plan in plans)
        for partition, _count in GATE78_PARTITION_COUNTS
    } == dict(GATE78_PARTITION_COUNTS)
    assert {length for plan in plans for length in plan.segment_lengths} == {
        2,
        3,
        4,
        5,
    }
    assert len({family for plan in plans for family in plan.action_family_ids}) >= 4
    assert all(plan.next_session_boundary for plan in plans)
    assert all(len(plan.session_one_turns) == len(plan.session_two_turns) == 2 for plan in plans)
    assert all(
        isinstance(value, float)
        for plan in plans
        for value in plan.user_prior
    )


def test_gate78_bundle_admits_and_loads_only_requested_partition(tmp_path) -> None:
    export_gate78_shared_trace_bundle(output_dir=tmp_path)
    verification = verify_gate78_shared_trace_bundle(tmp_path)

    assert verification["passed"] is True
    assert verification["consumer_admission"] is True
    assert verification["locked_consumption_count"] == 0
    locked = load_gate78_partition(
        tmp_path,
        seed=709,
        partition="trace-locked-confirmation",
    )
    assert len(locked) == dict(GATE78_PARTITION_COUNTS)[
        "trace-locked-confirmation"
    ]
    assert all(row.partition == "trace-locked-confirmation" for row in locked)


def test_gate78_bundle_rejects_lineage_mutation(tmp_path) -> None:
    export_gate78_shared_trace_bundle(output_dir=tmp_path)
    episode_path = tmp_path / f"seed_{GATE78_TRACE_SEEDS[0]}" / "episodes.jsonl"
    rows = episode_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["difficulty"] = 0.99
    rows[0] = json.dumps(payload, sort_keys=True)
    episode_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    verification = verify_gate78_shared_trace_bundle(tmp_path)
    assert verification["passed"] is False
    with pytest.raises(RuntimeError, match="consumer admission"):
        load_gate78_partition(
            tmp_path,
            seed=701,
            partition="trace-train",
        )


def test_gate78_rejects_unregistered_seed_and_partition(tmp_path) -> None:
    with pytest.raises(ValueError, match="not preregistered"):
        build_gate78_episode_plans(700)

    export_gate78_shared_trace_bundle(output_dir=tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_gate78_partition(
            tmp_path,
            seed=701,
            partition="trace-future",
        )
