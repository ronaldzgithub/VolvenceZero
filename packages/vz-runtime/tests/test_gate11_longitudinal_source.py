from __future__ import annotations

import json

import pytest

from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SESSION_SIZE,
    GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
    _derive_seed_packet,
    build_gate11_longitudinal_source_plans,
    validate_gate11_longitudinal_source_prefix,
)
from volvence_zero.agent.shared_settled_trace import (
    SHARED_SETTLED_TRACE_COUNT_PER_SEED,
    SHARED_SETTLED_TRACE_PARTITION_COUNTS,
)


def _synthetic_record(plan):
    prediction_id = f"{plan.transition_id}:prediction"
    outcome_id = f"{plan.transition_id}:outcome"
    return {
        "schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "transition_id": plan.transition_id,
        "seed": plan.seed,
        "global_index": plan.global_index,
        "partition": plan.partition,
        "context_id": plan.context_id,
        "user_id": plan.user_id,
        "domain": plan.domain,
        "episode_phase": plan.episode_phase,
        "knowledge_key": plan.knowledge_key,
        "lineage": {
            "session_id": plan.transition_id,
            "prediction_id": prediction_id,
            "prediction_ref": f"{plan.transition_id}::{prediction_id}",
            "environment_event_id": f"{plan.transition_id}:event:settlement",
            "environment_source_event_id": (
                f"{plan.transition_id}:event:prediction"
            ),
            "environment_outcome_id": outcome_id,
            "observed_at": plan.global_index,
        },
        "prediction": {"prediction_id": prediction_id},
        "actual_outcome": {
            "action_context": {"environment_outcome_id": outcome_id}
        },
        "prediction_error": {"magnitude": 0.1},
        "environment_outcome": {"outcome_id": outcome_id},
        "credit_snapshot": {},
        "temporal_snapshot": {},
        "memory_snapshot": {},
        "action_selection": {},
        "substrate": {
            "runtime_origin": "hf-local",
            "fallback_active": False,
            "residual_sequence_length": 2,
            "mutation_applied": False,
        },
        "latency": {
            "prediction_turn_ms": 1.0,
            "settlement_turn_ms": 1.0,
            "session_post_slow_job_ms": 1.0,
        },
        "longitudinal": {
            "capture_mode": "fresh-isolated-micro-session",
            "consumer_session_boundary_interval": (
                GATE11_LONGITUDINAL_SESSION_SIZE
            ),
            "consumer_owner_persistence_required": True,
        },
        "settled": True,
        "record_sha256": "test-only",
    }


def test_gate11_longitudinal_registry_is_fresh_and_complete() -> None:
    assert set(GATE11_LONGITUDINAL_SOURCE_SEEDS).isdisjoint(
        {401, 409, 419}
    )
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        plans = build_gate11_longitudinal_source_plans(seed)
        assert len(plans) == SHARED_SETTLED_TRACE_COUNT_PER_SEED
        assert tuple(
            (
                partition,
                sum(plan.partition == partition for plan in plans),
            )
            for partition, _ in SHARED_SETTLED_TRACE_PARTITION_COUNTS
        ) == SHARED_SETTLED_TRACE_PARTITION_COUNTS


def test_gate11_longitudinal_prefix_rejects_reordering() -> None:
    plans = build_gate11_longitudinal_source_plans(1201)
    records = [
        _synthetic_record(plans[0]),
        _synthetic_record(plans[2]),
    ]
    with pytest.raises(ValueError, match="prefix drift"):
        validate_gate11_longitudinal_source_prefix(
            records=records,
            plans=plans,
        )


def test_gate11_longitudinal_complete_packet_is_fresh_and_admissible(
    tmp_path,
) -> None:
    plans = build_gate11_longitudinal_source_plans(1201)
    records = [_synthetic_record(plan) for plan in plans]
    (tmp_path / "transitions.jsonl").write_text(
        "\n",
        encoding="utf-8",
    )
    _derive_seed_packet(
        output_dir=tmp_path,
        seed=1201,
        records=records,
        runtime_fingerprint="runtime-descriptor-sha256:test",
    )
    verdict = json.loads(
        (tmp_path / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    assert verdict["status"] == "trace-contract-supported"
    assert verdict["consumer_admission"] == "seed-ready"
