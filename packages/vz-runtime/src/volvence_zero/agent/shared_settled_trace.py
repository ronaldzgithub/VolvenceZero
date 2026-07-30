"""Immutable settled trace factory shared by Gate 4, Gate 5 and Gate 6.

This is an out-of-turn exporter over public runtime results.  It does not add
a runtime slot and it never reconstructs owner state.  One append unit is one
fully settled two-turn micro-session, which makes resume exact without
restoring private runner counters.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.environment import (
    EnvironmentActionSchema,
    EnvironmentActorRef,
    EnvironmentEventKind,
    EnvironmentFrame,
    EnvironmentMeasurement,
    EnvironmentOutcome,
    build_user_input_environment_event,
)
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    LocalSubstrateRuntimeMode,
    OpenWeightResidualRuntime,
    SubstrateFallbackMode,
    build_transformers_runtime_with_fallback,
)


SHARED_SETTLED_TRACE_SCHEMA_VERSION = (
    "gate456-shared-settled-trace.v1"
)
SHARED_SETTLED_TRACE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SHARED_SETTLED_TRACE_SEEDS = (401, 409, 419)
SHARED_SETTLED_TRACE_PARTITION_COUNTS = (
    ("trace-train", 300),
    ("trace-heldout-context", 150),
    ("trace-locked-confirmation", 60),
)
SHARED_SETTLED_TRACE_COUNT_PER_SEED = 510
SHARED_SETTLED_TRACE_REQUIRED_FILES = (
    "manifest.yaml",
    "transitions.jsonl",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)


@dataclass(frozen=True)
class SharedTraceContextSpec:
    partition: str
    context_id: str
    user_id: str
    domain: str
    old_knowledge_key: str
    new_knowledge_key: str
    transition_count: int


@dataclass(frozen=True)
class SharedTraceTransitionPlan:
    transition_id: str
    seed: int
    global_index: int
    partition: str
    context_id: str
    user_id: str
    domain: str
    episode_phase: str
    knowledge_key: str
    turn_one_text: str
    turn_two_text: str
    task_progress: float
    action_payoff: float
    discrete_milestone: bool


SHARED_SETTLED_TRACE_CONTEXTS = (
    SharedTraceContextSpec(
        "trace-train",
        "train-orchard",
        "user-train-01",
        "orchard planning",
        "orchard-irrigation-baseline",
        "orchard-irrigation-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-library",
        "user-train-02",
        "library circulation",
        "library-catalog-baseline",
        "library-catalog-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-workshop",
        "user-train-03",
        "workshop scheduling",
        "workshop-capacity-baseline",
        "workshop-capacity-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-clinic",
        "user-train-04",
        "clinic handoff",
        "clinic-handoff-baseline",
        "clinic-handoff-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-harbor",
        "user-train-05",
        "harbor routing",
        "harbor-routing-baseline",
        "harbor-routing-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-greenhouse",
        "user-train-06",
        "greenhouse control",
        "greenhouse-control-baseline",
        "greenhouse-control-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-studio",
        "user-train-07",
        "studio production",
        "studio-production-baseline",
        "studio-production-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-observatory",
        "user-train-08",
        "observatory planning",
        "observatory-plan-baseline",
        "observatory-plan-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-kitchen",
        "user-train-09",
        "kitchen service",
        "kitchen-service-baseline",
        "kitchen-service-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-train",
        "train-depot",
        "user-train-10",
        "depot dispatch",
        "depot-dispatch-baseline",
        "depot-dispatch-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-heldout-context",
        "heldout-apiary",
        "user-heldout-01",
        "apiary care",
        "apiary-care-baseline",
        "apiary-care-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-heldout-context",
        "heldout-brewery",
        "user-heldout-02",
        "brewery control",
        "brewery-control-baseline",
        "brewery-control-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-heldout-context",
        "heldout-museum",
        "user-heldout-03",
        "museum operations",
        "museum-operations-baseline",
        "museum-operations-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-heldout-context",
        "heldout-theater",
        "user-heldout-04",
        "theater blocking",
        "theater-blocking-baseline",
        "theater-blocking-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-heldout-context",
        "heldout-geology",
        "user-heldout-05",
        "geology survey",
        "geology-survey-baseline",
        "geology-survey-revision",
        30,
    ),
    SharedTraceContextSpec(
        "trace-locked-confirmation",
        "locked-luthiery",
        "user-locked-01",
        "luthiery setup",
        "luthiery-setup-baseline",
        "luthiery-setup-revision",
        20,
    ),
    SharedTraceContextSpec(
        "trace-locked-confirmation",
        "locked-bathymetry",
        "user-locked-02",
        "bathymetry survey",
        "bathymetry-survey-baseline",
        "bathymetry-survey-revision",
        20,
    ),
    SharedTraceContextSpec(
        "trace-locked-confirmation",
        "locked-mycology",
        "user-locked-03",
        "mycology culture",
        "mycology-culture-baseline",
        "mycology-culture-revision",
        20,
    ),
)

_EPISODE_PHASES = (
    "old-recall",
    "new-introduce",
    "new-revision",
    "old-retention",
)


def build_shared_trace_plans(
    seed: int,
) -> tuple[SharedTraceTransitionPlan, ...]:
    if seed not in SHARED_SETTLED_TRACE_SEEDS:
        raise ValueError(
            f"Shared trace seed {seed} is not preregistered"
        )
    plans: list[SharedTraceTransitionPlan] = []
    for context in SHARED_SETTLED_TRACE_CONTEXTS:
        for context_index in range(context.transition_count):
            global_index = len(plans)
            phase = _EPISODE_PHASES[context_index % len(_EPISODE_PHASES)]
            knowledge_key = (
                context.old_knowledge_key
                if phase in {"old-recall", "old-retention"}
                else context.new_knowledge_key
            )
            transition_id = (
                f"shared-trace-s{seed:03d}-t{global_index:04d}"
            )
            turn_one = (
                f"Context {context.context_id} for {context.user_id}. "
                f"In {context.domain}, review the observable record "
                f"{knowledge_key} for episode {context_index:02d} and "
                "state one bounded next action."
            )
            turn_two = (
                f"Observable follow-up for {knowledge_key}: the bounded "
                f"action completed under phase {phase}. Integrate this "
                "result while preserving the earlier context."
            )
            task_progress = {
                "old-recall": 0.4,
                "new-introduce": 0.2,
                "new-revision": -0.2,
                "old-retention": 0.5,
            }[phase]
            action_payoff = {
                "old-recall": 0.3,
                "new-introduce": 0.1,
                "new-revision": -0.1,
                "old-retention": 0.4,
            }[phase]
            plans.append(
                SharedTraceTransitionPlan(
                    transition_id=transition_id,
                    seed=seed,
                    global_index=global_index,
                    partition=context.partition,
                    context_id=context.context_id,
                    user_id=context.user_id,
                    domain=context.domain,
                    episode_phase=phase,
                    knowledge_key=knowledge_key,
                    turn_one_text=turn_one,
                    turn_two_text=turn_two,
                    task_progress=task_progress,
                    action_payoff=action_payoff,
                    discrete_milestone=(context_index % 5 == 4),
                )
            )
    if len(plans) != SHARED_SETTLED_TRACE_COUNT_PER_SEED:
        raise RuntimeError(
            "Shared trace registry count drifted: "
            f"expected={SHARED_SETTLED_TRACE_COUNT_PER_SEED}, "
            f"actual={len(plans)}"
        )
    actual_counts = tuple(
        (
            partition,
            sum(plan.partition == partition for plan in plans),
        )
        for partition, _ in SHARED_SETTLED_TRACE_PARTITION_COUNTS
    )
    if actual_counts != SHARED_SETTLED_TRACE_PARTITION_COUNTS:
        raise RuntimeError(
            "Shared trace partition counts drifted: "
            f"{actual_counts!r}"
        )
    return tuple(plans)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"Shared trace cannot serialize {type(value).__name__}"
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _actor_frame(user_id: str) -> EnvironmentFrame:
    return EnvironmentFrame(
        actor=EnvironmentActorRef(
            actor_id=user_id,
            actor_kind="interlocutor",
        ),
        active_speaker_id=user_id,
        addressee_ids=("self",),
        subject_ids=(user_id,),
        audience_ids=("self",),
    )


def _active_config() -> FinalRolloutConfig:
    return FinalRolloutConfig(
        substrate=WiringLevel.ACTIVE,
        memory=WiringLevel.ACTIVE,
        dual_track=WiringLevel.ACTIVE,
        evaluation=WiringLevel.ACTIVE,
        regime=WiringLevel.ACTIVE,
        credit=WiringLevel.ACTIVE,
        reflection=WiringLevel.ACTIVE,
        temporal=WiringLevel.ACTIVE,
    )


def _snapshot_value(result, slot_name: str) -> object:
    snapshot = result.active_snapshots.get(slot_name)
    if snapshot is None:
        raise RuntimeError(
            f"Shared trace turn lacks active snapshot {slot_name!r}"
        )
    return snapshot.value


async def run_shared_trace_transition(
    *,
    plan: SharedTraceTransitionPlan,
    runtime: OpenWeightResidualRuntime,
    schema_version: str = SHARED_SETTLED_TRACE_SCHEMA_VERSION,
    runner: AgentSessionRunner | None = None,
    config: FinalRolloutConfig | None = None,
) -> dict[str, Any]:
    """Run one formal two-turn transition and return one settled record.

    The default path retains the v1 one-transition micro-session contract.
    A versioned longitudinal source may inject a persistent ``runner`` and
    its own schema version so several settled transitions share one real
    session before an explicit owner persistence/restart boundary.
    """

    active_runner = runner or AgentSessionRunner(
        session_id=plan.transition_id,
        config=config or _active_config(),
        default_residual_runtime=runtime,
        allow_live_substrate_mutation=False,
    )
    frame = _actor_frame(plan.user_id)
    base_timestamp = plan.seed * 1_000_000 + plan.global_index * 10
    first_event = build_user_input_environment_event(
        event_id=f"{plan.transition_id}:event:prediction",
        user_input=plan.turn_one_text,
        scene_id=plan.context_id,
        timestamp_ms=base_timestamp + 1,
        frame=frame,
        provenance=schema_version,
    )
    first_started = time.perf_counter()
    first = await active_runner.run_turn(
        plan.turn_one_text,
        environment_event=first_event,
    )
    first_latency_ms = (time.perf_counter() - first_started) * 1000.0
    prediction = first.next_prediction
    if prediction is None or not prediction.prediction_id:
        raise RuntimeError(
            f"{plan.transition_id} did not publish owner prediction id"
        )
    outcome_id = f"{plan.transition_id}:outcome"
    environment_outcome = EnvironmentOutcome(
        outcome_id=outcome_id,
        event_id=first_event.event_id,
        outcome_kind=EnvironmentEventKind.TOOL_RESULT,
        action_id=f"{plan.transition_id}:action",
        status="completed",
        summary=(
            f"Typed observable result for {plan.knowledge_key}"
        ),
        detail=plan.turn_two_text,
        confidence=1.0,
        prediction_id=prediction.prediction_id,
        evidence=(f"trace://{plan.transition_id}",),
        latency_ms=10 + (plan.global_index % 17),
        monetary_cost=0.0,
        reversibility="reversible",
        environment_state_delta_kind=plan.episode_phase,
        measurement=EnvironmentMeasurement(
            task_progress=plan.task_progress,
            action_payoff=plan.action_payoff,
            terminal=True,
            discrete_milestone=plan.discrete_milestone,
        ),
        action_schema=EnvironmentActionSchema(
            schema_id=f"schema:{plan.domain.replace(' ', '-')}",
            applicability_conditions=(
                f"context={plan.context_id}",
                f"phase={plan.episode_phase}",
            ),
            action_steps=(
                "review-observable-record",
                "apply-bounded-action",
                "observe-result",
            ),
            description=(
                f"Reviewed bounded action schema for {plan.domain}."
            ),
        ),
        situation_summary=(
            f"{plan.domain} context before observable outcome."
        ),
    )
    active_runner.submit_environment_outcome(environment_outcome)
    second_event = build_user_input_environment_event(
        event_id=f"{plan.transition_id}:event:settlement",
        user_input=plan.turn_two_text,
        scene_id=plan.context_id,
        timestamp_ms=base_timestamp + 2,
        frame=frame,
        provenance=schema_version,
    )
    second_started = time.perf_counter()
    second = await active_runner.run_turn(
        plan.turn_two_text,
        environment_event=second_event,
    )
    second_latency_ms = (
        time.perf_counter() - second_started
    ) * 1000.0
    if (
        second.evaluated_prediction is None
        or second.actual_outcome is None
        or second.prediction_error is None
    ):
        raise RuntimeError(
            f"{plan.transition_id} did not settle prediction outcome PE"
        )
    action_context = second.actual_outcome.action_context
    lineage_matches = (
        second.evaluated_prediction.prediction_id
        == prediction.prediction_id
        == action_context.prediction_id
        == environment_outcome.prediction_id
        and action_context.environment_outcome_id == outcome_id
        and action_context.environment_event_id
        == second_event.event_id
    )
    if not lineage_matches:
        raise RuntimeError(
            f"{plan.transition_id} owner lineage mismatch"
        )
    slow_started = time.perf_counter()
    active_runner.begin_new_context(
        reason=f"shared-trace-settle:{plan.transition_id}"
    )
    slow_results = await active_runner.drain_session_post_slow_loop()
    slow_latency_ms = (time.perf_counter() - slow_started) * 1000.0
    substrate_mutation_applied = bool(
        second.online_fast_substrate_result is not None
        and second.online_fast_substrate_result.applied
    )
    if substrate_mutation_applied:
        raise RuntimeError(
            f"{plan.transition_id} mutated frozen substrate"
        )
    record = {
        "schema_version": schema_version,
        "transition_id": plan.transition_id,
        "seed": plan.seed,
        "global_index": plan.global_index,
        "partition": plan.partition,
        "context_id": plan.context_id,
        "user_id": plan.user_id,
        "domain": plan.domain,
        "episode_phase": plan.episode_phase,
        "knowledge_key": plan.knowledge_key,
        "input": {
            "prediction_turn": plan.turn_one_text,
            "settlement_turn": plan.turn_two_text,
        },
        "lineage": {
            "session_id": active_runner.session_id,
            "prediction_id": prediction.prediction_id,
            "prediction_ref": (
                f"{plan.transition_id}::{prediction.prediction_id}"
            ),
            "environment_event_id": second_event.event_id,
            "environment_source_event_id": first_event.event_id,
            "environment_outcome_id": outcome_id,
            "observed_at": second_event.timestamp_ms,
        },
        "prediction": _jsonable(second.evaluated_prediction),
        "actual_outcome": _jsonable(second.actual_outcome),
        "prediction_error": _jsonable(second.prediction_error),
        "environment_outcome": _jsonable(environment_outcome),
        "credit_snapshot": _jsonable(
            _snapshot_value(second, "credit")
        ),
        "temporal_snapshot": _jsonable(
            _snapshot_value(second, "temporal_abstraction")
        ),
        "memory_snapshot": _jsonable(
            _snapshot_value(second, "memory")
        ),
        "action_selection": {
            "active_regime": second.active_regime,
            "active_abstract_action": second.active_abstract_action,
            "joint_schedule_action": second.joint_schedule_action,
            "track_z_t_codes": _jsonable(second.track_z_t_codes),
        },
        "substrate": {
            "model_id": second.substrate_model_id,
            "runtime_origin": second.substrate_runtime_origin,
            "fallback_active": second.substrate_fallback_active,
            "capture_source": second.substrate_capture_source,
            "residual_sequence_length": (
                second.substrate_residual_sequence_length
            ),
            "is_frozen": runtime.is_frozen,
            "mutation_applied": substrate_mutation_applied,
        },
        "latency": {
            "prediction_turn_ms": first_latency_ms,
            "settlement_turn_ms": second_latency_ms,
            "session_post_slow_job_ms": slow_latency_ms,
            "session_post_completed_job_count": len(slow_results),
        },
        "settled": True,
    }
    record["record_sha256"] = hashlib.sha256(
        _canonical_json_bytes(record)
    ).hexdigest()
    return record


def build_strict_local_shared_trace_runtime():
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    disable_progress_bars()
    local_snapshot = snapshot_download(
        repo_id=SHARED_SETTLED_TRACE_MODEL_ID,
        local_files_only=True,
    )
    return build_transformers_runtime_with_fallback(
        model_id=SHARED_SETTLED_TRACE_MODEL_ID,
        model_source=local_snapshot,
        device="cpu",
        activation_width=8,
        local_files_only=True,
        fallback_mode=SubstrateFallbackMode.DENY,
        runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
        allow_live_substrate_mutation=False,
    )


def _runtime_fingerprint(runtime: OpenWeightResidualRuntime) -> str:
    from huggingface_hub import snapshot_download

    local_snapshot = Path(
        snapshot_download(
            repo_id=SHARED_SETTLED_TRACE_MODEL_ID,
            local_files_only=True,
        )
    )
    descriptor = {
        "model_id": runtime.model_id,
        "runtime_origin": runtime.runtime_origin,
        "runtime_type": type(runtime).__name__,
        "is_frozen": runtime.is_frozen,
        "control_basis_rank": runtime.control_basis_rank,
        "runtime_mode": LocalSubstrateRuntimeMode.STRICT_LOCAL.value,
        "fallback_mode": SubstrateFallbackMode.DENY.value,
        "activation_width": 8,
        "local_snapshot_commit": local_snapshot.name,
    }
    return "runtime-descriptor-sha256:" + hashlib.sha256(
        _canonical_json_bytes(descriptor)
    ).hexdigest()


def shared_trace_runtime_fingerprint(
    runtime: OpenWeightResidualRuntime,
) -> str:
    """Publish the frozen trace runtime descriptor fingerprint.

    Versioned evidence factories use this public readout instead of
    importing the shared trace module's serialization internals.
    """

    return _runtime_fingerprint(runtime)


def load_shared_trace_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(
                f"Blank shared trace line at {line_number}"
            )
        payload = json.loads(line)
        record_sha = payload.pop("record_sha256", None)
        expected_sha = hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()
        payload["record_sha256"] = record_sha
        if record_sha != expected_sha:
            raise ValueError(
                "Shared trace record digest mismatch at line "
                f"{line_number}"
            )
        records.append(payload)
    return records


def validate_shared_trace_prefix(
    *,
    records: Sequence[Mapping[str, Any]],
    expected_plans: Sequence[SharedTraceTransitionPlan],
) -> None:
    if len(records) > len(expected_plans):
        raise ValueError("Shared trace has more rows than frozen registry")
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        plan = expected_plans[index]
        expected = {
            "transition_id": plan.transition_id,
            "seed": plan.seed,
            "global_index": plan.global_index,
            "partition": plan.partition,
            "context_id": plan.context_id,
            "user_id": plan.user_id,
            "episode_phase": plan.episode_phase,
            "knowledge_key": plan.knowledge_key,
        }
        mismatches = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Shared trace prefix drift at row "
                f"{index}: {mismatches!r}"
            )
        transition_id = str(record["transition_id"])
        if transition_id in seen_ids:
            raise ValueError(
                f"Duplicate shared trace transition {transition_id!r}"
            )
        seen_ids.add(transition_id)
        if not record.get("settled"):
            raise ValueError(
                f"Shared trace row {transition_id!r} is not settled"
            )


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(record) + b"\n"
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.write_text(
        "".join(
            _canonical_json_bytes(row).decode("utf-8") + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_progress(
    *,
    output_dir: Path,
    seed: int,
    completed: int,
    total: int,
    runtime_fingerprint: str,
) -> None:
    progress = output_dir / "progress.json"
    temporary = output_dir / "progress.json.tmp"
    _write_json(
        temporary,
        {
            "schema_version": SHARED_SETTLED_TRACE_SCHEMA_VERSION,
            "seed": seed,
            "completed_transition_count": completed,
            "total_transition_count": total,
            "runtime_fingerprint": runtime_fingerprint,
        },
    )
    temporary.replace(progress)


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _derive_packet(
    *,
    output_dir: Path,
    seed: int,
    records: Sequence[Mapping[str, Any]],
    runtime_fingerprint: str,
) -> tuple[Path, ...]:
    expected_plans = build_shared_trace_plans(seed)
    validate_shared_trace_prefix(
        records=records,
        expected_plans=expected_plans,
    )
    partition_counts = {
        partition: sum(
            record["partition"] == partition for record in records
        )
        for partition, _ in SHARED_SETTLED_TRACE_PARTITION_COUNTS
    }
    lineage_ids = [
        str(record["lineage"]["prediction_ref"])
        for record in records
    ]
    outcome_ids = [
        str(record["lineage"]["environment_outcome_id"])
        for record in records
    ]
    valid_lineage_count = sum(
        bool(record["lineage"]["prediction_id"])
        and bool(record["lineage"]["prediction_ref"])
        and bool(record["lineage"]["session_id"])
        and bool(record["lineage"]["environment_event_id"])
        and bool(record["lineage"]["environment_outcome_id"])
        and record["lineage"]["observed_at"] is not None
        for record in records
    )
    duplicate_count = (
        len(lineage_ids)
        - len(set(lineage_ids))
        + len(outcome_ids)
        - len(set(outcome_ids))
    )
    mismatch_count = sum(
        record["prediction"]["prediction_id"]
        != record["lineage"]["prediction_id"]
        or record["actual_outcome"]["action_context"][
            "environment_outcome_id"
        ]
        != record["lineage"]["environment_outcome_id"]
        for record in records
    )
    coverage = (
        valid_lineage_count / len(records) if records else 0.0
    )
    runtime_origins = {
        record["substrate"]["runtime_origin"] for record in records
    }
    fallback_count = sum(
        bool(record["substrate"]["fallback_active"])
        for record in records
    )
    mutation_count = sum(
        bool(record["substrate"]["mutation_applied"])
        for record in records
    )
    empty_residual_count = sum(
        int(record["substrate"]["residual_sequence_length"]) <= 0
        for record in records
    )
    complete = len(records) == len(expected_plans)
    gates = {
        "settled_count_510": complete,
        "partition_counts_exact": (
            complete
            and tuple(partition_counts.items())
            == SHARED_SETTLED_TRACE_PARTITION_COUNTS
        ),
        "lineage_coverage_100_percent": coverage == 1.0,
        "lineage_mismatch_zero": mismatch_count == 0,
        "duplicate_settlement_zero": duplicate_count == 0,
        "runtime_origin_hf_local": runtime_origins == {"hf-local"},
        "fallback_zero": fallback_count == 0,
        "empty_residual_zero": empty_residual_count == 0,
        "substrate_mutation_zero": mutation_count == 0,
    }
    status = (
        "trace-contract-supported"
        if complete and all(gates.values())
        else "in-progress"
        if not complete
        else "invalid"
    )
    trace_digest = hashlib.sha256(
        b"".join(
            _canonical_json_bytes(record) + b"\n"
            for record in records
        )
    ).hexdigest()
    manifest = {
        "schema_version": SHARED_SETTLED_TRACE_SCHEMA_VERSION,
        "suite_id": f"gate456-shared-trace-seed-{seed}",
        "seed": seed,
        "seed_schedule": list(SHARED_SETTLED_TRACE_SEEDS),
        "model_id": SHARED_SETTLED_TRACE_MODEL_ID,
        "device": "cpu",
        "runtime_mode": "strict-local",
        "fallback_mode": "deny",
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_verification": (
            "runtime descriptor; runtime does not expose weights digest"
        ),
        "partition_counts": dict(
            SHARED_SETTLED_TRACE_PARTITION_COUNTS
        ),
        "context_registry": [
            asdict(context)
            for context in SHARED_SETTLED_TRACE_CONTEXTS
        ],
        "locked_confirmation": True,
        "consumer_loading_rule": "whole-partition-only",
        "required_files": list(
            SHARED_SETTLED_TRACE_REQUIRED_FILES
        ),
        "trace_sha256": trace_digest,
        "provenance": {
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "working_tree_dirty": bool(
                _git_output("status", "--porcelain")
                not in {"", "unknown"}
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    latency_rows = [
        record["latency"] for record in records
    ]

    def mean(field: str) -> float:
        return (
            sum(float(row[field]) for row in latency_rows)
            / len(latency_rows)
            if latency_rows
            else 0.0
        )

    ablation = {
        "schema_version": SHARED_SETTLED_TRACE_SCHEMA_VERSION,
        "status": status,
        "completed_transition_count": len(records),
        "expected_transition_count": len(expected_plans),
        "partition_counts": partition_counts,
        "lineage": {
            "coverage": coverage,
            "accepted_mismatch_count": mismatch_count,
            "duplicate_settlement_count": duplicate_count,
        },
        "substrate": {
            "runtime_origins": sorted(runtime_origins),
            "fallback_count": fallback_count,
            "empty_residual_count": empty_residual_count,
            "mutation_count": mutation_count,
        },
        "latency_diagnostic": {
            "mean_prediction_turn_ms": mean("prediction_turn_ms"),
            "mean_settlement_turn_ms": mean("settlement_turn_ms"),
            "mean_session_post_slow_job_ms": mean(
                "session_post_slow_job_ms"
            ),
        },
        "gates": gates,
    }
    verdict = {
        "schema_version": SHARED_SETTLED_TRACE_SCHEMA_VERSION,
        "gate_scope": "Gate 4/5/6 shared settled trace contract",
        "status": status,
        "trace_contract_status": status,
        "gate4_causal_status": "not-evaluated",
        "gate5_causal_status": "not-evaluated",
        "gate6_causal_status": "not-evaluated",
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
    }
    rollback = {
        "schema_version": SHARED_SETTLED_TRACE_SCHEMA_VERSION,
        "runtime_owner_mutated_by_export": False,
        "substrate_mutation_count": mutation_count,
        "rollback_action": "stop generation and retain settled prefix",
        "consumer_admission": (
            "allowed" if status == "trace-contract-supported"
            else "denied"
        ),
    }
    _write_json(output_dir / "manifest.yaml", manifest)
    _write_json(output_dir / "ablation_results.json", ablation)
    _write_json(output_dir / "promotion_verdict.json", verdict)
    _write_json(output_dir / "rollback_evidence.json", rollback)
    _write_jsonl(
        output_dir / "predictions.jsonl",
        [
            {
                **record["lineage"],
                "transition_id": record["transition_id"],
                "partition": record["partition"],
                "prediction": record["prediction"],
            }
            for record in records
        ],
    )
    _write_jsonl(
        output_dir / "outcomes.jsonl",
        [
            {
                **record["lineage"],
                "transition_id": record["transition_id"],
                "partition": record["partition"],
                "actual_outcome": record["actual_outcome"],
                "environment_outcome": record["environment_outcome"],
            }
            for record in records
        ],
    )
    _write_jsonl(
        output_dir / "prediction_errors.jsonl",
        [
            {
                **record["lineage"],
                "transition_id": record["transition_id"],
                "partition": record["partition"],
                "prediction_error": record["prediction_error"],
            }
            for record in records
        ],
    )
    _write_jsonl(
        output_dir / "segments.jsonl",
        [
            {
                "transition_id": record["transition_id"],
                "partition": record["partition"],
                "context_id": record["context_id"],
                "temporal_snapshot": record["temporal_snapshot"],
            }
            for record in records
        ],
    )
    _write_jsonl(
        output_dir / "credit.jsonl",
        [
            {
                "transition_id": record["transition_id"],
                "partition": record["partition"],
                "credit_snapshot": record["credit_snapshot"],
            }
            for record in records
        ],
    )
    _write_jsonl(
        output_dir / "state_diff.jsonl",
        [
            {
                "transition_id": record["transition_id"],
                "partition": record["partition"],
                "episode_phase": record["episode_phase"],
                "knowledge_key": record["knowledge_key"],
                "memory_snapshot": record["memory_snapshot"],
            }
            for record in records
        ],
    )
    _write_jsonl(
        output_dir / "action_selection.jsonl",
        [
            {
                "transition_id": record["transition_id"],
                "partition": record["partition"],
                "action_selection": record["action_selection"],
            }
            for record in records
        ],
    )
    (output_dir / "report.md").write_text(
        "\n".join(
            (
                "# Gate 4/5/6 shared settled trace",
                "",
                f"- seed: `{seed}`",
                f"- status: `{status}`",
                (
                    "- completed transitions: "
                    f"`{len(records)}/{len(expected_plans)}`"
                ),
                f"- lineage coverage: `{coverage:.6f}`",
                f"- trace sha256: `{trace_digest}`",
                "",
            )
        ),
        encoding="utf-8",
    )
    written = tuple(
        output_dir / name
        for name in SHARED_SETTLED_TRACE_REQUIRED_FILES
    )
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(
            f"Shared trace packet missing files {missing!r}"
        )
    return written


async def generate_shared_settled_trace(
    *,
    output_dir: str | Path,
    seed: int,
    max_transitions: int | None = None,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    plans = build_shared_trace_plans(seed)
    transition_path = target / "transitions.jsonl"
    records = load_shared_trace_records(transition_path)
    validate_shared_trace_prefix(
        records=records,
        expected_plans=plans,
    )
    runtime = build_strict_local_shared_trace_runtime()
    if (
        runtime.runtime_origin != "hf-local"
        or not runtime.is_frozen
    ):
        raise RuntimeError(
            "Shared trace strict-local frozen Qwen runtime unavailable"
        )
    runtime_fingerprint = _runtime_fingerprint(runtime)
    progress_path = target / "progress.json"
    if progress_path.is_file():
        progress = json.loads(
            progress_path.read_text(encoding="utf-8")
        )
        if (
            progress["seed"] != seed
            or progress["runtime_fingerprint"]
            != runtime_fingerprint
        ):
            raise ValueError(
                "Shared trace resume provenance mismatch"
            )
    stop_count = len(plans)
    if max_transitions is not None:
        if max_transitions < 1:
            raise ValueError("max_transitions must be positive")
        stop_count = min(stop_count, max_transitions)
    for plan in plans[len(records) : stop_count]:
        record = await run_shared_trace_transition(
            plan=plan,
            runtime=runtime,
        )
        _append_record(transition_path, record)
        records.append(record)
        _write_progress(
            output_dir=target,
            seed=seed,
            completed=len(records),
            total=len(plans),
            runtime_fingerprint=runtime_fingerprint,
        )
    return _derive_packet(
        output_dir=target,
        seed=seed,
        records=records,
        runtime_fingerprint=runtime_fingerprint,
    )


def generate_shared_settled_trace_sync(
    *,
    output_dir: str | Path,
    seed: int,
    max_transitions: int | None = None,
) -> tuple[Path, ...]:
    return asyncio.run(
        generate_shared_settled_trace(
            output_dir=output_dir,
            seed=seed,
            max_transitions=max_transitions,
        )
    )


def aggregate_shared_settled_trace_campaign(
    *,
    campaign_dir: str | Path,
) -> tuple[Path, ...]:
    """Validate all preregistered seed corpora and write campaign verdict."""

    root = Path(campaign_dir)
    seed_summaries: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    trace_digests: set[str] = set()
    total_count = 0
    total_partition_counts = {
        partition: 0
        for partition, _ in SHARED_SETTLED_TRACE_PARTITION_COUNTS
    }
    for seed in SHARED_SETTLED_TRACE_SEEDS:
        seed_dir = root / f"seed_{seed}"
        manifest = json.loads(
            (seed_dir / "manifest.yaml").read_text(encoding="utf-8")
        )
        ablation = json.loads(
            (seed_dir / "ablation_results.json").read_text(
                encoding="utf-8"
            )
        )
        verdict = json.loads(
            (seed_dir / "promotion_verdict.json").read_text(
                encoding="utf-8"
            )
        )
        records = load_shared_trace_records(
            seed_dir / "transitions.jsonl"
        )
        validate_shared_trace_prefix(
            records=records,
            expected_plans=build_shared_trace_plans(seed),
        )
        actual_trace_digest = hashlib.sha256(
            b"".join(
                _canonical_json_bytes(record) + b"\n"
                for record in records
            )
        ).hexdigest()
        if actual_trace_digest != manifest["trace_sha256"]:
            raise ValueError(
                f"Shared trace seed {seed} manifest digest mismatch"
            )
        partition_counts = {
            partition: sum(
                record["partition"] == partition
                for record in records
            )
            for partition, _ in SHARED_SETTLED_TRACE_PARTITION_COUNTS
        }
        for partition, count in partition_counts.items():
            total_partition_counts[partition] += count
        total_count += len(records)
        fingerprints.add(str(manifest["runtime_fingerprint"]))
        trace_digests.add(actual_trace_digest)
        seed_summaries.append(
            {
                "seed": seed,
                "status": verdict["status"],
                "transition_count": len(records),
                "partition_counts": partition_counts,
                "lineage": ablation["lineage"],
                "substrate": ablation["substrate"],
                "runtime_fingerprint": manifest[
                    "runtime_fingerprint"
                ],
                "trace_sha256": actual_trace_digest,
            }
        )
    expected_total = (
        SHARED_SETTLED_TRACE_COUNT_PER_SEED
        * len(SHARED_SETTLED_TRACE_SEEDS)
    )
    expected_partition_totals = {
        partition: count * len(SHARED_SETTLED_TRACE_SEEDS)
        for partition, count in SHARED_SETTLED_TRACE_PARTITION_COUNTS
    }
    gates = {
        "three_seed_packets_present": (
            len(seed_summaries) == len(SHARED_SETTLED_TRACE_SEEDS)
        ),
        "settled_transition_count_1530": total_count == expected_total,
        "partition_totals_exact": (
            total_partition_counts == expected_partition_totals
        ),
        "all_seed_contracts_supported": all(
            summary["status"] == "trace-contract-supported"
            for summary in seed_summaries
        ),
        "runtime_fingerprint_shared": len(fingerprints) == 1,
        "trace_digests_distinct": (
            len(trace_digests) == len(SHARED_SETTLED_TRACE_SEEDS)
        ),
        "lineage_all_green": all(
            summary["lineage"]["coverage"] == 1.0
            and summary["lineage"]["accepted_mismatch_count"] == 0
            and summary["lineage"]["duplicate_settlement_count"] == 0
            for summary in seed_summaries
        ),
        "substrate_all_green": all(
            summary["substrate"]["runtime_origins"] == ["hf-local"]
            and summary["substrate"]["fallback_count"] == 0
            and summary["substrate"]["empty_residual_count"] == 0
            and summary["substrate"]["mutation_count"] == 0
            for summary in seed_summaries
        ),
    }
    status = (
        "trace-contract-supported"
        if all(gates.values())
        else "invalid"
    )
    aggregate_manifest = {
        "schema_version": SHARED_SETTLED_TRACE_SCHEMA_VERSION,
        "campaign_id": "gate456-shared-settled-trace-20260730",
        "seed_schedule": list(SHARED_SETTLED_TRACE_SEEDS),
        "expected_transition_count": expected_total,
        "expected_partition_totals": expected_partition_totals,
        "runtime_fingerprint": (
            next(iter(fingerprints)) if len(fingerprints) == 1 else None
        ),
        "locked_confirmation": True,
        "consumer_loading_rule": "whole-partition-only",
        "seed_summaries": seed_summaries,
    }
    aggregate_verdict = {
        "schema_version": SHARED_SETTLED_TRACE_SCHEMA_VERSION,
        "gate_scope": "Gate 4/5/6 shared trace aggregate",
        "status": status,
        "trace_contract_status": status,
        "consumer_admission": (
            "allowed" if status == "trace-contract-supported"
            else "denied"
        ),
        "gate4_causal_status": "not-evaluated",
        "gate5_causal_status": "not-evaluated",
        "gate6_causal_status": "not-evaluated",
        "gates": gates,
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
    }
    manifest_path = root / "aggregate_manifest.json"
    verdict_path = root / "aggregate_verdict.json"
    report_path = root / "aggregate_report.md"
    _write_json(manifest_path, aggregate_manifest)
    _write_json(verdict_path, aggregate_verdict)
    report_path.write_text(
        "\n".join(
            (
                "# Gate 4/5/6 shared trace aggregate",
                "",
                f"- status: `{status}`",
                f"- settled transitions: `{total_count}`",
                (
                    "- partition totals: `"
                    + json.dumps(
                        total_partition_counts,
                        sort_keys=True,
                    )
                    + "`"
                ),
                (
                    "- runtime fingerprint: `"
                    + str(aggregate_manifest["runtime_fingerprint"])
                    + "`"
                ),
                "- Gate 4/5/6 causal verdicts: `not-evaluated`",
                "",
            )
        ),
        encoding="utf-8",
    )
    return (manifest_path, verdict_path, report_path)


__all__ = [
    "SHARED_SETTLED_TRACE_CONTEXTS",
    "SHARED_SETTLED_TRACE_COUNT_PER_SEED",
    "SHARED_SETTLED_TRACE_MODEL_ID",
    "SHARED_SETTLED_TRACE_PARTITION_COUNTS",
    "SHARED_SETTLED_TRACE_REQUIRED_FILES",
    "SHARED_SETTLED_TRACE_SCHEMA_VERSION",
    "SHARED_SETTLED_TRACE_SEEDS",
    "SharedTraceContextSpec",
    "SharedTraceTransitionPlan",
    "aggregate_shared_settled_trace_campaign",
    "build_shared_trace_plans",
    "build_strict_local_shared_trace_runtime",
    "generate_shared_settled_trace",
    "generate_shared_settled_trace_sync",
    "load_shared_trace_records",
    "run_shared_trace_transition",
    "validate_shared_trace_prefix",
]
