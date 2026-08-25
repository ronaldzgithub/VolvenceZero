"""Fresh real-substrate source for the fourth gate campaign.

The source keeps the frozen Gate 4/5/6 scenario registry but uses fresh seeds
with isolated capture micro-sessions.  Gate 11 and Gate 5 consumers apply the
real owner persistence boundary every ten transitions.  Separating capture
from state replay prevents accumulated retrieval context from contaminating
the matched frozen-substrate variable.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

from volvence_zero.agent.shared_settled_trace import (
    SHARED_SETTLED_TRACE_CONTEXTS,
    SHARED_SETTLED_TRACE_COUNT_PER_SEED,
    SHARED_SETTLED_TRACE_MODEL_ID,
    SHARED_SETTLED_TRACE_PARTITION_COUNTS,
    SharedTraceTransitionPlan,
    build_strict_local_shared_trace_runtime,
    run_shared_trace_transition,
    shared_trace_runtime_fingerprint,
)
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel


GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION = (
    "gate11-longitudinal-settled-trace.v2"
)
GATE11_LONGITUDINAL_SOURCE_SEEDS = (1201, 1213, 1223)
GATE11_LONGITUDINAL_SESSION_SIZE = 10
GATE11_LONGITUDINAL_SOURCE_REQUIRED_FILES = (
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
_EPISODE_PHASES = (
    "old-recall",
    "new-introduce",
    "new-revision",
    "old-retention",
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(row) + b"\n")
        handle.flush()


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


def _capture_config() -> FinalRolloutConfig:
    """Keep State-KV carriers out of the matched source capture."""

    return FinalRolloutConfig(
        substrate=WiringLevel.ACTIVE,
        memory=WiringLevel.ACTIVE,
        dual_track=WiringLevel.ACTIVE,
        evaluation=WiringLevel.ACTIVE,
        regime=WiringLevel.ACTIVE,
        credit=WiringLevel.ACTIVE,
        reflection=WiringLevel.ACTIVE,
        temporal=WiringLevel.ACTIVE,
        personal_conditioning=WiringLevel.DISABLED,
        relationship_conditioning=WiringLevel.DISABLED,
        conditioning_router=WiringLevel.DISABLED,
    )


def build_gate11_longitudinal_source_plans(
    seed: int,
) -> tuple[SharedTraceTransitionPlan, ...]:
    """Build the frozen fresh-seed registry for one longitudinal source."""

    if seed not in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        raise ValueError(
            f"Gate 11 longitudinal seed {seed} is not preregistered"
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
                f"gate11-long-s{seed:04d}-t{global_index:04d}"
            )
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
                    turn_one_text=(
                        f"Context {context.context_id} for {context.user_id}. "
                        f"In {context.domain}, review the observable record "
                        f"{knowledge_key} for episode {context_index:02d} and "
                        "state one bounded next action."
                    ),
                    turn_two_text=(
                        f"Observable follow-up for {knowledge_key}: the bounded "
                        f"action completed under phase {phase}. Integrate this "
                        "result while preserving the earlier context."
                    ),
                    task_progress={
                        "old-recall": 0.4,
                        "new-introduce": 0.2,
                        "new-revision": -0.2,
                        "old-retention": 0.5,
                    }[phase],
                    action_payoff={
                        "old-recall": 0.3,
                        "new-introduce": 0.1,
                        "new-revision": -0.1,
                        "old-retention": 0.4,
                    }[phase],
                    discrete_milestone=(context_index % 5 == 4),
                )
            )
    if len(plans) != SHARED_SETTLED_TRACE_COUNT_PER_SEED:
        raise RuntimeError(
            "Gate 11 longitudinal registry count drifted: "
            f"expected={SHARED_SETTLED_TRACE_COUNT_PER_SEED}, "
            f"actual={len(plans)}"
        )
    return tuple(plans)


def load_gate11_longitudinal_source_records(
    path: str | Path,
) -> list[dict[str, Any]]:
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
                f"Blank Gate 11 longitudinal source line at {line_number}"
            )
        payload = json.loads(line)
        actual = payload.pop("record_sha256", None)
        expected = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        payload["record_sha256"] = actual
        if actual != expected:
            raise ValueError(
                "Gate 11 longitudinal source digest mismatch at line "
                f"{line_number}"
            )
        records.append(payload)
    return records


def validate_gate11_longitudinal_source_prefix(
    *,
    records: Sequence[Mapping[str, Any]],
    plans: Sequence[SharedTraceTransitionPlan],
) -> None:
    if len(records) > len(plans):
        raise ValueError("Longitudinal source has more rows than registry")
    seen: set[str] = set()
    for index, record in enumerate(records):
        plan = plans[index]
        expected = {
            "schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
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
                f"Longitudinal source prefix drift at {index}: {mismatches!r}"
            )
        transition_id = str(record["transition_id"])
        if transition_id in seen:
            raise ValueError(
                f"Duplicate longitudinal transition {transition_id!r}"
            )
        seen.add(transition_id)
        if not record.get("settled"):
            raise ValueError(f"{transition_id} is not settled")


def _derive_seed_packet(
    *,
    output_dir: Path,
    seed: int,
    records: Sequence[Mapping[str, Any]],
    runtime_fingerprint: str,
) -> tuple[Path, ...]:
    plans = build_gate11_longitudinal_source_plans(seed)
    validate_gate11_longitudinal_source_prefix(
        records=records,
        plans=plans,
    )
    partition_counts = {
        partition: sum(record["partition"] == partition for record in records)
        for partition, _ in SHARED_SETTLED_TRACE_PARTITION_COUNTS
    }
    complete = len(records) == len(plans)
    lineage_refs = [
        str(record["lineage"]["prediction_ref"]) for record in records
    ]
    outcome_ids = [
        str(record["lineage"]["environment_outcome_id"]) for record in records
    ]
    lineage_complete = all(
        record["lineage"]["prediction_id"]
        and record["lineage"]["prediction_ref"]
        and record["lineage"]["session_id"]
        and record["lineage"]["environment_event_id"]
        and record["lineage"]["environment_outcome_id"]
        for record in records
    )
    duplicate_count = (
        len(lineage_refs)
        - len(set(lineage_refs))
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
    fallback_count = sum(
        bool(record["substrate"]["fallback_active"]) for record in records
    )
    mutation_count = sum(
        bool(record["substrate"]["mutation_applied"]) for record in records
    )
    empty_residual_count = sum(
        int(record["substrate"]["residual_sequence_length"]) <= 0
        for record in records
    )
    session_ids = {
        str(record["lineage"]["session_id"]) for record in records
    }
    gates = {
        "settled_count_510": complete,
        "partition_counts_exact": (
            complete
            and tuple(partition_counts.items())
            == SHARED_SETTLED_TRACE_PARTITION_COUNTS
        ),
        "lineage_complete": lineage_complete,
        "lineage_mismatch_zero": mismatch_count == 0,
        "duplicate_settlement_zero": duplicate_count == 0,
        "runtime_origin_hf_local": {
            record["substrate"]["runtime_origin"] for record in records
        }
        == {"hf-local"},
        "fallback_zero": fallback_count == 0,
        "empty_residual_zero": empty_residual_count == 0,
        "substrate_mutation_zero": mutation_count == 0,
        "fresh_capture_micro_session_count_510": (
            complete and len(session_ids) == len(records)
        ),
        "consumer_session_boundary_declared": all(
            int(
                record["longitudinal"][
                    "consumer_session_boundary_interval"
                ]
            )
            == GATE11_LONGITUDINAL_SESSION_SIZE
            for record in records
        ),
    }
    status = (
        "trace-contract-supported"
        if complete and all(gates.values())
        else "in-progress"
        if not complete
        else "invalid"
    )
    trace_digest = hashlib.sha256(
        b"".join(_canonical_bytes(record) + b"\n" for record in records)
    ).hexdigest()
    manifest = {
        "schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "suite_id": f"gate11-longitudinal-source-seed-{seed}",
        "seed": seed,
        "seed_schedule": list(GATE11_LONGITUDINAL_SOURCE_SEEDS),
        "model_id": SHARED_SETTLED_TRACE_MODEL_ID,
        "runtime_mode": "strict-local",
        "fallback_mode": "deny",
        "runtime_fingerprint": runtime_fingerprint,
        "partition_counts": dict(SHARED_SETTLED_TRACE_PARTITION_COUNTS),
        "session_size": GATE11_LONGITUDINAL_SESSION_SIZE,
        "context_registry": [
            asdict(context) for context in SHARED_SETTLED_TRACE_CONTEXTS
        ],
        "locked_confirmation": True,
        "consumer_loading_rule": "whole-three-seed-campaign-only",
        "trace_sha256": trace_digest,
        "required_files": list(GATE11_LONGITUDINAL_SOURCE_REQUIRED_FILES),
        "provenance": {
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "working_tree_dirty": (
                _git_output("status", "--porcelain")
                not in {"", "unknown"}
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    ablation = {
        "schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "status": status,
        "completed_transition_count": len(records),
        "expected_transition_count": len(plans),
        "partition_counts": partition_counts,
        "fresh_capture_session_count": len(session_ids),
        "consumer_session_boundary_interval": (
            GATE11_LONGITUDINAL_SESSION_SIZE
        ),
        "lineage": {
            "coverage": float(lineage_complete) if records else 0.0,
            "accepted_mismatch_count": mismatch_count,
            "duplicate_settlement_count": duplicate_count,
        },
        "substrate": {
            "runtime_origins": sorted(
                {
                    record["substrate"]["runtime_origin"]
                    for record in records
                }
            ),
            "fallback_count": fallback_count,
            "empty_residual_count": empty_residual_count,
            "mutation_count": mutation_count,
        },
        "gates": gates,
    }
    verdict = {
        "schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "gate_scope": "Gate 11 / Gate 5 fresh longitudinal source",
        "status": status,
        "consumer_admission": (
            "seed-ready" if status == "trace-contract-supported" else "denied"
        ),
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
    }
    rollback = {
        "schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "runtime_owner_mutated_by_export": False,
        "substrate_mutation_count": mutation_count,
        "rollback_action": (
            "stop generation; retain immutable settled prefix and "
            "filesystem owner checkpoints"
        ),
    }
    _write_json(output_dir / "manifest.yaml", manifest)
    _write_json(output_dir / "ablation_results.json", ablation)
    _write_json(output_dir / "promotion_verdict.json", verdict)
    _write_json(output_dir / "rollback_evidence.json", rollback)
    row_specs = {
        "predictions.jsonl": lambda record: {
            **record["lineage"],
            "transition_id": record["transition_id"],
            "partition": record["partition"],
            "prediction": record["prediction"],
        },
        "outcomes.jsonl": lambda record: {
            **record["lineage"],
            "transition_id": record["transition_id"],
            "partition": record["partition"],
            "actual_outcome": record["actual_outcome"],
            "environment_outcome": record["environment_outcome"],
        },
        "prediction_errors.jsonl": lambda record: {
            **record["lineage"],
            "transition_id": record["transition_id"],
            "partition": record["partition"],
            "prediction_error": record["prediction_error"],
        },
        "segments.jsonl": lambda record: {
            "transition_id": record["transition_id"],
            "partition": record["partition"],
            "context_id": record["context_id"],
            "temporal_snapshot": record["temporal_snapshot"],
        },
        "credit.jsonl": lambda record: {
            "transition_id": record["transition_id"],
            "partition": record["partition"],
            "credit_snapshot": record["credit_snapshot"],
        },
        "state_diff.jsonl": lambda record: {
            "transition_id": record["transition_id"],
            "partition": record["partition"],
            "episode_phase": record["episode_phase"],
            "knowledge_key": record["knowledge_key"],
            "memory_snapshot": record["memory_snapshot"],
            "longitudinal": record["longitudinal"],
        },
        "action_selection.jsonl": lambda record: {
            "transition_id": record["transition_id"],
            "partition": record["partition"],
            "action_selection": record["action_selection"],
        },
    }
    for name, builder in row_specs.items():
        _write_jsonl(output_dir / name, [builder(record) for record in records])
    (output_dir / "report.md").write_text(
        "\n".join(
            (
                "# Gate 11 / Gate 5 fresh longitudinal source",
                "",
                f"- seed: `{seed}`",
                f"- status: `{status}`",
                f"- settled transitions: `{len(records)}/{len(plans)}`",
                f"- fresh capture micro-sessions: `{len(session_ids)}`",
                (
                    "- consumer persistence boundary interval: "
                    f"`{GATE11_LONGITUDINAL_SESSION_SIZE}`"
                ),
                f"- trace sha256: `{trace_digest}`",
                "",
            )
        ),
        encoding="utf-8",
    )
    written = tuple(
        output_dir / name
        for name in GATE11_LONGITUDINAL_SOURCE_REQUIRED_FILES
    )
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(
            f"Gate 11 longitudinal source missing {missing!r}"
        )
    return written


async def generate_gate11_longitudinal_source(
    *,
    output_dir: str | Path,
    seed: int,
    max_transitions: int | None = None,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    plans = build_gate11_longitudinal_source_plans(seed)
    transition_path = target / "transitions.jsonl"
    records = load_gate11_longitudinal_source_records(transition_path)
    validate_gate11_longitudinal_source_prefix(
        records=records,
        plans=plans,
    )
    runtime = build_strict_local_shared_trace_runtime()
    if runtime.runtime_origin != "hf-local" or not runtime.is_frozen:
        raise RuntimeError(
            "Gate 11 longitudinal strict-local frozen runtime unavailable"
        )
    runtime_fingerprint = shared_trace_runtime_fingerprint(runtime)
    stop_count = len(plans)
    if max_transitions is not None:
        if max_transitions < 1:
            raise ValueError("max_transitions must be positive")
        stop_count = min(stop_count, max_transitions)
    for plan_index in range(len(records), stop_count):
        plan = plans[plan_index]
        record = await run_shared_trace_transition(
            plan=plan,
            runtime=runtime,
            schema_version=GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
            config=_capture_config(),
        )
        record.pop("record_sha256", None)
        record["longitudinal"] = {
            "capture_mode": "fresh-isolated-micro-session",
            "consumer_session_boundary_interval": (
                GATE11_LONGITUDINAL_SESSION_SIZE
            ),
            "consumer_owner_persistence_required": True,
        }
        record["record_sha256"] = hashlib.sha256(
            _canonical_bytes(record)
        ).hexdigest()
        _append_jsonl(transition_path, record)
        records.append(record)
        _write_json(
            target / "progress.json",
            {
                "schema_version": (
                    GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION
                ),
                "seed": seed,
                "completed_transition_count": len(records),
                "total_transition_count": len(plans),
                "runtime_fingerprint": runtime_fingerprint,
            },
        )
    return _derive_seed_packet(
        output_dir=target,
        seed=seed,
        records=records,
        runtime_fingerprint=runtime_fingerprint,
    )


def aggregate_gate11_longitudinal_source(
    *,
    campaign_dir: str | Path,
) -> tuple[Path, ...]:
    root = Path(campaign_dir)
    summaries: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    trace_digests: set[str] = set()
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        seed_dir = root / f"seed_{seed}"
        manifest = json.loads(
            (seed_dir / "manifest.yaml").read_text(encoding="utf-8")
        )
        verdict = json.loads(
            (seed_dir / "promotion_verdict.json").read_text(
                encoding="utf-8"
            )
        )
        records = load_gate11_longitudinal_source_records(
            seed_dir / "transitions.jsonl"
        )
        validate_gate11_longitudinal_source_prefix(
            records=records,
            plans=build_gate11_longitudinal_source_plans(seed),
        )
        actual_digest = hashlib.sha256(
            b"".join(_canonical_bytes(record) + b"\n" for record in records)
        ).hexdigest()
        if actual_digest != manifest["trace_sha256"]:
            raise ValueError(
                f"Longitudinal seed {seed} manifest digest mismatch"
            )
        fingerprints.add(str(manifest["runtime_fingerprint"]))
        trace_digests.add(actual_digest)
        summaries.append(
            {
                "seed": seed,
                "status": verdict["status"],
                "transition_count": len(records),
                "runtime_fingerprint": manifest["runtime_fingerprint"],
                "trace_sha256": actual_digest,
            }
        )
    gates = {
        "three_seed_packets_present": (
            len(summaries) == len(GATE11_LONGITUDINAL_SOURCE_SEEDS)
        ),
        "settled_transition_count_1530": sum(
            summary["transition_count"] for summary in summaries
        )
        == (
            SHARED_SETTLED_TRACE_COUNT_PER_SEED
            * len(GATE11_LONGITUDINAL_SOURCE_SEEDS)
        ),
        "all_seed_contracts_supported": all(
            summary["status"] == "trace-contract-supported"
            for summary in summaries
        ),
        "runtime_fingerprint_shared": len(fingerprints) == 1,
        "trace_digests_distinct": (
            len(trace_digests) == len(GATE11_LONGITUDINAL_SOURCE_SEEDS)
        ),
    }
    status = (
        "trace-contract-supported" if all(gates.values()) else "invalid"
    )
    manifest = {
        "schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "campaign_id": "gate11-longitudinal-source-20260730",
        "seed_schedule": list(GATE11_LONGITUDINAL_SOURCE_SEEDS),
        "runtime_fingerprint": (
            next(iter(fingerprints)) if len(fingerprints) == 1 else None
        ),
        "expected_transition_count": (
            SHARED_SETTLED_TRACE_COUNT_PER_SEED
            * len(GATE11_LONGITUDINAL_SOURCE_SEEDS)
        ),
        "locked_confirmation": True,
        "consumer_loading_rule": "whole-three-seed-campaign-only",
        "seed_summaries": summaries,
    }
    verdict = {
        "schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "gate_scope": "Gate 11 / Gate 5 longitudinal source aggregate",
        "status": status,
        "consumer_admission": (
            "allowed" if status == "trace-contract-supported" else "denied"
        ),
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "gates": gates,
    }
    report = "\n".join(
        (
            "# Gate 11 / Gate 5 longitudinal source aggregate",
            "",
            f"- status: `{status}`",
            f"- consumer admission: `{verdict['consumer_admission']}`",
            (
                "- settled transitions: "
                f"`{sum(s['transition_count'] for s in summaries)}`"
            ),
            "",
        )
    )
    _write_json(root / "aggregate_manifest.json", manifest)
    _write_json(root / "aggregate_verdict.json", verdict)
    (root / "aggregate_report.md").write_text(report, encoding="utf-8")
    return (
        root / "aggregate_manifest.json",
        root / "aggregate_verdict.json",
        root / "aggregate_report.md",
    )


__all__ = [
    "GATE11_LONGITUDINAL_SESSION_SIZE",
    "GATE11_LONGITUDINAL_SOURCE_REQUIRED_FILES",
    "GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION",
    "GATE11_LONGITUDINAL_SOURCE_SEEDS",
    "aggregate_gate11_longitudinal_source",
    "build_gate11_longitudinal_source_plans",
    "generate_gate11_longitudinal_source",
    "load_gate11_longitudinal_source_records",
    "validate_gate11_longitudinal_source_prefix",
]
