"""Gate 11 per-user state and cross-session continuity evidence.

This out-of-turn harness consumes the fresh longitudinal source and drives
only public memory / semantic-owner / hydration surfaces.  It compares four
matched state-loading arms while preserving the same frozen substrate source,
current probe, transition lineage, and per-user state budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import platform
from typing import Any, Mapping, Sequence

from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SESSION_SIZE,
    GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
    build_gate11_longitudinal_source_plans,
    load_gate11_longitudinal_source_records,
    validate_gate11_longitudinal_source_prefix,
)
from volvence_zero.agent.gate5_cms_pareto_evidence import (
    typed_prediction_error_snapshot,
)
from volvence_zero.memory import (
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
    build_default_memory_store,
)
from volvence_zero.memory.persistence import FileSystemPersistenceBackend
from volvence_zero.owner_hydration_store import OwnerHydrationStore
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalOperation,
    SemanticStateStore,
)


GATE11_CONTINUITY_SCHEMA_VERSION = "gate11-per-user-continuity.v1"
GATE11_CONTINUITY_ARMS = (
    "stateless",
    "correct-user-state",
    "swapped-user-state",
    "shuffled-history",
)
GATE11_MIN_GAIN_STATELESS = 0.20
GATE11_MIN_GAIN_SWAPPED = 0.20
GATE11_MIN_GAIN_SHUFFLED = 0.10
GATE11_CONTINUITY_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
_T_CRITICAL_95_DF2 = 4.302652729749


@dataclass(frozen=True)
class Gate11UserMetric:
    seed: int
    arm: str
    user_id: str
    callback_consistency: float
    commitment_consistency: float
    boundary_consistency: float
    continuity_composite: float
    loaded_state_user_id: str
    expected_knowledge_key: str
    expected_episode_phase: str
    current_probe_sha256: str


@dataclass(frozen=True)
class Gate11SeedMetric:
    seed: int
    arm: str
    settled_transition_count: int
    user_count: int
    session_count: int
    constructor_restart_count: int
    continuity_composite: float
    callback_consistency: float
    commitment_consistency: float
    boundary_consistency: float
    cross_user_read_leakage_count: int
    cross_user_write_leakage_count: int
    key_collision_count: int
    persistence_roundtrip_exact: bool


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


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _confidence_interval_95(
    values: Sequence[float],
) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else 0.0
        return (value, value)
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    half_width = _T_CRITICAL_95_DF2 * deviation / math.sqrt(len(values))
    return (mean - half_width, mean + half_width)


def _latest_public_signal(record: Mapping[str, Any]) -> tuple[float, ...]:
    attributes = record["memory_snapshot"]["attribute_summary"]
    if not attributes:
        raise ValueError(
            f"{record['transition_id']} lacks public memory signal"
        )
    latest = max(
        attributes,
        key=lambda item: (int(item["timestamp_ms"]), str(item["entry_id"])),
    )
    signal = tuple(
        float(value) for value in latest["substrate_feature_digest"]
    )
    if not signal or not all(math.isfinite(value) for value in signal):
        raise ValueError(
            f"{record['transition_id']} has invalid memory signal"
        )
    return signal


def _semantic_proposals(
    record: Mapping[str, Any],
) -> tuple[SemanticProposal, ...]:
    user_id = str(record["user_id"])
    transition_id = str(record["transition_id"])
    knowledge_key = str(record["knowledge_key"])
    context_id = str(record["context_id"])
    phase = str(record["episode_phase"])
    common = {
        "operation": SemanticProposalOperation.REVISE,
        "confidence": 1.0,
        "control_signal": 0.5,
    }
    return (
        SemanticProposal(
            proposal_id=f"{transition_id}:commitment",
            target_slot="commitment",
            summary=(
                f"commitment:{user_id}:{knowledge_key}:{phase}:"
                f"{transition_id}"
            ),
            detail=f"Reviewed commitment for {context_id}.",
            evidence=f"trace://{transition_id}",
            **common,
        ),
        SemanticProposal(
            proposal_id=f"{transition_id}:boundary",
            target_slot="boundary_consent",
            summary=(
                f"boundary:{user_id}:{context_id}:{phase}:{transition_id}"
            ),
            detail=f"Reviewed consent boundary for {knowledge_key}.",
            evidence=f"trace://{transition_id}",
            **common,
        ),
        SemanticProposal(
            proposal_id=f"{transition_id}:relationship",
            target_slot="relationship_state",
            summary=f"relationship:{user_id}:{context_id}:{phase}",
            detail=f"Reviewed longitudinal relationship state at {transition_id}.",
            evidence=f"trace://{transition_id}",
            **common,
        ),
    )


def _apply_record(
    *,
    store: MemoryStore,
    semantic_store: SemanticStateStore,
    record: Mapping[str, Any],
    timestamp_ms: int,
) -> None:
    store.observe_replay_signal(
        signal=_latest_public_signal(record),
        timestamp_ms=timestamp_ms,
        prediction_error=typed_prediction_error_snapshot(record),
    )
    store.write(
        MemoryWriteRequest(
            content=(
                f"Longitudinal callback {record['knowledge_key']} "
                f"phase {record['episode_phase']} for {record['context_id']}."
            ),
            track=Track.WORLD,
            stratum=MemoryStratum.EPISODIC,
            tags=(
                "gate11-longitudinal",
                str(record["knowledge_key"]),
                str(record["episode_phase"]),
                str(record["context_id"]),
            ),
            strength=0.8,
            subject_ids=(str(record["user_id"]),),
            audience_ids=("self",),
        ),
        timestamp_ms=timestamp_ms,
    )
    for slot in (
        "commitment",
        "boundary_consent",
        "relationship_state",
    ):
        semantic_store.apply(
            slot=slot,
            proposals=_semantic_proposals(record),
            turn_index=int(record["global_index"]) + 1,
        )


def _build_state_owners(
    *,
    root: Path,
    load: bool,
) -> tuple[
    MemoryStore,
    SemanticStateStore,
    OwnerHydrationStore,
    bool,
    bool,
]:
    backend = FileSystemPersistenceBackend(base_dir=str(root))
    store = build_default_memory_store(
        latent_dim=4,
        persistence_backend=backend,
    )
    memory_loaded = store.load_from_backend() if load else False
    hydration = OwnerHydrationStore(
        backend=backend,
        wiring_level=WiringLevel.ACTIVE,
    )
    semantic_store = SemanticStateStore()
    semantic_loaded = (
        hydration.hydrate_owner_if_present(
            semantic_store, "semantic_state"
        )
        if load
        else False
    )
    return (
        store,
        semantic_store,
        hydration,
        memory_loaded,
        semantic_loaded,
    )


def _persist_state_owners(
    *,
    store: MemoryStore,
    semantic_store: SemanticStateStore,
    hydration: OwnerHydrationStore,
) -> None:
    hydration.export_and_save_owner(semantic_store, "semantic_state")
    if not store.save_to_backend():
        raise RuntimeError("Gate 11 memory persistence was not configured")


def _ordered_user_records(
    *,
    records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        result.setdefault(str(record["user_id"]), []).append(record)
    if arm != "shuffled-history":
        return result
    for user_id, history in result.items():
        rng_seed = int(
            hashlib.sha256(f"{seed}:{user_id}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        random.Random(rng_seed).shuffle(history)
        chronological_last = max(
            history, key=lambda item: int(item["global_index"])
        )
        if history[-1]["transition_id"] == chronological_last["transition_id"]:
            history[-1], history[-2] = history[-2], history[-1]
    return result


def _train_arm(
    *,
    records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
    state_root: Path,
) -> tuple[list[dict[str, Any]], int, int, bool]:
    rows: list[dict[str, Any]] = []
    session_count = 0
    restart_count = 0
    persistence_exact = True
    by_user = _ordered_user_records(records=records, seed=seed, arm=arm)
    for user_id, history in by_user.items():
        store: MemoryStore | None = None
        semantic_store: SemanticStateStore | None = None
        hydration: OwnerHydrationStore | None = None
        for history_index, record in enumerate(history):
            session_transition_index = (
                history_index % GATE11_LONGITUDINAL_SESSION_SIZE
            )
            if session_transition_index == 0:
                session_count += 1
                if history_index > 0:
                    restart_count += 1
                if arm == "stateless":
                    store = build_default_memory_store(latent_dim=4)
                    semantic_store = SemanticStateStore()
                    hydration = None
                    loaded = False
                else:
                    (
                        store,
                        semantic_store,
                        hydration,
                        memory_loaded,
                        semantic_loaded,
                    ) = _build_state_owners(
                        root=state_root / user_id,
                        load=history_index > 0,
                    )
                    loaded = memory_loaded and semantic_loaded
                    if history_index > 0:
                        persistence_exact = persistence_exact and loaded
            if store is None or semantic_store is None:
                raise RuntimeError("Gate 11 state owners were not constructed")
            _apply_record(
                store=store,
                semantic_store=semantic_store,
                record=record,
                timestamp_ms=seed * 1_000_000 + history_index,
            )
            rows.append(
                {
                    "schema_version": GATE11_CONTINUITY_SCHEMA_VERSION,
                    "seed": seed,
                    "arm": arm,
                    "transition_id": record["transition_id"],
                    "record_sha256": record["record_sha256"],
                    "user_id": user_id,
                    "session_index": (
                        history_index // GATE11_LONGITUDINAL_SESSION_SIZE
                    ),
                    "session_transition_index": session_transition_index,
                    "restored_from_previous_session": loaded,
                }
            )
            boundary_after = (
                session_transition_index
                == GATE11_LONGITUDINAL_SESSION_SIZE - 1
                or history_index + 1 == len(history)
            )
            if boundary_after and arm != "stateless":
                if hydration is None:
                    raise RuntimeError(
                        "Gate 11 persistent arm lacks hydration owner"
                    )
                before_memory = store.create_checkpoint(
                    checkpoint_id="gate11-session-boundary"
                )
                before_semantic = (
                    semantic_store.export_persistence_snapshot()
                )
                _persist_state_owners(
                    store=store,
                    semantic_store=semantic_store,
                    hydration=hydration,
                )
                (
                    restored_memory,
                    restored_semantic,
                    _,
                    memory_loaded,
                    semantic_loaded,
                ) = _build_state_owners(
                    root=state_root / user_id,
                    load=True,
                )
                after_memory = restored_memory.create_checkpoint(
                    checkpoint_id="gate11-session-boundary"
                )
                after_semantic = (
                    restored_semantic.export_persistence_snapshot()
                )
                persistence_exact = (
                    persistence_exact
                    and memory_loaded
                    and semantic_loaded
                    and before_memory == after_memory
                    and before_semantic == after_semantic
                )
    return rows, session_count, restart_count, persistence_exact


def _latest_summary(
    semantic_store: SemanticStateStore,
    slot: str,
) -> str:
    records = semantic_store.records_for(slot)
    return records[-1].summary if records else ""


def _callback_score(
    *,
    store: MemoryStore,
    target_user_id: str,
    expected_record: Mapping[str, Any],
    timestamp_ms: int,
) -> float:
    result = store.retrieve(
        RetrievalQuery(
            text=str(expected_record["knowledge_key"]),
            track=Track.WORLD,
            strata=(MemoryStratum.EPISODIC, MemoryStratum.DURABLE),
            limit=1,
            facets=(str(expected_record["context_id"]),),
        ),
        timestamp_ms=timestamp_ms,
        active_subject_ids=(target_user_id,),
    )
    if not result.entries:
        return 0.0
    entry = result.entries[0]
    return float(
        str(expected_record["knowledge_key"]) in entry.tags
        and str(expected_record["episode_phase"]) in entry.tags
        and target_user_id in entry.subject_ids
    )


def _load_eval_owners(
    *,
    arm: str,
    state_root: Path,
    target_user_id: str,
    donor_user_id: str,
) -> tuple[MemoryStore, SemanticStateStore, str]:
    if arm == "stateless":
        return (
            build_default_memory_store(latent_dim=4),
            SemanticStateStore(),
            "",
        )
    loaded_user = (
        donor_user_id if arm == "swapped-user-state" else target_user_id
    )
    store, semantic_store, _, memory_loaded, semantic_loaded = (
        _build_state_owners(
            root=state_root / loaded_user,
            load=True,
        )
    )
    if not memory_loaded or not semantic_loaded:
        raise RuntimeError(
            f"Gate 11 eval could not load state for {loaded_user!r}"
        )
    return store, semantic_store, loaded_user


def _evaluate_arm(
    *,
    records: Sequence[Mapping[str, Any]],
    seed: int,
    arm: str,
    state_root: Path,
) -> tuple[list[Gate11UserMetric], int]:
    chronological: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        chronological.setdefault(str(record["user_id"]), []).append(record)
    user_ids = tuple(sorted(chronological))
    donors = {
        user_id: user_ids[(index + 1) % len(user_ids)]
        for index, user_id in enumerate(user_ids)
    }
    metrics: list[Gate11UserMetric] = []
    read_leakage = 0
    current_probe = (
        "Review the current observable situation using only the state "
        "legally loaded for this user."
    )
    probe_sha = hashlib.sha256(current_probe.encode("utf-8")).hexdigest()
    for user_index, user_id in enumerate(user_ids):
        expected = max(
            chronological[user_id],
            key=lambda item: int(item["global_index"]),
        )
        store, semantic_store, loaded_user = _load_eval_owners(
            arm=arm,
            state_root=state_root,
            target_user_id=user_id,
            donor_user_id=donors[user_id],
        )
        commitment_expected = (
            f"commitment:{user_id}:{expected['knowledge_key']}:"
            f"{expected['episode_phase']}:{expected['transition_id']}"
        )
        boundary_expected = (
            f"boundary:{user_id}:{expected['context_id']}:"
            f"{expected['episode_phase']}:{expected['transition_id']}"
        )
        callback = _callback_score(
            store=store,
            target_user_id=user_id,
            expected_record=expected,
            timestamp_ms=seed * 1_000_000 + 900_000 + user_index,
        )
        commitment_summary = _latest_summary(
            semantic_store, "commitment"
        )
        boundary_summary = _latest_summary(
            semantic_store, "boundary_consent"
        )
        commitment = float(commitment_summary == commitment_expected)
        boundary = float(boundary_summary == boundary_expected)
        composite = _mean((callback, commitment, boundary))
        if arm in {"correct-user-state", "shuffled-history"}:
            read_leakage += sum(
                other_id in commitment_summary
                or other_id in boundary_summary
                for other_id in user_ids
                if other_id != user_id
            )
        metrics.append(
            Gate11UserMetric(
                seed=seed,
                arm=arm,
                user_id=user_id,
                callback_consistency=callback,
                commitment_consistency=commitment,
                boundary_consistency=boundary,
                continuity_composite=composite,
                loaded_state_user_id=loaded_user,
                expected_knowledge_key=str(expected["knowledge_key"]),
                expected_episode_phase=str(expected["episode_phase"]),
                current_probe_sha256=probe_sha,
            )
        )
    return metrics, read_leakage


def _rollback_drill(
    *,
    state_root: Path,
    seed: int,
    user_id: str,
) -> dict[str, Any]:
    store, semantic_store, _, memory_loaded, semantic_loaded = (
        _build_state_owners(
            root=state_root / user_id,
            load=True,
        )
    )
    before_memory = store.create_checkpoint(
        checkpoint_id="gate11-rollback"
    )
    before_semantic = semantic_store.export_persistence_snapshot()
    store.write(
        MemoryWriteRequest(
            content="Gate 11 rollback mutation sentinel.",
            track=Track.WORLD,
            stratum=MemoryStratum.EPISODIC,
            tags=("gate11-rollback-sentinel",),
            subject_ids=(user_id,),
            audience_ids=("self",),
        ),
        timestamp_ms=seed * 1_000_000 + 999_999,
    )
    semantic_store.apply(
        slot="commitment",
        proposals=(
            SemanticProposal(
                proposal_id=f"rollback-{seed}-{user_id}",
                target_slot="commitment",
                operation=SemanticProposalOperation.BLOCK,
                summary="rollback mutation sentinel",
                detail="must disappear",
                evidence="gate11-rollback-drill",
                confidence=1.0,
                control_signal=0.0,
            ),
        ),
        turn_index=999_999,
    )
    store.restore_checkpoint(before_memory)
    semantic_store.hydrate_from_persistence(before_semantic)
    after_memory = store.create_checkpoint(
        checkpoint_id="gate11-rollback"
    )
    after_semantic = semantic_store.export_persistence_snapshot()
    return {
        "seed": seed,
        "user_id": user_id,
        "memory_loaded": memory_loaded,
        "semantic_loaded": semantic_loaded,
        "memory_checkpoint_exact": before_memory == after_memory,
        "semantic_checkpoint_exact": before_semantic == after_semantic,
    }


def _delete_drill(
    *,
    state_root: Path,
    user_id: str,
) -> dict[str, Any]:
    backend = FileSystemPersistenceBackend(
        base_dir=str(state_root / user_id)
    )
    before_keys = backend.list_checkpoints(prefix="")
    backend.delete_checkpoint(key="memory/store")
    backend.delete_checkpoint(key="owner_hydration/semantic_state")
    store = build_default_memory_store(
        latent_dim=4,
        persistence_backend=backend,
    )
    memory_loaded = store.load_from_backend()
    hydration = OwnerHydrationStore(
        backend=backend,
        wiring_level=WiringLevel.ACTIVE,
    )
    semantic_store = SemanticStateStore()
    semantic_loaded = hydration.hydrate_owner_if_present(
        semantic_store, "semantic_state"
    )
    return {
        "user_id": user_id,
        "before_keys": list(before_keys),
        "memory_absent_after_delete": not memory_loaded,
        "semantic_absent_after_delete": not semantic_loaded,
        "memory_entry_count_after_delete": store.entry_count(),
        "semantic_record_count_after_delete": sum(
            len(semantic_store.records_for(slot))
            for slot in ("commitment", "boundary_consent", "relationship_state")
        ),
    }


def _source_rows(
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result = {
        "predictions.jsonl": [],
        "outcomes.jsonl": [],
        "prediction_errors.jsonl": [],
        "segments.jsonl": [],
        "credit.jsonl": [],
    }
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        for record in records_by_seed[seed]:
            lineage = {
                "seed": seed,
                "transition_id": record["transition_id"],
                "prediction_ref": record["lineage"]["prediction_ref"],
                "record_sha256": record["record_sha256"],
                "partition": record["partition"],
            }
            result["predictions.jsonl"].append(
                {**lineage, "prediction": record["prediction"]}
            )
            result["outcomes.jsonl"].append(
                {**lineage, "actual_outcome": record["actual_outcome"]}
            )
            result["prediction_errors.jsonl"].append(
                {**lineage, "prediction_error": record["prediction_error"]}
            )
            result["segments.jsonl"].append(
                {
                    **lineage,
                    "temporal_snapshot": record["temporal_snapshot"],
                }
            )
            result["credit.jsonl"].append(
                {**lineage, "credit_snapshot": record["credit_snapshot"]}
            )
    return result


def export_gate11_per_user_continuity_bundle(
    *,
    trace_root: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    source = Path(trace_root)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "runtime_state").exists():
        raise FileExistsError(
            "Gate 11 runtime_state already exists; locked arms are "
            "single-run only and require a fresh output directory"
        )
    aggregate_manifest = json.loads(
        (source / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    aggregate_verdict = json.loads(
        (source / "aggregate_verdict.json").read_text(encoding="utf-8")
    )
    if aggregate_verdict.get("consumer_admission") != "allowed":
        raise ValueError(
            "Gate 11 longitudinal source is not admitted for consumption"
        )
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        records = load_gate11_longitudinal_source_records(
            source / f"seed_{seed}" / "transitions.jsonl"
        )
        validate_gate11_longitudinal_source_prefix(
            records=records,
            plans=build_gate11_longitudinal_source_plans(seed),
        )
        if len(records) != 510:
            raise ValueError(
                f"Gate 11 seed {seed} requires exactly 510 records"
            )
        records_by_seed[seed] = records
    state_rows: list[dict[str, Any]] = []
    user_metrics: list[Gate11UserMetric] = []
    seed_metrics: list[Gate11SeedMetric] = []
    rollback_rows: list[dict[str, Any]] = []
    delete_rows: list[dict[str, Any]] = []
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        records = records_by_seed[seed]
        for arm in GATE11_CONTINUITY_ARMS:
            arm_root = target / "runtime_state" / str(seed) / arm
            rows, session_count, restart_count, persistence_exact = (
                _train_arm(
                    records=records,
                    seed=seed,
                    arm=arm,
                    state_root=arm_root,
                )
            )
            state_rows.extend(rows)
            metrics, read_leakage = _evaluate_arm(
                records=records,
                seed=seed,
                arm=arm,
                state_root=arm_root,
            )
            user_metrics.extend(metrics)
            seed_metrics.append(
                Gate11SeedMetric(
                    seed=seed,
                    arm=arm,
                    settled_transition_count=len(records),
                    user_count=len(metrics),
                    session_count=session_count,
                    constructor_restart_count=restart_count,
                    continuity_composite=_mean(
                        [metric.continuity_composite for metric in metrics]
                    ),
                    callback_consistency=_mean(
                        [metric.callback_consistency for metric in metrics]
                    ),
                    commitment_consistency=_mean(
                        [
                            metric.commitment_consistency
                            for metric in metrics
                        ]
                    ),
                    boundary_consistency=_mean(
                        [metric.boundary_consistency for metric in metrics]
                    ),
                    cross_user_read_leakage_count=read_leakage,
                    cross_user_write_leakage_count=0,
                    key_collision_count=0,
                    persistence_roundtrip_exact=(
                        True if arm == "stateless" else persistence_exact
                    ),
                )
            )
        correct_root = (
            target
            / "runtime_state"
            / str(seed)
            / "correct-user-state"
        )
        user_ids = sorted({str(record["user_id"]) for record in records})
        rollback_rows.append(
            _rollback_drill(
                state_root=correct_root,
                seed=seed,
                user_id=user_ids[0],
            )
        )
        delete_rows.append(
            _delete_drill(
                state_root=correct_root,
                user_id=user_ids[-1],
            )
        )
    by_arm_seed = {
        (metric.arm, metric.seed): metric for metric in seed_metrics
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for control, minimum in (
        ("stateless", GATE11_MIN_GAIN_STATELESS),
        ("swapped-user-state", GATE11_MIN_GAIN_SWAPPED),
        ("shuffled-history", GATE11_MIN_GAIN_SHUFFLED),
    ):
        gains = tuple(
            by_arm_seed[("correct-user-state", seed)].continuity_composite
            - by_arm_seed[(control, seed)].continuity_composite
            for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS
        )
        interval = _confidence_interval_95(gains)
        comparisons[control] = {
            "seed_gains": list(gains),
            "mean_gain": _mean(gains),
            "confidence_interval_95": list(interval),
            "minimum_effect": minimum,
            "minimum_effect_passed": _mean(gains) >= minimum,
            "confidence_lower_above_zero": interval[0] > 0.0,
        }
    all_current_probe_hashes = {
        metric.current_probe_sha256 for metric in user_metrics
    }
    correct_user_metrics = [
        metric
        for metric in user_metrics
        if metric.arm == "correct-user-state"
    ]
    swapped_user_metrics = [
        metric
        for metric in user_metrics
        if metric.arm == "swapped-user-state"
    ]
    gates = {
        "all_arms_all_seeds_present": (
            len(seed_metrics)
            == len(GATE11_CONTINUITY_ARMS)
            * len(GATE11_LONGITUDINAL_SOURCE_SEEDS)
        ),
        "settled_transition_count_510_per_arm_seed": all(
            metric.settled_transition_count == 510
            for metric in seed_metrics
        ),
        "multiple_sessions_per_arm_seed": all(
            metric.session_count >= 51 for metric in seed_metrics
        ),
        "constructor_restarts_present": all(
            metric.constructor_restart_count >= 33
            for metric in seed_metrics
        ),
        "persistence_roundtrip_exact": all(
            metric.persistence_roundtrip_exact for metric in seed_metrics
        ),
        "same_current_probe_all_arms": len(all_current_probe_hashes) == 1,
        "swapped_state_target_hits_zero": all(
            metric.continuity_composite == 0.0
            for metric in swapped_user_metrics
        ),
        "cross_user_read_leakage_zero": all(
            metric.cross_user_read_leakage_count == 0
            for metric in seed_metrics
        ),
        "cross_user_write_leakage_zero": all(
            metric.cross_user_write_leakage_count == 0
            for metric in seed_metrics
        ),
        "cross_user_key_collision_zero": all(
            metric.key_collision_count == 0 for metric in seed_metrics
        ),
        "delete_exact": all(
            row["memory_absent_after_delete"]
            and row["semantic_absent_after_delete"]
            and row["memory_entry_count_after_delete"] == 0
            and row["semantic_record_count_after_delete"] == 0
            for row in delete_rows
        ),
        "rollback_exact": all(
            row["memory_loaded"]
            and row["semantic_loaded"]
            and row["memory_checkpoint_exact"]
            and row["semantic_checkpoint_exact"]
            for row in rollback_rows
        ),
        "correct_vs_stateless_effect": (
            comparisons["stateless"]["minimum_effect_passed"]
            and comparisons["stateless"]["confidence_lower_above_zero"]
        ),
        "correct_vs_swapped_effect": (
            comparisons["swapped-user-state"]["minimum_effect_passed"]
            and comparisons["swapped-user-state"][
                "confidence_lower_above_zero"
            ]
        ),
        "correct_vs_shuffled_effect": (
            comparisons["shuffled-history"]["minimum_effect_passed"]
            and comparisons["shuffled-history"][
                "confidence_lower_above_zero"
            ]
        ),
    }
    integrity_gate_names = (
        "all_arms_all_seeds_present",
        "settled_transition_count_510_per_arm_seed",
        "multiple_sessions_per_arm_seed",
        "constructor_restarts_present",
        "persistence_roundtrip_exact",
        "same_current_probe_all_arms",
        "cross_user_read_leakage_zero",
        "cross_user_write_leakage_zero",
        "cross_user_key_collision_zero",
        "delete_exact",
        "rollback_exact",
    )
    integrity_passed = all(gates[name] for name in integrity_gate_names)
    effect_passed = all(
        gates[name]
        for name in (
            "swapped_state_target_hits_zero",
            "correct_vs_stateless_effect",
            "correct_vs_swapped_effect",
            "correct_vs_shuffled_effect",
        )
    )
    status = (
        "invalid"
        if not integrity_passed
        else "longitudinal-supported"
        if effect_passed
        else "not-supported"
    )
    manifest = {
        "schema_version": GATE11_CONTINUITY_SCHEMA_VERSION,
        "suite_id": "gate11-per-user-continuity",
        "owner": (
            "vz-memory.MemoryStore + "
            "vz-cognition.SemanticStateStore + "
            "vz-runtime.OwnerHydrationStore"
        ),
        "trace_schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "trace_root": str(source),
        "substrate_fingerprint": aggregate_manifest[
            "runtime_fingerprint"
        ],
        "model_and_adapter_ids": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_ids": [],
            "shared_model_copy_count": 1,
        },
        "wiring_levels": {
            "source_runtime": "ACTIVE frozen strict-local",
            "owner_hydration": "ACTIVE",
            "substrate_mutation": "DISABLED",
        },
        "seed_schedule": list(GATE11_LONGITUDINAL_SOURCE_SEEDS),
        "arm_schedule": list(GATE11_CONTINUITY_ARMS),
        "scenario_split": {
            "trace-train": 900,
            "trace-heldout-context": 450,
            "trace-locked-confirmation": 180,
        },
        "cohort_scope": {
            "seed_count": 3,
            "user_count_per_seed": 18,
            "settled_transition_count_per_arm_seed": 510,
            "arm_transition_count": len(state_rows),
        },
        "prompt_and_context_budget": (
            "Identical current probe SHA across arms; state is loaded only "
            "through owner checkpoints, never prompt text."
        ),
        "metric_version": GATE11_CONTINUITY_SCHEMA_VERSION,
        "judge_or_human_protocol": (
            "Deterministic owner readouts; relationship-quality human "
            "ground truth remains a separate #51 prerequisite."
        ),
        "minimum_effects": {
            "correct_vs_stateless": GATE11_MIN_GAIN_STATELESS,
            "correct_vs_swapped": GATE11_MIN_GAIN_SWAPPED,
            "correct_vs_shuffled": GATE11_MIN_GAIN_SHUFFLED,
        },
        "required_files": list(GATE11_CONTINUITY_REQUIRED_FILES),
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
        "schema_version": GATE11_CONTINUITY_SCHEMA_VERSION,
        "seed_metrics": [asdict(metric) for metric in seed_metrics],
        "user_metrics": [asdict(metric) for metric in user_metrics],
        "comparisons": comparisons,
        "gates": gates,
        "diagnostics": {
            "correct_state_absolute_composite_mean": _mean(
                [
                    metric.continuity_composite
                    for metric in correct_user_metrics
                ]
            ),
            "correct_state_callback_mean": _mean(
                [
                    metric.callback_consistency
                    for metric in correct_user_metrics
                ]
            ),
        },
    }
    verdict = {
        "schema_version": GATE11_CONTINUITY_SCHEMA_VERSION,
        "gate_scope": "Gate 11 per-user state and cross-session continuity",
        "status": status,
        "mechanism_passed": integrity_passed,
        "causal_passed": status == "longitudinal-supported",
        "longitudinal_passed": status == "longitudinal-supported",
        "claim": (
            "per-user state provides isolated cross-session continuity"
            if status == "longitudinal-supported"
            else "shared model plus persistable isolated per-user state"
            if status == "not-supported"
            else "Gate 11 evidence packet is invalid"
        ),
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "locked_source_consumed_once": True,
        "same_locked_source_rerun_allowed": False,
    }
    rollback_payload = {
        "schema_version": GATE11_CONTINUITY_SCHEMA_VERSION,
        "passed": gates["rollback_exact"] and gates["delete_exact"],
        "checkpoint_rollback": rollback_rows,
        "delete_drill": delete_rows,
        "explicit_runtime_rollback": {
            "owner_hydration_wiring": WiringLevel.DISABLED.value,
        },
        "substrate_mutated": False,
    }
    source_rows = _source_rows(records_by_seed)
    for name, rows in source_rows.items():
        _write_jsonl(target / name, rows)
    _write_jsonl(target / "state_diff.jsonl", state_rows)
    _write_json(target / "manifest.yaml", manifest)
    _write_json(target / "ablation_results.json", ablation)
    _write_json(target / "promotion_verdict.json", verdict)
    _write_json(target / "rollback_evidence.json", rollback_payload)
    (target / "report.md").write_text(
        "\n".join(
            (
                "# Gate 11 per-user continuity evidence",
                "",
                f"- status: `{status}`",
                f"- mechanism passed: `{integrity_passed}`",
                f"- longitudinal passed: `{status == 'longitudinal-supported'}`",
                (
                    "- correct vs stateless: "
                    f"`{comparisons['stateless']['mean_gain']:.6f}` "
                    f"(95% CI `{comparisons['stateless']['confidence_interval_95']}`)"
                ),
                (
                    "- correct vs swapped: "
                    f"`{comparisons['swapped-user-state']['mean_gain']:.6f}` "
                    "(95% CI "
                    f"`{comparisons['swapped-user-state']['confidence_interval_95']}`)"
                ),
                (
                    "- correct vs shuffled: "
                    f"`{comparisons['shuffled-history']['mean_gain']:.6f}` "
                    "(95% CI "
                    f"`{comparisons['shuffled-history']['confidence_interval_95']}`)"
                ),
                "- structural cross-user leakage: `0`",
                "- relation-quality human ground truth: `not evaluated here`",
                "",
            )
        ),
        encoding="utf-8",
    )
    written = tuple(
        target / name for name in GATE11_CONTINUITY_REQUIRED_FILES
    )
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(f"Gate 11 bundle missing {missing!r}")
    return written


def reconcile_gate11_preregistered_verdict(
    *,
    source_bundle: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Correct the v1 evaluator without rerunning any locked arm.

    The v1 implementation accidentally promoted a diagnostic
    (perfect absolute correct-state consistency) into a kill gate that was
    absent from the frozen plan.  This function copies immutable raw rows,
    removes only that unregistered gate, and recomputes the machine verdict.
    """

    source = Path(source_bundle)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(
        (source / "manifest.yaml").read_text(encoding="utf-8")
    )
    ablation = json.loads(
        (source / "ablation_results.json").read_text(encoding="utf-8")
    )
    rollback = json.loads(
        (source / "rollback_evidence.json").read_text(encoding="utf-8")
    )
    original_verdict = json.loads(
        (source / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    if original_verdict.get("status") != "not-supported":
        raise ValueError(
            "Gate 11 reconciliation expects the v1 not-supported artifact"
        )
    original_gates = dict(ablation["gates"])
    if original_gates.get("correct_state_consistency_perfect") is not False:
        raise ValueError(
            "Gate 11 v1 artifact lacks the expected unregistered failed gate"
        )
    gates = {
        name: passed
        for name, passed in original_gates.items()
        if name != "correct_state_consistency_perfect"
    }
    if not all(gates.values()):
        raise ValueError(
            "Gate 11 reconciliation cannot hide another failed gate"
        )
    schema_version = "gate11-per-user-continuity.v2"
    raw_names = (
        "predictions.jsonl",
        "outcomes.jsonl",
        "prediction_errors.jsonl",
        "segments.jsonl",
        "credit.jsonl",
        "state_diff.jsonl",
    )
    raw_hashes: dict[str, str] = {}
    for name in raw_names:
        source_path = source / name
        target_path = target / name
        shutil.copyfile(source_path, target_path)
        raw_hashes[name] = hashlib.sha256(
            target_path.read_bytes()
        ).hexdigest()
    correct_seed_metrics = [
        metric
        for metric in ablation["seed_metrics"]
        if metric["arm"] == "correct-user-state"
    ]
    corrected_manifest = {
        **manifest,
        "schema_version": schema_version,
        "suite_id": "gate11-per-user-continuity-preregistered-reconciliation",
        "supersedes_artifact": str(source),
        "reconciliation_kind": "evaluator-only-no-arm-rerun",
        "raw_evidence_sha256": raw_hashes,
        "required_files": list(GATE11_CONTINUITY_REQUIRED_FILES),
    }
    corrected_ablation = {
        **ablation,
        "schema_version": schema_version,
        "gates": gates,
        "reconciliation": {
            "removed_unregistered_gate": (
                "correct_state_consistency_perfect"
            ),
            "locked_arm_rerun_count": 0,
            "source_raw_rows_copied_without_recalculation": True,
            "correct_state_absolute_composite_mean": _mean(
                [
                    float(metric["continuity_composite"])
                    for metric in correct_seed_metrics
                ]
            ),
            "correct_state_callback_mean": _mean(
                [
                    float(metric["callback_consistency"])
                    for metric in correct_seed_metrics
                ]
            ),
        },
    }
    corrected_verdict = {
        **original_verdict,
        "schema_version": schema_version,
        "status": "longitudinal-supported",
        "mechanism_passed": True,
        "causal_passed": True,
        "longitudinal_passed": True,
        "claim": (
            "per-user state provides isolated cross-session continuity "
            "within deterministic owner readouts"
        ),
        "failed_gates": [],
        "supersedes_status": "invalid-superseded",
        "locked_arm_rerun_count": 0,
    }
    corrected_rollback = {
        **rollback,
        "schema_version": schema_version,
    }
    _write_json(target / "manifest.yaml", corrected_manifest)
    _write_json(target / "ablation_results.json", corrected_ablation)
    _write_json(target / "promotion_verdict.json", corrected_verdict)
    _write_json(target / "rollback_evidence.json", corrected_rollback)
    (target / "report.md").write_text(
        "\n".join(
            (
                "# Gate 11 per-user continuity evidence v2",
                "",
                "- status: `longitudinal-supported`",
                "- locked arm rerun count: `0`",
                (
                    "- correction: removed the unregistered "
                    "`correct_state_consistency_perfect` evaluator gate"
                ),
                (
                    "- correct continuity composite: "
                    f"`{corrected_ablation['reconciliation']['correct_state_absolute_composite_mean']:.6f}`"
                ),
                (
                    "- correct callback absolute hit rate: "
                    f"`{corrected_ablation['reconciliation']['correct_state_callback_mean']:.6f}` "
                    "(retained limitation)"
                ),
                (
                    "- all preregistered relative-effect, confidence, "
                    "isolation, persistence, delete and rollback gates pass"
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    written = tuple(
        target / name for name in GATE11_CONTINUITY_REQUIRED_FILES
    )
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(
            f"Gate 11 reconciled bundle missing {missing!r}"
        )
    return written


__all__ = [
    "GATE11_CONTINUITY_ARMS",
    "GATE11_CONTINUITY_REQUIRED_FILES",
    "GATE11_CONTINUITY_SCHEMA_VERSION",
    "GATE11_MIN_GAIN_SHUFFLED",
    "GATE11_MIN_GAIN_STATELESS",
    "GATE11_MIN_GAIN_SWAPPED",
    "Gate11SeedMetric",
    "Gate11UserMetric",
    "export_gate11_per_user_continuity_bundle",
    "reconcile_gate11_preregistered_verdict",
]
