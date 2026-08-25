"""Fresh immutable multi-session source corpus for the Gate 7/8 campaign.

The factory is an out-of-turn evidence utility.  It publishes no runtime slot,
does not infer semantics from text, and carries only structured environment
plans plus numeric context/user priors.  Consumers must verify the admitted
bundle before constructing owner-native traces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


GATE78_TRACE_SCHEMA_VERSION = "gate78-shared-trace.v2"
GATE78_TRACE_SEEDS = (701, 709, 719)
GATE78_PARTITION_COUNTS = (
    ("trace-train", 24),
    ("trace-development-heldout", 12),
    ("trace-locked-confirmation", 12),
)
GATE78_EPISODE_COUNT_PER_SEED = 48
GATE78_SOURCE_DESCRIPTOR = (
    "frozen-synthetic-residual-v2|prefix-only|expert-action-vector|"
    "session-boundary-explicit"
)
GATE7_V3_TRACE_SCHEMA_VERSION = "gate78-shared-trace.v3"
GATE7_V3_TRACE_SEEDS = (727, 733, 739)
GATE7_V3_SOURCE_DESCRIPTOR = (
    "frozen-synthetic-residual-v3|prefix-only|expert-action-vector|"
    "session-boundary-explicit"
)


@dataclass(frozen=True)
class Gate78ContextSpec:
    context_id: str
    domain: str
    route: tuple[str, ...]
    context_centroid: tuple[float, ...]


@dataclass(frozen=True)
class Gate78EpisodePlan:
    episode_id: str
    seed: int
    global_index: int
    partition: str
    context_id: str
    domain: str
    user_prior_id: str
    user_prior: tuple[float, ...]
    context_centroid: tuple[float, ...]
    route: tuple[str, ...]
    action_family_ids: tuple[str, ...]
    segment_lengths: tuple[int, ...]
    difficulty: float
    session_one_turns: tuple[str, ...]
    session_two_turns: tuple[str, ...]
    next_session_boundary: str


@dataclass(frozen=True)
class Gate78TraceProfile:
    schema_version: str
    suite_id: str
    seeds: tuple[int, ...]
    source_descriptor: str


GATE78_V2_TRACE_PROFILE = Gate78TraceProfile(
    schema_version=GATE78_TRACE_SCHEMA_VERSION,
    suite_id="gate78-shared-trace-v2",
    seeds=GATE78_TRACE_SEEDS,
    source_descriptor=GATE78_SOURCE_DESCRIPTOR,
)
GATE7_V3_TRACE_PROFILE = Gate78TraceProfile(
    schema_version=GATE7_V3_TRACE_SCHEMA_VERSION,
    suite_id="gate78-shared-trace-v3",
    seeds=GATE7_V3_TRACE_SEEDS,
    source_descriptor=GATE7_V3_SOURCE_DESCRIPTOR,
)


_CONTEXTS = (
    Gate78ContextSpec(
        context_id="orchard-transfer",
        domain="orchard transfer control",
        route=("entry", "alpha", "beta", "delta"),
        context_centroid=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    Gate78ContextSpec(
        context_id="harbor-compose",
        domain="harbor route composition",
        route=("entry", "beta", "gamma", "epsilon"),
        context_centroid=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ),
    Gate78ContextSpec(
        context_id="clinic-recovery",
        domain="clinic recovery handoff",
        route=("entry", "delta", "beta", "gamma"),
        context_centroid=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    ),
    Gate78ContextSpec(
        context_id="observatory-loop",
        domain="observatory loop planning",
        route=("entry", "hub", "beta", "epsilon"),
        context_centroid=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    ),
    Gate78ContextSpec(
        context_id="workshop-branch",
        domain="workshop branch scheduling",
        route=("entry", "alpha", "gamma", "epsilon"),
        context_centroid=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ),
    Gate78ContextSpec(
        context_id="greenhouse-chain",
        domain="greenhouse chained control",
        route=("entry", "delta", "alpha", "beta", "epsilon"),
        context_centroid=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ),
)
_NON_ACTION_LOCATIONS = {"entry", "hub"}
_USER_PRIORS = (
    (0.10, 0.30, 0.70, 0.90),
    (0.90, 0.10, 0.30, 0.70),
    (0.70, 0.90, 0.10, 0.30),
    (0.30, 0.70, 0.90, 0.10),
)
_SEGMENT_LENGTHS = (2, 3, 4, 5)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _partition_for_index(global_index: int) -> str:
    cursor = 0
    for partition, count in GATE78_PARTITION_COUNTS:
        cursor += count
        if global_index < cursor:
            return partition
    raise IndexError(f"Gate 7/8 episode index out of range: {global_index}")


def build_gate78_episode_plans(
    seed: int,
    *,
    profile: Gate78TraceProfile = GATE78_V2_TRACE_PROFILE,
) -> tuple[Gate78EpisodePlan, ...]:
    if seed not in profile.seeds:
        raise ValueError(f"Gate 7/8 trace seed {seed} is not preregistered")
    plans: list[Gate78EpisodePlan] = []
    seed_offset = profile.seeds.index(seed)
    for global_index in range(GATE78_EPISODE_COUNT_PER_SEED):
        context = _CONTEXTS[(global_index + seed_offset) % len(_CONTEXTS)]
        partition = _partition_for_index(global_index)
        prior_index = (global_index * 3 + seed_offset) % len(_USER_PRIORS)
        action_family_ids = tuple(
            location
            for location in context.route
            if location not in _NON_ACTION_LOCATIONS
        )
        segment_lengths = tuple(
            _SEGMENT_LENGTHS[
                (global_index + action_index + seed_offset)
                % len(_SEGMENT_LENGTHS)
            ]
            for action_index, _family_id in enumerate(action_family_ids)
        )
        difficulty = round(
            0.35 + 0.50 * ((global_index % 12) / 11.0),
            6,
        )
        episode_id = f"gate78-s{seed}-e{global_index:03d}"
        plans.append(
            Gate78EpisodePlan(
                episode_id=episode_id,
                seed=seed,
                global_index=global_index,
                partition=partition,
                context_id=context.context_id,
                domain=context.domain,
                user_prior_id=f"prior-{prior_index}",
                user_prior=_USER_PRIORS[prior_index],
                context_centroid=context.context_centroid,
                route=context.route,
                action_family_ids=action_family_ids,
                segment_lengths=segment_lengths,
                difficulty=difficulty,
                session_one_turns=(
                    (
                        f"Session one observation {episode_id} in "
                        f"{context.domain}; establish the bounded route."
                    ),
                    (
                        f"Session one delayed outcome {episode_id}; close the "
                        "observed segment without adding future state."
                    ),
                ),
                session_two_turns=(
                    (
                        f"Session two cold start {episode_id}; resume from the "
                        "audited prior and compose the next route segment."
                    ),
                    (
                        f"Session two terminal observation {episode_id}; "
                        "record the delayed composition outcome."
                    ),
                ),
                next_session_boundary=f"{episode_id}:session-1->session-2",
            )
        )
    return tuple(plans)


def _episode_row(
    plan: Gate78EpisodePlan,
    *,
    profile: Gate78TraceProfile,
) -> dict[str, object]:
    return {
        **asdict(plan),
        "schema_version": profile.schema_version,
        "session_count": 2,
        "source_descriptor": profile.source_descriptor,
    }


def _write_jsonl(path: Path, rows: tuple[Mapping[str, object], ...]) -> None:
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _pairwise_centroid_mae(
    centroids: tuple[tuple[float, ...], ...],
) -> float:
    distances = tuple(
        sum(abs(left - right) for left, right in zip(a, b, strict=True))
        / len(a)
        for index, a in enumerate(centroids)
        for b in centroids[index + 1 :]
    )
    return min(distances) if distances else 0.0


def _seed_admission(
    rows: tuple[dict[str, object], ...],
    lineage_rows: tuple[dict[str, object], ...],
    *,
    profile: Gate78TraceProfile,
) -> dict[str, object]:
    counts = {
        partition: sum(row["partition"] == partition for row in rows)
        for partition, _count in GATE78_PARTITION_COUNTS
    }
    centroid_by_context = {
        str(row["context_id"]): tuple(float(value) for value in row["context_centroid"])
        for row in rows
    }
    segment_lengths = {
        int(length)
        for row in rows
        for length in row["segment_lengths"]
    }
    action_families = {
        str(family)
        for row in rows
        for family in row["action_family_ids"]
    }
    user_prior_fact_leakage_count = sum(
        any(not isinstance(value, (int, float)) for value in row["user_prior"])
        for row in rows
    )
    gates = {
        "immutable-lineage-complete": (
            len(lineage_rows) == len(rows)
            and all(lineage["row_sha256"] == _sha256(row) for row, lineage in zip(rows, lineage_rows, strict=True))
        ),
        "source-fingerprint-consistent": (
            {row["source_descriptor"] for row in rows}
            == {profile.source_descriptor}
        ),
        "partition-counts-exact": counts
        == dict(GATE78_PARTITION_COUNTS),
        "explicit-session-boundary-complete": all(
            row["session_count"] == 2 and row["next_session_boundary"]
            for row in rows
        ),
        "variable-segment-lengths-present": segment_lengths
        == set(_SEGMENT_LENGTHS),
        "multiple-action-families-present": len(action_families) >= 4,
        "context-centroids-separated": _pairwise_centroid_mae(
            tuple(centroid_by_context.values())
        )
        >= 0.15,
        "user-prior-fact-leakage-zero": user_prior_fact_leakage_count == 0,
        "difficulty-gradient-present": (
            min(float(row["difficulty"]) for row in rows) == 0.35
            and max(float(row["difficulty"]) for row in rows) == 0.85
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "partition_counts": counts,
        "action_families": sorted(action_families),
        "segment_lengths": sorted(segment_lengths),
        "minimum_context_centroid_pairwise_mae": (
            _pairwise_centroid_mae(tuple(centroid_by_context.values()))
        ),
        "user_prior_fact_leakage_count": user_prior_fact_leakage_count,
    }


def export_gate78_shared_trace_bundle(
    *,
    output_dir: str | Path,
    profile: Gate78TraceProfile = GATE78_V2_TRACE_PROFILE,
) -> tuple[Path, ...]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in profile.seeds:
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rows = tuple(
            _episode_row(plan, profile=profile)
            for plan in build_gate78_episode_plans(seed, profile=profile)
        )
        lineage_rows = tuple(
            {
                "episode_id": row["episode_id"],
                "partition": row["partition"],
                "row_sha256": _sha256(row),
                "source_descriptor_sha256": _sha256(
                    row["source_descriptor"]
                ),
            }
            for row in rows
        )
        admission = _seed_admission(
            rows,
            lineage_rows,
            profile=profile,
        )
        manifest = {
            "schema_version": profile.schema_version,
            "suite_id": profile.suite_id,
            "seed": seed,
            "episode_count": len(rows),
            "partition_counts": dict(GATE78_PARTITION_COUNTS),
            "source_descriptor": profile.source_descriptor,
            "source_fingerprint": _sha256(profile.source_descriptor),
            "episodes_sha256": _sha256(rows),
            "lineage_sha256": _sha256(lineage_rows),
            "consumer_admission": admission["passed"],
            "locked_partition": "trace-locked-confirmation",
            "locked_consumption_count": 0,
        }
        files = {
            "episodes.jsonl": rows,
            "lineage.jsonl": lineage_rows,
        }
        for filename, payload in files.items():
            path = seed_dir / filename
            _write_jsonl(path, payload)
            written.append(path)
        for filename, payload in (
            ("manifest.yaml", manifest),
            ("admission.json", admission),
        ):
            path = seed_dir / filename
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(path)
        report_path = seed_dir / "report.md"
        report_path.write_text(
            (
                f"# Gate 7/8 shared trace seed {seed}\n\n"
                f"- schema: `{profile.schema_version}`\n"
                f"- episodes: `{len(rows)}`\n"
                f"- consumer admission: `{admission['passed']}`\n"
                "- locked rows have not been consumed by a development probe.\n"
            ),
            encoding="utf-8",
        )
        written.append(report_path)
        seed_manifests.append(manifest)
    aggregate = {
        "schema_version": profile.schema_version,
        "suite_id": profile.suite_id,
        "seed_schedule": list(profile.seeds),
        "partition_counts_per_seed": dict(GATE78_PARTITION_COUNTS),
        "source_fingerprint": _sha256(profile.source_descriptor),
        "consumer_admission": all(
            bool(manifest["consumer_admission"])
            for manifest in seed_manifests
        ),
        "seed_manifest_sha256": _sha256(seed_manifests),
        "locked_consumption_count": 0,
    }
    aggregate_path = root / "aggregate_manifest.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(aggregate_path)
    return tuple(written)


def verify_gate78_shared_trace_bundle(
    root: str | Path,
    *,
    profile: Gate78TraceProfile = GATE78_V2_TRACE_PROFILE,
) -> dict[str, object]:
    source = Path(root)
    aggregate = json.loads(
        (source / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    seed_results: list[dict[str, object]] = []
    for seed in profile.seeds:
        seed_dir = source / f"seed_{seed}"
        rows = _read_jsonl(seed_dir / "episodes.jsonl")
        lineage_rows = _read_jsonl(seed_dir / "lineage.jsonl")
        manifest = json.loads(
            (seed_dir / "manifest.yaml").read_text(encoding="utf-8")
        )
        admission = _seed_admission(
            rows,
            lineage_rows,
            profile=profile,
        )
        digest_match = (
            manifest["episodes_sha256"] == _sha256(rows)
            and manifest["lineage_sha256"] == _sha256(lineage_rows)
            and manifest["source_fingerprint"]
            == _sha256(profile.source_descriptor)
        )
        seed_results.append(
            {
                "seed": seed,
                "passed": bool(admission["passed"]) and digest_match,
                "digest_match": digest_match,
                "admission": admission,
            }
        )
    passed = (
        aggregate["schema_version"] == profile.schema_version
        and aggregate["suite_id"] == profile.suite_id
        and aggregate["seed_schedule"] == list(profile.seeds)
        and aggregate["source_fingerprint"]
        == _sha256(profile.source_descriptor)
        and all(bool(result["passed"]) for result in seed_results)
    )
    return {
        "schema_version": profile.schema_version,
        "passed": passed,
        "consumer_admission": passed,
        "seed_results": seed_results,
        "locked_consumption_count": aggregate["locked_consumption_count"],
    }


def load_gate78_partition(
    root: str | Path,
    *,
    seed: int,
    partition: str,
    profile: Gate78TraceProfile = GATE78_V2_TRACE_PROFILE,
) -> tuple[Gate78EpisodePlan, ...]:
    verification = verify_gate78_shared_trace_bundle(
        root,
        profile=profile,
    )
    if not verification["consumer_admission"]:
        raise RuntimeError("Gate 7/8 source corpus failed consumer admission")
    if seed not in profile.seeds:
        raise ValueError(f"Gate 7/8 trace seed {seed} is not preregistered")
    allowed = {name for name, _count in GATE78_PARTITION_COUNTS}
    if partition not in allowed:
        raise ValueError(
            f"Unsupported Gate 7/8 partition {partition!r}; "
            f"expected one of {tuple(sorted(allowed))}"
        )
    rows = _read_jsonl(Path(root) / f"seed_{seed}" / "episodes.jsonl")
    return tuple(
        Gate78EpisodePlan(
            episode_id=str(row["episode_id"]),
            seed=int(row["seed"]),
            global_index=int(row["global_index"]),
            partition=str(row["partition"]),
            context_id=str(row["context_id"]),
            domain=str(row["domain"]),
            user_prior_id=str(row["user_prior_id"]),
            user_prior=tuple(float(value) for value in row["user_prior"]),
            context_centroid=tuple(
                float(value) for value in row["context_centroid"]
            ),
            route=tuple(str(value) for value in row["route"]),
            action_family_ids=tuple(
                str(value) for value in row["action_family_ids"]
            ),
            segment_lengths=tuple(
                int(value) for value in row["segment_lengths"]
            ),
            difficulty=float(row["difficulty"]),
            session_one_turns=tuple(
                str(value) for value in row["session_one_turns"]
            ),
            session_two_turns=tuple(
                str(value) for value in row["session_two_turns"]
            ),
            next_session_boundary=str(row["next_session_boundary"]),
        )
        for row in rows
        if row["partition"] == partition
    )
