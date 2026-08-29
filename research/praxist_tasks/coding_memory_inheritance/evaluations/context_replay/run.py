#!/usr/bin/env python3
"""Replay frozen Packet 2 trajectories against one structured context policy."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from lifeform_domain_coding.lab.generation import EnvSpec
from lifeform_domain_coding.lab.tasks import ChainTask, generate_task_chain
from lifeform_domain_coding.lab.trajectory import read_trajectory
from lifeform_evolution.coding_lab_arms import approx_tokens, recall_for_task
from lifeform_evolution.coding_lab_observer import CodingLabChainObserver

TASK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TASK_ROOT.parents[2]
POLICY_SCHEMA_PATH = TASK_ROOT / "assets" / "harness" / "policy.schema.json"
DATASET_MANIFEST_PATH = (
    TASK_ROOT / "assets" / "dataset_metadata" / "public_replay_manifest.json"
)

EVALUATOR_VERSION = "coding-memory-context-replay.v2"
POLICY_VERSION = "coding-memory-inheritance-policy.v1"
MAX_TOKEN_RATIO = 0.10
MIN_RECALLED_SELECTION_COVERAGE = 0.75
MIN_FAILED_SELECTION_COVERAGE = 0.95
REFERENCE_UNITS = 8
EPISODES_PER_CHAIN = 10
ENV_SEED = 20260812
CONVENTION_IDS = ("convention_export_all",)
TRUNCATION_MARKER = "\n... [digest truncated at budget]"

SECTION_IDS = (
    "recalled_entries",
    "memory_description",
    "memory_durable_summary",
    "memory_episodic_summary",
    "prediction_error",
    "plan_intent",
    "open_loop",
    "belief_assumption",
    "execution_result",
    "user_model",
)

DEFAULT_POLICY: dict[str, Any] = {
    "max_context_chars": 3500,
    "max_recalled_entries": 8,
    "sections": list(SECTION_IDS),
    "recalled_entry_order": "owner_order",
    "truncation_strategy": "legacy_tail_marker",
    "deduplicate_exact_lines": False,
    "generic_section_budget_fraction": 1.0,
}

MODE_CHAINS: dict[str, tuple[int, ...]] = {
    "preliminary": (0,),
    "complete": tuple(range(REFERENCE_UNITS)),
}


class EvaluationError(RuntimeError):
    """Raised when candidate or frozen-corpus integrity is invalid."""


@dataclass(frozen=True)
class RenderedPack:
    text: str
    available_entry_lines: tuple[str, ...]
    available_failed_entry_lines: tuple[str, ...]
    selected_entry_lines: tuple[str, ...]
    selected_failed_entry_lines: tuple[str, ...]
    retained_entry_lines: tuple[str, ...]
    retained_failed_entry_lines: tuple[str, ...]
    strict_budget_passed: bool


@dataclass(frozen=True)
class ReplayRow:
    chain_index: int
    episode_index: int
    context_chars: int
    context_tokens: int
    steelman_context_tokens: int
    available_entries: int
    selected_entries: int
    retained_entries: int
    available_failed_entries: int
    selected_failed_entries: int
    retained_failed_entries: int
    strict_budget_passed: bool


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"missing {context}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read {context} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{context} must contain a JSON object: {path}")
    return value


def _load_policy(variant_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_path = variant_dir / "policy.json"
    raw = _load_json(policy_path, context="candidate policy")
    schema = _load_json(POLICY_SCHEMA_PATH, context="policy schema")
    errors = sorted(Draft202012Validator(schema).iter_errors(raw), key=lambda item: list(item.path))
    if errors:
        details = "; ".join(error.message for error in errors[:5])
        raise EvaluationError(f"candidate policy violates policy.schema.json: {details}")
    resolved = dict(DEFAULT_POLICY)
    resolved.update(raw)
    if resolved["schema_version"] != POLICY_VERSION:
        raise EvaluationError(f"unsupported policy schema_version: {resolved['schema_version']!r}")
    if len(set(resolved["sections"])) != len(resolved["sections"]):
        raise EvaluationError("policy sections must be unique")
    if "recalled_entries" not in resolved["sections"]:
        raise EvaluationError("policy sections must retain the recalled_entries surface")
    unknown = sorted(set(resolved["sections"]) - set(SECTION_IDS))
    if unknown:
        raise EvaluationError(f"policy has unsupported sections: {unknown}")
    effective = {key: resolved[key] for key in DEFAULT_POLICY}
    return resolved, effective


def _verify_file(path: Path, expected_sha256: str, expected_bytes: int, *, context: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"cannot read frozen {context} {path}: {exc}") from exc
    if len(data) != expected_bytes:
        raise EvaluationError(
            f"frozen {context} byte count changed: expected {expected_bytes}, got {len(data)}"
        )
    actual = _sha256_bytes(data)
    if actual != expected_sha256:
        raise EvaluationError(
            f"frozen {context} sha256 changed: expected {expected_sha256}, got {actual}"
        )
    return data


def _verify_corpus() -> tuple[dict[str, Any], Path, dict[tuple[str, int, int], int]]:
    manifest = _load_json(DATASET_MANIFEST_PATH, context="public replay manifest")
    corpus = manifest["corpus"]
    prereg = manifest["preregistration"]
    corpus_root = REPO_ROOT / str(corpus["relative_root"])
    report_path = corpus_root / "report.json"
    report_data = _verify_file(
        report_path,
        str(corpus["report_sha256"]),
        int(corpus["report_bytes"]),
        context="Packet 2 report",
    )
    prereg_path = REPO_ROOT / str(prereg["relative_path"])
    prereg_data = _verify_file(
        prereg_path,
        str(prereg["sha256"]),
        int(prereg["bytes"]),
        context="Packet 2 preregistration",
    )
    declared_prereg = _sha256_bytes(prereg_data.rstrip(b"\n"))
    if declared_prereg != prereg["declared_design_sha256"]:
        raise EvaluationError("Packet 2 preregistration design digest no longer matches the manifest")

    trajectory_paths: list[Path] = []
    for chain_index in corpus["chains"]:
        trajectory_paths.extend(
            sorted(
                (corpus_root / "brain" / f"chain-{int(chain_index):02d}" / "trajectories").glob(
                    "episode-*.jsonl"
                )
            )
        )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(trajectory_paths, key=lambda item: item.relative_to(corpus_root).as_posix()):
        relative = path.relative_to(corpus_root).as_posix()
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise EvaluationError(
                f"cannot read frozen trajectory {path}: {exc}"
            ) from exc
        total_bytes += len(data)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    if len(trajectory_paths) != int(corpus["brain_trajectory_file_count"]):
        raise EvaluationError("frozen trajectory file count changed")
    if total_bytes != int(corpus["brain_trajectory_bytes"]):
        raise EvaluationError("frozen trajectory byte count changed")
    if digest.hexdigest() != corpus["brain_trajectory_tree_sha256"]:
        raise EvaluationError("frozen trajectory tree digest changed")

    report = json.loads(report_data)
    if report.get("prereg_sha256") != prereg["declared_design_sha256"]:
        raise EvaluationError("Packet 2 report is not bound to the frozen preregistration")
    steelman_tokens: dict[tuple[str, int, int], int] = {}
    for row in report.get("episodes", []):
        if not isinstance(row, dict):
            raise EvaluationError("Packet 2 report contains a non-object episode")
        key = (str(row.get("arm")), int(row.get("chain_index", -1)), int(row.get("episode_index", -1)))
        if key in steelman_tokens:
            raise EvaluationError(f"Packet 2 report contains duplicate episode identity {key!r}")
        steelman_tokens[key] = int(row.get("context_tokens_approx", -1))
    return manifest, corpus_root, steelman_tokens


def _entry_line(entry: Any) -> str:
    return f"[memory:entry:{entry.stratum}] {entry.content}"


def _ordered_entries(recalled: tuple[Any, ...], policy: dict[str, Any]) -> tuple[Any, ...]:
    entries = list(recalled)
    order = policy["recalled_entry_order"]
    if order == "strongest_first":
        entries.sort(key=lambda item: (-float(item.strength), -int(item.last_accessed_ms), str(item.entry_id)))
    elif order == "failures_first":
        entries.sort(
            key=lambda item: (
                "outcome:fail" not in item.tags,
                -float(item.strength),
                -int(item.last_accessed_ms),
                str(item.entry_id),
            )
        )
    elif order == "recent_first":
        entries.sort(key=lambda item: (-int(item.created_at_ms), str(item.entry_id)))
    elif order != "owner_order":
        raise EvaluationError(f"unsupported recalled_entry_order: {order!r}")
    return tuple(entries[: int(policy["max_recalled_entries"])])


def _section_lines(
    observer: CodingLabChainObserver,
    recalled: tuple[Any, ...],
    policy: dict[str, Any],
) -> tuple[
    list[tuple[str, str]],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    selected_entries = _ordered_entries(recalled, policy)
    selected_entry_lines = tuple(_entry_line(entry) for entry in selected_entries)
    selected_failed_lines = tuple(
        _entry_line(entry) for entry in selected_entries if "outcome:fail" in entry.tags
    )
    available_entry_lines = tuple(_entry_line(entry) for entry in recalled)
    available_failed_lines = tuple(
        _entry_line(entry) for entry in recalled if "outcome:fail" in entry.tags
    )

    snapshots = observer.session.runner.upstream_snapshots
    groups: dict[str, list[str]] = {section: [] for section in SECTION_IDS}
    groups["recalled_entries"] = list(selected_entry_lines)
    memory_snapshot = snapshots.get("memory")
    if memory_snapshot is not None:
        value = memory_snapshot.value
        if str(value.description):
            groups["memory_description"].append(f"[memory] {value.description}")
        if str(value.durable_summary):
            groups["memory_durable_summary"].append(
                f"[memory:durable] {value.durable_summary}"
            )
        if str(value.episodic_summary):
            groups["memory_episodic_summary"].append(
                f"[memory:episodic] {value.episodic_summary}"
            )
    for slot in (
        "prediction_error",
        "plan_intent",
        "open_loop",
        "belief_assumption",
        "execution_result",
        "user_model",
    ):
        published = snapshots.get(slot)
        if published is None:
            continue
        description = str(published.value.description)
        if description:
            groups[slot].append(f"[{slot}] {description}")

    lines: list[tuple[str, str]] = []
    for section in policy["sections"]:
        lines.extend((section, line) for line in groups[section])
    return (
        lines,
        available_entry_lines,
        available_failed_lines,
        selected_entry_lines,
        selected_failed_lines,
    )


def _render_pack(
    observer: CodingLabChainObserver,
    recalled: tuple[Any, ...],
    policy: dict[str, Any],
) -> RenderedPack:
    (
        lines,
        available_entry_lines,
        available_failed_lines,
        selected_entry_lines,
        selected_failed_lines,
    ) = _section_lines(observer, recalled, policy)
    if policy["deduplicate_exact_lines"]:
        seen: set[str] = set()
        deduplicated: list[tuple[str, str]] = []
        for section, line in lines:
            if line in seen:
                continue
            seen.add(line)
            deduplicated.append((section, line))
        lines = deduplicated

    generic_fraction = float(policy["generic_section_budget_fraction"])
    if generic_fraction < 1.0:
        generic_cap = math.floor(int(policy["max_context_chars"]) * generic_fraction)
        generic_used = 0
        capped: list[tuple[str, str]] = []
        for section, line in lines:
            if section == "recalled_entries":
                capped.append((section, line))
                continue
            line_cost = len(line) + (1 if generic_used else 0)
            if generic_used + line_cost <= generic_cap:
                capped.append((section, line))
                generic_used += line_cost
        lines = capped

    budget = int(policy["max_context_chars"])
    strategy = str(policy["truncation_strategy"])
    raw = "\n".join(line for _, line in lines)
    if len(raw) <= budget:
        rendered = raw
    elif strategy == "legacy_tail_marker":
        rendered = raw[:budget] + TRUNCATION_MARKER
    elif strategy == "bounded_tail":
        keep = max(0, budget - len(TRUNCATION_MARKER))
        rendered = raw[:keep] + TRUNCATION_MARKER
    elif strategy == "whole_line":
        kept: list[str] = []
        for _, line in lines:
            candidate = "\n".join((*kept, line))
            if len(candidate) <= budget:
                kept.append(line)
        rendered = "\n".join(kept)
    else:
        raise EvaluationError(f"unsupported truncation_strategy: {strategy!r}")

    rendered_full_lines = set(rendered.splitlines())
    retained = tuple(line for line in selected_entry_lines if line in rendered_full_lines)
    retained_failed = tuple(
        line for line in selected_failed_lines if line in rendered_full_lines
    )
    return RenderedPack(
        text=rendered,
        available_entry_lines=available_entry_lines,
        available_failed_entry_lines=available_failed_lines,
        selected_entry_lines=selected_entry_lines,
        selected_failed_entry_lines=selected_failed_lines,
        retained_entry_lines=retained,
        retained_failed_entry_lines=retained_failed,
        strict_budget_passed=len(rendered) <= budget,
    )


def _task_from_trajectory(
    expected: ChainTask,
    trajectory_path: Path,
) -> None:
    events = read_trajectory(trajectory_path)
    presented = next(
        (event for event in events if event["event_type"] == "task_presented"),
        None,
    )
    if presented is None:
        raise EvaluationError(f"trajectory has no task_presented event: {trajectory_path}")
    payload = presented["payload"]
    if payload.get("task_id") != expected.task_id or payload.get("category") != expected.category:
        raise EvaluationError(
            f"trajectory/task generator mismatch at {trajectory_path}: "
            f"expected {expected.task_id}/{expected.category}, got "
            f"{payload.get('task_id')}/{payload.get('category')}"
        )


async def _replay(
    *,
    chains: tuple[int, ...],
    corpus_root: Path,
    steelman_tokens: dict[tuple[str, int, int], int],
    policy: dict[str, Any],
    state_root: Path,
) -> tuple[ReplayRow, ...]:
    rows: list[ReplayRow] = []
    for chain_index in chains:
        spec = EnvSpec(
            env_seed=ENV_SEED + chain_index * 13,
            convention_ids=CONVENTION_IDS,
        )
        tasks = generate_task_chain(
            spec,
            chain_seed=chain_index,
            length=EPISODES_PER_CHAIN,
        )
        observer = CodingLabChainObserver(
            chain_id=f"praxist-replay-{chain_index:02d}",
            brain_state_root=state_root / f"chain-{chain_index:02d}",
        )
        for episode_index, task in enumerate(tasks):
            trajectory_path = (
                corpus_root
                / "brain"
                / f"chain-{chain_index:02d}"
                / "trajectories"
                / f"episode-{episode_index:03d}.jsonl"
            )
            _task_from_trajectory(task, trajectory_path)
            recalled = recall_for_task(observer, task=task)
            pack = _render_pack(observer, recalled, policy)
            steelman_key = ("steelman", chain_index, episode_index)
            if steelman_key not in steelman_tokens:
                raise EvaluationError(f"missing frozen steelman row {steelman_key!r}")
            rows.append(
                ReplayRow(
                    chain_index=chain_index,
                    episode_index=episode_index,
                    context_chars=len(pack.text),
                    context_tokens=approx_tokens(pack.text),
                    steelman_context_tokens=steelman_tokens[steelman_key],
                    available_entries=len(pack.available_entry_lines),
                    selected_entries=len(pack.selected_entry_lines),
                    retained_entries=len(pack.retained_entry_lines),
                    available_failed_entries=len(pack.available_failed_entry_lines),
                    selected_failed_entries=len(pack.selected_failed_entry_lines),
                    retained_failed_entries=len(pack.retained_failed_entry_lines),
                    strict_budget_passed=pack.strict_budget_passed,
                )
            )
            await observer.observe_episode(
                episode_index=episode_index,
                trajectory_path=trajectory_path,
            )
    return tuple(rows)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def _metrics(rows: tuple[ReplayRow, ...], *, elapsed: float, complete: bool) -> dict[str, Any]:
    measured = tuple(row for row in rows if row.episode_index > 0)
    if not measured:
        raise EvaluationError("replay produced no post-first-episode contexts")
    candidate_tokens = sum(row.context_tokens for row in measured)
    baseline_tokens = sum(row.steelman_context_tokens for row in measured)
    if baseline_tokens <= 0:
        raise EvaluationError("frozen steelman context token total is not positive")
    context_ratio = candidate_tokens / baseline_tokens
    chain_ratios: list[float] = []
    for chain_index in sorted({row.chain_index for row in measured}):
        chain_rows = tuple(row for row in measured if row.chain_index == chain_index)
        denominator = sum(row.steelman_context_tokens for row in chain_rows)
        if denominator <= 0:
            raise EvaluationError(f"chain {chain_index} has no positive steelman context")
        chain_ratios.append(sum(row.context_tokens for row in chain_rows) / denominator)
    available = sum(row.available_entries for row in measured)
    selected = sum(row.selected_entries for row in measured)
    retained = sum(row.retained_entries for row in measured)
    failed_available = sum(row.available_failed_entries for row in measured)
    failed_selected = sum(row.selected_failed_entries for row in measured)
    failed_retained = sum(row.retained_failed_entries for row in measured)
    recall_selection_coverage = _ratio(selected, available)
    failed_selection_coverage = _ratio(failed_selected, failed_available)
    recall_retention = _ratio(retained, selected)
    failed_retention = _ratio(failed_retained, failed_selected)
    strict_pass_rate = statistics.fmean(1.0 if row.strict_budget_passed else 0.0 for row in measured)
    scaling_margin = MAX_TOKEN_RATIO - context_ratio
    worst_margin = MAX_TOKEN_RATIO - max(chain_ratios)
    selection_gate = (
        recall_selection_coverage >= MIN_RECALLED_SELECTION_COVERAGE
        and failed_selection_coverage >= MIN_FAILED_SELECTION_COVERAGE
    )
    render_retention_gate = recall_retention == 1.0 and failed_retention == 1.0
    retention_gate = selection_gate and render_retention_gate
    scaling_gate = context_ratio <= MAX_TOKEN_RATIO
    strict_gate = strict_pass_rate == 1.0
    promotion_eligible = complete and retention_gate and scaling_gate and strict_gate
    return {
        "scaling_margin": scaling_margin,
        "context_token_ratio": context_ratio,
        "worst_chain_scaling_margin": worst_margin,
        "recalled_entry_selection_coverage": recall_selection_coverage,
        "failed_entry_selection_coverage": failed_selection_coverage,
        "recalled_entry_retention": recall_retention,
        "failed_entry_retention": failed_retention,
        "strict_budget_pass_rate": strict_pass_rate,
        "mean_context_tokens": statistics.fmean(row.context_tokens for row in measured),
        "candidate_context_tokens": candidate_tokens,
        "steelman_context_tokens": baseline_tokens,
        "recalled_entries_available": available,
        "recalled_entries_selected": selected,
        "recalled_entries_retained": retained,
        "failed_entries_available": failed_available,
        "failed_entries_selected": failed_selected,
        "failed_entries_retained": failed_retained,
        "evaluation_units": len({row.chain_index for row in rows}),
        "evaluated_contexts": len(measured),
        "evaluator_wall_seconds": elapsed,
        "protocol_integrity_passed": True,
        "protocol_integrity_failed": False,
        "selection_gate_passed": selection_gate,
        "render_retention_gate_passed": render_retention_gate,
        "retention_gate_passed": retention_gate,
        "scaling_gate_passed": scaling_gate,
        "strict_budget_passed": strict_gate,
        "scored_complete": complete,
        "promotion_eligible": promotion_eligible,
        "is_smoke_eval": not complete,
        "partial": not complete,
        "scout_only": not complete,
        "validation_only": not complete,
        "validation_only_result": not complete,
        "suspect_protocol": False,
        "suspect_leakage": False,
        "late_after_generation_boundary": False,
    }


def _producer_identity() -> dict[str, Any]:
    result: dict[str, Any] = {}
    generation = os.environ.get("PRAXIST_GENERATION_ID", "").strip()
    peer = os.environ.get("PRAXIST_PEER_ID", "").strip()
    if generation:
        result["generation_id"] = int(generation)
    if peer:
        result["peer_id"] = peer
    return result


def _design_dimensions(policy: dict[str, Any]) -> dict[str, str]:
    return {
        "mechanism_family": str(policy["truncation_strategy"]),
        "intervention_surface": "declarative_context_composition",
        "intent": "optimize",
        "semantic_family": f"sections_{len(policy['sections'])}",
        "parent_lineage": "packet2_v3_baseline",
        "novelty_axis": str(policy["recalled_entry_order"]),
    }


def _failure_modes(metrics: dict[str, Any]) -> list[str]:
    failures = []
    for field, label in (
        ("selection_gate_passed", "recalled_evidence_selection_floor_failed"),
        ("render_retention_gate_passed", "selected_evidence_truncated"),
        ("scaling_gate_passed", "scaling_gate_failed"),
        ("strict_budget_passed", "strict_budget_overrun"),
    ):
        if not metrics[field]:
            failures.append(label)
    return failures


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "evaluation_summary.json"
    temporary = output_dir / ".evaluation_summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _build_summary(
    *,
    mode: str,
    policy: dict[str, Any],
    effective_policy: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    replication_of_effective_config_sha256: str = "",
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical evaluator result contract from measured metrics."""

    complete = mode == "complete"
    chains = MODE_CHAINS[mode]
    effort_ratio = len(chains) / REFERENCE_UNITS
    coverage_ratio = effort_ratio
    effective_config = {
        "evaluator_version": EVALUATOR_VERSION,
        "mode": mode,
        "policy": effective_policy,
        "public_replay_tree_sha256": manifest["corpus"]["brain_trajectory_tree_sha256"],
        "reference_units": REFERENCE_UNITS,
        "selected_chains": list(chains),
    }
    effective_digest = _canonical_digest(effective_config)
    replication_status = "not_requested"
    if replication_of_effective_config_sha256:
        replication_status = (
            "matched"
            if replication_of_effective_config_sha256 == effective_digest
            else "mismatched"
        )
    failures = _failure_modes(metrics)
    source_lane = "performance" if complete else "task_candidate"
    if complete and metrics["promotion_eligible"]:
        valence = "positive"
    elif complete and metrics["retention_gate_passed"]:
        valence = "mixed"
    elif complete:
        valence = "negative"
    else:
        valence = "neutral"
    producer_identity = dict(producer or {})
    return {
        "schema_version": 1,
        "protocol": EVALUATOR_VERSION,
        "protocol_version": 1,
        "variant_id": policy["variant_id"],
        "variant_name": policy["variant_id"],
        **producer_identity,
        "score": metrics["scaling_margin"],
        "metrics": metrics,
        "frontier_lane": source_lane,
        "promotion_lane": source_lane,
        "evidence_stage": mode,
        "eval_stage": mode,
        "tier": mode,
        "result_status": "scored_complete" if complete else "preliminary",
        "scored_complete": complete,
        "complete_eval": complete,
        "promotion_eligible": metrics["promotion_eligible"],
        "parent_authorized": complete and metrics["retention_gate_passed"],
        "close_eligible": complete,
        "effort_ratio": effort_ratio,
        "coverage_ratio": coverage_ratio,
        "actual_evaluation_units": len(chains),
        "reference_evaluation_units": REFERENCE_UNITS,
        "evaluation_units_completed": len(chains),
        "evaluation_units_required": REFERENCE_UNITS,
        "effective_config": effective_config,
        "effective_config_complete": True,
        "effective_config_digest": effective_digest,
        "effective_config_schema": "coding-memory-context-replay-effective-config.v1",
        "replication_of_effective_config_sha256": (
            replication_of_effective_config_sha256
        ),
        "replication_effective_config_status": replication_status,
        "design_dimensions": _design_dimensions(policy),
        "changed_modules": ["policy.json"],
        "method_class": "declarative_memory_inheritance_policy",
        "protocol_integrity": {
            "passed": True,
            "frozen_assets_attested": True,
            "outcome_dependent_selection": False,
            "candidate_contains_executable_code": False,
            "public_replay_only": True,
        },
        "retention_contract": {
            "min_recalled_selection_coverage": MIN_RECALLED_SELECTION_COVERAGE,
            "min_failed_selection_coverage": MIN_FAILED_SELECTION_COVERAGE,
            "selected_entries_must_be_full_lines": True,
        },
        "dataset": {
            "visibility": manifest["visibility"],
            "source_base_revision": manifest["source_base_revision"],
            "selected_chains": list(chains),
            "episodes_per_chain": EPISODES_PER_CHAIN,
            "trajectory_tree_sha256": manifest["corpus"][
                "brain_trajectory_tree_sha256"
            ],
            "formal_validation_overlap": True,
        },
        "extra": {
            "frontier_lane": source_lane,
            "promotion_lane": source_lane,
            "evidence_stage": mode,
            "protocol_name": EVALUATOR_VERSION,
            "effort_ratio": effort_ratio,
            "coverage_ratio": coverage_ratio,
            "completed_required_eval_units": len(chains),
            "complete_protocol_evaluation_units": REFERENCE_UNITS,
            "is_negative": valence == "negative",
            "evidence_valence": valence,
            "failure_mode": ",".join(failures),
            "disconfirming_claim_ids": ["claim:coding-memory-scaling"]
            if failures
            else [],
            "next_step_intent": "sealed_formal_validation"
            if metrics["promotion_eligible"]
            else "repair_or_complete_validation",
            "public_replay_is_not_formal_quality_evidence": True,
            "design_dimensions": _design_dimensions(policy),
            "effective_config_digest": effective_digest,
            **producer_identity,
        },
    }


def evaluate(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    variant_dir = args.variant_dir.expanduser().resolve(strict=True)
    if not variant_dir.is_dir():
        raise EvaluationError(f"variant path is not a directory: {variant_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy, effective_policy = _load_policy(variant_dir)
    manifest, corpus_root, steelman_tokens = _verify_corpus()
    chains = MODE_CHAINS[args.mode]
    with tempfile.TemporaryDirectory(prefix="coding-memory-replay-", dir=args.output_dir) as state_dir:
        rows = asyncio.run(
            _replay(
                chains=chains,
                corpus_root=corpus_root,
                steelman_tokens=steelman_tokens,
                policy=policy,
                state_root=Path(state_dir),
            )
        )
    elapsed = time.perf_counter() - started
    complete = args.mode == "complete"
    metrics = _metrics(rows, elapsed=elapsed, complete=complete)
    summary = _build_summary(
        mode=args.mode,
        policy=policy,
        effective_policy=effective_policy,
        manifest=manifest,
        metrics=metrics,
        replication_of_effective_config_sha256=(
            args.replication_of_effective_config_sha256
        ),
        producer=_producer_identity(),
    )
    return _write_summary(args.output_dir, summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one structured coding-memory inheritance policy."
    )
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE_CHAINS), default="preliminary")
    parser.add_argument(
        "--replication-of-effective-config-sha256",
        default="",
        help="Optional selected-parent effective-config digest for an exact replication claim.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    try:
        summary_path = evaluate(args)
    except (EvaluationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"summary": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
