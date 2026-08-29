"""Typed ResearchOpportunity discovery and approval-gated submission.

The scanner consumes only validated Forge failure-pattern records.  Routing is
an exact registry lookup over protocol fields; causal prose is copied as
evidence but never used to select a task, model, or launch profile.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .config import ForgeConfig
from .foundation import (
    ForgeError,
    SchemaStore,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_text,
    utc_now,
)
from .research_control import (
    inspect_research_request,
    submit_research_request,
    validate_research_request,
)
from .research_promotion import validate_research_task

SCHEMA_NAME = "research_opportunity.schema.json"

_REGISTRY_VERSION = "forge-research-task-registry.v1"
_OPPORTUNITY_VERSION = "forge-research-opportunity.v1"
_ROUTING_VERSION = "forge-research-opportunity-routing.v1"
_PATTERN_VERSION = "forge-failure-pattern.v3"
_SCANNER_ID = "forge:research-opportunity-scanner.v1"
_OCCURRENCE_CAP = 1000


class ResearchOpportunityError(ForgeError):
    """Raised when typed discovery, registry routing, or immutable state is unsafe."""


@dataclass(frozen=True)
class ResearchOpportunityStatus:
    opportunity_id: str
    pattern_id: str
    priority_score: int
    state: str
    blocker_codes: tuple[str, ...]
    opportunity_path: Path
    routing_path: Path
    mapping_id: str | None
    request_id: str | None
    request_path: Path | None


@dataclass(frozen=True)
class ResearchScanResult:
    failure_patterns_path: Path
    registry_path: Path
    statuses: tuple[ResearchOpportunityStatus, ...]
    discovered_count: int
    new_request_count: int


@dataclass(frozen=True)
class _PatternRecord:
    line_number: int
    value: dict[str, Any]
    sha256: str


@dataclass(frozen=True)
class _ResolvedBinding:
    mapping_id: str
    binding_sha256: str
    task_id: str
    owner: str
    capability_axes: tuple[str, ...]
    task_manifest: Path
    task_project: Path
    praxist_executable: Path
    run_dir: Path
    config_file: Path | None
    agent_system: str | None
    runtime: str | None
    codex_native: bool
    model_provider: str | None
    model: str
    strategy: str
    cohort: int | None
    generations: int | None
    startup_timeout_seconds: int


def scan_research_opportunities(
    *,
    config: ForgeConfig,
    failure_patterns_path: Path,
    registry_path: Path | None = None,
) -> ResearchScanResult:
    """Run one bounded typed scan and submit at most the registry policy limit."""

    patterns_path = _resolve_regular_file(
        failure_patterns_path,
        context="Forge failure-pattern JSONL",
    )
    selected_registry = registry_path or config.paths.forge_root / "research_task_registry.yaml"
    registry, resolved_registry = _load_registry(config, selected_registry)
    registry_ref = _content_ref(config, resolved_registry, context="research task registry")
    pattern_records = _load_failure_patterns(config, patterns_path)
    source_ref = _content_ref(config, patterns_path, context="Forge failure-pattern JSONL")
    scan_lock = config.paths.artifacts_root / "research_opportunities" / ".scan.lock"

    with _exclusive_lock(scan_lock):
        opportunities = tuple(
            _seal_opportunity(
                config=config,
                source_ref=source_ref,
                record=record,
            )
            for record in pattern_records
        )
        ordered = sorted(
            opportunities,
            key=lambda item: (-int(item[0]["priority"]["score"]), str(item[0]["opportunity_id"])),
        )
        remaining = int(registry["policy"]["max_new_requests_per_scan"])
        statuses: list[ResearchOpportunityStatus] = []
        new_requests = 0
        for opportunity, opportunity_path, opportunity_sha256 in ordered:
            pattern = opportunity["source"]["record"]
            nomination_blockers = tuple(str(value) for value in opportunity["nomination"]["blocker_codes"])
            if nomination_blockers:
                routing_path = _seal_routing_receipt(
                    config=config,
                    opportunity=opportunity,
                    opportunity_path=opportunity_path,
                    opportunity_sha256=opportunity_sha256,
                    registry_ref=registry_ref,
                    mapping=None,
                    decision="NEEDS_TASK_DESIGN",
                    blocker_codes=nomination_blockers,
                    request=None,
                )
                statuses.append(
                    _status(
                        opportunity=opportunity,
                        opportunity_path=opportunity_path,
                        routing_path=routing_path,
                        state="NEEDS_TASK_DESIGN",
                        blocker_codes=nomination_blockers,
                    )
                )
                continue

            mapping = _match_mapping(registry, pattern)
            if mapping is None:
                blockers = ("NO_REGISTERED_TASK",)
                routing_path = _seal_routing_receipt(
                    config=config,
                    opportunity=opportunity,
                    opportunity_path=opportunity_path,
                    opportunity_sha256=opportunity_sha256,
                    registry_ref=registry_ref,
                    mapping=None,
                    decision="NEEDS_TASK_DESIGN",
                    blocker_codes=blockers,
                    request=None,
                )
                statuses.append(
                    _status(
                        opportunity=opportunity,
                        opportunity_path=opportunity_path,
                        routing_path=routing_path,
                        state="NEEDS_TASK_DESIGN",
                        blocker_codes=blockers,
                    )
                )
                continue

            binding = _resolve_binding(
                config=config,
                registry_path=resolved_registry,
                opportunity=opportunity,
                mapping=mapping,
            )
            existing = _find_submitted_request(
                config=config,
                opportunity=opportunity,
                opportunity_path=opportunity_path,
                opportunity_sha256=opportunity_sha256,
                binding=binding,
            )
            if existing is not None:
                request_id, request_path = existing
                routing_path = _seal_routing_receipt(
                    config=config,
                    opportunity=opportunity,
                    opportunity_path=opportunity_path,
                    opportunity_sha256=opportunity_sha256,
                    registry_ref=registry_ref,
                    mapping=binding,
                    decision="SUBMITTED_FOR_A0",
                    blocker_codes=(),
                    request=(request_id, request_path),
                )
                request_state = inspect_research_request(
                    config=config,
                    request_path=request_path,
                ).state
                statuses.append(
                    _status(
                        opportunity=opportunity,
                        opportunity_path=opportunity_path,
                        routing_path=routing_path,
                        state=request_state,
                        blocker_codes=(),
                        binding=binding,
                        request=(request_id, request_path),
                    )
                )
                continue

            if remaining == 0:
                blockers = ("SCAN_SUBMISSION_LIMIT_REACHED",)
                routing_path = _seal_routing_receipt(
                    config=config,
                    opportunity=opportunity,
                    opportunity_path=opportunity_path,
                    opportunity_sha256=opportunity_sha256,
                    registry_ref=registry_ref,
                    mapping=binding,
                    decision="DEFERRED_BY_SCAN_LIMIT",
                    blocker_codes=blockers,
                    request=None,
                )
                statuses.append(
                    _status(
                        opportunity=opportunity,
                        opportunity_path=opportunity_path,
                        routing_path=routing_path,
                        state="DEFERRED_BY_SCAN_LIMIT",
                        blocker_codes=blockers,
                        binding=binding,
                    )
                )
                continue

            request = submit_research_request(
                config=config,
                task_manifest_path=binding.task_manifest,
                task_project_path=binding.task_project,
                praxist_executable=binding.praxist_executable,
                run_dir=binding.run_dir,
                requested_by=_SCANNER_ID,
                reason=_submission_reason(opportunity, binding),
                trigger_kind="forge_failure_pattern",
                evidence_paths=(opportunity_path,),
                config_file=binding.config_file,
                agent_system=binding.agent_system,
                runtime=binding.runtime,
                codex_native=binding.codex_native,
                model_provider=binding.model_provider,
                model=binding.model,
                strategy=binding.strategy,
                cohort=binding.cohort,
                generations=binding.generations,
                startup_timeout_seconds=binding.startup_timeout_seconds,
            )
            remaining -= 1
            new_requests += 1
            routing_path = _seal_routing_receipt(
                config=config,
                opportunity=opportunity,
                opportunity_path=opportunity_path,
                opportunity_sha256=opportunity_sha256,
                registry_ref=registry_ref,
                mapping=binding,
                decision="SUBMITTED_FOR_A0",
                blocker_codes=(),
                request=(request.request_id, request.request_path),
            )
            statuses.append(
                _status(
                    opportunity=opportunity,
                    opportunity_path=opportunity_path,
                    routing_path=routing_path,
                    state="AWAITING_RESEARCH_APPROVAL",
                    blocker_codes=(),
                    binding=binding,
                    request=(request.request_id, request.request_path),
                )
            )

    return ResearchScanResult(
        failure_patterns_path=patterns_path,
        registry_path=resolved_registry,
        statuses=tuple(statuses),
        discovered_count=len(opportunities),
        new_request_count=new_requests,
    )


def validate_research_opportunity(
    *,
    config: ForgeConfig,
    opportunity_path: Path,
) -> dict[str, Any]:
    """Validate one immutable ResearchOpportunity and its canonical location."""

    opportunity, _, _ = _load_opportunity(config, opportunity_path)
    return opportunity


def _load_failure_patterns(
    config: ForgeConfig,
    path: Path,
) -> tuple[_PatternRecord, ...]:
    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ResearchOpportunityError(f"cannot read Forge failure-pattern JSONL {path}: {exc}") from exc
    records: list[_PatternRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResearchOpportunityError(f"invalid failure-pattern JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ResearchOpportunityError(f"failure-pattern record at {path}:{line_number} must be an object")
        schema_store.validate(raw, "failure_pattern.schema.json")
        if raw.get("schema_version") != _PATTERN_VERSION:
            raise ResearchOpportunityError(
                f"research scan requires {_PATTERN_VERSION}, got {raw.get('schema_version')!r}"
            )
        _validate_pattern_identity(raw)
        pattern_id = str(raw["pattern_id"])
        if pattern_id in seen_ids:
            raise ResearchOpportunityError(f"duplicate failure pattern_id in scan input: {pattern_id}")
        seen_ids.add(pattern_id)
        record_sha256 = sha256_text(canonical_json(raw))
        records.append(
            _PatternRecord(
                line_number=line_number,
                value=dict(raw),
                sha256=record_sha256,
            )
        )
    return tuple(records)


def _validate_pattern_identity(pattern: Mapping[str, Any]) -> None:
    identity_payload = {
        "verifier_cause": pattern["verifier_cause"],
        "agent_behavior_cause": pattern["agent_behavior_cause"],
        "exposed_mechanism": pattern["exposed_mechanism"],
        "evidence_refs": pattern["evidence_refs"],
    }
    expected = "fp_" + sha256_text(canonical_json(identity_payload))[:16]
    if pattern["pattern_id"] != expected:
        raise ResearchOpportunityError("failure pattern_id does not match its canonical causal/evidence payload")


def _seal_opportunity(
    *,
    config: ForgeConfig,
    source_ref: dict[str, str],
    record: _PatternRecord,
) -> tuple[dict[str, Any], Path, str]:
    key_payload = {
        "source_kind": "forge_failure_pattern",
        "source_schema_version": record.value["schema_version"],
        "pattern_id": record.value["pattern_id"],
        "record_sha256": record.sha256,
    }
    opportunity_key = f"research-opportunity-key:{sha256_text(canonical_json(key_payload))}"
    existing = _find_existing_opportunity(config, opportunity_key, record.sha256)
    if existing is not None:
        return existing

    blockers = _nomination_blockers(record.value)
    payload: dict[str, Any] = {
        "schema_version": _OPPORTUNITY_VERSION,
        "opportunity_key": opportunity_key,
        "source": {
            "kind": "forge_failure_pattern",
            "artifact": source_ref,
            "line_number": record.line_number,
            "record_sha256": record.sha256,
            "record": record.value,
        },
        "nomination": {
            "readiness": "NEEDS_TASK_DESIGN" if blockers else "ROUTABLE",
            "editable_component": record.value["editable_component"],
            "editable_target": record.value["editable_target"],
            "blocker_codes": list(blockers),
        },
        "priority": {
            "policy": "typed_occurrence_count_capped.v1",
            "occurrence_count": record.value["occurrence_count"],
            "occurrence_cap": _OCCURRENCE_CAP,
            "score": min(int(record.value["occurrence_count"]), _OCCURRENCE_CAP),
        },
        "authority": {
            "nomination_only": True,
            "research_start_authorized": False,
            "formal_validation_performed": False,
            "production_promotion_authorized": False,
            "runtime_wiring_changed": False,
            "evaluation_is_learning_source": False,
        },
        "created_at": utc_now(),
    }
    payload["opportunity_id"] = _artifact_id(
        "research-opportunity",
        payload,
        "opportunity_id",
    )
    _validate_schema(config, payload, _OPPORTUNITY_VERSION)
    key_digest = opportunity_key.partition(":")[2]
    opportunity_digest = str(payload["opportunity_id"]).partition(":")[2]
    destination = (
        config.paths.artifacts_root / "research_opportunities" / key_digest / opportunity_digest / "opportunity.json"
    )
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=payload,
        expected_version=_OPPORTUNITY_VERSION,
        identity_field="opportunity_id",
    )
    return payload, destination, _sha256_file(destination, context="ResearchOpportunity")


def _find_existing_opportunity(
    config: ForgeConfig,
    opportunity_key: str,
    record_sha256: str,
) -> tuple[dict[str, Any], Path, str] | None:
    key_digest = opportunity_key.partition(":")[2]
    paths = tuple(
        sorted((config.paths.artifacts_root / "research_opportunities" / key_digest).glob("*/opportunity.json"))
    )
    matches: list[tuple[dict[str, Any], Path, str]] = []
    for path in paths:
        opportunity, resolved, digest = _load_opportunity(config, path)
        if opportunity["opportunity_key"] != opportunity_key:
            raise ResearchOpportunityError("ResearchOpportunity key directory contains another key")
        if opportunity["source"]["record_sha256"] == record_sha256:
            matches.append((opportunity, resolved, digest))
    if len(matches) > 1:
        raise ResearchOpportunityError(f"multiple immutable ResearchOpportunity artifacts claim key {opportunity_key}")
    return matches[0] if matches else None


def _nomination_blockers(pattern: Mapping[str, Any]) -> tuple[str, ...]:
    blockers: list[str] = []
    if pattern["surface_status"] != "in-surface":
        blockers.append("OUT_OF_EDITABLE_SURFACE")
    if pattern["editable_component"] is None:
        blockers.append("MISSING_EDITABLE_COMPONENT")
    if pattern["editable_target"] is None:
        blockers.append("MISSING_EDITABLE_TARGET")
    return tuple(blockers)


def _load_registry(
    config: ForgeConfig,
    registry_path: Path,
) -> tuple[dict[str, Any], Path]:
    resolved = _resolve_regular_file(registry_path, context="research task registry")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ResearchOpportunityError(f"invalid research task registry YAML {resolved}: {exc}") from exc
    except OSError as exc:
        raise ResearchOpportunityError(f"cannot read research task registry {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResearchOpportunityError("research task registry must contain a mapping")
    SchemaStore(config.paths.forge_root / "schemas").validate(raw, SCHEMA_NAME)
    if raw.get("schema_version") != _REGISTRY_VERSION:
        raise ResearchOpportunityError(f"unsupported research task registry: {raw.get('schema_version')!r}")
    _validate_registry_uniqueness(raw)
    return raw, resolved


def _validate_registry_uniqueness(registry: Mapping[str, Any]) -> None:
    mapping_ids: set[str] = set()
    matches: set[tuple[str, str | None]] = set()
    component_targets: dict[str, set[str | None]] = {}
    for mapping in registry["mappings"]:
        mapping_id = str(mapping["mapping_id"])
        if mapping_id in mapping_ids:
            raise ResearchOpportunityError(f"duplicate research mapping_id: {mapping_id}")
        mapping_ids.add(mapping_id)
        match = mapping["match"]
        key = (str(match["editable_component"]), match["editable_target"])
        if key in matches:
            raise ResearchOpportunityError(f"duplicate research mapping match: {key!r}")
        matches.add(key)
        component_targets.setdefault(key[0], set()).add(key[1])
    for component, targets in component_targets.items():
        if None in targets and len(targets) > 1:
            raise ResearchOpportunityError(f"component-wide and target-specific mappings overlap for {component!r}")


def _match_mapping(
    registry: Mapping[str, Any],
    pattern: Mapping[str, Any],
) -> dict[str, Any] | None:
    component = pattern["editable_component"]
    target = pattern["editable_target"]
    candidates = [
        mapping
        for mapping in registry["mappings"]
        if mapping["match"]["editable_component"] == component and mapping["match"]["editable_target"] in {None, target}
    ]
    if len(candidates) > 1:
        raise ResearchOpportunityError(
            f"ambiguous exact research-task mapping for component={component!r}, target={target!r}"
        )
    return dict(candidates[0]) if candidates else None


def _resolve_binding(
    *,
    config: ForgeConfig,
    registry_path: Path,
    opportunity: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> _ResolvedBinding:
    base = registry_path.parent
    task_manifest = _resolve_registry_file(base, mapping["task_manifest"], "research task manifest")
    task_project = _resolve_registry_directory(base, mapping["task_project"], "Praxist task project")
    executable = _resolve_registry_file(base, mapping["praxist_executable"], "Praxist executable")
    run_root = _resolve_registry_output_root(base, mapping["run_root"])
    launch = mapping["launch"]
    config_file = (
        _resolve_registry_file(base, launch["config_file"], "Praxist config file")
        if launch["config_file"] is not None
        else None
    )
    task = validate_research_task(config=config, task_path=task_manifest)
    task_ref = _content_ref(config, task_manifest, context="research task manifest")
    executable_ref = _absolute_content_ref(executable, context="Praxist executable")
    config_ref = _absolute_content_ref(config_file, context="Praxist config file") if config_file is not None else None
    profile = _normalized_launch_profile(launch, config_ref=config_ref)
    normalized_binding = {
        "mapping_id": mapping["mapping_id"],
        "binding_revision": mapping["binding_revision"],
        "task_manifest": task_ref,
        "task_project": str(task_project),
        "task_project_manifest_sha256": task["praxist"]["task_project_manifest_sha256"],
        "praxist_executable": executable_ref,
        "run_root": str(run_root),
        "launch": profile,
    }
    binding_sha256 = sha256_text(canonical_json(normalized_binding))
    opportunity_digest = str(opportunity["opportunity_id"]).partition(":")[2]
    run_id = f"run_{opportunity_digest[:20]}_{binding_sha256[:12]}"
    run_dir = run_root / run_id
    return _ResolvedBinding(
        mapping_id=str(mapping["mapping_id"]),
        binding_sha256=binding_sha256,
        task_id=str(task["task_id"]),
        owner=str(task["owner"]),
        capability_axes=tuple(str(value) for value in task["capability_axes"]),
        task_manifest=task_manifest,
        task_project=task_project,
        praxist_executable=executable,
        run_dir=run_dir,
        config_file=config_file,
        agent_system=profile["agent_system"],
        runtime=profile["runtime"],
        codex_native=profile["codex_native"],
        model_provider=profile["model_provider"],
        model=profile["model"],
        strategy=profile["strategy"],
        cohort=profile["cohort"],
        generations=profile["generations"],
        startup_timeout_seconds=profile["startup_timeout_seconds"],
    )


def _normalized_launch_profile(
    launch: Mapping[str, Any],
    *,
    config_ref: dict[str, str] | None,
) -> dict[str, Any]:
    agent_system = launch["agent_system"]
    runtime = launch["runtime"]
    model_provider = launch["model_provider"]
    if launch["codex_native"]:
        agent_system = agent_system or "codex_sdk"
        runtime = runtime or "agent_runtime:codex_sdk"
        model_provider = model_provider or "model_provider:openai_compatible"
    return {
        "config_file": config_ref,
        "agent_system": agent_system,
        "runtime": runtime,
        "codex_native": launch["codex_native"],
        "model_provider": model_provider,
        "model": launch["model"],
        "strategy": launch["strategy"],
        "cohort": launch["cohort"],
        "generations": launch["generations"],
        "startup_timeout_seconds": launch["startup_timeout_seconds"],
    }


def _find_submitted_request(
    *,
    config: ForgeConfig,
    opportunity: Mapping[str, Any],
    opportunity_path: Path,
    opportunity_sha256: str,
    binding: _ResolvedBinding,
) -> tuple[str, Path] | None:
    opportunity_ref = {
        "locator": _portable_locator(config, opportunity_path),
        "sha256": opportunity_sha256,
    }
    routed: list[tuple[str, Path]] = []
    for path in sorted((opportunity_path.parent / "routes").glob("*.json")):
        receipt = _load_routing_receipt(
            config=config,
            routing_path=path,
            opportunity=opportunity,
            opportunity_sha256=opportunity_sha256,
        )
        mapping = receipt["mapping"]
        if (
            receipt["decision"] == "SUBMITTED_FOR_A0"
            and mapping is not None
            and mapping["binding_sha256"] == binding.binding_sha256
        ):
            request_ref = receipt["request"]
            if request_ref is None:
                raise ResearchOpportunityError("submitted routing receipt is missing its Request")
            expected_mapping = _mapping_ref(binding)
            if mapping != expected_mapping:
                raise ResearchOpportunityError(
                    "routing receipt task identity does not match the current exact task binding"
                )
            request_path = _verify_content_ref(
                config,
                request_ref["artifact"],
                context="routed ResearchRequest",
            )
            request = validate_research_request(
                config=config,
                request_path=request_path,
                verify_bindings=False,
            )
            if request["request_id"] != request_ref["request_id"]:
                raise ResearchOpportunityError("routing receipt request_id does not match its Request")
            if not _request_matches_binding(
                config=config,
                request=request,
                opportunity_ref=opportunity_ref,
                pattern_id=str(opportunity["source"]["record"]["pattern_id"]),
                binding=binding,
            ):
                raise ResearchOpportunityError("routing receipt Request does not match the current exact task binding")
            routed.append((str(request["request_id"]), request_path))
    if routed:
        unique = {(request_id, str(path)) for request_id, path in routed}
        if len(unique) != 1:
            raise ResearchOpportunityError("multiple ResearchRequests are routed from one opportunity binding")
        return routed[0]

    recovered: list[tuple[str, Path]] = []
    control_root = config.paths.artifacts_root / "research_control"
    for request_path in sorted(control_root.glob("*/*/request.json")):
        request = validate_research_request(
            config=config,
            request_path=request_path,
            verify_bindings=False,
        )
        if _request_matches_binding(
            config=config,
            request=request,
            opportunity_ref=opportunity_ref,
            pattern_id=str(opportunity["source"]["record"]["pattern_id"]),
            binding=binding,
        ):
            recovered.append((str(request["request_id"]), request_path.resolve()))
    if len(recovered) > 1:
        raise ResearchOpportunityError("multiple orphan ResearchRequests match one opportunity binding")
    return recovered[0] if recovered else None


def _request_matches_binding(
    *,
    config: ForgeConfig,
    request: Mapping[str, Any],
    opportunity_ref: dict[str, str],
    pattern_id: str,
    binding: _ResolvedBinding,
) -> bool:
    task_ref = _content_ref(config, binding.task_manifest, context="research task manifest")
    executable_ref = _absolute_content_ref(binding.praxist_executable, context="Praxist executable")
    config_ref = (
        _absolute_content_ref(binding.config_file, context="Praxist config file")
        if binding.config_file is not None
        else None
    )
    expected_profile = {
        "config_file": config_ref,
        "agent_system": binding.agent_system,
        "runtime": binding.runtime,
        "codex_native": binding.codex_native,
        "model_provider": binding.model_provider,
        "model": binding.model,
        "strategy": binding.strategy,
        "cohort": binding.cohort,
        "generations": binding.generations,
        "startup_timeout_seconds": binding.startup_timeout_seconds,
    }
    return (
        request["trigger"]
        == {
            "kind": "forge_failure_pattern",
            "submitted_by": _SCANNER_ID,
            "rationale": _submission_reason_from_ids(
                pattern_id=pattern_id,
                mapping_id=binding.mapping_id,
            ),
            "evidence": [opportunity_ref],
        }
        and request["bindings"]["research_task"] == task_ref
        and request["bindings"]["task_project"]["root"] == str(binding.task_project)
        and request["bindings"]["praxist"]["executable"] == executable_ref
        and request["launch"]["run_dir"] == str(binding.run_dir)
        and request["launch"]["run_id"] == binding.run_dir.name
        and request["launch"]["profile"] == expected_profile
    )


def _submission_reason(
    opportunity: Mapping[str, Any],
    binding: _ResolvedBinding,
) -> str:
    pattern_id = str(opportunity["source"]["record"]["pattern_id"])
    return _submission_reason_from_ids(pattern_id=pattern_id, mapping_id=binding.mapping_id)


def _submission_reason_from_ids(*, pattern_id: str, mapping_id: str) -> str:
    return f"Typed failure pattern {pattern_id} mapped by {mapping_id}."


def _seal_routing_receipt(
    *,
    config: ForgeConfig,
    opportunity: Mapping[str, Any],
    opportunity_path: Path,
    opportunity_sha256: str,
    registry_ref: dict[str, str],
    mapping: _ResolvedBinding | None,
    decision: str,
    blocker_codes: tuple[str, ...],
    request: tuple[str, Path] | None,
) -> Path:
    request_payload = None
    if request is not None:
        request_id, request_path = request
        request_payload = {
            "request_id": request_id,
            "artifact": _content_ref(config, request_path, context="ResearchRequest"),
        }
    payload: dict[str, Any] = {
        "schema_version": _ROUTING_VERSION,
        "opportunity_id": opportunity["opportunity_id"],
        "opportunity_sha256": opportunity_sha256,
        "registry": registry_ref,
        "mapping": (
            _mapping_ref(mapping)
            if mapping is not None
            else None
        ),
        "decision": decision,
        "blocker_codes": list(blocker_codes),
        "request": request_payload,
        "authority": {
            "human_research_approval_required": True,
            "research_start_authorized": False,
            "formal_validation_authorized": False,
            "production_promotion_authorized": False,
            "runtime_wiring_changed": False,
            "evaluation_is_learning_source": False,
        },
        "created_at": utc_now(),
    }
    _validate_routing_semantics(payload)
    payload["routing_id"] = _artifact_id(
        "research-opportunity-routing",
        payload,
        "routing_id",
    )
    _validate_schema(config, payload, _ROUTING_VERSION)
    digest = str(payload["routing_id"]).partition(":")[2]
    destination = opportunity_path.parent / "routes" / f"{digest}.json"
    _write_immutable_artifact(
        config=config,
        destination=destination,
        payload=payload,
        expected_version=_ROUTING_VERSION,
        identity_field="routing_id",
    )
    return destination


def _mapping_ref(mapping: _ResolvedBinding) -> dict[str, Any]:
    return {
        "mapping_id": mapping.mapping_id,
        "binding_sha256": mapping.binding_sha256,
        "task_id": mapping.task_id,
        "owner": mapping.owner,
        "capability_axes": list(mapping.capability_axes),
    }


def _validate_routing_semantics(receipt: Mapping[str, Any]) -> None:
    decision = receipt["decision"]
    mapping = receipt["mapping"]
    request = receipt["request"]
    blockers = tuple(receipt["blocker_codes"])
    if decision == "NEEDS_TASK_DESIGN":
        if mapping is not None or request is not None or not blockers:
            raise ResearchOpportunityError("NEEDS_TASK_DESIGN requires blockers and no mapping/request")
    elif decision == "DEFERRED_BY_SCAN_LIMIT":
        if mapping is None or request is not None or blockers != ("SCAN_SUBMISSION_LIMIT_REACHED",):
            raise ResearchOpportunityError("deferred routing receipt has inconsistent fields")
    elif decision == "SUBMITTED_FOR_A0":
        if mapping is None or request is None or blockers:
            raise ResearchOpportunityError("submitted routing receipt requires mapping/request and no blockers")
    else:
        raise ResearchOpportunityError(f"unsupported routing decision: {decision!r}")


def _load_opportunity(
    config: ForgeConfig,
    opportunity_path: Path,
) -> tuple[dict[str, Any], Path, str]:
    resolved = _resolve_regular_file(opportunity_path, context="ResearchOpportunity")
    payload = read_json(resolved)
    _validate_schema(config, payload, _OPPORTUNITY_VERSION)
    if payload["opportunity_id"] != _artifact_id(
        "research-opportunity",
        payload,
        "opportunity_id",
    ):
        raise ResearchOpportunityError("ResearchOpportunity identity does not match its payload")
    record = payload["source"]["record"]
    _validate_pattern_identity(record)
    record_sha256 = sha256_text(canonical_json(record))
    if record_sha256 != payload["source"]["record_sha256"]:
        raise ResearchOpportunityError("ResearchOpportunity source record digest mismatch")
    key_payload = {
        "source_kind": "forge_failure_pattern",
        "source_schema_version": record["schema_version"],
        "pattern_id": record["pattern_id"],
        "record_sha256": record_sha256,
    }
    expected_key = f"research-opportunity-key:{sha256_text(canonical_json(key_payload))}"
    if payload["opportunity_key"] != expected_key:
        raise ResearchOpportunityError("ResearchOpportunity key does not match its source record")
    blockers = _nomination_blockers(record)
    expected_readiness = "NEEDS_TASK_DESIGN" if blockers else "ROUTABLE"
    if payload["nomination"] != {
        "readiness": expected_readiness,
        "editable_component": record["editable_component"],
        "editable_target": record["editable_target"],
        "blocker_codes": list(blockers),
    }:
        raise ResearchOpportunityError("ResearchOpportunity nomination does not match typed source fields")
    expected_score = min(int(record["occurrence_count"]), _OCCURRENCE_CAP)
    if payload["priority"] != {
        "policy": "typed_occurrence_count_capped.v1",
        "occurrence_count": record["occurrence_count"],
        "occurrence_cap": _OCCURRENCE_CAP,
        "score": expected_score,
    }:
        raise ResearchOpportunityError("ResearchOpportunity priority does not match typed occurrence policy")
    opportunity_digest = str(payload["opportunity_id"]).partition(":")[2]
    key_digest = str(payload["opportunity_key"]).partition(":")[2]
    expected_parent = (
        config.paths.artifacts_root / "research_opportunities" / key_digest / opportunity_digest
    ).resolve(strict=False)
    if resolved.name != "opportunity.json" or resolved.parent != expected_parent:
        raise ResearchOpportunityError("ResearchOpportunity is not stored at its canonical path")
    return payload, resolved, _sha256_file(resolved, context="ResearchOpportunity")


def _load_routing_receipt(
    *,
    config: ForgeConfig,
    routing_path: Path,
    opportunity: Mapping[str, Any],
    opportunity_sha256: str,
) -> dict[str, Any]:
    resolved = _resolve_regular_file(routing_path, context="ResearchOpportunity routing receipt")
    payload = read_json(resolved)
    _validate_schema(config, payload, _ROUTING_VERSION)
    if payload["routing_id"] != _artifact_id(
        "research-opportunity-routing",
        payload,
        "routing_id",
    ):
        raise ResearchOpportunityError("routing receipt identity does not match its payload")
    _validate_routing_semantics(payload)
    if payload["opportunity_id"] != opportunity["opportunity_id"]:
        raise ResearchOpportunityError("routing receipt references another ResearchOpportunity")
    if payload["opportunity_sha256"] != opportunity_sha256:
        raise ResearchOpportunityError("routing receipt ResearchOpportunity digest mismatch")
    expected_name = str(payload["routing_id"]).partition(":")[2] + ".json"
    key_digest = str(opportunity["opportunity_key"]).partition(":")[2]
    opportunity_digest = str(opportunity["opportunity_id"]).partition(":")[2]
    expected_parent = (
        config.paths.artifacts_root / "research_opportunities" / key_digest / opportunity_digest / "routes"
    ).resolve(strict=False)
    if resolved.parent != expected_parent:
        raise ResearchOpportunityError("routing receipt is outside its ResearchOpportunity directory")
    if resolved.name != expected_name:
        raise ResearchOpportunityError("routing receipt is not stored at its canonical path")
    return payload


def _status(
    *,
    opportunity: Mapping[str, Any],
    opportunity_path: Path,
    routing_path: Path,
    state: str,
    blocker_codes: tuple[str, ...],
    binding: _ResolvedBinding | None = None,
    request: tuple[str, Path] | None = None,
) -> ResearchOpportunityStatus:
    return ResearchOpportunityStatus(
        opportunity_id=str(opportunity["opportunity_id"]),
        pattern_id=str(opportunity["source"]["record"]["pattern_id"]),
        priority_score=int(opportunity["priority"]["score"]),
        state=state,
        blocker_codes=blocker_codes,
        opportunity_path=opportunity_path,
        routing_path=routing_path,
        mapping_id=binding.mapping_id if binding is not None else None,
        request_id=request[0] if request is not None else None,
        request_path=request[1] if request is not None else None,
    )


def _resolve_registry_file(base: Path, value: Any, context: str) -> Path:
    path = _registry_path(base, value, context)
    return _resolve_regular_file(path, context=context)


def _resolve_registry_directory(base: Path, value: Any, context: str) -> Path:
    path = _registry_path(base, value, context)
    if path.is_symlink():
        raise ResearchOpportunityError(f"{context} may not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchOpportunityError(f"missing {context}: {path}") from exc
    if not resolved.is_dir():
        raise ResearchOpportunityError(f"{context} must be a directory: {resolved}")
    return resolved


def _resolve_registry_output_root(base: Path, value: Any) -> Path:
    path = _registry_path(base, value, "Praxist run root")
    if path.is_symlink():
        raise ResearchOpportunityError(f"Praxist run root may not be a symlink: {path}")
    resolved = path.resolve(strict=False)
    if resolved.exists() and not resolved.is_dir():
        raise ResearchOpportunityError(f"Praxist run root must be a directory: {resolved}")
    return resolved


def _registry_path(base: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ResearchOpportunityError(f"{context} path must be non-empty")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _resolve_regular_file(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ResearchOpportunityError(f"{context} may not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchOpportunityError(f"missing {context}: {expanded}") from exc
    if not resolved.is_file():
        raise ResearchOpportunityError(f"{context} must be a regular file: {resolved}")
    return resolved


def _content_ref(config: ForgeConfig, path: Path, *, context: str) -> dict[str, str]:
    resolved = _resolve_regular_file(path, context=context)
    return {
        "locator": _portable_locator(config, resolved),
        "sha256": _sha256_file(resolved, context=context),
    }


def _absolute_content_ref(path: Path, *, context: str) -> dict[str, str]:
    resolved = _resolve_regular_file(path, context=context)
    return {
        "locator": str(resolved),
        "sha256": _sha256_file(resolved, context=context),
    }


def _verify_content_ref(
    config: ForgeConfig,
    content_ref: Mapping[str, Any],
    *,
    context: str,
) -> Path:
    locator = str(content_ref["locator"])
    path = Path(locator).expanduser()
    if not path.is_absolute():
        relative = PurePosixPath(locator)
        if not locator or "\\" in locator or ".." in relative.parts or "." in relative.parts:
            raise ResearchOpportunityError(f"unsafe content locator: {locator!r}")
        path = config.paths.repo_root / Path(*relative.parts)
    resolved = _resolve_regular_file(path, context=context)
    actual = _sha256_file(resolved, context=context)
    if actual != content_ref["sha256"]:
        raise ResearchOpportunityError(f"{context} digest mismatch")
    return resolved


def _portable_locator(config: ForgeConfig, path: Path) -> str:
    repo_root = config.paths.repo_root.resolve()
    return path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path)


def _artifact_id(prefix: str, payload: Mapping[str, Any], identity_field: str) -> str:
    identity_payload = {key: value for key, value in payload.items() if key not in {identity_field, "created_at"}}
    return f"{prefix}:{sha256_text(canonical_json(identity_payload))}"


def _validate_schema(
    config: ForgeConfig,
    payload: dict[str, Any],
    expected_version: str,
) -> None:
    SchemaStore(config.paths.forge_root / "schemas").validate(payload, SCHEMA_NAME)
    if payload.get("schema_version") != expected_version:
        raise ResearchOpportunityError(
            f"expected schema_version {expected_version!r}, got {payload.get('schema_version')!r}"
        )


def _write_immutable_artifact(
    *,
    config: ForgeConfig,
    destination: Path,
    payload: dict[str, Any],
    expected_version: str,
    identity_field: str,
) -> None:
    root = (config.paths.artifacts_root / "research_opportunities").resolve(strict=False)
    target = destination.expanduser().resolve(strict=False)
    if not target.is_relative_to(root):
        raise ResearchOpportunityError(
            "ResearchOpportunity artifacts may only be written below artifacts/research_opportunities"
        )
    if target.exists():
        existing = read_json(target)
        _validate_schema(config, existing, expected_version)
        if existing[identity_field] != payload[identity_field]:
            raise ResearchOpportunityError(f"refusing to overwrite another immutable artifact: {target}")
        existing_body = {key: value for key, value in existing.items() if key not in {identity_field, "created_at"}}
        payload_body = {key: value for key, value in payload.items() if key not in {identity_field, "created_at"}}
        if existing_body != payload_body:
            raise ResearchOpportunityError(f"refusing to overwrite changed immutable artifact: {target}")
        return
    _write_create_only_json(target, payload)


def _write_create_only_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ResearchOpportunityError(f"refusing to overwrite create-only artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _sha256_file(path: Path, *, context: str) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise ResearchOpportunityError(f"cannot read {context} at {path}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise ResearchOpportunityError(f"cannot open artifact directory for fsync {path}: {exc}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ResearchOpportunityError(f"cannot fsync artifact directory {path}: {exc}") from exc
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Forge scan currently targets POSIX hosts.
        raise ResearchOpportunityError("research scan requires POSIX file locking") from exc
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ResearchOpportunityError(f"cannot open research scan lock {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise
