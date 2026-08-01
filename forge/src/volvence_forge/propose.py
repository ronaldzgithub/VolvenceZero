"""Generate narrow, evidence-bound proposal bundles without mutating targets."""

from __future__ import annotations

import difflib
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import ForgeConfig
from .foundation import (
    BackendError,
    EmbeddingBackend,
    ForgeError,
    PromptStore,
    SchemaStore,
    StructuredBackend,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    cosine_similarity,
    sha256_text,
    utc_now,
    utc_stamp,
)


@dataclass(frozen=True)
class ProposeResult:
    output_dir: Path
    report_path: Path
    proposal_dirs: tuple[Path, ...]
    skipped_duplicates: int


def propose_changes(
    *,
    config: ForgeConfig,
    failure_patterns_path: Path,
    backend: StructuredBackend,
    embedder: EmbeddingBackend,
    output_dir: Path | None = None,
    max_proposals: int = 3,
    candidates_per_pattern: int = 1,
) -> ProposeResult:
    if max_proposals <= 0:
        raise ForgeError("max_proposals must be positive")
    if not 1 <= candidates_per_pattern <= 16:
        raise ForgeError("candidates_per_pattern must be within [1, 16]")
    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    prompt_store = PromptStore(config.paths.forge_root / "prompts")
    patterns = _load_patterns(failure_patterns_path, schema_store)
    candidates = [pattern for pattern in patterns if pattern["surface_status"] == "in-surface"]
    if not candidates:
        raise ForgeError("No in-surface failure pattern is eligible for a proposal")
    destination = output_dir or config.paths.artifacts_root / f"forge_propose_{utc_stamp()}"
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    proposals_root = destination / "proposals"
    proposals_root.mkdir()

    prior_summaries = _read_prior_summaries(config.paths.ledger_path)
    comparison_texts = list(prior_summaries)
    comparison_vectors = embedder.encode(comparison_texts) if comparison_texts else np.empty((0, 0), dtype=np.float64)
    proposal_dirs: list[Path] = []
    skipped_duplicates = 0
    skips: list[str] = []
    population = tuple(
        pattern
        for pattern in candidates
        for _candidate_index in range(candidates_per_pattern)
    )
    for pattern in population:
        if len(proposal_dirs) >= max_proposals:
            break
        target = pattern["editable_target"]
        if not isinstance(target, str):
            raise ForgeError(f"In-surface pattern {pattern['pattern_id']} is missing editable_target")
        entry = config.editable_entry_for(target)
        if entry is None:
            raise ForgeError(f"Pattern target is no longer editable under current policy: {target}")
        target_path = config.resolve_target(target, must_exist=True)
        before = target_path.read_text(encoding="utf-8")
        response = backend.complete_json(
            system=prompt_store.render("proposal.system.md"),
            user=prompt_store.render(
                "proposal.user.md",
                failure_pattern=json.dumps(pattern, ensure_ascii=False, indent=2, sort_keys=True),
                target_path=target,
                target_content=before[:16000],
                prior_proposals=json.dumps(prior_summaries[-30:], ensure_ascii=False, indent=2),
            ),
            schema=schema_store.load("proposal_candidate.schema.json"),
        )
        schema_store.validate(response, "proposal_candidate.schema.json")
        _validate_candidate(response, pattern, config)
        after = _render_candidate(
            before=before,
            content=response["section_content"],
            operation=response["operation"],
            document_path=response.get("document_path"),
            target=target,
            requires_offline_gate=entry.requires_offline_gate,
        )
        patch = _unified_diff(target, before, after)
        novelty_text = f"{target}\n{response['targeted_fix']}\n{response['section_content']}"
        novelty_vector = embedder.encode([novelty_text])[0]
        if comparison_vectors.size:
            maximum_similarity = max(cosine_similarity(novelty_vector, vector) for vector in comparison_vectors)
            if maximum_similarity >= config.proposal_duplicate_similarity:
                skipped_duplicates += 1
                skips.append(
                    f"{pattern['pattern_id']}: semantic duplicate (similarity={maximum_similarity:.3f})"
                )
                continue
        comparison_vectors = (
            np.vstack((comparison_vectors, novelty_vector)) if comparison_vectors.size else np.asarray([novelty_vector])
        )
        comparison_texts.append(novelty_text)

        proposal_identity = canonical_json(
            {"pattern_id": pattern["pattern_id"], "target": target, "patch_sha256": sha256_text(patch)}
        )
        proposal_id = f"pr_{sha256_text(proposal_identity)[:16]}"
        proposal_dir = proposals_root / proposal_id
        proposal_dir.mkdir()
        manifesto = _manifesto(
            proposal_id=proposal_id,
            pattern=pattern,
            target=target,
            before=before,
            response=response,
            backend=backend,
            rollback_command=(
                "git apply --reverse "
                f"{shlex.quote(os.path.relpath(proposal_dir / 'patch.diff', config.paths.repo_root))}"
            ),
        )
        schema_store.validate(manifesto, "proposal_manifesto.schema.json")
        atomic_write_text(proposal_dir / "patch.diff", patch)
        atomic_write_json(proposal_dir / "manifesto.json", manifesto)
        atomic_write_json(proposal_dir / "failure_pattern.json", pattern)
        proposal_dirs.append(proposal_dir)

    report_path = destination / "report.md"
    atomic_write_text(
        report_path,
        _render_report(
            backend=backend,
            embedder=embedder,
            proposal_dirs=proposal_dirs,
            skipped=skips,
            source=failure_patterns_path,
        ),
    )
    return ProposeResult(
        output_dir=destination,
        report_path=report_path,
        proposal_dirs=tuple(proposal_dirs),
        skipped_duplicates=skipped_duplicates,
    )


def _load_patterns(path: Path, schema_store: SchemaStore) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ForgeError(f"Missing failure pattern file: {path}") from exc
    patterns: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            pattern = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ForgeError(f"Invalid pattern JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(pattern, dict):
            raise ForgeError(f"Pattern must be an object at {path}:{line_number}")
        schema_store.validate(pattern, "failure_pattern.schema.json")
        patterns.append(pattern)
    return patterns


def _validate_candidate(response: dict[str, Any], pattern: dict[str, Any], config: ForgeConfig) -> None:
    target = response["target"]
    if target != pattern["editable_target"]:
        raise BackendError(
            f"Proposer changed the frozen target for {pattern['pattern_id']}: "
            f"expected {pattern['editable_target']!r}, got {target!r}"
        )
    entry = config.editable_entry_for(target)
    if entry is None:
        raise BackendError(f"Proposer selected a protected or out-of-surface target: {target}")
    operation = response["operation"]
    document_path = response.get("document_path")
    if operation == "append_section":
        if document_path is not None:
            raise BackendError("append_section must not declare document_path")
        section = response["section_content"].strip()
        if not section.startswith("#"):
            raise BackendError("append_section content must begin with a Markdown heading")
        if entry.requires_offline_gate:
            raise BackendError("runtime semantic assets require a structured append operation")
    elif operation == "append_yaml_sequence_item":
        if not entry.requires_offline_gate or not target.endswith("/scenes.yaml"):
            raise BackendError("append_yaml_sequence_item is limited to gated scenes.yaml assets")
        if document_path != "/scenes":
            raise BackendError("scenes.yaml append must use document_path=/scenes")
    elif operation == "append_json_array_item":
        if not entry.requires_offline_gate:
            raise BackendError("append_json_array_item requires an offline-gated asset")
        if target.endswith("/ssot_fragment.json"):
            if document_path not in {"/paths", "/arc_specs"}:
                raise BackendError("ssot_fragment append path must be /paths or /arc_specs")
        elif target.endswith("/companion_playbook_overlay.json"):
            if document_path != "/playbook_rules":
                raise BackendError(
                    "companion playbook overlay append must use document_path=/playbook_rules"
                )
        else:
            raise BackendError("append_json_array_item target is not an approved structured asset")
    else:
        raise BackendError(f"Unsupported proposal operation: {operation}")
    expected_preserve = set(pattern["preserve_behaviors"])
    returned_preserve = set(response["preserve_behaviors"])
    if not expected_preserve.issubset(returned_preserve):
        missing = sorted(expected_preserve - returned_preserve)
        raise BackendError(f"Proposer dropped passing behaviors that must be preserved: {missing}")


def _append_section(before: str, section: str) -> str:
    normalized_section = section.strip() + "\n"
    if normalized_section.strip() in before:
        raise BackendError("Proposed section already exists in target content")
    separator = "\n" if before.endswith("\n") else "\n\n"
    return before + separator + normalized_section


def _render_candidate(
    *,
    before: str,
    content: str,
    operation: str,
    document_path: str | None,
    target: str,
    requires_offline_gate: bool,
) -> str:
    if operation == "append_section":
        return _append_section(before, content)
    if not requires_offline_gate:
        raise BackendError("structured document operations require an offline-gated component")
    if operation == "append_yaml_sequence_item":
        return _append_yaml_sequence_item(before, content, target=target)
    if operation == "append_json_array_item":
        if document_path is None:
            raise BackendError("append_json_array_item requires document_path")
        return _append_json_array_item(before, content, document_path=document_path, target=target)
    raise BackendError(f"Unsupported proposal operation: {operation}")


def _append_yaml_sequence_item(before: str, fragment: str, *, target: str) -> str:
    if not before.endswith("\n"):
        raise BackendError(f"Runtime YAML asset must end with a newline: {target}")
    normalized = fragment.rstrip() + "\n"
    if not normalized.startswith("  - "):
        raise BackendError("scenes.yaml fragment must start with a two-space sequence item")
    try:
        baseline = yaml.safe_load(before)
        fragment_document = yaml.safe_load("scenes:\n" + normalized)
        candidate = yaml.safe_load(before + normalized)
    except yaml.YAMLError as exc:
        raise BackendError(f"Proposed YAML fragment is invalid for {target}: {exc}") from exc
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise BackendError("Runtime YAML asset must remain a mapping")
    baseline_scenes = baseline.get("scenes")
    candidate_scenes = candidate.get("scenes")
    fragment_scenes = fragment_document.get("scenes") if isinstance(fragment_document, dict) else None
    if not isinstance(baseline_scenes, list) or not isinstance(candidate_scenes, list):
        raise BackendError("scenes.yaml must contain a scenes sequence")
    if not isinstance(fragment_scenes, list) or len(fragment_scenes) != 1:
        raise BackendError("Proposal must append exactly one scene")
    if candidate_scenes[:-1] != baseline_scenes:
        raise BackendError("Runtime scene proposal may not rewrite existing scenes")
    _ensure_unique_identifier(candidate_scenes, "scenario_id", target)
    return before + normalized


def _append_json_array_item(
    before: str,
    fragment: str,
    *,
    document_path: str,
    target: str,
) -> str:
    try:
        baseline = json.loads(before)
        item = json.loads(fragment)
    except json.JSONDecodeError as exc:
        raise BackendError(f"Proposed JSON fragment is invalid for {target}: {exc}") from exc
    canonical_before = json.dumps(baseline, ensure_ascii=False, indent=2) + "\n"
    if canonical_before != before:
        raise BackendError(f"Runtime JSON asset is not in the frozen canonical format: {target}")
    if not isinstance(baseline, dict) or not isinstance(item, dict):
        raise BackendError("Runtime JSON asset and appended item must be mappings")
    key = document_path.removeprefix("/")
    collection = baseline.get(key)
    if not isinstance(collection, list):
        raise BackendError(f"JSON document path {document_path} is not a sequence")
    collection.append(item)
    identifiers = {
        "paths": "path_id",
        "arc_specs": "arc_spec_id",
        "playbook_rules": "rule_id",
    }
    identifier = identifiers.get(key)
    if identifier is None:
        raise BackendError(f"Unsupported JSON append collection: {key}")
    _ensure_unique_identifier(collection, identifier, target)
    if key == "playbook_rules":
        _ensure_unique_identifier(collection, "problem_pattern", target)
    return json.dumps(baseline, ensure_ascii=False, indent=2) + "\n"


def _ensure_unique_identifier(items: list[Any], field: str, target: str) -> None:
    identifiers: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise BackendError(f"{target} item {index} must be a mapping")
        value = item.get(field)
        if not isinstance(value, str) or not value:
            raise BackendError(f"{target} item {index} lacks {field}")
        identifiers.append(value)
    if len(identifiers) != len(set(identifiers)):
        raise BackendError(f"{target} contains duplicate {field} values")


def _unified_diff(target: str, before: str, after: str) -> str:
    patch = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{target}",
            tofile=f"b/{target}",
        )
    )
    if not patch:
        raise BackendError("Proposer produced an empty patch")
    return patch


def _manifesto(
    *,
    proposal_id: str,
    pattern: dict[str, Any],
    target: str,
    before: str,
    response: dict[str, Any],
    backend: StructuredBackend,
    rollback_command: str,
) -> dict[str, Any]:
    prediction = response["prediction"]
    return {
        "schema_version": "forge-proposal-manifesto.v1",
        "proposal_id": proposal_id,
        "pattern_id": pattern["pattern_id"],
        "target": target,
        "target_preimage_sha256": sha256_text(before),
        "evidence": pattern["evidence_refs"],
        "root_cause": response["root_cause"],
        "targeted_fix": response["targeted_fix"],
        "predicted_impact": {
            **prediction,
            "baseline_value": pattern["occurrence_count"],
        },
        "at_risk_regressions": response["at_risk_regressions"],
        "preserve_behaviors": pattern["preserve_behaviors"],
        "rollback": {
            "method": "reverse_patch",
            "working_directory": "repository_root",
            "command": rollback_command,
        },
        "generator": {"backend": backend.backend_name, "model": backend.model_name},
        "created_at": utc_now(),
    }


def _read_prior_summaries(ledger_path: Path) -> tuple[str, ...]:
    if not ledger_path.exists():
        return ()
    summaries: list[str] = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ForgeError(f"Invalid ledger JSON at {ledger_path}:{line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ForgeError(f"Ledger event must be an object at {ledger_path}:{line_number}")
        if event.get("event") != "proposal_decision":
            continue
        summary = event.get("proposal_summary")
        if not isinstance(summary, str) or not summary:
            raise ForgeError(f"Proposal decision lacks proposal_summary at {ledger_path}:{line_number}")
        summaries.append(summary)
    return tuple(summaries)


def _render_report(
    *,
    backend: StructuredBackend,
    embedder: EmbeddingBackend,
    proposal_dirs: list[Path],
    skipped: list[str],
    source: Path,
) -> str:
    lines = [
        "# Forge Proposal Report",
        "",
        f"- Source patterns: `{source}`",
        f"- Generator: `{backend.backend_name}` / `{backend.model_name}`",
        f"- Diversity embedding: `{embedder.model_name}`",
        f"- Proposal bundles: {len(proposal_dirs)}",
        f"- Semantic duplicates skipped: {len(skipped)}",
        "",
        "## Bundles",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in proposal_dirs)
    lines.extend(["", "## Skipped", ""])
    lines.extend(f"- {item}" for item in skipped)
    if not skipped:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)
