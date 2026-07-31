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
) -> ProposeResult:
    if max_proposals <= 0:
        raise ForgeError("max_proposals must be positive")
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
    for pattern in candidates:
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
        after = _append_section(before, response["section_content"])
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
    if config.editable_entry_for(target) is None:
        raise BackendError(f"Proposer selected a protected or out-of-surface target: {target}")
    section = response["section_content"].strip()
    if not section.startswith("#"):
        raise BackendError("append_section content must begin with a Markdown heading")
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
