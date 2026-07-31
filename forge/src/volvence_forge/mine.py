"""Semantic, schema-bound failure mining for Forge.

The module requires injected structured-output and embedding backends. There is
no keyword or hash fallback: unavailable semantic infrastructure is an
observable failure, not a hidden second decision owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ForgeConfig
from .foundation import (
    BackendError,
    EmbeddingBackend,
    PromptStore,
    SchemaStore,
    StructuredBackend,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    normalized,
    utc_now,
    utc_stamp,
)
from .sources import (
    BenchBundleSource,
    EvidenceRef,
    PlanSource,
    SourceBundle,
    TranscriptSource,
    VerdictSource,
    source_bundle_digest,
)


@dataclass(frozen=True)
class FailureAnalysis:
    verifier_cause: str
    agent_behavior_cause: str
    exposed_mechanism: str
    confidence: float
    preserve_behaviors: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "FailureAnalysis":
        fields = (
            "verifier_cause",
            "agent_behavior_cause",
            "exposed_mechanism",
        )
        values = {field: raw.get(field) for field in fields}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError("failure analysis causal fields must be non-empty strings")
        confidence = raw.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("failure analysis confidence must be numeric")
        if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("failure analysis confidence must be in [0, 1]")
        preserve = raw.get("preserve_behaviors", ())
        if not isinstance(preserve, Sequence) or isinstance(preserve, (str, bytes)):
            raise ValueError("preserve_behaviors must be a sequence")
        preserve_values = tuple(
            item.strip() for item in preserve if isinstance(item, str) and item.strip()
        )
        return cls(
            verifier_cause=values["verifier_cause"].strip(),
            agent_behavior_cause=values["agent_behavior_cause"].strip(),
            exposed_mechanism=values["exposed_mechanism"].strip(),
            confidence=float(confidence),
            preserve_behaviors=preserve_values,
        )


@dataclass(frozen=True)
class MineResult:
    output_dir: Path
    pattern_path: Path
    report_path: Path
    pattern_count: int
    in_surface_count: int


def _analysis_text(analysis: FailureAnalysis) -> str:
    return " ".join(
        (
            analysis.verifier_cause,
            analysis.agent_behavior_cause,
            analysis.exposed_mechanism,
        )
    )


def _source_objects(
    bundle: SourceBundle,
) -> tuple[TranscriptSource | VerdictSource | BenchBundleSource, ...]:
    """Select only sources carrying explicit structured failure evidence."""

    failed_transcripts = tuple(source for source in bundle.transcripts if source.error_refs)
    failed_verdicts = tuple(source for source in bundle.verdicts if source.failed_gate_refs)
    failed_bench = tuple(source for source in bundle.bench_bundles if source.failure_refs)
    return failed_transcripts + failed_verdicts + failed_bench


def _fallback_ref(
    source: TranscriptSource | VerdictSource | PlanSource | BenchBundleSource,
) -> EvidenceRef:
    record = source.analysis_record()
    excerpt = canonical_json(record)[:1200]
    digest = hashlib.sha256(f"{source.source_id}\nsource\n{excerpt}".encode()).hexdigest()
    return EvidenceRef(
        source_id=source.source_id,
        source_kind=record["source_kind"],
        locator="source",
        excerpt=excerpt,
        digest=digest,
    )


def _evidence_refs(
    source: TranscriptSource | VerdictSource | PlanSource | BenchBundleSource,
) -> tuple[EvidenceRef, ...]:
    if isinstance(source, TranscriptSource) and source.error_refs:
        return source.error_refs
    if isinstance(source, VerdictSource) and source.failed_gate_refs:
        return source.failed_gate_refs
    if isinstance(source, BenchBundleSource) and source.failure_refs:
        return source.failure_refs
    return (_fallback_ref(source),)


def _validate_embedding_matrix(values: np.ndarray, *, expected_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] == 0:
        raise ValueError(f"embedding backend returned invalid shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("embedding backend returned non-finite values")
    return np.vstack([normalized(row) for row in array])


def _pattern_id(payload: Mapping[str, object]) -> str:
    return "fp_" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]


def mine_bundle(
    *,
    bundle: SourceBundle,
    config: ForgeConfig,
    structured_backend: StructuredBackend,
    embedding_backend: EmbeddingBackend,
    schema_store: SchemaStore,
    prompt_store: PromptStore,
) -> tuple[dict[str, object], ...]:
    """Analyze sources and semantically cluster their causal failure records."""

    analysis_schema = schema_store.load("failure_analysis.schema.json")
    system_prompt = prompt_store.render("failure_mining.system.md")
    records: list[tuple[Any, FailureAnalysis, np.ndarray]] = []
    campaign_context = [plan.analysis_record() for plan in bundle.plans]
    for source in _source_objects(bundle):
        source_record = {**source.analysis_record(), "campaign_context": campaign_context}
        response = structured_backend.complete_json(
            system=system_prompt,
            user=prompt_store.render(
                "failure_mining.user.md",
                source_record=canonical_json(source_record),
                passing_behaviors=canonical_json(
                    source_record.get("passing_behaviors", [])
                ),
            ),
            schema=analysis_schema,
        )
        raw_records = response.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("failure analysis backend must return a records list")
        analyses = tuple(FailureAnalysis.from_mapping(raw) for raw in raw_records)
        if not analyses:
            continue
        vectors = _validate_embedding_matrix(
            embedding_backend.encode([_analysis_text(analysis) for analysis in analyses]),
            expected_rows=len(analyses),
        )
        for index, analysis in enumerate(analyses):
            records.append((source, analysis, vectors[index]))

    clusters: list[dict[str, object]] = []
    for source, analysis, vector in records:
        target_cluster: dict[str, object] | None = None
        for cluster in clusters:
            centroid = cluster["centroid"]
            assert isinstance(centroid, np.ndarray)
            if float(np.dot(vector, centroid)) >= config.cluster_similarity:
                target_cluster = cluster
                break
        if target_cluster is None:
            target_cluster = {"centroid": vector.copy(), "items": []}
            clusters.append(target_cluster)
        items = target_cluster["items"]
        assert isinstance(items, list)
        items.append((source, analysis, vector))
        matrix = np.vstack([item[2] for item in items])
        target_cluster["centroid"] = normalized(matrix.mean(axis=0))

    surface_assets = config.editable_assets()
    surface_texts = [
        f"component={entry.component}\npath={relative}\n{entry.semantic_description}\n"
        f"{path.read_text(encoding='utf-8')[:6000]}"
        for entry, relative, path in surface_assets
    ]
    patterns: list[dict[str, object]] = []
    for cluster in clusters:
        items = cluster["items"]
        assert isinstance(items, list) and items
        analyses = [item[1] for item in items]
        sources = [item[0] for item in items]
        centroid = cluster["centroid"]
        assert isinstance(centroid, np.ndarray)
        if surface_assets:
            surface_vectors = _validate_embedding_matrix(
                embedding_backend.encode(surface_texts),
                expected_rows=len(surface_assets),
            )
            scores = surface_vectors @ centroid
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
        else:
            best_index = -1
            best_score = -1.0
        if best_index >= 0 and best_score >= config.minimum_surface_similarity:
            target_entry, target_relative, _target_path = surface_assets[best_index]
            editable_target = target_relative
            editable_component = target_entry.component
            surface_status = "in-surface"
        else:
            editable_target = None
            editable_component = None
            surface_status = "out-of-surface"
        evidence_refs: list[dict[str, object]] = []
        seen_refs: set[tuple[str, str]] = set()
        for source in sources:
            for reference in _evidence_refs(source):
                identity = (reference.source_id, reference.locator)
                if identity in seen_refs:
                    continue
                seen_refs.add(identity)
                evidence_refs.append(reference.as_dict())
        representative = analyses[0]
        base = {
            "verifier_cause": representative.verifier_cause,
            "agent_behavior_cause": representative.agent_behavior_cause,
            "exposed_mechanism": representative.exposed_mechanism,
            "evidence_refs": evidence_refs,
        }
        pattern = {
            "schema_version": "forge-failure-pattern.v2",
            "pattern_id": _pattern_id(base),
            "title": representative.exposed_mechanism[:240],
            **base,
            "occurrence_count": len(items),
            "source_kinds": sorted({source.analysis_record()["source_kind"] for source in sources}),
            "centroid_digest": hashlib.sha256(centroid.tobytes()).hexdigest(),
            "editable_target": editable_target,
            "editable_component": editable_component,
            "surface_status": surface_status,
            "surface_similarity": best_score,
            "preserve_behaviors": sorted(
                {
                    behavior
                    for analysis in analyses
                    for behavior in analysis.preserve_behaviors
                }
            ),
        }
        schema_store.validate(pattern, "failure_pattern.schema.json")
        patterns.append(pattern)
    return tuple(sorted(patterns, key=lambda item: str(item["pattern_id"])))


def write_failure_patterns(patterns: Sequence[Mapping[str, object]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for pattern in patterns:
            handle.write(canonical_json(dict(pattern)) + "\n")
    return output


def mine_failures(
    *,
    config: ForgeConfig,
    sources: SourceBundle,
    embedder: EmbeddingBackend,
    backend: StructuredBackend | None,
    output_dir: Path | None = None,
) -> MineResult:
    """Run package-1 mining and write a bounded artifact directory."""

    if backend is None:
        raise BackendError("mine requires a structured backend; use --backend replay or --backend openai")
    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    prompt_store = PromptStore(config.paths.forge_root / "prompts")
    patterns = mine_bundle(
        bundle=sources,
        config=config,
        structured_backend=backend,
        embedding_backend=embedder,
        schema_store=schema_store,
        prompt_store=prompt_store,
    )
    destination = (output_dir or config.paths.artifacts_root / f"forge_mine_{utc_stamp()}").resolve()
    destination.mkdir(parents=True, exist_ok=False)
    pattern_path = write_failure_patterns(patterns, destination / "failure_patterns.jsonl")
    inventory = {
        "schema_version": "forge-source-inventory.v2",
        "created_at": utc_now(),
        "source_bundle_digest": source_bundle_digest(sources),
        "analysis_backend": backend.backend_name,
        "analysis_model": backend.model_name,
        "embedding_model": embedder.model_name,
        "evidence_since": sources.evidence_since,
        "counts": {
            "transcripts": len(sources.transcripts),
            "promotion_verdicts": len(sources.verdicts),
            "plans": len(sources.plans),
            "bench_bundles": len(sources.bench_bundles),
            "explicit_failed_sources": len(_source_objects(sources)),
            "failure_patterns": len(patterns),
        },
    }
    atomic_write_json(destination / "source_inventory.json", inventory)
    prediction_checks = _prediction_checks(
        patterns,
        config.paths.ledger_path,
        evidence_since=sources.evidence_since,
    )
    atomic_write_json(
        destination / "prediction_checks.json",
        {
            "schema_version": "forge-prediction-checks.v2",
            "evidence_since": sources.evidence_since,
            "checks": prediction_checks,
        },
    )
    report_path = destination / "report.md"
    atomic_write_text(report_path, _render_mine_report(inventory, patterns, prediction_checks))
    return MineResult(
        output_dir=destination,
        pattern_path=pattern_path,
        report_path=report_path,
        pattern_count=len(patterns),
        in_surface_count=sum(pattern["surface_status"] == "in-surface" for pattern in patterns),
    )


def _prediction_checks(
    patterns: Sequence[Mapping[str, object]],
    ledger_path: Path,
    *,
    evidence_since: str | None,
) -> list[dict[str, object]]:
    if not ledger_path.exists():
        return []
    counts = {str(pattern["pattern_id"]): int(pattern["occurrence_count"]) for pattern in patterns}
    checks: list[dict[str, object]] = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid ledger JSON at {ledger_path}:{line_number}: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"ledger event must be an object at {ledger_path}:{line_number}")
        if event.get("event") != "proposal_decision" or event.get("decision") != "applied":
            continue
        prediction = event.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError(f"applied ledger event lacks prediction at {ledger_path}:{line_number}")
        pattern_id = prediction.get("pattern_id")
        baseline = prediction.get("baseline_value")
        expected_delta = prediction.get("expected_delta")
        if not isinstance(pattern_id, str) or not isinstance(baseline, int) or not isinstance(expected_delta, int):
            raise ValueError(f"invalid prediction fields at {ledger_path}:{line_number}")
        observed = counts.get(pattern_id, 0)
        applied_at = event.get("timestamp")
        if not isinstance(applied_at, str):
            raise ValueError(f"applied ledger event lacks timestamp at {ledger_path}:{line_number}")
        comparable = evidence_since is not None and _iso_timestamp(evidence_since) >= _iso_timestamp(applied_at)
        if not comparable:
            status = "inconclusive"
        else:
            status = "fulfilled" if observed <= max(0, baseline + expected_delta) else "refuted"
        checks.append(
            {
                "proposal_id": event.get("proposal_id"),
                "pattern_id": pattern_id,
                "baseline_value": baseline,
                "expected_delta": expected_delta,
                "observed_value": observed,
                "evidence_since": evidence_since,
                "applied_at": applied_at,
                "status": status,
            }
        )
    return checks


def _iso_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp {value!r}: {exc}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone: {value!r}")
    return parsed


def _render_mine_report(
    inventory: Mapping[str, object],
    patterns: Sequence[Mapping[str, object]],
    checks: Sequence[Mapping[str, object]],
) -> str:
    counts = inventory["counts"]
    assert isinstance(counts, dict)
    lines = [
        "# Forge Failure Mining Report",
        "",
        f"- Created: {inventory['created_at']}",
        f"- Analysis backend: `{inventory['analysis_backend']}` / `{inventory['analysis_model']}`",
        f"- Embedding model: `{inventory['embedding_model']}`",
        f"- Inputs: {counts['transcripts']} transcripts, {counts['promotion_verdicts']} verdicts, "
        f"{counts['plans']} plans, {counts['bench_bundles']} bench bundles",
        f"- Evidence since: `{inventory['evidence_since'] or 'unbounded'}`",
        f"- Explicit failed sources: {counts['explicit_failed_sources']}",
        f"- Failure patterns: {counts['failure_patterns']}",
        "",
        "## Failure patterns",
        "",
    ]
    for pattern in patterns:
        target = pattern["editable_target"] or "out-of-surface"
        lines.extend(
            (
                f"### {pattern['pattern_id']}",
                "",
                f"- Occurrences: {pattern['occurrence_count']}",
                f"- Surface: `{pattern['surface_status']}` → `{target}` "
                f"(similarity={float(pattern['surface_similarity']):.3f})",
                f"- Verifier cause: {pattern['verifier_cause']}",
                f"- Agent behavior: {pattern['agent_behavior_cause']}",
                f"- Mechanism: {pattern['exposed_mechanism']}",
                "",
            )
        )
    lines.extend(("## Prediction checks", ""))
    if checks:
        for check in checks:
            lines.append(
                f"- `{check['proposal_id']}` / `{check['pattern_id']}`: **{check['status']}** "
                f"(baseline={check['baseline_value']}, observed={check['observed_value']}, "
                f"expected_delta={check['expected_delta']})"
            )
    else:
        lines.append("- No previously applied proposal is awaiting longitudinal comparison.")
    lines.append("")
    return "\n".join(lines)
