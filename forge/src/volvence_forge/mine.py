"""Semantic, schema-bound failure mining for Forge.

The module requires injected structured-output and embedding backends. There is
no keyword or hash fallback: unavailable semantic infrastructure is an
observable failure, not a hidden second decision owner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ForgeConfig
from .foundation import (
    EmbeddingBackend,
    PromptStore,
    SchemaStore,
    StructuredBackend,
    canonical_json,
    normalized,
)
from .sources import (
    EvidenceRef,
    PlanSource,
    SourceBundle,
    TranscriptSource,
    VerdictSource,
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


def _analysis_text(analysis: FailureAnalysis) -> str:
    return " ".join(
        (
            analysis.verifier_cause,
            analysis.agent_behavior_cause,
            analysis.exposed_mechanism,
        )
    )


def _source_objects(bundle: SourceBundle) -> tuple[TranscriptSource | VerdictSource | PlanSource, ...]:
    return bundle.transcripts + bundle.verdicts + bundle.plans


def _fallback_ref(source: TranscriptSource | VerdictSource | PlanSource) -> EvidenceRef:
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


def _evidence_refs(source: TranscriptSource | VerdictSource | PlanSource) -> tuple[EvidenceRef, ...]:
    if isinstance(source, TranscriptSource) and source.error_refs:
        return source.error_refs
    if isinstance(source, VerdictSource) and source.failed_gate_refs:
        return source.failed_gate_refs
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
    for source in _source_objects(bundle):
        source_record = source.analysis_record()
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

    surface_entries = config.editable
    surface_texts = [entry.semantic_description for entry in surface_entries]
    patterns: list[dict[str, object]] = []
    for cluster in clusters:
        items = cluster["items"]
        assert isinstance(items, list) and items
        analyses = [item[1] for item in items]
        sources = [item[0] for item in items]
        centroid = cluster["centroid"]
        assert isinstance(centroid, np.ndarray)
        surface_vectors = _validate_embedding_matrix(
            embedding_backend.encode(surface_texts),
            expected_rows=len(surface_entries),
        )
        scores = surface_vectors @ centroid
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        if best_score >= config.minimum_surface_similarity:
            target = surface_entries[best_index]
            assets = [
                relative
                for entry, relative, _path in config.editable_assets()
                if entry.component == target.component
            ]
            if assets:
                editable_target = assets[0]
                editable_component = target.component
                surface_status = "in-surface"
            else:
                editable_target = None
                editable_component = None
                surface_status = "out-of-surface"
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
            "schema_version": "forge-failure-pattern.v1",
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
