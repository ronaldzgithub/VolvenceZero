"""Hard quality, provenance, leakage, and distribution audits."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .canonical import canonical_json, stable_hash, trajectory_from_json
from .contracts import (
    AnnotationSource,
    ExperienceTrajectory,
    GenerationTier,
    TrainingUse,
)
from .projections import ProjectionRecord, ProjectionView

AUDIT_SCHEMA_VERSION = "synthetic-audit.v1"

_EMAIL_RE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9])")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[\s-])?(?:\(\d{2,4}\)|\d{2,4})"
    r"[\s-]\d{3,4}[\s-]\d{4}(?!\d)"
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


@dataclass(frozen=True)
class AuditCheck:
    check_id: str
    passed: bool
    hard_gate: bool
    observed: int | float | str
    expected: str
    details_json: str

    def __post_init__(self) -> None:
        if not self.check_id.strip():
            raise ValueError("AuditCheck.check_id must be non-empty")
        try:
            json.loads(self.details_json)
        except json.JSONDecodeError as error:
            raise ValueError("AuditCheck.details_json must be valid JSON") from error


@dataclass(frozen=True)
class AuditReport:
    schema_version: str
    run_id: str
    trajectory_count: int
    passed: bool
    hard_failure_count: int
    checks: tuple[AuditCheck, ...]
    trajectory_hashes: tuple[str, ...]
    distribution_json: str

    def __post_init__(self) -> None:
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError("AuditReport.schema_version mismatch")
        if not self.run_id.strip():
            raise ValueError("AuditReport.run_id must be non-empty")
        if self.trajectory_count < 0:
            raise ValueError("AuditReport.trajectory_count must be non-negative")
        if self.hard_failure_count != sum(1 for check in self.checks if check.hard_gate and not check.passed):
            raise ValueError("AuditReport.hard_failure_count mismatch")
        if self.passed != (self.hard_failure_count == 0):
            raise ValueError("AuditReport.passed mismatch")
        try:
            json.loads(self.distribution_json)
        except json.JSONDecodeError as error:
            raise ValueError("AuditReport.distribution_json is invalid") from error


def audit_trajectories(
    trajectories: tuple[ExperienceTrajectory, ...],
    *,
    expected_count: int | None = None,
    expected_scenario_count: int = 96,
) -> AuditReport:
    if not trajectories:
        raise ValueError("cannot audit an empty trajectory set")
    run_ids = {item.provenance.run_id for item in trajectories}
    if len(run_ids) != 1:
        raise ValueError(f"audit input mixes run ids: {sorted(run_ids)}")
    checks: list[AuditCheck] = []
    checks.append(
        _check(
            "trajectory_count",
            passed=expected_count is None or len(trajectories) == expected_count,
            hard_gate=expected_count is not None,
            observed=len(trajectories),
            expected=("declared expected_count" if expected_count is not None else "informational"),
            details={"expected_count": expected_count},
        )
    )
    scenario_ids = {item.scenario_ref for item in trajectories}
    checks.append(
        _check(
            "scenario_coverage",
            passed=len(scenario_ids) == expected_scenario_count,
            hard_gate=True,
            observed=len(scenario_ids),
            expected=str(expected_scenario_count),
            details={"scenario_ids": sorted(scenario_ids)},
        )
    )

    hashes = tuple(stable_hash(item) for item in trajectories)
    round_trip_failures = []
    for trajectory, digest in zip(trajectories, hashes, strict=True):
        reconstructed = trajectory_from_json(canonical_json(trajectory))
        if reconstructed != trajectory or stable_hash(reconstructed) != digest:
            round_trip_failures.append(trajectory.trajectory_id)
    checks.append(
        _check(
            "schema_round_trip_hash",
            passed=not round_trip_failures,
            hard_gate=True,
            observed=len(round_trip_failures),
            expected="0 failures",
            details={"failures": round_trip_failures},
        )
    )

    duplicate_ids = _duplicates(item.trajectory_id for item in trajectories)
    duplicate_hashes = _duplicates(hashes)
    checks.append(
        _check(
            "trajectory_identity_uniqueness",
            passed=not duplicate_ids and not duplicate_hashes,
            hard_gate=True,
            observed=len(duplicate_ids) + len(duplicate_hashes),
            expected="0 duplicate ids and 0 duplicate full records",
            details={
                "duplicate_ids": duplicate_ids,
                "duplicate_hashes": duplicate_hashes,
            },
        )
    )

    lineage_violations = _annotation_lineage_violations(trajectories)
    checks.append(
        _check(
            "annotation_lineage_policy",
            passed=not lineage_violations,
            hard_gate=True,
            observed=len(lineage_violations),
            expected="0 source/training-use violations",
            details={"violations": lineage_violations},
        )
    )

    split_violations = _split_lineage_violations(trajectories)
    checks.append(
        _check(
            "split_lineage_zero_leakage",
            passed=not split_violations,
            hard_gate=True,
            observed=len(split_violations),
            expected="0 persona/latent-arc cross-split overlaps",
            details={"violations": split_violations},
        )
    )

    transcript_by_id = {item.trajectory_id: _transcript_text(item) for item in trajectories}
    exact_cross_split = _exact_cross_split_duplicates(
        trajectories,
        transcript_by_id=transcript_by_id,
    )
    checks.append(
        _check(
            "exact_transcript_cross_split",
            passed=not exact_cross_split,
            hard_gate=True,
            observed=len(exact_cross_split),
            expected="0 exact cross-split transcript duplicates",
            details={"pairs": exact_cross_split},
        )
    )

    rendered = tuple(item for item in trajectories if item.generation_tier is GenerationTier.RENDERED)
    near_cross_split = _near_cross_split_duplicates(
        rendered,
        transcript_by_id=transcript_by_id,
    )
    checks.append(
        _check(
            "near_duplicate_cross_split",
            passed=not near_cross_split,
            hard_gate=bool(rendered),
            observed=len(near_cross_split),
            expected="0 rendered cross-split pairs with similarity >= 0.94",
            details={"pairs": near_cross_split},
        )
    )

    pii_secret_hits = _pii_secret_hits(trajectories)
    checks.append(
        _check(
            "pii_secret_scan",
            passed=not pii_secret_hits,
            hard_gate=True,
            observed=len(pii_secret_hits),
            expected="0 high-confidence PII/secret patterns",
            details={"hits": pii_secret_hits},
        )
    )

    provenance_violations = _provenance_violations(trajectories)
    checks.append(
        _check(
            "heldout_copyright_source_isolation",
            passed=not provenance_violations,
            hard_gate=True,
            observed=len(provenance_violations),
            expected="fully synthetic proprietary provenance only",
            details={"violations": provenance_violations},
        )
    )

    rendered_slot_violations = _rendered_slot_violations(rendered)
    checks.append(
        _check(
            "rendered_text_slot_integrity",
            passed=not rendered_slot_violations,
            hard_gate=bool(rendered),
            observed=len(rendered_slot_violations),
            expected="no structural placeholders in rendered trajectories",
            details={"violations": rendered_slot_violations},
        )
    )

    live = tuple(item for item in trajectories if item.generation_tier is GenerationTier.LIVE_THROUGH)
    snapshot_violations = _snapshot_lineage_violations(live)
    checks.append(
        _check(
            "runtime_snapshot_lineage",
            passed=not snapshot_violations,
            hard_gate=bool(live),
            observed=len(snapshot_violations),
            expected="all live turns reference hash-valid public snapshots",
            details={"violations": snapshot_violations},
        )
    )

    distribution = _distribution(trajectories)
    distribution_violations = _distribution_violations(distribution)
    checks.append(
        _check(
            "distribution_coverage",
            passed=not distribution_violations,
            hard_gate=True,
            observed=len(distribution_violations),
            expected="all splits/languages/families represented",
            details={"violations": distribution_violations},
        )
    )

    hard_failure_count = sum(1 for check in checks if check.hard_gate and not check.passed)
    return AuditReport(
        schema_version=AUDIT_SCHEMA_VERSION,
        run_id=next(iter(run_ids)),
        trajectory_count=len(trajectories),
        passed=hard_failure_count == 0,
        hard_failure_count=hard_failure_count,
        checks=tuple(checks),
        trajectory_hashes=tuple(sorted(hashes)),
        distribution_json=canonical_json(distribution),
    )


def audit_projection_records(
    records: tuple[ProjectionRecord, ...],
) -> tuple[AuditCheck, ...]:
    leakage: list[str] = []
    for record in records:
        if (
            record.view
            in {
                ProjectionView.EVALUATION_ONLY,
                ProjectionView.HUMAN_REVIEW_QUEUE,
            }
            and record.training_use is TrainingUse.TARGET
        ):
            leakage.append(record.record_id)
        payload = json.loads(record.payload_json)
        source = payload.get("source") if isinstance(payload, dict) else None
        if (
            source
            in {
                AnnotationSource.MODEL_PREDICTION.value,
                AnnotationSource.EVALUATION_READOUT.value,
            }
            and record.training_use is TrainingUse.TARGET
        ):
            leakage.append(record.record_id)
    hash_failures = [record.record_id for record in records if len(record.master_trajectory_hash) != 64]
    return (
        _check(
            "projection_eval_training_isolation",
            passed=not leakage,
            hard_gate=True,
            observed=len(leakage),
            expected="0 eval/model records used as targets",
            details={"violations": leakage},
        ),
        _check(
            "projection_master_hash_lineage",
            passed=not hash_failures,
            hard_gate=True,
            observed=len(hash_failures),
            expected="all records cite a master SHA-256",
            details={"violations": hash_failures},
        ),
    )


def write_audit_bundle(
    report: AuditReport,
    trajectories: tuple[ExperienceTrajectory, ...],
    *,
    output_dir: Path,
    actual_cost_usd: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "audit-report.json"
    report_path.write_text(f"{canonical_json(report)}\n", encoding="utf-8")
    distribution = json.loads(report.distribution_json)

    dataset_card = output_dir / "dataset-card.md"
    dataset_card.write_text(
        "\n".join(
            (
                "# Unified Synthetic Experience Corpus v1",
                "",
                f"- Run: `{report.run_id}`",
                f"- Trajectories: {report.trajectory_count}",
                f"- Hard-gate status: {'PASS' if report.passed else 'FAIL'}",
                "- Origin: fully original synthetic world/FSM states; no held-out benchmark loading.",
                "- Evidence layers: generator truth, rendered text, and runtime observations are separate.",
                (
                    "- Intended uses: relationship encoder, expression SFT, "
                    "semantic/social owners, memory retrieval, temporal SSL, "
                    "internal RL, evaluation."
                ),
                (
                    "- Prohibited uses: treating model/judge output as hard truth; "
                    "claiming real-user consent or human annotation."
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    field_dictionary = output_dir / "field-dictionary.md"
    field_dictionary.write_text(
        "\n".join(
            (
                "# Field Dictionary",
                "",
                "- `truth_frames`: generator-owned observable/private world state and response contract.",
                "- `sessions[].turns[].text`: structural or LLM-rendered expression only.",
                "- `snapshot_frames`: immutable public runtime observations with payload hash.",
                "- `annotations`: ontology/version/source/training-use/evidence lineage.",
                "- `artifacts`: content-addressed external references; no embedded mutable objects.",
                "- `quality`: hard/readout checks; evaluation remains downstream.",
                "",
            )
        ),
        encoding="utf-8",
    )
    handbook = output_dir / "annotation-handbook.md"
    handbook.write_text(
        "\n".join(
            (
                "# Annotation Handbook",
                "",
                "1. `generator_truth` and `environment_fact` may be targets when their evidence refs resolve.",
                "2. `runtime_snapshot` is feature-only unless a task explicitly learns a runtime readout.",
                "3. `model_prediction` and `evaluation_readout` are never targets.",
                "4. `human_annotation` requires a real annotator id and evidence; never synthesize it.",
                "5. Keep owner, track, timescale, scope and adjudication explicit.",
                "6. Never assign hand-authored semantics to `z_t` or `beta_t`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    coverage_path = output_dir / "coverage-matrix.json"
    coverage_path.write_text(
        json.dumps(distribution, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cost_path = output_dir / "cost-model-report.json"
    model_counts = Counter(item.provenance.model_id or "none" for item in trajectories)
    cost_path.write_text(
        json.dumps(
            {
                "actual_cost_usd": actual_cost_usd,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model_distribution": dict(sorted(model_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        report_path,
        dataset_card,
        field_dictionary,
        handbook,
        coverage_path,
        cost_path,
    )


def _annotation_lineage_violations(
    trajectories: tuple[ExperienceTrajectory, ...],
) -> list[str]:
    violations: list[str] = []
    for trajectory in trajectories:
        frame_ids = {frame.frame_id for frame in trajectory.truth_frames}
        snapshot_ids = {frame.snapshot_id for frame in trajectory.snapshot_frames}
        artifact_ids = {item.artifact_id for item in trajectory.artifacts}
        evidence_ids = frame_ids | snapshot_ids | artifact_ids
        for annotation in trajectory.annotations:
            if (
                annotation.source
                in {
                    AnnotationSource.MODEL_PREDICTION,
                    AnnotationSource.EVALUATION_READOUT,
                }
                and annotation.training_use is TrainingUse.TARGET
            ):
                violations.append(annotation.annotation_id + ":model-as-target")
            if annotation.source is AnnotationSource.HUMAN_ANNOTATION:
                if annotation.annotator_id is None:
                    violations.append(annotation.annotation_id + ":missing-human-id")
            if not set(annotation.evidence_refs).issubset(evidence_ids):
                violations.append(annotation.annotation_id + ":unresolved-evidence")
    return violations


def _split_lineage_violations(
    trajectories: tuple[ExperienceTrajectory, ...],
) -> list[str]:
    by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for trajectory in trajectories:
        metadata = {item.key: item.value for item in trajectory.metadata}
        for key in ("persona_id", "latent_arc_id"):
            value = metadata.get(key)
            if value is None:
                by_key[(key, "<missing>")].add(trajectory.split.value)
            else:
                by_key[(key, value)].add(trajectory.split.value)
    return [
        f"{key}:{value}:{sorted(splits)}"
        for (key, value), splits in sorted(by_key.items())
        if len(splits) > 1 or value == "<missing>"
    ]


def _exact_cross_split_duplicates(
    trajectories: tuple[ExperienceTrajectory, ...],
    *,
    transcript_by_id: dict[str, str],
) -> list[dict[str, str]]:
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[dict[str, str]] = []
    for trajectory in trajectories:
        digest = hashlib.sha256(transcript_by_id[trajectory.trajectory_id].encode("utf-8")).hexdigest()
        previous = seen.get(digest)
        if previous is not None and previous[1] != trajectory.split.value:
            duplicates.append(
                {
                    "left": previous[0],
                    "right": trajectory.trajectory_id,
                    "hash": digest,
                }
            )
        else:
            seen[digest] = (trajectory.trajectory_id, trajectory.split.value)
    return duplicates


def _near_cross_split_duplicates(
    trajectories: tuple[ExperienceTrajectory, ...],
    *,
    transcript_by_id: dict[str, str],
) -> list[dict[str, object]]:
    buckets: dict[int, list[tuple[ExperienceTrajectory, int]]] = defaultdict(list)
    for trajectory in trajectories:
        signature = _simhash(transcript_by_id[trajectory.trajectory_id])
        buckets[signature >> 48].append((trajectory, signature))
    duplicates: list[dict[str, object]] = []
    for entries in buckets.values():
        for left_index, (left, left_signature) in enumerate(entries):
            for right, right_signature in entries[left_index + 1 :]:
                if left.split is right.split:
                    continue
                similarity = 1.0 - ((left_signature ^ right_signature).bit_count() / 64.0)
                if similarity >= 0.94:
                    duplicates.append(
                        {
                            "left": left.trajectory_id,
                            "right": right.trajectory_id,
                            "similarity": round(similarity, 6),
                        }
                    )
    return duplicates


def _pii_secret_hits(
    trajectories: tuple[ExperienceTrajectory, ...],
) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for trajectory in trajectories:
        text = _transcript_text(trajectory)
        for kind, pattern in (
            ("email", _EMAIL_RE),
            ("phone", _PHONE_RE),
            *((f"secret_{index}", pattern) for index, pattern in enumerate(_SECRET_PATTERNS)),
        ):
            match = pattern.search(text)
            if match is not None:
                hits.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "kind": kind,
                        "digest": hashlib.sha256(match.group(0).encode("utf-8")).hexdigest(),
                    }
                )
    return hits


def _provenance_violations(
    trajectories: tuple[ExperienceTrajectory, ...],
) -> list[str]:
    violations: list[str] = []
    for trajectory in trajectories:
        provenance = trajectory.provenance
        if provenance.consent_basis != "fully_synthetic":
            violations.append(trajectory.trajectory_id + ":consent-basis")
        if provenance.license_id != "Proprietary-Synthetic-v1":
            violations.append(trajectory.trajectory_id + ":license")
        if provenance.source_kind not in {
            "synthetic_world_fsm",
            "synthetic_world_fsm_plus_llm_render",
            "synthetic_world_fsm_plus_live_through",
            "synthetic_world_fsm_plus_llm_render_plus_live_through",
        }:
            violations.append(trajectory.trajectory_id + ":source-kind")
        if any(
            token in trajectory.scenario_ref.casefold() for token in ("heldout", "held-out", "companionbench-private")
        ):
            violations.append(trajectory.trajectory_id + ":heldout-ref")
    return violations


def _rendered_slot_violations(
    trajectories: tuple[ExperienceTrajectory, ...],
) -> list[str]:
    violations: list[str] = []
    for trajectory in trajectories:
        for session in trajectory.sessions:
            for turn in session.turns:
                if turn.text.startswith("[STRUCTURAL SLOT") or turn.text.startswith("【结构槽"):
                    violations.append(turn.turn_id)
                if turn.text.startswith("[ASSISTANT SLOT") or turn.text.startswith("【待渲染助手槽"):
                    violations.append(turn.turn_id)
    return violations


def _snapshot_lineage_violations(
    trajectories: tuple[ExperienceTrajectory, ...],
) -> list[str]:
    violations: list[str] = []
    for trajectory in trajectories:
        snapshots = {frame.snapshot_id: frame for frame in trajectory.snapshot_frames}
        for frame in trajectory.snapshot_frames:
            payload = json.loads(frame.payload_json)
            if stable_hash(payload) != frame.value_hash:
                violations.append(frame.snapshot_id + ":hash")
        for session in trajectory.sessions:
            for turn in session.turns:
                for ref in turn.snapshot_refs:
                    if ref not in snapshots:
                        violations.append(turn.turn_id + ":missing:" + ref)
    return violations


def _distribution(
    trajectories: tuple[ExperienceTrajectory, ...],
) -> dict[str, object]:
    turn_lengths = [
        len(turn.text) for trajectory in trajectories for session in trajectory.sessions for turn in session.turns
    ]
    label_counts = Counter(annotation.label_key for trajectory in trajectories for annotation in trajectory.annotations)
    return {
        "split": dict(sorted(Counter(item.split.value for item in trajectories).items())),
        "language": dict(sorted(Counter(item.language for item in trajectories).items())),
        "family": dict(sorted(Counter(item.family for item in trajectories).items())),
        "generation_tier": dict(sorted(Counter(item.generation_tier.value for item in trajectories).items())),
        "annotation_label": dict(sorted(label_counts.items())),
        "turn_length": {
            "min": min(turn_lengths),
            "max": max(turn_lengths),
            "mean": sum(turn_lengths) / len(turn_lengths),
            "p95": _percentile(turn_lengths, 0.95),
        },
    }


def _distribution_violations(distribution: dict[str, object]) -> list[str]:
    violations: list[str] = []
    split = distribution["split"]
    language = distribution["language"]
    family = distribution["family"]
    if not isinstance(split, dict) or set(split) != {"train", "val", "test"}:
        violations.append("split coverage")
    if not isinstance(language, dict) or set(language) != {
        "zh",
        "en",
        "bilingual",
    }:
        violations.append("language coverage")
    if not isinstance(family, dict) or len(family) != 16:
        violations.append("family coverage")
    return violations


def _simhash(text: str) -> int:
    tokens = tuple(_normalized_tokens(text))
    shingles = (
        tuple(" ".join(tokens[index : index + 5]) for index in range(len(tokens) - 4)) if len(tokens) >= 5 else tokens
    )
    weights = [0] * 64
    for shingle in shingles:
        digest = int.from_bytes(
            hashlib.sha256(shingle.encode("utf-8")).digest()[:8],
            "big",
        )
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    signature = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << bit
    return signature


def _normalized_tokens(text: str):
    token: list[str] = []
    for character in text.casefold():
        if character.isalnum():
            token.append(character)
        elif token:
            yield "".join(token)
            token = []
    if token:
        yield "".join(token)


def _transcript_text(trajectory: ExperienceTrajectory) -> str:
    return "\n".join(turn.text for session in trajectory.sessions for turn in session.turns)


def _duplicates(values) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def _check(
    check_id: str,
    *,
    passed: bool,
    hard_gate: bool,
    observed: int | float | str,
    expected: str,
    details: object,
) -> AuditCheck:
    return AuditCheck(
        check_id=check_id,
        passed=passed,
        hard_gate=hard_gate,
        observed=observed,
        expected=expected,
        details_json=canonical_json(details),
    )


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditCheck",
    "AuditReport",
    "audit_projection_records",
    "audit_trajectories",
    "write_audit_bundle",
]
