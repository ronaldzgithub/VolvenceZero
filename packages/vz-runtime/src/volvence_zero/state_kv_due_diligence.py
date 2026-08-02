"""Frozen State KV P6 evidence manifest and seven-conclusion report."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

FREEZE_MANIFEST_SCHEMA_VERSION = "state-kv-freeze-manifest.v2"
DUE_DILIGENCE_SCHEMA_VERSION = "state-kv-due-diligence.v1"


@dataclass(frozen=True)
class FrozenEvidenceArtifact:
    evidence_id: str
    path: str
    sha256: str
    schema_version: str
    observed_state: str


@dataclass(frozen=True)
class StateKVFreezeManifest:
    freeze_id: str
    model_id: str
    model_weights_sha256: str
    prefix_manifest_path: str
    prefix_manifest_sha256: str
    prefix_artifact_id: str
    experiment_config_json: str
    experiment_config_sha256: str
    profile_labels: tuple[str, ...]
    generation_seeds: tuple[int, ...]
    scenario_sets: tuple[str, ...]
    metric_definitions: tuple[str, ...]
    judge_panel: tuple[str, ...]
    evidence: tuple[FrozenEvidenceArtifact, ...]

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
            "freeze_id": self.freeze_id,
            "model": {
                "model_id": self.model_id,
                "weights_sha256": self.model_weights_sha256,
            },
            "prefix_manifest": {
                "path": self.prefix_manifest_path,
                "sha256": self.prefix_manifest_sha256,
            },
            "prefix_artifact_id": self.prefix_artifact_id,
            "experiment_config": json.loads(self.experiment_config_json),
            "experiment_config_sha256": self.experiment_config_sha256,
            "profile_labels": list(self.profile_labels),
            "generation_seeds": list(self.generation_seeds),
            "scenario_sets": list(self.scenario_sets),
            "metric_definitions": list(self.metric_definitions),
            "judge_panel": list(self.judge_panel),
            "evidence": [
                {
                    "evidence_id": artifact.evidence_id,
                    "path": artifact.path,
                    "sha256": artifact.sha256,
                    "schema_version": artifact.schema_version,
                    "observed_state": artifact.observed_state,
                }
                for artifact in self.evidence
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


@dataclass(frozen=True)
class DueDiligenceConclusion:
    conclusion_id: str
    statement: str
    state: str
    evidence_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class StateKVDueDiligenceReport:
    freeze_id: str
    conclusions: tuple[DueDiligenceConclusion, ...]
    evidence_fingerprints: tuple[tuple[str, str], ...]

    @property
    def gate_state(self) -> str:
        return (
            "complete"
            if all(item.state == "proven" for item in self.conclusions)
            else "partial"
        )

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": DUE_DILIGENCE_SCHEMA_VERSION,
            "gate_state": self.gate_state,
            "freeze_id": self.freeze_id,
            "conclusions": [
                {
                    "conclusion_id": item.conclusion_id,
                    "statement": item.statement,
                    "state": item.state,
                    "evidence_ids": list(item.evidence_ids),
                    "detail": item.detail,
                }
                for item in self.conclusions
            ],
            "evidence_fingerprints": dict(self.evidence_fingerprints),
            "summary": {
                "proven": sum(
                    item.state == "proven" for item in self.conclusions
                ),
                "not_yet_proven": sum(
                    item.state == "not-yet-proven"
                    for item in self.conclusions
                ),
            },
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observed_state(payload: Mapping[str, object]) -> str:
    for key in (
        "gate_state",
        "verdict_state",
        "court_state",
        "p5d_decision",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return "reported"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_freeze_payload(
    *,
    model_id: str,
    model_weights_sha256: str,
    prefix_manifest_path: str,
    prefix_manifest_sha256: str,
    prefix_artifact_id: str,
    experiment_config_json: str,
    experiment_config_sha256: str,
    profile_labels: Sequence[str],
    generation_seeds: Sequence[int],
    scenario_sets: Sequence[str],
    metric_definitions: Sequence[str],
    judge_panel: Sequence[str],
    evidence: Sequence[FrozenEvidenceArtifact],
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_weights_sha256": model_weights_sha256,
        "prefix_manifest_path": prefix_manifest_path,
        "prefix_manifest_sha256": prefix_manifest_sha256,
        "prefix_artifact_id": prefix_artifact_id,
        "experiment_config": json.loads(experiment_config_json),
        "experiment_config_sha256": experiment_config_sha256,
        "profile_labels": list(profile_labels),
        "generation_seeds": list(generation_seeds),
        "scenario_sets": list(scenario_sets),
        "metric_definitions": list(metric_definitions),
        "judge_panel": list(judge_panel),
        "evidence": [
            (
                item.evidence_id,
                item.path,
                item.sha256,
                item.schema_version,
                item.observed_state,
            )
            for item in evidence
        ],
    }


def build_freeze_manifest(
    *,
    repo_root: Path,
    prefix_manifest_path: str,
    evidence_paths: Mapping[str, str],
    profile_labels: Sequence[str],
    generation_seeds: Sequence[int],
    scenario_sets: Sequence[str],
    metric_definitions: Sequence[str],
    judge_panel: Sequence[str],
    experiment_config: Mapping[str, object],
) -> StateKVFreezeManifest:
    prefix_path = repo_root / prefix_manifest_path
    prefix_payload = json.loads(prefix_path.read_text(encoding="utf-8"))
    model_id = str(prefix_payload["model_id"])
    model_weights_sha256 = str(prefix_payload["weights_sha256"])
    prefix_artifact_id = str(prefix_payload["artifact_id"])
    prefix_manifest_sha256 = _sha256(prefix_path)
    experiment_config_json = _canonical_json(experiment_config)
    experiment_config_sha256 = _sha256_text(experiment_config_json)

    evidence: list[FrozenEvidenceArtifact] = []
    for evidence_id, relative_path in sorted(evidence_paths.items()):
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"freeze evidence {evidence_id!r} is missing: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence.append(
            FrozenEvidenceArtifact(
                evidence_id=evidence_id,
                path=relative_path,
                sha256=_sha256(path),
                schema_version=str(payload.get("schema_version", "")),
                observed_state=_observed_state(payload),
            )
        )
    canonical = _canonical_freeze_payload(
        model_id=model_id,
        model_weights_sha256=model_weights_sha256,
        prefix_manifest_path=prefix_manifest_path,
        prefix_manifest_sha256=prefix_manifest_sha256,
        prefix_artifact_id=prefix_artifact_id,
        experiment_config_json=experiment_config_json,
        experiment_config_sha256=experiment_config_sha256,
        profile_labels=profile_labels,
        generation_seeds=generation_seeds,
        scenario_sets=scenario_sets,
        metric_definitions=metric_definitions,
        judge_panel=judge_panel,
        evidence=evidence,
    )
    freeze_id = _sha256_text(_canonical_json(canonical))
    manifest = StateKVFreezeManifest(
        freeze_id=freeze_id,
        model_id=model_id,
        model_weights_sha256=model_weights_sha256,
        prefix_manifest_path=prefix_manifest_path,
        prefix_manifest_sha256=prefix_manifest_sha256,
        prefix_artifact_id=prefix_artifact_id,
        experiment_config_json=experiment_config_json,
        experiment_config_sha256=experiment_config_sha256,
        profile_labels=tuple(profile_labels),
        generation_seeds=tuple(generation_seeds),
        scenario_sets=tuple(scenario_sets),
        metric_definitions=tuple(metric_definitions),
        judge_panel=tuple(judge_panel),
        evidence=tuple(evidence),
    )
    validate_freeze_manifest(manifest)
    return manifest


def validate_freeze_manifest(manifest: StateKVFreezeManifest) -> None:
    hex_digits = frozenset("0123456789abcdef")
    if (
        len(manifest.freeze_id) != 64
        or set(manifest.freeze_id) - hex_digits
    ):
        raise ValueError("freeze_id must be a SHA-256 hex digest")
    if (
        not manifest.model_id
        or len(manifest.model_weights_sha256) != 64
        or set(manifest.model_weights_sha256) - hex_digits
    ):
        raise ValueError("freeze manifest requires a model id and weight SHA-256")
    if (
        not manifest.prefix_manifest_path
        or len(manifest.prefix_manifest_sha256) != 64
        or set(manifest.prefix_manifest_sha256) - hex_digits
    ):
        raise ValueError(
            "freeze manifest requires a prefix manifest path and SHA-256"
        )
    if not manifest.prefix_artifact_id:
        raise ValueError("freeze manifest requires prefix_artifact_id")
    expected_config_sha256 = _sha256_text(manifest.experiment_config_json)
    if manifest.experiment_config_sha256 != expected_config_sha256:
        raise ValueError("freeze manifest experiment config SHA-256 mismatch")
    try:
        config_payload = json.loads(manifest.experiment_config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("freeze manifest experiment config is invalid JSON") from exc
    if not isinstance(config_payload, Mapping) or not config_payload:
        raise ValueError("freeze manifest experiment config must be a non-empty object")
    collections = (
        ("profile_labels", manifest.profile_labels),
        ("generation_seeds", manifest.generation_seeds),
        ("scenario_sets", manifest.scenario_sets),
        ("metric_definitions", manifest.metric_definitions),
        ("judge_panel", manifest.judge_panel),
    )
    for name, values in collections:
        if not values or len(values) != len(set(values)):
            raise ValueError(
                f"freeze manifest {name} must be non-empty and unique"
            )
    evidence_ids = tuple(item.evidence_id for item in manifest.evidence)
    if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("freeze manifest evidence ids must be non-empty and unique")
    for artifact in manifest.evidence:
        if (
            not artifact.schema_version
            or len(artifact.sha256) != 64
            or set(artifact.sha256) - hex_digits
        ):
            raise ValueError(
                f"invalid frozen evidence record {artifact.evidence_id!r}"
            )
    canonical = _canonical_freeze_payload(
        model_id=manifest.model_id,
        model_weights_sha256=manifest.model_weights_sha256,
        prefix_manifest_path=manifest.prefix_manifest_path,
        prefix_manifest_sha256=manifest.prefix_manifest_sha256,
        prefix_artifact_id=manifest.prefix_artifact_id,
        experiment_config_json=manifest.experiment_config_json,
        experiment_config_sha256=manifest.experiment_config_sha256,
        profile_labels=manifest.profile_labels,
        generation_seeds=manifest.generation_seeds,
        scenario_sets=manifest.scenario_sets,
        metric_definitions=manifest.metric_definitions,
        judge_panel=manifest.judge_panel,
        evidence=manifest.evidence,
    )
    expected_freeze_id = _sha256_text(_canonical_json(canonical))
    if manifest.freeze_id != expected_freeze_id:
        raise ValueError(
            "freeze_id does not match the canonical manifest payload"
        )


def verify_frozen_evidence(
    *,
    repo_root: Path,
    manifest: StateKVFreezeManifest,
) -> dict[str, Mapping[str, object]]:
    validate_freeze_manifest(manifest)
    prefix_path = repo_root / manifest.prefix_manifest_path
    actual_prefix_sha256 = _sha256(prefix_path)
    if actual_prefix_sha256 != manifest.prefix_manifest_sha256:
        raise ValueError(
            "frozen prefix manifest changed: "
            f"expected {manifest.prefix_manifest_sha256}, "
            f"got {actual_prefix_sha256}"
        )
    payloads: dict[str, Mapping[str, object]] = {}
    for artifact in manifest.evidence:
        path = repo_root / artifact.path
        actual = _sha256(path)
        if actual != artifact.sha256:
            raise ValueError(
                f"frozen evidence changed for {artifact.evidence_id!r}: "
                f"expected {artifact.sha256}, got {actual}"
            )
        payloads[artifact.evidence_id] = json.loads(
            path.read_text(encoding="utf-8")
        )
    return payloads


def _gate_pass(payloads: Mapping[str, Mapping[str, object]], key: str) -> bool:
    payload = payloads[key]
    return (
        payload.get("gate_state") == "pass"
        or payload.get("court_state") == "pass"
        or str(payload.get("verdict_state", "")).startswith("retain-")
    )


def _claim_pass(
    payloads: Mapping[str, Mapping[str, object]],
    key: str,
    claim_name: str,
) -> bool:
    claims = payloads[key].get("claims", ())
    return any(
        isinstance(claim, Mapping)
        and claim.get("claim") == claim_name
        and claim.get("state") == "pass"
        for claim in claims
    )


def _claim_state(
    payloads: Mapping[str, Mapping[str, object]],
    key: str,
    claim_name: str,
) -> str:
    claims = payloads[key].get("claims", ())
    for claim in claims:
        if (
            isinstance(claim, Mapping)
            and claim.get("claim") == claim_name
        ):
            return str(claim.get("state", ""))
    return ""


def _five_arm_prefix_beats_residual(
    payloads: Mapping[str, Mapping[str, object]],
) -> bool:
    """Require the frozen P3 five-arm verdict, not only a live P4 carrier.

    C3 is comparative: the Prefix-KV arm must retain identification while
    the matched pure residual arm remains statistically compatible with
    chance.  A P4 attention diagnostic alone only proves that the carrier is
    live; it cannot establish increment over residual projection.
    """

    payload = payloads["five_arm_identification"]
    if (
        payload.get("verdict_state") != "retain-strict"
        or payload.get("candidate_arm") != "state-kv-arm-g-prefix-pure"
    ):
        return False
    matching = payload.get("matching", ())
    if not isinstance(matching, Sequence) or isinstance(matching, (str, bytes)):
        return False
    for item in matching:
        if not isinstance(item, Mapping):
            continue
        if item.get("arm") != "state-kv-arm-e-pure":
            continue
        ci_low = item.get("ci_low")
        ci_high = item.get("ci_high")
        return (
            isinstance(ci_low, (int, float))
            and isinstance(ci_high, (int, float))
            and float(ci_low) <= 0.5 <= float(ci_high)
        )
    return False


def _carrier_matches_frozen_prefix(
    *,
    manifest: StateKVFreezeManifest,
    payloads: Mapping[str, Mapping[str, object]],
) -> bool:
    payload = payloads["carrier_diagnostics"]
    return (
        payload.get("carrier_is_live") is True
        and payload.get("prefix_artifact_id") == manifest.prefix_artifact_id
        and _claim_pass(
            payloads,
            "carrier_diagnostics",
            "claim_slot_attention_read",
        )
    )


def build_due_diligence_report(
    *,
    repo_root: Path,
    manifest: StateKVFreezeManifest,
) -> StateKVDueDiligenceReport:
    """Map design §11.3 conclusions to frozen evidence without overclaiming."""

    payloads = verify_frozen_evidence(
        repo_root=repo_root,
        manifest=manifest,
    )
    conclusion_2 = (
        _gate_pass(payloads, "retention")
        and _gate_pass(payloads, "cost")
        and _gate_pass(payloads, "judge_court")
        and _claim_pass(
            payloads,
            "quality_noninferiority",
            "claim_quality_noninferior_to_bprime",
        )
    )
    conclusion_3 = (
        conclusion_2
        and _carrier_matches_frozen_prefix(
            manifest=manifest,
            payloads=payloads,
        )
        and _five_arm_prefix_beats_residual(payloads)
    )
    conclusion_4 = _gate_pass(payloads, "control_dim")
    control_decision = str(
        payloads["control_dim"].get("p5d_decision", "unreported")
    )
    credit_mechanism = _claim_pass(
        payloads,
        "credit_longitudinal",
        "claim_credit_feedback_applied_increment_grows",
    )
    credit_outcome_state = _claim_state(
        payloads,
        "credit_longitudinal",
        "claim_credit_feedback_improves_matched_outcome",
    )
    conclusion_5 = credit_mechanism and credit_outcome_state == "pass"
    conclusion_6 = (
        _gate_pass(payloads, "deployment")
        and _gate_pass(payloads, "generation_seed")
        and _gate_pass(payloads, "safety_negatives")
        and _claim_pass(
            payloads,
            "safety_negatives",
            "claim_stale_conditioning_is_inert",
        )
        and _claim_pass(
            payloads,
            "safety_negatives",
            "claim_latent_state_resists_output_extraction",
        )
    )
    conclusions = (
        DueDiligenceConclusion(
            conclusion_id="C1",
            statement="正确状态相对 prompt/RAG 有稳定增益",
            state="not-yet-proven",
            evidence_ids=("retention",),
            detail=(
                "B/C/D（人工 prompt、RAG、matched LoRA）未按同预算实现；"
                "现有 B-prime 不能替代全部 prompt/RAG 对照。"
            ),
        ),
        DueDiligenceConclusion(
            conclusion_id="C2",
            statement="潜状态相对同信息量文本质量不劣且成本占优",
            state="proven" if conclusion_2 else "not-yet-proven",
            evidence_ids=(
                "identification",
                "retention",
                "judge_court",
                "quality_noninferiority",
                "cost",
            ),
            detail=(
                "Prefix-KV 在冻结同基底、多场景、多 seed、多裁判的"
                "G-vs-B-prime 配对非劣效门通过，且成本与延迟门通过。"
                if conclusion_2
                else "质量或成本的冻结证据门未全部通过。"
            ),
        ),
        DueDiligenceConclusion(
            conclusion_id="C3",
            statement="State KV 相对残差投影有增量且不退化为偏置",
            state="proven" if conclusion_3 else "not-yet-proven",
            evidence_ids=(
                "five_arm_identification",
                "carrier_diagnostics",
                "temporal_causal",
            ),
            detail=(
                "冻结五臂中 G-prefix retain-strict、E-pure 仍与随机相容，"
                "且同一标准 Prefix artifact 的 slot-attention 非退化门通过。"
                if conclusion_3
                else (
                    "标准 artifact 的 live slot-attention 与五臂 Prefix-vs-"
                    "residual 对照未同时通过；线性可读出或单臂 retain 不能"
                    "替代相对残差增量证据。"
                )
            ),
        ),
        DueDiligenceConclusion(
            conclusion_id="C4",
            statement="扩展动态残差相对已有 3 维控制有增量",
            state="proven" if conclusion_4 else "not-yet-proven",
            evidence_ids=("control_dim",),
            detail=(
                "Matched full-dimension、rank-3 与 dynamic-off 三臂通过；"
                "允许控制基 artifact 进入后续 OFFLINE ModificationGate。"
                if conclusion_4
                else (
                    "D0 已完成 matched full-dimension、rank-3 与 "
                    f"dynamic-off 三臂，结论为 {control_decision}；"
                    "候选 artifact 未获准进入生产，保留 rank-3。"
                )
            ),
        ),
        DueDiligenceConclusion(
            conclusion_id="C5",
            statement="Prediction Error 闭环相对静态组合有增量",
            state="proven" if conclusion_5 else "not-yet-proven",
            evidence_ids=("bank_gain", "credit_longitudinal"),
            detail=(
                "J 对 I 的长 session 增量与 matched outcome 门均通过。"
                if conclusion_5
                else (
                    "I/J 长 session 已有冻结观测窗口；"
                    f"mechanism_pass={credit_mechanism}，"
                    f"matched_outcome={credit_outcome_state or 'missing'}。"
                    "响应分叉不能替代质量/真实结局增量。"
                )
            ),
        ),
        DueDiligenceConclusion(
            conclusion_id="C6",
            statement="错用户/过期/撤销无残留且潜状态不可抽取",
            state="proven" if conclusion_6 else "not-yet-proven",
            evidence_ids=(
                "deployment",
                "generation_seed",
                "safety_negatives",
            ),
            detail=(
                "冻结 deployment/generation-seed 门通过；freshness=0 状态"
                "在 carrier 边界被拒且 baseline-equivalent/applied=false，"
                "24 样本直接提示与 held-out 线性探针均低于预注册抽取阈值。"
                if conclusion_6
                else (
                    "deployment、generation-seed、stale-state 与 extraction "
                    "attack 冻结证据未全部通过。"
                )
            ),
        ),
        DueDiligenceConclusion(
            conclusion_id="C7",
            statement="World/Environment/Object bank 改善未见场景迁移",
            state="not-yet-proven",
            evidence_ids=("bank_gain",),
            detail=(
                "World/Environment/Object bank 尚未实现；当前 Personal/"
                "Relationship bank-gain 门状态为 "
                f"{payloads['bank_gain'].get('gate_state', 'missing')}，"
                "bank 数量保持冻结。"
            ),
        ),
    )
    return StateKVDueDiligenceReport(
        freeze_id=manifest.freeze_id,
        conclusions=conclusions,
        evidence_fingerprints=tuple(
            (artifact.evidence_id, artifact.sha256)
            for artifact in manifest.evidence
        ),
    )


def freeze_manifest_from_json(payload: Mapping[str, object]) -> StateKVFreezeManifest:
    if payload.get("schema_version") != FREEZE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported freeze manifest schema_version "
            f"{payload.get('schema_version')!r}"
        )
    expected_keys = {
        "schema_version",
        "freeze_id",
        "model",
        "prefix_manifest",
        "prefix_artifact_id",
        "experiment_config",
        "experiment_config_sha256",
        "profile_labels",
        "generation_seeds",
        "scenario_sets",
        "metric_definitions",
        "judge_panel",
        "evidence",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            "freeze manifest fields do not match schema: "
            f"missing={sorted(expected_keys - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected_keys)}"
        )
    model = payload["model"]
    if not isinstance(model, Mapping):
        raise TypeError("freeze manifest model must be an object")
    prefix_manifest = payload["prefix_manifest"]
    if not isinstance(prefix_manifest, Mapping):
        raise TypeError("freeze manifest prefix_manifest must be an object")
    experiment_config = payload["experiment_config"]
    if not isinstance(experiment_config, Mapping):
        raise TypeError("freeze manifest experiment_config must be an object")
    evidence_payload = payload["evidence"]
    if not isinstance(evidence_payload, Sequence):
        raise TypeError("freeze manifest evidence must be an array")
    if not all(isinstance(item, Mapping) for item in evidence_payload):
        raise TypeError(
            "freeze manifest evidence entries must all be objects"
        )
    experiment_config_json = _canonical_json(experiment_config)
    manifest = StateKVFreezeManifest(
        freeze_id=str(payload["freeze_id"]),
        model_id=str(model["model_id"]),
        model_weights_sha256=str(model["weights_sha256"]),
        prefix_manifest_path=str(prefix_manifest["path"]),
        prefix_manifest_sha256=str(prefix_manifest["sha256"]),
        prefix_artifact_id=str(payload["prefix_artifact_id"]),
        experiment_config_json=experiment_config_json,
        experiment_config_sha256=str(payload["experiment_config_sha256"]),
        profile_labels=tuple(str(value) for value in payload["profile_labels"]),
        generation_seeds=tuple(
            int(value) for value in payload["generation_seeds"]
        ),
        scenario_sets=tuple(str(value) for value in payload["scenario_sets"]),
        metric_definitions=tuple(
            str(value) for value in payload["metric_definitions"]
        ),
        judge_panel=tuple(str(value) for value in payload["judge_panel"]),
        evidence=tuple(
            FrozenEvidenceArtifact(
                evidence_id=str(item["evidence_id"]),
                path=str(item["path"]),
                sha256=str(item["sha256"]),
                schema_version=str(item["schema_version"]),
                observed_state=str(item["observed_state"]),
            )
            for item in evidence_payload
        ),
    )
    validate_freeze_manifest(manifest)
    return manifest


__all__ = [
    "DUE_DILIGENCE_SCHEMA_VERSION",
    "FREEZE_MANIFEST_SCHEMA_VERSION",
    "DueDiligenceConclusion",
    "FrozenEvidenceArtifact",
    "StateKVDueDiligenceReport",
    "StateKVFreezeManifest",
    "build_due_diligence_report",
    "build_freeze_manifest",
    "freeze_manifest_from_json",
    "validate_freeze_manifest",
    "verify_frozen_evidence",
]
