"""Unified immutable manifest for a reviewed character package.

``LifeformTemplate`` remains the semantic/lived-state artifact, while model-
side Prefix/KV and optional PEFT LoRA remain substrate artifacts.  This module
owns the L2 manifest that binds those references to one base model and one
shared common-adapter release.  It is an offline/package contract, not a new
runtime owner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

CHARACTER_PACKAGE_MANIFEST_SCHEMA_VERSION = "character-package-manifest.v1"
CHARACTER_ARTIFACT_REF_SCHEMA_VERSION = "character-artifact-ref.v1"
CHARACTER_LORA_REF_SCHEMA_VERSION = "character-lora-ref.v1"
CHARACTER_FIDELITY_EVIDENCE_SCHEMA_VERSION = "character-fidelity-evidence.v1"
CHARACTER_PACKAGE_GATE_SCHEMA_VERSION = "character-package-gate-record.v1"

_REVALIDATION_MODES = frozenset({"full-rebake", "fidelity-only"})
_EVIDENCE_SOURCES = frozenset(
    {"system_self_eval", "llm_judge", "external_validated"}
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_path(path: Path) -> str:
    """Digest a file or a directory tree without following symlinks."""

    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise ValueError(f"artifact directory {path} is empty.")
    for item in files:
        if item.is_symlink():
            raise ValueError(
                f"artifact directory {path} contains symlink {item}; "
                "package digests require regular files."
            )
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class CharacterArtifactRef:
    """Content-addressed reference to one immutable file artifact."""

    locator: str
    sha256: str
    artifact_id: str
    media_type: str
    schema_version: str = CHARACTER_ARTIFACT_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHARACTER_ARTIFACT_REF_SCHEMA_VERSION:
            raise ValueError(
                "character artifact ref schema_version must be "
                f"{CHARACTER_ARTIFACT_REF_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("locator", self.locator),
            ("sha256", self.sha256),
            ("artifact_id", self.artifact_id),
            ("media_type", self.media_type),
        ):
            if not value.strip():
                raise ValueError(f"character artifact ref {name} must be non-empty.")
        _validate_sha256(self.sha256, field_name="artifact ref sha256")


@dataclass(frozen=True)
class CharacterLoRARef:
    """Optional heavyweight PEFT checkpoint reference."""

    locator: str
    sha256: str
    training_plan_hash: str
    parameter_count: int
    backend_id: str = "peft-character-lora-v1"
    schema_version: str = CHARACTER_LORA_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHARACTER_LORA_REF_SCHEMA_VERSION:
            raise ValueError(
                "character LoRA ref schema_version must be "
                f"{CHARACTER_LORA_REF_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("locator", self.locator),
            ("sha256", self.sha256),
            ("training_plan_hash", self.training_plan_hash),
            ("backend_id", self.backend_id),
        ):
            if not value.strip():
                raise ValueError(f"character LoRA ref {name} must be non-empty.")
        _validate_sha256(self.sha256, field_name="character LoRA sha256")
        if self.parameter_count <= 0:
            raise ValueError("character LoRA parameter_count must be positive.")


@dataclass(frozen=True)
class CharacterFidelityEvidence:
    """Held-out behavior evidence bound to one adapter compatibility anchor."""

    report_ref: CharacterArtifactRef
    evidence_source: str
    verdict: str
    passed: bool
    held_out: bool
    source_immutable: bool
    feedback_free: bool
    includes_character_lora: bool
    common_adapter_version: str
    compatibility_fingerprint: str
    schema_version: str = CHARACTER_FIDELITY_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHARACTER_FIDELITY_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                "character fidelity evidence schema_version must be "
                f"{CHARACTER_FIDELITY_EVIDENCE_SCHEMA_VERSION!r}."
            )
        if self.evidence_source not in _EVIDENCE_SOURCES:
            raise ValueError(
                "character fidelity evidence_source must be one of "
                f"{sorted(_EVIDENCE_SOURCES)!r}."
            )
        for name, value in (
            ("verdict", self.verdict),
            ("common_adapter_version", self.common_adapter_version),
            ("compatibility_fingerprint", self.compatibility_fingerprint),
        ):
            if not value.strip():
                raise ValueError(
                    f"character fidelity evidence {name} must be non-empty."
                )

    @property
    def active_eligible(self) -> bool:
        return all(
            (
                self.passed,
                self.held_out,
                self.source_immutable,
                self.feedback_free,
            )
        )


@dataclass(frozen=True)
class CharacterPackageGateRecord:
    """ModificationGate decision binding fidelity evidence to a package."""

    proposal_id: str
    decision: str
    desired_gate: str
    fidelity_report_sha256: str
    rollback_evidence: str
    is_reversible: bool
    common_adapter_version: str
    compatibility_fingerprint: str
    schema_version: str = CHARACTER_PACKAGE_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHARACTER_PACKAGE_GATE_SCHEMA_VERSION:
            raise ValueError(
                "character package gate schema_version must be "
                f"{CHARACTER_PACKAGE_GATE_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("proposal_id", self.proposal_id),
            ("decision", self.decision),
            ("desired_gate", self.desired_gate),
            ("fidelity_report_sha256", self.fidelity_report_sha256),
            ("rollback_evidence", self.rollback_evidence),
            ("common_adapter_version", self.common_adapter_version),
            ("compatibility_fingerprint", self.compatibility_fingerprint),
        ):
            if not value.strip():
                raise ValueError(
                    f"character package gate {name} must be non-empty."
                )
        if self.decision not in {"allow", "deny"}:
            raise ValueError(
                "character package gate decision must be 'allow' or 'deny'."
            )
        if self.desired_gate != "offline":
            raise ValueError(
                "character package promotion requires desired_gate='offline'."
            )
        _validate_sha256(
            self.fidelity_report_sha256,
            field_name="character package gate fidelity_report_sha256",
        )

    @property
    def allows_active(self) -> bool:
        return (
            self.decision == "allow"
            and self.desired_gate == "offline"
            and self.is_reversible
            and bool(self.rollback_evidence.strip())
        )


@dataclass(frozen=True)
class CharacterPackageManifest:
    """Unified L2 character package manifest."""

    schema_version: str
    package_id: str
    character_id: str
    character_name: str
    base_model_id: str
    common_adapter_version: str
    compatibility_fingerprint: str
    template_ref: CharacterArtifactRef
    prefix_kv_ref: CharacterArtifactRef | None
    lora_ref: CharacterLoRARef | None
    fidelity_evidence: CharacterFidelityEvidence | None
    gate_record: CharacterPackageGateRecord | None
    revalidation_mode: str
    description: str

    @classmethod
    def create(
        cls,
        *,
        character_id: str,
        character_name: str,
        base_model_id: str,
        common_adapter_version: str,
        compatibility_fingerprint: str,
        template_ref: CharacterArtifactRef,
        prefix_kv_ref: CharacterArtifactRef | None,
        lora_ref: CharacterLoRARef | None = None,
        fidelity_evidence: CharacterFidelityEvidence | None = None,
        gate_record: CharacterPackageGateRecord | None = None,
        revalidation_mode: str = "full-rebake",
        description: str,
    ) -> "CharacterPackageManifest":
        provisional = cls(
            schema_version=CHARACTER_PACKAGE_MANIFEST_SCHEMA_VERSION,
            package_id="pending",
            character_id=character_id,
            character_name=character_name,
            base_model_id=base_model_id,
            common_adapter_version=common_adapter_version,
            compatibility_fingerprint=compatibility_fingerprint,
            template_ref=template_ref,
            prefix_kv_ref=prefix_kv_ref,
            lora_ref=lora_ref,
            fidelity_evidence=fidelity_evidence,
            gate_record=gate_record,
            revalidation_mode=revalidation_mode,
            description=description,
        )
        return replace(provisional, package_id=provisional._canonical_id())

    def __post_init__(self) -> None:
        if self.schema_version != CHARACTER_PACKAGE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "character package manifest schema_version must be "
                f"{CHARACTER_PACKAGE_MANIFEST_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("package_id", self.package_id),
            ("character_id", self.character_id),
            ("character_name", self.character_name),
            ("base_model_id", self.base_model_id),
            ("common_adapter_version", self.common_adapter_version),
            ("compatibility_fingerprint", self.compatibility_fingerprint),
            ("description", self.description),
        ):
            if not value.strip():
                raise ValueError(
                    f"character package manifest {name} must be non-empty."
                )
        if self.revalidation_mode not in _REVALIDATION_MODES:
            raise ValueError(
                "character package revalidation_mode must be one of "
                f"{sorted(_REVALIDATION_MODES)!r}."
            )
        if self.package_id != "pending" and self.package_id != self._canonical_id():
            raise ValueError(
                "character package package_id does not match canonical payload."
            )

    @property
    def active_eligible(self) -> bool:
        evidence = self.fidelity_evidence
        gate = self.gate_record
        if self.prefix_kv_ref is None or evidence is None or gate is None:
            return False
        if not evidence.active_eligible or not gate.allows_active:
            return False
        if self.lora_ref is not None and not evidence.includes_character_lora:
            return False
        expected_proposal_prefix = (
            f"character-package:{self.ungated_candidate_id}:"
        )
        if not gate.proposal_id.startswith(expected_proposal_prefix):
            return False
        return (
            evidence.common_adapter_version == self.common_adapter_version
            and evidence.compatibility_fingerprint
            == self.compatibility_fingerprint
            and gate.common_adapter_version == self.common_adapter_version
            and gate.compatibility_fingerprint == self.compatibility_fingerprint
            and gate.fidelity_report_sha256 == evidence.report_ref.sha256
        )

    @property
    def ungated_candidate_id(self) -> str:
        """Content id of the exact carrier set evaluated before promotion."""

        candidate = replace(
            self,
            package_id="pending",
            fidelity_evidence=None,
            gate_record=None,
        )
        return candidate._canonical_id()

    def require_active(self) -> None:
        if not self.active_eligible:
            raise ValueError(
                "character package is not ACTIVE-eligible: a Prefix/KV ref, "
                "held-out immutable feedback-free fidelity pass, matching "
                "common-adapter fingerprints, and an allowed reversible "
                "OFFLINE gate record are required; character LoRA packages "
                "also require an adapter+LoRA evidence arm."
            )

    def assert_common_adapter(
        self,
        *,
        base_model_id: str,
        common_adapter_version: str,
        compatibility_fingerprint: str,
    ) -> None:
        expected = (
            self.base_model_id,
            self.common_adapter_version,
            self.compatibility_fingerprint,
        )
        actual = (
            base_model_id,
            common_adapter_version,
            compatibility_fingerprint,
        )
        if expected != actual:
            raise ValueError(
                "character package is incompatible with the active common "
                f"adapter: manifest={expected!r}, runtime={actual!r}."
            )

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "character_id": self.character_id,
            "character_name": self.character_name,
            "base_model_id": self.base_model_id,
            "common_adapter_version": self.common_adapter_version,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "template_ref": asdict(self.template_ref),
            "prefix_kv_ref": (
                asdict(self.prefix_kv_ref)
                if self.prefix_kv_ref is not None
                else None
            ),
            "lora_ref": asdict(self.lora_ref) if self.lora_ref is not None else None,
            "fidelity_evidence": (
                asdict(self.fidelity_evidence)
                if self.fidelity_evidence is not None
                else None
            ),
            "gate_record": (
                asdict(self.gate_record) if self.gate_record is not None else None
            ),
            "revalidation_mode": self.revalidation_mode,
            "description": self.description,
        }

    def _canonical_id(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._canonical_payload()).encode("utf-8")
        ).hexdigest()

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["package_id"] = self.package_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "CharacterPackageManifest":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("character package manifest must be a JSON object.")
        required = {
            "schema_version",
            "package_id",
            "character_id",
            "character_name",
            "base_model_id",
            "common_adapter_version",
            "compatibility_fingerprint",
            "template_ref",
            "prefix_kv_ref",
            "lora_ref",
            "fidelity_evidence",
            "gate_record",
            "revalidation_mode",
            "description",
        }
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        if missing or extra:
            raise ValueError(
                "character package manifest fields do not match schema; "
                f"missing={missing}, extra={extra}."
            )
        template_raw = _require_object(raw["template_ref"], "template_ref")
        prefix_raw = _optional_object(raw["prefix_kv_ref"], "prefix_kv_ref")
        lora_raw = _optional_object(raw["lora_ref"], "lora_ref")
        evidence_raw = _optional_object(
            raw["fidelity_evidence"], "fidelity_evidence"
        )
        gate_raw = _optional_object(raw["gate_record"], "gate_record")
        evidence = None
        if evidence_raw is not None:
            report_raw = _require_object(
                evidence_raw.pop("report_ref", None),
                "fidelity_evidence.report_ref",
            )
            evidence = CharacterFidelityEvidence(
                report_ref=CharacterArtifactRef(**report_raw),
                **evidence_raw,
            )
        return cls(
            schema_version=str(raw["schema_version"]),
            package_id=str(raw["package_id"]),
            character_id=str(raw["character_id"]),
            character_name=str(raw["character_name"]),
            base_model_id=str(raw["base_model_id"]),
            common_adapter_version=str(raw["common_adapter_version"]),
            compatibility_fingerprint=str(raw["compatibility_fingerprint"]),
            template_ref=CharacterArtifactRef(**template_raw),
            prefix_kv_ref=(
                CharacterArtifactRef(**prefix_raw)
                if prefix_raw is not None
                else None
            ),
            lora_ref=(CharacterLoRARef(**lora_raw) if lora_raw is not None else None),
            fidelity_evidence=evidence,
            gate_record=(
                CharacterPackageGateRecord(**gate_raw)
                if gate_raw is not None
                else None
            ),
            revalidation_mode=str(raw["revalidation_mode"]),
            description=str(raw["description"]),
        )


def resolve_artifact_path(
    ref: CharacterArtifactRef | CharacterLoRARef,
    *,
    manifest_path: Path,
) -> Path:
    path = Path(ref.locator).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def verify_manifest_artifacts(
    manifest: CharacterPackageManifest,
    *,
    manifest_path: Path,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    """Verify every referenced digest and return resolved artifact paths."""

    template_path = _verify_ref(manifest.template_ref, manifest_path=manifest_path)
    prefix_path = (
        _verify_ref(manifest.prefix_kv_ref, manifest_path=manifest_path)
        if manifest.prefix_kv_ref is not None
        else None
    )
    lora_path = (
        _verify_ref(manifest.lora_ref, manifest_path=manifest_path)
        if manifest.lora_ref is not None
        else None
    )
    evidence_path = (
        _verify_ref(
            manifest.fidelity_evidence.report_ref,
            manifest_path=manifest_path,
        )
        if manifest.fidelity_evidence is not None
        else None
    )
    return template_path, prefix_path, lora_path, evidence_path


def rebind_fidelity_only(
    manifest: CharacterPackageManifest,
    *,
    base_model_id: str,
    common_adapter_version: str,
    compatibility_fingerprint: str,
    fidelity_evidence: CharacterFidelityEvidence,
    gate_record: CharacterPackageGateRecord,
) -> CharacterPackageManifest:
    """Re-sign a fidelity-only package after a common-adapter upgrade."""

    if manifest.revalidation_mode != "fidelity-only":
        raise ValueError(
            "character package requires full-rebake; fidelity-only rebinding "
            "is not permitted by its manifest."
        )
    rebound = CharacterPackageManifest.create(
        character_id=manifest.character_id,
        character_name=manifest.character_name,
        base_model_id=base_model_id,
        common_adapter_version=common_adapter_version,
        compatibility_fingerprint=compatibility_fingerprint,
        template_ref=manifest.template_ref,
        prefix_kv_ref=manifest.prefix_kv_ref,
        lora_ref=manifest.lora_ref,
        fidelity_evidence=fidelity_evidence,
        gate_record=gate_record,
        revalidation_mode=manifest.revalidation_mode,
        description=manifest.description,
    )
    rebound.require_active()
    return rebound


def character_fidelity_evidence_from_json(
    payload: str,
) -> CharacterFidelityEvidence:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("character fidelity evidence must be a JSON object.")
    values = dict(raw)
    report_raw = _require_object(values.pop("report_ref", None), "report_ref")
    return CharacterFidelityEvidence(
        report_ref=CharacterArtifactRef(**report_raw),
        **values,
    )


def character_package_gate_record_from_json(
    payload: str,
) -> CharacterPackageGateRecord:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("character package gate record must be a JSON object.")
    return CharacterPackageGateRecord(**raw)


def _verify_ref(
    ref: CharacterArtifactRef | CharacterLoRARef,
    *,
    manifest_path: Path,
) -> Path:
    path = resolve_artifact_path(ref, manifest_path=manifest_path)
    actual = sha256_path(path)
    if actual != ref.sha256:
        raise ValueError(
            f"artifact digest mismatch for {path}: declared={ref.sha256}, "
            f"actual={actual}."
        )
    return path


def _validate_sha256(value: str, *, field_name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hexadecimal.") from exc


def _require_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"character package {name} must be an object.")
    return dict(value)


def _optional_object(value: object, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _require_object(value, name)


__all__ = [
    "CHARACTER_ARTIFACT_REF_SCHEMA_VERSION",
    "CHARACTER_FIDELITY_EVIDENCE_SCHEMA_VERSION",
    "CHARACTER_LORA_REF_SCHEMA_VERSION",
    "CHARACTER_PACKAGE_GATE_SCHEMA_VERSION",
    "CHARACTER_PACKAGE_MANIFEST_SCHEMA_VERSION",
    "CharacterArtifactRef",
    "CharacterFidelityEvidence",
    "CharacterLoRARef",
    "CharacterPackageGateRecord",
    "CharacterPackageManifest",
    "character_fidelity_evidence_from_json",
    "character_package_gate_record_from_json",
    "rebind_fidelity_only",
    "resolve_artifact_path",
    "sha256_path",
    "verify_manifest_artifacts",
]
