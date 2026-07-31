"""Versioned shared adapter bundle for one frozen substrate.

The common adapter is the process-wide L1 asset shared by every character.
It binds the three substrate-owned carriers that must advance together:

* the rare-heavy adapter checkpoint;
* the personal/relationship State-KV generator;
* the residual control basis used by ``z_t`` control.

The bundle is immutable and content-addressed.  Runtime loading is allowed
only for an OFFLINE gate record whose decision is ``allow``; omitting the
bundle is the byte-identical rollback path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from volvence_zero.substrate.control_basis import ControlBasisArtifact
from volvence_zero.substrate.prefix_kv_artifact import PrefixKVArtifact
from volvence_zero.substrate.residual_contracts import (
    SubstrateDeltaAdapterLayer,
    SubstrateRareHeavyCheckpoint,
)

COMMON_ADAPTER_BUNDLE_SCHEMA_VERSION = "common-adapter-bundle.v1"
COMMON_ADAPTER_GATE_SCHEMA_VERSION = "common-adapter-gate-record.v1"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class CommonAdapterGateRecord:
    """Auditable OFFLINE admission decision for a common adapter bundle."""

    proposal_id: str
    decision: str
    desired_gate: str
    validation_delta: float
    capacity_cost: float
    rollback_evidence: str
    is_reversible: bool
    evaluation_ref: str
    schema_version: str = COMMON_ADAPTER_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COMMON_ADAPTER_GATE_SCHEMA_VERSION:
            raise ValueError(
                "common adapter gate schema_version must be "
                f"{COMMON_ADAPTER_GATE_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("proposal_id", self.proposal_id),
            ("decision", self.decision),
            ("desired_gate", self.desired_gate),
            ("rollback_evidence", self.rollback_evidence),
            ("evaluation_ref", self.evaluation_ref),
        ):
            if not value.strip():
                raise ValueError(
                    f"common adapter gate {name} must be non-empty."
                )
        if self.decision not in {"allow", "deny"}:
            raise ValueError(
                "common adapter gate decision must be 'allow' or 'deny'."
            )
        if self.desired_gate != "offline":
            raise ValueError(
                "common adapter bundles require desired_gate='offline'."
            )
        if not all(
            math.isfinite(value)
            for value in (self.validation_delta, self.capacity_cost)
        ):
            raise ValueError(
                "common adapter gate numeric evidence must be finite."
            )
        if self.capacity_cost < 0.0:
            raise ValueError(
                "common adapter gate capacity_cost must be non-negative."
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
class CommonAdapterBundle:
    """One process-wide adapter bundle compatible with one frozen base."""

    schema_version: str
    bundle_id: str
    common_adapter_version: str
    base_model_id: str
    base_model_weights_sha256: str
    compatibility_fingerprint: str
    rare_heavy_checkpoint: SubstrateRareHeavyCheckpoint
    state_kv_artifact: PrefixKVArtifact
    control_basis_artifact: ControlBasisArtifact
    gate_record: CommonAdapterGateRecord
    description: str

    @classmethod
    def create(
        cls,
        *,
        common_adapter_version: str,
        base_model_id: str,
        base_model_weights_sha256: str,
        rare_heavy_checkpoint: SubstrateRareHeavyCheckpoint,
        state_kv_artifact: PrefixKVArtifact,
        control_basis_artifact: ControlBasisArtifact,
        gate_record: CommonAdapterGateRecord,
        description: str,
    ) -> "CommonAdapterBundle":
        compatibility_fingerprint = cls.build_compatibility_fingerprint(
            common_adapter_version=common_adapter_version,
            base_model_id=base_model_id,
            base_model_weights_sha256=base_model_weights_sha256,
            rare_heavy_checkpoint=rare_heavy_checkpoint,
            state_kv_artifact=state_kv_artifact,
            control_basis_artifact=control_basis_artifact,
        )
        provisional = cls(
            schema_version=COMMON_ADAPTER_BUNDLE_SCHEMA_VERSION,
            bundle_id="pending",
            common_adapter_version=common_adapter_version,
            base_model_id=base_model_id,
            base_model_weights_sha256=base_model_weights_sha256,
            compatibility_fingerprint=compatibility_fingerprint,
            rare_heavy_checkpoint=rare_heavy_checkpoint,
            state_kv_artifact=state_kv_artifact,
            control_basis_artifact=control_basis_artifact,
            gate_record=gate_record,
            description=description,
        )
        return replace(provisional, bundle_id=provisional._canonical_id())

    def __post_init__(self) -> None:
        if self.schema_version != COMMON_ADAPTER_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                "common adapter bundle schema_version must be "
                f"{COMMON_ADAPTER_BUNDLE_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("bundle_id", self.bundle_id),
            ("common_adapter_version", self.common_adapter_version),
            ("base_model_id", self.base_model_id),
            ("base_model_weights_sha256", self.base_model_weights_sha256),
            ("compatibility_fingerprint", self.compatibility_fingerprint),
            ("description", self.description),
        ):
            if not value.strip():
                raise ValueError(
                    f"common adapter bundle {name} must be non-empty."
                )
        if self.base_model_weights_sha256 == "legacy":
            raise ValueError(
                "common adapter bundle cannot use the legacy substrate "
                "fingerprint sentinel."
            )
        if len(self.base_model_weights_sha256) != 64:
            raise ValueError(
                "common adapter base_model_weights_sha256 must be a full "
                "64-character SHA-256 digest."
            )
        try:
            int(self.base_model_weights_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "common adapter base_model_weights_sha256 must be hexadecimal."
            ) from exc
        self._validate_nested_compatibility()
        expected_fingerprint = self.build_compatibility_fingerprint(
            common_adapter_version=self.common_adapter_version,
            base_model_id=self.base_model_id,
            base_model_weights_sha256=self.base_model_weights_sha256,
            rare_heavy_checkpoint=self.rare_heavy_checkpoint,
            state_kv_artifact=self.state_kv_artifact,
            control_basis_artifact=self.control_basis_artifact,
        )
        if self.compatibility_fingerprint != expected_fingerprint:
            raise ValueError(
                "common adapter compatibility_fingerprint does not match "
                "its base model and nested carriers."
            )
        if self.bundle_id != "pending" and self.bundle_id != self._canonical_id():
            raise ValueError(
                "common adapter bundle_id does not match its canonical payload."
            )

    def _validate_nested_compatibility(self) -> None:
        nested_model_ids = {
            self.rare_heavy_checkpoint.model_id,
            self.state_kv_artifact.model_id,
            self.control_basis_artifact.model_id,
        }
        if nested_model_ids != {self.base_model_id}:
            raise ValueError(
                "common adapter nested carrier model ids must all match "
                f"base_model_id={self.base_model_id!r}; got "
                f"{sorted(nested_model_ids)!r}."
            )
        if not self.rare_heavy_checkpoint.compatibility_fingerprint:
            raise ValueError(
                "common adapter rare-heavy checkpoint must publish a "
                "compatibility_fingerprint."
            )
        if not self.rare_heavy_checkpoint.adapter_layers:
            raise ValueError(
                "common adapter rare-heavy checkpoint must carry a non-empty "
                "adapter payload."
            )

    @staticmethod
    def build_compatibility_fingerprint(
        *,
        common_adapter_version: str,
        base_model_id: str,
        base_model_weights_sha256: str,
        rare_heavy_checkpoint: SubstrateRareHeavyCheckpoint,
        state_kv_artifact: PrefixKVArtifact,
        control_basis_artifact: ControlBasisArtifact,
    ) -> str:
        payload = {
            "common_adapter_version": common_adapter_version,
            "base_model_id": base_model_id,
            "base_model_weights_sha256": base_model_weights_sha256,
            "rare_heavy_compatibility_fingerprint": (
                rare_heavy_checkpoint.compatibility_fingerprint
            ),
            "rare_heavy_training_mode": rare_heavy_checkpoint.training_mode,
            "state_kv_artifact_id": state_kv_artifact.artifact_id,
            "state_kv_geometry": [
                state_kv_artifact.num_layers,
                state_kv_artifact.num_kv_heads,
                state_kv_artifact.head_dim,
            ],
            "control_basis_artifact_id": control_basis_artifact.artifact_id,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def active_eligible(self) -> bool:
        return self.gate_record.allows_active

    def require_active(self) -> None:
        if not self.active_eligible:
            raise ValueError(
                "common adapter bundle is not ACTIVE-eligible: an allowed, "
                "reversible OFFLINE ModificationGate record is required."
            )

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "common_adapter_version": self.common_adapter_version,
            "base_model_id": self.base_model_id,
            "base_model_weights_sha256": self.base_model_weights_sha256,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "rare_heavy_checkpoint": asdict(self.rare_heavy_checkpoint),
            "state_kv_artifact": json.loads(self.state_kv_artifact.to_json()),
            "control_basis_artifact": self.control_basis_artifact.as_json_dict(),
            "gate_record": asdict(self.gate_record),
            "description": self.description,
        }

    def _canonical_id(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._canonical_payload()).encode("utf-8")
        ).hexdigest()

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["bundle_id"] = self.bundle_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "CommonAdapterBundle":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("common adapter bundle must be a JSON object.")
        required = {
            "schema_version",
            "bundle_id",
            "common_adapter_version",
            "base_model_id",
            "base_model_weights_sha256",
            "compatibility_fingerprint",
            "rare_heavy_checkpoint",
            "state_kv_artifact",
            "control_basis_artifact",
            "gate_record",
            "description",
        }
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        if missing or extra:
            raise ValueError(
                "common adapter bundle fields do not match schema; "
                f"missing={missing}, extra={extra}."
            )
        checkpoint_raw = raw["rare_heavy_checkpoint"]
        if not isinstance(checkpoint_raw, dict):
            raise ValueError("rare_heavy_checkpoint must be an object.")
        checkpoint = _checkpoint_from_dict(checkpoint_raw)
        state_kv_raw = raw["state_kv_artifact"]
        control_raw = raw["control_basis_artifact"]
        gate_raw = raw["gate_record"]
        if not isinstance(state_kv_raw, dict):
            raise ValueError("state_kv_artifact must be an object.")
        if not isinstance(control_raw, dict):
            raise ValueError("control_basis_artifact must be an object.")
        if not isinstance(gate_raw, dict):
            raise ValueError("gate_record must be an object.")
        return cls(
            schema_version=str(raw["schema_version"]),
            bundle_id=str(raw["bundle_id"]),
            common_adapter_version=str(raw["common_adapter_version"]),
            base_model_id=str(raw["base_model_id"]),
            base_model_weights_sha256=str(raw["base_model_weights_sha256"]),
            compatibility_fingerprint=str(raw["compatibility_fingerprint"]),
            rare_heavy_checkpoint=checkpoint,
            state_kv_artifact=PrefixKVArtifact.from_json(
                json.dumps(state_kv_raw, ensure_ascii=False)
            ),
            control_basis_artifact=ControlBasisArtifact.from_json(
                json.dumps(control_raw, ensure_ascii=False)
            ),
            gate_record=CommonAdapterGateRecord(**gate_raw),
            description=str(raw["description"]),
        )


def _checkpoint_from_dict(raw: dict[str, Any]) -> SubstrateRareHeavyCheckpoint:
    fields = set(SubstrateRareHeavyCheckpoint.__dataclass_fields__)
    missing = sorted(fields - set(raw))
    extra = sorted(set(raw) - fields)
    if missing or extra:
        raise ValueError(
            "rare-heavy checkpoint fields do not match schema; "
            f"missing={missing}, extra={extra}."
        )
    layers_raw = raw.get("adapter_layers", [])
    if not isinstance(layers_raw, list):
        raise ValueError("rare-heavy adapter_layers must be an array.")
    layers = tuple(
        SubstrateDeltaAdapterLayer(
            layer_index=int(layer["layer_index"]),
            delta_vector=tuple(float(value) for value in layer["delta_vector"]),
            mean_abs_delta=float(layer["mean_abs_delta"]),
            description=str(layer["description"]),
        )
        for layer in layers_raw
    )
    values = dict(raw)
    values["semantic_anchor_bias"] = tuple(
        float(value) for value in raw["semantic_anchor_bias"]
    )
    values["adapter_layers"] = layers
    return SubstrateRareHeavyCheckpoint(**values)


__all__ = [
    "COMMON_ADAPTER_BUNDLE_SCHEMA_VERSION",
    "COMMON_ADAPTER_GATE_SCHEMA_VERSION",
    "CommonAdapterBundle",
    "CommonAdapterGateRecord",
]
