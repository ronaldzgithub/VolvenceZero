"""Versioned Relationship-bank Prefix-KV artifacts.

The generic PrefixKVArtifact owns bounded State-to-KV tensor math. This
wrapper freezes its semantic use for the Relationship bank so a Personal or
character artifact cannot be loaded under a Relationship carrier merely
because the attention geometry happens to match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from volvence_zero.conditioning_bank_contracts import ConditioningBankType
from volvence_zero.substrate.prefix_kv_artifact import (
    PrefixKVArtifact,
    PrefixKVGenerator,
    load_prefix_generator,
)

RELATIONSHIP_PREFIX_KV_SCHEMA_VERSION = "relationship-prefix-kv.v1"
RELATIONSHIP_PREFIX_KV_CARRIER_VERSION = "relationship-prefix-kv-carrier.v1"


@dataclass(frozen=True)
class RelationshipPrefixKVArtifact:
    """Content-addressed Relationship binding around one KV generator."""

    schema_version: str
    bank_type: str
    owner_schema_version: str
    readout_labels: tuple[str, ...]
    prefix_artifact: PrefixKVArtifact
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PREFIX_KV_SCHEMA_VERSION:
            raise ValueError(
                "Relationship Prefix-KV schema_version must be "
                f"{RELATIONSHIP_PREFIX_KV_SCHEMA_VERSION!r}."
            )
        if self.bank_type != ConditioningBankType.RELATIONSHIP.value:
            raise ValueError(
                "Relationship Prefix-KV bank_type must be 'relationship'."
            )
        if not self.owner_schema_version.strip():
            raise ValueError(
                "Relationship Prefix-KV owner_schema_version must be non-empty."
            )
        if not self.readout_labels:
            raise ValueError(
                "Relationship Prefix-KV readout_labels must be non-empty."
            )
        if len(set(self.readout_labels)) != len(self.readout_labels):
            raise ValueError(
                "Relationship Prefix-KV readout_labels must be unique."
            )
        if self.prefix_artifact.vector_labels != self.readout_labels:
            raise ValueError(
                "Relationship Prefix-KV readout_labels must match the nested "
                "prefix artifact vector_labels."
            )
        if self.prefix_artifact.norm_cap > 0.12:
            raise ValueError(
                "Relationship Prefix-KV norm_cap must not exceed 0.12."
            )
        if not self.description.strip():
            raise ValueError(
                "Relationship Prefix-KV description must be non-empty."
            )

    @property
    def artifact_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "bank_type": self.bank_type,
            "owner_schema_version": self.owner_schema_version,
            "readout_labels": list(self.readout_labels),
            "prefix_artifact": json.loads(self.prefix_artifact.to_json()),
            "description": self.description,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def carrier_version(self) -> str:
        return f"{RELATIONSHIP_PREFIX_KV_CARRIER_VERSION}:{self.artifact_id}"

    def to_json(self) -> str:
        payload = asdict(self)
        payload["prefix_artifact"] = json.loads(
            self.prefix_artifact.to_json()
        )
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "RelationshipPrefixKVArtifact":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError(
                "Relationship Prefix-KV artifact must be a JSON object."
            )
        required = {
            "artifact_id",
            "schema_version",
            "bank_type",
            "owner_schema_version",
            "readout_labels",
            "prefix_artifact",
            "description",
        }
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        if missing or extra:
            raise ValueError(
                "Relationship Prefix-KV fields do not match the frozen "
                f"schema; missing={missing}, extra={extra}"
            )
        nested = raw["prefix_artifact"]
        if not isinstance(nested, dict):
            raise ValueError(
                "Relationship Prefix-KV prefix_artifact must be an object."
            )
        artifact = cls(
            schema_version=str(raw["schema_version"]),
            bank_type=str(raw["bank_type"]),
            owner_schema_version=str(raw["owner_schema_version"]),
            readout_labels=tuple(str(value) for value in raw["readout_labels"]),
            prefix_artifact=PrefixKVArtifact.from_json(
                json.dumps(nested, ensure_ascii=False)
            ),
            description=str(raw["description"]),
        )
        if str(raw["artifact_id"]) != artifact.artifact_id:
            raise ValueError(
                "Relationship Prefix-KV artifact_id does not match its "
                "canonical payload."
            )
        return artifact


def bind_relationship_prefix_artifact(
    *,
    prefix_artifact: PrefixKVArtifact,
    owner_schema_version: str,
    readout_labels: Sequence[str],
    description: str,
) -> RelationshipPrefixKVArtifact:
    """Bind a trained generic generator to the Relationship owner contract."""

    return RelationshipPrefixKVArtifact(
        schema_version=RELATIONSHIP_PREFIX_KV_SCHEMA_VERSION,
        bank_type=ConditioningBankType.RELATIONSHIP.value,
        owner_schema_version=owner_schema_version,
        readout_labels=tuple(str(label) for label in readout_labels),
        prefix_artifact=prefix_artifact,
        description=description,
    )


def load_relationship_prefix_generator(
    *,
    torch_module: Any,
    artifact: RelationshipPrefixKVArtifact,
    expected_model_id: str,
    expected_num_layers: int,
    expected_num_kv_heads: int,
    expected_head_dim: int,
    device: Any,
    dtype: Any,
) -> PrefixKVGenerator:
    """Validate attention geometry and materialize the nested generator."""

    return load_prefix_generator(
        torch_module=torch_module,
        artifact=artifact.prefix_artifact,
        expected_model_id=expected_model_id,
        expected_num_layers=expected_num_layers,
        expected_num_kv_heads=expected_num_kv_heads,
        expected_head_dim=expected_head_dim,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "RELATIONSHIP_PREFIX_KV_CARRIER_VERSION",
    "RELATIONSHIP_PREFIX_KV_SCHEMA_VERSION",
    "RelationshipPrefixKVArtifact",
    "bind_relationship_prefix_artifact",
    "load_relationship_prefix_generator",
]
