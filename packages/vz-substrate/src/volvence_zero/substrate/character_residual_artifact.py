"""Content-addressed residual adapter package for a reviewed character.

The package is deliberately separate from relationship conditioning and the
generic rare-heavy checkpoint. It carries a frozen, target-model-specific
residual intervention trained from reviewed live-through traces.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from volvence_zero.substrate.residual_contracts import SubstrateDeltaAdapterLayer


CHARACTER_RESIDUAL_ADAPTER_SCHEMA_VERSION = "character-residual-adapter.v1"
CHARACTER_RESIDUAL_ADAPTER_MODE = "character-teacher-forced-residual-v1"
CHARACTER_RESIDUAL_DELTA_CAP = 0.18


@dataclass(frozen=True)
class CharacterResidualAdapterPackage:
    """Validated, target-model-specific character residual artifact."""

    schema_version: str
    package_id: str
    character_id: str
    character_name: str
    model_id: str
    source_live_through_model_id: str
    source_template_id: str
    source_template_integrity_hash: str
    source_live_through_proof: str
    hidden_size: int
    layer_indices: tuple[int, ...]
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...]
    training_mode: str
    training_loss: float
    sample_count: int
    description: str

    @classmethod
    def create(
        cls,
        *,
        character_id: str,
        character_name: str,
        model_id: str,
        source_live_through_model_id: str,
        source_template_id: str,
        source_template_integrity_hash: str,
        source_live_through_proof: str,
        hidden_size: int,
        adapter_layers: Sequence[SubstrateDeltaAdapterLayer],
        training_mode: str = CHARACTER_RESIDUAL_ADAPTER_MODE,
        training_loss: float,
        sample_count: int,
        description: str,
    ) -> "CharacterResidualAdapterPackage":
        layers = tuple(adapter_layers)
        payload = {
            "schema_version": CHARACTER_RESIDUAL_ADAPTER_SCHEMA_VERSION,
            "character_id": character_id,
            "character_name": character_name,
            "model_id": model_id,
            "source_live_through_model_id": source_live_through_model_id,
            "source_template_id": source_template_id,
            "source_template_integrity_hash": source_template_integrity_hash,
            "source_live_through_proof": source_live_through_proof,
            "hidden_size": hidden_size,
            "layer_indices": list(layer.layer_index for layer in layers),
            "adapter_layers": [_layer_payload(layer) for layer in layers],
            "training_mode": training_mode,
            "training_loss": float(training_loss),
            "sample_count": sample_count,
            "description": description,
        }
        package_id = _content_hash(payload)
        return cls(
            schema_version=CHARACTER_RESIDUAL_ADAPTER_SCHEMA_VERSION,
            package_id=package_id,
            character_id=character_id,
            character_name=character_name,
            model_id=model_id,
            source_live_through_model_id=source_live_through_model_id,
            source_template_id=source_template_id,
            source_template_integrity_hash=source_template_integrity_hash,
            source_live_through_proof=source_live_through_proof,
            hidden_size=hidden_size,
            layer_indices=tuple(layer.layer_index for layer in layers),
            adapter_layers=layers,
            training_mode=training_mode,
            training_loss=float(training_loss),
            sample_count=sample_count,
            description=description,
        )

    def __post_init__(self) -> None:
        if self.schema_version != CHARACTER_RESIDUAL_ADAPTER_SCHEMA_VERSION:
            raise ValueError(
                "character residual adapter schema_version must be "
                f"{CHARACTER_RESIDUAL_ADAPTER_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("character_id", self.character_id),
            ("character_name", self.character_name),
            ("model_id", self.model_id),
            ("source_live_through_model_id", self.source_live_through_model_id),
            ("source_template_id", self.source_template_id),
            ("source_template_integrity_hash", self.source_template_integrity_hash),
            ("source_live_through_proof", self.source_live_through_proof),
            ("training_mode", self.training_mode),
            ("description", self.description),
        ):
            if not value.strip():
                raise ValueError(f"character residual adapter {name} must be non-empty.")
        if self.hidden_size <= 0:
            raise ValueError("character residual adapter hidden_size must be positive.")
        if not self.layer_indices:
            raise ValueError("character residual adapter must target at least one layer.")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("character residual adapter layer_indices must be unique.")
        if tuple(layer.layer_index for layer in self.adapter_layers) != self.layer_indices:
            raise ValueError("character residual adapter layer metadata is inconsistent.")
        if not math.isfinite(self.training_loss) or self.training_loss < 0.0:
            raise ValueError("character residual adapter training_loss must be finite and non-negative.")
        if self.sample_count <= 0:
            raise ValueError("character residual adapter sample_count must be positive.")
        for layer in self.adapter_layers:
            if len(layer.delta_vector) != self.hidden_size:
                raise ValueError(
                    f"character residual adapter layer {layer.layer_index} width "
                    f"{len(layer.delta_vector)} does not match hidden_size={self.hidden_size}."
                )
            if any(
                not math.isfinite(value) or abs(value) > CHARACTER_RESIDUAL_DELTA_CAP
                for value in layer.delta_vector
            ):
                raise ValueError(
                    f"character residual adapter layer {layer.layer_index} exceeds "
                    f"delta cap {CHARACTER_RESIDUAL_DELTA_CAP}."
                )
        if self.package_id != self._canonical_id():
            raise ValueError("character residual adapter package_id does not match its payload.")

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "character_id": self.character_id,
            "character_name": self.character_name,
            "model_id": self.model_id,
            "source_live_through_model_id": self.source_live_through_model_id,
            "source_template_id": self.source_template_id,
            "source_template_integrity_hash": self.source_template_integrity_hash,
            "source_live_through_proof": self.source_live_through_proof,
            "hidden_size": self.hidden_size,
            "layer_indices": list(self.layer_indices),
            "adapter_layers": [_layer_payload(layer) for layer in self.adapter_layers],
            "training_mode": self.training_mode,
            "training_loss": self.training_loss,
            "sample_count": self.sample_count,
            "description": self.description,
        }

    def _canonical_id(self) -> str:
        return _content_hash(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["package_id"] = self.package_id
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "CharacterResidualAdapterPackage":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("character residual adapter package must be a JSON object.")
        required = {
            "schema_version",
            "package_id",
            "character_id",
            "character_name",
            "model_id",
            "source_live_through_model_id",
            "source_template_id",
            "source_template_integrity_hash",
            "source_live_through_proof",
            "hidden_size",
            "layer_indices",
            "adapter_layers",
            "training_mode",
            "training_loss",
            "sample_count",
            "description",
        }
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        if missing or extra:
            raise ValueError(
                "character residual adapter fields do not match schema; "
                f"missing={missing}, extra={extra}"
            )
        layers_raw = raw["adapter_layers"]
        if not isinstance(layers_raw, list):
            raise ValueError("character residual adapter adapter_layers must be a list.")
        layers = tuple(_layer_from_payload(item) for item in layers_raw)
        return cls(
            schema_version=str(raw["schema_version"]),
            package_id=str(raw["package_id"]),
            character_id=str(raw["character_id"]),
            character_name=str(raw["character_name"]),
            model_id=str(raw["model_id"]),
            source_live_through_model_id=str(raw["source_live_through_model_id"]),
            source_template_id=str(raw["source_template_id"]),
            source_template_integrity_hash=str(raw["source_template_integrity_hash"]),
            source_live_through_proof=str(raw["source_live_through_proof"]),
            hidden_size=int(raw["hidden_size"]),
            layer_indices=tuple(int(value) for value in raw["layer_indices"]),
            adapter_layers=layers,
            training_mode=str(raw["training_mode"]),
            training_loss=float(raw["training_loss"]),
            sample_count=int(raw["sample_count"]),
            description=str(raw["description"]),
        )


def load_character_residual_deltas(
    *,
    torch_module: Any,
    package: CharacterResidualAdapterPackage,
    expected_model_id: str,
    expected_hidden_size: int,
    available_layer_indices: Sequence[int],
    device: Any,
) -> dict[int, Any]:
    """Validate package/runtime geometry and materialize frozen deltas."""

    if package.model_id != expected_model_id:
        raise ValueError(
            f"character residual adapter model_id {package.model_id!r} does not "
            f"match runtime {expected_model_id!r}."
        )
    if package.hidden_size != expected_hidden_size:
        raise ValueError(
            f"character residual adapter hidden_size {package.hidden_size} does "
            f"not match runtime {expected_hidden_size}."
        )
    unavailable = sorted(set(package.layer_indices) - set(available_layer_indices))
    if unavailable:
        raise ValueError(
            "character residual adapter targets unavailable runtime layers: "
            f"{unavailable}"
        )
    return {
        layer.layer_index: torch_module.tensor(
            layer.delta_vector,
            dtype=torch_module.float32,
            device=device,
        )
        for layer in package.adapter_layers
    }


def _layer_payload(layer: SubstrateDeltaAdapterLayer) -> dict[str, Any]:
    return {
        "layer_index": layer.layer_index,
        "delta_vector": list(layer.delta_vector),
        "mean_abs_delta": layer.mean_abs_delta,
        "description": layer.description,
    }


def _layer_from_payload(raw: Any) -> SubstrateDeltaAdapterLayer:
    if not isinstance(raw, dict):
        raise ValueError("character residual adapter layer must be an object.")
    return SubstrateDeltaAdapterLayer(
        layer_index=int(raw["layer_index"]),
        delta_vector=tuple(float(value) for value in raw["delta_vector"]),
        mean_abs_delta=float(raw["mean_abs_delta"]),
        description=str(raw["description"]),
    )


def _content_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "CHARACTER_RESIDUAL_ADAPTER_MODE",
    "CHARACTER_RESIDUAL_ADAPTER_SCHEMA_VERSION",
    "CHARACTER_RESIDUAL_DELTA_CAP",
    "CharacterResidualAdapterPackage",
    "load_character_residual_deltas",
]
