"""Bounded owner summary for one strict full-prompt residual capture.

Only the substrate owner may traverse token-level residual activations and
interpret its feature-surface names.  Evidence orchestrators consume the
frozen bounded summary published here instead of rebuilding substrate state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import cast, Protocol

from volvence_zero.substrate.residual_contracts import (
    FeatureSignal,
    OpenWeightRuntimeCapture,
    ResidualActivation,
    ResidualSequenceStep,
)


STRICT_CAPTURE_AUDIT_SCHEMA_VERSION = "strict-capture-audit-summary.v1"
_FEATURE_NAMES = (
    "hook_layer_coverage",
    "hook_fire_rate",
    "token_step_coverage",
    "residual_sequence_present",
    "fallback_active",
)


class _Digest(Protocol):
    def update(self, payload: bytes) -> None: ...


@dataclass(frozen=True)
class StrictCaptureAuditSummary:
    """Frozen, bounded readout of a potentially very large owner capture."""

    residual_sequence_length: int
    residual_step_continuity_exact: bool
    capture_layer_exact: bool
    capture_width_exact: bool
    residual_activation_value_count: int
    finite_residual_activation_value_count: int
    capture_values_all_finite: bool
    residual_sequence_sha256: str
    latest_activation_width: int
    latest_activation_sha256: str
    latest_matches_sequence_exact: bool
    top_logit_count: int
    top_logits_finite_nonempty: bool
    top_logits_sha256: str
    selected_feature_values: tuple[tuple[str, float | None], ...]
    description_sha256: str
    schema_version: str = STRICT_CAPTURE_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str:
            raise TypeError("strict capture audit schema_version must be exact text")
        if self.schema_version != STRICT_CAPTURE_AUDIT_SCHEMA_VERSION:
            raise ValueError("strict capture audit schema drift")
        for name in (
            "residual_sequence_length",
            "residual_activation_value_count",
            "finite_residual_activation_value_count",
            "latest_activation_width",
            "top_logit_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"StrictCaptureAuditSummary.{name} is invalid")
        for name in (
            "residual_step_continuity_exact",
            "capture_layer_exact",
            "capture_width_exact",
            "capture_values_all_finite",
            "top_logits_finite_nonempty",
            "latest_matches_sequence_exact",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"StrictCaptureAuditSummary.{name} must be bool")
        for name in (
            "residual_sequence_sha256",
            "latest_activation_sha256",
            "top_logits_sha256",
            "description_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.selected_feature_values) is not tuple or any(
            type(item) is not tuple or len(item) != 2 for item in self.selected_feature_values
        ):
            raise TypeError("strict capture selected features must be exact frozen pairs")
        if tuple(name for name, _ in self.selected_feature_values) != _FEATURE_NAMES:
            raise ValueError("strict capture selected feature set/order drift")
        for name, value in self.selected_feature_values:
            if value is not None:
                if type(value) is not float:
                    raise TypeError(f"strict capture feature {name} must be float or None")
                if not math.isfinite(value):
                    raise ValueError(f"strict capture feature {name} is not finite")
        if self.finite_residual_activation_value_count > self.residual_activation_value_count:
            raise ValueError("strict capture finite residual count exceeds total count")
        if self.capture_values_all_finite is not (
            self.finite_residual_activation_value_count == self.residual_activation_value_count
        ):
            raise ValueError("strict capture finite residual flag/count drift")
        if self.top_logit_count == 0 and self.top_logits_finite_nonempty:
            raise ValueError("strict capture empty logits cannot be finite-nonempty")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "residual_sequence_length": self.residual_sequence_length,
            "residual_step_continuity_exact": (self.residual_step_continuity_exact),
            "capture_layer_exact": self.capture_layer_exact,
            "capture_width_exact": self.capture_width_exact,
            "residual_activation_value_count": (self.residual_activation_value_count),
            "finite_residual_activation_value_count": (self.finite_residual_activation_value_count),
            "capture_values_all_finite": self.capture_values_all_finite,
            "residual_sequence_sha256": self.residual_sequence_sha256,
            "latest_activation_width": self.latest_activation_width,
            "latest_activation_sha256": self.latest_activation_sha256,
            "latest_matches_sequence_exact": (self.latest_matches_sequence_exact),
            "top_logit_count": self.top_logit_count,
            "top_logits_finite_nonempty": self.top_logits_finite_nonempty,
            "top_logits_sha256": self.top_logits_sha256,
            "selected_feature_values": dict(self.selected_feature_values),
            "description_sha256": self.description_sha256,
        }


def audit_strict_capture(
    capture: OpenWeightRuntimeCapture,
    *,
    expected_layer_index: int,
    expected_activation_width: int,
) -> StrictCaptureAuditSummary:
    """Read one public owner capture and publish only a bounded summary."""

    if type(capture) is not OpenWeightRuntimeCapture:
        raise TypeError("strict capture audit requires OpenWeightRuntimeCapture")
    if type(expected_layer_index) is not int or expected_layer_index < 0:
        raise ValueError("expected_layer_index must be a non-negative int")
    if type(expected_activation_width) is not int or expected_activation_width <= 0:
        raise ValueError("expected_activation_width must be a positive int")

    _require_bool(
        capture.personal_conditioning_applied,
        "personal_conditioning_applied",
    )
    sequence = _require_tuple(capture.residual_sequence, "residual_sequence")
    residual_digest = hashlib.sha256()
    residual_digest.update(b"strict-capture-residual-sequence.v1\x00")
    _update_unsigned_count(residual_digest, len(sequence), "sequence length")
    step_continuity = True
    layer_exact = bool(sequence)
    width_exact = bool(sequence)
    all_finite = True
    value_count = 0
    finite_value_count = 0
    for expected_step, step in enumerate(sequence):
        if type(step) is not ResidualSequenceStep:
            raise TypeError("strict capture residual sequence entries must be exact ResidualSequenceStep")
        actual_step = _require_int(
            step.step,
            f"residual_sequence[{expected_step}].step",
        )
        if actual_step != expected_step:
            step_continuity = False
        _require_text(step.token, f"residual_sequence[{expected_step}].token")
        _validate_feature_surface(
            step.feature_surface,
            label=f"residual_sequence[{expected_step}].feature_surface",
        )
        _require_text(
            step.description,
            f"residual_sequence[{expected_step}].description",
        )
        activations = _require_tuple(
            step.residual_activations,
            f"residual_sequence[{expected_step}].residual_activations",
        )
        residual_digest.update(b"\x01")
        _update_unsigned_count(
            residual_digest,
            expected_step,
            "sequence position",
        )
        _update_signed_int(residual_digest, actual_step, "sequence step")
        _update_unsigned_count(
            residual_digest,
            len(activations),
            "activation count",
        )
        if len(activations) != 1:
            layer_exact = False
            width_exact = False
        for activation_position, activation in enumerate(activations):
            if type(activation) is not ResidualActivation:
                raise TypeError("strict capture residual activations must be exact ResidualActivation")
            layer_index = _require_int(
                activation.layer_index,
                "residual activation layer_index",
            )
            activation_step = _require_int(
                activation.step,
                "residual activation step",
            )
            values = _require_tuple(
                activation.activation,
                f"residual_sequence[{expected_step}].activation",
            )
            if activation_position != 0 or layer_index != expected_layer_index or activation_step != expected_step:
                layer_exact = False
            if activation_position != 0 or len(values) != expected_activation_width:
                width_exact = False
            residual_digest.update(b"\x02")
            _update_unsigned_count(
                residual_digest,
                activation_position,
                "activation position",
            )
            _update_signed_int(
                residual_digest,
                layer_index,
                "activation layer_index",
            )
            _update_signed_int(
                residual_digest,
                activation_step,
                "activation step",
            )
            _update_unsigned_count(
                residual_digest,
                len(values),
                "activation width",
            )
            for value in values:
                numeric = _require_float(value, "residual activation value")
                value_count += 1
                if math.isfinite(numeric):
                    finite_value_count += 1
                else:
                    all_finite = False
                residual_digest.update(struct.pack("!d", numeric))

    latest_activations = _validate_activation_tuple(
        capture.residual_activations,
        label="residual_activations",
    )
    expected_latest_activations = sequence[-1].residual_activations if sequence else ()
    latest_activation_sha256 = _activation_tuple_sha256(latest_activations)
    expected_latest_activation_sha256 = _activation_tuple_sha256(expected_latest_activations)
    latest_matches_sequence_exact = latest_activation_sha256 == expected_latest_activation_sha256
    latest_activation_values: tuple[float, ...] = ()
    if len(latest_activations) == 1:
        latest_activation_values = latest_activations[0].activation

    token_logits_raw = _require_tuple(capture.token_logits, "token_logits")
    token_logits = tuple(_require_float(value, "token_logits value") for value in token_logits_raw)
    top_logits_finite = bool(token_logits) and all(math.isfinite(value) for value in token_logits)
    return StrictCaptureAuditSummary(
        residual_sequence_length=len(sequence),
        residual_step_continuity_exact=step_continuity,
        capture_layer_exact=layer_exact,
        capture_width_exact=width_exact,
        residual_activation_value_count=value_count,
        finite_residual_activation_value_count=finite_value_count,
        capture_values_all_finite=all_finite,
        residual_sequence_sha256=residual_digest.hexdigest(),
        latest_activation_width=len(latest_activation_values),
        latest_activation_sha256=latest_activation_sha256,
        latest_matches_sequence_exact=latest_matches_sequence_exact,
        top_logit_count=len(token_logits),
        top_logits_finite_nonempty=top_logits_finite,
        top_logits_sha256=_float_sequence_sha256(token_logits),
        selected_feature_values=_selected_feature_values(capture.feature_surface),
        description_sha256=_sha256_bytes(_require_text(capture.description, "description").encode("utf-8")),
    )


def _selected_feature_values(
    feature_surface: object,
) -> tuple[tuple[str, float | None], ...]:
    features = _validate_feature_surface(
        feature_surface,
        label="feature_surface",
    )
    counts = {name: 0 for name in _FEATURE_NAMES}
    first_values: dict[str, float | None] = {name: None for name in _FEATURE_NAMES}
    for feature in features:
        if feature.name not in counts:
            continue
        values = feature.values
        counts[feature.name] += 1
        if counts[feature.name] == 1:
            if len(values) == 1:
                first_values[feature.name] = values[0]
            else:
                first_values[feature.name] = None
    return tuple(
        (
            name,
            first_values[name]
            if counts[name] == 1 and first_values[name] is not None and math.isfinite(first_values[name])
            else None,
        )
        for name in _FEATURE_NAMES
    )


def _validate_feature_surface(
    value: object,
    *,
    label: str,
) -> tuple[FeatureSignal, ...]:
    features = _require_tuple(value, label)
    for index, feature in enumerate(features):
        if type(feature) is not FeatureSignal:
            raise TypeError(f"strict capture {label}[{index}] must be exact FeatureSignal")
        _require_text(feature.name, f"{label}[{index}].name")
        values = _require_tuple(
            feature.values,
            f"{label}[{index}].values",
        )
        for feature_value in values:
            _require_float(feature_value, f"{label}[{index}].value")
        _require_text(feature.source, f"{label}[{index}].source")
        if feature.layer_hint is not None:
            _require_int(
                feature.layer_hint,
                f"{label}[{index}].layer_hint",
            )
    return cast(tuple[FeatureSignal, ...], features)


def _validate_activation_tuple(
    value: object,
    *,
    label: str,
) -> tuple[ResidualActivation, ...]:
    activations = _require_tuple(value, label)
    for index, activation in enumerate(activations):
        if type(activation) is not ResidualActivation:
            raise TypeError(f"strict capture {label}[{index}] must be exact ResidualActivation")
        _require_int(
            activation.layer_index,
            f"{label}[{index}].layer_index",
        )
        _require_int(activation.step, f"{label}[{index}].step")
        values = _require_tuple(
            activation.activation,
            f"{label}[{index}].activation",
        )
        for activation_value in values:
            _require_float(activation_value, f"{label}[{index}].value")
    return cast(tuple[ResidualActivation, ...], activations)


def _activation_tuple_sha256(
    activations: tuple[ResidualActivation, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"strict-capture-latest-activations.v1\x00")
    _update_unsigned_count(
        digest,
        len(activations),
        "latest activation count",
    )
    for position, activation in enumerate(activations):
        _update_unsigned_count(
            digest,
            position,
            "latest activation position",
        )
        _update_signed_int(
            digest,
            activation.layer_index,
            "latest layer_index",
        )
        _update_signed_int(
            digest,
            activation.step,
            "latest activation step",
        )
        _update_unsigned_count(
            digest,
            len(activation.activation),
            "latest activation width",
        )
        for value in activation.activation:
            digest.update(struct.pack("!d", value))
    return digest.hexdigest()


def _float_sequence_sha256(values: tuple[float, ...]) -> str:
    return _sha256_bytes(b"".join(struct.pack("!d", value) for value in values))


def _update_unsigned_count(
    digest: _Digest,
    value: int,
    label: str,
) -> None:
    try:
        payload = struct.pack("!Q", value)
    except struct.error as exc:
        raise ValueError(f"strict capture {label} exceeds uint64") from exc
    digest.update(payload)


def _update_signed_int(
    digest: _Digest,
    value: int,
    label: str,
) -> None:
    try:
        payload = struct.pack("!q", value)
    except struct.error as exc:
        raise ValueError(f"strict capture {label} exceeds int64") from exc
    digest.update(payload)


def _require_tuple(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"strict capture {label} must be an exact tuple")
    return value


def _require_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"strict capture {label} must be exact text")
    return value


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"strict capture {label} must be an exact int")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"strict capture {label} must be an exact bool")
    return value


def _require_float(value: object, label: str) -> float:
    if type(value) is not float:
        raise TypeError(f"strict capture {label} must be an exact float")
    return value


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"strict capture {label} must be a lowercase SHA-256")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = (
    "STRICT_CAPTURE_AUDIT_SCHEMA_VERSION",
    "StrictCaptureAuditSummary",
    "audit_strict_capture",
)
