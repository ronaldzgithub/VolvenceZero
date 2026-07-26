"""Versioned State-KV prefix artifacts (learned state -> per-layer K/V).

The residual carrier frozen in ``personal_conditioning_projector`` is bounded
by ``scale <= 0.12`` on a single hidden state, which measured out at a ~0.3%
relative perturbation and produced byte-identical outputs across users. This
module owns the higher-bandwidth carrier: the personal-conditioning owner's
frozen 16-dimensional readout is mapped to a short per-layer key/value prefix
that the frozen substrate attends over.

Three properties are contractual rather than incidental:

1. **The base model is never mutated.** An artifact carries floats only. It is
   consumed by prepending tensors to the attention cache; no weight is touched,
   and omitting the artifact restores the previous path exactly.
2. **Bandwidth is bounded by measurement, not by taste.** Every generated
   key/value vector is capped at ``norm_cap`` times a *measured* per-layer
   reference norm recorded at bake time. Real Qwen2.5-0.5B key norms span
   259.7 (layer 0) to ~14 (middle layers), so a single global bound would be
   meaningless. Unbounded prefixes are not a theoretical risk: norm-matched
   random prefixes at gain 0.25 already collapse this substrate's output to
   ``'......'``.
3. **No user facts travel in the artifact.** It stores generator coefficients
   keyed to the frozen coordinate set, never dialogue, memory, or per-user
   tensors. A user's prefix exists only for the duration of one generate call.

The artifact is inference-only. Training owns its own torch module and exports
here, so a loaded artifact can never accumulate gradient state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_VECTOR_LABELS,
)

PREFIX_KV_SCHEMA_VERSION = "state-kv-prefix.v1"
TEACHER_DISTILLED_PREFIX_TRAINING_MODE = "teacher-distilled-prefix-v1"

# Upper bound on ``norm_cap`` itself. A prefix key at parity with real key
# norms is not "strong personalization", it is an out-of-distribution token the
# substrate has never attended over; the measured failure mode is degenerate
# text, not a differently-toned answer.
MAX_PREFIX_NORM_CAP = 0.5


def _is_finite_matrix(rows: Sequence[Sequence[float]]) -> bool:
    return all(math.isfinite(value) for row in rows for value in row)


@dataclass(frozen=True)
class PrefixKVArtifact:
    """Float-only State->KV generator compatible with one frozen substrate.

    The generator is deliberately low-rank::

        h      = tanh(encoder_rows @ state_vector + encoder_bias)   # (rank,)
        k_flat = key_projection[l] @ h + key_bias[l]                # (out_dim,)
        K[l]   = cap(reshape(k_flat, (slots, kv_heads, head_dim)), l)

    Rank is a contract term, not a hyperparameter detail: it bounds how much of
    the 16-dimensional readout can reach attention, which is what keeps the
    carrier auditable against the "it memorized the user" objection.
    """

    schema_version: str
    model_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_slots: int
    bottleneck_rank: int
    vector_labels: tuple[str, ...]
    encoder_rows: tuple[tuple[float, ...], ...]
    encoder_bias: tuple[float, ...]
    key_projection: tuple[tuple[tuple[float, ...], ...], ...]
    key_bias: tuple[tuple[float, ...], ...]
    value_projection: tuple[tuple[tuple[float, ...], ...], ...]
    value_bias: tuple[tuple[float, ...], ...]
    reference_key_norms: tuple[float, ...]
    reference_value_norms: tuple[float, ...]
    norm_cap: float
    training_mode: str
    source_fingerprint: str
    sample_count: int
    description: str

    @property
    def output_width(self) -> int:
        """Flat width of one layer's prefix tensor."""

        return self.num_slots * self.num_kv_heads * self.head_dim

    def __post_init__(self) -> None:
        if self.schema_version != PREFIX_KV_SCHEMA_VERSION:
            raise ValueError(
                f"prefix artifact schema_version must be "
                f"{PREFIX_KV_SCHEMA_VERSION!r}."
            )
        if not self.model_id.strip():
            raise ValueError("prefix artifact model_id must be non-empty.")
        for name, value in (
            ("num_layers", self.num_layers),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
            ("num_slots", self.num_slots),
            ("bottleneck_rank", self.bottleneck_rank),
        ):
            if value <= 0:
                raise ValueError(f"prefix artifact {name} must be positive.")
        if self.vector_labels != PERSONAL_CONDITIONING_VECTOR_LABELS:
            raise ValueError(
                "prefix artifact vector_labels must match "
                "personal-conditioning.v1."
            )
        rank = self.bottleneck_rank
        width = self.output_width
        coordinates = len(self.vector_labels)

        if len(self.encoder_rows) != rank:
            raise ValueError(
                "prefix artifact encoder_rows must carry one row per "
                "bottleneck rank."
            )
        if any(len(row) != coordinates for row in self.encoder_rows):
            raise ValueError(
                "every prefix encoder row must span the frozen coordinate set."
            )
        if len(self.encoder_bias) != rank:
            raise ValueError("prefix artifact encoder_bias must match rank.")

        for name, projection, bias in (
            ("key", self.key_projection, self.key_bias),
            ("value", self.value_projection, self.value_bias),
        ):
            if len(projection) != self.num_layers:
                raise ValueError(
                    f"prefix artifact {name}_projection must carry one block "
                    "per layer."
                )
            if len(bias) != self.num_layers:
                raise ValueError(
                    f"prefix artifact {name}_bias must carry one block per "
                    "layer."
                )
            for block in projection:
                if len(block) != width:
                    raise ValueError(
                        f"each {name}_projection block must have "
                        f"{width} rows (slots x kv_heads x head_dim)."
                    )
                if any(len(row) != rank for row in block):
                    raise ValueError(
                        f"every {name}_projection row must span the bottleneck."
                    )
                if not _is_finite_matrix(block):
                    raise ValueError(
                        f"prefix artifact {name}_projection must be finite."
                    )
            for block in bias:
                if len(block) != width:
                    raise ValueError(
                        f"each {name}_bias block must have {width} entries."
                    )
            if not _is_finite_matrix(bias):
                raise ValueError(f"prefix artifact {name}_bias must be finite.")

        if not _is_finite_matrix(self.encoder_rows):
            raise ValueError("prefix artifact encoder_rows must be finite.")
        if any(not math.isfinite(value) for value in self.encoder_bias):
            raise ValueError("prefix artifact encoder_bias must be finite.")

        for name, norms in (
            ("reference_key_norms", self.reference_key_norms),
            ("reference_value_norms", self.reference_value_norms),
        ):
            if len(norms) != self.num_layers:
                raise ValueError(
                    f"prefix artifact {name} must carry one norm per layer."
                )
            if any(not math.isfinite(value) or value <= 0.0 for value in norms):
                # A non-positive reference norm would disable the cap for that
                # layer while still looking like a bounded artifact.
                raise ValueError(
                    f"prefix artifact {name} must be positive and finite; they "
                    "are measured from the frozen substrate at bake time."
                )

        if not 0.0 < self.norm_cap <= MAX_PREFIX_NORM_CAP:
            raise ValueError(
                "prefix artifact norm_cap must be in "
                f"(0, {MAX_PREFIX_NORM_CAP}]; a prefix at parity with real key "
                "norms is out-of-distribution for the frozen substrate."
            )
        if self.training_mode != TEACHER_DISTILLED_PREFIX_TRAINING_MODE:
            raise ValueError(
                "unsupported prefix training_mode "
                f"{self.training_mode!r}; expected "
                f"{TEACHER_DISTILLED_PREFIX_TRAINING_MODE!r}."
            )
        if not self.source_fingerprint.strip():
            raise ValueError(
                "prefix artifact source_fingerprint must be non-empty."
            )
        if self.sample_count <= 0:
            raise ValueError("prefix artifact sample_count must be positive.")
        if not self.description.strip():
            raise ValueError("prefix artifact description must be non-empty.")

    @property
    def artifact_id(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        payload = asdict(self)
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "PrefixKVArtifact":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("prefix artifact must be a JSON object.")
        required_fields = {
            "artifact_id",
            "schema_version",
            "model_id",
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "num_slots",
            "bottleneck_rank",
            "vector_labels",
            "encoder_rows",
            "encoder_bias",
            "key_projection",
            "key_bias",
            "value_projection",
            "value_bias",
            "reference_key_norms",
            "reference_value_norms",
            "norm_cap",
            "training_mode",
            "source_fingerprint",
            "sample_count",
            "description",
        }
        missing = sorted(required_fields - set(raw))
        extra = sorted(set(raw) - required_fields)
        if missing or extra:
            raise ValueError(
                "prefix artifact fields do not match the frozen schema; "
                f"missing={missing}, extra={extra}"
            )
        declared_id = str(raw.pop("artifact_id"))

        def _matrix(key: str) -> tuple[tuple[float, ...], ...]:
            return tuple(
                tuple(float(value) for value in row) for row in raw[key]
            )

        def _blocks(key: str) -> tuple[tuple[tuple[float, ...], ...], ...]:
            return tuple(
                tuple(tuple(float(value) for value in row) for row in block)
                for block in raw[key]
            )

        artifact = cls(
            schema_version=str(raw["schema_version"]),
            model_id=str(raw["model_id"]),
            num_layers=int(raw["num_layers"]),
            num_kv_heads=int(raw["num_kv_heads"]),
            head_dim=int(raw["head_dim"]),
            num_slots=int(raw["num_slots"]),
            bottleneck_rank=int(raw["bottleneck_rank"]),
            vector_labels=tuple(str(value) for value in raw["vector_labels"]),
            encoder_rows=_matrix("encoder_rows"),
            encoder_bias=tuple(float(value) for value in raw["encoder_bias"]),
            key_projection=_blocks("key_projection"),
            key_bias=_matrix("key_bias"),
            value_projection=_blocks("value_projection"),
            value_bias=_matrix("value_bias"),
            reference_key_norms=tuple(
                float(value) for value in raw["reference_key_norms"]
            ),
            reference_value_norms=tuple(
                float(value) for value in raw["reference_value_norms"]
            ),
            norm_cap=float(raw["norm_cap"]),
            training_mode=str(raw["training_mode"]),
            source_fingerprint=str(raw["source_fingerprint"]),
            sample_count=int(raw["sample_count"]),
            description=str(raw["description"]),
        )
        if declared_id != artifact.artifact_id:
            raise ValueError(
                "prefix artifact_id does not match its canonical payload."
            )
        return artifact


def build_teacher_distilled_prefix_artifact(
    *,
    model_id: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_slots: int,
    bottleneck_rank: int,
    encoder_rows: Sequence[Sequence[float]],
    encoder_bias: Sequence[float],
    key_projection: Sequence[Sequence[Sequence[float]]],
    key_bias: Sequence[Sequence[float]],
    value_projection: Sequence[Sequence[Sequence[float]]],
    value_bias: Sequence[Sequence[float]],
    reference_key_norms: Sequence[float],
    reference_value_norms: Sequence[float],
    norm_cap: float,
    source_fingerprint: str,
    sample_count: int,
) -> PrefixKVArtifact:
    """Freeze a trained generator into the versioned artifact."""

    return PrefixKVArtifact(
        schema_version=PREFIX_KV_SCHEMA_VERSION,
        model_id=model_id,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_slots=num_slots,
        bottleneck_rank=bottleneck_rank,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        encoder_rows=tuple(
            tuple(float(value) for value in row) for row in encoder_rows
        ),
        encoder_bias=tuple(float(value) for value in encoder_bias),
        key_projection=tuple(
            tuple(tuple(float(value) for value in row) for row in block)
            for block in key_projection
        ),
        key_bias=tuple(
            tuple(float(value) for value in block) for block in key_bias
        ),
        value_projection=tuple(
            tuple(tuple(float(value) for value in row) for row in block)
            for block in value_projection
        ),
        value_bias=tuple(
            tuple(float(value) for value in block) for block in value_bias
        ),
        reference_key_norms=tuple(float(v) for v in reference_key_norms),
        reference_value_norms=tuple(float(v) for v in reference_value_norms),
        norm_cap=float(norm_cap),
        training_mode=TEACHER_DISTILLED_PREFIX_TRAINING_MODE,
        source_fingerprint=source_fingerprint,
        sample_count=sample_count,
        description=(
            "State-KV prefix generator distilled from the frozen substrate's "
            f"own text-state arm; layers={num_layers} slots={num_slots} "
            f"rank={bottleneck_rank} norm_cap={norm_cap}."
        ),
    )


class PrefixKVGenerator:
    """Materialized, inference-only view of one artifact on one device.

    Holds plain tensors and runs under ``no_grad``: the prefix is a function of
    the owner's readout, never a parameter that a live session can drift.
    """

    def __init__(
        self,
        *,
        torch_module: Any,
        artifact: PrefixKVArtifact,
        device: Any,
        dtype: Any,
    ) -> None:
        self._torch = torch_module
        self._artifact = artifact
        self._device = device
        self._dtype = dtype

        def make(rows: Any) -> Any:
            return torch_module.tensor(
                rows, dtype=torch_module.float32, device=device
            )

        self._encoder = make(artifact.encoder_rows)
        self._encoder_bias = make(artifact.encoder_bias)
        self._key_projection = make(artifact.key_projection)
        self._key_bias = make(artifact.key_bias)
        self._value_projection = make(artifact.value_projection)
        self._value_bias = make(artifact.value_bias)
        self._key_caps = make(artifact.reference_key_norms) * artifact.norm_cap
        self._value_caps = (
            make(artifact.reference_value_norms) * artifact.norm_cap
        )

    @property
    def artifact(self) -> PrefixKVArtifact:
        return self._artifact

    @property
    def num_slots(self) -> int:
        return self._artifact.num_slots

    def build(self, state_vector: Sequence[float]) -> list[tuple[Any, Any]]:
        """Return ``[(key, value)]`` per layer, shaped for the attention cache.

        Each tensor is ``(1, kv_heads, slots, head_dim)`` -- the layout
        ``Cache.update`` concatenates against.
        """

        torch = self._torch
        artifact = self._artifact
        if len(state_vector) != len(artifact.vector_labels):
            raise ValueError(
                f"state vector has {len(state_vector)} coordinates; the frozen "
                f"readout has {len(artifact.vector_labels)}"
            )
        with torch.no_grad():
            state = torch.tensor(
                [float(value) for value in state_vector],
                dtype=torch.float32,
                device=self._device,
            )
            hidden = torch.tanh(self._encoder @ state + self._encoder_bias)
            keys = self._key_projection @ hidden + self._key_bias
            values = self._value_projection @ hidden + self._value_bias
            shape = (
                artifact.num_layers,
                artifact.num_slots,
                artifact.num_kv_heads,
                artifact.head_dim,
            )
            keys = _cap_per_layer(
                torch, keys.reshape(shape), self._key_caps
            )
            values = _cap_per_layer(
                torch, values.reshape(shape), self._value_caps
            )
            # (layers, slots, kv_heads, head_dim) -> per layer
            # (1, kv_heads, slots, head_dim)
            keys = keys.permute(0, 2, 1, 3).unsqueeze(1)
            values = values.permute(0, 2, 1, 3).unsqueeze(1)
            return [
                (
                    keys[index].to(self._dtype).contiguous(),
                    values[index].to(self._dtype).contiguous(),
                )
                for index in range(artifact.num_layers)
            ]


def _cap_per_layer(torch_module: Any, tensor: Any, caps: Any) -> Any:
    """Scale each head vector down to its layer's measured norm budget.

    Scaling down only: a generator that under-uses its budget is allowed to,
    and rescaling *up* would manufacture bandwidth the training run never
    validated.
    """

    norms = tensor.norm(dim=-1, keepdim=True)
    limits = caps.reshape(-1, 1, 1, 1)
    factor = torch_module.clamp(limits / norms.clamp_min(1e-8), max=1.0)
    return tensor * factor


def load_prefix_generator(
    *,
    torch_module: Any,
    artifact: PrefixKVArtifact,
    expected_model_id: str,
    expected_num_layers: int,
    expected_num_kv_heads: int,
    expected_head_dim: int,
    device: Any,
    dtype: Any,
) -> PrefixKVGenerator:
    """Validate artifact/runtime compatibility, then materialize the generator.

    Every mismatch raises. A prefix silently reshaped to fit a different
    attention geometry would still generate text, and the resulting arm would
    look like evidence about relationship state rather than about a bug.
    """

    if artifact.model_id != expected_model_id:
        raise ValueError(
            f"prefix artifact model_id {artifact.model_id!r} does not match "
            f"runtime {expected_model_id!r}."
        )
    for name, declared, expected in (
        ("num_layers", artifact.num_layers, expected_num_layers),
        ("num_kv_heads", artifact.num_kv_heads, expected_num_kv_heads),
        ("head_dim", artifact.head_dim, expected_head_dim),
    ):
        if declared != expected:
            raise ValueError(
                f"prefix artifact {name} {declared} does not match runtime "
                f"{expected}."
            )
    return PrefixKVGenerator(
        torch_module=torch_module,
        artifact=artifact,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "MAX_PREFIX_NORM_CAP",
    "PREFIX_KV_SCHEMA_VERSION",
    "TEACHER_DISTILLED_PREFIX_TRAINING_MODE",
    "PrefixKVArtifact",
    "PrefixKVGenerator",
    "build_teacher_distilled_prefix_artifact",
    "load_prefix_generator",
]
