"""Frozen-model adapters for Relationship Lab product baselines.

The live BGE adapter passes an exact immutable revision to the actual
``SentenceTransformer`` load.  A parent process may materialize its output for
typed :class:`ProductBaselineInput` values into a canonical content-addressed
table; fresh children then query that immutable table without loading BGE
again.

Only public history/current-observation strings are representable at the table
builder boundary.  Evaluator truth, owner snapshots, outcomes that have not yet
become public history, PE/credit, and steering state are intentionally absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from lifeform_domain_emogpt.lab import canonical_json, sha256_json
from lifeform_evolution.relationship_lab_product_baselines import (
    ProductBaselineInput,
    ProductHistorySemanticEmbedder,
)


BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
PRECOMPUTED_PUBLIC_EMBEDDING_RECORD_SCHEMA_VERSION = "relationship-product-public-embedding-record.v1"
PRECOMPUTED_PUBLIC_EMBEDDING_TABLE_SCHEMA_VERSION = "relationship-product-public-embedding-table.v2"

_SHA256_LENGTH = 64
_REVISION_LENGTH = 40
_BGE_SOURCE_NAME_PREFIX = f"sentence-transformer:{BGE_M3_MODEL_ID}@revision:"
_BGE_SOURCE_NAME_SUFFIXES = (
    "/live-public-exact-text-cache-v1",
    "/model-adapter-v1",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_non_empty_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_sha256(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")


def _require_lower_hex_revision(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _REVISION_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase 40-hex revision")
    return value


def _parse_bge_source_embedder_name(value: object) -> tuple[str, str]:
    _require_non_empty_text(value, "source_embedder_name")
    assert isinstance(value, str)
    if not value.startswith(_BGE_SOURCE_NAME_PREFIX):
        raise ValueError(
            "source_embedder_name must bind the exact BAAI/bge-m3 source and revision"
        )
    for suffix in _BGE_SOURCE_NAME_SUFFIXES:
        if value.endswith(suffix):
            revision = value[len(_BGE_SOURCE_NAME_PREFIX) : -len(suffix)]
            _require_lower_hex_revision(revision, "source BGE revision")
            canonical_name = f"{_BGE_SOURCE_NAME_PREFIX}{revision}{suffix}"
            if value != canonical_name:
                raise ValueError("source_embedder_name must use canonical BGE identity syntax")
            return BGE_M3_MODEL_ID, revision
    raise ValueError("source_embedder_name uses an unsupported BGE adapter identity")


def _parse_source_embedder_identity(value: object) -> tuple[str, str | None]:
    """Parse formal BGE identity or an unmistakable test-only identity."""

    _require_non_empty_text(value, "source_embedder_name")
    assert isinstance(value, str)
    if value.startswith("fake-test-only/"):
        label = value.removeprefix("fake-test-only/")
        if not label or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in label):
            raise ValueError("fake-test-only source identity must use one canonical slug")
        return value, None
    return _parse_bge_source_embedder_name(value)


class _SentenceEmbeddingVector(Protocol):
    def tolist(self) -> list[float]: ...


class _SentenceEmbeddingModel(Protocol):
    def encode(
        self,
        text: str,
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> _SentenceEmbeddingVector: ...


class RevisionPinnedProductHistorySemanticEmbedder(ProductHistorySemanticEmbedder, Protocol):
    """Public semantic encoder with inspectable, exact model provenance."""

    model_source: str
    model_revision: str


class RevisionPinnedBgeM3PublicSemanticEmbedder:
    """Lazy BGE-M3 adapter whose revision is passed to the actual model load."""

    def __init__(
        self,
        *,
        model_revision: str = BGE_M3_MODEL_REVISION,
        device: str | None = None,
        model_factory: Callable[..., _SentenceEmbeddingModel] | None = None,
    ) -> None:
        self._model_revision = _require_lower_hex_revision(
            model_revision,
            "model_revision",
        )
        if device is not None:
            _require_non_empty_text(device, "device")
        self._device = device
        self._model_factory = model_factory
        self._model: _SentenceEmbeddingModel | None = None

    @property
    def model_source(self) -> str:
        return BGE_M3_MODEL_ID

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def name(self) -> str:
        return (
            f"{_BGE_SOURCE_NAME_PREFIX}{self._model_revision}"
            "/model-adapter-v1"
        )

    def _ensure_model(self) -> _SentenceEmbeddingModel:
        if self._model is None:
            factory = self._model_factory
            if factory is None:
                from sentence_transformers import SentenceTransformer

                factory = SentenceTransformer
            self._model = factory(
                BGE_M3_MODEL_ID,
                revision=self._model_revision,
                device=self._device,
                local_files_only=True,
            )
        return self._model

    def embed(self, text: str) -> tuple[float, ...]:
        _require_non_empty_text(text, "text")
        encoded = self._ensure_model().encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        raw_values = encoded.tolist()
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError("BGE embedder must return a non-empty one-dimensional vector")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_values):
            raise ValueError("BGE embedding values must be numeric")
        vector = tuple(float(value) for value in raw_values)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("BGE embedding values must be finite")
        return vector


def bge_m3_public_semantic_embedder(
    *,
    device: str | None = None,
    model_revision: str = BGE_M3_MODEL_REVISION,
    model_factory: Callable[..., _SentenceEmbeddingModel] | None = None,
) -> RevisionPinnedProductHistorySemanticEmbedder:
    """Return a lazy, local-only BGE-M3 adapter pinned to one exact revision."""

    return RevisionPinnedBgeM3PublicSemanticEmbedder(
        model_revision=model_revision,
        device=device,
        model_factory=model_factory,
    )


@dataclass(frozen=True)
class PrecomputedPublicEmbeddingRecord:
    """One exact public UTF-8 string and its finite hexadecimal vector."""

    text: str
    embedding_hex: tuple[str, ...]
    schema_version: str = PRECOMPUTED_PUBLIC_EMBEDDING_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty_text(self.text, "text")
        if not isinstance(self.embedding_hex, tuple) or not self.embedding_hex:
            raise ValueError("embedding_hex must be a non-empty tuple")
        for index, encoded in enumerate(self.embedding_hex):
            _require_non_empty_text(encoded, f"embedding_hex[{index}]")
            try:
                value = float.fromhex(encoded)
            except ValueError as exc:
                raise ValueError(f"embedding_hex[{index}] must be a hexadecimal float") from exc
            if not math.isfinite(value):
                raise ValueError(f"embedding_hex[{index}] must be finite")
            if value.hex() != encoded:
                raise ValueError(f"embedding_hex[{index}] must use canonical float.hex() form")
        if self.schema_version != PRECOMPUTED_PUBLIC_EMBEDDING_RECORD_SCHEMA_VERSION:
            raise ValueError("public embedding record schema_version mismatch")

    @property
    def text_sha256(self) -> str:
        return _sha256_text(self.text)

    @property
    def embedding(self) -> tuple[float, ...]:
        return tuple(float.fromhex(value) for value in self.embedding_hex)

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "embedding_hex": list(self.embedding_hex),
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._core_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self._core_payload(), "artifact_id": self.artifact_id}

    @classmethod
    def from_payload(cls, payload: object) -> PrecomputedPublicEmbeddingRecord:
        if not isinstance(payload, dict):
            raise ValueError("public embedding record payload must be an object")
        expected_keys = {
            "schema_version",
            "text",
            "text_sha256",
            "embedding_hex",
            "artifact_id",
        }
        if set(payload) != expected_keys:
            raise ValueError("public embedding record payload keys mismatch")
        raw_embedding = payload["embedding_hex"]
        if not isinstance(raw_embedding, list) or not all(isinstance(value, str) for value in raw_embedding):
            raise ValueError("public embedding record embedding_hex must be a string list")
        record = cls(
            text=payload["text"],
            embedding_hex=tuple(raw_embedding),
            schema_version=payload["schema_version"],
        )
        _require_sha256(payload["text_sha256"], "record text_sha256")
        _require_sha256(payload["artifact_id"], "record artifact_id")
        if payload["text_sha256"] != record.text_sha256:
            raise ValueError("public embedding record text_sha256 mismatch")
        if payload["artifact_id"] != record.artifact_id:
            raise ValueError("public embedding record artifact_id mismatch")
        return record


@dataclass(frozen=True)
class PrecomputedPublicEmbeddingTable:
    """Immutable, content-addressed public embedding table for fresh children."""

    source_embedder_name: str
    embedding_width: int
    records: tuple[PrecomputedPublicEmbeddingRecord, ...]
    schema_version: str = PRECOMPUTED_PUBLIC_EMBEDDING_TABLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _parse_source_embedder_identity(self.source_embedder_name)
        _require_positive_int(self.embedding_width, "embedding_width")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("records must be a non-empty tuple")
        if not all(isinstance(record, PrecomputedPublicEmbeddingRecord) for record in self.records):
            raise ValueError("records must contain only PrecomputedPublicEmbeddingRecord values")
        expected_order = tuple(sorted(self.records, key=lambda record: (record.text_sha256, record.text)))
        if self.records != expected_order:
            raise ValueError("public embedding records must use canonical sha256/text order")
        digests = tuple(record.text_sha256 for record in self.records)
        if len(set(digests)) != len(digests):
            raise ValueError("public embedding record text digests must be unique")
        if any(len(record.embedding_hex) != self.embedding_width for record in self.records):
            raise ValueError("all public embedding records must match embedding_width")
        if self.schema_version != PRECOMPUTED_PUBLIC_EMBEDDING_TABLE_SCHEMA_VERSION:
            raise ValueError("public embedding table schema_version mismatch")

    @property
    def source_model_id(self) -> str:
        model_id, _revision = _parse_source_embedder_identity(self.source_embedder_name)
        return model_id

    @property
    def source_model_revision(self) -> str | None:
        _model_id, revision = _parse_source_embedder_identity(self.source_embedder_name)
        return revision

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_embedder_name": self.source_embedder_name,
            "source_model_id": self.source_model_id,
            "source_model_revision": self.source_model_revision,
            "embedding_width": self.embedding_width,
            "records": [record.to_payload() for record in self.records],
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._core_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self._core_payload(), "artifact_id": self.artifact_id}

    def to_json(self) -> str:
        return canonical_json(self.to_payload()) + "\n"

    @classmethod
    def from_payload(cls, payload: object) -> PrecomputedPublicEmbeddingTable:
        if not isinstance(payload, dict):
            raise ValueError("public embedding table payload must be an object")
        expected_keys = {
            "schema_version",
            "source_embedder_name",
            "source_model_id",
            "source_model_revision",
            "embedding_width",
            "records",
            "artifact_id",
        }
        if set(payload) != expected_keys:
            raise ValueError("public embedding table payload keys mismatch")
        raw_records = payload["records"]
        if not isinstance(raw_records, list):
            raise ValueError("public embedding table records must be a list")
        table = cls(
            source_embedder_name=payload["source_embedder_name"],
            embedding_width=payload["embedding_width"],
            records=tuple(PrecomputedPublicEmbeddingRecord.from_payload(record) for record in raw_records),
            schema_version=payload["schema_version"],
        )
        if payload["source_model_id"] != table.source_model_id:
            raise ValueError("public embedding table source_model_id mismatch")
        if payload["source_model_revision"] != table.source_model_revision:
            raise ValueError("public embedding table source_model_revision mismatch")
        _require_sha256(payload["artifact_id"], "table artifact_id")
        if payload["artifact_id"] != table.artifact_id:
            raise ValueError("public embedding table artifact_id mismatch")
        return table

    @classmethod
    def from_json(cls, raw: str) -> PrecomputedPublicEmbeddingTable:
        if not isinstance(raw, str):
            raise TypeError("raw public embedding table must be a string")

        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            parsed: dict[str, object] = {}
            for key, value in pairs:
                if key in parsed:
                    raise ValueError(f"public embedding table contains duplicate JSON key: {key}")
                parsed[key] = value
            return parsed

        try:
            payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise ValueError("public embedding table must be valid JSON") from exc
        table = cls.from_payload(payload)
        if raw != table.to_json():
            raise ValueError("public embedding table must use exact canonical JSON bytes")
        return table


def _validated_embedding(
    embedder: ProductHistorySemanticEmbedder,
    *,
    text: str,
) -> tuple[float, ...]:
    vector = embedder.embed(text)
    if not isinstance(vector, tuple) or not vector:
        raise ValueError("semantic embedder must return a non-empty tuple")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        raise ValueError("semantic embedding values must be numeric")
    normalized = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("semantic embedding values must be finite")
    return normalized


def build_precomputed_public_embedding_table(
    *,
    embedder: RevisionPinnedProductHistorySemanticEmbedder,
    public_inputs: tuple[ProductBaselineInput, ...],
) -> PrecomputedPublicEmbeddingTable:
    """Embed every unique public string represented by typed baseline inputs."""

    _require_non_empty_text(embedder.name, "embedder.name")
    source_model_id, source_model_revision = _parse_bge_source_embedder_name(embedder.name)
    if embedder.model_source != source_model_id:
        raise ValueError("embedder.model_source disagrees with its canonical source name")
    if embedder.model_revision != source_model_revision:
        raise ValueError("embedder.model_revision disagrees with its canonical source name")
    if not isinstance(public_inputs, tuple) or not public_inputs:
        raise ValueError("public_inputs must be a non-empty tuple")
    if not all(isinstance(item, ProductBaselineInput) for item in public_inputs):
        raise TypeError("public_inputs must contain only ProductBaselineInput values")

    text_by_digest: dict[str, str] = {}
    for public_input in public_inputs:
        texts = (
            *(block.semantic_text for block in public_input.history),
            public_input.current_observation.content,
        )
        for text in texts:
            digest = _sha256_text(text)
            existing = text_by_digest.get(digest)
            if existing is not None and existing != text:
                raise RuntimeError("sha256 collision between distinct public text values")
            text_by_digest[digest] = text

    records: list[PrecomputedPublicEmbeddingRecord] = []
    expected_width: int | None = None
    for digest, text in sorted(text_by_digest.items()):
        vector = _validated_embedding(embedder, text=text)
        if expected_width is None:
            expected_width = len(vector)
        elif len(vector) != expected_width:
            raise ValueError(
                f"semantic embedding width mismatch for public text {digest}: "
                f"expected {expected_width}, got {len(vector)}"
            )
        records.append(
            PrecomputedPublicEmbeddingRecord(
                text=text,
                embedding_hex=tuple(value.hex() for value in vector),
            )
        )
    assert expected_width is not None
    return PrecomputedPublicEmbeddingTable(
        source_embedder_name=embedder.name,
        embedding_width=expected_width,
        records=tuple(records),
    )


def write_precomputed_public_embedding_table(
    table: PrecomputedPublicEmbeddingTable,
    *,
    path: pathlib.Path,
) -> pathlib.Path:
    """Write one canonical table create-only; existing evidence is immutable."""

    if not isinstance(table, PrecomputedPublicEmbeddingTable):
        raise TypeError("table must be PrecomputedPublicEmbeddingTable")
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(table.to_json())
        handle.flush()
        os.fsync(handle.fileno())
    loaded = load_precomputed_public_embedding_table(target)
    if loaded.artifact_id != table.artifact_id:
        raise RuntimeError("written public embedding table failed artifact verification")
    return target


def load_precomputed_public_embedding_table(
    path: pathlib.Path,
) -> PrecomputedPublicEmbeddingTable:
    """Load and fully verify a canonical content-addressed table."""

    source = pathlib.Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"public embedding table does not exist: {source}")
    try:
        raw = source.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("public embedding table must be exact UTF-8") from exc
    return PrecomputedPublicEmbeddingTable.from_json(raw)


class MissingPublicSemanticEmbeddingError(KeyError):
    """Raised when a fresh child requests text absent from the frozen table."""


@dataclass(frozen=True)
class PrecomputedPublicSemanticEmbedder:
    """Read-only semantic embedder backed solely by one verified table."""

    table: PrecomputedPublicEmbeddingTable
    _records_by_sha256: Mapping[str, PrecomputedPublicEmbeddingRecord] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.table, PrecomputedPublicEmbeddingTable):
            raise TypeError("table must be PrecomputedPublicEmbeddingTable")
        records = {record.text_sha256: record for record in self.table.records}
        object.__setattr__(self, "_records_by_sha256", MappingProxyType(records))

    @property
    def name(self) -> str:
        return f"{self.table.source_embedder_name}/precomputed-public-table@sha256:{self.table.artifact_id}"

    def embed(self, text: str) -> tuple[float, ...]:
        _require_non_empty_text(text, "text")
        digest = _sha256_text(text)
        try:
            record = self._records_by_sha256[digest]
        except KeyError as exc:
            raise MissingPublicSemanticEmbeddingError(
                f"public text sha256 {digest} is absent from table {self.table.artifact_id}"
            ) from exc
        if record.text != text:
            raise RuntimeError("sha256 collision while querying the public embedding table")
        return record.embedding


__all__ = [
    "BGE_M3_MODEL_ID",
    "BGE_M3_MODEL_REVISION",
    "MissingPublicSemanticEmbeddingError",
    "PRECOMPUTED_PUBLIC_EMBEDDING_RECORD_SCHEMA_VERSION",
    "PRECOMPUTED_PUBLIC_EMBEDDING_TABLE_SCHEMA_VERSION",
    "PrecomputedPublicEmbeddingRecord",
    "PrecomputedPublicEmbeddingTable",
    "PrecomputedPublicSemanticEmbedder",
    "RevisionPinnedBgeM3PublicSemanticEmbedder",
    "RevisionPinnedProductHistorySemanticEmbedder",
    "bge_m3_public_semantic_embedder",
    "build_precomputed_public_embedding_table",
    "load_precomputed_public_embedding_table",
    "write_precomputed_public_embedding_table",
]
