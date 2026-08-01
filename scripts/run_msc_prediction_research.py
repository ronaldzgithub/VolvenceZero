#!/usr/bin/env python3
"""Run the real-human MSC N+1 capacity/four-arm research lane.

This runner never vendors corpus text and never routes mismatch around the PE
owner.  It intentionally labels its bounded-state ``volvence`` arm as a
prototype: only an external collector using the complete runtime may attest
``volvence_full_stack=True`` for formal R4 adjudication.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import tempfile
import time
from typing import Any, Iterable

from companion_bench.msc_corpus import MSCDyad, load_msc_split
from companion_bench.prediction_research import (
    PREDICTION_ARMS,
    CapacityObservation,
    MSCDialogueTurn,
    MSCNextTurnExample,
    PredictionObservation,
    adjudicate_capacity_ladder,
    adjudicate_prediction_experiment,
    build_msc_next_turn_examples,
    examples_fingerprint,
    render_long_context,
    render_stateless_context,
    render_summary_retrieval_context,
)
from volvence_zero.prediction import ForwardRepresentationBatch, PredictionErrorModule
from volvence_zero.substrate import (
    SubstrateFingerprint,
    SubstrateForwardRepresentationLineage,
    SubstrateForwardRepresentationPublisher,
    SubstrateForwardRepresentationSnapshot,
    build_transformers_runtime_with_fallback,
    fingerprint_model_weight_files,
)

from msc_prediction_checkpoint import PredictionRunCheckpointStore


def _jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)  # type: ignore[arg-type]
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing final result differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: object) -> None:
    _write_immutable_bytes(
        path,
        (
            json.dumps(payload, indent=2, sort_keys=True, default=_jsonable)
            + "\n"
        ).encode("utf-8"),
    )


def _stable_subset(dyads: tuple[MSCDyad, ...], limit: int | None) -> tuple[MSCDyad, ...]:
    if limit is None:
        return dyads
    if limit < 1:
        raise ValueError("dyad limits must be positive")
    return tuple(
        sorted(
            dyads,
            key=lambda row: hashlib.sha256(row.dyad_id.encode("utf-8")).hexdigest(),
        )[:limit]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_vector(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"!{len(values)}d", *values)).hexdigest()


def _source_sha256s(texts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts)


def _audit_payload(audit: object, *, msc_root: Path) -> dict[str, object]:
    payload = asdict(audit)  # type: ignore[arg-type]
    audit_path = Path(str(payload["path"])).resolve()
    try:
        payload["path"] = audit_path.relative_to(msc_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"MSC audit path {audit_path} is outside corpus root {msc_root.resolve()}"
        ) from exc
    payload["path_base"] = "msc_root"
    return payload


def _hashed_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact lineage file does not exist: {path}")
    return {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


class FrozenSentenceEncoder:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        max_seq_length: int,
        batch_size: int,
    ) -> None:
        if max_seq_length < 8 or batch_size < 1:
            raise ValueError("encoder max_seq_length/batch_size are invalid")
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "MSC prediction research requires sentence-transformers and "
                "huggingface-hub"
            ) from exc
        snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True))
        self._model = SentenceTransformer(
            str(snapshot), local_files_only=True, device=device
        )
        transformer = self._model[0]
        declared_limit = int(self._model.max_seq_length)
        position_limit = int(transformer.auto_model.config.max_position_embeddings)
        self.max_seq_length = min(max_seq_length, declared_limit, position_limit)
        self._model.max_seq_length = self.max_seq_length
        self._tokenizer = self._model.tokenizer
        self._tokenizer.truncation_side = "left"
        self.batch_size = batch_size
        self.model_id = model_id
        self.device = device
        self.embedding_dim = int(self._model.get_sentence_embedding_dimension())
        digest = hashlib.sha256()
        digest.update(model_id.encode("utf-8"))
        digest.update(str(self.max_seq_length).encode("ascii"))
        digest.update(b"recency-left-truncation")
        for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
            relative = path.relative_to(snapshot).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(_sha256_file(path).encode("ascii"))
        self.fingerprint = digest.hexdigest()

    def encode(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[tuple[float, ...], ...], float]:
        if not texts:
            return (), 0.0
        started = time.perf_counter()
        values = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (
            tuple(tuple(float(value) for value in row) for row in values),
            elapsed_ms / len(texts),
        )

    def token_cost(self, text: str) -> tuple[int, int]:
        token_count = len(
            self._tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        )
        return (
            max(1, min(token_count, self.max_seq_length)),
            max(0, token_count - self.max_seq_length),
        )


def _unit(values: Iterable[float]) -> tuple[float, ...]:
    row = tuple(values)
    norm = math.sqrt(sum(value * value for value in row))
    if norm <= 1e-12:
        return tuple(0.0 for _ in row)
    return tuple(value / norm for value in row)


def _mean_vectors(
    vectors: tuple[tuple[float, ...], ...], *, dimension: int
) -> tuple[float, ...]:
    if not vectors:
        return tuple(0.0 for _ in range(dimension))
    return tuple(
        sum(row[index] for row in vectors) / len(vectors)
        for index in range(dimension)
    )


def _ema_vectors(
    vectors: tuple[tuple[float, ...], ...], *, dimension: int, decay: float = 0.85
) -> tuple[float, ...]:
    state = tuple(0.0 for _ in range(dimension))
    for vector in vectors:
        state = tuple(
            decay * previous + (1.0 - decay) * current
            for previous, current in zip(state, vector, strict=True)
        )
    return state


def _all_atomic_texts(
    examples_by_split: dict[str, tuple[MSCNextTurnExample, ...]]
) -> tuple[str, ...]:
    texts: dict[str, None] = {}
    for examples in examples_by_split.values():
        for example in examples:
            texts[example.target_text] = None
            for persona in example.personas:
                texts[persona] = None
            for turn in example.history:
                texts[turn.text] = None
    return tuple(texts)


def _target_atomic_texts(
    examples_by_split: dict[str, tuple[MSCNextTurnExample, ...]],
) -> tuple[str, ...]:
    texts: dict[str, None] = {}
    for examples in examples_by_split.values():
        for example in examples:
            texts[example.target_text] = None
            texts[example.latest_text] = None
    return tuple(
        sorted(
            texts,
            key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
        )
    )


def _embedding_map(
    encoder: FrozenSentenceEncoder,
    texts: tuple[str, ...],
) -> tuple[dict[str, tuple[float, ...]], float]:
    rows, per_item_ms = encoder.encode(texts)
    return dict(zip(texts, rows, strict=True)), per_item_ms


def _substrate_target_contract(
    *, model_id: str
) -> tuple[Path, SubstrateFingerprint]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "MSC substrate target capture requires huggingface-hub"
        ) from exc
    snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    model_fingerprint = SubstrateFingerprint(
        model_id=model_id,
        version=snapshot.name,
        weights_sha256=fingerprint_model_weight_files(snapshot),
    )
    return snapshot, model_fingerprint


def _build_substrate_target_publisher(
    *,
    model_id: str,
    snapshot: Path,
    model_fingerprint: SubstrateFingerprint,
    device: str,
    activation_width: int,
    layer_indices: tuple[int, ...] | None,
) -> SubstrateForwardRepresentationPublisher:
    runtime = build_transformers_runtime_with_fallback(
        model_id=model_id,
        model_source=str(snapshot),
        device=device,
        layer_indices=layer_indices,
        hook_layer_selection="middle",
        activation_width=activation_width,
        local_files_only=True,
        fallback_mode="deny",
        runtime_mode="strict-local",
    )
    return SubstrateForwardRepresentationPublisher(
        runtime,
        model_fingerprint=model_fingerprint,
    )


def _substrate_target_map(
    publisher: SubstrateForwardRepresentationPublisher,
    texts: tuple[str, ...],
) -> tuple[dict[str, tuple[float, ...]], SubstrateForwardRepresentationSnapshot]:
    sample_sources = tuple(
        (f"text:{hashlib.sha256(value.encode('utf-8')).hexdigest()}", value)
        for value in texts
    )
    snapshot = publisher.publish(sample_sources)
    return (
        {
            text: row.values
            for text, row in zip(texts, snapshot.representations, strict=True)
        },
        snapshot,
    )


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("MSC prediction checkpoints require numpy") from exc
    return np


def _array_vectors(
    value: object,
    *,
    rows: int,
    dimension: int | None,
    field: str,
) -> tuple[tuple[float, ...], ...]:
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != rows:
        raise ValueError(
            f"{field} checkpoint shape mismatch: expected {rows} rows, got {array.shape}"
        )
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(
            f"{field} checkpoint dimension mismatch: expected {dimension}, "
            f"got {array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{field} checkpoint contains non-finite values")
    return tuple(tuple(float(item) for item in row) for row in array.tolist())


def _array_ints(value: object, *, rows: int, field: str) -> tuple[int, ...]:
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{field} checkpoint shape mismatch")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{field} checkpoint dtype must be integer")
    return tuple(int(item) for item in array.tolist())


def _array_floats(value: object, *, rows: int, field: str) -> tuple[float, ...]:
    np = _require_numpy()
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != rows:
        raise ValueError(f"{field} checkpoint shape mismatch")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} checkpoint contains non-finite values")
    return tuple(float(item) for item in array.tolist())


def _lineage_from_payload(value: object) -> SubstrateForwardRepresentationLineage:
    if not isinstance(value, dict):
        raise ValueError("substrate target checkpoint lineage must be an object")
    model = value.get("model_fingerprint")
    if not isinstance(model, dict):
        raise ValueError("substrate target checkpoint model fingerprint is invalid")
    try:
        model_fingerprint = SubstrateFingerprint(
            model_id=str(model["model_id"]),
            version=str(model["version"]),
            weights_sha256=str(model["weights_sha256"]),
        )
        return SubstrateForwardRepresentationLineage(
            schema_version=str(value["schema_version"]),
            snapshot_fingerprint=str(value["snapshot_fingerprint"]),
            model_fingerprint=model_fingerprint,
            runtime_origin=str(value["runtime_origin"]),
            readout_kind=str(value["readout_kind"]),
            layer_indices=tuple(int(item) for item in value["layer_indices"]),
            activation_widths=tuple(
                int(item) for item in value["activation_widths"]
            ),
            representation_dim=int(value["representation_dim"]),
        )
    except KeyError as exc:
        raise ValueError(
            f"substrate target checkpoint lineage lacks {exc.args[0]!r}"
        ) from exc


def _load_or_build_atomic_embeddings(
    *,
    store: PredictionRunCheckpointStore,
    encoder: FrozenSentenceEncoder,
    texts: tuple[str, ...],
) -> tuple[dict[str, tuple[float, ...]], float]:
    np = _require_numpy()
    metadata = {
        "encoder_fingerprint": encoder.fingerprint,
        "embedding_dim": encoder.embedding_dim,
        "source_sha256": _source_sha256s(texts),
        "raw_text_retained": False,
    }
    unit = "contexts/atomic"
    arrays = store.load_arrays(
        unit=unit,
        relative_path="checkpoints/contexts/atomic.npz",
        expected_metadata=metadata,
    )
    if arrays is None:
        vectors, per_item_ms = encoder.encode(texts)
        store.save_arrays(
            unit=unit,
            relative_path="checkpoints/contexts/atomic.npz",
            metadata=metadata,
            arrays={
                "vectors": np.asarray(vectors, dtype=np.float64),
                "per_item_ms": np.asarray((per_item_ms,), dtype=np.float64),
            },
        )
        return dict(zip(texts, vectors, strict=True)), per_item_ms
    vectors = _array_vectors(
        arrays.get("vectors"),
        rows=len(texts),
        dimension=encoder.embedding_dim,
        field="atomic context vectors",
    )
    latency = _array_floats(
        arrays.get("per_item_ms"), rows=1, field="atomic context latency"
    )[0]
    return dict(zip(texts, vectors, strict=True)), latency


def _load_or_build_substrate_targets(
    *,
    store: PredictionRunCheckpointStore,
    texts: tuple[str, ...],
    model_id: str,
    device: str,
    activation_width: int,
    layer_indices: tuple[int, ...] | None,
) -> tuple[
    dict[str, tuple[float, ...]],
    SubstrateForwardRepresentationLineage,
    SubstrateFingerprint,
]:
    np = _require_numpy()
    snapshot, model_fingerprint = _substrate_target_contract(model_id=model_id)
    source_hashes = _source_sha256s(texts)
    metadata = {
        "model_fingerprint": asdict(model_fingerprint),
        "requested_activation_width": activation_width,
        "requested_layer_indices": layer_indices,
        "source_sha256": source_hashes,
        "raw_text_retained": False,
    }
    unit = "targets/substrate"
    arrays = store.load_arrays(
        unit=unit,
        relative_path="checkpoints/targets/substrate.npz",
        expected_metadata=metadata,
    )
    if arrays is None:
        publisher = _build_substrate_target_publisher(
            model_id=model_id,
            snapshot=snapshot,
            model_fingerprint=model_fingerprint,
            device=device,
            activation_width=activation_width,
            layer_indices=layer_indices,
        )
        embedding_map, target_snapshot = _substrate_target_map(publisher, texts)
        vectors = tuple(embedding_map[text] for text in texts)
        value_hashes = tuple(
            row.values_sha256 for row in target_snapshot.representations
        )
        store.save_arrays(
            unit=unit,
            relative_path="checkpoints/targets/substrate.npz",
            metadata=metadata,
            arrays={
                "vectors": np.asarray(vectors, dtype=np.float64),
                "values_sha256": np.asarray(value_hashes),
                "lineage_json": np.asarray(
                    json.dumps(asdict(target_snapshot.lineage), sort_keys=True)
                ),
            },
        )
        return embedding_map, target_snapshot.lineage, model_fingerprint
    lineage_raw = arrays.get("lineage_json")
    if lineage_raw is None:
        raise ValueError("substrate target checkpoint lacks lineage_json")
    lineage = _lineage_from_payload(json.loads(str(lineage_raw.item())))
    if lineage.model_fingerprint != model_fingerprint:
        raise ValueError("substrate target checkpoint model fingerprint drift")
    vectors = _array_vectors(
        arrays.get("vectors"),
        rows=len(texts),
        dimension=lineage.representation_dim,
        field="substrate target vectors",
    )
    hashes_raw = arrays.get("values_sha256")
    if hashes_raw is None:
        raise ValueError("substrate target checkpoint lacks values_sha256")
    value_hashes = tuple(str(item) for item in hashes_raw.tolist())
    if len(value_hashes) != len(vectors) or any(
        expected != _sha256_vector(vector)
        for expected, vector in zip(value_hashes, vectors, strict=True)
    ):
        raise ValueError("substrate target checkpoint vector hash drift")
    return dict(zip(texts, vectors, strict=True)), lineage, model_fingerprint


def _load_or_build_arm_vectors(
    *,
    store: PredictionRunCheckpointStore,
    arm: str,
    split: str,
    examples: tuple[MSCNextTurnExample, ...],
    encoder: FrozenSentenceEncoder,
    atomic_embeddings: dict[str, tuple[float, ...]],
    atomic_latency_ms: float,
    retrieval_count: int,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    np = _require_numpy()
    metadata = {
        "arm": arm,
        "split": split,
        "encoder_fingerprint": encoder.fingerprint,
        "example_fingerprint": examples_fingerprint(examples),
        "sample_ids": tuple(example.sample_id for example in examples),
        "retrieval_count": retrieval_count,
        "raw_text_retained": False,
    }
    unit = f"contexts/{arm}/{split}"
    relative_path = f"checkpoints/contexts/{arm}-{split}.npz"
    arrays = store.load_arrays(
        unit=unit,
        relative_path=relative_path,
        expected_metadata=metadata,
    )
    if arrays is None:
        prepared = _prepare_arm_vectors(
            arm=arm,
            examples=examples,
            encoder=encoder,
            atomic_embeddings=atomic_embeddings,
            atomic_latency_ms=atomic_latency_ms,
            retrieval_count=retrieval_count,
        )
        vectors, tokens, truncated, latency = prepared
        store.save_arrays(
            unit=unit,
            relative_path=relative_path,
            metadata=metadata,
            arrays={
                "vectors": np.asarray(vectors, dtype=np.float64),
                "tokens": np.asarray(tokens, dtype=np.int64),
                "truncated": np.asarray(truncated, dtype=np.int64),
                "latency_ms": np.asarray(latency, dtype=np.float64),
            },
        )
        return prepared
    return (
        _array_vectors(
            arrays.get("vectors"),
            rows=len(examples),
            dimension=encoder.embedding_dim,
            field=f"{arm}/{split} context vectors",
        ),
        _array_ints(arrays.get("tokens"), rows=len(examples), field="tokens"),
        _array_ints(
            arrays.get("truncated"), rows=len(examples), field="truncated"
        ),
        _array_floats(
            arrays.get("latency_ms"), rows=len(examples), field="latency_ms"
        ),
    )


def _retrieved_turns(
    example: MSCNextTurnExample,
    *,
    embeddings: dict[str, tuple[float, ...]],
    count: int,
) -> tuple[MSCDialogueTurn, ...]:
    query = embeddings[example.latest_text]
    candidates = tuple(example.history[:-1])
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: sum(
            left * right
            for left, right in zip(
                query, embeddings[item[1].text], strict=True
            )
        ),
        reverse=True,
    )[:count]
    return tuple(candidates[index] for index, _ in sorted(ranked))


def _volvence_state(
    example: MSCNextTurnExample,
    *,
    embeddings: dict[str, tuple[float, ...]],
    dimension: int,
) -> tuple[float, ...]:
    target_history = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
    )
    previous_sessions = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
        and turn.session_index < example.session_index
    )
    current_session = tuple(
        embeddings[turn.text]
        for turn in example.history
        if turn.speaker == example.target_speaker
        and turn.session_index == example.session_index
    )
    personas = tuple(embeddings[text] for text in example.personas)
    latest = embeddings[example.latest_text]
    ema = _ema_vectors(target_history, dimension=dimension)
    persona = _mean_vectors(personas, dimension=dimension)
    consolidated = _mean_vectors(previous_sessions, dimension=dimension)
    within_session = _mean_vectors(current_session, dimension=dimension)
    return _unit(
        0.45 * latest[index]
        + 0.25 * ema[index]
        + 0.15 * persona[index]
        + 0.10 * consolidated[index]
        + 0.05 * within_session[index]
        for index in range(dimension)
    )


def _prepare_arm_vectors(
    *,
    arm: str,
    examples: tuple[MSCNextTurnExample, ...],
    encoder: FrozenSentenceEncoder,
    atomic_embeddings: dict[str, tuple[float, ...]],
    atomic_latency_ms: float,
    retrieval_count: int,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    if arm == "volvence":
        started = time.perf_counter()
        vectors = tuple(
            _volvence_state(
                example,
                embeddings=atomic_embeddings,
                dimension=encoder.embedding_dim,
            )
            for example in examples
        )
        state_ms = (time.perf_counter() - started) * 1000.0 / max(len(examples), 1)
        costs = tuple(encoder.token_cost(example.latest_text) for example in examples)
        return (
            vectors,
            tuple(cost[0] for cost in costs),
            tuple(cost[1] for cost in costs),
            tuple(atomic_latency_ms + state_ms for _ in examples),
        )
    if arm == "stateless":
        texts = tuple(render_stateless_context(example) for example in examples)
    elif arm == "long_context":
        texts = tuple(render_long_context(example) for example in examples)
    elif arm == "summary_retrieval":
        texts = tuple(
            render_summary_retrieval_context(
                example,
                retrieved_turns=_retrieved_turns(
                    example,
                    embeddings=atomic_embeddings,
                    count=retrieval_count,
                ),
            )
            for example in examples
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    vectors, per_item_ms = encoder.encode(texts)
    costs = tuple(encoder.token_cost(text) for text in texts)
    return (
        vectors,
        tuple(cost[0] for cost in costs),
        tuple(cost[1] for cost in costs),
        tuple(per_item_ms for _ in examples),
    )


def _batches(indices: tuple[int, ...], batch_size: int) -> Iterable[tuple[int, ...]]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _make_batch(
    *,
    batch_id: str,
    examples: tuple[MSCNextTurnExample, ...],
    context_vectors: tuple[tuple[float, ...], ...],
    target_vectors: tuple[tuple[float, ...], ...],
    persistence_vectors: tuple[tuple[float, ...], ...],
    indices: tuple[int, ...],
    target_lineage: SubstrateForwardRepresentationLineage,
) -> ForwardRepresentationBatch:
    return ForwardRepresentationBatch(
        batch_id=batch_id,
        sample_ids=tuple(examples[index].sample_id for index in indices),
        context_representations=tuple(context_vectors[index] for index in indices),
        target_representations=tuple(target_vectors[index] for index in indices),
        persistence_representations=tuple(
            persistence_vectors[index] for index in indices
        ),
        history_turns=tuple(examples[index].history_turns for index in indices),
        target_lineage=target_lineage,
    )


def _train_head(
    *,
    train_examples: tuple[MSCNextTurnExample, ...],
    train_contexts: tuple[tuple[float, ...], ...],
    train_targets: tuple[tuple[float, ...], ...],
    train_persistence: tuple[tuple[float, ...], ...],
    n_z: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    target_lineage: SubstrateForwardRepresentationLineage,
) -> PredictionErrorModule:
    module = PredictionErrorModule()
    module.configure_forward_representation_head(
        input_dim=len(train_contexts[0]),
        target_dim=len(train_targets[0]),
        n_z=n_z,
        seed=seed,
        learning_rate=learning_rate,
        device=device,
    )
    rng = random.Random(seed)
    for epoch in range(epochs):
        indices = list(range(len(train_examples)))
        rng.shuffle(indices)
        for batch_index, batch_indices in enumerate(
            _batches(tuple(indices), batch_size)
        ):
            module.process_forward_representation_batch(
                _make_batch(
                    batch_id=f"train-e{epoch}-b{batch_index}",
                    examples=train_examples,
                    context_vectors=train_contexts,
                    target_vectors=train_targets,
                    persistence_vectors=train_persistence,
                    indices=batch_indices,
                    target_lineage=target_lineage,
                ),
                update=True,
            )
    return module


def _evaluate_head(
    *,
    module: PredictionErrorModule,
    examples: tuple[MSCNextTurnExample, ...],
    context_vectors: tuple[tuple[float, ...], ...],
    target_vectors: tuple[tuple[float, ...], ...],
    persistence_vectors: tuple[tuple[float, ...], ...],
    batch_size: int,
    target_lineage: SubstrateForwardRepresentationLineage,
) -> tuple[tuple[Any, ...], tuple[float, ...], str]:
    settlements = []
    latencies = []
    fingerprint = ""
    indices = tuple(range(len(examples)))
    for batch_index, batch_indices in enumerate(_batches(indices, batch_size)):
        snapshot = module.process_forward_representation_batch(
            _make_batch(
                batch_id=f"eval-b{batch_index}",
                examples=examples,
                context_vectors=context_vectors,
                target_vectors=target_vectors,
                persistence_vectors=persistence_vectors,
                indices=batch_indices,
                target_lineage=target_lineage,
            ),
            update=False,
        )
        settlements.extend(snapshot.settlements)
        latencies.extend(
            snapshot.elapsed_ms / snapshot.sample_count
            for _ in snapshot.settlements
        )
        fingerprint = snapshot.parameter_fingerprint
    return tuple(settlements), tuple(latencies), fingerprint


def _targets(
    examples: tuple[MSCNextTurnExample, ...],
    embeddings: dict[str, tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(embeddings[example.target_text] for example in examples)


def _persistence(
    examples: tuple[MSCNextTurnExample, ...],
    embeddings: dict[str, tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(embeddings[example.latest_text] for example in examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accept-noncommercial-license", action="store_true", required=True
    )
    parser.add_argument(
        "--encoder", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--substrate-model", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--substrate-device", default="auto")
    parser.add_argument("--substrate-activation-width", type=int, default=896)
    parser.add_argument("--substrate-layer-indices", type=int, nargs="+")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--retrieval-count", type=int, default=4)
    parser.add_argument("--train-dyads", type=int, default=24)
    parser.add_argument("--validation-dyads", type=int, default=12)
    parser.add_argument("--heldout-dyads", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.accept_noncommercial_license:
        raise SystemExit("MSC license acceptance flag is required")
    if (
        args.epochs < 1
        or args.retrieval_count < 1
        or args.substrate_activation_width < 1
    ):
        raise ValueError(
            "epochs/retrieval-count/substrate-activation-width must be positive"
        )
    seeds = tuple(args.seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("research run requires at least three unique non-negative seeds")

    output = args.output.resolve()
    msc_root = args.msc_root.resolve()
    corpus_provenance = msc_root.parent / "DOWNLOAD_PROVENANCE.json"
    if not corpus_provenance.is_file():
        raise FileNotFoundError(
            "MSC artifact capture requires DOWNLOAD_PROVENANCE.json next to the "
            f"extracted corpus root; missing {corpus_provenance}"
        )
    try:
        corpus_provenance_payload = json.loads(
            corpus_provenance.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"MSC corpus provenance is not valid JSON: {corpus_provenance}"
        ) from exc
    if not isinstance(corpus_provenance_payload, dict):
        raise ValueError("MSC corpus provenance root must be a JSON object")
    if corpus_provenance_payload.get("schema_version") != "msc-download-provenance.v1":
        raise ValueError("MSC corpus provenance schema_version is unsupported")

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = (
        Path(__file__).resolve(),
        repository_root / "scripts/companion_test_plan_common.py",
        repository_root / "scripts/msc_prediction_checkpoint.py",
        repository_root / "scripts/run_msc_prediction_test_plan.py",
        repository_root
        / "packages/companion-bench/src/companion_bench/msc_corpus.py",
        repository_root
        / "packages/companion-bench/src/companion_bench/prediction_research.py",
        repository_root
        / "packages/vz-cognition/src/volvence_zero/prediction/error.py",
        repository_root
        / "packages/vz-cognition/src/volvence_zero/prediction/forward_representation.py",
        repository_root
        / "packages/vz-substrate/src/volvence_zero/substrate/forward_representation.py",
    )
    source_hashes = {
        path.relative_to(repository_root).as_posix(): _sha256_file(path)
        for path in source_paths
    }
    layer_indices = (
        tuple(args.substrate_layer_indices)
        if args.substrate_layer_indices is not None
        else None
    )
    run_configuration = {
        "schema_version": "msc-n-plus-one-resumable-run.v1",
        "msc_root": os.fspath(msc_root),
        "corpus_provenance_sha256": _sha256_file(corpus_provenance),
        "encoder": args.encoder,
        "device": args.device,
        "substrate_model": args.substrate_model,
        "substrate_device": args.substrate_device,
        "substrate_activation_width": args.substrate_activation_width,
        "substrate_layer_indices": layer_indices,
        "max_seq_length": args.max_seq_length,
        "encoder_batch_size": args.encoder_batch_size,
        "head_batch_size": args.head_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "retrieval_count": args.retrieval_count,
        "train_dyads": args.train_dyads,
        "validation_dyads": args.validation_dyads,
        "heldout_dyads": args.heldout_dyads,
        "seeds": seeds,
        "source_sha256": source_hashes,
    }
    store = PredictionRunCheckpointStore(
        output_dir=output,
        configuration=run_configuration,
        resume=args.resume,
    )

    split_payload: dict[str, tuple[MSCDyad, ...]] = {}
    audits = {}
    limits = {
        "train": args.train_dyads,
        "validation": args.validation_dyads,
        "heldout": args.heldout_dyads,
    }
    for split in ("train", "validation", "heldout"):
        dyads, audit = load_msc_split(msc_root, split=split, strict=True)
        split_payload[split] = _stable_subset(dyads, limits[split])
        audits[split] = audit
    examples_by_split = {
        split: build_msc_next_turn_examples(dyads)
        for split, dyads in split_payload.items()
    }
    corpus_index = {
        split: {
            "official_audit": _audit_payload(audits[split], msc_root=msc_root),
            "selected_dyads": len(split_payload[split]),
            "prediction_examples": len(examples_by_split[split]),
            "example_fingerprint": examples_fingerprint(examples_by_split[split]),
        }
        for split in ("train", "validation", "heldout")
    }
    cached_corpus_index = store.load_json(
        unit="corpus/index",
        relative_path="checkpoints/corpus/index.json",
    )
    if cached_corpus_index is None:
        store.save_json(
            unit="corpus/index",
            relative_path="checkpoints/corpus/index.json",
            payload=corpus_index,
        )
    elif cached_corpus_index != corpus_index:
        raise ValueError("MSC corpus/example fingerprint drift on resume")

    encoder = FrozenSentenceEncoder(
        model_id=args.encoder,
        device=args.device,
        max_seq_length=args.max_seq_length,
        batch_size=args.encoder_batch_size,
    )
    atomic_texts = _all_atomic_texts(examples_by_split)
    atomic_embeddings, atomic_latency_ms = _load_or_build_atomic_embeddings(
        store=store,
        encoder=encoder,
        texts=atomic_texts,
    )
    target_texts = _target_atomic_texts(examples_by_split)
    target_embeddings, target_lineage, target_model_fingerprint = (
        _load_or_build_substrate_targets(
            store=store,
            texts=target_texts,
            model_id=args.substrate_model,
            device=args.substrate_device,
            activation_width=args.substrate_activation_width,
            layer_indices=layer_indices,
        )
    )
    targets_by_split = {
        split: _targets(examples, target_embeddings)
        for split, examples in examples_by_split.items()
    }
    persistence_by_split = {
        split: _persistence(examples, target_embeddings)
        for split, examples in examples_by_split.items()
    }

    prepared: dict[
        tuple[str, str],
        tuple[
            tuple[tuple[float, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            tuple[float, ...],
        ],
    ] = {}
    for arm in PREDICTION_ARMS:
        for split, examples in examples_by_split.items():
            prepared[(arm, split)] = _load_or_build_arm_vectors(
                store=store,
                arm=arm,
                split=split,
                examples=examples,
                encoder=encoder,
                atomic_embeddings=atomic_embeddings,
                atomic_latency_ms=atomic_latency_ms,
                retrieval_count=args.retrieval_count,
            )

    capacity_rows: list[CapacityObservation] = []
    for n_z in (3, 16, 64, 256):
        for seed in seeds:
            unit = f"capacity/nz-{n_z}/seed-{seed}"
            relative_path = f"checkpoints/capacity/nz-{n_z}-seed-{seed}.json"
            cached = store.load_json(unit=unit, relative_path=relative_path)
            if cached is None:
                module = _train_head(
                    train_examples=examples_by_split["train"],
                    train_contexts=prepared[("volvence", "train")][0],
                    train_targets=targets_by_split["train"],
                    train_persistence=persistence_by_split["train"],
                    n_z=n_z,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.head_batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    target_lineage=target_lineage,
                )
                settlements, _, _ = _evaluate_head(
                    module=module,
                    examples=examples_by_split["validation"],
                    context_vectors=prepared[("volvence", "validation")][0],
                    target_vectors=targets_by_split["validation"],
                    persistence_vectors=persistence_by_split["validation"],
                    batch_size=args.head_batch_size,
                    target_lineage=target_lineage,
                )
                row = CapacityObservation(
                    forward_head_n_z=n_z,
                    seed=seed,
                    split="validation",
                    mean_cosine_similarity=sum(
                        item.cosine_similarity for item in settlements
                    )
                    / len(settlements),
                    mean_squared_error=sum(
                        item.mean_squared_error for item in settlements
                    )
                    / len(settlements),
                )
                store.save_json(
                    unit=unit,
                    relative_path=relative_path,
                    payload=asdict(row),
                )
            else:
                if not isinstance(cached, dict):
                    raise ValueError(f"capacity checkpoint is not an object: {unit}")
                row = CapacityObservation(**cached)
                if row.forward_head_n_z != n_z or row.seed != seed:
                    raise ValueError(f"capacity checkpoint identity drift: {unit}")
            capacity_rows.append(row)
    capacity_verdict = adjudicate_capacity_ladder(
        tuple(capacity_rows),
        complete_train=(
            len(split_payload["train"]) == audits["train"].conversation_count
        ),
        complete_validation=(
            len(split_payload["validation"])
            == audits["validation"].conversation_count
        ),
    )

    observations: list[PredictionObservation] = []
    head_fingerprints: dict[str, str] = {}
    chosen_n_z = capacity_verdict.best_forward_head_n_z
    for arm in PREDICTION_ARMS:
        for seed in seeds:
            unit = f"heldout/{arm}/seed-{seed}"
            relative_path = f"checkpoints/heldout/{arm}-seed-{seed}.json"
            cached = store.load_json(unit=unit, relative_path=relative_path)
            if cached is None:
                module = _train_head(
                    train_examples=examples_by_split["train"],
                    train_contexts=prepared[(arm, "train")][0],
                    train_targets=targets_by_split["train"],
                    train_persistence=persistence_by_split["train"],
                    n_z=chosen_n_z,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.head_batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    target_lineage=target_lineage,
                )
                settlements, head_latency, fingerprint = _evaluate_head(
                    module=module,
                    examples=examples_by_split["heldout"],
                    context_vectors=prepared[(arm, "heldout")][0],
                    target_vectors=targets_by_split["heldout"],
                    persistence_vectors=persistence_by_split["heldout"],
                    batch_size=args.head_batch_size,
                    target_lineage=target_lineage,
                )
                tokens = prepared[(arm, "heldout")][1]
                truncated = prepared[(arm, "heldout")][2]
                context_latency = prepared[(arm, "heldout")][3]
                rows = tuple(
                    PredictionObservation(
                        arm=arm,
                        seed=seed,
                        sample_id=example.sample_id,
                        dyad_id=example.dyad_id,
                        session_index=example.session_index,
                        history_turns=example.history_turns,
                        cosine_similarity=settlement.cosine_similarity,
                        mean_squared_error=settlement.mean_squared_error,
                        persistence_cosine_similarity=(
                            settlement.persistence_cosine_similarity
                        ),
                        persistence_mean_squared_error=(
                            settlement.persistence_mean_squared_error
                        ),
                        context_token_count=tokens[index],
                        context_truncated_tokens=truncated[index],
                        latency_ms=context_latency[index] + head_latency[index],
                    )
                    for index, (example, settlement) in enumerate(
                        zip(
                            examples_by_split["heldout"],
                            settlements,
                            strict=True,
                        )
                    )
                )
                store.save_json(
                    unit=unit,
                    relative_path=relative_path,
                    payload={
                        "arm": arm,
                        "seed": seed,
                        "selected_forward_head_n_z": chosen_n_z,
                        "target_snapshot_fingerprint": (
                            target_lineage.snapshot_fingerprint
                        ),
                        "head_fingerprint": fingerprint,
                        "observations": tuple(asdict(row) for row in rows),
                    },
                )
            else:
                if not isinstance(cached, dict):
                    raise ValueError(f"heldout checkpoint is not an object: {unit}")
                if (
                    cached.get("arm") != arm
                    or cached.get("seed") != seed
                    or cached.get("selected_forward_head_n_z") != chosen_n_z
                    or cached.get("target_snapshot_fingerprint")
                    != target_lineage.snapshot_fingerprint
                ):
                    raise ValueError(f"heldout checkpoint identity drift: {unit}")
                raw_rows = cached.get("observations")
                fingerprint = cached.get("head_fingerprint")
                if not isinstance(raw_rows, list) or not isinstance(fingerprint, str):
                    raise ValueError(f"heldout checkpoint payload is invalid: {unit}")
                rows = tuple(
                    PredictionObservation(**raw_row)
                    for raw_row in raw_rows
                    if isinstance(raw_row, dict)
                )
                if len(rows) != len(raw_rows):
                    raise ValueError(f"heldout checkpoint row is invalid: {unit}")
                if any(row.arm != arm or row.seed != seed for row in rows):
                    raise ValueError(f"heldout checkpoint row identity drift: {unit}")
            observations.extend(rows)
            head_fingerprints[f"{arm}:seed{seed}"] = fingerprint

    prediction_verdict = adjudicate_prediction_experiment(
        tuple(observations),
        heldout_sorted_id_sha256=audits["heldout"].sorted_id_sha256,
        encoder_fingerprint=target_model_fingerprint.weights_sha256,
        volvence_full_stack=False,
    )
    manifest = {
        "schema_version": "msc-n-plus-one-research.v1",
        "license_policy": "noncommercial-research-only",
        "corpus": corpus_index,
        "encoder": {
            "role": "context-only-prototype; not the N+1 target owner",
            "model_id": encoder.model_id,
            "fingerprint": encoder.fingerprint,
            "embedding_dim": encoder.embedding_dim,
            "max_seq_length": encoder.max_seq_length,
            "device": encoder.device,
        },
        "target_representation": {
            "owner": "vz-substrate",
            "lineage": asdict(target_lineage),
            "captured_text_count": len(target_texts),
            "raw_text_retained": False,
        },
        "training": {
            "seeds": seeds,
            "forward_head_capacity_n_z": (3, 16, 64, 256),
            "selected_forward_head_n_z": chosen_n_z,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "head_batch_size": args.head_batch_size,
            "head_fingerprints": head_fingerprints,
        },
        "checkpointing": {
            "schema_version": "msc-prediction-run-state.v1",
            "resume_supported": True,
            "resume_requires_exact_configuration_and_source_sha": True,
            "immutable_unit_count": len(store.immutable_file_manifest()),
            "mutable_control_state": "run_state.json",
            "mutable_control_state_is_evidence": False,
            "intermediate_effect_analysis_allowed": False,
            "raw_corpus_text_retained": False,
        },
        "arms": {
            "volvence": (
                "PE-owned bounded recurrent relationship-state prototype; "
                "NOT a full runtime-stack attestation"
            ),
            "stateless": "persona plus latest partner message",
            "long_context": (
                "complete rendered relationship history, recency-truncated by "
                "the frozen encoder at the audited max sequence length"
            ),
            "summary_retrieval": (
                "persona summary plus frozen-encoder top-k extractive memories"
            ),
        },
        "claim_boundary": (
            "The target is now a frozen substrate-owned N+1 residual representation, "
            "but this remains pilot evidence: contexts still use MiniLM, long_context "
            f"is limited to {encoder.max_seq_length} tokens, and volvence bypasses "
            "the complete runtime stack."
        ),
        "execution_status": {
            "evidence_level": "mechanism-only-pilot",
            "thesis_status": "not-evaluated",
            "formal_experiment_executed": False,
            "completed_blockers": (
                "official-msc-corpus",
                "substrate-n-plus-one-target",
            ),
            "remaining_blockers": (
                "same-substrate-long-context-steelman",
                "complete-volvence-runtime-arm",
                "temporal-controller-capacity-ladder",
            ),
        },
        "run_configuration": run_configuration,
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "corpus_provenance.json", corpus_provenance_payload)
    _write_json(output / "capacity_ladder.json", capacity_verdict)
    _write_json(output / "prediction_verdict.json", prediction_verdict)
    _write_immutable_bytes(
        output / "prediction_observations.jsonl",
        "".join(
            json.dumps(asdict(observation), sort_keys=True) + "\n"
            for observation in observations
        ).encode("utf-8"),
    )
    _write_immutable_bytes(
        output / "report.md",
        "\n".join(
            (
                "# MSC N+1 prediction research report",
                "",
                f"- Evidence level: `{prediction_verdict.evidence_level}`",
                "- Thesis status: `not-evaluated`",
                "- Formal experiment executed: `false`",
                (
                    "- N+1 target owner: `vz-substrate` / "
                    f"`{target_lineage.readout_kind}`"
                ),
                (
                    "- N+1 target model: "
                    f"`{target_model_fingerprint.to_short_id()}`"
                ),
                f"- Thesis exit: `{prediction_verdict.thesis_exit}`",
                (
                    "- Forward-head capacity exit: "
                    f"`{capacity_verdict.forward_head_claim_exit}`"
                ),
                f"- Selected PE forward_head_n_z: `{chosen_n_z}`",
                (
                    "- Longest-session cosine advantage vs long context: "
                    f"`{prediction_verdict.longest_quality_advantage:.6f}`"
                ),
                f"- Token ratio: `{prediction_verdict.longest_token_ratio:.6f}`",
                f"- Latency ratio: `{prediction_verdict.longest_latency_ratio:.6f}`",
                "- Resume checkpoints: `complete`",
                "",
                "This is pilot-only: the bounded-state arm is not the complete ",
                "Volvence runtime stack, and partial corpus runs cannot select thesis v3.",
                "",
            )
        ).encode("utf-8"),
    )

    current_source_hashes = {
        path.relative_to(repository_root).as_posix(): _sha256_file(path)
        for path in source_paths
    }
    if current_source_hashes != source_hashes:
        raise ValueError("MSC runner source drifted during execution")
    result_names = (
        "manifest.json",
        "corpus_provenance.json",
        "capacity_ladder.json",
        "prediction_verdict.json",
        "prediction_observations.jsonl",
        "report.md",
    )
    _write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": "msc-mechanism-artifact.v2",
            "evidence_level": "mechanism-only-pilot",
            "thesis_status": "not-evaluated",
            "formal_experiment_executed": False,
            "corpus_provenance_source": _hashed_file(corpus_provenance),
            "source_files": {
                path.relative_to(repository_root).as_posix(): _hashed_file(path)
                for path in source_paths
            },
            "result_files": {
                name: _hashed_file(output / name) for name in result_names
            },
            "checkpoint_files": store.immutable_file_manifest(),
            "mutable_control_files_excluded": {
                "run_state.json": (
                    "resume journal only; not an effect input or evidence result"
                )
            },
            "hash_scope": (
                "artifact_manifest.json is excluded to avoid a recursive self-hash"
            ),
        },
    )
    store.mark_complete()
    print(output)
    print(prediction_verdict.description)
    print(capacity_verdict.description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
