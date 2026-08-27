from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError

import pytest

from lifeform_domain_emogpt.lab import RelationshipAction, canonical_json
from lifeform_evolution.relationship_lab_baseline import HFStatelessRelationshipActionPolicy
from lifeform_evolution.relationship_lab_product_baselines import (
    FrozenProductChatMessage,
    ProductBaselineInput,
    ProductCurrentObservation,
    ProductPublicHistoryBlock,
)
from lifeform_evolution.relationship_lab_product_batch_model_adapter import (
    bge_m3_batch_public_semantic_embedder,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    MissingPublicSemanticEmbeddingError,
    PrecomputedPublicEmbeddingTable,
    PrecomputedPublicSemanticEmbedder,
    RevisionPinnedBgeM3PublicSemanticEmbedder,
    bge_m3_public_semantic_embedder,
    bge_m3_weight_pinned_embedder_identity,
    build_precomputed_public_embedding_table,
    load_precomputed_public_embedding_table,
    write_precomputed_public_embedding_table,
)


class _FakeTensor:
    def __init__(self, token_count: int) -> None:
        self.shape = (1, token_count)


class _RecordingChatTokenizer:
    def __init__(self) -> None:
        self.template_calls: list[tuple[list[dict[str, str]], bool, bool]] = []
        self.tokenize_calls: list[tuple[str, str]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.template_calls.append((messages, tokenize, add_generation_prompt))
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        return rendered + "|assistant:"

    def __call__(self, rendered: str, *, return_tensors: str) -> dict[str, _FakeTensor]:
        self.tokenize_calls.append((rendered, return_tensors))
        return {"input_ids": _FakeTensor(len(rendered.encode("utf-8")) + 7)}


class _RecordingEmbedder:
    model_source = BGE_M3_MODEL_ID
    model_revision = BGE_M3_MODEL_REVISION
    weights_sha256 = BGE_M3_WEIGHT_BYTES_SHA256
    sentence_transformers_version = BGE_M3_SENTENCE_TRANSFORMERS_VERSION
    name = bge_m3_weight_pinned_embedder_identity(
        model_revision=model_revision,
        weights_sha256=weights_sha256,
        sentence_transformers_version=sentence_transformers_version,
        identity_kind="model-adapter-v2",
    )

    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self._vectors = vectors
        self.seen: list[str] = []

    def embed(self, text: str) -> tuple[float, ...]:
        self.seen.append(text)
        return self._vectors[text]


def _public_input(*contents: str, current: str = "current public observation") -> ProductBaselineInput:
    return ProductBaselineInput(
        history=tuple(
            ProductPublicHistoryBlock(
                ordinal=index,
                exchange_id=f"public-exchange-{index}",
                user_messages=(content,),
                assistant_outcome=canonical_json(
                    {
                        "action_id": RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                        "observed_outcome_id": "felt_heard",
                        "rendered_user_reaction": f"public reaction {index}",
                    }
                ),
            )
            for index, content in enumerate(contents)
        ),
        current_observation=ProductCurrentObservation(content=current),
    )


def test_hf_policy_public_counter_uses_exact_generation_chat_template_path() -> None:
    tokenizer = _RecordingChatTokenizer()
    policy = object.__new__(HFStatelessRelationshipActionPolicy)
    policy._tokenizer = tokenizer
    policy.tokenizer_id = "fake-tokenizer@sha256:" + "a" * 64
    messages = (
        FrozenProductChatMessage(role="system", content="frozen system contract"),
        FrozenProductChatMessage(role="assistant", content="public prior response"),
        FrozenProductChatMessage(role="user", content="public current input"),
    )

    count = policy.count_message_tokens(messages=messages)

    expected_payload = [{"role": message.role, "content": message.content} for message in messages]
    assert tokenizer.template_calls == [(expected_payload, False, True)]
    assert tokenizer.tokenize_calls[0][1] == "pt"
    assert count == len(tokenizer.tokenize_calls[0][0].encode("utf-8")) + 7
    assert policy.tokenizer_id.startswith("fake-tokenizer@sha256:")

    with pytest.raises(ValueError, match="final contextual message"):
        policy.count_message_tokens(messages=(FrozenProductChatMessage(role="assistant", content="not final user"),))


class _FakeEmbeddingVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeEmbeddingMatrix:
    def __init__(self, rows: object) -> None:
        self._rows = rows

    def tolist(self) -> object:
        return self._rows


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool, bool, bool]] = []

    def encode(
        self,
        text: str,
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> _FakeEmbeddingVector:
        self.calls.append(
            (text, normalize_embeddings, convert_to_numpy, show_progress_bar)
        )
        return _FakeEmbeddingVector([1.0, -0.25])


class _FakeBatchSentenceTransformer:
    def __init__(self, rows: object | None = None) -> None:
        self.calls: list[
            tuple[list[str], int, bool, bool, bool]
        ] = []
        self._rows = rows

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> _FakeEmbeddingMatrix:
        self.calls.append(
            (
                list(texts),
                batch_size,
                normalize_embeddings,
                convert_to_numpy,
                show_progress_bar,
            )
        )
        rows = self._rows
        if rows is None:
            rows = [
                [float(index + 1), float(-(index + 1))]
                for index, _text in enumerate(texts)
            ]
        return _FakeEmbeddingMatrix(rows)


def _fake_bge_snapshot(tmp_path, *, weight_bytes: bytes = b"frozen fake bge weights"):
    snapshot = tmp_path / "bge-snapshot"
    snapshot.mkdir()
    (snapshot / "pytorch_model.bin").write_bytes(weight_bytes)
    return snapshot, hashlib.sha256(weight_bytes).hexdigest()


def test_bge_m3_adapter_verifies_offline_snapshot_before_model_factory(tmp_path) -> None:
    snapshot, weights_sha256 = _fake_bge_snapshot(tmp_path)
    model = _FakeSentenceTransformer()
    factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    runtime_calls: list[str] = []

    def factory(*args: object, **kwargs: object) -> _FakeSentenceTransformer:
        factory_calls.append((args, kwargs))
        return model

    def runtime_version(distribution_name: str) -> str:
        runtime_calls.append(distribution_name)
        return BGE_M3_SENTENCE_TRANSFORMERS_VERSION

    embedder = bge_m3_public_semantic_embedder(
        device="cpu",
        weights_sha256=weights_sha256,
        model_factory=factory,
        snapshot_path=snapshot,
        runtime_version_resolver=runtime_version,
    )

    assert isinstance(embedder, RevisionPinnedBgeM3PublicSemanticEmbedder)
    assert factory_calls == []
    assert embedder.model_source == BGE_M3_MODEL_ID
    assert embedder.model_revision == BGE_M3_MODEL_REVISION
    assert embedder.weights_sha256 == weights_sha256
    assert embedder.sentence_transformers_version == BGE_M3_SENTENCE_TRANSFORMERS_VERSION
    assert embedder.name == bge_m3_weight_pinned_embedder_identity(
        model_revision=BGE_M3_MODEL_REVISION,
        weights_sha256=weights_sha256,
        sentence_transformers_version=BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
        identity_kind="model-adapter-v2",
    )
    assert embedder.embed("public evidence only") == (1.0, -0.25)
    assert runtime_calls == ["sentence-transformers"]
    assert factory_calls == [
        (
            (str(snapshot.resolve()),),
            {
                "device": "cpu",
                "local_files_only": True,
            },
        )
    ]
    assert model.calls == [("public evidence only", True, True, False)]

    with pytest.raises(ValueError, match="40-hex revision"):
        bge_m3_public_semantic_embedder(model_revision="main")


def test_bge_m3_adapter_rejects_wrong_weight_or_runtime_before_fake_factory(
    tmp_path,
) -> None:
    snapshot, weights_sha256 = _fake_bge_snapshot(tmp_path)
    factory_calls: list[object] = []

    def factory(*args: object, **kwargs: object) -> _FakeSentenceTransformer:
        factory_calls.append((args, kwargs))
        return _FakeSentenceTransformer()

    wrong_weight = bge_m3_public_semantic_embedder(
        device="cpu",
        weights_sha256="0" * 64,
        model_factory=factory,
        snapshot_path=snapshot,
        runtime_version_resolver=lambda _name: BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    )
    with pytest.raises(ValueError, match="raw weight bytes sha256 mismatch"):
        wrong_weight.embed("public evidence")
    assert factory_calls == []

    wrong_runtime = bge_m3_public_semantic_embedder(
        device="cpu",
        weights_sha256=weights_sha256,
        model_factory=factory,
        snapshot_path=snapshot,
        runtime_version_resolver=lambda _name: "0.0.0",
    )
    with pytest.raises(ValueError, match="runtime version mismatch"):
        wrong_runtime.embed("public evidence")
    assert factory_calls == []


def test_bge_m3_adapter_snapshot_resolver_is_local_only_and_revision_pinned(
    tmp_path,
) -> None:
    snapshot, weights_sha256 = _fake_bge_snapshot(tmp_path)
    resolver_calls: list[dict[str, object]] = []

    def snapshot_resolver(**kwargs: object) -> str:
        resolver_calls.append(kwargs)
        return str(snapshot)

    embedder = bge_m3_public_semantic_embedder(
        device="cpu",
        weights_sha256=weights_sha256,
        model_factory=lambda *_args, **_kwargs: _FakeSentenceTransformer(),
        snapshot_resolver=snapshot_resolver,
        runtime_version_resolver=lambda _name: BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    )
    assert embedder.embed("public evidence") == (1.0, -0.25)
    assert resolver_calls == [
        {
            "repo_id": BGE_M3_MODEL_ID,
            "revision": BGE_M3_MODEL_REVISION,
            "local_files_only": True,
        }
    ]


def test_bge_m3_batch_adapter_preserves_order_and_fixed_encode_contract(
    tmp_path,
) -> None:
    snapshot, weights_sha256 = _fake_bge_snapshot(tmp_path)
    model = _FakeBatchSentenceTransformer()
    embedder = bge_m3_batch_public_semantic_embedder(
        device="cuda",
        weights_sha256=weights_sha256,
        model_factory=lambda *_args, **_kwargs: model,
        snapshot_path=snapshot,
        runtime_version_resolver=lambda _name: BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    )

    texts = ("public alpha", "public beta", "public alpha")
    assert embedder.embed_many(texts, batch_size=2) == (
        (1.0, -1.0),
        (2.0, -2.0),
        (3.0, -3.0),
    )
    assert model.calls == [([*texts], 2, True, True, False)]


@pytest.mark.parametrize(
    ("texts", "batch_size", "message"),
    [
        ((), 2, "non-empty tuple"),
        (["public"], 2, "non-empty tuple"),
        (("",), 2, "non-empty string"),
        (("public",), 0, "positive integer"),
        (("public",), True, "positive integer"),
    ],
)
def test_bge_m3_batch_adapter_rejects_bad_input_before_model_load(
    tmp_path,
    texts,
    batch_size,
    message,
) -> None:
    snapshot, weights_sha256 = _fake_bge_snapshot(tmp_path)
    factory_calls: list[object] = []

    def factory(*args: object, **kwargs: object) -> _FakeBatchSentenceTransformer:
        factory_calls.append((args, kwargs))
        return _FakeBatchSentenceTransformer()

    embedder = bge_m3_batch_public_semantic_embedder(
        weights_sha256=weights_sha256,
        model_factory=factory,
        snapshot_path=snapshot,
        runtime_version_resolver=lambda _name: BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    )
    with pytest.raises(ValueError, match=message):
        embedder.embed_many(texts, batch_size=batch_size)
    assert factory_calls == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[1.0, 2.0]], "row count"),
        ([[1.0], []], "must be non-empty"),
        ([[1.0], [2.0, 3.0]], "fixed width"),
        ([[1.0], [True]], "must be numeric"),
        ([[1.0], [float("nan")]], "must be finite"),
    ],
)
def test_bge_m3_batch_adapter_rejects_malformed_model_output(
    tmp_path,
    rows,
    message,
) -> None:
    snapshot, weights_sha256 = _fake_bge_snapshot(tmp_path)
    model = _FakeBatchSentenceTransformer(rows)
    embedder = bge_m3_batch_public_semantic_embedder(
        weights_sha256=weights_sha256,
        model_factory=lambda *_args, **_kwargs: model,
        snapshot_path=snapshot,
        runtime_version_resolver=lambda _name: BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    )
    with pytest.raises(ValueError, match=message):
        embedder.embed_many(("first", "second"), batch_size=2)
    assert len(model.calls) == 1


def test_precomputed_table_build_load_and_strict_query_are_content_addressed(
    tmp_path,
    monkeypatch,
) -> None:
    first = _public_input("public alpha", "public beta")
    second = _public_input("public gamma", "public beta", current="second public current")
    all_texts = {
        *(block.semantic_text for block in first.history),
        *(block.semantic_text for block in second.history),
        first.current_observation.content,
        second.current_observation.content,
    }
    vectors = {text: (float(index + 1), float(-(index + 1))) for index, text in enumerate(sorted(all_texts))}
    source_embedder = _RecordingEmbedder(vectors)

    table = build_precomputed_public_embedding_table(
        embedder=source_embedder,
        public_inputs=(first, second),
    )
    path = tmp_path / "public_embeddings.json"
    real_fsync = os.fsync
    fsync_calls: list[int] = []

    def recording_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    monkeypatch.setattr(
        "lifeform_evolution.relationship_lab_product_model_adapters.os.fsync",
        recording_fsync,
    )
    write_precomputed_public_embedding_table(table, path=path)
    loaded = load_precomputed_public_embedding_table(path)
    adapter = PrecomputedPublicSemanticEmbedder(loaded)

    assert set(source_embedder.seen) == all_texts
    assert source_embedder.seen.count(first.history[1].semantic_text) == 1
    assert loaded.artifact_id == table.artifact_id
    assert loaded.to_json() == table.to_json()
    assert loaded.source_model_id == BGE_M3_MODEL_ID
    assert loaded.source_model_revision == BGE_M3_MODEL_REVISION
    assert loaded.source_weights_sha256 == BGE_M3_WEIGHT_BYTES_SHA256
    assert (
        loaded.source_sentence_transformers_version
        == BGE_M3_SENTENCE_TRANSFORMERS_VERSION
    )
    assert loaded.to_payload()["source_model_revision"] == BGE_M3_MODEL_REVISION
    assert adapter.weights_sha256 == BGE_M3_WEIGHT_BYTES_SHA256
    assert (
        adapter.sentence_transformers_version
        == BGE_M3_SENTENCE_TRANSFORMERS_VERSION
    )
    assert len(fsync_calls) == 1
    assert adapter.name.endswith(f"sha256:{table.artifact_id}")
    for text in all_texts:
        assert adapter.embed(text) == vectors[text]
    with pytest.raises(MissingPublicSemanticEmbeddingError, match="absent from table"):
        adapter.embed("unmaterialized public text")
    with pytest.raises(FileExistsError):
        write_precomputed_public_embedding_table(table, path=path)
    with pytest.raises(FrozenInstanceError):
        table.embedding_width = 7  # type: ignore[misc]


def test_precomputed_table_rejects_tampering_and_embedding_width_drift() -> None:
    public_input = _public_input("first public text", "second public text")
    vectors = {
        public_input.history[0].semantic_text: (1.0, 0.0),
        public_input.history[1].semantic_text: (0.0, 1.0, 2.0),
        public_input.current_observation.content: (1.0, 1.0),
    }
    with pytest.raises(ValueError, match="width mismatch"):
        build_precomputed_public_embedding_table(
            embedder=_RecordingEmbedder(vectors),
            public_inputs=(public_input,),
        )

    stable_vectors = {text: (1.0, 0.0) for text in vectors}
    table = build_precomputed_public_embedding_table(
        embedder=_RecordingEmbedder(stable_vectors),
        public_inputs=(public_input,),
    )
    tampered = json.loads(table.to_json())
    tampered["records"][0]["embedding_hex"][0] = (0.5).hex()

    with pytest.raises(ValueError, match="record artifact_id mismatch"):
        PrecomputedPublicEmbeddingTable.from_payload(tampered)


def test_table_builder_rejects_unpinned_or_inconsistent_bge_provenance_before_embed() -> None:
    public_input = _public_input("public source")
    vectors = {
        public_input.history[0].semantic_text: (1.0, 0.0),
        public_input.current_observation.content: (0.0, 1.0),
    }

    unpinned = _RecordingEmbedder(vectors)
    unpinned.name = f"sentence-transformer:{BGE_M3_MODEL_ID}/model-adapter-v1"
    with pytest.raises(ValueError, match="exact BAAI/bge-m3 source and revision"):
        build_precomputed_public_embedding_table(
            embedder=unpinned,
            public_inputs=(public_input,),
        )
    assert unpinned.seen == []

    inconsistent = _RecordingEmbedder(vectors)
    inconsistent.model_revision = "0" * 40
    with pytest.raises(ValueError, match="model_revision disagrees"):
        build_precomputed_public_embedding_table(
            embedder=inconsistent,
            public_inputs=(public_input,),
        )
    assert inconsistent.seen == []

    inconsistent_weights = _RecordingEmbedder(vectors)
    inconsistent_weights.weights_sha256 = "0" * 64
    with pytest.raises(ValueError, match="weights_sha256 disagrees"):
        build_precomputed_public_embedding_table(
            embedder=inconsistent_weights,
            public_inputs=(public_input,),
        )
    assert inconsistent_weights.seen == []


def test_table_loader_rejects_duplicate_keys_and_noncanonical_raw_bytes(tmp_path) -> None:
    public_input = _public_input("public source")
    vectors = {
        public_input.history[0].semantic_text: (1.0, 0.0),
        public_input.current_observation.content: (0.0, 1.0),
    }
    table = build_precomputed_public_embedding_table(
        embedder=_RecordingEmbedder(vectors),
        public_inputs=(public_input,),
    )

    duplicate_raw = table.to_json().replace(
        '"embedding_width":2',
        '"embedding_width":2,"embedding_width":2',
        1,
    )
    assert duplicate_raw != table.to_json()
    with pytest.raises(ValueError, match="duplicate JSON key: embedding_width"):
        PrecomputedPublicEmbeddingTable.from_json(duplicate_raw)

    noncanonical_path = tmp_path / "noncanonical.json"
    noncanonical_path.write_text(
        json.dumps(table.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="exact canonical JSON bytes"):
        load_precomputed_public_embedding_table(noncanonical_path)

    crlf_path = tmp_path / "crlf.json"
    crlf_path.write_bytes(table.to_json().replace("\n", "\r\n").encode("utf-8"))
    with pytest.raises(ValueError, match="exact canonical JSON bytes"):
        load_precomputed_public_embedding_table(crlf_path)


def test_table_payload_binds_explicit_source_and_revision_fields() -> None:
    public_input = _public_input("public source")
    vectors = {
        public_input.history[0].semantic_text: (1.0, 0.0),
        public_input.current_observation.content: (0.0, 1.0),
    }
    table = build_precomputed_public_embedding_table(
        embedder=_RecordingEmbedder(vectors),
        public_inputs=(public_input,),
    )
    tampered = table.to_payload()
    tampered["source_model_revision"] = "0" * 40

    with pytest.raises(ValueError, match="source_model_revision mismatch"):
        PrecomputedPublicEmbeddingTable.from_payload(tampered)


def test_test_only_table_identity_is_explicitly_non_bge_and_stays_content_addressed() -> None:
    public_input = _public_input("public source")
    vectors = {
        public_input.history[0].semantic_text: (1.0, 0.0),
        public_input.current_observation.content: (0.0, 1.0),
    }
    formal = build_precomputed_public_embedding_table(
        embedder=_RecordingEmbedder(vectors),
        public_inputs=(public_input,),
    )
    test_only = PrecomputedPublicEmbeddingTable(
        source_embedder_name="fake-test-only/product-horizon",
        embedding_width=formal.embedding_width,
        records=formal.records,
    )
    loaded = PrecomputedPublicEmbeddingTable.from_json(test_only.to_json())

    assert loaded.source_model_id == "fake-test-only/product-horizon"
    assert loaded.source_model_revision is None
    assert loaded.source_weights_sha256 is None
    assert loaded.source_sentence_transformers_version is None
    assert loaded.artifact_id == test_only.artifact_id
    with pytest.raises(ValueError, match="canonical slug"):
        PrecomputedPublicEmbeddingTable(
            source_embedder_name="fake-test-only/Not Canonical",
            embedding_width=formal.embedding_width,
            records=formal.records,
        )


def test_legacy_unweighted_v2_table_identity_remains_strictly_loadable() -> None:
    public_input = _public_input("public source")
    vectors = {
        public_input.history[0].semantic_text: (1.0, 0.0),
        public_input.current_observation.content: (0.0, 1.0),
    }
    current = build_precomputed_public_embedding_table(
        embedder=_RecordingEmbedder(vectors),
        public_inputs=(public_input,),
    )
    legacy = PrecomputedPublicEmbeddingTable(
        source_embedder_name=(
            f"sentence-transformer:{BGE_M3_MODEL_ID}"
            f"@revision:{BGE_M3_MODEL_REVISION}/model-adapter-v1"
        ),
        embedding_width=current.embedding_width,
        records=current.records,
    )

    loaded = PrecomputedPublicEmbeddingTable.from_json(legacy.to_json())

    assert loaded.source_model_revision == BGE_M3_MODEL_REVISION
    assert loaded.source_weights_sha256 is None
    assert loaded.source_sentence_transformers_version is None
    assert loaded.to_json() == legacy.to_json()
