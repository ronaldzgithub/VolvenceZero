from __future__ import annotations

import hashlib
import io
import json
from dataclasses import FrozenInstanceError

import pytest

from lifeform_domain_emogpt.lab import RelationshipAction, canonical_json, sha256_json
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_product_baseline_dispatcher import (
    DEFAULT_PRODUCT_BASELINE_BGE_REVISION,
    DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION,
    DEFAULT_PRODUCT_BASELINE_BGE_WEIGHTS_SHA256,
    DEFAULT_PRODUCT_BASELINE_MODEL_REVISION,
    ProductBaselineCurrentObservationLineage,
    ProductBaselineDecisionBoundary,
    ProductBaselineDispatcherConfig,
    ProductBaselineDispatcherReceivedResponse,
    ProductBaselineDispatcherRequest,
    ProductBaselineHistoryBlockLineage,
    ProductBaselineSemanticMode,
    RevisionPinnedCachedPublicSemanticEmbedder,
    build_product_baseline_dispatcher_suite,
    export_live_public_embedding_table,
    parse_product_baseline_dispatcher_response_line,
    serve_product_baseline_jsonl,
    validate_product_baseline_dispatcher_suite,
)
from lifeform_evolution.relationship_lab_product_baselines import (
    FrozenProductChatMessage,
    ProductBaselineArm,
    ProductBaselineInput,
    ProductBaselineTokenBudget,
    ProductCurrentObservation,
    ProductPublicHistoryBlock,
    RelationshipProductBaselineSuite,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    PrecomputedPublicEmbeddingRecord,
    PrecomputedPublicEmbeddingTable,
    load_precomputed_public_embedding_table,
    write_precomputed_public_embedding_table,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _assistant_outcome(label: str) -> str:
    return canonical_json(
        {
            "action_id": RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
            "observed_outcome_id": "felt_heard",
            "rendered_user_reaction": f"public reaction {label}",
        }
    )


class _FakeResidentPolicy:
    model_id = "fake-resident-qwen"
    weights_sha256 = "1" * 64
    generation_config_sha256 = "2" * 64
    tokenizer_id = "fake-exact-tokenizer@revision:test"

    def __init__(self, *, invalid_seed: int | None = None, failing_seed: int | None = None) -> None:
        self.max_new_tokens = 8
        self.invalid_seed = invalid_seed
        self.failing_seed = failing_seed
        self.calls: list[int] = []

    @staticmethod
    def _count(messages: tuple[tuple[str, str], ...]) -> int:
        return 3 + sum(len(role.encode("utf-8")) + len(content.encode("utf-8")) for role, content in messages)

    def count_message_tokens(
        self,
        *,
        messages: tuple[FrozenProductChatMessage, ...],
    ) -> int:
        return self._count(tuple((message.role, message.content) for message in messages))

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        self.calls.append(seed)
        if seed == self.failing_seed:
            raise RuntimeError("synthetic resident policy failure")
        chosen = (
            None
            if seed == self.invalid_seed
            else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
        )
        raw_output = (
            "invalid model output"
            if chosen is None
            else canonical_json({"action_id": chosen.value})
        )
        return StatelessActionCompletion(
            raw_output=raw_output,
            chosen_action_id=chosen,
            prompt_tokens=self._count(
                tuple((message["role"], message["content"]) for message in messages)
            ),
            completion_tokens=3,
        )


class _FakePublicEmbedder:
    name = (
        "fake-public-semantic@revision:"
        + DEFAULT_PRODUCT_BASELINE_BGE_REVISION
        + "/precomputed-public-table"
    )

    def embed(self, text: str) -> tuple[float, ...]:
        total = float(sum(text.encode("utf-8")))
        return (total + 1.0, float(len(text.encode("utf-8"))) + 1.0)


class _FlushRecordingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _public_input() -> ProductBaselineInput:
    return ProductBaselineInput(
        history=(
            ProductPublicHistoryBlock(
                ordinal=0,
                exchange_id="public-history-session-00",
                user_messages=("first public context", "first public user message"),
                assistant_outcome=_assistant_outcome("first"),
            ),
            ProductPublicHistoryBlock(
                ordinal=1,
                exchange_id="public-history-session-01",
                user_messages=("second public context", "second public user message"),
                assistant_outcome=_assistant_outcome("second"),
            ),
            ProductPublicHistoryBlock(
                ordinal=2,
                exchange_id="public-history-session-02",
                user_messages=("third public context", "third public user message"),
                assistant_outcome=_assistant_outcome("third"),
            ),
        ),
        current_observation=ProductCurrentObservation(content="current public observation"),
    )


def _request(
    *,
    nonce: str,
    arm: ProductBaselineArm,
    seed: int,
    top_k: int | None,
) -> ProductBaselineDispatcherRequest:
    public_input = _public_input()
    return ProductBaselineDispatcherRequest(
        nonce=nonce,
        arm=arm,
        public_plan_artifact_id=_digest("public-plan"),
        subject_scope=_digest("subject-scope"),
        decision_boundary=ProductBaselineDecisionBoundary(
            current_session_id="public-decision-session-03",
            decision_id="public-decision-03",
            decision_index=3,
        ),
        ordered_source_session_ids=(
            *(block.exchange_id for block in public_input.history),
            "public-decision-session-03",
        ),
        ordered_source_block_artifact_ids=tuple(
            block.artifact_id for block in public_input.history
        ),
        public_ledger_artifact_id=_digest("public-ledger"),
        public_input=public_input,
        history_block_lineage=tuple(
            ProductBaselineHistoryBlockLineage(
                ordinal=block.ordinal,
                block_artifact_id=block.artifact_id,
                public_ledger_entry_artifact_id=_digest(f"ledger-entry-{block.ordinal}"),
            )
            for block in public_input.history
        ),
        current_observation_lineage=ProductBaselineCurrentObservationLineage(
            observation_artifact_id=public_input.current_observation.artifact_id,
            public_ledger_entry_artifact_id=_digest("ledger-current"),
        ),
        seed=seed,
        top_k=top_k,
    )


def _suite(policy: _FakeResidentPolicy | None = None) -> RelationshipProductBaselineSuite:
    resident = policy or _FakeResidentPolicy()
    return RelationshipProductBaselineSuite(
        policy=resident,
        token_counter=resident,
        token_budget=ProductBaselineTokenBudget(
            context_window_tokens=32768,
            generation_reserve_tokens=8,
        ),
        semantic_embedder=_FakePublicEmbedder(),
    )


def _request_line(request: ProductBaselineDispatcherRequest) -> str:
    return canonical_json(request.to_payload()) + "\n"


def test_request_round_trip_binds_public_plan_ledger_and_rejects_truth_or_tampering() -> None:
    request = _request(
        nonce="native-1",
        arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        seed=7,
        top_k=None,
    )
    parsed = ProductBaselineDispatcherRequest.from_payload(request.to_payload())

    assert parsed == request
    assert parsed.to_payload() == request.to_payload()
    with pytest.raises(FrozenInstanceError):
        parsed.seed = 9  # type: ignore[misc]

    forbidden = request.to_payload()
    forbidden["preferred_action_id"] = "respect_space_with_return_option"
    with pytest.raises(ValueError, match="extra=.*preferred_action_id"):
        ProductBaselineDispatcherRequest.from_payload(forbidden)

    missing_ledger = request.to_payload()
    del missing_ledger["public_ledger_artifact_id"]
    with pytest.raises(ValueError, match="missing=.*public_ledger_artifact_id"):
        ProductBaselineDispatcherRequest.from_payload(missing_ledger)

    tampered_text = request.to_payload()
    tampered_text["public_input"]["history"][0]["user_messages"][0] = (  # type: ignore[index]
        "tampered public text"
    )
    with pytest.raises(ValueError, match="semantic_text_sha256 mismatch"):
        ProductBaselineDispatcherRequest.from_payload(tampered_text)

    tampered_lineage = request.to_payload()
    tampered_lineage["history_block_lineage"][0]["block_artifact_id"] = "f" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="history_block_lineage"):
        ProductBaselineDispatcherRequest.from_payload(tampered_lineage)

    split_exchange_lineage = request.to_payload()
    split_exchange_lineage["ordered_source_session_ids"][0] = "unbound-message-session"  # type: ignore[index]
    with pytest.raises(ValueError, match="one complete history exchange"):
        ProductBaselineDispatcherRequest.from_payload(split_exchange_lineage)


def test_resident_loop_constructs_once_preserves_nonce_order_and_runs_both_arms() -> None:
    policy = _FakeResidentPolicy()
    suite = _suite(policy)
    factory_calls = 0

    def factory() -> RelationshipProductBaselineSuite:
        nonlocal factory_calls
        factory_calls += 1
        return suite

    native = _request(
        nonce="native-first",
        arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        seed=11,
        top_k=None,
    )
    rag = _request(
        nonce="rag-second",
        arm=ProductBaselineArm.SELECTIVE_SEMANTIC_RAG,
        seed=12,
        top_k=8,
    )
    output = _FlushRecordingStream()
    status = serve_product_baseline_jsonl(
        input_stream=io.StringIO(_request_line(native) + _request_line(rag)),
        output_stream=output,
        error_stream=io.StringIO(),
        suite_factory=factory,
    )
    responses = tuple(json.loads(line) for line in output.getvalue().splitlines())

    assert status == 0
    assert factory_calls == 1
    assert policy.calls == [11, 12]
    assert tuple(response["nonce"] for response in responses) == (
        "native-first",
        "rag-second",
    )
    assert tuple(response["result"]["arm"] for response in responses) == (
        ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY.value,
        ProductBaselineArm.SELECTIVE_SEMANTIC_RAG.value,
    )
    assert all(response["status"] == "ok" for response in responses)
    assert all(response["valid"] for response in responses)
    assert all(
        response["generation_config_sha256"] == policy.generation_config_sha256
        for response in responses
    )
    assert all(
        response["token_receipt"]["generation_reserve_tokens"] == 8
        for response in responses
    )
    rag_receipt = responses[1]["result"]["retrieval_receipt"]
    assert rag_receipt["requested_top_k"] == 8
    assert rag_receipt["effective_top_k"] == len(_public_input().history) == 3
    assert output.flush_count == 2
    parsed_responses = tuple(
        parse_product_baseline_dispatcher_response_line(line + "\n")
        for line in output.getvalue().splitlines()
    )
    assert all(
        isinstance(response, ProductBaselineDispatcherReceivedResponse)
        for response in parsed_responses
    )
    assert tuple(response.nonce for response in parsed_responses) == (
        "native-first",
        "rag-second",
    )


def test_response_parser_rejects_noncanonical_bytes_and_nested_artifact_tampering() -> None:
    request = _request(
        nonce="response-tamper",
        arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        seed=17,
        top_k=None,
    )
    output = _FlushRecordingStream()
    status = serve_product_baseline_jsonl(
        input_stream=io.StringIO(_request_line(request)),
        output_stream=output,
        error_stream=io.StringIO(),
        suite_factory=lambda: _suite(_FakeResidentPolicy()),
    )
    assert status == 0

    parsed = parse_product_baseline_dispatcher_response_line(output.getvalue())
    assert isinstance(parsed, ProductBaselineDispatcherReceivedResponse)
    assert parsed.result_payload["artifact_id"] == parsed.result_artifact_id

    payload = json.loads(output.getvalue())
    payload["result"]["action_completion"]["raw_output"] = "tampered"
    tampered = canonical_json(payload) + "\n"
    with pytest.raises(ValueError, match="action_completion artifact_id mismatch"):
        parse_product_baseline_dispatcher_response_line(tampered)


def test_response_parser_rejects_self_consistent_effective_top_k_drift() -> None:
    request = _request(
        nonce="rag-effective-k-tamper",
        arm=ProductBaselineArm.SELECTIVE_SEMANTIC_RAG,
        seed=18,
        top_k=8,
    )
    output = _FlushRecordingStream()
    status = serve_product_baseline_jsonl(
        input_stream=io.StringIO(_request_line(request)),
        output_stream=output,
        error_stream=io.StringIO(),
        suite_factory=lambda: _suite(_FakeResidentPolicy()),
    )
    assert status == 0
    payload = json.loads(output.getvalue())
    retrieval = payload["result"]["retrieval_receipt"]
    retrieval["effective_top_k"] = 2
    retrieval["artifact_id"] = sha256_json(
        {key: value for key, value in retrieval.items() if key != "artifact_id"}
    )
    result = payload["result"]
    result["artifact_id"] = sha256_json(
        {key: value for key, value in result.items() if key != "artifact_id"}
    )

    with pytest.raises(ValueError, match="effective_top_k does not equal min"):
        parse_product_baseline_dispatcher_response_line(canonical_json(payload) + "\n")

    noncanonical = json.dumps(json.loads(output.getvalue())) + "\n"
    with pytest.raises(ValueError, match="canonical JSON"):
        parse_product_baseline_dispatcher_response_line(noncanonical)


def test_invalid_model_output_is_an_ordered_valid_false_result_not_a_second_parser() -> None:
    policy = _FakeResidentPolicy(invalid_seed=99)
    request = _request(
        nonce="invalid-output",
        arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        seed=99,
        top_k=None,
    )
    output = _FlushRecordingStream()

    status = serve_product_baseline_jsonl(
        input_stream=io.StringIO(_request_line(request)),
        output_stream=output,
        error_stream=io.StringIO(),
        suite_factory=lambda: _suite(policy),
    )
    response = json.loads(output.getvalue())

    assert status == 0
    assert response["status"] == "ok"
    assert response["nonce"] == "invalid-output"
    assert response["action_id"] is None
    assert response["valid"] is False
    assert response["result"]["action_completion"]["raw_output"] == "invalid model output"


def test_noncanonical_or_failing_request_emits_one_fatal_flush_and_stops() -> None:
    policy = _FakeResidentPolicy(failing_seed=22)
    first = _request(
        nonce="first-ok",
        arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        seed=21,
        top_k=None,
    )
    failing = _request(
        nonce="second-fails",
        arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        seed=22,
        top_k=None,
    )
    never_run = _request(
        nonce="third-never-runs",
        arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        seed=23,
        top_k=None,
    )
    output = _FlushRecordingStream()
    errors = _FlushRecordingStream()

    status = serve_product_baseline_jsonl(
        input_stream=io.StringIO(
            _request_line(first) + _request_line(failing) + _request_line(never_run)
        ),
        output_stream=output,
        error_stream=errors,
        suite_factory=lambda: _suite(policy),
    )
    responses = tuple(json.loads(line) for line in output.getvalue().splitlines())

    assert status == 1
    assert policy.calls == [21, 22]
    assert tuple(response["status"] for response in responses) == ("ok", "fatal")
    assert responses[-1]["nonce"] == "second-fails"
    assert responses[-1]["error_type"] == "RuntimeError"
    assert "synthetic resident policy failure" in responses[-1]["error_message"]
    assert "Traceback" in errors.getvalue()
    assert output.flush_count == 2

    noncanonical_output = _FlushRecordingStream()
    pretty = json.dumps(first.to_payload(), sort_keys=False) + "\n"
    noncanonical_status = serve_product_baseline_jsonl(
        input_stream=io.StringIO(pretty + _request_line(never_run)),
        output_stream=noncanonical_output,
        error_stream=io.StringIO(),
        suite_factory=lambda: _suite(_FakeResidentPolicy()),
    )
    fatal = json.loads(noncanonical_output.getvalue())
    assert noncanonical_status == 1
    assert fatal["status"] == "fatal"
    assert "canonical JSON" in fatal["error_message"]


def test_dispatcher_rejects_split_execution_identity_and_under_reserved_generation() -> None:
    policy = _FakeResidentPolicy()
    split_counter = _FakeResidentPolicy()
    split_suite = RelationshipProductBaselineSuite(
        policy=policy,
        token_counter=split_counter,
        token_budget=ProductBaselineTokenBudget(
            context_window_tokens=32768,
            generation_reserve_tokens=8,
        ),
        semantic_embedder=_FakePublicEmbedder(),
    )
    with pytest.raises(ValueError, match="same object"):
        validate_product_baseline_dispatcher_suite(split_suite)

    policy.max_new_tokens = 9
    with pytest.raises(ValueError, match="generation reserve"):
        validate_product_baseline_dispatcher_suite(_suite(policy))


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeSentenceModel:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(
        self,
        text: str,
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> _FakeVector:
        assert normalize_embeddings
        assert convert_to_numpy
        assert not show_progress_bar
        self.seen.append(text)
        return _FakeVector([float(len(text)), 1.0])


def test_live_revision_pinned_bge_caches_exact_public_text_and_exports_table(tmp_path) -> None:
    weight_bytes = b"dispatcher fake bge weight bytes"
    snapshot = tmp_path / "bge-snapshot"
    snapshot.mkdir()
    (snapshot / "pytorch_model.bin").write_bytes(weight_bytes)
    weights_sha256 = hashlib.sha256(weight_bytes).hexdigest()
    model = _FakeSentenceModel()
    factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def factory(*args: object, **kwargs: object) -> _FakeSentenceModel:
        factory_calls.append((args, kwargs))
        return model

    embedder = RevisionPinnedCachedPublicSemanticEmbedder(
        model_source="BAAI/bge-m3",
        model_revision=DEFAULT_PRODUCT_BASELINE_BGE_REVISION,
        weights_sha256=weights_sha256,
        sentence_transformers_version=(
            DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION
        ),
        device="cuda",
        model_factory=factory,
        snapshot_path=snapshot,
        runtime_version_resolver=lambda _name: (
            DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION
        ),
    )
    first = embedder.embed("first actual public reaction")
    replay = embedder.embed("first actual public reaction")
    second = embedder.embed("second actual public reaction")

    assert first == replay
    assert first != second
    assert len(factory_calls) == 1
    assert factory_calls[0][0] == (str(snapshot.resolve()),)
    assert factory_calls[0][1]["local_files_only"] is True
    assert model.seen == ["first actual public reaction", "second actual public reaction"]
    assert embedder.cached_text_count == 2
    assert "live-public-exact-text-cache" in embedder.name
    assert DEFAULT_PRODUCT_BASELINE_BGE_REVISION in embedder.name
    assert weights_sha256 in embedder.name
    assert DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION in embedder.name

    policy = _FakeResidentPolicy()
    suite = RelationshipProductBaselineSuite(
        policy=policy,
        token_counter=policy,
        token_budget=ProductBaselineTokenBudget(
            context_window_tokens=32768,
            generation_reserve_tokens=8,
        ),
        semantic_embedder=embedder,
    )
    output_path = tmp_path / "actual_public_embeddings.json"
    export_live_public_embedding_table(suite=suite, path=output_path)
    loaded = load_precomputed_public_embedding_table(output_path)

    assert loaded.source_embedder_name == embedder.name
    assert loaded.source_weights_sha256 == weights_sha256
    assert (
        loaded.source_sentence_transformers_version
        == DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION
    )
    assert len(loaded.records) == 2
    assert {record.text for record in loaded.records} == {
        "first actual public reaction",
        "second actual public reaction",
    }
    with pytest.raises(FileExistsError):
        export_live_public_embedding_table(suite=suite, path=output_path)

    rag_result = suite.run_selective_semantic_rag(
        public_input=_public_input(),
        seed=47,
        top_k=2,
    )
    assert rag_result.retrieval_receipt.embedder_id == embedder.name
    assert weights_sha256 in rag_result.retrieval_receipt.embedder_id


def test_formal_config_binds_revisions_and_deterministic_generation_without_loading_models(
    monkeypatch,
) -> None:
    import lifeform_evolution.relationship_lab_product_baseline_dispatcher as dispatcher

    captured_policy_kwargs: dict[str, object] = {}
    policy = _FakeResidentPolicy()

    def fake_policy(**kwargs: object) -> _FakeResidentPolicy:
        captured_policy_kwargs.update(kwargs)
        policy.max_new_tokens = int(kwargs["max_new_tokens"])
        return policy

    live_embedder = _FakePublicEmbedder()
    captured_bge_kwargs: dict[str, object] = {}

    def fake_live_embedder(**kwargs: object) -> _FakePublicEmbedder:
        captured_bge_kwargs.update(kwargs)
        return live_embedder

    monkeypatch.setattr(dispatcher, "HFStatelessRelationshipActionPolicy", fake_policy)
    monkeypatch.setattr(
        dispatcher,
        "RevisionPinnedCachedPublicSemanticEmbedder",
        fake_live_embedder,
    )
    config = ProductBaselineDispatcherConfig(
        prefill_chunk_size=2048,
        generation_use_cache=True,
        schema_constrained_decoding=True,
        semantic_mode=ProductBaselineSemanticMode.LIVE_BGE_M3_CACHED,
    )
    suite = build_product_baseline_dispatcher_suite(config)

    assert config.model_revision == DEFAULT_PRODUCT_BASELINE_MODEL_REVISION
    assert config.bge_model_revision == DEFAULT_PRODUCT_BASELINE_BGE_REVISION
    assert config.bge_weights_sha256 == DEFAULT_PRODUCT_BASELINE_BGE_WEIGHTS_SHA256
    assert (
        config.bge_sentence_transformers_version
        == DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION
    )
    assert config.context_window_tokens == 32768
    assert config.generation_reserve_tokens == 64
    assert captured_policy_kwargs == {
        "model_source": config.model_source,
        "model_id": config.model_id,
        "model_revision": DEFAULT_PRODUCT_BASELINE_MODEL_REVISION,
        "device": "cuda",
        "torch_dtype": "float16",
        "local_files_only": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 64,
        "prefill_chunk_size": 2048,
        "generation_use_cache": True,
        "schema_constrained_decoding": True,
    }
    assert captured_bge_kwargs["model_revision"] == DEFAULT_PRODUCT_BASELINE_BGE_REVISION
    assert (
        captured_bge_kwargs["weights_sha256"]
        == DEFAULT_PRODUCT_BASELINE_BGE_WEIGHTS_SHA256
    )
    assert (
        captured_bge_kwargs["sentence_transformers_version"]
        == DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION
    )
    assert suite.policy is suite.token_counter is policy
    assert suite.token_budget.generation_reserve_tokens == policy.max_new_tokens == 64


def test_dispatcher_config_rejects_unbounded_or_uncached_chunked_prefill() -> None:
    with pytest.raises(ValueError, match="generation_use_cache=True"):
        ProductBaselineDispatcherConfig(
            prefill_chunk_size=2048,
            generation_use_cache=False,
        )
    with pytest.raises(ValueError, match="positive integer"):
        ProductBaselineDispatcherConfig(
            prefill_chunk_size=True,
            generation_use_cache=True,
        )
    with pytest.raises(TypeError, match="schema_constrained_decoding"):
        ProductBaselineDispatcherConfig(schema_constrained_decoding=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weights_sha256"):
        ProductBaselineDispatcherConfig(bge_weights_sha256="not-a-digest")
    with pytest.raises(ValueError, match="canonical package version"):
        ProductBaselineDispatcherConfig(bge_sentence_transformers_version="not a version")
    with pytest.raises(ValueError, match="must be exactly BAAI/bge-m3"):
        ProductBaselineDispatcherConfig(bge_model_source="other/embedder")


def test_dispatcher_schema_constraint_is_legacy_off_and_requires_explicit_cli_flag() -> None:
    import lifeform_evolution.relationship_lab_product_baseline_dispatcher as dispatcher

    assert ProductBaselineDispatcherConfig().schema_constrained_decoding is False
    assert dispatcher._parse_args([]).schema_constrained_decoding is False
    assert (
        dispatcher._parse_args(["--schema-constrained-decoding"]).schema_constrained_decoding
        is True
    )
    defaults = dispatcher._parse_args([])
    assert defaults.bge_weights_sha256 == DEFAULT_PRODUCT_BASELINE_BGE_WEIGHTS_SHA256
    assert (
        defaults.bge_sentence_transformers_version
        == DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION
    )
    explicit = dispatcher._parse_args(
        [
            "--bge-weights-sha256",
            "a" * 64,
            "--bge-sentence-transformers-version",
            "9.8.7",
        ]
    )
    assert explicit.bge_weights_sha256 == "a" * 64
    assert explicit.bge_sentence_transformers_version == "9.8.7"


def test_dispatcher_precomputed_mode_strictly_accepts_legacy_v2_identity(
    tmp_path,
    monkeypatch,
) -> None:
    import lifeform_evolution.relationship_lab_product_baseline_dispatcher as dispatcher

    record = PrecomputedPublicEmbeddingRecord(
        text="legacy public text",
        embedding_hex=((1.0).hex(),),
    )
    table = PrecomputedPublicEmbeddingTable(
        source_embedder_name=(
            "sentence-transformer:BAAI/bge-m3@revision:"
            f"{DEFAULT_PRODUCT_BASELINE_BGE_REVISION}/model-adapter-v1"
        ),
        embedding_width=1,
        records=(record,),
    )
    table_path = tmp_path / "legacy-v2-table.json"
    write_precomputed_public_embedding_table(table, path=table_path)
    policy = _FakeResidentPolicy()
    monkeypatch.setattr(
        dispatcher,
        "HFStatelessRelationshipActionPolicy",
        lambda **_kwargs: policy,
    )

    suite = build_product_baseline_dispatcher_suite(
        ProductBaselineDispatcherConfig(
            semantic_mode=ProductBaselineSemanticMode.PRECOMPUTED,
            precomputed_embedding_table_path=table_path,
        )
    )

    assert suite.policy is policy
    assert suite.semantic_embedder.name.startswith(table.source_embedder_name)
