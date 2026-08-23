from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from lifeform_domain_emogpt.lab import RelationshipAction, canonical_json, sha256_json
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_product_baselines import (
    FrozenProductChatMessage,
    ProductBaselineArm,
    ProductBaselineContextWindowError,
    ProductBaselineInput,
    ProductBaselineTokenBudget,
    ProductCurrentObservation,
    ProductPublicHistoryBlock,
    RelationshipProductBaselineSuite,
    product_baseline_prompt_path,
)


class _FakeExactTokenizer:
    tokenizer_id = "fake-chat-template-v1"

    def count_message_tokens(self, *, messages: tuple[FrozenProductChatMessage, ...]) -> int:
        return self._count(tuple((message.role, message.content) for message in messages))

    @staticmethod
    def _count(messages: tuple[tuple[str, str], ...]) -> int:
        # Three generation-template tokens plus two framing tokens per message.
        return 3 + sum(2 + len(content.split()) for _role, content in messages)


class _FakeContextualPolicy:
    model_id = "fake-shared-contextual-policy"
    weights_sha256 = sha256_json("fake-product-policy-weights")
    prompt_sha256 = sha256_json("unused-stateless-prompt")
    generation_config_sha256 = sha256_json("fake-product-generation-config")
    tokenizer_id = _FakeExactTokenizer.tokenizer_id

    def __init__(
        self,
        *,
        chosen_action_id: RelationshipAction | None = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
        prompt_token_offset: int = 0,
        completion_tokens: int = 2,
        max_new_tokens: int = 4,
    ) -> None:
        self.chosen_action_id = chosen_action_id
        self.prompt_token_offset = prompt_token_offset
        self.completion_tokens = completion_tokens
        self.max_new_tokens = max_new_tokens
        self.calls: list[tuple[tuple[tuple[str, str], ...], int]] = []

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        raise AssertionError("product contextual baselines must not call the stateless policy entrypoint")

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        frozen_messages = tuple((message["role"], message["content"]) for message in messages)
        self.calls.append((frozen_messages, seed))
        raw_output = (
            json.dumps({"action_id": self.chosen_action_id.value}, separators=(",", ":"))
            if self.chosen_action_id is not None
            else "not valid action JSON"
        )
        return StatelessActionCompletion(
            raw_output=raw_output,
            chosen_action_id=self.chosen_action_id,
            prompt_tokens=_FakeExactTokenizer._count(frozen_messages) + self.prompt_token_offset,
            completion_tokens=self.completion_tokens,
        )

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class _RecordingEmbedder:
    name = "fake-semantic-embedder-v1"

    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self._vectors = vectors
        self.seen: list[str] = []

    def embed(self, text: str) -> tuple[float, ...]:
        self.seen.append(text)
        return self._vectors[text]


def _assistant_outcome(
    label: str,
    *,
    action: RelationshipAction = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
    outcome_id: str = "felt_heard",
) -> str:
    return canonical_json(
        {
            "action_id": action.value,
            "observed_outcome_id": outcome_id,
            "rendered_user_reaction": f"public reaction {label}",
        }
    )


def _public_input(*contents: str) -> ProductBaselineInput:
    return ProductBaselineInput(
        history=tuple(
            ProductPublicHistoryBlock(
                ordinal=index,
                exchange_id=f"public-exchange-{index}",
                user_messages=(content,),
                assistant_outcome=_assistant_outcome(str(index)),
            )
            for index, content in enumerate(contents)
        ),
        current_observation=ProductCurrentObservation(content="current public observation"),
    )


def _exchange_message_contents(
    blocks: tuple[ProductPublicHistoryBlock, ...],
) -> tuple[str, ...]:
    return tuple(
        content
        for block in blocks
        for content in (
            *block.user_messages,
            block.assistant_action_message,
            block.user_outcome_feedback_message,
        )
    )


def _suite(
    *,
    policy: _FakeContextualPolicy | None = None,
    budget: ProductBaselineTokenBudget | None = None,
    embedder: _RecordingEmbedder | None = None,
) -> RelationshipProductBaselineSuite:
    return RelationshipProductBaselineSuite(
        policy=policy or _FakeContextualPolicy(),
        token_counter=_FakeExactTokenizer(),
        token_budget=budget or ProductBaselineTokenBudget(
            context_window_tokens=1024,
            generation_reserve_tokens=8,
        ),
        semantic_embedder=embedder,
    )


def _prompt_and_current_tokens(arm: ProductBaselineArm, public_input: ProductBaselineInput) -> int:
    prompt = product_baseline_prompt_path(arm).read_text(encoding="utf-8").strip()
    messages = (
        ("system", prompt),
        ("user", public_input.current_observation.content),
    )
    return _FakeExactTokenizer._count(messages)


def test_public_input_is_frozen_chronological_and_content_addressed() -> None:
    public_input = _public_input("first public turn", "second public turn")
    identical = _public_input("first public turn", "second public turn")
    changed = _public_input("first public turn", "changed second turn")

    assert public_input.artifact_id == identical.artifact_id
    assert public_input.artifact_id != changed.artifact_id
    assert public_input.to_payload()["artifact_id"] == public_input.artifact_id
    assert public_input.history[0].semantic_text_sha256 == hashlib.sha256(
        public_input.history[0].semantic_text.encode("utf-8")
    ).hexdigest()
    with pytest.raises(FrozenInstanceError):
        public_input.history[0].assistant_outcome = "mutated"  # type: ignore[misc]
    with pytest.raises(ValueError, match="tuple"):
        ProductBaselineInput(  # type: ignore[arg-type]
            history=[public_input.history[0]],
            current_observation=public_input.current_observation,
        )
    with pytest.raises(ValueError, match="contiguous and chronological"):
        ProductBaselineInput(
            history=(public_input.history[1], public_input.history[0]),
            current_observation=public_input.current_observation,
        )
    with pytest.raises(ValueError, match="non-empty tuple"):
        ProductPublicHistoryBlock(
            ordinal=0,
            exchange_id="incomplete-exchange",
            user_messages=(),
            assistant_outcome="orphaned assistant outcome",
        )
    with pytest.raises(ValueError, match="canonical JSON"):
        ProductPublicHistoryBlock(
            ordinal=0,
            exchange_id="invalid-json",
            user_messages=("public turn",),
            assistant_outcome="not-json",
        )
    with pytest.raises(ValueError, match="contain exactly"):
        ProductPublicHistoryBlock(
            ordinal=0,
            exchange_id="extra-field",
            user_messages=("public turn",),
            assistant_outcome=canonical_json(
                {
                    "action_id": RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
                    "observed_outcome_id": "felt_heard",
                    "rendered_user_reaction": "public reaction",
                    "unexpected": True,
                }
            ),
        )
    with pytest.raises(ValueError, match="canonical JSON serialization"):
        ProductPublicHistoryBlock(
            ordinal=0,
            exchange_id="non-canonical-json",
            user_messages=("public turn",),
            assistant_outcome=(
                '{"rendered_user_reaction":"public reaction",'
                '"observed_outcome_id":"felt_heard",'
                '"action_id":"stay_present_without_probe"}'
            ),
        )


def test_history_triple_is_not_an_assistant_few_shot_output_schema() -> None:
    public_input = _public_input("public history")
    policy = _FakeContextualPolicy()

    _suite(policy=policy).run_native_chronological_full_history(
        public_input=public_input,
        seed=16,
    )

    messages = policy.calls[0][0]
    assistant_messages = [json.loads(content) for role, content in messages if role == "assistant"]
    assert assistant_messages == [
        {"action_id": RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value}
    ]
    feedback_messages = [
        json.loads(content)
        for role, content in messages
        if role == "user" and content.startswith('{"observed_outcome_id"')
    ]
    assert feedback_messages == [
        {
            "observed_outcome_id": "felt_heard",
            "rendered_user_reaction": "public reaction 0",
        }
    ]
    assert all(
        set(payload) == {"action_id"}
        for payload in assistant_messages
    )


def test_native_full_history_drops_oldest_complete_exchanges_and_receipts_exact_budget() -> None:
    public_input = _public_input(
        "oldest one two three four",
        "middle one two three",
        "newest one two",
    )
    reserve = 7
    base_tokens = _prompt_and_current_tokens(
        ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        public_input,
    )
    # Each exchange expands atomically into its user messages plus paired
    # assistant outcome. Fit the newest two complete exchanges only.
    newest_two_increment = sum(
        2 + len(content.split())
        for content in _exchange_message_contents(public_input.history[1:])
    )
    budget = ProductBaselineTokenBudget(
        context_window_tokens=base_tokens + newest_two_increment + reserve,
        generation_reserve_tokens=reserve,
    )
    policy = _FakeContextualPolicy()
    result = _suite(policy=policy, budget=budget).run_native_chronological_full_history(
        public_input=public_input,
        seed=17,
    )

    assert len(policy.calls) == 1
    call_messages, call_seed = policy.calls[0]
    assert call_seed == 17
    assert tuple(content for _role, content in call_messages[1:-1]) == (
        _exchange_message_contents(public_input.history[1:])
    )
    assert result.action_completion.chosen_action_id is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
    assert result.action_completion.valid
    assert result.truncation_receipt.was_truncated
    assert (
        result.truncation_receipt.reason
        == "oldest_complete_exchange_units_until_budget_fit"
    )
    assert result.truncation_receipt.dropped_oldest_block_artifact_ids == (
        public_input.history[0].artifact_id,
    )
    assert result.context_receipt.included_block_artifact_ids == tuple(
        block.artifact_id for block in public_input.history[1:]
    )
    assert result.context_receipt.prompt_and_current_tokens == base_tokens
    assert result.context_receipt.final_prompt_tokens == (
        base_tokens + newest_two_increment
    )
    assert result.context_receipt.total_reserved_tokens == budget.context_window_tokens
    assert result.context_receipt.history_increment_tokens == newest_two_increment
    assert result.retrieval_receipt.selected_block_artifact_ids == tuple(
        block.artifact_id for block in public_input.history
    )
    assert result.retrieval_receipt.requested_top_k is None
    assert result.retrieval_receipt.effective_top_k == len(public_input.history)
    assert result.artifact_id == result.to_payload()["artifact_id"]


def test_native_full_history_without_truncation_preserves_every_block_and_is_replay_stable() -> None:
    public_input = _public_input("turn zero", "turn one")
    suite = _suite()

    first = suite.run_native_chronological_full_history(public_input=public_input, seed=23)
    second = suite.run_native_chronological_full_history(public_input=public_input, seed=23)

    assert not first.truncation_receipt.was_truncated
    assert first.truncation_receipt.dropped_oldest_block_artifact_ids == ()
    assert first.artifact_id == second.artifact_id
    assert first.context_receipt.rendered_messages_sha256 == second.context_receipt.rendered_messages_sha256
    with pytest.raises(FrozenInstanceError):
        first.seed = 99  # type: ignore[misc]


def test_semantic_rag_is_deterministic_top_k_less_than_n_and_renders_chronologically() -> None:
    public_input = _public_input("score point eight", "tie first", "tie second", "irrelevant")
    vectors = {
        public_input.current_observation.content: (1.0, 0.0),
        public_input.history[0].semantic_text: (0.8, 0.6),
        public_input.history[1].semantic_text: (1.0, 0.0),
        public_input.history[2].semantic_text: (1.0, 0.0),
        public_input.history[3].semantic_text: (0.0, 1.0),
    }
    embedder = _RecordingEmbedder(vectors)
    policy = _FakeContextualPolicy(chosen_action_id=RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION)
    suite = _suite(policy=policy, embedder=embedder)

    result = suite.run_selective_semantic_rag(public_input=public_input, seed=31, top_k=3)

    assert embedder.seen == [
        public_input.current_observation.content,
        *(block.semantic_text for block in public_input.history),
    ]
    ranked_ids = tuple(
        candidate.block_artifact_id for candidate in result.retrieval_receipt.ranked_candidates
    )
    assert ranked_ids == (
        public_input.history[1].artifact_id,
        public_input.history[2].artifact_id,
        public_input.history[0].artifact_id,
        public_input.history[3].artifact_id,
    )
    assert result.retrieval_receipt.selected_block_artifact_ids == ranked_ids[:3]
    assert result.retrieval_receipt.selected_chronological_block_artifact_ids == (
        public_input.history[0].artifact_id,
        public_input.history[1].artifact_id,
        public_input.history[2].artifact_id,
    )
    call_messages, _seed = policy.calls[0]
    assert tuple(content for _role, content in call_messages[1:-1]) == (
        _exchange_message_contents(public_input.history[:3])
    )
    assert result.action_completion.chosen_action_id is RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION

    all_selected = suite.run_selective_semantic_rag(
        public_input=public_input,
        seed=32,
        top_k=len(public_input.history) + 2,
    )
    assert all_selected.retrieval_receipt.requested_top_k == 6
    assert all_selected.retrieval_receipt.effective_top_k == 4
    assert all_selected.retrieval_receipt.selected_chronological_block_artifact_ids == tuple(
        block.artifact_id for block in public_input.history
    )
    with pytest.raises(ValueError, match=r"min\(requested_top_k, candidate_count\)"):
        replace(all_selected.retrieval_receipt, effective_top_k=3)


def test_semantic_rag_scores_and_renders_complete_paired_exchange_units() -> None:
    public_input = ProductBaselineInput(
        history=(
            ProductPublicHistoryBlock(
                ordinal=0,
                exchange_id="exchange-with-space-outcome",
                user_messages=("same public context", "same public user message"),
                assistant_outcome=_assistant_outcome(
                    "assistant respected space and the user relaxed",
                    action=RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                ),
            ),
            ProductPublicHistoryBlock(
                ordinal=1,
                exchange_id="exchange-with-probe-outcome",
                user_messages=("same public context", "same public user message"),
                assistant_outcome=_assistant_outcome(
                    "assistant probed and the user withdrew",
                    outcome_id="over_directive",
                ),
            ),
        ),
        current_observation=ProductCurrentObservation(content="current public observation"),
    )
    embedder = _RecordingEmbedder(
        {
            public_input.current_observation.content: (1.0, 0.0),
            public_input.history[0].semantic_text: (0.0, 1.0),
            public_input.history[1].semantic_text: (1.0, 0.0),
        }
    )
    policy = _FakeContextualPolicy()

    result = _suite(policy=policy, embedder=embedder).run_selective_semantic_rag(
        public_input=public_input,
        seed=33,
        top_k=1,
    )

    assert embedder.seen == [
        public_input.current_observation.content,
        public_input.history[0].semantic_text,
        public_input.history[1].semantic_text,
    ]
    assert result.retrieval_receipt.selected_block_artifact_ids == (
        public_input.history[1].artifact_id,
    )
    rendered_contents = tuple(content for _role, content in policy.calls[0][0][1:-1])
    assert rendered_contents == _exchange_message_contents(public_input.history[1:])
    assert public_input.history[1].assistant_action_message in rendered_contents
    assert public_input.history[1].user_outcome_feedback_message in rendered_contents
    assert public_input.history[0].user_outcome_feedback_message not in rendered_contents


def test_semantic_rag_passes_only_public_text_to_embedder() -> None:
    public_input = _public_input("public history alpha", "public history beta")
    embedder = _RecordingEmbedder(
        {
            public_input.current_observation.content: (1.0, 0.0),
            public_input.history[0].semantic_text: (1.0, 0.0),
            public_input.history[1].semantic_text: (0.0, 1.0),
        }
    )

    _suite(embedder=embedder).run_selective_semantic_rag(
        public_input=public_input,
        seed=5,
        top_k=1,
    )

    assert set(embedder.seen) == {
        public_input.current_observation.content,
        *(block.semantic_text for block in public_input.history),
    }
    assert all("owner" not in value and "evaluator" not in value and "truth" not in value for value in embedder.seen)


def test_prompt_current_and_reserve_must_fit_before_policy_call() -> None:
    public_input = _public_input("one history block")
    base_tokens = _prompt_and_current_tokens(
        ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        public_input,
    )
    policy = _FakeContextualPolicy()
    suite = _suite(
        policy=policy,
        budget=ProductBaselineTokenBudget(
            context_window_tokens=base_tokens + 3,
            generation_reserve_tokens=4,
        ),
    )

    with pytest.raises(ProductBaselineContextWindowError, match="system prompt"):
        suite.run_native_chronological_full_history(public_input=public_input, seed=1)
    assert policy.calls == []


def test_exact_counter_mismatch_and_generation_reserve_overflow_fail_loudly() -> None:
    public_input = _public_input("one history block", "another history block")
    mismatched_policy = _FakeContextualPolicy(prompt_token_offset=1)
    with pytest.raises(ValueError, match="counter disagrees"):
        _suite(policy=mismatched_policy).run_native_chronological_full_history(
            public_input=public_input,
            seed=2,
        )

    reserve_overflow_policy = _FakeContextualPolicy(completion_tokens=9)
    with pytest.raises(ValueError, match="generation token reserve"):
        _suite(
            policy=reserve_overflow_policy,
            budget=ProductBaselineTokenBudget(
                context_window_tokens=1024,
                generation_reserve_tokens=8,
            ),
        ).run_native_chronological_full_history(public_input=public_input, seed=2)


def test_suite_rejects_tokenizer_drift_and_under_reserve_before_generation() -> None:
    policy = _FakeContextualPolicy()
    policy.tokenizer_id = "different-generation-tokenizer"
    with pytest.raises(ValueError, match="tokenizer identities differ"):
        _suite(policy=policy)
    assert policy.calls == []

    under_reserved_policy = _FakeContextualPolicy(max_new_tokens=9)
    with pytest.raises(ValueError, match="reserve must cover"):
        _suite(
            policy=under_reserved_policy,
            budget=ProductBaselineTokenBudget(
                context_window_tokens=1024,
                generation_reserve_tokens=8,
            ),
        )
    assert under_reserved_policy.calls == []

    mutable_policy = _FakeContextualPolicy()
    suite = _suite(policy=mutable_policy)
    mutable_policy.tokenizer_id = "drift-after-suite-construction"
    with pytest.raises(ValueError, match="tokenizer identities differ"):
        suite.run_native_chronological_full_history(
            public_input=_public_input("public history"),
            seed=3,
        )
    assert mutable_policy.calls == []


def test_result_rejects_non_covering_or_out_of_order_truncation_partition() -> None:
    public_input = _public_input(
        "oldest one two three four five six",
        "middle one two three four five",
        "newest one two",
    )
    reserve = 8
    base_tokens = _prompt_and_current_tokens(
        ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
        public_input,
    )
    newest_increment = sum(
        2 + len(content.split())
        for content in _exchange_message_contents(public_input.history[-1:])
    )
    result = _suite(
        budget=ProductBaselineTokenBudget(
            context_window_tokens=base_tokens + newest_increment + reserve,
            generation_reserve_tokens=reserve,
        )
    ).run_native_chronological_full_history(public_input=public_input, seed=4)

    assert result.truncation_receipt.dropped_oldest_block_artifact_ids == tuple(
        block.artifact_id for block in public_input.history[:2]
    )
    wrong_boundary = replace(
        result.truncation_receipt,
        dropped_oldest_block_artifact_ids=(hashlib.sha256(b"outside-selection").hexdigest(),),
    )
    with pytest.raises(ValueError, match="exactly partition"):
        replace(result, truncation_receipt=wrong_boundary)

    reversed_prefix = replace(
        result.truncation_receipt,
        dropped_oldest_block_artifact_ids=tuple(
            block.artifact_id for block in reversed(public_input.history[:2])
        ),
    )
    with pytest.raises(ValueError, match="chronological order"):
        replace(result, truncation_receipt=reversed_prefix)


def test_invalid_policy_completion_stays_invalid_without_a_second_parser() -> None:
    public_input = _public_input("first", "second")
    result = _suite(
        policy=_FakeContextualPolicy(chosen_action_id=None),
    ).run_native_chronological_full_history(public_input=public_input, seed=7)

    assert result.action_completion.raw_output == "not valid action JSON"
    assert result.action_completion.chosen_action_id is None
    assert not result.action_completion.valid


def test_semantic_rag_requires_embedder_and_nonempty_strict_selection() -> None:
    public_input = _public_input("only block")
    with pytest.raises(ValueError, match="semantic_embedder"):
        _suite().run_selective_semantic_rag(public_input=public_input, seed=1, top_k=1)

    embedder = _RecordingEmbedder(
        {
            public_input.current_observation.content: (1.0,),
            public_input.history[0].semantic_text: (1.0,),
        }
    )
    suite = _suite(embedder=embedder)
    selected_all = suite.run_selective_semantic_rag(
        public_input=public_input,
        seed=1,
        top_k=4,
    )
    assert selected_all.retrieval_receipt.requested_top_k == 4
    assert selected_all.retrieval_receipt.effective_top_k == 1
    with pytest.raises(ValueError, match="positive"):
        suite.run_selective_semantic_rag(public_input=public_input, seed=1, top_k=0)
