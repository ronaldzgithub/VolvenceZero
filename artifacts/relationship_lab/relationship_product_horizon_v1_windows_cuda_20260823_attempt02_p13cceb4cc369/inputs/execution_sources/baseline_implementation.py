"""Strong public-context baselines for the Relationship Lab product pilot.

The two arms in this module share one externally composed contextual action
policy and one exact chat-token counter.  They receive only frozen public
history blocks plus the current public observation.  No owner snapshot,
evaluator record, latent truth, future outcome, or reward object is accepted
by the API.

This module only prepares bounded public context and records what the frozen
policy returned.  It does not own memory, retrieval state, relationship
semantics, evaluation, PE/credit, or steering.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from lifeform_domain_emogpt.lab import RelationshipAction, canonical_json, sha256_json
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_packet1 import ContextualRelationshipActionPolicy


PRODUCT_BASELINE_INPUT_SCHEMA_VERSION = "relationship-product-baseline-input.v2"
PRODUCT_BASELINE_RESULT_SCHEMA_VERSION = "relationship-product-baseline-result.v2"
PRODUCT_BASELINE_CONTEXT_RECEIPT_SCHEMA_VERSION = "relationship-product-baseline-context-receipt.v2"
PRODUCT_BASELINE_RETRIEVAL_RECEIPT_SCHEMA_VERSION = "relationship-product-baseline-retrieval-receipt.v2"
PRODUCT_BASELINE_TRUNCATION_RECEIPT_SCHEMA_VERSION = "relationship-product-baseline-truncation-receipt.v2"
PRODUCT_BASELINE_ACTION_COMPLETION_SCHEMA_VERSION = "relationship-product-baseline-action-completion.v1"

_SHA256_LENGTH = 64


class ProductBaselineArm(str, Enum):
    """The two strong, non-learning product-pilot comparison arms."""

    NATIVE_CHRONOLOGICAL_FULL_HISTORY = "native_chronological_full_history"
    SELECTIVE_SEMANTIC_RAG = "selective_semantic_rag"


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def product_baseline_prompt_path(arm: ProductBaselineArm) -> pathlib.Path:
    """Return the frozen steelman prompt reused by one product baseline arm."""

    names = {
        ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY: "relationship_lab_full_history_steelman_v2.txt",
        ProductBaselineArm.SELECTIVE_SEMANTIC_RAG: "relationship_lab_rag_steelman_v2.txt",
    }
    return _asset_dir() / "prompts" / names[arm]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _require_non_empty_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


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


class _ContentAddressedPayload:
    """Mixin for immutable records whose ids cover their complete core payload."""

    def _core_payload(self) -> dict[str, object]:
        raise NotImplementedError

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._core_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self._core_payload(), "artifact_id": self.artifact_id}


@dataclass(frozen=True)
class ProductPublicHistoryBlock(_ContentAddressedPayload):
    """One indivisible public exchange with its paired assistant outcome."""

    ordinal: int
    exchange_id: str
    user_messages: tuple[str, ...]
    assistant_outcome: str
    schema_version: str = "relationship-product-public-history-exchange.v1"

    def __post_init__(self) -> None:
        _require_non_negative_int(self.ordinal, "ordinal")
        _require_non_empty_text(self.exchange_id, "exchange_id")
        if not isinstance(self.user_messages, tuple) or not self.user_messages:
            raise ValueError("user_messages must be a non-empty tuple")
        for index, message in enumerate(self.user_messages):
            _require_non_empty_text(message, f"user_messages[{index}]")
        _require_non_empty_text(self.assistant_outcome, "assistant_outcome")
        if self.schema_version != "relationship-product-public-history-exchange.v1":
            raise ValueError("public history exchange schema_version mismatch")

    @property
    def semantic_text(self) -> str:
        """Injective public-only representation passed to semantic retrieval."""

        return canonical_json(
            {
                "assistant_outcome": self.assistant_outcome,
                "user_messages": list(self.user_messages),
            }
        )

    @property
    def semantic_text_sha256(self) -> str:
        return _sha256_text(self.semantic_text)

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ordinal": self.ordinal,
            "exchange_id": self.exchange_id,
            "user_messages": list(self.user_messages),
            "assistant_outcome": self.assistant_outcome,
            "semantic_text_sha256": self.semantic_text_sha256,
        }


@dataclass(frozen=True)
class ProductCurrentObservation(_ContentAddressedPayload):
    """The public current user observation; no evaluator sidecar is representable."""

    content: str
    schema_version: str = "relationship-product-current-observation.v1"

    def __post_init__(self) -> None:
        _require_non_empty_text(self.content, "content")
        if self.schema_version != "relationship-product-current-observation.v1":
            raise ValueError("current observation schema_version mismatch")

    @property
    def content_sha256(self) -> str:
        return _sha256_text(self.content)

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "content": self.content,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class ProductBaselineInput(_ContentAddressedPayload):
    """Frozen SUT-visible input shared by both baseline arms."""

    history: tuple[ProductPublicHistoryBlock, ...]
    current_observation: ProductCurrentObservation
    schema_version: str = PRODUCT_BASELINE_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.history, tuple):
            raise ValueError("history must be a tuple of frozen public blocks")
        if not all(isinstance(block, ProductPublicHistoryBlock) for block in self.history):
            raise ValueError("history must contain only ProductPublicHistoryBlock values")
        ordinals = tuple(block.ordinal for block in self.history)
        if ordinals != tuple(range(len(self.history))):
            raise ValueError("history exchange ordinals must be contiguous and chronological")
        exchange_ids = tuple(block.exchange_id for block in self.history)
        if len(set(exchange_ids)) != len(exchange_ids):
            raise ValueError("history exchange_id values must be unique")
        if not isinstance(self.current_observation, ProductCurrentObservation):
            raise ValueError("current_observation must be ProductCurrentObservation")
        if self.schema_version != PRODUCT_BASELINE_INPUT_SCHEMA_VERSION:
            raise ValueError("product baseline input schema_version mismatch")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "history": [block.to_payload() for block in self.history],
            "current_observation": self.current_observation.to_payload(),
        }


@dataclass(frozen=True)
class ProductBaselineTokenBudget(_ContentAddressedPayload):
    """Exact model-window budget, including the reserved completion headroom."""

    context_window_tokens: int
    generation_reserve_tokens: int
    schema_version: str = "relationship-product-baseline-token-budget.v1"

    def __post_init__(self) -> None:
        _require_positive_int(self.context_window_tokens, "context_window_tokens")
        _require_positive_int(self.generation_reserve_tokens, "generation_reserve_tokens")
        if self.generation_reserve_tokens >= self.context_window_tokens:
            raise ValueError("generation_reserve_tokens must be smaller than context_window_tokens")
        if self.schema_version != "relationship-product-baseline-token-budget.v1":
            raise ValueError("product baseline token budget schema_version mismatch")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "context_window_tokens": self.context_window_tokens,
            "generation_reserve_tokens": self.generation_reserve_tokens,
        }


@dataclass(frozen=True)
class FrozenProductChatMessage(_ContentAddressedPayload):
    """Immutable message form used by the exact tokenizer counter."""

    role: str
    content: str
    schema_version: str = "relationship-product-baseline-chat-message.v1"

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError("chat message role is unsupported")
        _require_non_empty_text(self.content, "content")
        if self.schema_version != "relationship-product-baseline-chat-message.v1":
            raise ValueError("product chat message schema_version mismatch")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "content": self.content,
        }


class ExactProductMessageTokenCounter(Protocol):
    """Composition-root adapter over the exact tokenizer/chat template."""

    tokenizer_id: str

    def count_message_tokens(self, *, messages: tuple[FrozenProductChatMessage, ...]) -> int:
        """Count the exact input ids with generation prompt and special tokens."""


class ProductHistorySemanticEmbedder(Protocol):
    """Read-only semantic embedder; only public strings are passed to it."""

    name: str

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class ProductBaselineRetrievalCandidate(_ContentAddressedPayload):
    """One deterministically ranked public RAG candidate."""

    retrieval_rank: int
    block_artifact_id: str
    block_ordinal: int
    cosine_score_hex: str
    schema_version: str = "relationship-product-baseline-retrieval-candidate.v1"

    def __post_init__(self) -> None:
        _require_positive_int(self.retrieval_rank, "retrieval_rank")
        _require_sha256(self.block_artifact_id, "block_artifact_id")
        _require_non_negative_int(self.block_ordinal, "block_ordinal")
        _require_non_empty_text(self.cosine_score_hex, "cosine_score_hex")
        try:
            score = float.fromhex(self.cosine_score_hex)
        except ValueError as exc:
            raise ValueError("cosine_score_hex must be a finite hexadecimal float") from exc
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError("cosine_score_hex must encode a finite cosine in [-1, 1]")
        if self.schema_version != "relationship-product-baseline-retrieval-candidate.v1":
            raise ValueError("retrieval candidate schema_version mismatch")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "retrieval_rank": self.retrieval_rank,
            "block_artifact_id": self.block_artifact_id,
            "block_ordinal": self.block_ordinal,
            "cosine_score_hex": self.cosine_score_hex,
        }


@dataclass(frozen=True)
class ProductBaselineRetrievalReceipt(_ContentAddressedPayload):
    """Auditable selection receipt before any context-window truncation."""

    arm: ProductBaselineArm
    input_artifact_id: str
    strategy: str
    candidate_count: int
    requested_top_k: int | None
    effective_top_k: int
    embedder_id: str | None
    query_content_sha256: str | None
    ranked_candidates: tuple[ProductBaselineRetrievalCandidate, ...]
    selected_block_artifact_ids: tuple[str, ...]
    selected_chronological_block_artifact_ids: tuple[str, ...]
    schema_version: str = PRODUCT_BASELINE_RETRIEVAL_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.input_artifact_id, "input_artifact_id")
        _require_non_negative_int(self.candidate_count, "candidate_count")
        _require_non_negative_int(self.effective_top_k, "effective_top_k")
        if not isinstance(self.ranked_candidates, tuple):
            raise ValueError("ranked_candidates must be a tuple")
        if not all(
            isinstance(candidate, ProductBaselineRetrievalCandidate)
            for candidate in self.ranked_candidates
        ):
            raise ValueError("ranked_candidates must contain only frozen retrieval candidates")
        if not isinstance(self.selected_block_artifact_ids, tuple):
            raise ValueError("selected_block_artifact_ids must be a tuple")
        if not isinstance(self.selected_chronological_block_artifact_ids, tuple):
            raise ValueError("selected_chronological_block_artifact_ids must be a tuple")
        for digest in (*self.selected_block_artifact_ids, *self.selected_chronological_block_artifact_ids):
            _require_sha256(digest, "selected block artifact id")
        if set(self.selected_block_artifact_ids) != set(self.selected_chronological_block_artifact_ids):
            raise ValueError("retrieval-ranked and chronological selections must contain the same blocks")
        if self.arm is ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY:
            if (
                self.strategy != "all_public_exchanges_chronological"
                or self.requested_top_k is not None
            ):
                raise ValueError("native full-history retrieval receipt is inconsistent")
            if self.embedder_id is not None or self.query_content_sha256 is not None or self.ranked_candidates:
                raise ValueError("native full-history receipt must not claim semantic retrieval")
            if len(self.selected_block_artifact_ids) != self.candidate_count:
                raise ValueError("native full-history must select every public exchange")
            if self.effective_top_k != self.candidate_count:
                raise ValueError("native full-history effective_top_k must equal candidate_count")
        elif self.arm is ProductBaselineArm.SELECTIVE_SEMANTIC_RAG:
            if self.strategy != "cosine_desc_ordinal_asc_sha256_asc_v1":
                raise ValueError("semantic RAG strategy mismatch")
            _require_non_empty_text(self.embedder_id, "embedder_id")
            _require_sha256(self.query_content_sha256, "query_content_sha256")
            _require_positive_int(self.requested_top_k, "requested_top_k")
            assert self.requested_top_k is not None
            _require_positive_int(self.candidate_count, "candidate_count")
            expected_effective_top_k = min(self.requested_top_k, self.candidate_count)
            if self.effective_top_k != expected_effective_top_k:
                raise ValueError(
                    "semantic RAG effective_top_k must equal min(requested_top_k, candidate_count)"
                )
            if len(self.ranked_candidates) != self.candidate_count:
                raise ValueError("semantic RAG must receipt every ranked candidate")
            if len(self.selected_block_artifact_ids) != self.effective_top_k:
                raise ValueError("semantic RAG selection size must equal effective_top_k")
            ranks = tuple(candidate.retrieval_rank for candidate in self.ranked_candidates)
            if ranks != tuple(range(1, self.candidate_count + 1)):
                raise ValueError("semantic RAG candidate ranks must be contiguous")
            expected_selected = tuple(
                candidate.block_artifact_id
                for candidate in self.ranked_candidates[: self.effective_top_k]
            )
            if self.selected_block_artifact_ids != expected_selected:
                raise ValueError("semantic RAG selection must be the ranked effective_top_k")
        else:  # pragma: no cover - Enum construction closes this branch
            raise ValueError("unsupported product baseline arm")
        if self.schema_version != PRODUCT_BASELINE_RETRIEVAL_RECEIPT_SCHEMA_VERSION:
            raise ValueError("product baseline retrieval receipt schema_version mismatch")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arm": self.arm.value,
            "input_artifact_id": self.input_artifact_id,
            "strategy": self.strategy,
            "candidate_count": self.candidate_count,
            "requested_top_k": self.requested_top_k,
            "effective_top_k": self.effective_top_k,
            "embedder_id": self.embedder_id,
            "query_content_sha256": self.query_content_sha256,
            "ranked_candidates": [candidate.to_payload() for candidate in self.ranked_candidates],
            "selected_block_artifact_ids": list(self.selected_block_artifact_ids),
            "selected_chronological_block_artifact_ids": list(
                self.selected_chronological_block_artifact_ids
            ),
        }


@dataclass(frozen=True)
class ProductBaselineTruncationReceipt(_ContentAddressedPayload):
    """Explicit whole-block truncation receipt for one model call."""

    input_artifact_id: str
    initial_prompt_tokens: int
    final_prompt_tokens: int
    dropped_oldest_block_artifact_ids: tuple[str, ...]
    included_block_artifact_ids: tuple[str, ...]
    was_truncated: bool
    granularity: str = "complete_public_exchange_unit"
    reason: str = "none"
    schema_version: str = PRODUCT_BASELINE_TRUNCATION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.input_artifact_id, "input_artifact_id")
        _require_non_negative_int(self.initial_prompt_tokens, "initial_prompt_tokens")
        _require_non_negative_int(self.final_prompt_tokens, "final_prompt_tokens")
        if not isinstance(self.dropped_oldest_block_artifact_ids, tuple):
            raise ValueError("dropped_oldest_block_artifact_ids must be a tuple")
        if not isinstance(self.included_block_artifact_ids, tuple):
            raise ValueError("included_block_artifact_ids must be a tuple")
        for digest in (*self.dropped_oldest_block_artifact_ids, *self.included_block_artifact_ids):
            _require_sha256(digest, "truncation block artifact id")
        if set(self.dropped_oldest_block_artifact_ids) & set(self.included_block_artifact_ids):
            raise ValueError("a history block cannot be both dropped and included")
        if self.granularity != "complete_public_exchange_unit":
            raise ValueError("truncation granularity must remain whole-exchange")
        expected_reason = (
            "oldest_complete_exchange_units_until_budget_fit"
            if self.was_truncated
            else "none"
        )
        if self.reason != expected_reason:
            raise ValueError("truncation reason does not match was_truncated")
        if self.was_truncated != bool(self.dropped_oldest_block_artifact_ids):
            raise ValueError("truncation flag must exactly reflect dropped blocks")
        if self.final_prompt_tokens > self.initial_prompt_tokens:
            raise ValueError("truncation cannot increase the prompt token count")
        if self.schema_version != PRODUCT_BASELINE_TRUNCATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("product baseline truncation receipt schema_version mismatch")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_artifact_id": self.input_artifact_id,
            "initial_prompt_tokens": self.initial_prompt_tokens,
            "final_prompt_tokens": self.final_prompt_tokens,
            "dropped_oldest_block_artifact_ids": list(self.dropped_oldest_block_artifact_ids),
            "included_block_artifact_ids": list(self.included_block_artifact_ids),
            "was_truncated": self.was_truncated,
            "granularity": self.granularity,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ProductBaselineContextReceipt(_ContentAddressedPayload):
    """Exact final context and token-budget receipt passed to the policy."""

    arm: ProductBaselineArm
    input_artifact_id: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    arm_prompt_sha256: str
    tokenizer_id: str
    token_budget_artifact_id: str
    prompt_and_current_tokens: int
    final_prompt_tokens: int
    history_increment_tokens: int
    generation_reserve_tokens: int
    total_reserved_tokens: int
    context_window_tokens: int
    included_block_artifact_ids: tuple[str, ...]
    rendered_messages_sha256: str
    schema_version: str = PRODUCT_BASELINE_CONTEXT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.input_artifact_id, "input_artifact_id")
        _require_non_empty_text(self.model_id, "model_id")
        _require_sha256(self.weights_sha256, "weights_sha256")
        _require_sha256(self.generation_config_sha256, "generation_config_sha256")
        _require_sha256(self.arm_prompt_sha256, "arm_prompt_sha256")
        _require_non_empty_text(self.tokenizer_id, "tokenizer_id")
        _require_sha256(self.token_budget_artifact_id, "token_budget_artifact_id")
        for field_name, value in (
            ("prompt_and_current_tokens", self.prompt_and_current_tokens),
            ("final_prompt_tokens", self.final_prompt_tokens),
            ("history_increment_tokens", self.history_increment_tokens),
            ("generation_reserve_tokens", self.generation_reserve_tokens),
            ("total_reserved_tokens", self.total_reserved_tokens),
            ("context_window_tokens", self.context_window_tokens),
        ):
            _require_non_negative_int(value, field_name)
        if self.final_prompt_tokens != self.prompt_and_current_tokens + self.history_increment_tokens:
            raise ValueError("history_increment_tokens must reconcile the exact final prompt count")
        if self.total_reserved_tokens != self.final_prompt_tokens + self.generation_reserve_tokens:
            raise ValueError("total_reserved_tokens must include prompt and generation reserve")
        if self.total_reserved_tokens > self.context_window_tokens:
            raise ValueError("reserved tokens exceed the model context window")
        if not isinstance(self.included_block_artifact_ids, tuple):
            raise ValueError("included_block_artifact_ids must be a tuple")
        for digest in self.included_block_artifact_ids:
            _require_sha256(digest, "included block artifact id")
        _require_sha256(self.rendered_messages_sha256, "rendered_messages_sha256")
        if self.schema_version != PRODUCT_BASELINE_CONTEXT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("product baseline context receipt schema_version mismatch")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arm": self.arm.value,
            "input_artifact_id": self.input_artifact_id,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "arm_prompt_sha256": self.arm_prompt_sha256,
            "tokenizer_id": self.tokenizer_id,
            "token_budget_artifact_id": self.token_budget_artifact_id,
            "prompt_and_current_tokens": self.prompt_and_current_tokens,
            "final_prompt_tokens": self.final_prompt_tokens,
            "history_increment_tokens": self.history_increment_tokens,
            "generation_reserve_tokens": self.generation_reserve_tokens,
            "total_reserved_tokens": self.total_reserved_tokens,
            "context_window_tokens": self.context_window_tokens,
            "included_block_artifact_ids": list(self.included_block_artifact_ids),
            "rendered_messages_sha256": self.rendered_messages_sha256,
        }


@dataclass(frozen=True)
class ProductBaselineActionCompletion(_ContentAddressedPayload):
    """Content-addressed copy of the existing policy's strict completion."""

    raw_output: str
    chosen_action_id: RelationshipAction | None
    prompt_tokens: int
    completion_tokens: int
    schema_version: str = PRODUCT_BASELINE_ACTION_COMPLETION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.raw_output, str):
            raise ValueError("raw_output must be a string")
        if self.chosen_action_id is not None and not isinstance(self.chosen_action_id, RelationshipAction):
            raise ValueError("chosen_action_id must be a RelationshipAction or None")
        _require_non_negative_int(self.prompt_tokens, "prompt_tokens")
        _require_non_negative_int(self.completion_tokens, "completion_tokens")
        if self.schema_version != PRODUCT_BASELINE_ACTION_COMPLETION_SCHEMA_VERSION:
            raise ValueError("product baseline action completion schema_version mismatch")

    @property
    def valid(self) -> bool:
        return self.chosen_action_id is not None

    @classmethod
    def from_policy_completion(cls, completion: StatelessActionCompletion) -> ProductBaselineActionCompletion:
        if not isinstance(completion, StatelessActionCompletion):
            raise TypeError("policy must return StatelessActionCompletion")
        return cls(
            raw_output=completion.raw_output,
            chosen_action_id=completion.chosen_action_id,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raw_output": self.raw_output,
            "chosen_action_id": self.chosen_action_id.value if self.chosen_action_id is not None else None,
            "valid": self.valid,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True)
class ProductBaselineResult(_ContentAddressedPayload):
    """One immutable action plus context/retrieval/truncation evidence bundle."""

    arm: ProductBaselineArm
    seed: int
    input_artifact_id: str
    action_completion: ProductBaselineActionCompletion
    context_receipt: ProductBaselineContextReceipt
    retrieval_receipt: ProductBaselineRetrievalReceipt
    truncation_receipt: ProductBaselineTruncationReceipt
    schema_version: str = PRODUCT_BASELINE_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_negative_int(self.seed, "seed")
        _require_sha256(self.input_artifact_id, "input_artifact_id")
        if not isinstance(self.action_completion, ProductBaselineActionCompletion):
            raise ValueError("action_completion must be a frozen product baseline completion")
        if not isinstance(self.context_receipt, ProductBaselineContextReceipt):
            raise ValueError("context_receipt must be a frozen product baseline context receipt")
        if not isinstance(self.retrieval_receipt, ProductBaselineRetrievalReceipt):
            raise ValueError("retrieval_receipt must be a frozen product baseline retrieval receipt")
        if not isinstance(self.truncation_receipt, ProductBaselineTruncationReceipt):
            raise ValueError("truncation_receipt must be a frozen product baseline truncation receipt")
        receipts = (
            self.context_receipt,
            self.retrieval_receipt,
            self.truncation_receipt,
        )
        if any(receipt.input_artifact_id != self.input_artifact_id for receipt in receipts):
            raise ValueError("all product baseline receipts must bind the same input")
        if self.context_receipt.arm is not self.arm or self.retrieval_receipt.arm is not self.arm:
            raise ValueError("all product baseline receipts must bind the same arm")
        if self.action_completion.prompt_tokens != self.context_receipt.final_prompt_tokens:
            raise ValueError("action completion and context receipt prompt counts differ")
        if self.context_receipt.included_block_artifact_ids != self.truncation_receipt.included_block_artifact_ids:
            raise ValueError("context and truncation receipts disagree about included blocks")
        chronological_selection = (
            self.retrieval_receipt.selected_chronological_block_artifact_ids
        )
        truncation_partition = (
            *self.truncation_receipt.dropped_oldest_block_artifact_ids,
            *self.truncation_receipt.included_block_artifact_ids,
        )
        if truncation_partition != chronological_selection:
            raise ValueError(
                "truncation dropped/included blocks must exactly partition the "
                "retrieval selection in chronological order"
            )
        if len(set(truncation_partition)) != len(truncation_partition):
            raise ValueError("truncation partition must not contain duplicate blocks")
        if self.schema_version != PRODUCT_BASELINE_RESULT_SCHEMA_VERSION:
            raise ValueError("product baseline result schema_version mismatch")

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arm": self.arm.value,
            "seed": self.seed,
            "input_artifact_id": self.input_artifact_id,
            "action_completion": self.action_completion.to_payload(),
            "context_receipt": self.context_receipt.to_payload(),
            "retrieval_receipt": self.retrieval_receipt.to_payload(),
            "truncation_receipt": self.truncation_receipt.to_payload(),
        }


class ProductBaselineContextWindowError(ValueError):
    """Raised before generation when prompt/current/reserve cannot fit."""


def _count_message_tokens(
    counter: ExactProductMessageTokenCounter,
    messages: tuple[FrozenProductChatMessage, ...],
) -> int:
    count = counter.count_message_tokens(messages=messages)
    _require_non_negative_int(count, "exact message token count")
    return count


def _messages_for_blocks(
    *,
    prompt: str,
    blocks: tuple[ProductPublicHistoryBlock, ...],
    current_observation: ProductCurrentObservation,
) -> tuple[FrozenProductChatMessage, ...]:
    messages: list[FrozenProductChatMessage] = [
        FrozenProductChatMessage(role="system", content=prompt)
    ]
    for block in blocks:
        messages.extend(
            FrozenProductChatMessage(role="user", content=content)
            for content in block.user_messages
        )
        messages.append(
            FrozenProductChatMessage(
                role="assistant",
                content=block.assistant_outcome,
            )
        )
    messages.append(
        FrozenProductChatMessage(role="user", content=current_observation.content)
    )
    return tuple(messages)


def _policy_messages(
    messages: tuple[FrozenProductChatMessage, ...],
) -> tuple[dict[str, str], ...]:
    return tuple({"role": message.role, "content": message.content} for message in messages)


def _rendered_messages_sha256(messages: tuple[FrozenProductChatMessage, ...]) -> str:
    return sha256_json(tuple(message._core_payload() for message in messages))


def _validated_embedding(embedder: ProductHistorySemanticEmbedder, text: str) -> tuple[float, ...]:
    vector = embedder.embed(text)
    if not isinstance(vector, tuple) or not vector:
        raise ValueError("semantic embedder must return a non-empty tuple")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        raise ValueError("semantic embedding values must be numbers")
    normalized = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("semantic embedding values must be finite")
    return normalized


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError(f"semantic embedding width mismatch: {len(left)} vs {len(right)}")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    score = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    ) / (
        left_norm * right_norm
    )
    if not math.isfinite(score):
        raise ValueError("semantic cosine must be finite")
    return max(-1.0, min(1.0, score))


@dataclass(frozen=True)
class RelationshipProductBaselineSuite:
    """Externally composed two-arm suite sharing one policy and token budget."""

    policy: ContextualRelationshipActionPolicy
    token_counter: ExactProductMessageTokenCounter
    token_budget: ProductBaselineTokenBudget
    semantic_embedder: ProductHistorySemanticEmbedder | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.token_budget, ProductBaselineTokenBudget):
            raise TypeError("token_budget must be ProductBaselineTokenBudget")
        for field_name in ("model_id", "weights_sha256", "generation_config_sha256"):
            value = getattr(self.policy, field_name)
            if field_name.endswith("sha256"):
                _require_sha256(value, f"policy.{field_name}")
            else:
                _require_non_empty_text(value, f"policy.{field_name}")
        self._validate_generation_contract()
        if self.semantic_embedder is not None:
            _require_non_empty_text(self.semantic_embedder.name, "semantic_embedder.name")

    def _validate_generation_contract(self) -> None:
        """Fail before generation if counting and generation can diverge."""

        policy_tokenizer_id = self.policy.tokenizer_id
        counter_tokenizer_id = self.token_counter.tokenizer_id
        _require_non_empty_text(policy_tokenizer_id, "policy.tokenizer_id")
        _require_non_empty_text(counter_tokenizer_id, "token_counter.tokenizer_id")
        if policy_tokenizer_id != counter_tokenizer_id:
            raise ValueError("policy and exact token counter tokenizer identities differ")
        maximum_generation_tokens = self.policy.max_new_tokens
        _require_positive_int(maximum_generation_tokens, "policy.max_new_tokens")
        if self.token_budget.generation_reserve_tokens < maximum_generation_tokens:
            raise ValueError("generation reserve must cover policy.max_new_tokens")

    def run_native_chronological_full_history(
        self,
        *,
        public_input: ProductBaselineInput,
        seed: int,
    ) -> ProductBaselineResult:
        """Run all public exchanges, dropping oldest whole exchanges only if required."""

        self._validate_call(public_input=public_input, seed=seed)
        selected = public_input.history
        selected_ids = tuple(block.artifact_id for block in selected)
        retrieval_receipt = ProductBaselineRetrievalReceipt(
            arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
            input_artifact_id=public_input.artifact_id,
            strategy="all_public_exchanges_chronological",
            candidate_count=len(public_input.history),
            requested_top_k=None,
            effective_top_k=len(public_input.history),
            embedder_id=None,
            query_content_sha256=None,
            ranked_candidates=(),
            selected_block_artifact_ids=selected_ids,
            selected_chronological_block_artifact_ids=selected_ids,
        )
        return self._run_selected_blocks(
            arm=ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY,
            public_input=public_input,
            selected_blocks=selected,
            retrieval_receipt=retrieval_receipt,
            seed=seed,
        )

    def run_selective_semantic_rag(
        self,
        *,
        public_input: ProductBaselineInput,
        seed: int,
        top_k: int,
    ) -> ProductBaselineResult:
        """Run deterministic cosine top-k retrieval over public block content only."""

        self._validate_call(public_input=public_input, seed=seed)
        if self.semantic_embedder is None:
            raise ValueError("selective semantic RAG requires an injected semantic_embedder")
        _require_positive_int(top_k, "top_k")
        if not public_input.history:
            raise ValueError("selective semantic RAG requires at least one public exchange")

        query = _validated_embedding(self.semantic_embedder, public_input.current_observation.content)
        scored: list[tuple[float, ProductPublicHistoryBlock]] = []
        for block in public_input.history:
            vector = _validated_embedding(self.semantic_embedder, block.semantic_text)
            scored.append((_cosine(query, vector), block))
        scored.sort(key=lambda item: (-item[0], item[1].ordinal, item[1].artifact_id))
        ranked_candidates = tuple(
            ProductBaselineRetrievalCandidate(
                retrieval_rank=index,
                block_artifact_id=block.artifact_id,
                block_ordinal=block.ordinal,
                cosine_score_hex=score.hex(),
            )
            for index, (score, block) in enumerate(scored, start=1)
        )
        effective_top_k = min(top_k, len(scored))
        ranked_selected = tuple(block for _score, block in scored[:effective_top_k])
        chronological_selected = tuple(sorted(ranked_selected, key=lambda block: block.ordinal))
        retrieval_receipt = ProductBaselineRetrievalReceipt(
            arm=ProductBaselineArm.SELECTIVE_SEMANTIC_RAG,
            input_artifact_id=public_input.artifact_id,
            strategy="cosine_desc_ordinal_asc_sha256_asc_v1",
            candidate_count=len(public_input.history),
            requested_top_k=top_k,
            effective_top_k=effective_top_k,
            embedder_id=self.semantic_embedder.name,
            query_content_sha256=public_input.current_observation.content_sha256,
            ranked_candidates=ranked_candidates,
            selected_block_artifact_ids=tuple(block.artifact_id for block in ranked_selected),
            selected_chronological_block_artifact_ids=tuple(
                block.artifact_id for block in chronological_selected
            ),
        )
        return self._run_selected_blocks(
            arm=ProductBaselineArm.SELECTIVE_SEMANTIC_RAG,
            public_input=public_input,
            selected_blocks=chronological_selected,
            retrieval_receipt=retrieval_receipt,
            seed=seed,
        )

    def _validate_call(self, *, public_input: ProductBaselineInput, seed: int) -> None:
        self._validate_generation_contract()
        if not isinstance(public_input, ProductBaselineInput):
            raise TypeError("public_input must be ProductBaselineInput")
        _require_non_negative_int(seed, "seed")

    def _run_selected_blocks(
        self,
        *,
        arm: ProductBaselineArm,
        public_input: ProductBaselineInput,
        selected_blocks: tuple[ProductPublicHistoryBlock, ...],
        retrieval_receipt: ProductBaselineRetrievalReceipt,
        seed: int,
    ) -> ProductBaselineResult:
        prompt_path = product_baseline_prompt_path(arm)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"product baseline prompt is missing: {prompt_path}")
        prompt_bytes = prompt_path.read_bytes()
        prompt = prompt_bytes.decode("utf-8").strip()
        _require_non_empty_text(prompt, "product baseline prompt")

        base_messages = _messages_for_blocks(
            prompt=prompt,
            blocks=(),
            current_observation=public_input.current_observation,
        )
        prompt_and_current_tokens = _count_message_tokens(self.token_counter, base_messages)
        if (
            prompt_and_current_tokens + self.token_budget.generation_reserve_tokens
            > self.token_budget.context_window_tokens
        ):
            raise ProductBaselineContextWindowError(
                "system prompt + current observation + generation reserve exceed the context window"
            )

        kept_blocks = list(selected_blocks)
        dropped_blocks: list[ProductPublicHistoryBlock] = []
        messages = _messages_for_blocks(
            prompt=prompt,
            blocks=tuple(kept_blocks),
            current_observation=public_input.current_observation,
        )
        initial_prompt_tokens = _count_message_tokens(self.token_counter, messages)
        final_prompt_tokens = initial_prompt_tokens
        while (
            final_prompt_tokens + self.token_budget.generation_reserve_tokens
            > self.token_budget.context_window_tokens
        ):
            if not kept_blocks:  # pragma: no cover - base fit is checked above
                raise ProductBaselineContextWindowError("unable to fit public context into the context window")
            dropped_blocks.append(kept_blocks.pop(0))
            messages = _messages_for_blocks(
                prompt=prompt,
                blocks=tuple(kept_blocks),
                current_observation=public_input.current_observation,
            )
            final_prompt_tokens = _count_message_tokens(self.token_counter, messages)

        completion = self.policy.choose_from_messages(messages=_policy_messages(messages), seed=seed)
        action_completion = ProductBaselineActionCompletion.from_policy_completion(completion)
        if action_completion.prompt_tokens != final_prompt_tokens:
            raise ValueError(
                "exact token counter disagrees with the contextual policy prompt_tokens receipt"
            )
        if action_completion.completion_tokens > self.token_budget.generation_reserve_tokens:
            raise ValueError("policy completion exceeded the frozen generation token reserve")

        included_ids = tuple(block.artifact_id for block in kept_blocks)
        truncation_receipt = ProductBaselineTruncationReceipt(
            input_artifact_id=public_input.artifact_id,
            initial_prompt_tokens=initial_prompt_tokens,
            final_prompt_tokens=final_prompt_tokens,
            dropped_oldest_block_artifact_ids=tuple(block.artifact_id for block in dropped_blocks),
            included_block_artifact_ids=included_ids,
            was_truncated=bool(dropped_blocks),
            reason=(
                "oldest_complete_exchange_units_until_budget_fit"
                if dropped_blocks
                else "none"
            ),
        )
        context_receipt = ProductBaselineContextReceipt(
            arm=arm,
            input_artifact_id=public_input.artifact_id,
            model_id=self.policy.model_id,
            weights_sha256=self.policy.weights_sha256,
            generation_config_sha256=self.policy.generation_config_sha256,
            arm_prompt_sha256=_sha256_bytes(prompt_bytes),
            tokenizer_id=self.token_counter.tokenizer_id,
            token_budget_artifact_id=self.token_budget.artifact_id,
            prompt_and_current_tokens=prompt_and_current_tokens,
            final_prompt_tokens=final_prompt_tokens,
            history_increment_tokens=final_prompt_tokens - prompt_and_current_tokens,
            generation_reserve_tokens=self.token_budget.generation_reserve_tokens,
            total_reserved_tokens=final_prompt_tokens + self.token_budget.generation_reserve_tokens,
            context_window_tokens=self.token_budget.context_window_tokens,
            included_block_artifact_ids=included_ids,
            rendered_messages_sha256=_rendered_messages_sha256(messages),
        )
        return ProductBaselineResult(
            arm=arm,
            seed=seed,
            input_artifact_id=public_input.artifact_id,
            action_completion=action_completion,
            context_receipt=context_receipt,
            retrieval_receipt=retrieval_receipt,
            truncation_receipt=truncation_receipt,
        )


__all__ = [
    "ExactProductMessageTokenCounter",
    "FrozenProductChatMessage",
    "ProductBaselineActionCompletion",
    "ProductBaselineArm",
    "ProductBaselineContextReceipt",
    "ProductBaselineContextWindowError",
    "ProductBaselineInput",
    "ProductBaselineResult",
    "ProductBaselineRetrievalCandidate",
    "ProductBaselineRetrievalReceipt",
    "ProductBaselineTokenBudget",
    "ProductBaselineTruncationReceipt",
    "ProductCurrentObservation",
    "ProductHistorySemanticEmbedder",
    "ProductPublicHistoryBlock",
    "RelationshipProductBaselineSuite",
    "product_baseline_prompt_path",
]
