"""Resident JSONL dispatcher for Relationship Lab product baselines.

The dispatcher is a deliberately narrow process boundary.  A parent may send
only one content-addressed :class:`ProductBaselineInput`, a public baseline arm,
and generation controls.  Evaluator truth, owner snapshots, future outcomes,
PE/credit, and campaign state are not representable in the request schema.

One frozen Hugging Face policy, its exact tokenizer counter, and one verified
precomputed public-embedding table are constructed before the first request
and reused until EOF.  A request or execution failure emits one typed fatal
response, logs the exception, flushes both streams, and terminates the loop
with a non-zero status instead of continuing with ambiguous model state.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TextIO

from lifeform_domain_emogpt.lab import RelationshipAction, canonical_json, sha256_json
from lifeform_evolution.relationship_lab_baseline import (
    DEFAULT_STATELESS_MODEL_ID,
    DEFAULT_STATELESS_MODEL_SOURCE,
    HFStatelessRelationshipActionPolicy,
)
from lifeform_evolution.relationship_lab_product_baselines import (
    ProductBaselineArm,
    ProductBaselineInput,
    ProductBaselineResult,
    ProductBaselineTokenBudget,
    ProductCurrentObservation,
    ProductPublicHistoryBlock,
    RelationshipProductBaselineSuite,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    PrecomputedPublicEmbeddingRecord,
    PrecomputedPublicEmbeddingTable,
    PrecomputedPublicSemanticEmbedder,
    RevisionPinnedBgeM3PublicSemanticEmbedder,
    bge_m3_weight_pinned_embedder_identity,
    load_precomputed_public_embedding_table,
    write_precomputed_public_embedding_table,
)


PRODUCT_BASELINE_DISPATCHER_REQUEST_SCHEMA_VERSION = (
    "relationship-product-baseline-dispatcher-request.v2"
)
PRODUCT_BASELINE_DISPATCHER_RESPONSE_SCHEMA_VERSION = (
    "relationship-product-baseline-dispatcher-response.v2"
)
PRODUCT_BASELINE_DISPATCHER_FATAL_SCHEMA_VERSION = (
    "relationship-product-baseline-dispatcher-fatal.v1"
)

_REQUEST_KEYS = frozenset(
    {
        "nonce",
        "arm",
        "public_plan_artifact_id",
        "subject_scope",
        "decision_boundary",
        "ordered_source_session_ids",
        "ordered_source_block_artifact_ids",
        "public_ledger_artifact_id",
        "public_input",
        "history_block_lineage",
        "current_observation_lineage",
        "seed",
        "top_k",
    }
)
_HISTORY_BLOCK_KEYS = frozenset(
    {
        "schema_version",
        "ordinal",
        "exchange_id",
        "user_messages",
        "assistant_outcome",
        "semantic_text_sha256",
        "artifact_id",
    }
)
_CURRENT_OBSERVATION_KEYS = frozenset(
    {"schema_version", "content", "content_sha256", "artifact_id"}
)
_BASELINE_INPUT_KEYS = frozenset(
    {"schema_version", "history", "current_observation", "artifact_id"}
)
_SHA256_LENGTH = 64

DEFAULT_PRODUCT_BASELINE_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_PRODUCT_BASELINE_BGE_SOURCE = BGE_M3_MODEL_ID
DEFAULT_PRODUCT_BASELINE_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DEFAULT_PRODUCT_BASELINE_BGE_WEIGHTS_SHA256 = BGE_M3_WEIGHT_BYTES_SHA256
DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION = (
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION
)


class ProductBaselineSemanticMode(str, Enum):
    """The two honest public-only semantic embedding paths."""

    PRECOMPUTED = "precomputed"
    LIVE_BGE_M3_CACHED = "live_bge_m3_cached"


def _require_exact_keys(payload: object, expected: frozenset[str], field_name: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be an object")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{field_name} keys mismatch: missing={missing}, extra={extra}")
    return payload


def _require_non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_lower_hex_revision(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase 40-hex revision")
    return value


def _parse_public_history_block(payload: object, *, index: int) -> ProductPublicHistoryBlock:
    raw = _require_exact_keys(payload, _HISTORY_BLOCK_KEYS, f"public_input.history[{index}]")
    raw_user_messages = raw["user_messages"]
    if not isinstance(raw_user_messages, list):
        raise ValueError(f"public_input.history[{index}].user_messages must be a list")
    block = ProductPublicHistoryBlock(
        ordinal=raw["ordinal"],
        exchange_id=raw["exchange_id"],
        user_messages=tuple(raw_user_messages),
        assistant_outcome=raw["assistant_outcome"],
        schema_version=raw["schema_version"],
    )
    semantic_text_sha256 = _require_sha256(
        raw["semantic_text_sha256"],
        f"public_input.history[{index}].semantic_text_sha256",
    )
    artifact_id = _require_sha256(
        raw["artifact_id"],
        f"public_input.history[{index}].artifact_id",
    )
    if semantic_text_sha256 != block.semantic_text_sha256:
        raise ValueError(f"public_input.history[{index}] semantic_text_sha256 mismatch")
    if artifact_id != block.artifact_id:
        raise ValueError(f"public_input.history[{index}] artifact_id mismatch")
    return block


def _parse_current_observation(payload: object) -> ProductCurrentObservation:
    raw = _require_exact_keys(
        payload,
        _CURRENT_OBSERVATION_KEYS,
        "public_input.current_observation",
    )
    observation = ProductCurrentObservation(
        content=raw["content"],
        schema_version=raw["schema_version"],
    )
    content_sha256 = _require_sha256(
        raw["content_sha256"],
        "public_input.current_observation.content_sha256",
    )
    artifact_id = _require_sha256(
        raw["artifact_id"],
        "public_input.current_observation.artifact_id",
    )
    if content_sha256 != observation.content_sha256:
        raise ValueError("public_input.current_observation content_sha256 mismatch")
    if artifact_id != observation.artifact_id:
        raise ValueError("public_input.current_observation artifact_id mismatch")
    return observation


def parse_product_baseline_input(payload: object) -> ProductBaselineInput:
    """Strictly reconstruct and verify every public input layer."""

    raw = _require_exact_keys(payload, _BASELINE_INPUT_KEYS, "public_input")
    raw_history = raw["history"]
    if not isinstance(raw_history, list):
        raise ValueError("public_input.history must be a list")
    public_input = ProductBaselineInput(
        history=tuple(
            _parse_public_history_block(block, index=index)
            for index, block in enumerate(raw_history)
        ),
        current_observation=_parse_current_observation(raw["current_observation"]),
        schema_version=raw["schema_version"],
    )
    artifact_id = _require_sha256(raw["artifact_id"], "public_input.artifact_id")
    if artifact_id != public_input.artifact_id:
        raise ValueError("public_input artifact_id mismatch")
    return public_input


@dataclass(frozen=True)
class ProductBaselineHistoryBlockLineage:
    """Campaign-issued public-ledger reference for one exact history block."""

    ordinal: int
    block_artifact_id: str
    public_ledger_entry_artifact_id: str

    def __post_init__(self) -> None:
        _require_non_negative_int(self.ordinal, "history lineage ordinal")
        _require_sha256(self.block_artifact_id, "history lineage block_artifact_id")
        _require_sha256(
            self.public_ledger_entry_artifact_id,
            "history lineage public_ledger_entry_artifact_id",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "block_artifact_id": self.block_artifact_id,
            "public_ledger_entry_artifact_id": self.public_ledger_entry_artifact_id,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ProductBaselineHistoryBlockLineage:
        raw = _require_exact_keys(
            payload,
            frozenset({"ordinal", "block_artifact_id", "public_ledger_entry_artifact_id"}),
            "history_block_lineage entry",
        )
        return cls(
            ordinal=_require_non_negative_int(raw["ordinal"], "history lineage ordinal"),
            block_artifact_id=_require_sha256(
                raw["block_artifact_id"],
                "history lineage block_artifact_id",
            ),
            public_ledger_entry_artifact_id=_require_sha256(
                raw["public_ledger_entry_artifact_id"],
                "history lineage public_ledger_entry_artifact_id",
            ),
        )


@dataclass(frozen=True)
class ProductBaselineCurrentObservationLineage:
    """Campaign-issued public-ledger reference for the current observation."""

    observation_artifact_id: str
    public_ledger_entry_artifact_id: str

    def __post_init__(self) -> None:
        _require_sha256(self.observation_artifact_id, "observation lineage artifact_id")
        _require_sha256(
            self.public_ledger_entry_artifact_id,
            "observation lineage public_ledger_entry_artifact_id",
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "observation_artifact_id": self.observation_artifact_id,
            "public_ledger_entry_artifact_id": self.public_ledger_entry_artifact_id,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ProductBaselineCurrentObservationLineage:
        raw = _require_exact_keys(
            payload,
            frozenset({"observation_artifact_id", "public_ledger_entry_artifact_id"}),
            "current_observation_lineage",
        )
        return cls(
            observation_artifact_id=_require_sha256(
                raw["observation_artifact_id"],
                "observation lineage artifact_id",
            ),
            public_ledger_entry_artifact_id=_require_sha256(
                raw["public_ledger_entry_artifact_id"],
                "observation lineage public_ledger_entry_artifact_id",
            ),
        )


@dataclass(frozen=True)
class ProductBaselineDecisionBoundary:
    """Public source coordinates for the current decision only."""

    current_session_id: str
    decision_id: str
    decision_index: int

    def __post_init__(self) -> None:
        _require_non_empty_text(self.current_session_id, "decision boundary current_session_id")
        _require_non_empty_text(self.decision_id, "decision boundary decision_id")
        _require_non_negative_int(self.decision_index, "decision boundary decision_index")

    def to_payload(self) -> dict[str, object]:
        return {
            "current_session_id": self.current_session_id,
            "decision_id": self.decision_id,
            "decision_index": self.decision_index,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ProductBaselineDecisionBoundary:
        raw = _require_exact_keys(
            payload,
            frozenset({"current_session_id", "decision_id", "decision_index"}),
            "decision_boundary",
        )
        return cls(
            current_session_id=_require_non_empty_text(
                raw["current_session_id"],
                "decision boundary current_session_id",
            ),
            decision_id=_require_non_empty_text(
                raw["decision_id"],
                "decision boundary decision_id",
            ),
            decision_index=_require_non_negative_int(
                raw["decision_index"],
                "decision boundary decision_index",
            ),
        )


@dataclass(frozen=True)
class ProductBaselineDispatcherRequest:
    """The complete and intentionally tiny SUT-visible JSONL request."""

    nonce: str
    arm: ProductBaselineArm
    public_plan_artifact_id: str
    subject_scope: str
    decision_boundary: ProductBaselineDecisionBoundary
    ordered_source_session_ids: tuple[str, ...]
    ordered_source_block_artifact_ids: tuple[str, ...]
    public_ledger_artifact_id: str
    public_input: ProductBaselineInput
    history_block_lineage: tuple[ProductBaselineHistoryBlockLineage, ...]
    current_observation_lineage: ProductBaselineCurrentObservationLineage
    seed: int
    top_k: int | None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.nonce, "nonce")
        if not isinstance(self.arm, ProductBaselineArm):
            raise ValueError("arm must be a ProductBaselineArm")
        _require_sha256(self.public_plan_artifact_id, "public_plan_artifact_id")
        _require_sha256(self.subject_scope, "subject_scope")
        if not isinstance(self.decision_boundary, ProductBaselineDecisionBoundary):
            raise ValueError("decision_boundary must be ProductBaselineDecisionBoundary")
        if not isinstance(self.ordered_source_session_ids, tuple) or not all(
            isinstance(value, str) and value.strip() for value in self.ordered_source_session_ids
        ):
            raise ValueError("ordered_source_session_ids must be a non-empty-text tuple")
        if len(set(self.ordered_source_session_ids)) != len(self.ordered_source_session_ids):
            raise ValueError("ordered_source_session_ids must not contain duplicates")
        if not self.ordered_source_session_ids:
            raise ValueError("ordered_source_session_ids must not be empty")
        if self.ordered_source_session_ids[-1] != self.decision_boundary.current_session_id:
            raise ValueError("ordered_source_session_ids must end at the current decision boundary")
        if not isinstance(self.ordered_source_block_artifact_ids, tuple):
            raise ValueError("ordered_source_block_artifact_ids must be a tuple")
        for digest in self.ordered_source_block_artifact_ids:
            _require_sha256(digest, "ordered source block artifact id")
        if not isinstance(self.public_input, ProductBaselineInput):
            raise ValueError("public_input must be ProductBaselineInput")
        expected_session_ids = (
            *(block.exchange_id for block in self.public_input.history),
            self.decision_boundary.current_session_id,
        )
        if self.ordered_source_session_ids != expected_session_ids:
            raise ValueError(
                "ordered_source_session_ids must bind one complete history exchange per "
                "prior session and end at the current decision"
            )
        expected_block_ids = tuple(block.artifact_id for block in self.public_input.history)
        if self.ordered_source_block_artifact_ids != expected_block_ids:
            raise ValueError(
                "ordered_source_block_artifact_ids must exactly match public_input.history"
            )
        _require_sha256(self.public_ledger_artifact_id, "public_ledger_artifact_id")
        if not isinstance(self.history_block_lineage, tuple) or not all(
            isinstance(item, ProductBaselineHistoryBlockLineage)
            for item in self.history_block_lineage
        ):
            raise ValueError("history_block_lineage must be a tuple of frozen lineage rows")
        if not isinstance(
            self.current_observation_lineage,
            ProductBaselineCurrentObservationLineage,
        ):
            raise ValueError(
                "current_observation_lineage must be ProductBaselineCurrentObservationLineage"
            )
        expected_history_refs = tuple(
            (block.ordinal, block.artifact_id) for block in self.public_input.history
        )
        actual_history_refs = tuple(
            (item.ordinal, item.block_artifact_id) for item in self.history_block_lineage
        )
        if actual_history_refs != expected_history_refs:
            raise ValueError("history_block_lineage does not exactly match public_input.history")
        if (
            self.current_observation_lineage.observation_artifact_id
            != self.public_input.current_observation.artifact_id
        ):
            raise ValueError(
                "current_observation_lineage does not match public_input.current_observation"
            )
        _require_non_negative_int(self.seed, "seed")
        if self.arm is ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY:
            if self.top_k is not None:
                raise ValueError("native chronological history requires top_k=null")
        elif self.arm is ProductBaselineArm.SELECTIVE_SEMANTIC_RAG:
            _require_positive_int(self.top_k, "top_k")
            if not self.public_input.history:
                raise ValueError("selective semantic RAG requires at least one public exchange")
        else:  # pragma: no cover - enum construction closes this branch
            raise ValueError("unsupported product baseline arm")

    def to_payload(self) -> dict[str, object]:
        return {
            "nonce": self.nonce,
            "arm": self.arm.value,
            "public_plan_artifact_id": self.public_plan_artifact_id,
            "subject_scope": self.subject_scope,
            "decision_boundary": self.decision_boundary.to_payload(),
            "ordered_source_session_ids": list(self.ordered_source_session_ids),
            "ordered_source_block_artifact_ids": list(self.ordered_source_block_artifact_ids),
            "public_ledger_artifact_id": self.public_ledger_artifact_id,
            "public_input": self.public_input.to_payload(),
            "history_block_lineage": [item.to_payload() for item in self.history_block_lineage],
            "current_observation_lineage": self.current_observation_lineage.to_payload(),
            "seed": self.seed,
            "top_k": self.top_k,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ProductBaselineDispatcherRequest:
        raw = _require_exact_keys(payload, _REQUEST_KEYS, "dispatcher request")
        nonce = _require_non_empty_text(raw["nonce"], "nonce")
        if not isinstance(raw["arm"], str):
            raise ValueError("arm must be a string")
        try:
            arm = ProductBaselineArm(raw["arm"])
        except ValueError as exc:
            raise ValueError("arm must identify the native-history or semantic-RAG baseline") from exc
        raw_history_lineage = raw["history_block_lineage"]
        if not isinstance(raw_history_lineage, list):
            raise ValueError("history_block_lineage must be a list")
        raw_session_ids = raw["ordered_source_session_ids"]
        if not isinstance(raw_session_ids, list):
            raise ValueError("ordered_source_session_ids must be a list")
        raw_block_ids = raw["ordered_source_block_artifact_ids"]
        if not isinstance(raw_block_ids, list):
            raise ValueError("ordered_source_block_artifact_ids must be a list")
        return cls(
            nonce=nonce,
            arm=arm,
            public_plan_artifact_id=_require_sha256(
                raw["public_plan_artifact_id"],
                "public_plan_artifact_id",
            ),
            subject_scope=_require_sha256(raw["subject_scope"], "subject_scope"),
            decision_boundary=ProductBaselineDecisionBoundary.from_payload(
                raw["decision_boundary"]
            ),
            ordered_source_session_ids=tuple(
                _require_non_empty_text(value, f"ordered_source_session_ids[{index}]")
                for index, value in enumerate(raw_session_ids)
            ),
            ordered_source_block_artifact_ids=tuple(
                _require_sha256(value, f"ordered_source_block_artifact_ids[{index}]")
                for index, value in enumerate(raw_block_ids)
            ),
            public_ledger_artifact_id=_require_sha256(
                raw["public_ledger_artifact_id"],
                "public_ledger_artifact_id",
            ),
            public_input=parse_product_baseline_input(raw["public_input"]),
            history_block_lineage=tuple(
                ProductBaselineHistoryBlockLineage.from_payload(item)
                for item in raw_history_lineage
            ),
            current_observation_lineage=(
                ProductBaselineCurrentObservationLineage.from_payload(
                    raw["current_observation_lineage"]
                )
            ),
            seed=_require_non_negative_int(raw["seed"], "seed"),
            top_k=raw["top_k"],
        )


@dataclass(frozen=True)
class ProductBaselineDispatcherTokenReceipt:
    """Small exact-token projection bound to the full returned result."""

    action_completion_artifact_id: str
    context_receipt_artifact_id: str
    prompt_tokens: int
    completion_tokens: int
    generation_reserve_tokens: int
    total_reserved_tokens: int
    context_window_tokens: int

    def __post_init__(self) -> None:
        _require_sha256(self.action_completion_artifact_id, "action_completion_artifact_id")
        _require_sha256(self.context_receipt_artifact_id, "context_receipt_artifact_id")
        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "generation_reserve_tokens",
            "total_reserved_tokens",
            "context_window_tokens",
        ):
            _require_non_negative_int(getattr(self, field_name), field_name)
        if self.completion_tokens > self.generation_reserve_tokens:
            raise ValueError("completion_tokens exceed generation_reserve_tokens")
        if self.total_reserved_tokens > self.context_window_tokens:
            raise ValueError("total_reserved_tokens exceed context_window_tokens")

    @classmethod
    def from_result(cls, result: ProductBaselineResult) -> ProductBaselineDispatcherTokenReceipt:
        return cls(
            action_completion_artifact_id=result.action_completion.artifact_id,
            context_receipt_artifact_id=result.context_receipt.artifact_id,
            prompt_tokens=result.action_completion.prompt_tokens,
            completion_tokens=result.action_completion.completion_tokens,
            generation_reserve_tokens=result.context_receipt.generation_reserve_tokens,
            total_reserved_tokens=result.context_receipt.total_reserved_tokens,
            context_window_tokens=result.context_receipt.context_window_tokens,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "action_completion_artifact_id": self.action_completion_artifact_id,
            "context_receipt_artifact_id": self.context_receipt_artifact_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "generation_reserve_tokens": self.generation_reserve_tokens,
            "total_reserved_tokens": self.total_reserved_tokens,
            "context_window_tokens": self.context_window_tokens,
        }


@dataclass(frozen=True)
class ProductBaselineDispatcherResponse:
    """One ordered successful response with the complete baseline evidence."""

    nonce: str
    result: ProductBaselineResult
    action_id: RelationshipAction | None
    valid: bool
    generation_config_sha256: str
    token_receipt: ProductBaselineDispatcherTokenReceipt
    schema_version: str = PRODUCT_BASELINE_DISPATCHER_RESPONSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty_text(self.nonce, "nonce")
        if not isinstance(self.result, ProductBaselineResult):
            raise ValueError("result must be ProductBaselineResult")
        if self.action_id is not None and not isinstance(self.action_id, RelationshipAction):
            raise ValueError("action_id must be a RelationshipAction or None")
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a bool")
        _require_sha256(self.generation_config_sha256, "generation_config_sha256")
        if not isinstance(self.token_receipt, ProductBaselineDispatcherTokenReceipt):
            raise ValueError("token_receipt must be ProductBaselineDispatcherTokenReceipt")
        completion = self.result.action_completion
        if self.action_id is not completion.chosen_action_id or self.valid is not completion.valid:
            raise ValueError("action/valid projection disagrees with the full result")
        if self.generation_config_sha256 != self.result.context_receipt.generation_config_sha256:
            raise ValueError("generation_config_sha256 projection disagrees with the full result")
        expected_token_receipt = ProductBaselineDispatcherTokenReceipt.from_result(self.result)
        if self.token_receipt != expected_token_receipt:
            raise ValueError("token_receipt projection disagrees with the full result")
        chronological_selection = (
            self.result.retrieval_receipt.selected_chronological_block_artifact_ids
        )
        settled_selection = (
            *self.result.truncation_receipt.dropped_oldest_block_artifact_ids,
            *self.result.truncation_receipt.included_block_artifact_ids,
        )
        if settled_selection != chronological_selection:
            raise ValueError(
                "dropped plus included blocks must exactly reconstruct retrieval selection"
            )
        if self.schema_version != PRODUCT_BASELINE_DISPATCHER_RESPONSE_SCHEMA_VERSION:
            raise ValueError("dispatcher response schema_version mismatch")

    @classmethod
    def from_result(
        cls,
        *,
        nonce: str,
        result: ProductBaselineResult,
    ) -> ProductBaselineDispatcherResponse:
        return cls(
            nonce=nonce,
            result=result,
            action_id=result.action_completion.chosen_action_id,
            valid=result.action_completion.valid,
            generation_config_sha256=result.context_receipt.generation_config_sha256,
            token_receipt=ProductBaselineDispatcherTokenReceipt.from_result(result),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "nonce": self.nonce,
            "status": "ok",
            "result": self.result.to_payload(),
            "action_id": self.action_id.value if self.action_id is not None else None,
            "valid": self.valid,
            "generation_config_sha256": self.generation_config_sha256,
            "token_receipt": self.token_receipt.to_payload(),
        }


@dataclass(frozen=True)
class ProductBaselineDispatcherFatalResponse:
    """Terminal process-boundary failure; no later request may execute."""

    nonce: str | None
    error_type: str
    error_message: str
    schema_version: str = PRODUCT_BASELINE_DISPATCHER_FATAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.nonce is not None:
            _require_non_empty_text(self.nonce, "nonce")
        _require_non_empty_text(self.error_type, "error_type")
        _require_non_empty_text(self.error_message, "error_message")
        if self.schema_version != PRODUCT_BASELINE_DISPATCHER_FATAL_SCHEMA_VERSION:
            raise ValueError("dispatcher fatal schema_version mismatch")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "nonce": self.nonce,
            "status": "fatal",
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ProductBaselineDispatcherFatalResponse:
        raw = _require_exact_keys(
            payload,
            frozenset(
                {"schema_version", "nonce", "status", "error_type", "error_message"}
            ),
            "dispatcher fatal response",
        )
        if raw["status"] != "fatal":
            raise ValueError("dispatcher fatal response status mismatch")
        return cls(
            nonce=raw["nonce"],
            error_type=raw["error_type"],
            error_message=raw["error_message"],
            schema_version=raw["schema_version"],
        )


_SUCCESS_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "nonce",
        "status",
        "result",
        "action_id",
        "valid",
        "generation_config_sha256",
        "token_receipt",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "arm",
        "seed",
        "input_artifact_id",
        "action_completion",
        "context_receipt",
        "retrieval_receipt",
        "truncation_receipt",
        "artifact_id",
    }
)
_ACTION_COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "raw_output",
        "chosen_action_id",
        "valid",
        "prompt_tokens",
        "completion_tokens",
        "artifact_id",
    }
)
_CONTEXT_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "arm",
        "input_artifact_id",
        "model_id",
        "weights_sha256",
        "generation_config_sha256",
        "arm_prompt_sha256",
        "tokenizer_id",
        "token_budget_artifact_id",
        "prompt_and_current_tokens",
        "final_prompt_tokens",
        "history_increment_tokens",
        "generation_reserve_tokens",
        "total_reserved_tokens",
        "context_window_tokens",
        "included_block_artifact_ids",
        "rendered_messages_sha256",
        "artifact_id",
    }
)
_RETRIEVAL_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "arm",
        "input_artifact_id",
        "strategy",
        "candidate_count",
        "requested_top_k",
        "effective_top_k",
        "embedder_id",
        "query_content_sha256",
        "ranked_candidates",
        "selected_block_artifact_ids",
        "selected_chronological_block_artifact_ids",
        "artifact_id",
    }
)
_RETRIEVAL_CANDIDATE_KEYS = frozenset(
    {
        "schema_version",
        "retrieval_rank",
        "block_artifact_id",
        "block_ordinal",
        "cosine_score_hex",
        "artifact_id",
    }
)
_TRUNCATION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "input_artifact_id",
        "initial_prompt_tokens",
        "final_prompt_tokens",
        "dropped_oldest_block_artifact_ids",
        "included_block_artifact_ids",
        "was_truncated",
        "granularity",
        "reason",
        "artifact_id",
    }
)
_TOKEN_RECEIPT_KEYS = frozenset(
    {
        "action_completion_artifact_id",
        "context_receipt_artifact_id",
        "prompt_tokens",
        "completion_tokens",
        "generation_reserve_tokens",
        "total_reserved_tokens",
        "context_window_tokens",
    }
)


def _validate_content_addressed_object(
    payload: object,
    *,
    expected_keys: frozenset[str],
    field_name: str,
) -> dict[str, object]:
    raw = _require_exact_keys(payload, expected_keys, field_name)
    artifact_id = _require_sha256(raw["artifact_id"], f"{field_name}.artifact_id")
    core = {key: value for key, value in raw.items() if key != "artifact_id"}
    if artifact_id != sha256_json(core):
        raise ValueError(f"{field_name} artifact_id mismatch")
    return raw


@dataclass(frozen=True)
class ProductBaselineDispatcherReceivedResponse:
    """Strict, immutable consumer receipt for one canonical success line."""

    canonical_response_json: str
    nonce: str
    result_artifact_id: str
    action_id: RelationshipAction | None
    valid: bool
    generation_config_sha256: str
    token_receipt: ProductBaselineDispatcherTokenReceipt

    def __post_init__(self) -> None:
        _require_non_empty_text(self.canonical_response_json, "canonical_response_json")
        _require_non_empty_text(self.nonce, "nonce")
        _require_sha256(self.result_artifact_id, "result_artifact_id")
        if self.action_id is not None and not isinstance(self.action_id, RelationshipAction):
            raise ValueError("action_id must be a RelationshipAction or None")
        if not isinstance(self.valid, bool) or self.valid is not (self.action_id is not None):
            raise ValueError("valid must exactly reflect action_id presence")
        _require_sha256(self.generation_config_sha256, "generation_config_sha256")
        if not isinstance(self.token_receipt, ProductBaselineDispatcherTokenReceipt):
            raise TypeError("token_receipt must be ProductBaselineDispatcherTokenReceipt")

    @property
    def result_payload(self) -> dict[str, object]:
        response = json.loads(self.canonical_response_json)
        result = response["result"]
        if not isinstance(result, dict):  # pragma: no cover - parser establishes this
            raise RuntimeError("stored canonical response result is not an object")
        return result


def _parse_token_receipt(payload: object) -> ProductBaselineDispatcherTokenReceipt:
    raw = _require_exact_keys(payload, _TOKEN_RECEIPT_KEYS, "dispatcher token_receipt")
    return ProductBaselineDispatcherTokenReceipt(
        action_completion_artifact_id=raw["action_completion_artifact_id"],
        context_receipt_artifact_id=raw["context_receipt_artifact_id"],
        prompt_tokens=raw["prompt_tokens"],
        completion_tokens=raw["completion_tokens"],
        generation_reserve_tokens=raw["generation_reserve_tokens"],
        total_reserved_tokens=raw["total_reserved_tokens"],
        context_window_tokens=raw["context_window_tokens"],
    )


def _validate_received_result(payload: object) -> dict[str, object]:
    result = _require_exact_keys(payload, _RESULT_KEYS, "dispatcher result")
    result_artifact_id = _require_sha256(
        result["artifact_id"],
        "dispatcher result.artifact_id",
    )
    action = _validate_content_addressed_object(
        result["action_completion"],
        expected_keys=_ACTION_COMPLETION_KEYS,
        field_name="dispatcher result.action_completion",
    )
    context = _validate_content_addressed_object(
        result["context_receipt"],
        expected_keys=_CONTEXT_RECEIPT_KEYS,
        field_name="dispatcher result.context_receipt",
    )
    retrieval = _validate_content_addressed_object(
        result["retrieval_receipt"],
        expected_keys=_RETRIEVAL_RECEIPT_KEYS,
        field_name="dispatcher result.retrieval_receipt",
    )
    truncation = _validate_content_addressed_object(
        result["truncation_receipt"],
        expected_keys=_TRUNCATION_RECEIPT_KEYS,
        field_name="dispatcher result.truncation_receipt",
    )
    ranked = retrieval["ranked_candidates"]
    if not isinstance(ranked, list):
        raise ValueError("dispatcher result ranked_candidates must be a list")
    candidates = tuple(
        _validate_content_addressed_object(
            candidate,
            expected_keys=_RETRIEVAL_CANDIDATE_KEYS,
            field_name=f"dispatcher result.ranked_candidates[{index}]",
        )
        for index, candidate in enumerate(ranked)
    )
    candidate_count = _require_non_negative_int(
        retrieval["candidate_count"],
        "dispatcher result retrieval candidate_count",
    )
    effective_top_k = _require_non_negative_int(
        retrieval["effective_top_k"],
        "dispatcher result retrieval effective_top_k",
    )
    selected_ranked = retrieval["selected_block_artifact_ids"]
    selected = retrieval["selected_chronological_block_artifact_ids"]
    dropped = truncation["dropped_oldest_block_artifact_ids"]
    included = truncation["included_block_artifact_ids"]
    for field_name, values in (
        ("selected_block_artifact_ids", selected_ranked),
        ("selected_chronological_block_artifact_ids", selected),
        ("dropped_oldest_block_artifact_ids", dropped),
        ("included_block_artifact_ids", included),
    ):
        if not isinstance(values, list):
            raise ValueError(f"dispatcher result {field_name} must be a list")
        for index, value in enumerate(values):
            _require_sha256(value, f"dispatcher result {field_name}[{index}]")
    if tuple(dropped) + tuple(included) != tuple(selected):
        raise ValueError("dispatcher result truncation does not partition retrieval selection")
    if len(set(selected)) != len(selected):
        raise ValueError("dispatcher result retrieval selection contains duplicate exchanges")
    if set(selected_ranked) != set(selected) or len(selected_ranked) != len(selected):
        raise ValueError("dispatcher result ranked/chronological selections differ")
    arm = retrieval["arm"]
    if arm == ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY.value:
        if retrieval["requested_top_k"] is not None:
            raise ValueError("native dispatcher result requested_top_k must be null")
        if candidates:
            raise ValueError("native dispatcher result must not contain ranked candidates")
        if effective_top_k != candidate_count or len(selected_ranked) != candidate_count:
            raise ValueError("native dispatcher result must select every public exchange")
    elif arm == ProductBaselineArm.SELECTIVE_SEMANTIC_RAG.value:
        requested_top_k = _require_positive_int(
            retrieval["requested_top_k"],
            "dispatcher result retrieval requested_top_k",
        )
        if candidate_count <= 0:
            raise ValueError("semantic dispatcher result requires public exchange candidates")
        if len(candidates) != candidate_count:
            raise ValueError("dispatcher result ranked candidate count mismatch")
        if effective_top_k != min(requested_top_k, candidate_count):
            raise ValueError(
                "dispatcher result effective_top_k does not equal min(requested_top_k, candidate_count)"
            )
        ranked_candidate_ids = tuple(candidate["block_artifact_id"] for candidate in candidates)
        if tuple(selected_ranked) != ranked_candidate_ids[:effective_top_k]:
            raise ValueError("dispatcher result RAG selection is not the ranked effective top-k")
    else:
        raise ValueError("dispatcher result retrieval arm is unsupported")
    ordinal_by_id = {
        candidate["block_artifact_id"]: candidate["block_ordinal"] for candidate in candidates
    }
    if candidates and len(ordinal_by_id) != len(candidates):
        raise ValueError("dispatcher result ranked candidate block ids must be unique")
    if candidates:
        try:
            selected_ordinals = tuple(ordinal_by_id[value] for value in selected)
        except KeyError as exc:
            raise ValueError("dispatcher result selected block is absent from ranked candidates") from exc
        if selected_ordinals != tuple(sorted(selected_ordinals)):
            raise ValueError("dispatcher result RAG selection is not chronological")
    input_artifact_id = _require_sha256(result["input_artifact_id"], "result input_artifact_id")
    for receipt_name, receipt in (
        ("context_receipt", context),
        ("retrieval_receipt", retrieval),
        ("truncation_receipt", truncation),
    ):
        if receipt["input_artifact_id"] != input_artifact_id:
            raise ValueError(f"dispatcher result {receipt_name} input lineage mismatch")
    if context["included_block_artifact_ids"] != included:
        raise ValueError("dispatcher result context/truncation included blocks differ")
    if action["prompt_tokens"] != context["final_prompt_tokens"]:
        raise ValueError("dispatcher result action/context prompt token counts differ")
    if context["arm"] != result["arm"] or retrieval["arm"] != result["arm"]:
        raise ValueError("dispatcher result arm lineage mismatch")
    result_core = {key: value for key, value in result.items() if key != "artifact_id"}
    if result_artifact_id != sha256_json(result_core):
        raise ValueError("dispatcher result artifact_id mismatch")
    return result


def parse_product_baseline_dispatcher_response_line(
    raw_line: str,
) -> ProductBaselineDispatcherReceivedResponse | ProductBaselineDispatcherFatalResponse:
    """Parse one canonical response and reverify every nested artifact id."""

    if not isinstance(raw_line, str):
        raise TypeError("raw dispatcher response line must be a string")
    canonical_raw = raw_line[:-1] if raw_line.endswith("\n") else raw_line
    if canonical_raw.endswith("\r"):
        canonical_raw = canonical_raw[:-1]
    try:
        payload = json.loads(canonical_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("dispatcher response line must be valid JSON") from exc
    if canonical_raw != canonical_json(payload):
        raise ValueError("dispatcher response line must use canonical JSON bytes")
    if not isinstance(payload, dict):
        raise ValueError("dispatcher response must be an object")
    if payload.get("status") == "fatal":
        return ProductBaselineDispatcherFatalResponse.from_payload(payload)
    raw = _require_exact_keys(payload, _SUCCESS_RESPONSE_KEYS, "dispatcher success response")
    if raw["status"] != "ok":
        raise ValueError("dispatcher success response status mismatch")
    if raw["schema_version"] != PRODUCT_BASELINE_DISPATCHER_RESPONSE_SCHEMA_VERSION:
        raise ValueError("dispatcher success response schema_version mismatch")
    nonce = _require_non_empty_text(raw["nonce"], "dispatcher response nonce")
    result = _validate_received_result(raw["result"])
    action_payload = result["action_completion"]
    context_payload = result["context_receipt"]
    assert isinstance(action_payload, dict)
    assert isinstance(context_payload, dict)
    raw_action_id = raw["action_id"]
    if raw_action_id is None:
        action_id = None
    elif isinstance(raw_action_id, str):
        try:
            action_id = RelationshipAction(raw_action_id)
        except ValueError as exc:
            raise ValueError("dispatcher response action_id is outside the closed surface") from exc
    else:
        raise ValueError("dispatcher response action_id must be a string or null")
    if type(raw["valid"]) is not bool:
        raise ValueError("dispatcher response valid must be a bool")
    valid = raw["valid"]
    expected_action_value = action_id.value if action_id is not None else None
    if action_payload["chosen_action_id"] != expected_action_value:
        raise ValueError("dispatcher response action projection mismatch")
    if action_payload["valid"] is not valid or valid is not (action_id is not None):
        raise ValueError("dispatcher response valid projection mismatch")
    generation_config_sha256 = _require_sha256(
        raw["generation_config_sha256"],
        "dispatcher response generation_config_sha256",
    )
    if context_payload["generation_config_sha256"] != generation_config_sha256:
        raise ValueError("dispatcher response generation config projection mismatch")
    token_receipt = _parse_token_receipt(raw["token_receipt"])
    expected_token_values = {
        "action_completion_artifact_id": action_payload["artifact_id"],
        "context_receipt_artifact_id": context_payload["artifact_id"],
        "prompt_tokens": action_payload["prompt_tokens"],
        "completion_tokens": action_payload["completion_tokens"],
        "generation_reserve_tokens": context_payload["generation_reserve_tokens"],
        "total_reserved_tokens": context_payload["total_reserved_tokens"],
        "context_window_tokens": context_payload["context_window_tokens"],
    }
    if token_receipt.to_payload() != expected_token_values:
        raise ValueError("dispatcher response token receipt projection mismatch")
    return ProductBaselineDispatcherReceivedResponse(
        canonical_response_json=canonical_raw,
        nonce=nonce,
        result_artifact_id=result["artifact_id"],
        action_id=action_id,
        valid=valid,
        generation_config_sha256=generation_config_sha256,
        token_receipt=token_receipt,
    )


class RevisionPinnedCachedPublicSemanticEmbedder:
    """Byte/runtime-pinned BGE with an exact-text, public-request-only cache.

    The dispatcher is the only caller and passes only strings reconstructed
    from a verified ``ProductBaselineInput``.  The cache key is the UTF-8
    sha256 and a collision check retains the exact text.  Nothing about the
    campaign environment, evaluator truth, or future settlement is accepted.
    """

    def __init__(
        self,
        *,
        model_source: str,
        model_revision: str,
        weights_sha256: str,
        sentence_transformers_version: str,
        device: str,
        model_factory: Callable[..., object] | None = None,
        snapshot_path: pathlib.Path | None = None,
        snapshot_resolver: Callable[..., str | pathlib.Path] | None = None,
        runtime_version_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self._model_source = _require_non_empty_text(model_source, "BGE model_source")
        if self._model_source != BGE_M3_MODEL_ID:
            raise ValueError(f"BGE model_source must be exactly {BGE_M3_MODEL_ID}")
        self._model_revision = _require_lower_hex_revision(
            model_revision,
            "BGE model_revision",
        )
        self._weights_sha256 = _require_sha256(
            weights_sha256,
            "BGE weights_sha256",
        )
        self._sentence_transformers_version = _require_non_empty_text(
            sentence_transformers_version,
            "BGE sentence_transformers_version",
        )
        self._device = _require_non_empty_text(device, "BGE device")
        self._source_embedder = RevisionPinnedBgeM3PublicSemanticEmbedder(
            model_revision=self._model_revision,
            weights_sha256=self._weights_sha256,
            sentence_transformers_version=self._sentence_transformers_version,
            device=self._device,
            model_factory=model_factory,
            snapshot_path=snapshot_path,
            snapshot_resolver=snapshot_resolver,
            runtime_version_resolver=runtime_version_resolver,
        )
        self._cache: dict[str, tuple[str, tuple[float, ...]]] = {}
        self._embedding_width: int | None = None

    @property
    def name(self) -> str:
        return bge_m3_weight_pinned_embedder_identity(
            model_revision=self._model_revision,
            weights_sha256=self._weights_sha256,
            sentence_transformers_version=self._sentence_transformers_version,
            identity_kind="live-public-exact-text-cache-v2",
        )

    @property
    def model_source(self) -> str:
        return self._model_source

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def weights_sha256(self) -> str:
        return self._weights_sha256

    @property
    def sentence_transformers_version(self) -> str:
        return self._sentence_transformers_version

    def embed(self, text: str) -> tuple[float, ...]:
        exact_text = _require_non_empty_text(text, "public semantic text")
        import hashlib

        digest = hashlib.sha256(exact_text.encode("utf-8")).hexdigest()
        cached = self._cache.get(digest)
        if cached is not None:
            cached_text, vector = cached
            if cached_text != exact_text:
                raise RuntimeError("sha256 collision in live public semantic cache")
            return vector
        vector = self._source_embedder.embed(exact_text)
        if self._embedding_width is None:
            self._embedding_width = len(vector)
        elif len(vector) != self._embedding_width:
            raise ValueError("BGE embedding width changed within the resident process")
        self._cache[digest] = (exact_text, vector)
        return vector

    @property
    def cached_text_count(self) -> int:
        return len(self._cache)

    def export_table(self) -> PrecomputedPublicEmbeddingTable:
        """Freeze every actually observed public string into an offline table."""

        if not self._cache or self._embedding_width is None:
            raise ValueError("cannot export an empty live public semantic cache")
        records = tuple(
            sorted(
                (
                    PrecomputedPublicEmbeddingRecord(
                        text=text,
                        embedding_hex=tuple(value.hex() for value in vector),
                    )
                    for text, vector in self._cache.values()
                ),
                key=lambda record: (record.text_sha256, record.text),
            )
        )
        return PrecomputedPublicEmbeddingTable(
            source_embedder_name=self.name,
            embedding_width=self._embedding_width,
            records=records,
        )


@dataclass(frozen=True)
class ProductBaselineDispatcherConfig:
    """Composition-root settings for the single resident model process."""

    model_source: str = DEFAULT_STATELESS_MODEL_SOURCE
    model_id: str = DEFAULT_STATELESS_MODEL_ID
    model_revision: str = DEFAULT_PRODUCT_BASELINE_MODEL_REVISION
    device: str = "cuda"
    torch_dtype: str = "float16"
    context_window_tokens: int = 32768
    generation_reserve_tokens: int = 64
    prefill_chunk_size: int | None = None
    generation_use_cache: bool | None = None
    schema_constrained_decoding: bool = False
    semantic_mode: ProductBaselineSemanticMode = ProductBaselineSemanticMode.LIVE_BGE_M3_CACHED
    precomputed_embedding_table_path: pathlib.Path | None = None
    bge_model_source: str = DEFAULT_PRODUCT_BASELINE_BGE_SOURCE
    bge_model_revision: str = DEFAULT_PRODUCT_BASELINE_BGE_REVISION
    bge_weights_sha256: str = DEFAULT_PRODUCT_BASELINE_BGE_WEIGHTS_SHA256
    bge_sentence_transformers_version: str = (
        DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION
    )
    bge_device: str = "cuda"
    export_embedding_table_path: pathlib.Path | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.model_source, "model_source")
        _require_non_empty_text(self.model_id, "model_id")
        _require_lower_hex_revision(self.model_revision, "model_revision")
        _require_non_empty_text(self.device, "device")
        if self.torch_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("torch_dtype must be auto, bfloat16, float16, or float32")
        _require_positive_int(self.context_window_tokens, "context_window_tokens")
        reserve = _require_positive_int(
            self.generation_reserve_tokens,
            "generation_reserve_tokens",
        )
        if reserve < 4:
            raise ValueError("generation_reserve_tokens must be >= 4")
        if reserve >= self.context_window_tokens:
            raise ValueError("generation_reserve_tokens must be smaller than context_window_tokens")
        if self.prefill_chunk_size is not None:
            chunk_size = _require_positive_int(
                self.prefill_chunk_size,
                "prefill_chunk_size",
            )
            if chunk_size >= self.context_window_tokens:
                raise ValueError("prefill_chunk_size must be smaller than context_window_tokens")
            if self.generation_use_cache is not True:
                raise ValueError("chunked prefill requires generation_use_cache=True")
        if self.generation_use_cache is not None and not isinstance(
            self.generation_use_cache,
            bool,
        ):
            raise TypeError("generation_use_cache must be bool or None")
        if not isinstance(self.schema_constrained_decoding, bool):
            raise TypeError("schema_constrained_decoding must be bool")
        if not isinstance(self.semantic_mode, ProductBaselineSemanticMode):
            raise TypeError("semantic_mode must be ProductBaselineSemanticMode")
        if _require_non_empty_text(self.bge_model_source, "bge_model_source") != BGE_M3_MODEL_ID:
            raise ValueError(f"bge_model_source must be exactly {BGE_M3_MODEL_ID}")
        bge_m3_weight_pinned_embedder_identity(
            model_revision=self.bge_model_revision,
            weights_sha256=self.bge_weights_sha256,
            sentence_transformers_version=self.bge_sentence_transformers_version,
            identity_kind="live-public-exact-text-cache-v2",
        )
        _require_non_empty_text(self.bge_device, "bge_device")
        if self.semantic_mode is ProductBaselineSemanticMode.PRECOMPUTED:
            if not isinstance(self.precomputed_embedding_table_path, pathlib.Path):
                raise TypeError("precomputed mode requires precomputed_embedding_table_path")
            if self.export_embedding_table_path is not None:
                raise ValueError("precomputed mode cannot export a live embedding cache")
        elif self.precomputed_embedding_table_path is not None:
            raise ValueError("live cached mode must not receive a precomputed table path")
        if self.export_embedding_table_path is not None and not isinstance(
            self.export_embedding_table_path,
            pathlib.Path,
        ):
            raise TypeError("export_embedding_table_path must be pathlib.Path or None")


def build_product_baseline_dispatcher_suite(
    config: ProductBaselineDispatcherConfig,
) -> RelationshipProductBaselineSuite:
    """Load the policy and verified public table exactly once for a process."""

    if not isinstance(config, ProductBaselineDispatcherConfig):
        raise TypeError("config must be ProductBaselineDispatcherConfig")
    policy = HFStatelessRelationshipActionPolicy(
        model_source=config.model_source,
        model_id=config.model_id,
        model_revision=config.model_revision,
        device=config.device,
        torch_dtype=config.torch_dtype,
        local_files_only=True,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=config.generation_reserve_tokens,
        prefill_chunk_size=config.prefill_chunk_size,
        generation_use_cache=config.generation_use_cache,
        schema_constrained_decoding=config.schema_constrained_decoding,
    )
    if config.semantic_mode is ProductBaselineSemanticMode.PRECOMPUTED:
        assert config.precomputed_embedding_table_path is not None
        table = load_precomputed_public_embedding_table(config.precomputed_embedding_table_path)
        if table.source_model_id != config.bge_model_source:
            raise ValueError(
                "precomputed public embedding table does not bind the frozen BGE source"
            )
        if table.source_model_revision != config.bge_model_revision:
            raise ValueError(
                "precomputed public embedding table does not bind the frozen BGE revision"
            )
        table_weights = table.source_weights_sha256
        table_runtime = table.source_sentence_transformers_version
        if (table_weights is None) != (table_runtime is None):  # pragma: no cover - parser invariant
            raise ValueError("precomputed BGE identity has incomplete weight/runtime lineage")
        if table_weights is not None and table_weights != config.bge_weights_sha256:
            raise ValueError(
                "precomputed public embedding table does not bind the frozen BGE weights"
            )
        if (
            table_runtime is not None
            and table_runtime != config.bge_sentence_transformers_version
        ):
            raise ValueError(
                "precomputed public embedding table does not bind the frozen "
                "sentence-transformers runtime"
            )
        semantic_embedder = PrecomputedPublicSemanticEmbedder(table)
    else:
        semantic_embedder = RevisionPinnedCachedPublicSemanticEmbedder(
            model_source=config.bge_model_source,
            model_revision=config.bge_model_revision,
            weights_sha256=config.bge_weights_sha256,
            sentence_transformers_version=(
                config.bge_sentence_transformers_version
            ),
            device=config.bge_device,
        )
    return RelationshipProductBaselineSuite(
        policy=policy,
        token_counter=policy,
        token_budget=ProductBaselineTokenBudget(
            context_window_tokens=config.context_window_tokens,
            generation_reserve_tokens=config.generation_reserve_tokens,
        ),
        semantic_embedder=semantic_embedder,
    )


def export_live_public_embedding_table(
    *,
    suite: RelationshipProductBaselineSuite,
    path: pathlib.Path,
) -> pathlib.Path:
    """Export only a live cached embedder; precomputed mode is already frozen."""

    embedder = suite.semantic_embedder
    if not isinstance(embedder, RevisionPinnedCachedPublicSemanticEmbedder):
        raise TypeError("suite does not contain a live cached public semantic embedder")
    return write_precomputed_public_embedding_table(embedder.export_table(), path=path)


def validate_product_baseline_dispatcher_suite(
    suite: RelationshipProductBaselineSuite,
) -> None:
    """Require generation and exact counting to share one execution identity."""

    if not isinstance(suite, RelationshipProductBaselineSuite):
        raise TypeError("suite_factory must return RelationshipProductBaselineSuite")
    if suite.policy is not suite.token_counter:
        raise ValueError("dispatcher policy and exact token counter must be the same object")
    maximum_generation_tokens = suite.policy.max_new_tokens
    _require_positive_int(maximum_generation_tokens, "policy.max_new_tokens")
    if suite.token_budget.generation_reserve_tokens < maximum_generation_tokens:
        raise ValueError("generation reserve must cover policy.max_new_tokens")


def validate_product_baseline_dispatcher_result(
    *,
    request: ProductBaselineDispatcherRequest,
    result: ProductBaselineResult,
) -> None:
    """Close retrieval/truncation and public chronology before serialization."""

    if result.input_artifact_id != request.public_input.artifact_id:
        raise ValueError("baseline result does not bind the dispatched public input")
    if result.arm is not request.arm or result.seed != request.seed:
        raise ValueError("baseline result arm/seed lineage does not match the request")
    retrieval = result.retrieval_receipt
    truncation = result.truncation_receipt
    if retrieval.candidate_count != len(request.public_input.history):
        raise ValueError("retrieval candidate_count does not match dispatched exchange history")
    if request.arm is ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY:
        if retrieval.requested_top_k is not None:
            raise ValueError("native result must not claim a requested_top_k")
    else:
        if retrieval.requested_top_k != request.top_k:
            raise ValueError("RAG result requested_top_k does not match the dispatched request")
        assert request.top_k is not None
        if retrieval.effective_top_k != min(
            request.top_k,
            len(request.public_input.history),
        ):
            raise ValueError("RAG result effective_top_k drifted from min(requested K, history N)")
    chronological_selection = retrieval.selected_chronological_block_artifact_ids
    if (
        *truncation.dropped_oldest_block_artifact_ids,
        *truncation.included_block_artifact_ids,
    ) != chronological_selection:
        raise ValueError("truncation rows do not partition the retrieval selection")
    ordinal_by_artifact_id = {
        block.artifact_id: block.ordinal for block in request.public_input.history
    }
    if len(ordinal_by_artifact_id) != len(request.public_input.history):
        raise ValueError("public history block artifact ids must be unique")
    try:
        selected_ordinals = tuple(
            ordinal_by_artifact_id[artifact_id] for artifact_id in chronological_selection
        )
    except KeyError as exc:
        raise ValueError("retrieval selected a block outside the dispatched public input") from exc
    if selected_ordinals != tuple(sorted(selected_ordinals)):
        raise ValueError("retrieval chronological selection is not in public source order")


def dispatch_product_baseline_request(
    *,
    suite: RelationshipProductBaselineSuite,
    request: ProductBaselineDispatcherRequest,
) -> ProductBaselineDispatcherResponse:
    """Synchronously execute one already verified public request."""

    if not isinstance(request, ProductBaselineDispatcherRequest):
        raise TypeError("request must be ProductBaselineDispatcherRequest")
    if request.arm is ProductBaselineArm.NATIVE_CHRONOLOGICAL_FULL_HISTORY:
        result = suite.run_native_chronological_full_history(
            public_input=request.public_input,
            seed=request.seed,
        )
    elif request.arm is ProductBaselineArm.SELECTIVE_SEMANTIC_RAG:
        assert request.top_k is not None
        result = suite.run_selective_semantic_rag(
            public_input=request.public_input,
            seed=request.seed,
            top_k=request.top_k,
        )
    else:  # pragma: no cover - request enum closes this branch
        raise ValueError("unsupported product baseline arm")
    validate_product_baseline_dispatcher_result(request=request, result=result)
    return ProductBaselineDispatcherResponse.from_result(
        nonce=request.nonce,
        result=result,
    )


def _write_jsonl(output_stream: TextIO, payload: dict[str, object]) -> None:
    output_stream.write(canonical_json(payload))
    output_stream.write("\n")
    output_stream.flush()


def serve_product_baseline_jsonl(
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    suite_factory: Callable[[], RelationshipProductBaselineSuite],
    error_stream: TextIO | None = None,
    shutdown_hook: Callable[[RelationshipProductBaselineSuite], None] | None = None,
) -> int:
    """Serve ordered JSONL until clean EOF or the first terminal failure."""

    errors = sys.stderr if error_stream is None else error_stream
    try:
        suite = suite_factory()
        validate_product_baseline_dispatcher_suite(suite)
    except Exception as exc:  # process boundary: log, emit typed fatal, terminate
        traceback.print_exception(exc, file=errors)
        errors.flush()
        _write_jsonl(
            output_stream,
            ProductBaselineDispatcherFatalResponse(
                nonce=None,
                error_type=type(exc).__name__,
                error_message=str(exc) or repr(exc),
            ).to_payload(),
        )
        return 1

    for raw_line in input_stream:
        nonce: str | None = None
        try:
            payload = json.loads(raw_line)
            canonical_raw = raw_line[:-1] if raw_line.endswith("\n") else raw_line
            if canonical_raw.endswith("\r"):
                canonical_raw = canonical_raw[:-1]
            if canonical_raw != canonical_json(payload):
                raise ValueError("dispatcher request line must use canonical JSON bytes")
            if isinstance(payload, dict) and isinstance(payload.get("nonce"), str):
                candidate_nonce = payload["nonce"]
                if candidate_nonce.strip():
                    nonce = candidate_nonce
            request = ProductBaselineDispatcherRequest.from_payload(payload)
            response = dispatch_product_baseline_request(suite=suite, request=request)
        except Exception as exc:  # process boundary: log, emit typed fatal, terminate
            traceback.print_exception(exc, file=errors)
            errors.flush()
            _write_jsonl(
                output_stream,
                ProductBaselineDispatcherFatalResponse(
                    nonce=nonce,
                    error_type=type(exc).__name__,
                    error_message=str(exc) or repr(exc),
                ).to_payload(),
            )
            return 1
        _write_jsonl(output_stream, response.to_payload())
    if shutdown_hook is not None:
        try:
            shutdown_hook(suite)
        except Exception as exc:  # process boundary: log, emit typed fatal, terminate
            traceback.print_exception(exc, file=errors)
            errors.flush()
            _write_jsonl(
                output_stream,
                ProductBaselineDispatcherFatalResponse(
                    nonce=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc) or repr(exc),
                ).to_payload(),
            )
            return 1
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resident Relationship Lab native-history/RAG baseline dispatcher",
    )
    parser.add_argument("--model-source", default=DEFAULT_STATELESS_MODEL_SOURCE)
    parser.add_argument("--model-id", default=DEFAULT_STATELESS_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_PRODUCT_BASELINE_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="float16",
    )
    parser.add_argument("--context-window-tokens", type=int, default=32768)
    parser.add_argument("--generation-reserve-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int)
    parser.add_argument(
        "--generation-use-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--schema-constrained-decoding",
        action="store_true",
        help="constrain generation to one of the three canonical action JSON objects",
    )
    parser.add_argument(
        "--semantic-mode",
        choices=tuple(mode.value for mode in ProductBaselineSemanticMode),
        default=ProductBaselineSemanticMode.LIVE_BGE_M3_CACHED.value,
    )
    parser.add_argument("--precomputed-embedding-table", type=pathlib.Path)
    parser.add_argument("--bge-model-source", default=DEFAULT_PRODUCT_BASELINE_BGE_SOURCE)
    parser.add_argument("--bge-model-revision", default=DEFAULT_PRODUCT_BASELINE_BGE_REVISION)
    parser.add_argument(
        "--bge-weights-sha256",
        default=DEFAULT_PRODUCT_BASELINE_BGE_WEIGHTS_SHA256,
    )
    parser.add_argument(
        "--bge-sentence-transformers-version",
        default=DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION,
    )
    parser.add_argument("--bge-device", default="cuda")
    parser.add_argument("--export-public-embedding-table", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    config = ProductBaselineDispatcherConfig(
        model_source=args.model_source,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        torch_dtype=args.torch_dtype,
        context_window_tokens=args.context_window_tokens,
        generation_reserve_tokens=args.generation_reserve_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        generation_use_cache=args.generation_use_cache,
        schema_constrained_decoding=args.schema_constrained_decoding,
        semantic_mode=ProductBaselineSemanticMode(args.semantic_mode),
        precomputed_embedding_table_path=args.precomputed_embedding_table,
        bge_model_source=args.bge_model_source,
        bge_model_revision=args.bge_model_revision,
        bge_weights_sha256=args.bge_weights_sha256,
        bge_sentence_transformers_version=(
            args.bge_sentence_transformers_version
        ),
        bge_device=args.bge_device,
        export_embedding_table_path=args.export_public_embedding_table,
    )
    shutdown_hook: Callable[[RelationshipProductBaselineSuite], None] | None = None
    if config.export_embedding_table_path is not None:
        export_path = config.export_embedding_table_path

        def _shutdown_hook(suite: RelationshipProductBaselineSuite) -> None:
            export_live_public_embedding_table(suite=suite, path=export_path)

        shutdown_hook = _shutdown_hook
    return serve_product_baseline_jsonl(
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        suite_factory=lambda: build_product_baseline_dispatcher_suite(config),
        shutdown_hook=shutdown_hook,
    )


__all__ = [
    "PRODUCT_BASELINE_DISPATCHER_FATAL_SCHEMA_VERSION",
    "PRODUCT_BASELINE_DISPATCHER_REQUEST_SCHEMA_VERSION",
    "PRODUCT_BASELINE_DISPATCHER_RESPONSE_SCHEMA_VERSION",
    "DEFAULT_PRODUCT_BASELINE_BGE_REVISION",
    "DEFAULT_PRODUCT_BASELINE_BGE_SENTENCE_TRANSFORMERS_VERSION",
    "DEFAULT_PRODUCT_BASELINE_BGE_WEIGHTS_SHA256",
    "DEFAULT_PRODUCT_BASELINE_MODEL_REVISION",
    "ProductBaselineCurrentObservationLineage",
    "ProductBaselineDecisionBoundary",
    "ProductBaselineDispatcherConfig",
    "ProductBaselineDispatcherFatalResponse",
    "ProductBaselineDispatcherRequest",
    "ProductBaselineDispatcherReceivedResponse",
    "ProductBaselineDispatcherResponse",
    "ProductBaselineDispatcherTokenReceipt",
    "ProductBaselineHistoryBlockLineage",
    "ProductBaselineSemanticMode",
    "RevisionPinnedCachedPublicSemanticEmbedder",
    "build_product_baseline_dispatcher_suite",
    "dispatch_product_baseline_request",
    "export_live_public_embedding_table",
    "parse_product_baseline_input",
    "parse_product_baseline_dispatcher_response_line",
    "serve_product_baseline_jsonl",
    "validate_product_baseline_dispatcher_result",
    "validate_product_baseline_dispatcher_suite",
]
