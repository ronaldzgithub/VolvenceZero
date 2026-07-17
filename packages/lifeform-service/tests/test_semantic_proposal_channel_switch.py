"""Channel-level semantic-proposal ablation switch (experiment 2 of
docs/specs/semantic-grounding-evidence.md).

``VZ_SEMANTIC_PROPOSAL_CHANNEL=noop`` must force the explicit NoOp
runtime through the single vertical-factory choke point, even when a
runtime with HF internals is present; invalid values must fail loudly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lifeform_service import verticals
from volvence_zero.semantic_state import NoOpSemanticProposalRuntime


def test_default_channel_is_llm(monkeypatch) -> None:
    monkeypatch.delenv(verticals._SEMANTIC_PROPOSAL_CHANNEL_ENV, raising=False)
    assert verticals._semantic_proposal_channel() == "llm"


def test_noop_channel_forces_noop_even_with_hf_internals(monkeypatch) -> None:
    monkeypatch.setenv(verticals._SEMANTIC_PROPOSAL_CHANNEL_ENV, "noop")
    runtime = SimpleNamespace(
        _model=object(), _tokenizer=object(), _device="cpu"
    )
    built = verticals._build_llm_semantic_runtime_from_runtime(runtime)
    assert isinstance(built, NoOpSemanticProposalRuntime)


def test_noop_channel_applies_to_none_runtime(monkeypatch) -> None:
    monkeypatch.setenv(verticals._SEMANTIC_PROPOSAL_CHANNEL_ENV, "noop")
    built = verticals._build_llm_semantic_runtime_from_runtime(None)
    assert isinstance(built, NoOpSemanticProposalRuntime)


def test_invalid_channel_value_fails_loudly(monkeypatch) -> None:
    monkeypatch.setenv(verticals._SEMANTIC_PROPOSAL_CHANNEL_ENV, "off")
    with pytest.raises(ValueError, match="VZ_SEMANTIC_PROPOSAL_CHANNEL"):
        verticals._build_llm_semantic_runtime_from_runtime(None)


def test_llm_channel_preserves_fallback_behaviour(monkeypatch) -> None:
    monkeypatch.setenv(verticals._SEMANTIC_PROPOSAL_CHANNEL_ENV, "llm")
    # No HF internals -> no LLM runtime (existing synthetic-fallback rule).
    assert verticals._build_llm_semantic_runtime_from_runtime(None) is None
    assert (
        verticals._build_llm_semantic_runtime_from_runtime(SimpleNamespace())
        is None
    )
