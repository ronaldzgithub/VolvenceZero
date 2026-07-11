# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Tests for the cross-family extractor guard and embedder selection."""

from __future__ import annotations

import argparse

import pytest

from companion_ref_harness import cli
from companion_ref_harness.embed import HashingEmbedder, SentenceTransformerEmbedder
from companion_ref_harness.policy import parse_component_set


def _ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        use_stub_summary_extractor=False,
        summary_extractor_model=None,
        summary_extractor_base_url=None,
        embedder="hashing",
        embedder_device=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_require_cross_family_extractor_raises_without_config() -> None:
    components = parse_component_set("summary,embed,user_model,episodic")
    with pytest.raises(SystemExit):
        cli._require_cross_family_extractor(args=_ns(), components=components)


def test_require_cross_family_extractor_ok_with_config() -> None:
    components = parse_component_set("summary")
    cli._require_cross_family_extractor(
        args=_ns(
            summary_extractor_model="anthropic/claude",
            summary_extractor_base_url="https://api.anthropic.com/v1",
        ),
        components=components,
    )


def test_require_cross_family_extractor_stub_exempt() -> None:
    components = parse_component_set("summary")
    cli._require_cross_family_extractor(
        args=_ns(use_stub_summary_extractor=True), components=components,
    )


def test_require_cross_family_extractor_embed_only_exempt() -> None:
    # embed uses the embedder, not the LLM extractor, so it does not require one.
    components = parse_component_set("embed")
    cli._require_cross_family_extractor(args=_ns(), components=components)


def test_require_cross_family_extractor_env_override(monkeypatch) -> None:
    monkeypatch.setenv("REFH_ALLOW_SAME_FAMILY_EXTRACTOR", "1")
    components = parse_component_set("summary")
    cli._require_cross_family_extractor(args=_ns(), components=components)


def test_build_embedder_selects_bge_m3_lazily() -> None:
    components = parse_component_set("embed")
    emb = cli._build_embedder(components, args=_ns(embedder="bge-m3"))
    # constructed but not yet loaded (lazy) -> no network/model download here
    assert isinstance(emb, SentenceTransformerEmbedder)


def test_build_embedder_hashing_choice() -> None:
    components = parse_component_set("embed")
    emb = cli._build_embedder(components, args=_ns(embedder="hashing"))
    assert isinstance(emb, HashingEmbedder)
