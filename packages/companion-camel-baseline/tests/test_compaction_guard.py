# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Tests for the cross-family compaction extractor guard (camel backend)."""

from __future__ import annotations

import argparse

import pytest

from companion_camel_baseline import cli


def _ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        upstream_base_url="http://127.0.0.1:8000/v1?mode=raw",
        upstream_model="lifeform-raw",
        upstream_key_env="LIFEFORM_LOCAL_API_KEY",
        compaction_base_url=None,
        compaction_model=None,
        compaction_key_env=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_compaction_requires_cross_family() -> None:
    with pytest.raises(SystemExit):
        cli._resolve_compaction_config(_ns())


def test_compaction_ok_when_configured() -> None:
    base, model, key_env = cli._resolve_compaction_config(
        _ns(
            compaction_base_url="https://openrouter.ai/api/v1",
            compaction_model="anthropic/claude-3.7-sonnet",
            compaction_key_env="OPENROUTER_API_KEY",
        )
    )
    assert model == "anthropic/claude-3.7-sonnet"
    assert base == "https://openrouter.ai/api/v1"
    assert key_env == "OPENROUTER_API_KEY"


def test_compaction_env_override_falls_back_to_substrate(monkeypatch) -> None:
    monkeypatch.setenv("CAMEL_ALLOW_SAME_FAMILY_EXTRACTOR", "1")
    base, model, key_env = cli._resolve_compaction_config(_ns())
    assert model == "lifeform-raw"
    assert key_env == "LIFEFORM_LOCAL_API_KEY"
