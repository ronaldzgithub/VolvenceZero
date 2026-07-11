# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""CLI argument-parsing smoke tests.

These tests exercise only the argument parser and the upstream /
extractor factory helpers — they do not actually start the aiohttp
server (because that would bind a port). The full server flow is
covered by ``test_server_*.py``.
"""

from __future__ import annotations

import argparse

import pytest

from companion_ref_harness import cli


def _ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        verbose=False,
        subcommand="serve",
        port=8500,
        host="127.0.0.1",
        upstream_family="passthrough",
        upstream_base_url=None,
        upstream_model="ref-harness/echo",
        upstream_key_env=None,
        components="summary",
        summary_extractor_base_url=None,
        summary_extractor_model=None,
        summary_extractor_key_env=None,
        summary_extractor_family=None,
        use_stub_summary_extractor=True,
        embedder="hashing",
        embedder_device=None,
        store_mode="memory",
        store_path="./companion-ref-harness.sqlite3",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_serve_minimum_valid() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([
        "serve",
        "--upstream-family", "passthrough",
        "--upstream-model", "ref-harness/echo",
        "--components", "summary",
        "--use-stub-summary-extractor",
        "--store-mode", "memory",
    ])
    assert args.subcommand == "serve"
    assert args.upstream_family == "passthrough"
    assert args.components == "summary"


def test_parser_rejects_unknown_subcommand() -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["danceparty"])


def test_parser_components_default_is_summary() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["serve"])
    assert args.components == "summary"


def test_parser_version_subcommand() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["version"])
    assert args.subcommand == "version"


# ---------------------------------------------------------------------------
# Upstream / extractor factories
# ---------------------------------------------------------------------------


def test_build_upstream_passthrough_does_not_require_keys() -> None:
    upstream = cli._build_upstream_client(
        args=_ns(upstream_family="passthrough"),
        family=cli.UpstreamFamily.PASSTHROUGH,
    )
    assert isinstance(upstream, cli.EchoUpstreamClient)
    assert upstream.model == "ref-harness/echo"


def test_build_upstream_openai_compat_requires_url_model_key() -> None:
    with pytest.raises(SystemExit, match="--upstream-base-url"):
        cli._build_upstream_client(
            args=_ns(
                upstream_family="openai-compat",
                upstream_base_url=None,
                upstream_model=None,
                upstream_key_env=None,
            ),
            family=cli.UpstreamFamily.OPENAI_COMPAT,
        )


def test_build_upstream_openai_compat_with_missing_env_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_RH_NEVER_SET_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TEST_RH_NEVER_SET_API_KEY"):
        cli._build_upstream_client(
            args=_ns(
                upstream_family="openai-compat",
                upstream_base_url="https://api.example.com/v1",
                upstream_model="example/m-1",
                upstream_key_env="TEST_RH_NEVER_SET_API_KEY",
            ),
            family=cli.UpstreamFamily.OPENAI_COMPAT,
        )


def test_build_extractor_stub_when_flag_set() -> None:
    resolved = cli._build_extractor_call(
        args=_ns(use_stub_summary_extractor=True),
        family=cli.UpstreamFamily.PASSTHROUGH,
    )
    assert isinstance(cli._build_summary_extractor(resolved), cli.StubSummaryExtractor)


def test_build_extractor_passthrough_family_forces_stub() -> None:
    """passthrough has no LLM to call, so the factory should produce a stub."""
    resolved = cli._build_extractor_call(
        args=_ns(
            use_stub_summary_extractor=False,
            summary_extractor_family="passthrough",
        ),
        family=cli.UpstreamFamily.OPENAI_COMPAT,
    )
    assert isinstance(cli._build_summary_extractor(resolved), cli.StubSummaryExtractor)


def test_build_extractor_with_no_target_fails_loudly() -> None:
    """If neither flags nor upstream provide a target, we should fail clean."""
    with pytest.raises(SystemExit, match="memory extractors require"):
        cli._build_extractor_call(
            args=_ns(
                use_stub_summary_extractor=False,
                upstream_base_url=None,
                upstream_model=None,
                upstream_key_env=None,
            ),
            family=cli.UpstreamFamily.OPENAI_COMPAT,
        )


def test_build_components_for_full_stack() -> None:
    """All four memory components build their handlers when enabled."""
    from companion_ref_harness.policy import parse_component_set

    components = parse_component_set("summary,embed,user_model,episodic")
    resolved = cli._build_extractor_call(
        args=_ns(use_stub_summary_extractor=True),
        family=cli.UpstreamFamily.PASSTHROUGH,
    )
    assert cli._build_embedder(components, args=_ns()) is not None
    assert cli._build_user_fact_extractor(components, resolved) is not None
    assert cli._build_episodic_extractor(components, resolved) is not None
    # Disabled components return None.
    summary_only = parse_component_set("summary")
    assert cli._build_embedder(summary_only, args=_ns()) is None
    assert cli._build_user_fact_extractor(summary_only, resolved) is None
    assert cli._build_episodic_extractor(summary_only, resolved) is None
