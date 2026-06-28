# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""CLI parsing + backend-construction tests (no server boot)."""

from __future__ import annotations

import pytest

from companion_camel_baseline import __version__
from companion_camel_baseline.backend import EchoCamelBackend
from companion_camel_baseline.cli import _build_backend, _build_parser, main


def test_version_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == __version__


def test_serve_parser_defaults() -> None:
    args = _build_parser().parse_args(["serve"])
    assert args.subcommand == "serve"
    assert args.port == 8600
    assert args.backend == "camel"
    assert args.store_mode == "sqlite"


def test_echo_backend_needs_no_upstream() -> None:
    args = _build_parser().parse_args(["serve", "--backend", "echo"])
    backend = _build_backend(args)
    assert isinstance(backend, EchoCamelBackend)


def test_camel_backend_requires_upstream_flags() -> None:
    args = _build_parser().parse_args(["serve", "--backend", "camel"])
    with pytest.raises(SystemExit):
        _build_backend(args)
