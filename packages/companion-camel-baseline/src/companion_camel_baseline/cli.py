# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""``companion-camel-baseline`` console entry point.

Subcommands:

* ``serve`` — boot the aiohttp server.
* ``version`` — print the wheel version.

The ``serve`` subcommand is explicit about every upstream parameter (base URL,
model, key env) — the reproducibility contract requires the boot command shown
in a paper or pitch deck to specify the upstream exactly, with no implicit
model fallback.
"""

from __future__ import annotations

import argparse
import enum
import logging
import os
import pathlib
import sys

from companion_camel_baseline import __version__
from companion_camel_baseline.backend import (
    CamelBackend,
    CamelChatAgentBackend,
    EchoCamelBackend,
)
from companion_camel_baseline.memory_store import StoreMode, open_store


_LOG = logging.getLogger("companion_camel_baseline.cli")


class BackendKind(str, enum.Enum):
    ECHO = "echo"
    CAMEL = "camel"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(verbose=args.verbose)
    if args.subcommand == "serve":
        return _cmd_serve(args)
    if args.subcommand == "version":
        print(__version__)
        return 0
    parser.error(f"unknown subcommand {args.subcommand!r}")  # pragma: no cover
    return 2  # pragma: no cover


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companion-camel-baseline",
        description=(
            "CAMEL agent-framework baseline for CompanionBench. Boots an "
            "OpenAI-compatible /v1/chat/completions endpoint that wraps an "
            "upstream substrate with a CAMEL ChatAgent + cross-session memory."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable DEBUG-level logging",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    serve = sub.add_parser("serve", help="boot the baseline server")
    serve.add_argument("--port", type=int, default=8600, help="TCP port (default: 8600)")
    serve.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    serve.add_argument(
        "--backend",
        default=BackendKind.CAMEL.value,
        choices=[b.value for b in BackendKind],
        help=(
            "agent backend. 'camel' uses the real CAMEL ChatAgent (requires the "
            "[camel] extra + a configured upstream); 'echo' is the deterministic "
            "no-network smoke backend."
        ),
    )
    serve.add_argument(
        "--default-system-prompt",
        default="You are a long-running companion AI.",
        help="system prompt used when the request supplies none.",
    )
    # --- upstream (camel backend only) ---
    serve.add_argument(
        "--upstream-base-url",
        default=None,
        help="upstream OpenAI-compatible chat-completions root URL (camel backend).",
    )
    serve.add_argument(
        "--upstream-model",
        default=None,
        help="upstream model identifier (camel backend).",
    )
    serve.add_argument(
        "--upstream-key-env",
        default=None,
        help="env var holding the upstream API key (camel backend).",
    )
    # --- store ---
    serve.add_argument(
        "--store-mode",
        default=StoreMode.SQLITE.value,
        choices=[m.value for m in StoreMode],
        help="memory-record backend (default: sqlite)",
    )
    serve.add_argument(
        "--store-path",
        default="./companion-camel-baseline.sqlite3",
        help="SQLite file path when --store-mode=sqlite.",
    )

    sub.add_parser("version", help="print the wheel version")
    return parser


def _setup_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cmd_serve(args: argparse.Namespace) -> int:
    from aiohttp import web

    from companion_camel_baseline.server import build_app

    backend = _build_backend(args)
    store_mode = StoreMode(args.store_mode)
    store_path = (
        pathlib.Path(args.store_path) if store_mode is StoreMode.SQLITE else None
    )
    store = open_store(store_mode, sqlite_path=store_path)

    app = build_app(
        backend=backend,
        store=store,
        default_system_prompt=args.default_system_prompt,
    )
    _LOG.info(
        "companion-camel-baseline serving on http://%s:%d (backend=%s model=%s store=%s)",
        args.host, args.port, args.backend, backend.model, store_mode.value,
    )
    try:
        web.run_app(app, host=args.host, port=args.port, print=None)
    except KeyboardInterrupt:  # pragma: no cover - manual interrupt
        _LOG.info("interrupted; shutting down")
    return 0


def _build_backend(args: argparse.Namespace) -> CamelBackend:
    kind = BackendKind(args.backend)
    if kind is BackendKind.ECHO:
        return EchoCamelBackend()
    if kind is BackendKind.CAMEL:
        if not args.upstream_base_url or not args.upstream_model or not args.upstream_key_env:
            raise SystemExit(
                "--upstream-base-url, --upstream-model, and --upstream-key-env are "
                "required for --backend camel (no implicit model fallback)."
            )
        api_key = _required_env(args.upstream_key_env)
        return CamelChatAgentBackend(
            model_type=args.upstream_model,
            upstream_base_url=args.upstream_base_url,
            upstream_api_key=api_key,
        )
    raise SystemExit(f"unhandled backend: {args.backend}")  # pragma: no cover


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"required environment variable {name!r} is not set. "
            "Set it before invoking `companion-camel-baseline serve`."
        )
    return value


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
