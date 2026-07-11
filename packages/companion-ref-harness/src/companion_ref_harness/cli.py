# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""``companion-ref-harness`` console entry point.

Subcommands:

* ``serve`` — boot the aiohttp server.
* ``version`` — print the wheel version.

The ``serve`` subcommand is intentionally explicit about every
upstream parameter (base URL, model, key env var, family) — the
benchmark reproducibility contract requires that the harness boot
command shown in a paper or pitch deck specifies the upstream
exactly, with no implicit defaults beyond OpenAI-compat shape.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import sys

from companion_ref_harness import __version__
from companion_ref_harness.embed import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
)
from companion_ref_harness.episodic import (
    EpisodicExtractor,
    LLMEpisodicExtractor,
    StubEpisodicExtractor,
)
from companion_ref_harness.policy import ComponentSet, parse_component_set, HarnessComponent
from companion_ref_harness.session_summary import (
    LLMSummaryExtractor,
    StubSummaryExtractor,
    SummaryExtractor,
)
from companion_ref_harness.store.sqlite_store import StoreMode, open_store
from companion_ref_harness.upstream_client import (
    AnthropicUpstreamClient,
    EchoUpstreamClient,
    OpenAICompatUpstreamClient,
    UpstreamClient,
    UpstreamFamily,
    parse_upstream_family,
)
from companion_ref_harness.user_model import (
    LLMUserFactExtractor,
    StubUserFactExtractor,
    UserFactExtractor,
)


_LOG = logging.getLogger("companion_ref_harness.cli")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companion-ref-harness",
        description=(
            "Reference Companion Harness for CompanionBench. Boots an "
            "OpenAI-compatible /v1/chat/completions endpoint that wraps "
            "any upstream substrate with a configurable cross-session "
            "memory layer."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable DEBUG-level logging",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    serve = sub.add_parser("serve", help="boot the harness server")
    serve.add_argument(
        "--port",
        type=int,
        default=8500,
        help="TCP port to bind (default: 8500)",
    )
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="hostname to bind (default: 127.0.0.1)",
    )
    # --- upstream ---
    serve.add_argument(
        "--upstream-family",
        default=UpstreamFamily.OPENAI_COMPAT.value,
        choices=[f.value for f in UpstreamFamily],
        help="upstream protocol family (default: openai-compat)",
    )
    serve.add_argument(
        "--upstream-base-url",
        default=None,
        help=(
            "upstream chat-completions root URL. Required unless "
            "--upstream-family=passthrough."
        ),
    )
    serve.add_argument(
        "--upstream-model",
        default=None,
        help=(
            "upstream model identifier. Required unless "
            "--upstream-family=passthrough."
        ),
    )
    serve.add_argument(
        "--upstream-key-env",
        default=None,
        help=(
            "name of the environment variable holding the upstream API "
            "key. Required unless --upstream-family=passthrough."
        ),
    )
    # --- components ---
    serve.add_argument(
        "--components",
        default="summary",
        help=(
            "comma-separated component set. Allowed: "
            f"{','.join(c.value for c in HarnessComponent)}. "
            "Use '' (empty) for passthrough mode. All four components are "
            "wired (summary + embed retrieval + user_model + episodic)."
        ),
    )
    # --- summary extractor ---
    serve.add_argument(
        "--summary-extractor-base-url",
        default=None,
        help=(
            "base URL for the summary extractor LLM. Defaults to "
            "--upstream-base-url (NOT recommended — same-family extractor "
            "is the 'GPT-5 writes its own crib-notes' criticism)."
        ),
    )
    serve.add_argument(
        "--summary-extractor-model",
        default=None,
        help=(
            "model id for the summary extractor. Defaults to "
            "--upstream-model. Cross-family is the recommended setup."
        ),
    )
    serve.add_argument(
        "--summary-extractor-key-env",
        default=None,
        help=(
            "env var for the summary extractor API key. Defaults to "
            "--upstream-key-env."
        ),
    )
    serve.add_argument(
        "--summary-extractor-family",
        default=None,
        choices=[f.value for f in UpstreamFamily],
        help=(
            "summary extractor family. Defaults to --upstream-family. "
            "Set differently from --upstream-family for cross-family "
            "extraction."
        ),
    )
    serve.add_argument(
        "--use-stub-summary-extractor",
        action="store_true",
        help=(
            "use the deterministic StubSummaryExtractor instead of "
            "calling an LLM. Intended for tests / smoke runs without "
            "API keys."
        ),
    )
    # --- embed retrieval ---
    serve.add_argument(
        "--embedder",
        default="bge-m3",
        choices=["bge-m3", "hashing"],
        help=(
            "embedder for the H-B retrieval component. 'bge-m3' uses the "
            "BAAI/bge-m3 multilingual sentence-transformer (semantic "
            "retrieval; requires the [embed] extra + first-run download). "
            "'hashing' is the dependency-free bag-of-tokens fallback for "
            "offline / smoke runs. (default: bge-m3)"
        ),
    )
    serve.add_argument(
        "--embedder-device",
        default=None,
        help="device for the sentence-transformer embedder (e.g. cuda, cpu). "
        "Default lets sentence-transformers pick.",
    )
    # --- store ---
    serve.add_argument(
        "--store-mode",
        default=StoreMode.SQLITE.value,
        choices=[m.value for m in StoreMode],
        help="component-state backend (default: sqlite)",
    )
    serve.add_argument(
        "--store-path",
        default="./companion-ref-harness.sqlite3",
        help=(
            "SQLite file path when --store-mode=sqlite. Ignored for "
            "memory mode. (default: ./companion-ref-harness.sqlite3)"
        ),
    )

    sub.add_parser("version", help="print the wheel version")
    return parser


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from aiohttp import web  # local import; CLI users without web get a clearer error

    from companion_ref_harness.server import build_app

    components = parse_component_set(args.components)
    family = parse_upstream_family(args.upstream_family)
    _require_cross_family_extractor(args=args, components=components)
    upstream = _build_upstream_client(args=args, family=family)
    extractor_call = _build_extractor_call(args=args, family=family)
    summary_extractor = _build_summary_extractor(extractor_call)
    embedder = _build_embedder(components, args=args)
    user_fact_extractor = _build_user_fact_extractor(components, extractor_call)
    episodic_extractor = _build_episodic_extractor(components, extractor_call)
    store_mode = StoreMode(args.store_mode)
    store_path = (
        pathlib.Path(args.store_path)
        if store_mode is StoreMode.SQLITE else None
    )
    store = open_store(store_mode, sqlite_path=store_path)

    app = build_app(
        upstream=upstream,
        store=store,
        components=components,
        summary_extractor=summary_extractor,
        embedder=embedder,
        user_fact_extractor=user_fact_extractor,
        episodic_extractor=episodic_extractor,
    )
    _LOG.info(
        "companion-ref-harness serving on http://%s:%d (components=%s family=%s model=%s store=%s)",
        args.host,
        args.port,
        components.to_csv() or "(passthrough)",
        family.value,
        upstream.model,
        store_mode.value,
    )
    try:
        web.run_app(app, host=args.host, port=args.port, print=None)
    except KeyboardInterrupt:  # pragma: no cover - manual interrupt
        _LOG.info("interrupted; shutting down")
    return 0


_LLM_EXTRACTOR_COMPONENTS = (
    HarnessComponent.SUMMARY,
    HarnessComponent.USER_MODEL,
    HarnessComponent.EPISODIC,
)


def _require_cross_family_extractor(
    *, args: argparse.Namespace, components: ComponentSet,
) -> None:
    """Fail loudly if LLM memory components run without a cross-family extractor.

    The summary / user_model / episodic components extract memory with an
    LLM. If left to fall back to the substrate upstream (same Qwen family),
    the harness effectively gets same-family "crib notes", which biases the
    same-substrate ablation. Require an explicit ``--summary-extractor-model``
    + ``--summary-extractor-base-url`` (a different family from the substrate)
    whenever these components are active. Stub mode is exempt (no LLM at all),
    and ``REFH_ALLOW_SAME_FAMILY_EXTRACTOR=1`` is a documented escape hatch.
    """
    if args.use_stub_summary_extractor:
        return
    needs_extractor = any(components.has(c) for c in _LLM_EXTRACTOR_COMPONENTS)
    if not needs_extractor:
        return
    if os.environ.get("REFH_ALLOW_SAME_FAMILY_EXTRACTOR", "").strip() in {
        "1", "true", "True", "yes",
    }:
        return
    if not args.summary_extractor_model or not args.summary_extractor_base_url:
        raise SystemExit(
            "ref-harness memory components (summary/user_model/episodic) require a "
            "cross-family extractor: set --summary-extractor-model and "
            "--summary-extractor-base-url to a NON-substrate model family. "
            "Falling back to the substrate upstream is a same-family 'crib-notes' "
            "bias. Use --use-stub-summary-extractor for offline tests, or set "
            "REFH_ALLOW_SAME_FAMILY_EXTRACTOR=1 to intentionally override."
        )


def _build_upstream_client(
    *,
    args: argparse.Namespace,
    family: UpstreamFamily,
) -> UpstreamClient:
    if family is UpstreamFamily.PASSTHROUGH:
        return EchoUpstreamClient(model=args.upstream_model or "ref-harness/echo")
    if not args.upstream_base_url or not args.upstream_model or not args.upstream_key_env:
        raise SystemExit(
            "--upstream-base-url, --upstream-model, and --upstream-key-env are "
            "required unless --upstream-family=passthrough."
        )
    api_key = _required_env(args.upstream_key_env)
    if family is UpstreamFamily.OPENAI_COMPAT:
        return OpenAICompatUpstreamClient(
            base_url=args.upstream_base_url,
            api_key=api_key,
            model=args.upstream_model,
        )
    if family is UpstreamFamily.ANTHROPIC:
        return AnthropicUpstreamClient(
            base_url=args.upstream_base_url,
            api_key=api_key,
            model=args.upstream_model,
        )
    raise SystemExit(f"unhandled upstream family: {family.value}")  # pragma: no cover


@dataclasses.dataclass(frozen=True)
class _ExtractorCall:
    """Resolved LLM extractor target shared by summary / user-model / episodic.

    ``call`` is ``None`` when the deterministic stub extractors should be used
    (``--use-stub-summary-extractor`` or a passthrough family).
    """

    model: str | None
    call: "object | None"  # Callable[[list[dict[str,str]]], Awaitable[str]] | None


def _build_extractor_call(
    *,
    args: argparse.Namespace,
    family: UpstreamFamily,
) -> _ExtractorCall:
    """Resolve the single cross-family extractor target used by all close
    components. Stub mode (no LLM) returns an empty :class:`_ExtractorCall`."""

    if args.use_stub_summary_extractor:
        return _ExtractorCall(model=None, call=None)
    extractor_family = parse_upstream_family(
        args.summary_extractor_family or family.value,
    )
    if extractor_family is UpstreamFamily.PASSTHROUGH:
        return _ExtractorCall(model=None, call=None)
    base_url = args.summary_extractor_base_url or args.upstream_base_url
    model = args.summary_extractor_model or args.upstream_model
    key_env = args.summary_extractor_key_env or args.upstream_key_env
    if not base_url or not model or not key_env:
        raise SystemExit(
            "memory extractors require base_url, model, and key_env. Either set "
            "--summary-extractor-* flags, fall back to --upstream-* flags, or "
            "set --use-stub-summary-extractor."
        )
    api_key = _required_env(key_env)
    if extractor_family is UpstreamFamily.OPENAI_COMPAT:
        client = OpenAICompatUpstreamClient(base_url=base_url, api_key=api_key, model=model)
    elif extractor_family is UpstreamFamily.ANTHROPIC:
        client = AnthropicUpstreamClient(base_url=base_url, api_key=api_key, model=model)
    else:
        raise SystemExit(
            f"unhandled extractor family: {extractor_family.value}"
        )  # pragma: no cover

    async def _call(messages: list[dict[str, str]]) -> str:
        resp = await client.chat(
            messages=messages,
            max_tokens=600,
            temperature=0.0,
            session_id=None,
            user_id=None,
        )
        return resp.text

    return _ExtractorCall(model=model, call=_call)


def _build_summary_extractor(resolved: _ExtractorCall) -> SummaryExtractor:
    if resolved.call is None or resolved.model is None:
        return StubSummaryExtractor()
    return LLMSummaryExtractor(model=resolved.model, upstream_call=resolved.call)


def _build_embedder(
    components: ComponentSet, *, args: argparse.Namespace,
) -> Embedder | None:
    if not components.has(HarnessComponent.EMBED):
        return None
    choice = getattr(args, "embedder", "bge-m3")
    if choice == "hashing":
        return HashingEmbedder()
    if choice == "bge-m3":
        return SentenceTransformerEmbedder(
            model_id="BAAI/bge-m3",
            device=getattr(args, "embedder_device", None),
        )
    raise ValueError(f"unknown --embedder {choice!r}")


def _build_user_fact_extractor(
    components: ComponentSet, resolved: _ExtractorCall,
) -> UserFactExtractor | None:
    if not components.has(HarnessComponent.USER_MODEL):
        return None
    if resolved.call is None or resolved.model is None:
        return StubUserFactExtractor()
    return LLMUserFactExtractor(model=resolved.model, upstream_call=resolved.call)


def _build_episodic_extractor(
    components: ComponentSet, resolved: _ExtractorCall,
) -> EpisodicExtractor | None:
    if not components.has(HarnessComponent.EPISODIC):
        return None
    if resolved.call is None or resolved.model is None:
        return StubEpisodicExtractor()
    return LLMEpisodicExtractor(model=resolved.model, upstream_call=resolved.call)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"required environment variable {name!r} is not set. "
            "Set it before invoking `companion-ref-harness serve`."
        )
    return value


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
