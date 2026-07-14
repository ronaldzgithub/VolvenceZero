"""``lifeform-serve`` CLI \u2014 start the HTTP service.

Default vertical is ``companion`` (loaded from ``lifeform-domain-emogpt``)
when present. Other verticals can be selected with ``--vertical NAME``;
``--list-verticals`` prints what is available in the current install.

Substrate sharing
-----------------

By default the service runs in ``--substrate-mode synthetic`` which uses
the lightweight in-process synthetic substrate \u2014 no GPU, no model
weights, fine for tests and demos. For production on one GPU server,
pass ``--substrate-mode hf-shared --substrate-model-id Qwen/...`` and
the service loads ONE Qwen model at startup and shares it across every
session. The model is eagerly loaded before the listener binds, so the
first ``POST /v1/sessions/{id}/turns`` does not pay the model-load
latency.

Concurrency model: aiohttp runs a single-threaded asyncio event loop;
``runtime.generate(...)`` is sync and blocks the loop for its duration.
Concurrent sessions therefore serialise naturally on the model. If you
ever introduce ``run_in_executor`` for parallel decoding, you MUST also
add a ``threading.Lock`` around the runtime \u2014 the current default does
not need one because there is no parallelism in the inference path.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING

from aiohttp import web

from lifeform_service.app import create_app
from lifeform_service.alpha import AlphaServiceConfig, load_alpha_users
from lifeform_service.verticals import (
    default_vertical_name,
    discover_companion_ablation_verticals,
    discover_verticals,
)

if TYPE_CHECKING:
    from volvence_zero.substrate import OpenWeightResidualRuntime


_LOG = logging.getLogger("lifeform-serve")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-serve",
        description=(
            "Start a multi-tenant HTTP service that exposes a Volvence Zero "
            "lifeform from a chosen ``lifeform-domain-*`` vertical. Default "
            "deployment uses one shared synthetic substrate; for one-GPU "
            "production servers pass --substrate-mode hf-shared."
        ),
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind host (default 127.0.0.1).",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Bind port (default 8765).",
    )
    parser.add_argument(
        "--vertical",
        default=None,
        help=(
            "Vertical name to host. Defaults to the first installed vertical. "
            "Use --list-verticals to inspect the current install."
        ),
    )
    parser.add_argument(
        "--ablation-bundle",
        action="store_true",
        help=(
            "Host the reviewed Companion Bench ablation vertical bundle in one "
            "process. Mutually exclusive with --vertical; OpenAI-compat callers "
            "select the track with X-Compat-Vertical or ?vertical=."
        ),
    )
    parser.add_argument(
        "--max-sessions", type=int, default=256,
        help="Cap on concurrently live sessions before LRU eviction (default 256).",
    )
    parser.add_argument(
        "--idle-eviction-seconds", type=float, default=1800.0,
        help=(
            "Sessions idle longer than this many seconds are auto-closed. "
            "Pass 0 (or negative) to disable idle eviction."
        ),
    )
    parser.add_argument(
        "--substrate-mode",
        choices=("synthetic", "hf-shared"),
        default="synthetic",
        help=(
            "synthetic: no GPU, no model weights (default; safe for tests). "
            "hf-shared: load ONE Qwen-style model at startup and share it "
            "across every session. Requires the [hf] extras of vz-substrate."
        ),
    )
    parser.add_argument(
        "--substrate-model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "HF model id to load when --substrate-mode=hf-shared "
            "(default Qwen/Qwen2.5-0.5B-Instruct)."
        ),
    )
    parser.add_argument(
        "--substrate-device",
        default="auto",
        help="Torch device for the shared HF runtime (auto / cpu / cuda / cuda:0 / ...).",
    )
    parser.add_argument(
        "--substrate-local-files-only",
        action="store_true",
        help="Forbid HF Hub network fetches (use only the local cache).",
    )
    parser.add_argument(
        "--list-verticals", action="store_true",
        help="Print the verticals discovered in this install and exit.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Python logging level (default INFO).",
    )
    parser.add_argument(
        "--alpha-enabled",
        action="store_true",
        help="Enable closed-alpha service mode with required user identity.",
    )
    parser.add_argument(
        "--alpha-users-file",
        default=None,
        help="JSON list or {'users': [...]} allowlist for closed-alpha users.",
    )
    parser.add_argument(
        "--memory-scope-root-dir",
        default=None,
        help="Root directory for per-user scoped memory in alpha mode.",
    )
    parser.add_argument(
        "--evidence-root-dir",
        default=None,
        help="Root directory for closed-alpha session/deletion evidence bundles.",
    )
    parser.add_argument(
        "--service-version",
        default="closed-alpha-v0",
        help="Service version returned in alpha responses.",
    )
    parser.add_argument(
        "--policy-version",
        default="alpha-policy-v0",
        help="Policy version returned in alpha responses.",
    )
    parser.add_argument(
        "--require-alpha-preflight",
        action="store_true",
        help="Run closed-alpha preflight before binding the service.",
    )
    parser.add_argument(
        "--enable-openai-compat",
        action="store_true",
        help=(
            "Mount the OpenAI Chat Completions compatible router "
            "(POST /v1/chat/completions) on this app. The existing "
            "/v1/models route is owned by lifeform-service and is "
            "unaffected by this flag. Used by external benchmark "
            "harnesses (EQ-Bench 3, EmpathyBench, OpenAI Python "
            "client) — see lifeform-openai-compat wheel + "
            "docs/external/. Requires the lifeform-openai-compat "
            "wheel to be installed. Default: off (existing "
            "/v1/sessions/{id}/turns API only)."
        ),
    )
    parser.add_argument(
        "--openai-compat-api-key-env",
        default="LIFEFORM_LOCAL_API_KEY",
        help=(
            "Environment variable containing the Bearer API key required by "
            "the OpenAI-compatible /v1/chat/completions route when "
            "--enable-openai-compat is set (default LIFEFORM_LOCAL_API_KEY)."
        ),
    )
    parser.add_argument(
        "--protocol-approved-dir",
        default=None,
        help=(
            "Directory to persist approved protocols as JSON. When "
            "set, the protocol uptake service mirrors every "
            "approved candidate to '<dir>/<protocol_id>.json' and "
            "exposes the GET/POST/DELETE /v1/protocols/library "
            "routes so the chat UI can pick which persisted "
            "protocols to activate across restarts. Default: None "
            "(library mode disabled; approved protocols evaporate "
            "on restart)."
        ),
    )
    return parser


def _maybe_build_protocol_uptake_service(args: argparse.Namespace):
    """Construct a ProtocolUptakeService when persistence is requested.

    Returns ``None`` when ``--protocol-approved-dir`` is not set — in
    that case the CLI keeps the legacy single-vertical behavior of
    not mounting the protocol routes at all (matches pre-persistence
    behavior of this CLI; the richer ``start_browser_chat_qwen``
    scripts do their own wiring).

    Returns a configured :class:`ProtocolUptakeService` with the
    persistence store wired so approved protocols are mirrored to
    disk and library routes work.
    """

    approved_dir = (args.protocol_approved_dir or "").strip() or None
    if approved_dir is None:
        return None
    from pathlib import Path

    from lifeform_service.protocol_persistence import ProtocolPersistenceStore
    from lifeform_service.protocol_uptake import (
        ProtocolUptakeConfig,
        ProtocolUptakeService,
    )

    store = ProtocolPersistenceStore(Path(approved_dir))
    service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(),
        persistence=store,
    )
    persisted = store.list_all()
    if persisted:
        _LOG.info(
            "protocol library: discovered %d persisted protocol(s) in %s "
            "(use POST /v1/protocols/library/<id>/load to activate)",
            len(persisted), approved_dir,
        )
    else:
        _LOG.info(
            "protocol library: empty (no .json files under %s)",
            approved_dir,
        )
    return service


def _build_shared_substrate(args: argparse.Namespace) -> "OpenWeightResidualRuntime | None":
    """Construct the service-wide shared substrate runtime.

    Returns ``None`` for ``--substrate-mode synthetic`` so the vertical
    factory falls back to its default per-session synthetic runtime
    (which is cheap; no shared state needed).
    """
    if args.substrate_mode == "synthetic":
        _LOG.info("substrate_mode=synthetic; sessions will use per-session synthetic runtimes")
        return None

    if args.substrate_mode == "hf-shared":
        from volvence_zero.substrate import build_transformers_runtime_with_fallback

        _LOG.info(
            "substrate_mode=hf-shared; eagerly loading model_id=%s on device=%s "
            "(local_files_only=%s)",
            args.substrate_model_id,
            args.substrate_device,
            args.substrate_local_files_only,
        )
        runtime = build_transformers_runtime_with_fallback(
            model_id=args.substrate_model_id,
            device=args.substrate_device,
            local_files_only=args.substrate_local_files_only,
            # hf-shared means "serve THIS model". A silent builtin-fallback
            # substitute would violate the substrate contract (and poison
            # same-substrate benchmark runs), so load failures must raise.
            fallback_mode="deny",
            # Sharing requires frozen weights; explicit kwargs make the
            # invariant impossible to mis-configure. ``create_app``
            # double-checks via _enforce_frozen_for_sharing.
            allow_live_substrate_mutation=False,
        )
        _LOG.info(
            "shared substrate ready: model_id=%s runtime_origin=%s",
            getattr(runtime, "model_id", "?"),
            getattr(runtime, "runtime_origin", "?"),
        )
        return runtime

    raise ValueError(f"Unknown --substrate-mode {args.substrate_mode!r}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.ablation_bundle and args.vertical:
        print("--ablation-bundle cannot be combined with --vertical", file=sys.stderr)
        return 1

    try:
        discovered = (
            discover_companion_ablation_verticals()
            if args.ablation_bundle
            else discover_verticals()
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.list_verticals:
        if not discovered:
            print("No verticals available. Install lifeform-domain-emogpt or another vertical.")
            return 1
        for name, spec in discovered.items():
            print(
                f"{name}\ttemporal_bootstrap={spec.has_temporal_bootstrap}\t"
                f"regime_bootstrap={spec.has_regime_bootstrap}"
            )
        return 0

    if not discovered:
        print(
            "No verticals available. Install lifeform-domain-emogpt or another "
            "lifeform-domain-* before running the service.",
            file=sys.stderr,
        )
        return 1

    name = "companion" if args.ablation_bundle else (args.vertical or default_vertical_name())
    if name not in discovered:
        print(
            f"Unknown vertical {name!r}. Available: {sorted(discovered.keys())}",
            file=sys.stderr,
        )
        return 1

    spec = discovered[name]
    idle = args.idle_eviction_seconds if args.idle_eviction_seconds > 0 else None
    try:
        alpha_users = load_alpha_users(args.alpha_users_file)
    except Exception as exc:
        print(f"Failed to load --alpha-users-file: {exc}", file=sys.stderr)
        return 1
    alpha_config = AlphaServiceConfig(
        enabled=args.alpha_enabled,
        memory_scope_root_dir=args.memory_scope_root_dir,
        evidence_root_dir=args.evidence_root_dir,
        service_version=args.service_version,
        policy_version=args.policy_version,
        alpha_users=alpha_users,
        # D6 (#alpha-reload): remember the source file so the running
        # service can hot-reload the allow-list (endpoint / SIGHUP).
        alpha_users_path=args.alpha_users_file,
    )
    if args.alpha_enabled and args.memory_scope_root_dir is None:
        print("--alpha-enabled requires --memory-scope-root-dir", file=sys.stderr)
        return 1
    if args.require_alpha_preflight:
        from lifeform_evolution.closed_alpha_preflight import (
            format_closed_alpha_preflight_report,
            run_closed_alpha_preflight,
        )

        preflight_root = args.evidence_root_dir or "artifacts/lifeform_service_alpha"
        report = run_closed_alpha_preflight(
            artifacts_dir=f"{preflight_root}/preflight",
            scope_root_dir=f"{preflight_root}/preflight_scope",
        )
        print(format_closed_alpha_preflight_report(report))
        if not report.passed:
            return 1

    try:
        substrate_runtime = _build_shared_substrate(args)
    except ModuleNotFoundError as exc:
        print(
            f"--substrate-mode {args.substrate_mode} requires optional deps: {exc}\n"
            f"Install with: pip install 'vz-substrate[hf]'",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to build shared substrate runtime: {exc}", file=sys.stderr)
        return 1

    protocol_uptake_service = _maybe_build_protocol_uptake_service(args)

    app_kwargs = {
        "max_sessions": args.max_sessions,
        "idle_eviction_seconds": idle,
        "substrate_runtime": substrate_runtime,
        "alpha_config": alpha_config,
        "protocol_uptake_service": protocol_uptake_service,
    }
    if args.ablation_bundle:
        app = create_app(
            verticals=discovered,
            default_vertical=name,
            **app_kwargs,
        )
    else:
        app = create_app(
            vertical=spec,
            **app_kwargs,
        )
    if args.enable_openai_compat:
        # Deferred import: keeps lifeform-openai-compat an optional dep
        # (it is in the workspace but not in lifeform-service's pyproject
        # dependencies — the inverse direction is correct).
        try:
            from lifeform_openai_compat import add_openai_routes
        except ImportError as exc:
            print(
                f"--enable-openai-compat requires the lifeform-openai-compat "
                f"wheel: {exc}\n"
                f"Install with: pip install -e packages/lifeform-openai-compat",
                file=sys.stderr,
            )
            return 1
        api_key_env = args.openai_compat_api_key_env.strip()
        if not api_key_env:
            print(
                "--enable-openai-compat requires a non-empty "
                "--openai-compat-api-key-env name",
                file=sys.stderr,
            )
            return 1
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            print(
                f"--enable-openai-compat requires env var {api_key_env} "
                "to contain the local OpenAI-compatible API key",
                file=sys.stderr,
            )
            return 1
        add_openai_routes(app, api_keys=(api_key,))
    print(
        (
            "[lifeform-serve] ablation_bundle="
            f"{','.join(discovered.keys())}  default_vertical={name}  "
            if args.ablation_bundle
            else (
                f"[lifeform-serve] vertical={spec.name}  "
                f"temporal_bootstrap={spec.has_temporal_bootstrap}  "
                f"regime_bootstrap={spec.has_regime_bootstrap}  "
            )
        )
        + f"substrate_mode={args.substrate_mode}"
        + (f"  alpha_enabled={args.alpha_enabled}" if args.alpha_enabled else "")
        + (
            f"  openai_compat=on(auth_env={args.openai_compat_api_key_env})"
            if args.enable_openai_compat
            else ""
        )
        + (
            f"  model_id={args.substrate_model_id}"
            if args.substrate_mode == "hf-shared"
            else ""
        )
    )
    print(f"[lifeform-serve] listening on http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=lambda *_: None)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
