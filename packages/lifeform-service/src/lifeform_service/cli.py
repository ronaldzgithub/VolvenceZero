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
import sys
from typing import TYPE_CHECKING

from aiohttp import web

from lifeform_service.app import create_app
from lifeform_service.alpha import AlphaServiceConfig, load_alpha_users
from lifeform_service.verticals import default_vertical_name, discover_verticals

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
    return parser


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

    discovered = discover_verticals()
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

    name = args.vertical or default_vertical_name()
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

    app = create_app(
        vertical=spec,
        max_sessions=args.max_sessions,
        idle_eviction_seconds=idle,
        substrate_runtime=substrate_runtime,
        alpha_config=alpha_config,
    )
    print(
        f"[lifeform-serve] vertical={spec.name}  "
        f"temporal_bootstrap={spec.has_temporal_bootstrap}  "
        f"regime_bootstrap={spec.has_regime_bootstrap}  "
        f"substrate_mode={args.substrate_mode}"
        + (f"  alpha_enabled={args.alpha_enabled}" if args.alpha_enabled else "")
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
