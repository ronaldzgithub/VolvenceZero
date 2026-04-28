"""``lifeform-serve`` CLI \u2014 start the HTTP service.

Default vertical is ``companion`` (loaded from ``lifeform-domain-emogpt``)
when present. Other verticals can be selected with ``--vertical NAME``;
``--list-verticals`` prints what is available in the current install.
"""

from __future__ import annotations

import argparse
import logging
import sys

from aiohttp import web

from lifeform_service.app import create_app
from lifeform_service.verticals import default_vertical_name, discover_verticals


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-serve",
        description=(
            "Start a multi-tenant HTTP service that exposes a Volvence Zero "
            "lifeform from a chosen ``lifeform-domain-*`` vertical."
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
        "--list-verticals", action="store_true",
        help="Print the verticals discovered in this install and exit.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Python logging level (default INFO).",
    )
    return parser


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
    app = create_app(
        vertical=spec,
        max_sessions=args.max_sessions,
        idle_eviction_seconds=idle,
    )
    print(
        f"[lifeform-serve] vertical={spec.name}  "
        f"temporal_bootstrap={spec.has_temporal_bootstrap}  "
        f"regime_bootstrap={spec.has_regime_bootstrap}"
    )
    print(f"[lifeform-serve] listening on http://{args.host}:{args.port}")
    web.run_app(app, host=args.host, port=args.port, print=lambda *_: None)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
