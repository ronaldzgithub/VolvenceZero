"""Service layer: external surfaces.

Future home of:
- EmoGPT ``interface/*`` (HTTP / WebSocket / CLI servers)
- EmoGPT ``tenant/*`` (multi-tenant isolation, SQLite stores)
- EmoGPT ``persistence/*`` runtime/tenancy databases
- Operational scripts (deployment, db migrations) and supervisor configs

These are kept out of ``lifeform-core`` so that running the brain
in-process (e.g. for tests, benchmarks, or notebook experiments) does
NOT pull in HTTP/DB dependencies.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
