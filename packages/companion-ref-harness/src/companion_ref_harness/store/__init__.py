# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Persistent + in-memory stores for harness component state.

Each component owns its own table (or in-memory dict) and is the
only writer for that table. The :class:`HarnessPolicy` is the only
reader that aggregates across tables — it builds typed snapshots
from each owner and never reaches into another component's raw
representation. This mirrors the SSOT / snapshot-isolation rule in
``.cursor/rules/ssot-module-boundaries.mdc``.
"""

from __future__ import annotations

from companion_ref_harness.store.sqlite_store import (
    InMemoryHarnessStore,
    SqliteHarnessStore,
    StoreMode,
    open_store,
)

__all__ = (
    "InMemoryHarnessStore",
    "SqliteHarnessStore",
    "StoreMode",
    "open_store",
)
