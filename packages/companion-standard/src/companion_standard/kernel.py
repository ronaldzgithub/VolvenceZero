# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Snapshot — the immutable cross-module exchange container.

Part of the Relationship Representation Standard: a snapshot is the ONLY
sanctioned unit of data exchange between modules in a standard-conformant
runtime. It binds a slot name to an owner, a monotonically increasing
version, a wall-clock timestamp, and an immutable value.

Standard semantics (normative):

* One slot has exactly one owner; nobody else publishes to it.
* Snapshots are immutable after publication; consumers never mutate a
  received snapshot or its value.
* Consumers read published snapshot values; they do not reconstruct a
  producer's internal state from raw fields.

The propagation *mechanism* — dependency ordering, wiring levels, guard
enforcement, placeholder publication — is runtime implementation and
deliberately does not ship in the standard; any conformant runtime
provides its own.

Extracted from the upstream production runtime's kernel module (Phase A1
of the standard split); the runtime keeps its original import paths by
re-exporting from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class Snapshot(Generic[ValueT]):
    slot_name: str
    owner: str
    version: int
    timestamp_ms: int
    value: ValueT


__all__ = [
    "Snapshot",
]
