"""Grounded-decode hook protocol — substrate-side additive interface.

This module declares a tiny Protocol that the substrate runtime can
optionally call to verify that a generated text has citation-quality
support before being returned to the caller. The actual verification
logic lives in the lifeform-expression layer (see
``lifeform_expression.grounded_decoder.GroundedDecoder``); this
module only declares the **shape of the contract** so substrate-side
code can hold a reference without taking any policy stake.

Why the Protocol lives here:

* The substrate runtime is the layer that produces text, so when a
  future packet wants to short-circuit generation (cite-or-stop)
  rather than verify post-hoc, the hook needs to be reachable from
  inside ``vz-substrate``.
* Putting the Protocol here lets the substrate's pyproject not
  declare a ``lifeform-expression`` dependency to satisfy a type
  reference — that dependency direction is forbidden by R8.

This module is **interface only**. It contains:

* :class:`GroundingVerdict` — frozen result dataclass.
* :class:`GroundedDecodeHook` — Protocol with one method.

It does NOT import any lifeform module. CI in
``tests/contracts/test_import_boundaries.py`` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GroundingVerdict:
    """Result of a grounded-decode verification pass.

    A verdict is the load-bearing signal for the L3 enforcement
    contract:

    * ``passed``                  — every substantive assertion in the
                                    generated text was supported by
                                    at least one piece of citation-
                                    quality evidence.
    * ``unsupported_assertions``  — the assertions for which no
                                    supporting evidence was found.
                                    Empty when ``passed`` is True.
    * ``evidence_pointers``       — citation strings (one or more per
                                    supported assertion). The expression
                                    layer surfaces these to the caller
                                    or weaves them into the response.
    * ``rationale``               — short human-readable description
                                    of the decision; included so the
                                    audit log can record why a verdict
                                    was issued without re-running the
                                    decoder.
    """

    passed: bool
    unsupported_assertions: tuple[str, ...]
    evidence_pointers: tuple[str, ...]
    rationale: str


@runtime_checkable
class GroundedDecodeHook(Protocol):
    """Substrate-side Protocol for grounded-decode verification.

    Implementations live in the lifeform layer (e.g.,
    ``lifeform_expression.grounded_decoder.GroundedDecoder``). The
    substrate runtime accepts an instance via dependency injection
    and calls :meth:`verify` after producing text but before
    returning it to the caller.

    Implementations MUST raise an exception or return a verdict with
    ``passed=False``; they MUST NOT silently treat unsupported
    assertions as supported (no-swallow-errors invariant).
    """

    def verify(self, *, text: str) -> GroundingVerdict: ...


__all__ = [
    "GroundedDecodeHook",
    "GroundingVerdict",
]
