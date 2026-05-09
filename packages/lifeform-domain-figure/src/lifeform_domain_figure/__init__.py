"""Vertical: real-person digital revival from primary-source corpora.

This package is the monorepo-local application layer for "historical /
real-person figure to lifeform" work. It stands parallel to
``lifeform-domain-character`` (fictional characters) but enforces a
strictly different set of invariants:

* The source corpus is **evidence**, not narrative truth. Every claim
  the resulting lifeform produces must be traceable to a corpus
  citation (L3 grounding contract).
* Coverage is finite and known. Out-of-corpus topics must be refused
  or disclaimed (L4 not-known refusal contract).
* Style fidelity is statistical: tone / lexicon / sentence shape
  (L1 style prior contract).
* Stance fidelity is contrastive: the figure's documented positions
  versus contemporary opponents (L2 steering contract).

The wheel does NOT add a new kernel owner. Reviewed structured
artifacts compile into existing Volvence Zero owners (domain
knowledge, case memory, strategy playbook, boundary policy) just as
``lifeform-domain-character`` does, and additionally produce an
immutable :class:`FigureArtifactBundle` consumed at runtime by the
``lifeform-expression`` enforcement layer.

Public surface is added incrementally per the F1-F6 packet sequence
(see ``docs/specs/figure-vertical.md``). Early imports are
intentionally scoped: each packet adds its own modules and re-exports
here so consumers can pin against a stable namespace.
"""

from __future__ import annotations


__all__: tuple[str, ...] = ()
