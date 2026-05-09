"""Diagnostic counters for LLM-backed semantic / social proposal runtimes.

These counters live in ``vz-contracts`` because three runtimes in two
different wheels publish them and three owner snapshot types surface
them. Sharing the dataclass keeps R8 satisfied (one schema, one owner
per snapshot, immutable typed field) without forcing each consumer
to re-derive what "an LLM call attempt" means.

When does a counter advance:

* ``proposals_received_total`` advances on every ``runtime.propose(...)``
  call where the runtime actually invokes its provider (calls that
  short-circuit because the slot is out-of-scope or ``user_input`` is
  empty are NOT counted; those are not real "LLM attempts").
* ``proposals_parsed_ok`` advances when the strict parser returned at
  least one typed proposal (``parse_status == "ok"``).
* ``proposals_rejected_malformed_json`` advances when ``json.loads``
  raised or top-level shape was wrong (``parse_status == "parse_error"``).
* ``proposals_rejected_schema_mismatch`` advances when JSON parsed
  but no item survived schema / confidence validation
  (``parse_status == "empty_or_rejected"``).
* ``proposals_emitted_total`` advances by the number of typed
  proposals that survived schema validation AND owner-side confidence
  filtering. Owner-side filter (e.g. ToM
  ``min_proposal_confidence=0.50``) is applied in the owner module
  not in the runtime, so this can be lower than ``proposals_parsed_ok``.

Why not aggregate across runtimes:

The owner is the unit of evidence. Four ToM owners + one common-ground
owner each have their own runtime instance in default wiring, and a
0-records evidence run needs per-owner blame to be diagnosable. A
single global aggregator would obscure which runtime actually saw the
provider call.

Backward compatibility:

* ``LLMProposalAttemptCounters`` is a new typed shape; consumers that
  did not previously read it are unaffected.
* All snapshots that surface this field default it to ``None`` so a
  runtime without an LLM provider (NoOp or scripted) publishes
  ``proposal_diagnostics=None`` rather than a zero-valued counter —
  zero counters are reserved to mean "the runtime ran but emitted
  nothing", which is a real evidence signal.
"""

from __future__ import annotations

from dataclasses import dataclass


_VALID_PARSE_STATUS: frozenset[str] = frozenset(
    {"ok", "parse_error", "empty_or_rejected", "no_call"}
)


@dataclass(frozen=True)
class LLMProposalAttemptCounters:
    """Immutable typed snapshot of one LLM proposal runtime's call counters.

    Each field is cumulative for the lifetime of the runtime instance.
    The counters reset only when a new runtime is constructed (i.e. on
    process restart or explicit re-wiring). For per-turn deltas,
    consumers must subtract two consecutive snapshots themselves.

    Counter semantics:

    * ``proposals_received_total`` / ``proposals_parsed_ok`` /
      ``proposals_rejected_*`` are **call counts** — they advance by
      one per LLM call.
    * ``proposals_emitted_total`` is a **sum of items** — a single
      ToM call can emit four typed proposals (one per slot), so this
      counter can legitimately exceed ``proposals_parsed_ok`` even
      though every emitted item came from a parsed_ok call.

    Invariants enforced in ``__post_init__``:

    * All counters are non-negative ints.
    * ``proposals_received_total`` is at least the sum of the three
      reject categories plus ``proposals_parsed_ok`` (every received
      attempt has exactly one terminal status, with ``no_call`` as
      the bootstrap default before any attempt).
    * ``proposals_emitted_total > 0`` only when
      ``proposals_parsed_ok > 0`` (owner-side filtering may shrink
      but the runtime never invents typed items out of a parse
      failure).
    * ``last_parse_status`` is one of the documented enum values.
    """

    proposals_received_total: int
    proposals_parsed_ok: int
    proposals_rejected_malformed_json: int
    proposals_rejected_schema_mismatch: int
    proposals_emitted_total: int
    last_parse_status: str
    last_parse_error: str

    def __post_init__(self) -> None:
        for field_name in (
            "proposals_received_total",
            "proposals_parsed_ok",
            "proposals_rejected_malformed_json",
            "proposals_rejected_schema_mismatch",
            "proposals_emitted_total",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be int, got {type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0, got {value!r}")
        terminal_sum = (
            self.proposals_parsed_ok
            + self.proposals_rejected_malformed_json
            + self.proposals_rejected_schema_mismatch
        )
        if terminal_sum > self.proposals_received_total:
            raise ValueError(
                "Sum of terminal-status counters "
                f"(parsed_ok={self.proposals_parsed_ok}, "
                f"malformed_json={self.proposals_rejected_malformed_json}, "
                f"schema_mismatch={self.proposals_rejected_schema_mismatch}) "
                f"exceeds proposals_received_total={self.proposals_received_total}."
            )
        if self.proposals_emitted_total > 0 and self.proposals_parsed_ok == 0:
            raise ValueError(
                "proposals_emitted_total "
                f"({self.proposals_emitted_total}) > 0 but "
                "proposals_parsed_ok=0; runtime cannot emit typed "
                "proposals from a parse failure."
            )
        if self.last_parse_status not in _VALID_PARSE_STATUS:
            raise ValueError(
                f"last_parse_status={self.last_parse_status!r} not in "
                f"{sorted(_VALID_PARSE_STATUS)}."
            )

    @classmethod
    def empty(cls) -> "LLMProposalAttemptCounters":
        """Return the bootstrap counter (no runtime calls yet)."""
        return cls(
            proposals_received_total=0,
            proposals_parsed_ok=0,
            proposals_rejected_malformed_json=0,
            proposals_rejected_schema_mismatch=0,
            proposals_emitted_total=0,
            last_parse_status="no_call",
            last_parse_error="",
        )


__all__ = [
    "LLMProposalAttemptCounters",
]
