"""Mutable accumulator for LLM proposal runtime diagnostic counters.

Pairs with the immutable :class:`LLMProposalAttemptCounters` typed
contract in ``vz-contracts``. Lives in ``semantic_state`` because that
sub-package is the foundational layer for proposal-runtime
infrastructure; ``social`` imports it (matching the existing
``from volvence_zero.semantic_state import ...`` pattern in
``social/tom.py``), so the dependency direction stays one-way.

Why not just put a mutable dataclass in ``vz-contracts``:

``vz-contracts`` is the SSOT for IMMUTABLE typed shapes that cross
wheel boundaries. Mutable runtime accumulators have nothing to do
with cross-wheel contracts; they are private state of one runtime
instance. The accumulator here is what the runtime hot path mutates;
the runtime exposes ``snapshot()`` to produce a frozen
:class:`LLMProposalAttemptCounters` for owner snapshot publication.

Parse-status vocabulary:

The three terminal statuses (``"ok"`` / ``"parse_error"`` /
``"empty_or_rejected"``) match the strings produced by the
``_parse_*_with_diag`` helpers in
``vz-cognition.social._llm_debug``. We accept the strings as
parameters so this accumulator does not need to import a parse-state
enum from ``social``; the tightening would be one-directional and
add no safety.
"""

from __future__ import annotations

from volvence_zero.llm_proposal_diagnostics import LLMProposalAttemptCounters


class LLMProposalAttemptAccumulator:
    """Mutable per-runtime counter state for LLM proposal attempts.

    Contract:

    * ``record_attempt(parse_status, parse_error, parsed_count,
      emitted_count)`` is the only entry point that mutates state.
      Callers must pass ``parse_status`` from the documented enum
      (``"ok"`` / ``"parse_error"`` / ``"empty_or_rejected"``); any
      other value fails loudly so a typo in the runtime cannot
      silently produce malformed counters.
    * ``snapshot()`` returns an immutable
      :class:`LLMProposalAttemptCounters` reflecting current state.
      It does NOT reset state.
    * Bootstrap state (no attempts yet) reports
      ``last_parse_status="no_call"`` to distinguish "ran 0 times"
      from "ran once and parsed ok with 0 typed proposals".

    This class is single-threaded by design (matches the runtime
    propose path which is sequential per session).
    """

    __slots__ = (
        "_received_total",
        "_parsed_ok",
        "_rejected_malformed_json",
        "_rejected_schema_mismatch",
        "_emitted_total",
        "_last_parse_status",
        "_last_parse_error",
    )

    def __init__(self) -> None:
        self._received_total = 0
        self._parsed_ok = 0
        self._rejected_malformed_json = 0
        self._rejected_schema_mismatch = 0
        self._emitted_total = 0
        self._last_parse_status = "no_call"
        self._last_parse_error = ""

    def record_attempt(
        self,
        *,
        parse_status: str,
        parse_error: str | None,
        parsed_count: int,
        emitted_count: int,
    ) -> None:
        if parse_status not in {"ok", "parse_error", "empty_or_rejected"}:
            raise ValueError(
                f"record_attempt: parse_status={parse_status!r} not in "
                "{'ok', 'parse_error', 'empty_or_rejected'}."
            )
        if parsed_count < 0:
            raise ValueError(
                f"record_attempt: parsed_count={parsed_count!r} must be >= 0."
            )
        if emitted_count < 0:
            raise ValueError(
                f"record_attempt: emitted_count={emitted_count!r} must be >= 0."
            )
        if emitted_count > parsed_count:
            raise ValueError(
                f"record_attempt: emitted_count={emitted_count!r} cannot "
                f"exceed parsed_count={parsed_count!r} for a single "
                "call (owner-side filtering only shrinks)."
            )
        self._received_total += 1
        if parse_status == "ok":
            self._parsed_ok += 1
        elif parse_status == "parse_error":
            self._rejected_malformed_json += 1
        else:
            self._rejected_schema_mismatch += 1
        self._emitted_total += emitted_count
        self._last_parse_status = parse_status
        self._last_parse_error = (parse_error or "")[:240]

    def snapshot(self) -> LLMProposalAttemptCounters:
        return LLMProposalAttemptCounters(
            proposals_received_total=self._received_total,
            proposals_parsed_ok=self._parsed_ok,
            proposals_rejected_malformed_json=self._rejected_malformed_json,
            proposals_rejected_schema_mismatch=self._rejected_schema_mismatch,
            proposals_emitted_total=self._emitted_total,
            last_parse_status=self._last_parse_status,
            last_parse_error=self._last_parse_error,
        )


__all__ = [
    "LLMProposalAttemptAccumulator",
]
