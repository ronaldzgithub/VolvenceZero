"""Bundle-build gate: refuse to admit non-verified sources.

The gate is invoked by
:func:`lifeform_domain_figure.compiler.build_figure_artifact_bundle`
when the caller passes ``require_verification_pass=True``. It walks
every supplied :class:`SourceProvenance` and consults the supplied
:class:`VerificationLedger`; if any source's currently-effective
verdicts (per :meth:`VerificationLedger.latest_per_kind`) are not
all ``PASS`` for the *implemented* check kinds, the gate raises a
:class:`VerificationGateError` listing every failing source and the
specific reasons.

What "implemented" means
------------------------

The gate enforces all-PASS only for the subset listed in
:data:`IMPLEMENTED_CHECK_KINDS`. Deferred kinds (the 4 metadata-
dependent verifiers) are intentionally not blocking yet — the spec
acknowledges this is a phased rollout, and the contract test
``tests/contracts/test_bundle_admits_only_verified_sources.py``
fences the assumption: when a follow-up packet implements one of the
deferred kinds, it adds it to ``IMPLEMENTED_CHECK_KINDS`` and the
gate immediately starts requiring it. Any in-flight bundle missing
that coverage flips from PASS to FAIL on the next build, surfacing
the gap loudly.

What about ungated sources?
---------------------------

The gate operates on the supplied ``provenance_records`` tuple. A
curator declaring ``require_verification_pass=True`` is asserting
"every source I shipped is in this tuple". Sources without a
provenance record are out of scope of the gate (and out of scope of
audit); the curation flow refuses to ship those by attaching a
provenance to every typed source upstream. This is the same
discipline the existing
:func:`lifeform_domain_figure.corpus.provenance.fingerprint_provenance`
helper already assumes.
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_domain_figure.corpus.provenance import SourceProvenance
from lifeform_domain_figure.verification.ledger import VerificationLedger
from lifeform_domain_figure.verification.records import (
    CheckKind,
    IMPLEMENTED_CHECK_KINDS,
    Verdict,
    VerificationCheck,
)


class VerificationGateError(ValueError):
    """Raised when one or more sources fail the verification gate."""


@dataclass(frozen=True)
class _GateFailure:
    source_id: str
    byte_sha256: str
    check_kind: CheckKind
    reason: str


def _failures_for_provenance(
    provenance: SourceProvenance,
    ledger: VerificationLedger,
) -> tuple[_GateFailure, ...]:
    failures: list[_GateFailure] = []
    latest = ledger.latest_per_kind(provenance.byte_sha256)
    for kind in sorted(IMPLEMENTED_CHECK_KINDS, key=lambda k: k.value):
        check: VerificationCheck | None = latest.get(kind)
        if check is None:
            failures.append(
                _GateFailure(
                    source_id=provenance.source_id,
                    byte_sha256=provenance.byte_sha256,
                    check_kind=kind,
                    reason="missing-check (no entry in ledger)",
                )
            )
            continue
        if check.verdict is not Verdict.PASS:
            failures.append(
                _GateFailure(
                    source_id=provenance.source_id,
                    byte_sha256=provenance.byte_sha256,
                    check_kind=kind,
                    reason=(
                        f"verdict={check.verdict.value} "
                        f"reviewer_id={check.reviewer_id}"
                    ),
                )
            )
    return tuple(failures)


def assert_all_provenances_pass(
    provenances: tuple[SourceProvenance, ...],
    ledger: VerificationLedger,
) -> None:
    """Raise :class:`VerificationGateError` unless every source is all-PASS.

    All-PASS = for every :class:`CheckKind` in
    :data:`IMPLEMENTED_CHECK_KINDS`, the latest ledger check has
    ``verdict == Verdict.PASS``. Deferred kinds are not enforced yet
    (see module docstring).
    """

    if not isinstance(provenances, tuple):
        raise TypeError(
            f"assert_all_provenances_pass: provenances must be a tuple; "
            f"got {type(provenances).__name__}"
        )
    if not isinstance(ledger, VerificationLedger):
        raise TypeError(
            f"assert_all_provenances_pass: ledger must be a VerificationLedger; "
            f"got {type(ledger).__name__}"
        )
    if not provenances:
        raise VerificationGateError(
            "assert_all_provenances_pass: provenances is empty; "
            "require_verification_pass=True needs at least one source to gate"
        )
    all_failures: list[_GateFailure] = []
    for provenance in provenances:
        all_failures.extend(_failures_for_provenance(provenance, ledger))
    if all_failures:
        lines = [
            f"  - source_id={f.source_id} byte_sha256={f.byte_sha256[:12]}... "
            f"check_kind={f.check_kind.value} {f.reason}"
            for f in all_failures
        ]
        raise VerificationGateError(
            "verification gate refused bundle build; "
            f"{len(all_failures)} failure(s) across {len(provenances)} source(s):\n"
            + "\n".join(lines)
        )


__all__ = [
    "VerificationGateError",
    "assert_all_provenances_pass",
]
