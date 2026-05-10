"""Append-only persistent ledger of verification checks.

Layout under the configured ``root`` directory::

    root/
      verification/
        {source_byte_sha256}/
          checks.jsonl       # one VerificationCheck per line, append-only

Why this shape:

* Append-only means every verifier run / human review is preserved.
  A "latest wins" reduce (:meth:`latest_per_kind`) yields the
  effective verdict per check kind without losing history.
* Anchored by ``source_byte_sha256`` (the same key as L1
  :class:`RawDocument.raw_sha256` and
  :class:`SourceProvenance.byte_sha256`) so the verification record
  is content-addressable and trivially co-locatable with the
  cleaning store entries on disk.
* JSONL keeps each line self-describing and human-inspectable; no
  binary serialisation, no schema-coupled migration headaches.

The ledger never deletes or rewrites prior entries; a curator
overrides an auto verdict by appending a new ``human:...`` check
which wins under :meth:`latest_per_kind`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from lifeform_domain_figure.verification.records import (
    CheckKind,
    Verdict,
    VerificationCheck,
)


class VerificationLedger:
    """Filesystem-backed append-only ledger of :class:`VerificationCheck`."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError(
                f"VerificationLedger.root must be a Path; got {type(root).__name__}"
            )
        self._root = root
        self._verification_root = root / "verification"

    @property
    def root(self) -> Path:
        return self._root

    def append(self, check: VerificationCheck) -> Path:
        """Append ``check`` to its anchor's JSONL file. Return the file path."""

        if not isinstance(check, VerificationCheck):
            raise TypeError(
                f"VerificationLedger.append: check must be VerificationCheck; "
                f"got {type(check).__name__}"
            )
        anchor_dir = self._verification_root / check.source_byte_sha256
        anchor_dir.mkdir(parents=True, exist_ok=True)
        path = anchor_dir / "checks.jsonl"
        line = json.dumps(
            {
                "check_kind": check.check_kind.value,
                "verdict": check.verdict.value,
                "evidence": list(check.evidence),
                "reviewer_id": check.reviewer_id,
                "reviewed_at_iso": check.reviewed_at_iso,
                "source_byte_sha256": check.source_byte_sha256,
            },
            ensure_ascii=False,
        )
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return path

    def get_checks(self, source_byte_sha256: str) -> tuple[VerificationCheck, ...]:
        """Read every check for ``source_byte_sha256`` in append order.

        Returns an empty tuple when the anchor has no ledger entries.
        Order preserves the on-disk JSONL order (== append order, ==
        chronological for a sane filesystem); callers needing
        per-kind reduction should use :meth:`latest_per_kind`.
        """

        path = self._verification_root / source_byte_sha256 / "checks.jsonl"
        if not path.exists():
            return ()
        records: list[VerificationCheck] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                records.append(
                    VerificationCheck(
                        check_kind=CheckKind(payload["check_kind"]),
                        verdict=Verdict(payload["verdict"]),
                        evidence=tuple(str(item) for item in payload["evidence"]),
                        reviewer_id=str(payload["reviewer_id"]),
                        reviewed_at_iso=str(payload["reviewed_at_iso"]),
                        source_byte_sha256=str(payload["source_byte_sha256"]),
                    )
                )
        return tuple(records)

    def latest_per_kind(
        self, source_byte_sha256: str
    ) -> dict[CheckKind, VerificationCheck]:
        """Return the most-recent check per :class:`CheckKind` for the anchor.

        "Most recent" = last append-order occurrence (which is also
        last in the JSONL file). A human override appended after an
        auto check therefore wins; this is the sanctioned audit
        pattern (auto verifiers run frequently; humans adjudicate
        ``NEEDS_REVIEW`` and overrule false positives).
        """

        latest: dict[CheckKind, VerificationCheck] = {}
        for record in self.get_checks(source_byte_sha256):
            latest[record.check_kind] = record
        return latest

    def list_anchors(self) -> Iterator[str]:
        """Yield every anchor sha256 with at least one ledger entry, sorted."""

        if not self._verification_root.exists():
            return
        for entry in sorted(self._verification_root.iterdir()):
            if entry.is_dir() and (entry / "checks.jsonl").exists():
                yield entry.name


__all__ = ["VerificationLedger"]
