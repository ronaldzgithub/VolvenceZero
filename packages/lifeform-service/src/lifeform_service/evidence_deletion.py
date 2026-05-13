"""Evidence deletion policy + append-only deletion proof ledger.

Closed-alpha already deletes scoped memory through
``DELETE /v1/users/me/memory``; this module extends the surface to
``evidence_root_dir/sessions/*.json`` so a tenant admin or end user
can satisfy PIPL / GDPR removal requests without losing audit
integrity.

Design (debt #49 / cross-cutting-foundation-packet F-B):

* Deletions remove only the user's own evidence files; the deletion
  itself is recorded into ``evidence_deletion_ledger-YYYYMMDD.jsonl``
  (append-only, never deleted) so post-hoc audit can prove a
  deletion happened without disclosing the deleted content.
* The ledger entry stores: ``timestamp_iso`` / ``scope_key`` /
  ``deleted_file_count`` / ``deleted_file_sha256_set`` / ``actor`` /
  ``request_id`` / ``policy_version``.
* SHADOW: HTTP endpoint scaffolding lives in
  :mod:`lifeform_service.protocol_routes`; this module owns the
  policy + ledger writer only.

This is a SHADOW scaffold: real wire-up to existing
``DELETE /v1/users/me/memory`` flow lands as F-B subtask 4 (see
``docs/specs/evidence-deletion-protocol.md``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from volvence_zero.memory import EndUserIdentity, TenantIdentity


@dataclasses.dataclass(frozen=True)
class EvidenceDeletionPolicy:
    """How long to keep evidence + whether end-user deletion is allowed.

    * ``retention_days`` — hard maximum age (any older auto-purged by
      a separate sweeper, not by this module).
    * ``delete_on_user_request`` — must be True for PIPL / GDPR.
    * ``retain_deletion_proof`` — when True, deletion writes a
      placeholder receipt entry so audit chain has a hole-free trail.
    """

    retention_days: int = 365
    delete_on_user_request: bool = True
    retain_deletion_proof: bool = True

    def __post_init__(self) -> None:
        if self.retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        if not self.delete_on_user_request:
            # Closed-alpha disclaimer + commercialisation §8.1.4 require this.
            raise ValueError(
                "EvidenceDeletionPolicy.delete_on_user_request must be True; "
                "PIPL / GDPR forbid disabling end-user deletion."
            )


@dataclasses.dataclass(frozen=True)
class EvidenceDeletionRecord:
    """One row in the append-only deletion ledger."""

    timestamp_iso: str
    scope_key: str
    tenant_id: str | None
    end_user_id: str | None
    deleted_file_count: int
    deleted_file_sha256_set: tuple[str, ...]
    actor: str  # "end_user" / "tenant_admin" / "ops"
    request_id: str
    policy_version: str

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


def _ledger_path_for_today(evidence_root: Path) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return evidence_root / f"evidence_deletion_ledger-{today}.jsonl"


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delete_evidence_files_for_scope(
    *,
    evidence_root: Path,
    scope_key: str,
    actor: str,
    request_id: str,
    policy: EvidenceDeletionPolicy,
    tenant_identity: TenantIdentity | None = None,
    end_user_identity: EndUserIdentity | None = None,
    file_filter: Iterable[Path] | None = None,
) -> EvidenceDeletionRecord:
    """Delete evidence files for a scope and write the ledger entry.

    SHADOW behaviour: when ``file_filter`` is None, this scaffold
    walks ``evidence_root/sessions/`` for files whose name contains
    the scope key fragment. Real deletion (with proper file owner
    semantics + cross-restart audit) lands as F-B subtask 4.
    """

    if not policy.delete_on_user_request:  # defensive; __post_init__ already enforces
        raise PermissionError("policy disallows end-user deletion")

    sessions_dir = Path(evidence_root) / "sessions"
    targets: list[Path] = []
    if file_filter is not None:
        targets = [Path(p) for p in file_filter]
    elif sessions_dir.exists():
        targets = [
            p
            for p in sessions_dir.iterdir()
            if p.is_file() and scope_key.replace(":", "_") in p.name
        ]

    sha256_set: list[str] = []
    deleted = 0
    for path in targets:
        if not path.exists():
            continue
        sha256_set.append(_sha256_of_file(path))
        path.unlink()
        deleted += 1

    record = EvidenceDeletionRecord(
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
        scope_key=scope_key,
        tenant_id=tenant_identity.tenant_id if tenant_identity else None,
        end_user_id=end_user_identity.end_user_id if end_user_identity else None,
        deleted_file_count=deleted,
        deleted_file_sha256_set=tuple(sha256_set),
        actor=actor,
        request_id=request_id,
        policy_version="evidence-deletion-v0",
    )

    if policy.retain_deletion_proof:
        evidence_root_path = Path(evidence_root)
        evidence_root_path.mkdir(parents=True, exist_ok=True)
        ledger_path = _ledger_path_for_today(evidence_root_path)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json_line() + "\n")

    return record


__all__ = (
    "EvidenceDeletionPolicy",
    "EvidenceDeletionRecord",
    "delete_evidence_files_for_scope",
)
