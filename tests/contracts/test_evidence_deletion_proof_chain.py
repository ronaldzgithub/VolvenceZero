"""Contract tests for evidence deletion proof chain (debt #49).

Validates the SHADOW surface land in
:mod:`lifeform_service.evidence_deletion`:

1. ``EvidenceDeletionPolicy`` rejects ``delete_on_user_request=False``
   (PIPL / GDPR forbid disabling end-user deletion)
2. After deletion, the ledger entry exists and contains
   sha256 of deleted files (not their content)
3. Ledger is append-only across multiple deletes
4. Ledger schema is stable and policy_version pinned

See:

* ``docs/specs/evidence-deletion-protocol.md``
* ``docs/moving forward/cross-cutting-foundation-packet.md`` §2.2
* ``docs/known-debts.md`` #49
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifeform_service.evidence_deletion import (
    EvidenceDeletionPolicy,
    delete_evidence_files_for_scope,
)


def test_policy_rejects_disabling_user_deletion() -> None:
    with pytest.raises(ValueError, match="PIPL / GDPR"):
        EvidenceDeletionPolicy(delete_on_user_request=False)


def test_policy_accepts_default() -> None:
    policy = EvidenceDeletionPolicy()
    assert policy.retention_days == 365
    assert policy.delete_on_user_request is True
    assert policy.retain_deletion_proof is True


def test_policy_rejects_zero_retention() -> None:
    with pytest.raises(ValueError, match="retention_days must be > 0"):
        EvidenceDeletionPolicy(retention_days=0)


def test_deletion_writes_ledger_entry(tmp_path: Path) -> None:
    """Deletion records sha256 + count + scope, never content."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    file_a = sessions / "session-brand_a_alice-001.json"
    file_a.write_text(json.dumps({"secret": "hello"}), encoding="utf-8")
    file_b = sessions / "session-brand_a_alice-002.json"
    file_b.write_text(json.dumps({"secret": "world"}), encoding="utf-8")

    record = delete_evidence_files_for_scope(
        evidence_root=tmp_path,
        scope_key="brand_a:alice",
        actor="end_user",
        request_id="req-001",
        policy=EvidenceDeletionPolicy(),
    )

    assert record.deleted_file_count == 2
    assert len(record.deleted_file_sha256_set) == 2
    assert record.scope_key == "brand_a:alice"
    assert record.actor == "end_user"
    assert record.policy_version == "evidence-deletion-v0"
    # Files actually gone
    assert not file_a.exists()
    assert not file_b.exists()
    # Ledger written
    ledger_files = list(tmp_path.glob("evidence_deletion_ledger-*.jsonl"))
    assert len(ledger_files) == 1
    ledger_lines = ledger_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(ledger_lines) == 1
    payload = json.loads(ledger_lines[0])
    # Must include sha256 set, not content
    assert "deleted_file_sha256_set" in payload
    assert "secret" not in ledger_lines[0]  # no content leak
    assert "hello" not in ledger_lines[0]
    assert "world" not in ledger_lines[0]


def test_deletion_ledger_is_append_only_across_calls(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "session-brand_a_alice-001.json").write_text("{}", encoding="utf-8")
    (sessions / "session-brand_a_bob-002.json").write_text("{}", encoding="utf-8")

    delete_evidence_files_for_scope(
        evidence_root=tmp_path,
        scope_key="brand_a:alice",
        actor="end_user",
        request_id="req-001",
        policy=EvidenceDeletionPolicy(),
    )
    delete_evidence_files_for_scope(
        evidence_root=tmp_path,
        scope_key="brand_a:bob",
        actor="end_user",
        request_id="req-002",
        policy=EvidenceDeletionPolicy(),
    )

    ledger_files = list(tmp_path.glob("evidence_deletion_ledger-*.jsonl"))
    assert len(ledger_files) == 1
    lines = ledger_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    request_ids = {json.loads(line)["request_id"] for line in lines}
    assert request_ids == {"req-001", "req-002"}


def test_deletion_record_schema_stability() -> None:
    """Ledger entry has the documented field set (no drift)."""
    expected_fields = {
        "timestamp_iso",
        "scope_key",
        "tenant_id",
        "end_user_id",
        "deleted_file_count",
        "deleted_file_sha256_set",
        "actor",
        "request_id",
        "policy_version",
    }
    from lifeform_service.evidence_deletion import EvidenceDeletionRecord
    import dataclasses

    actual_fields = {f.name for f in dataclasses.fields(EvidenceDeletionRecord)}
    assert actual_fields == expected_fields, (
        f"EvidenceDeletionRecord fields drifted: "
        f"unexpected={actual_fields - expected_fields}, "
        f"missing={expected_fields - actual_fields}"
    )
