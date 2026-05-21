"""Tests for :class:`ProtocolPersistenceStore`.

Covers the disk-side contract:

* ``write`` is atomic (no partial files survive a crash mid-write).
* ``read`` round-trips identically (eq-based assertion).
* ``list_all`` discovers every well-formed JSON and skips bad files
  without failing the whole load.
* ``delete`` cleans up; ``exists`` reflects state.
* Filename sanitisation tolerates namespaced ``protocol_id`` like
  ``growth_advisor:cheng-laoshi`` on case-sensitive filesystems.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifeform_protocol_runtime import protocol_to_payload
from lifeform_service.protocol_persistence import (
    ProtocolPersistenceStore,
    _filename_for_protocol_id,
)
from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    FailureSignal,
    IdentityAssertion,
    ProtocolSourceKind,
    ReviewStatus,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
)


def _make_protocol(
    *,
    pid: str = "test:store-one",
    advisor: str = "Store Advisor",
) -> BehaviorProtocol:
    return BehaviorProtocol(
        protocol_id=pid,
        version="1.0.0",
        advisor_name=advisor,
        description="store test",
        source_kind=ProtocolSourceKind.PDF_UPTAKE,
        source_locator=f"uploads/{pid}.pdf",
        identity_assertion=IdentityAssertion(),
        boundary_contracts=(
            BoundaryContract(
                boundary_id="bd:one",
                description="d",
                trigger_reasons=("t",),
                severity=BoundarySeverity.SOFT_REMIND,
            ),
        ),
        activation_conditions=ActivationConditions(),
        strategy_priors=(
            StrategyPrior(
                rule_id="rule:one",
                problem_pattern="p",
                recommended_ordering=("s",),
                recommended_pacing="moderate",
            ),
        ),
        temporal_arc=TemporalArc(),
        success_signals=(
            SuccessSignal(
                signal_id="ss:one",
                description="ok",
                measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
            ),
        ),
        failure_signals=(
            FailureSignal(
                signal_id="fs:one",
                description="bad",
                measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            ),
        ),
        review_status=ReviewStatus.ACTIVE,
    )


def test_filename_sanitises_namespaced_ids(tmp_path: Path) -> None:
    # Colons are valid in protocol_id but not on Windows filenames.
    assert (
        _filename_for_protocol_id("growth_advisor:cheng-laoshi")
        == "growth_advisor_cheng-laoshi"
    )
    # Path-traversal characters get scrubbed.
    assert "/" not in _filename_for_protocol_id("a/b\\c")
    assert "\\" not in _filename_for_protocol_id("a/b\\c")
    with pytest.raises(ValueError):
        _filename_for_protocol_id("   ")


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    store = ProtocolPersistenceStore(tmp_path / "lib")
    proto = _make_protocol()
    path = store.write(proto)
    assert path.is_file()
    assert store.exists(proto.protocol_id)
    restored = store.read(proto.protocol_id)
    assert restored == proto


def test_list_all_returns_sorted(tmp_path: Path) -> None:
    store = ProtocolPersistenceStore(tmp_path / "lib")
    a = _make_protocol(pid="test:aaa")
    b = _make_protocol(pid="test:bbb")
    c = _make_protocol(pid="test:ccc")
    # Write in reverse order; list_all must still sort by id.
    store.write(c)
    store.write(a)
    store.write(b)
    listed = store.list_all()
    assert [p.protocol_id for p in listed] == [
        "test:aaa",
        "test:bbb",
        "test:ccc",
    ]


def test_list_all_skips_corrupted_files(tmp_path: Path, caplog) -> None:
    """One bad JSON must not block the rest of the library from loading."""
    store = ProtocolPersistenceStore(tmp_path / "lib")
    good = _make_protocol(pid="test:good")
    store.write(good)
    # Inject a corrupted neighbour with the same .json suffix.
    bad_path = store.approved_dir / "broken.json"
    bad_path.write_text("{not: valid: json", encoding="utf-8")
    # And a schema-version mismatch.
    stale_payload = protocol_to_payload(good)
    stale_payload["schema_version"] = "0.0-stale"
    stale_path = store.approved_dir / "stale.json"
    stale_path.write_text(json.dumps(stale_payload), encoding="utf-8")
    listed = store.list_all()
    assert [p.protocol_id for p in listed] == ["test:good"]


def test_write_is_atomic(tmp_path: Path) -> None:
    """A successful write leaves no stale ``.tmp`` neighbour behind."""
    store = ProtocolPersistenceStore(tmp_path / "lib")
    proto = _make_protocol()
    store.write(proto)
    stale_tmps = list(store.approved_dir.glob("*.tmp"))
    assert stale_tmps == []


def test_delete_removes_file(tmp_path: Path) -> None:
    store = ProtocolPersistenceStore(tmp_path / "lib")
    proto = _make_protocol()
    store.write(proto)
    assert store.delete(proto.protocol_id) is True
    assert store.exists(proto.protocol_id) is False
    # Idempotent: second delete returns False, not raise.
    assert store.delete(proto.protocol_id) is False


def test_read_raises_keyerror_when_absent(tmp_path: Path) -> None:
    store = ProtocolPersistenceStore(tmp_path / "lib")
    with pytest.raises(KeyError):
        store.read("nope:never-written")


def test_read_detects_filename_id_mismatch(tmp_path: Path) -> None:
    """File contents must agree with the filename we look up by.

    Catches the case where someone renames a file by hand to point
    at a different protocol_id — silent acceptance would lead to
    very confusing 'why is this protocol behaving like that one'
    bugs.
    """
    store = ProtocolPersistenceStore(tmp_path / "lib")
    proto = _make_protocol(pid="test:real")
    store.write(proto)
    # Rename file so the on-disk handle says "test:fake" but the
    # payload still has "test:real".
    real_path = store._path_for("test:real")
    fake_path = real_path.with_name("test_fake.json")
    real_path.rename(fake_path)
    with pytest.raises(ValueError, match="contains protocol_id"):
        store.read("test:fake")
