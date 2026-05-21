"""Integration tests for ``ProtocolUptakeService`` + ``ProtocolPersistenceStore``.

These cover the cross-restart flow the persistence packet was built
for:

1. Approve a pending candidate → it lands on disk AND in the
   in-memory registry.
2. A second service constructed against the same directory sees
   the approved protocol via ``list_library`` (this is the
   "survives restart" property).
3. The library and the active set are independent — unload from
   registry leaves disk untouched; delete from library unloads
   AND removes the disk file.

We exercise the public async API of ``ProtocolUptakeService``
directly (no HTTP), which is the same surface the HTTP route
handlers in ``protocol_routes.py`` use, so a passing test here
guarantees the route layer can rely on the same invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lifeform_service.protocol_persistence import ProtocolPersistenceStore
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    FailureSignal,
    IdentityAssertion,
    ProtocolProvenance,
    ProtocolSourceKind,
    ReviewStatus,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
)


def _make_candidate(pid: str = "test:uptake-one") -> BehaviorProtocolCandidate:
    protocol = BehaviorProtocol(
        protocol_id=pid,
        version="1.0.0",
        advisor_name="Uptake Advisor",
        description="test candidate",
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
        review_status=ReviewStatus.SHADOW,
    )
    return BehaviorProtocolCandidate(
        protocol=protocol,
        provenance=ProtocolProvenance(
            source_kind=ProtocolSourceKind.PDF_UPTAKE,
            source_locator=f"uploads/{pid}.pdf",
            extracted_at_iso="2026-05-22T00:00:00+00:00",
            extractor_id="test",
            confidence=1.0,
        ),
        requires_review=True,
    )


def _build_service(approved_dir: Path) -> ProtocolUptakeService:
    return ProtocolUptakeService(
        config=ProtocolUptakeConfig(),
        persistence=ProtocolPersistenceStore(approved_dir),
    )


@pytest.mark.asyncio
async def test_approve_persists_to_disk(tmp_path: Path) -> None:
    svc = _build_service(tmp_path / "lib")
    candidate = _make_candidate()
    await svc.submit_candidate(candidate)
    approved = await svc.approve_pending(
        candidate.protocol.protocol_id, reviewer_id="test"
    )
    # In-memory: shows up in active set.
    snapshot = svc.loaded_approved_snapshot()
    assert any(p.protocol_id == approved.protocol_id for p in snapshot)
    # Status flipped to ACTIVE on approval.
    assert approved.review_status == ReviewStatus.ACTIVE
    # On disk: file exists with the approved (ACTIVE) status.
    persisted = svc.persistence.read(approved.protocol_id)
    assert persisted == approved


@pytest.mark.asyncio
async def test_library_survives_restart(tmp_path: Path) -> None:
    """Approve in svc1, then a fresh svc2 must see it via list_library."""
    approved_dir = tmp_path / "lib"
    svc1 = _build_service(approved_dir)
    candidate = _make_candidate()
    await svc1.submit_candidate(candidate)
    await svc1.approve_pending(candidate.protocol.protocol_id, reviewer_id="t")
    # Simulate restart: drop svc1, build svc2 against same dir.
    svc2 = _build_service(approved_dir)
    library = await svc2.list_library()
    assert [p.protocol_id for p in library] == [candidate.protocol.protocol_id]
    # Fresh service starts with empty registry (active set NOT
    # persisted across restarts by design).
    assert svc2.loaded_approved_snapshot() == ()


@pytest.mark.asyncio
async def test_load_from_library_activates(tmp_path: Path) -> None:
    approved_dir = tmp_path / "lib"
    svc1 = _build_service(approved_dir)
    candidate = _make_candidate(pid="test:load")
    await svc1.submit_candidate(candidate)
    approved = await svc1.approve_pending(
        candidate.protocol.protocol_id, reviewer_id="t"
    )
    svc2 = _build_service(approved_dir)
    assert svc2.loaded_approved_snapshot() == ()
    loaded = await svc2.load_from_library(approved.protocol_id)
    assert loaded == approved
    snap = svc2.loaded_approved_snapshot()
    assert [p.protocol_id for p in snap] == [approved.protocol_id]


@pytest.mark.asyncio
async def test_unload_keeps_disk_file(tmp_path: Path) -> None:
    svc = _build_service(tmp_path / "lib")
    candidate = _make_candidate(pid="test:unload")
    await svc.submit_candidate(candidate)
    approved = await svc.approve_pending(
        candidate.protocol.protocol_id, reviewer_id="t"
    )
    assert svc.persistence.exists(approved.protocol_id)
    unloaded = await svc.unload_from_registry(approved.protocol_id)
    assert unloaded is True
    # Disk file still there.
    assert svc.persistence.exists(approved.protocol_id)
    # Active set empty.
    assert svc.loaded_approved_snapshot() == ()
    # Library still lists it (decoupled from active set).
    library = await svc.list_library()
    assert [p.protocol_id for p in library] == [approved.protocol_id]


@pytest.mark.asyncio
async def test_delete_from_library_removes_disk_and_active(tmp_path: Path) -> None:
    svc = _build_service(tmp_path / "lib")
    candidate = _make_candidate(pid="test:delete")
    await svc.submit_candidate(candidate)
    approved = await svc.approve_pending(
        candidate.protocol.protocol_id, reviewer_id="t"
    )
    removed = await svc.delete_from_library(approved.protocol_id)
    assert removed is True
    assert not svc.persistence.exists(approved.protocol_id)
    assert svc.loaded_approved_snapshot() == ()
    library = await svc.list_library()
    assert library == ()


@pytest.mark.asyncio
async def test_library_state_snapshot_marks_active(tmp_path: Path) -> None:
    """The UI binds Load/Unload visibility off ``is_active``; verify it."""
    svc = _build_service(tmp_path / "lib")
    a = _make_candidate(pid="test:a")
    b = _make_candidate(pid="test:b")
    await svc.submit_candidate(a)
    await svc.submit_candidate(b)
    await svc.approve_pending(a.protocol.protocol_id, reviewer_id="t")
    await svc.approve_pending(b.protocol.protocol_id, reviewer_id="t")
    # Unload b from registry but keep on disk.
    await svc.unload_from_registry(b.protocol.protocol_id)
    snap = svc.library_state_snapshot()
    by_id = {p.protocol_id: is_active for p, is_active in snap}
    assert by_id == {"test:a": True, "test:b": False}


@pytest.mark.asyncio
async def test_load_from_library_without_persistence_raises(tmp_path: Path) -> None:
    """No persistence wired → library calls must fail loudly, not silently."""
    svc = ProtocolUptakeService(config=ProtocolUptakeConfig(), persistence=None)
    assert await svc.list_library() == ()
    assert svc.library_state_snapshot() == ()
    with pytest.raises(RuntimeError, match="no persistence store wired"):
        await svc.load_from_library("nope")
    with pytest.raises(RuntimeError, match="no persistence store wired"):
        await svc.delete_from_library("nope")


@pytest.mark.asyncio
async def test_load_from_library_missing_id_raises(tmp_path: Path) -> None:
    svc = _build_service(tmp_path / "lib")
    with pytest.raises(KeyError):
        await svc.load_from_library("nope:never-approved")
