from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiohttp import web

from lifeform_service.alpha import AlphaServiceConfig
from lifeform_service.app import (
    _CHAT_UI_HTML,
    _error_middleware,
    _handle_relationship_continuity_metrics,
    _handle_relationship_memory,
    _handle_relationship_memory_action,
)
from lifeform_service.relationship_memory_console import (
    RelationshipMemoryAction,
    RelationshipMemoryActionConflictError,
    RelationshipMemoryActionLedger,
    RelationshipMemoryCorrectionKind,
)
from volvence_zero.brain import BrainSession
from volvence_zero.memory import (
    FileSystemPersistenceBackend,
    MemoryEntry,
    MemoryStratum,
    MemoryWriteRequest,
    Track,
    build_default_memory_store,
)
from volvence_zero.reflection import (
    ConsolidationScore,
    MemoryConsolidation,
    PolicyConsolidation,
    ReflectionSnapshot,
    RelationshipUpdateProposal,
)
from volvence_zero.runtime import Snapshot
from volvence_zero.semantic_state import SemanticProposalOperation


def _proposal(*, target: str = "memory", operation: str = "promote"):
    return RelationshipUpdateProposal(
        proposal_id="relationship-update:test",
        target_owner_slot=target,
        operation=operation,
        human_readable_description="Review this relationship update.",
        source_evidence=("memory_entry:memory-1",),
        confidence=0.8,
    )


def _entry(entry_id: str = "memory-1", content: str = "quiet follow-up"):
    return MemoryEntry(
        entry_id=entry_id,
        content=content,
        track=Track.SELF,
        stratum=MemoryStratum.DURABLE.value,
        created_at_ms=1,
        last_accessed_ms=1,
        strength=0.8,
        tags=("user_scope:alice",),
    )


class _FakeBrain:
    def __init__(self, *, proposal=None, entries=()):
        self.reflection = SimpleNamespace(
            relationship_update_proposals=(proposal,) if proposal is not None else ()
        )
        self.entries = tuple(entries)
        self.applied: list[str] = []
        self.continuity_calls = []

    def relationship_reflection_snapshot(self):
        return self.reflection

    def relationship_memory_entries(self):
        return self.entries

    def apply_confirmed_relationship_memory_proposal(self, *, proposal_id, timestamp_ms):
        self.applied.append(f"keep:{proposal_id}:{timestamp_ms}")
        return (f"promoted:{proposal_id}",)

    def delete_relationship_memory_entry(self, *, entry_id):
        target = next((entry for entry in self.entries if entry.entry_id == entry_id), None)
        if target is not None:
            self.entries = tuple(entry for entry in self.entries if entry.entry_id != entry_id)
        return target

    def rewrite_relationship_memory_entry(
        self, *, entry_id, replacement_content, timestamp_ms
    ):
        del timestamp_ms
        target = next((entry for entry in self.entries if entry.entry_id == entry_id), None)
        if target is None:
            return None
        replacement = _entry("memory-rewritten", replacement_content)
        self.entries = (replacement,)
        return replacement

    def relationship_continuity_readout(self, **kwargs):
        self.continuity_calls.append(kwargs)
        outcomes = kwargs["console_outcomes"]
        corrections = sum(item.is_correction for item in outcomes)
        wrong_user = sum(item.is_wrong_user_attribution for item in outcomes)
        return SimpleNamespace(
            to_json=lambda: {
                "wrong_user_attribution_rate": (
                    wrong_user / corrections if corrections else None
                ),
                "user_correction_rate": (
                    corrections / len(outcomes) if outcomes else None
                ),
                "wiring_level": "shadow",
            }
        )


class _FakeSession:
    def __init__(self, brain):
        self.brain_session = brain
        self.semantic_batches = []
        self.turn_summaries = ()
        self.dialogue_outcomes = []

    def submit_semantic_events(self, batch):
        self.semantic_batches.append(batch)
        return tuple(event.event_id for event in batch.events)

    def submit_dialogue_outcome(self, **kwargs):
        self.dialogue_outcomes.append(kwargs)
        return SimpleNamespace(evidence_id=f"outcome:{len(self.dialogue_outcomes)}")

    @property
    def latest_active_snapshots(self):
        return {}


class _FakeManager:
    def __init__(self, session, *, user_id: str = "alice"):
        self.session = session
        self.user_id = user_id

    async def get_session(self, session_id):
        assert session_id == "session-1"
        return self.session

    def session_end_user(self, session_id):
        assert session_id == "session-1"
        return self.user_id


def _app(*, brain, tmp_path):
    app = web.Application(middlewares=[_error_middleware])
    app["session_manager"] = _FakeManager(_FakeSession(brain))
    app["alpha_config"] = AlphaServiceConfig(
        enabled=True,
        alpha_users=frozenset({"alice"}),
        memory_scope_root_dir=str(tmp_path),
    )
    app["relationship_memory_action_ledger"] = RelationshipMemoryActionLedger()
    app.router.add_get(
        "/v1/users/me/relationship-memory",
        _handle_relationship_memory,
    )
    app.router.add_post(
        "/v1/users/me/relationship-memory/{item_id}/action",
        _handle_relationship_memory_action,
    )
    app.router.add_get(
        "/v1/users/me/continuity-metrics",
        _handle_relationship_continuity_metrics,
    )
    return app


async def test_relationship_memory_get_and_keep_are_idempotent(aiohttp_client, tmp_path):
    proposal = _proposal()
    brain = _FakeBrain(proposal=proposal, entries=(_entry(),))
    client = await aiohttp_client(_app(brain=brain, tmp_path=tmp_path))
    headers = {"X-Alpha-User": "alice"}

    response = await client.get(
        "/v1/users/me/relationship-memory?session_id=session-1",
        headers=headers,
    )
    assert response.status == 200
    body = await response.json()
    assert body["pending_proposals"][0]["proposal_id"] == proposal.proposal_id
    assert body["durable_entries"][0]["entry_id"] == "memory-1"

    action_path = (
        "/v1/users/me/relationship-memory/"
        "relationship-update:test/action"
    )
    first = await client.post(
        action_path,
        headers=headers,
        json={"session_id": "session-1", "action": "keep"},
    )
    second = await client.post(
        action_path,
        headers=headers,
        json={"session_id": "session-1", "action": "keep"},
    )
    assert first.status == 201
    assert second.status == 200
    assert (await first.json())["action_id"] == (await second.json())["action_id"]
    assert len(brain.applied) == 1


@pytest.mark.parametrize(
    "action",
    ("mark_sensitive", "no_proactive_mention"),
)
async def test_boundary_action_queues_typed_owner_event(
    aiohttp_client, tmp_path, action
):
    brain = _FakeBrain(entries=(_entry(),))
    app = _app(brain=brain, tmp_path=tmp_path)
    session = app["session_manager"].session
    client = await aiohttp_client(app)
    response = await client.post(
        "/v1/users/me/relationship-memory/memory-1/action",
        headers={"X-Alpha-User": "alice"},
        json={"session_id": "session-1", "action": action},
    )
    assert response.status == 201
    assert (await response.json())["status"] == "queued"
    event = session.semantic_batches[0].events[0]
    assert event.target_slot == "boundary_consent"
    assert event.operation is SemanticProposalOperation.BLOCK


async def test_session_only_resolves_without_owner_write(aiohttp_client, tmp_path):
    proposal = _proposal()
    brain = _FakeBrain(proposal=proposal)
    client = await aiohttp_client(_app(brain=brain, tmp_path=tmp_path))
    headers = {"X-Alpha-User": "alice"}
    response = await client.post(
        "/v1/users/me/relationship-memory/relationship-update:test/action",
        headers=headers,
        json={"session_id": "session-1", "action": "session_only"},
    )
    assert response.status == 201
    assert (await response.json())["status"] == "session_only"
    readout = await client.get(
        "/v1/users/me/relationship-memory?session_id=session-1",
        headers=headers,
    )
    assert (await readout.json())["pending_proposals"] == []
    assert brain.applied == []


async def test_durable_item_can_be_rewritten_then_deleted(aiohttp_client, tmp_path):
    brain = _FakeBrain(entries=(_entry(),))
    client = await aiohttp_client(_app(brain=brain, tmp_path=tmp_path))
    headers = {"X-Alpha-User": "alice"}
    rewritten = await client.post(
        "/v1/users/me/relationship-memory/memory-1/action",
        headers=headers,
        json={
            "session_id": "session-1",
            "action": "rewrite",
            "replacement": "quiet written follow-up",
        },
    )
    assert rewritten.status == 201
    rewritten_body = await rewritten.json()
    assert rewritten_body["replacement_entry_id"] == "memory-rewritten"

    deleted = await client.post(
        "/v1/users/me/relationship-memory/memory-rewritten/action",
        headers=headers,
        json={"session_id": "session-1", "action": "delete"},
    )
    assert deleted.status == 201
    assert (await deleted.json())["owner_operations"] == [
        "deleted:memory-rewritten"
    ]
    assert brain.entries == ()


async def test_relationship_memory_rejects_another_users_session(
    aiohttp_client, tmp_path
):
    app = _app(brain=_FakeBrain(entries=(_entry(),)), tmp_path=tmp_path)
    app["session_manager"].user_id = "bob"
    client = await aiohttp_client(app)
    response = await client.get(
        "/v1/users/me/relationship-memory?session_id=session-1",
        headers={"X-Alpha-User": "alice"},
    )
    assert response.status == 403


def test_action_ledger_rejects_conflicting_second_action() -> None:
    ledger = RelationshipMemoryActionLedger()
    keep_fp = ledger.request_fingerprint(
        action=RelationshipMemoryAction.KEEP,
        replacement=None,
    )
    ledger.record(
        user_id="alice",
        session_id="session-1",
        item_id="proposal-1",
        action=RelationshipMemoryAction.KEEP,
        request_fingerprint=keep_fp,
        status="applied",
        owner_operations=("promoted:memory-1",),
        replacement_entry_id=None,
        correction_kind=None,
        dialogue_outcome_evidence_id=None,
        dialogue_outcome_kind=None,
        created_at_ms=1,
        resolves_proposal=True,
    )
    delete_fp = ledger.request_fingerprint(
        action=RelationshipMemoryAction.DELETE,
        replacement=None,
    )
    with pytest.raises(RelationshipMemoryActionConflictError):
        ledger.ensure_proposal_open(
            user_id="alice",
            session_id="session-1",
            proposal_id="proposal-1",
            request_fingerprint=delete_fp,
        )


async def test_corrective_action_queues_typed_dialogue_outcome(
    aiohttp_client, tmp_path
):
    app = _app(brain=_FakeBrain(entries=(_entry(),)), tmp_path=tmp_path)
    session = app["session_manager"].session
    client = await aiohttp_client(app)
    response = await client.post(
        "/v1/users/me/relationship-memory/memory-1/action",
        headers={"X-Alpha-User": "alice"},
        json={
            "session_id": "session-1",
            "action": "delete",
            "correction_kind": "wrong_user_attribution",
        },
    )
    assert response.status == 201
    body = await response.json()
    assert body["dialogue_outcome_kind"] == "missed"
    assert body["correction_kind"] == "wrong_user_attribution"
    assert session.dialogue_outcomes[0]["kind"].value == "missed"
    assert session.dialogue_outcomes[0]["action_turn_index"] == -1
    metrics_response = await client.get(
        "/v1/users/me/continuity-metrics?session_id=session-1",
        headers={"X-Alpha-User": "alice"},
    )
    assert metrics_response.status == 200
    metrics = await metrics_response.json()
    assert metrics["wrong_user_attribution_rate"] == 1.0
    assert metrics["user_correction_rate"] == 1.0
    assert metrics["wiring_level"] == "shadow"


def test_action_ledger_round_trips_persistent_idempotency(tmp_path) -> None:
    ledger = RelationshipMemoryActionLedger(persistence_root=tmp_path)
    fingerprint = ledger.request_fingerprint(
        action=RelationshipMemoryAction.REWRITE,
        replacement="corrected",
        correction_kind=RelationshipMemoryCorrectionKind.CONTENT_INACCURATE,
    )
    first = ledger.record(
        user_id="alice",
        session_id="session-1",
        item_id="memory-1",
        action=RelationshipMemoryAction.REWRITE,
        request_fingerprint=fingerprint,
        status="applied",
        owner_operations=("rewritten",),
        replacement_entry_id="memory-2",
        correction_kind=RelationshipMemoryCorrectionKind.CONTENT_INACCURATE,
        dialogue_outcome_evidence_id="outcome-1",
        dialogue_outcome_kind=None,
        created_at_ms=1,
        resolves_proposal=False,
    )
    restored = RelationshipMemoryActionLedger(persistence_root=tmp_path)
    assert restored.existing(
        user_id="alice",
        session_id="session-1",
        item_id="memory-1",
        request_fingerprint=fingerprint,
    ) == first


def test_embedded_chat_ui_contains_relationship_memory_console() -> None:
    assert 'id="memoryConsoleSection"' in _CHAT_UI_HTML
    for action in (
        "keep",
        "session_only",
        "delete",
        "rewrite",
        "mark_sensitive",
        "no_proactive_mention",
    ):
        assert f'"{action}"' in _CHAT_UI_HTML


def test_brain_facade_applies_scoped_remember_proposal(tmp_path) -> None:
    source_entry = _entry("memory-new")
    proposal = RelationshipUpdateProposal(
        proposal_id="relationship-update:remember",
        target_owner_slot="memory",
        operation="remember",
        human_readable_description="Remember this reviewed item.",
        source_evidence=("memory_entry:memory-new",),
        confidence=0.9,
    )
    reflection = ReflectionSnapshot(
        memory_consolidation=MemoryConsolidation((source_entry,), (), (), ()),
        policy_consolidation=PolicyConsolidation((), (), ()),
        consolidation_score=ConsolidationScore(
            promotion_score=0.9,
            decay_score=0.1,
            threshold_delta=0.0,
            strategy_gain=0.0,
            regime_effectiveness_gain=0.0,
            confidence=0.9,
            description="test",
        ),
        interaction_trace_summary="test",
        tensions_identified=(),
        lessons_extracted=(),
        writeback_mode="proposal-only",
        review_required=True,
        description="test",
        relationship_update_proposals=(proposal,),
    )
    store = build_default_memory_store(
        persistence_backend=FileSystemPersistenceBackend(
            base_dir=str(tmp_path / "memory")
        )
    )
    runner = SimpleNamespace(
        upstream_snapshots={
            "reflection": Snapshot(
                slot_name="reflection",
                owner="ReflectionModule",
                version=1,
                timestamp_ms=1,
                value=reflection,
            )
        },
        memory_store=store,
        user_scope="alice",
    )
    brain = BrainSession(runner=runner)
    applied = brain.apply_confirmed_relationship_memory_proposal(
        proposal_id=proposal.proposal_id,
        timestamp_ms=2,
    )
    assert "durable:memory-new" in applied
    entries = brain.relationship_memory_entries()
    assert len(entries) == 1
    assert "user_scope:alice" in entries[0].tags


def test_brain_facade_rolls_back_when_memory_is_not_persistable() -> None:
    store = build_default_memory_store()
    entry = store.write(
        MemoryWriteRequest(
            content="quiet follow-up",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            tags=("user_scope:alice",),
            strength=0.8,
        ),
        timestamp_ms=1,
    )
    runner = SimpleNamespace(
        upstream_snapshots={},
        memory_store=store,
        user_scope="alice",
    )
    brain = BrainSession(runner=runner)
    with pytest.raises(RuntimeError, match="could not be persisted"):
        brain.delete_relationship_memory_entry(entry_id=entry.entry_id)
    assert brain.relationship_memory_entries() == (entry,)
    checkpoint = store.create_checkpoint(checkpoint_id="rollback-check")
    assert entry.entry_id in {item.entry_id for item in checkpoint.entry_attributes}
