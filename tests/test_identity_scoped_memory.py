"""Tests for per-user scoped MemoryStore + IdentityProvider (M4)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from volvence_zero.agent import default_active_runner
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import (
    ANONYMOUS_USER_SCOPE,
    AnonymousIdentityProvider,
    MemoryStratum,
    StaticIdentityProvider,
    UserIdentity,
    build_scoped_memory_store,
    delete_entries_for_scope,
    list_durable_entries_for_scope,
    scope_key_for,
    scoped_memory_dir,
)


def test_user_identity_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="user_id"):
        UserIdentity(user_id=" ", scope_key="x")
    with pytest.raises(ValueError, match="scope_key"):
        UserIdentity(user_id="alice", scope_key="")


def test_anonymous_provider_returns_none() -> None:
    provider = AnonymousIdentityProvider()
    assert provider.resolve("any-session") is None
    assert scope_key_for(None) == ANONYMOUS_USER_SCOPE


def test_static_provider_returns_configured_identity() -> None:
    identity = UserIdentity(user_id="alice", scope_key="alice")
    provider = StaticIdentityProvider(identity=identity)
    assert provider.resolve("sess-1") is identity
    assert provider.resolve("sess-2") is identity
    assert scope_key_for(identity) == "alice"


def test_build_scoped_memory_store_falls_back_for_anonymous() -> None:
    store = build_scoped_memory_store(identity=None)
    # Anonymous: no persistence backend.
    assert store._persistence_backend is None  # noqa: SLF001


def test_build_scoped_memory_store_requires_root_dir_for_identity(tmp_path: Path) -> None:
    identity = UserIdentity(user_id="alice", scope_key="alice")
    with pytest.raises(ValueError, match="root_dir is required"):
        build_scoped_memory_store(identity=identity, root_dir=None)


def test_build_scoped_memory_store_materializes_user_dir(tmp_path: Path) -> None:
    identity = UserIdentity(user_id="alice", scope_key="alice")
    store = build_scoped_memory_store(identity=identity, root_dir=tmp_path)
    assert store._persistence_backend is not None  # noqa: SLF001
    expected = scoped_memory_dir(root_dir=tmp_path, user_id="alice")
    assert expected.exists()


def test_identity_without_persist_permission_gets_in_memory_store(tmp_path: Path) -> None:
    identity = UserIdentity(
        user_id="alice",
        scope_key="alice",
        permissions=("inspect",),
    )
    store = build_scoped_memory_store(identity=identity, root_dir=tmp_path)
    assert store._persistence_backend is None  # noqa: SLF001


def test_brain_with_anonymous_provider_does_not_require_scope_root_dir() -> None:
    brain = Brain(BrainConfig())
    session = brain.create_session(session_id="anon-sess")
    assert session.runner._user_scope == ANONYMOUS_USER_SCOPE  # noqa: SLF001


def test_brain_with_identity_provider_builds_scoped_store_and_uses_user_scope(
    tmp_path: Path,
) -> None:
    identity = UserIdentity(user_id="alice", scope_key="alice")
    brain = Brain(
        BrainConfig(memory_scope_root_dir=str(tmp_path)),
        identity_provider=StaticIdentityProvider(identity=identity),
    )
    session = brain.create_session(session_id="alice-sess-1")
    runner = session.runner
    assert runner._user_scope == "alice"  # noqa: SLF001
    # The scoped user dir exists.
    expected = scoped_memory_dir(root_dir=tmp_path, user_id="alice")
    assert expected.exists()


def test_cross_user_scoped_entries_do_not_leak() -> None:
    """User A's rupture-repair entry must not surface under user B's scope.

    Uses ``AgentSessionRunner`` directly so the test keeps fast.
    """

    async def _run_for_user(scope: str) -> AgentSessionRunner:
        runner = AgentSessionRunner(
            user_scope=scope,
            config=FinalRolloutConfig(),
        )
        await runner.run_turn("first exchange")
        runner.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            confidence=0.9,
        )
        await runner.run_turn("second exchange, that felt cold")
        runner.begin_new_context(reason="test-scene-end")
        await runner.drain_session_post_slow_loop()
        return runner

    runner_alice = asyncio.run(_run_for_user("alice"))
    runner_bob = asyncio.run(_run_for_user("bob"))

    alice_entries = list_durable_entries_for_scope(
        runner_alice._memory_store,  # noqa: SLF001
        user_scope="alice",
    )
    alice_entries_under_bob = list_durable_entries_for_scope(
        runner_alice._memory_store,  # noqa: SLF001
        user_scope="bob",
    )
    bob_entries = list_durable_entries_for_scope(
        runner_bob._memory_store,  # noqa: SLF001
        user_scope="bob",
    )
    # Alice produced at least one entry tagged with her scope.
    assert alice_entries, "Alice should have a rupture_repair entry under her scope."
    # No leakage into Bob's scope when queried on Alice's store.
    assert alice_entries_under_bob == ()
    # Bob's store carries no Alice-scoped entries.
    bob_alice_lookup = list_durable_entries_for_scope(
        runner_bob._memory_store,  # noqa: SLF001
        user_scope="alice",
    )
    assert bob_alice_lookup == ()
    assert bob_entries, "Bob should have a rupture_repair entry under his scope."


def test_delete_entries_for_scope_removes_only_matching_entries(tmp_path: Path) -> None:
    async def _run_for_user(scope: str) -> AgentSessionRunner:
        runner = AgentSessionRunner(
            user_scope=scope,
            config=FinalRolloutConfig(),
        )
        await runner.run_turn("hi")
        runner.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.MISSED,
            source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
            confidence=0.9,
        )
        await runner.run_turn("that missed me")
        runner.begin_new_context(reason="t")
        await runner.drain_session_post_slow_loop()
        return runner

    runner = asyncio.run(_run_for_user("alice"))
    pre = list_durable_entries_for_scope(
        runner._memory_store,  # noqa: SLF001
        user_scope="alice",
    )
    assert pre, "Expected at least one alice rupture_repair entry."

    # Also seed an unrelated entry to verify we don't over-delete. Use
    # the reflection apply path via build_default_memory_store API: we
    # just manually construct an unrelated entry and write it through
    # artifact store to test selective deletion.
    from volvence_zero.memory import MemoryEntry, Track

    unrelated = MemoryEntry(
        entry_id="unrelated-1",
        content="unrelated",
        track=Track.WORLD,
        stratum=MemoryStratum.DURABLE.value,
        created_at_ms=1,
        last_accessed_ms=1,
        strength=0.9,
        tags=("other_tag",),
    )
    runner._memory_store._artifact_store.write(unrelated)  # noqa: SLF001

    deleted = delete_entries_for_scope(
        runner._memory_store,  # noqa: SLF001
        user_scope="alice",
    )
    assert deleted, "Expected delete_entries_for_scope to remove at least one id."

    post = list_durable_entries_for_scope(
        runner._memory_store,  # noqa: SLF001
        user_scope="alice",
    )
    assert post == ()
    # The unrelated entry is still there.
    all_durable = runner._memory_store._entries_for(MemoryStratum.DURABLE)  # noqa: SLF001
    assert any(entry.entry_id == "unrelated-1" for entry in all_durable)
