"""Contract: ``ProtocolUptakeService`` approved protocols MUST be
injected into every new ``LifeformSession`` via
``Lifeform.with_seed_protocols(...)`` so "upload PDF → approve →
AI behaves accordingly on the next session" actually holds.

Pre-this-packet semantics (broken):

* The ``ProtocolUptakeService._registry`` was the only consumer of
  approved protocols at runtime — operator visibility + audit trail
  only. No vertical's ``factory`` / ``alpha_factory`` read it, and
  the kernel session's own ``ProtocolRegistryModule`` was always
  default-constructed empty inside ``build_final_runtime_modules``.
  So an operator could approve a protocol, restart the session,
  and the AI's behaviour would not change.

Post-this-packet semantics (load-bearing, ``protocol-online-learning-active``):

* ``ProtocolUptakeService.loaded_approved_snapshot()`` returns the
  current approved ``BehaviorProtocol`` tuple (sync, RLock-safe).
* ``SessionManager.create_session`` calls
  ``_inject_uptake_seed_protocols(life)`` between the vertical
  factory call and ``life.create_session(...)``. When the service
  has approved protocols, this calls
  ``life.with_seed_protocols(approved)`` which returns a NEW
  ``Lifeform`` whose ``LifeformConfig.seed_protocols`` carries the
  approved tuple.
* On session construction, ``AgentSessionRunner.__init__`` builds
  ONE stable ``ProtocolRegistryModule`` and calls
  ``module.load_protocol(p)`` for each seed protocol. With the
  application stores injected, ``load_protocol`` auto-applies the
  compiled ``BoundaryPriorHint`` / ``PlaybookRule`` /
  ``DomainKnowledgeRecord`` / ``CaseMemoryRecord`` artifacts into
  the application owners in the same call.

What this test asserts:

1. ``loaded_approved_snapshot()`` returns ``()`` for an empty service.
2. After approving one payload-injected protocol, the snapshot
   returns a tuple containing that protocol.
3. ``SessionManager._inject_uptake_seed_protocols`` returns the
   original lifeform unchanged when no service is wired or the
   service is empty.
4. With the service wired AND non-empty, the helper returns a NEW
   ``Lifeform`` whose ``LifeformConfig.seed_protocols`` ends with
   the approved protocol (preserving any vertical-default seed).
5. ``create_session`` end-to-end: the runner's stable
   ``ProtocolRegistryModule.registry`` contains the approved
   protocol AND the application stores carry its compiled
   boundary / playbook artifacts (proves ``load_protocol``
   auto-apply ran via the new path).
"""

from __future__ import annotations

import pytest

from lifeform_core import Lifeform, LifeformConfig
from lifeform_protocol_runtime import inject_protocol_from_payload
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from lifeform_service.session_manager import SessionManager
from lifeform_service.vertical_registry import VerticalRegistry
from lifeform_service.verticals import VerticalSpec


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


_SAMPLE_PROTOCOL_SPEC: dict = {
    "protocol_id": "test-uptake:advisor-bot",
    "advisor_name": "advisor-bot",
    "description": "Test advisor protocol used by uptake-injection contract.",
    "boundaries": [
        {
            "boundary_id": "bd:test:no-medical",
            "description": "do not give medical advice",
            "trigger_reasons": ["medical question detected"],
            "blocked_topics": ["dosage"],
            "required_disclaimers": ["consult professional"],
            "severity": "hard_block",
        }
    ],
    "strategies": [
        {
            "rule_id": "rule:test:icebreak-first",
            "problem_pattern": "first contact",
            "recommended_ordering": ["greet", "ask context"],
            "recommended_pacing": "slow",
        }
    ],
}


async def _build_uptake_service_with_one_approved() -> ProtocolUptakeService:
    """Submit + approve one protocol via the API-injection path so
    the test does not need an LLM client.

    Async because :class:`ProtocolUptakeService` mutating routes go
    through ``asyncio.Lock``. Sync tests wrap the helper in
    ``asyncio.run`` themselves."""

    service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=None,
        ),
    )
    candidate = inject_protocol_from_payload(
        _SAMPLE_PROTOCOL_SPEC,
        request_id="req-uptake-injection-test",
        extractor_id="lifeform-service/test/protocol-uptake-injection",
    )
    pid = candidate.protocol.protocol_id
    await service.submit_candidate(candidate, note="contract test fixture")
    await service.approve_pending(pid, reviewer_id="test-reviewer")
    return service


def _build_uptake_service_sync() -> ProtocolUptakeService:
    import asyncio

    return asyncio.run(_build_uptake_service_with_one_approved())


def _build_minimal_lifeform() -> Lifeform:
    """Build a synthetic-substrate Lifeform with no seed protocols.

    The brain runs in synthetic substrate mode (no HF download); the
    test only inspects ``LifeformConfig`` field flow + the
    runner-held module's registry, so synthetic mode is sufficient.
    """
    return Lifeform(LifeformConfig())


def _make_factory_returning(life: Lifeform):
    return lambda _runtime: life


def _vertical_spec_with(life: Lifeform) -> VerticalSpec:
    return VerticalSpec(
        name="test-protocol-uptake-injection",
        factory=_make_factory_returning(life),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )


# ---------------------------------------------------------------------------
# 1. loaded_approved_snapshot() semantics
# ---------------------------------------------------------------------------


def test_snapshot_is_empty_when_no_protocol_approved() -> None:
    service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=None,
        ),
    )
    assert service.loaded_approved_snapshot() == ()


def test_snapshot_contains_approved_protocol() -> None:
    service = _build_uptake_service_sync()
    approved = service.loaded_approved_snapshot()
    assert len(approved) == 1
    assert approved[0].protocol_id == "test-uptake:advisor-bot"
    assert approved[0].advisor_name == "advisor-bot"


def test_snapshot_protocol_carries_boundary_and_strategy_definitions() -> None:
    """The injected payload's boundary / strategy MUST round-trip
    into the loaded protocol so the downstream
    ``load_protocol`` compile path produces the right artifacts."""

    service = _build_uptake_service_sync()
    protocol = service.loaded_approved_snapshot()[0]
    assert any(
        bc.boundary_id == "bd:test:no-medical"
        for bc in protocol.boundary_contracts
    )
    assert any(
        sp.rule_id == "rule:test:icebreak-first"
        for sp in protocol.strategy_priors
    )


# ---------------------------------------------------------------------------
# 2. SessionManager._inject_uptake_seed_protocols semantics
# ---------------------------------------------------------------------------


def test_inject_returns_unchanged_lifeform_when_no_service() -> None:
    life = _build_minimal_lifeform()
    spec = _vertical_spec_with(life)
    manager = SessionManager(
        vertical_registry=VerticalRegistry.single(spec, alpha_enabled=False),
        protocol_uptake_service=None,
    )
    same_life = manager._inject_uptake_seed_protocols(life)  # noqa: SLF001
    assert same_life is life


def test_inject_returns_unchanged_lifeform_when_service_empty() -> None:
    life = _build_minimal_lifeform()
    spec = _vertical_spec_with(life)
    empty_service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=None,
        ),
    )
    manager = SessionManager(
        vertical_registry=VerticalRegistry.single(spec, alpha_enabled=False),
        protocol_uptake_service=empty_service,
    )
    same_life = manager._inject_uptake_seed_protocols(life)  # noqa: SLF001
    assert same_life is life


def test_inject_appends_uptake_seed_protocols_when_service_non_empty() -> None:
    """Load-bearing invariant: the rebuilt Lifeform's
    ``LifeformConfig.seed_protocols`` MUST end with the approved
    protocol(s).

    This is what makes ``ProtocolRegistryModule.load_protocol``
    fire on session creation, which both (a) writes the protocol's
    compiled hint / rule / knowledge / case into the application
    owner stores AND (b) registers the protocol for online α/β
    PE-driven mixing.
    """

    life = _build_minimal_lifeform()
    original_seed_count = len(life.config.seed_protocols)
    spec = _vertical_spec_with(life)
    service = _build_uptake_service_sync()
    manager = SessionManager(
        vertical_registry=VerticalRegistry.single(spec, alpha_enabled=False),
        protocol_uptake_service=service,
    )

    rebuilt = manager._inject_uptake_seed_protocols(life)  # noqa: SLF001
    assert rebuilt is not life
    rebuilt_seed = rebuilt.config.seed_protocols
    assert len(rebuilt_seed) == original_seed_count + 1
    appended = rebuilt_seed[original_seed_count]
    assert appended.protocol_id == "test-uptake:advisor-bot"


@pytest.mark.asyncio
async def test_create_session_picks_up_uptake_protocol() -> None:
    """End-to-end: create_session through the manager must use the
    rebuilt Lifeform so the resulting session's stable
    ``ProtocolRegistryModule.registry`` contains the approved
    protocol AND the application stores carry its compiled
    artifacts.

    We intercept the rebuilt Lifeform via a stub vertical factory
    that returns a Lifeform we can inspect. After
    ``create_session`` returns, the runner must have been built
    from a config carrying the seeded protocol AND the stores
    must reflect the auto-apply that ``load_protocol`` ran during
    runner construction.
    """

    life = _build_minimal_lifeform()
    spec = _vertical_spec_with(life)
    service = await _build_uptake_service_with_one_approved()
    manager = SessionManager(
        vertical_registry=VerticalRegistry.single(spec, alpha_enabled=False),
        protocol_uptake_service=service,
    )
    session = await manager.create_session(session_id="contract-uptake-session")
    runner = session.brain_session.runner

    loaded_ids = {
        p.protocol_id
        for p in runner._protocol_registry_module.registry.loaded()  # noqa: SLF001
    }
    assert "test-uptake:advisor-bot" in loaded_ids

    rare_heavy_state = runner._application_rare_heavy_state  # noqa: SLF001
    assert any(
        "test-uptake:advisor-bot" in hint.hint_id
        for hint in rare_heavy_state.boundary_prior_hints
    )
    assert any(
        "test-uptake:advisor-bot" in rule.rule_id
        for rule in rare_heavy_state.distilled_playbook_rules
    )
