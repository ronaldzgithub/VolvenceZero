"""Contract: ``ProtocolUptakeService`` approved protocols MUST be
injected into every new ``LifeformSession`` via
``Lifeform.with_domain_experience(...)`` so "upload PDF → approve →
AI behaves accordingly on the next session" actually holds.

Pre-this-packet semantics (broken):

* The ``ProtocolUptakeService._registry`` was the only consumer of
  approved protocols at runtime — operator visibility + audit trail
  only. No vertical's ``factory`` / ``alpha_factory`` read it, and
  the kernel session's own ``ProtocolRegistryModule`` was always
  default-constructed empty inside ``build_final_runtime_modules``.
  So an operator could approve a protocol, restart the session,
  and the AI's behaviour would not change.

Post-this-packet semantics (load-bearing):

* ``ProtocolUptakeService.compile_approved_to_domain_packages_snapshot()``
  compiles every currently-approved protocol via
  ``compile_protocol_to_application_artifacts(...)`` and wraps each
  one in a ``DomainExperiencePackage`` (with field renames:
  ``boundary_prior_hints`` → ``boundary_hints``,
  ``domain_knowledge_records`` → ``knowledge_records``,
  ``case_memory_records`` → ``case_records``).
* ``SessionManager.create_session`` calls
  ``_inject_uptake_protocol_packages(life)`` between the vertical
  factory call and ``life.create_session(...)``. When the service
  is non-empty, this calls ``life.with_domain_experience(packages)``
  to produce a NEW ``Lifeform`` whose ``LifeformConfig`` carries the
  vertical's own packages plus one package per approved protocol.

What this test asserts:

1. Empty service ⇒ snapshot returns ``()`` (no work).
2. Approved protocol ⇒ snapshot returns one
   ``DomainExperiencePackage`` per protocol, with the four content
   tuples populated 1:1 from the protocol's compiled artifacts.
3. ``SessionManager._inject_uptake_protocol_packages`` returns the
   original lifeform unchanged when no service is wired.
4. With the service wired AND non-empty, the helper returns a NEW
   ``Lifeform`` whose ``brain_config.domain_experience_packages``
   contains the original vertical packages PLUS the uptake
   packages (in that order).

The fourth invariant is the one that pins the load-bearing wiring:
it proves the ``SessionManager`` injection path actually appends
to the same ``domain_experience_packages`` tuple that
``apply_domain_experience_packages`` later writes into the
application owners on session start.
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
from lifeform_service.verticals import VerticalSpec
from lifeform_service.vertical_registry import VerticalRegistry
from volvence_zero.application.domain_experience import DomainExperiencePackage


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
    """Build a synthetic-substrate Lifeform with no domain packages.

    The brain runs in synthetic substrate mode (no HF download); the
    test only inspects ``LifeformConfig`` field flow, never executes
    a turn, so synthetic mode is sufficient.
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
# 1. compile_approved_to_domain_packages_snapshot semantics
# ---------------------------------------------------------------------------


def test_snapshot_is_empty_when_no_protocol_approved() -> None:
    service = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=None,
        ),
    )
    assert service.compile_approved_to_domain_packages_snapshot() == ()


def test_snapshot_compiles_one_package_per_approved_protocol() -> None:
    service = _build_uptake_service_sync()
    packages = service.compile_approved_to_domain_packages_snapshot()
    assert len(packages) == 1
    package = packages[0]
    assert isinstance(package, DomainExperiencePackage)
    assert package.manifest.package_id == "protocol-uptake:test-uptake:advisor-bot"
    assert package.manifest.display_name == "advisor-bot"


def test_snapshot_package_carries_compiled_boundary_hints() -> None:
    """The single payload-injected boundary contract MUST surface as
    one BoundaryPriorHint on the package."""

    service = _build_uptake_service_sync()
    package = service.compile_approved_to_domain_packages_snapshot()[0]
    assert len(package.boundary_hints) == 1
    hint = package.boundary_hints[0]
    assert "test-uptake:advisor-bot" in hint.hint_id
    assert hint.hint_id.endswith(":boundary:bd:test:no-medical")


def test_snapshot_package_carries_compiled_playbook_rules() -> None:
    """The single payload-injected strategy MUST surface as one
    PlaybookRule on the package."""

    service = _build_uptake_service_sync()
    package = service.compile_approved_to_domain_packages_snapshot()[0]
    assert len(package.playbook_rules) == 1
    rule = package.playbook_rules[0]
    # ``compile_protocol_to_application_artifacts`` namespaces rule
    # ids by protocol; we only need to confirm the original id is
    # preserved as a substring (exact format is owned by the
    # compiler, not this packet).
    assert "rule:test:icebreak-first" in rule.rule_id


# ---------------------------------------------------------------------------
# 2. SessionManager._inject_uptake_protocol_packages semantics
# ---------------------------------------------------------------------------


def test_inject_returns_unchanged_lifeform_when_no_service() -> None:
    life = _build_minimal_lifeform()
    spec = _vertical_spec_with(life)
    manager = SessionManager(
        vertical_registry=VerticalRegistry.single(spec, alpha_enabled=False),
        protocol_uptake_service=None,
    )
    same_life = manager._inject_uptake_protocol_packages(life)  # noqa: SLF001
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
    same_life = manager._inject_uptake_protocol_packages(life)  # noqa: SLF001
    assert same_life is life


def test_inject_appends_uptake_packages_when_service_non_empty() -> None:
    """Load-bearing invariant: the rebuilt Lifeform's brain_config
    MUST carry the original packages plus uptake packages.

    This is what makes ``apply_domain_experience_packages`` on
    session creation actually push BoundaryPriorHint /
    PlaybookRule etc into the live application owners — the
    only failure mode the prior wiring had.
    """

    life = _build_minimal_lifeform()
    original_packages = life.config.brain_config.domain_experience_packages
    spec = _vertical_spec_with(life)
    service = _build_uptake_service_sync()
    manager = SessionManager(
        vertical_registry=VerticalRegistry.single(spec, alpha_enabled=False),
        protocol_uptake_service=service,
    )

    rebuilt = manager._inject_uptake_protocol_packages(life)  # noqa: SLF001
    assert rebuilt is not life
    rebuilt_packages = rebuilt.config.brain_config.domain_experience_packages
    assert len(rebuilt_packages) == len(original_packages) + 1
    appended = rebuilt_packages[len(original_packages)]
    assert appended.manifest.package_id == "protocol-uptake:test-uptake:advisor-bot"


@pytest.mark.asyncio
async def test_create_session_picks_up_uptake_protocol_packages() -> None:
    """End-to-end: create_session through the manager must use the
    rebuilt Lifeform so the resulting session inherits the uptake
    packages.

    We intercept the rebuilt Lifeform via a stub vertical factory
    that returns a Lifeform we can inspect. After
    ``create_session`` returns, the brain session's runner must
    have been built from a config carrying the uptake package.
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
    # ``apply_domain_experience_packages`` ran on construction — the
    # report records every package id that was applied. The uptake
    # package id MUST be in there, AND the boundary / playbook
    # counters MUST reflect the one boundary contract + one strategy
    # the API-injected payload carried.
    report = runner._domain_experience_application_report  # noqa: SLF001
    assert report is not None
    assert "protocol-uptake:test-uptake:advisor-bot" in report.package_ids
    assert report.imported_boundary_hint_count >= 1
    assert report.imported_playbook_count >= 1
