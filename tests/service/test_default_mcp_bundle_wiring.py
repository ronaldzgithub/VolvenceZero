from __future__ import annotations

from lifeform_core import Lifeform, LifeformConfig
from lifeform_service.session_manager import SessionManager
from lifeform_service.vertical_registry import VerticalRegistry
from lifeform_service.verticals import VerticalSpec


def _lifeform_factory(_runtime: object | None) -> Lifeform:
    return Lifeform(LifeformConfig())


def _manager(*, attach_default_mcp_bundle: bool) -> SessionManager:
    spec = VerticalSpec(
        name="test-default-mcp",
        factory=_lifeform_factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
    )
    return SessionManager(
        vertical_registry=VerticalRegistry.single(spec),
        attach_default_mcp_bundle=attach_default_mcp_bundle,
    )


async def test_session_manager_direct_path_does_not_auto_attach_default_mcp(
    monkeypatch,
) -> None:
    calls: list[Lifeform] = []

    def fake_attach(lifeform: Lifeform) -> Lifeform:
        calls.append(lifeform)
        return lifeform

    monkeypatch.setattr(
        "lifeform_service.session_manager.with_default_mcp_bundle",
        fake_attach,
    )

    manager = _manager(attach_default_mcp_bundle=False)
    await manager.create_session(session_id="direct-no-default-mcp")

    assert calls == []


async def test_session_manager_service_path_attaches_default_mcp_when_enabled(
    monkeypatch,
) -> None:
    calls: list[Lifeform] = []

    def fake_attach(lifeform: Lifeform) -> Lifeform:
        calls.append(lifeform)
        return lifeform

    monkeypatch.setattr(
        "lifeform_service.session_manager.with_default_mcp_bundle",
        fake_attach,
    )

    manager = _manager(attach_default_mcp_bundle=True)
    await manager.create_session(session_id="service-default-mcp")

    assert len(calls) == 1
