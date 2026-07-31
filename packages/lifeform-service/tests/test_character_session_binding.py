from __future__ import annotations

import pytest

from lifeform_service.session_manager import SessionManager
from lifeform_service.vertical_registry import VerticalRegistry
from lifeform_service.verticals import VerticalSpec


class _Session:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.turn_summaries = ()


class _Lifeform:
    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def create_session(self, *, session_id: str) -> _Session:
        return _Session(session_id)


def _spec(name: str, *, character_id: str = "") -> VerticalSpec:
    return VerticalSpec(
        name=name,
        factory=lambda _runtime: _Lifeform(),
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        character_id=character_id,
    )


def _manager() -> SessionManager:
    registry = VerticalRegistry.from_mapping(
        (
            _spec("zhang_wuji", character_id="zhang-wuji"),
            _spec("companion"),
        ),
        default_name="zhang_wuji",
    )
    return SessionManager(
        vertical_registry=registry,
        idle_eviction_seconds=None,
    )


async def test_session_binds_vertical_character_id_immutably() -> None:
    manager = _manager()

    first = await manager.create_session(session_id="first")
    second = await manager.create_session(
        session_id="second",
        character_id="zhang-wuji",
    )

    assert manager.character_id_for(first.session_id) == "zhang-wuji"
    assert manager.character_id_for(second.session_id) == "zhang-wuji"


async def test_session_rejects_character_id_drift_or_dynamic_character_selection() -> None:
    manager = _manager()

    with pytest.raises(ValueError, match="does not match"):
        await manager.create_session(
            session_id="wrong",
            character_id="another-character",
        )
    with pytest.raises(ValueError, match="cannot be selected dynamically"):
        await manager.create_session(
            session_id="generic",
            vertical_name="companion",
            character_id="zhang-wuji",
        )

    assert manager.session_count() == 0
