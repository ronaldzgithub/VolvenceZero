"""Tests for P0.3 per-ai_id memory root resolution."""

from __future__ import annotations

import pathlib

from dlaas_platform_launcher.instance_manager import InstanceManager


def _manager(base: str | None) -> InstanceManager:
    return InstanceManager(
        vertical_resolver=lambda _name: None,
        alpha_memory_scope_root_dir=base,
    )


def test_none_base_returns_none() -> None:
    mgr = _manager(None)
    assert mgr._resolve_memory_root_for(ai_id="ai_1", tenant_id="t1") is None


def test_default_shares_base_root(monkeypatch) -> None:
    monkeypatch.delenv("VZ_PER_AI_MEMORY_ROOT", raising=False)
    mgr = _manager("/data/mem")
    # Backward compat: without opt-in, all ai_ids share the base root.
    assert mgr._resolve_memory_root_for(ai_id="ai_1", tenant_id="t1") == "/data/mem"
    assert mgr._resolve_memory_root_for(ai_id="ai_2", tenant_id="t1") == "/data/mem"


def test_opt_in_per_ai_root(monkeypatch) -> None:
    monkeypatch.setenv("VZ_PER_AI_MEMORY_ROOT", "1")
    mgr = _manager("/data/mem")
    r1 = mgr._resolve_memory_root_for(ai_id="ai_1", tenant_id="t1")
    r2 = mgr._resolve_memory_root_for(ai_id="ai_2", tenant_id="t1")
    assert r1 == str(pathlib.Path("/data/mem") / "t1" / "ai_1")
    assert r2 == str(pathlib.Path("/data/mem") / "t1" / "ai_2")
    assert r1 != r2


def test_opt_in_blank_tenant_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("VZ_PER_AI_MEMORY_ROOT", "1")
    mgr = _manager("/data/mem")
    r = mgr._resolve_memory_root_for(ai_id="ai_1", tenant_id="")
    assert r == str(pathlib.Path("/data/mem") / "default-tenant" / "ai_1")
