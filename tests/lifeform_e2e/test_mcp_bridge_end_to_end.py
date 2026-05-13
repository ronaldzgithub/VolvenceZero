"""mcp-tools-bundle-bridge: end-to-end via the external/vz-bundle submodule.

Spawns the actual MCP server subprocess shipped in
``external/vz-bundle/`` (git submodule) and walks the full pipeline:

1. ``Lifeform.start()`` brings the pool up + populates the
   lifeform-scoped affordance registry / invoker.
2. ``Lifeform.create_session()`` returns a session whose
   ``mcp_invoker`` mirrors the populated invoker.
3. ``await session.flush_mcp_resources()`` ingests the bundle's
   ``knowledge/`` markdown files via the canonical
   ``IngestionPipeline`` (one INGESTION turn per envelope).
4. ``await invoker.invoke("vz-bundle.read_file", {...},
   plan_ref="p-001")`` succeeds and returns content.
5. ``await session.run_turn("...")`` produces a PE action context
   whose ``prediction_id`` matches ``"p-001"`` — the same lineage
   the long-horizon-closure packet established for in-process
   affordances applies to MCP-supplied affordances unchanged.

Skipped if the external bundle template is not present (e.g. when
running this test against a partial checkout).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from lifeform_affordance import AffordanceInvocationStatus
from lifeform_core.lifeform import Lifeform, LifeformConfig
from lifeform_mcp_bridge import MCPServerSpec
from volvence_zero.brain import BrainConfig


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_ROOT = _REPO_ROOT / "external" / "vz-bundle"
_MANIFEST_PATH = _BUNDLE_ROOT / ".vzbridge.yaml"


def _bundle_available() -> bool:
    return _BUNDLE_ROOT.is_dir() and _MANIFEST_PATH.is_file()


def _build_lifeform(*, sandbox_root: Path) -> Lifeform:
    spec = MCPServerSpec(
        name="vz-bundle",
        transport="stdio",
        command=(sys.executable, "-m", "vz_bundle.server"),
        env={
            "VZ_BUNDLE_SANDBOX_ROOT": str(sandbox_root),
            # Inherit Python path from the test environment so the
            # spawned subprocess can find ``vz_bundle`` (installed
            # via ``pip install -e external/vz-bundle``).
        },
        safety_manifest_path=str(_MANIFEST_PATH),
        autostart=True,
        call_timeout_seconds=15.0,
    )
    config = LifeformConfig(
        brain_config=BrainConfig(rare_heavy_enabled=False),
        mcp_server_specs=(spec,),
    )
    return Lifeform(config)


@pytest.mark.skipif(
    not _bundle_available(),
    reason="external/vz-bundle/ submodule not present in this checkout",
)
def test_mcp_bundle_tool_invocation_round_trip(tmp_path: Path) -> None:
    """Tool invocation succeeds + AffordanceInvocationResult is SUCCEEDED."""
    target = tmp_path / "hello.txt"
    target.write_text("hello from the bundle template", encoding="utf-8")
    lifeform = _build_lifeform(sandbox_root=tmp_path)

    async def _scenario() -> None:
        await lifeform.start()
        try:
            assert lifeform.mcp_invoker is not None
            registry = lifeform.mcp_registry
            assert registry is not None
            descriptor_names = sorted(registry.names())
            # Three reference tools, all prefixed with the server
            # name so they cannot clash with any in-process
            # affordances.
            assert "vz-bundle.read_file" in descriptor_names
            assert "vz-bundle.web_search" in descriptor_names
            assert "vz-bundle.python_eval" in descriptor_names

            session = lifeform.create_session(session_id="mcp-bundle-e2e")
            invoker = session.mcp_invoker
            assert invoker is not None

            # Warm the kernel so submit_tool_result has a session to land on.
            await session.run_turn("warm up the brain")
            result = await invoker.invoke(
                "vz-bundle.read_file",
                {"path": "hello.txt"},
                session=session.brain_session,
                event_id="mcp-bundle-read-1",
                granted_consents=frozenset({"filesystem_read"}),
                plan_ref="p-mcp-001",
            )
            assert result.status is AffordanceInvocationStatus.SUCCEEDED, (
                f"expected SUCCEEDED; got {result!r}"
            )
            assert result.payload is not None
            content_blocks = result.payload["content"]
            assert content_blocks[0]["text"] == "hello from the bundle template"
            assert result.tool_event_ids, (
                "tool_event_ids must be non-empty when the invoker routed "
                "through submit_tool_result"
            )

            # Long-horizon-closure plan_ref lineage applies to MCP tools too.
            next_turn = await session.run_turn("what did the read_file return?")
            pe_snapshot = next_turn.active_snapshots["prediction_error"].value
            assert pe_snapshot.action_context.prediction_id == "p-mcp-001"
            assert pe_snapshot.action_context.environment_outcome_id == (
                "mcp-bundle-read-1:outcome"
            )
        finally:
            await lifeform.shutdown()

    asyncio.run(_scenario())


@pytest.mark.skipif(
    not _bundle_available(),
    reason="external/vz-bundle/ submodule not present in this checkout",
)
def test_mcp_bundle_session_flush_resources_ingests_at_least_one_envelope(
    tmp_path: Path,
) -> None:
    """Resource adapter pulls the bundle's knowledge/*.md and the
    session's ``flush_mcp_resources`` drains them via the canonical
    ingestion pipeline. Idempotent on second call.
    """
    lifeform = _build_lifeform(sandbox_root=tmp_path)

    async def _scenario() -> None:
        await lifeform.start()
        try:
            session = lifeform.create_session(session_id="mcp-bundle-flush")
            assert session.pending_mcp_envelope_count >= 1, (
                "bundle template ships at least one knowledge/*.md "
                "file; resource adapter should produce >= 1 envelope"
            )
            reports = await session.flush_mcp_resources()
            assert len(reports) >= 1
            assert session.pending_mcp_envelope_count == 0
            # Idempotent: second call returns ()
            reports_again = await session.flush_mcp_resources()
            assert reports_again == ()
        finally:
            await lifeform.shutdown()

    asyncio.run(_scenario())


@pytest.mark.skipif(
    not _bundle_available(),
    reason="external/vz-bundle/ submodule not present in this checkout",
)
def test_mcp_bundle_disabled_wiring_skips_pool_construction(tmp_path: Path) -> None:
    """``mcp_bridge_wiring=DISABLED`` => no pool spawned, no
    affordance registered, no resources ingested. Used as the
    rollback escape hatch.
    """
    from volvence_zero.runtime import WiringLevel

    spec = MCPServerSpec(
        name="vz-bundle",
        command=(sys.executable, "-m", "vz_bundle.server"),
        safety_manifest_path=str(_MANIFEST_PATH),
        autostart=True,
    )
    config = LifeformConfig(
        brain_config=BrainConfig(rare_heavy_enabled=False),
        mcp_server_specs=(spec,),
        mcp_bridge_wiring=WiringLevel.DISABLED,
    )
    lifeform = Lifeform(config)

    async def _scenario() -> None:
        await lifeform.start()
        try:
            assert lifeform.mcp_started is True
            assert lifeform.mcp_pool is None
            assert lifeform.mcp_invoker is None
            session = lifeform.create_session(session_id="mcp-bundle-disabled")
            assert session.mcp_invoker is None
            assert session.pending_mcp_envelope_count == 0
        finally:
            await lifeform.shutdown()

    asyncio.run(_scenario())
