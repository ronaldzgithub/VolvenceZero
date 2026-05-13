"""mcp-tools-bundle-bridge: crash isolation.

Per ``docs/specs/mcp-bridge.md`` invariant 5: an MCP server process
crash MUST NOT crash the main lifeform process. Affected affordances
surface ``blocked_reason="mcp_unavailable:<server>"`` (or are
returned as backend failures) but the kernel keeps running.

Two scenarios:

1. Server raises an internal error mid-call -> next ``invoker.invoke``
   returns ``BACKEND_FAILED`` (or similar typed failure), the
   lifeform process never raises into the test runner.
2. Server is killed externally -> ``MCPClientPool.is_unavailable``
   reflects the loss; subsequent calls do not crash the main process.
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
        env={"VZ_BUNDLE_SANDBOX_ROOT": str(sandbox_root)},
        safety_manifest_path=str(_MANIFEST_PATH),
        autostart=True,
        restart_policy="never",  # crash test wants the server to stay dead
        call_timeout_seconds=10.0,
    )
    return Lifeform(
        LifeformConfig(
            brain_config=BrainConfig(rare_heavy_enabled=False),
            mcp_server_specs=(spec,),
        )
    )


@pytest.mark.skipif(
    not _bundle_available(),
    reason="external/vz-bundle/ submodule not present in this checkout",
)
def test_mcp_server_internal_error_returns_backend_failed_not_crash(
    tmp_path: Path,
) -> None:
    """Calling read_file with a path that escapes the sandbox makes
    the bundle return ``isError=True``. The bridge translates that
    into a ``BACKEND_FAILED`` invocation result; the main process
    keeps running and a follow-up call still works.
    """
    target = tmp_path / "good.txt"
    target.write_text("safe content", encoding="utf-8")
    lifeform = _build_lifeform(sandbox_root=tmp_path)

    async def _scenario() -> None:
        await lifeform.start()
        try:
            session = lifeform.create_session(session_id="mcp-crash-1")
            await session.run_turn("warm")
            invoker = session.mcp_invoker
            assert invoker is not None

            # Call 1: traversal escape -> isError -> BACKEND_FAILED.
            bad = await invoker.invoke(
                "vz-bundle.read_file",
                {"path": "../escape.txt"},
                session=session.brain_session,
                event_id="bad-call-1",
                granted_consents=frozenset({"filesystem_read"}),
            )
            assert bad.status is AffordanceInvocationStatus.BACKEND_FAILED, (
                f"server-side isError must surface as BACKEND_FAILED; got {bad!r}"
            )

            # Call 2: legitimate read still works AFTER the failure.
            ok = await invoker.invoke(
                "vz-bundle.read_file",
                {"path": "good.txt"},
                session=session.brain_session,
                event_id="ok-call-1",
                granted_consents=frozenset({"filesystem_read"}),
            )
            assert ok.status is AffordanceInvocationStatus.SUCCEEDED
        finally:
            await lifeform.shutdown()

    asyncio.run(_scenario())


@pytest.mark.skipif(
    not _bundle_available(),
    reason="external/vz-bundle/ submodule not present in this checkout",
)
def test_mcp_server_subprocess_killed_does_not_crash_main_process(
    tmp_path: Path,
) -> None:
    """Reach into the pool, find the live subprocess, kill it. Main
    process must not raise; ``pool.is_unavailable(name)`` becomes
    True; subsequent invocation returns BACKEND_FAILED rather than
    propagating an unhandled exception.
    """
    lifeform = _build_lifeform(sandbox_root=tmp_path)

    async def _scenario() -> None:
        await lifeform.start()
        try:
            session = lifeform.create_session(session_id="mcp-crash-2")
            await session.run_turn("warm")
            invoker = session.mcp_invoker
            assert invoker is not None
            pool = lifeform.mcp_pool
            assert pool is not None

            # Sanity: server is alive before we kill it.
            client = pool.client_for("vz-bundle")
            assert client.is_alive

            # Reach into the StdioMCPClient subprocess and kill it.
            # We poke ``_proc`` directly because the pool API does
            # not expose a "kill this server for testing" method by
            # design — production code never wants that path.
            proc = client._proc  # noqa: SLF001 — test-only escape
            assert proc is not None
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

            # Subsequent invoke must NOT raise into the test runner;
            # the invoker catches the MCPConnectionLostError and
            # returns BACKEND_FAILED.
            failed = await invoker.invoke(
                "vz-bundle.read_file",
                {"path": "anything.txt"},
                session=session.brain_session,
                event_id="post-kill-call",
                granted_consents=frozenset({"filesystem_read"}),
            )
            assert failed.status is AffordanceInvocationStatus.BACKEND_FAILED
            # Main lifeform run_turn still works after the bundle is dead.
            after = await session.run_turn("the bundle died but i still work")
            assert after.response.text  # any response, just not a crash
        finally:
            await lifeform.shutdown()

    asyncio.run(_scenario())
