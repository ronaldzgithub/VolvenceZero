"""Unit tests for SessionCultivationSink browse-tool wiring.

Verifies the fixed research() path: the configured tool descriptor is
invoked with the granted consents and the `top_k` count arg, and an
unsuccessful tool result degrades to no docs (study/reflect only).
"""

from __future__ import annotations

import asyncio

from lifeform_cultivation.sink import SessionCultivationSink


class _Result:
    def __init__(self, status: str, payload):
        self.status = status
        self.payload = payload


class _RecordingInvoker:
    """Records invoke() calls; scripts search + fetch results."""

    def __init__(self, *, search_ok: bool = True):
        self.calls: list[dict] = []
        self._search_ok = search_ok

    async def invoke(self, tool, params, *, session=None, contract_id=None, granted_consents=frozenset()):
        self.calls.append(
            {
                "tool": tool,
                "params": dict(params),
                "contract_id": contract_id,
                "granted_consents": set(granted_consents),
            }
        )
        if tool.endswith("search_web"):
            if not self._search_ok:
                return _Result("failed", None)
            return _Result(
                "succeeded",
                {"results": [{"url": "https://x/1", "title": "T1"}]},
            )
        # fetch
        return _Result("succeeded", {"text": "body text", "title": "T1"})


class _FakeSession:
    def __init__(self, invoker):
        self.mcp_invoker = invoker


def test_research_invokes_tool_with_consent_and_top_k():
    invoker = _RecordingInvoker(search_ok=True)
    sink = SessionCultivationSink(
        session=_FakeSession(invoker),
        search_tool="vz-bundle.search_web",
        fetch_tool="vz-bundle.fetch_webpage",
        tool_consents=frozenset({"network_egress"}),
    )
    docs = asyncio.run(sink.research("attachment theory", max_results=3))
    assert len(docs) == 1
    assert docs[0].url == "https://x/1"

    search_call = invoker.calls[0]
    assert search_call["tool"] == "vz-bundle.search_web"
    # The count arg is `top_k` (vz-bundle browse schema), not `max_results`.
    assert search_call["params"].get("top_k") == 3
    assert "max_results" not in search_call["params"]
    assert search_call["granted_consents"] == {"network_egress"}

    fetch_call = invoker.calls[1]
    assert fetch_call["tool"] == "vz-bundle.fetch_webpage"
    assert fetch_call["granted_consents"] == {"network_egress"}


def test_research_degrades_to_empty_when_search_fails():
    invoker = _RecordingInvoker(search_ok=False)
    sink = SessionCultivationSink(
        session=_FakeSession(invoker),
        tool_consents=frozenset({"network_egress"}),
    )
    docs = asyncio.run(sink.research("q", max_results=2))
    assert docs == ()
    # Only the search was attempted; no fetch on the degraded path.
    assert len(invoker.calls) == 1


def test_search_count_arg_is_configurable():
    invoker = _RecordingInvoker(search_ok=True)
    sink = SessionCultivationSink(
        session=_FakeSession(invoker),
        search_count_arg="max_results",
    )
    asyncio.run(sink.research("q", max_results=5))
    assert invoker.calls[0]["params"].get("max_results") == 5
