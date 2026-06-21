"""Cultivation sink — the kernel-facing edge of the self-study loop.

The :class:`CultivationSink` protocol abstracts the four kernel
operations the autonomous loop performs:

* ``research``  — acquire web material on a topic (search + fetch),
* ``ingest``    — feed material through the canonical ingestion pipeline,
* ``study``     — run one apprentice-mode study turn,
* ``reflect``   — drain the slow loop (R6) so reflection settles.

Keeping these behind a protocol means the engine is pure and testable
with a fake sink; the concrete :class:`SessionCultivationSink` binds a
live ``LifeformSession``. The sink NEVER touches cognitive state
directly — it only calls the documented session surface
(``run_turn`` / ``end_scene`` / ``mcp_invoker``) and the
``IngestionPipeline``. There is no token-space control logic here: the
study turn is a plain apprentice turn and the kernel owns the cognition.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from lifeform_core.types import TurnTriggerKind
from lifeform_ingestion import (
    IngestionComplianceProfile,
    IngestionPipeline,
    IngestionSourceKind,
    envelope_from_text,
)


@dataclass(frozen=True)
class ResearchDoc:
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class StudyTurn:
    text: str
    active_regime: str
    active_abstract_action: str


class CultivationSink(Protocol):
    async def research(
        self, query: str, *, max_results: int
    ) -> tuple[ResearchDoc, ...]: ...

    async def uptake_protocol(
        self, *, corpus_text: str, source_label: str
    ) -> str | None: ...

    async def ingest(self, *, corpus_text: str, source_uri: str) -> int: ...

    async def study(self, brief: str) -> StudyTurn: ...

    async def reflect(self, *, reason: str) -> None: ...

    def read_active_mixture(self) -> Any: ...


class SessionCultivationSink:
    """Concrete sink bound to one live ``LifeformSession``.

    ``contract_id`` is forwarded to the affordance invoker so the
    web-research tools resolve against the instance's contract tool
    policy. ``search_tool`` / ``fetch_tool`` default to the vz-bundle
    browse tool descriptor names (``search_web`` / ``fetch_webpage``),
    which are the architecturally-aligned web-research surface (cognition
    selects the affordance; no workflow is hardcoded in an app layer).
    """

    def __init__(
        self,
        *,
        session: Any,
        contract_id: str = "",
        ingestion_pipeline: IngestionPipeline | None = None,
        search_tool: str = "search_web",
        fetch_tool: str = "fetch_webpage",
        uploader: str = "cultivation",
        protocol_uptaker: "Callable[[str, str], Awaitable[str | None]] | None" = None,
        tool_consents: frozenset[str] = frozenset(),
        search_count_arg: str = "top_k",
    ) -> None:
        self._session = session
        self._contract_id = contract_id
        self._pipeline = ingestion_pipeline or IngestionPipeline()
        self._search_tool = search_tool
        self._fetch_tool = fetch_tool
        self._uploader = uploader
        # Consents the platform grants the browse tools for autonomous
        # research (the vz-bundle browse tools require ``network_egress``);
        # an empty set keeps the legacy behaviour. The result-count arg name
        # matches the search tool's schema (vz-bundle ``search_web`` expects
        # ``top_k``, not ``max_results``).
        self._tool_consents = tool_consents
        self._search_count_arg = search_count_arg
        # Bound by the platform layer to a ProtocolUptakeService:
        # (corpus_text, source_label) -> approved protocol_id | None.
        # When absent the sink degrades to raw corpus ingestion (which
        # does NOT engage the school-consolidation mechanism — see the
        # cultivation consistency plan).
        self._protocol_uptaker = protocol_uptaker

    async def research(
        self, query: str, *, max_results: int
    ) -> tuple[ResearchDoc, ...]:
        invoker = self._session.mcp_invoker
        search_result = await invoker.invoke(
            self._search_tool,
            {"query": query, self._search_count_arg: max_results},
            session=None,
            contract_id=self._contract_id or None,
            granted_consents=self._tool_consents,
        )
        if not _succeeded(search_result):
            # Web-research tools are not enabled on this instance's
            # contract (or the backend is unavailable). This is a
            # documented degraded mode, not a swallowed error: the
            # engine still runs a reflection-only study turn.
            return ()
        hits = _parse_search_hits(getattr(search_result, "payload", None))
        docs: list[ResearchDoc] = []
        for hit in hits[:max_results]:
            url = hit.get("url", "").strip()
            if not url:
                continue
            fetch_result = await invoker.invoke(
                self._fetch_tool,
                {"url": url},
                session=None,
                contract_id=self._contract_id or None,
                granted_consents=self._tool_consents,
            )
            if not _succeeded(fetch_result):
                continue
            text, title = _parse_fetched_page(getattr(fetch_result, "payload", None))
            if not text.strip():
                continue
            docs.append(
                ResearchDoc(
                    title=title or hit.get("title", "").strip() or url,
                    url=url,
                    text=text,
                )
            )
        return tuple(docs)

    async def uptake_protocol(
        self, *, corpus_text: str, source_label: str
    ) -> str | None:
        """Turn researched text into a BehaviorProtocol via the uptaker.

        Routes the material through the platform-bound
        ``ProtocolUptakeService`` (DocumentUptake: LLM structured
        extraction -> candidate -> review -> load). The protocol then
        competes in the active mixture on PE utility — this is the
        architecturally-correct path for forming a coherent school. When
        no uptaker is bound (e.g. LLM not configured) we fall back to raw
        corpus ingestion so the loop still makes progress, and return
        ``None`` to signal the degraded path to the engine.
        """

        if self._protocol_uptaker is None:
            await self.ingest(corpus_text=corpus_text, source_uri=source_label)
            return None
        return await self._protocol_uptaker(corpus_text, source_label)

    def read_active_mixture(self) -> Any:
        """Return the published ``active_mixture`` snapshot value or None.

        Reads the same ``latest_active_snapshots`` surface the dispatch
        cognition-enrichment uses. ``None`` when the protocol runtime is
        not publishing the slot in this environment.
        """

        snapshots = getattr(self._session, "latest_active_snapshots", None)
        if not snapshots:
            return None
        snapshot = snapshots.get("active_mixture")
        if snapshot is None:
            return None
        return getattr(snapshot, "value", None)

    async def ingest(self, *, corpus_text: str, source_uri: str) -> int:
        text = corpus_text.strip()
        if not text:
            return 0
        envelope = envelope_from_text(
            text,
            source_uri=source_uri,
            uploader=self._uploader,
            source_kind=IngestionSourceKind.CORPUS,
            compliance_profile=IngestionComplianceProfile.FORCED,
        )
        report = await self._pipeline.process_envelope(
            envelope,
            session=self._session,
            end_scene_after=False,
        )
        return int(report.processed_chunks)

    async def study(self, brief: str) -> StudyTurn:
        result = await self._session.run_turn(
            brief, trigger_kind=TurnTriggerKind.APPRENTICE
        )
        response_text = getattr(getattr(result, "response", None), "text", "") or ""
        active_regime = getattr(result, "active_regime", None) or ""
        abstract_action = getattr(result, "active_abstract_action", None) or ""
        return StudyTurn(
            text=response_text,
            active_regime=str(active_regime),
            active_abstract_action=str(abstract_action),
        )

    async def reflect(self, *, reason: str) -> None:
        await self._session.end_scene(reason=reason, drain_slow_loop=True)


def _succeeded(result: Any) -> bool:
    # AffordanceInvocationStatus is a ``str, Enum`` whose SUCCEEDED value
    # is the literal ``"succeeded"``; comparing against the string keeps
    # this wheel from hard-depending on lifeform-affordance.
    status = getattr(result, "status", None)
    return status is not None and str(getattr(status, "value", status)) == "succeeded"


def _parse_search_hits(payload: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Best-effort extraction of ``{url,title}`` hits from a search payload.

    Accepts both the flat ``{results: [...]}`` shape and the wrapped
    ``{content: [...]}`` shape. Returns ``[]`` for any other shape — a
    documented fallback (the engine degrades to reflection-only), not a
    silent error.
    """

    if not isinstance(payload, Mapping):
        return []
    raw_results: Any = payload.get("results")
    if raw_results is None:
        content = payload.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            raw_results = content
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        return []
    hits: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url", "") or "")
        title = str(item.get("title", "") or "")
        if url:
            hits.append({"url": url, "title": title})
    return hits


def _parse_fetched_page(payload: Mapping[str, Any] | None) -> tuple[str, str]:
    """Extract ``(text, title)`` from a fetch_webpage payload.

    Handles the ``{content: [{type:"text", text: ...}], title}`` shape
    and the flat ``{text, title}`` shape.
    """

    if not isinstance(payload, Mapping):
        return "", ""
    title = str(payload.get("title", "") or "")
    flat_text = payload.get("text")
    if isinstance(flat_text, str) and flat_text.strip():
        return flat_text, title
    content = payload.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                block_text = block.get("text")
                if isinstance(block_text, str) and block_text.strip():
                    parts.append(block_text)
        if parts:
            return "\n\n".join(parts), title
    return "", title


__all__ = [
    "CultivationSink",
    "ResearchDoc",
    "SessionCultivationSink",
    "StudyTurn",
]
