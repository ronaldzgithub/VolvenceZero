"""Service-level state for the protocol uptake HTTP routes.

What this owns:

* A list of **pending candidates** (extracted from PDF / MD /
  task-description / API injection) awaiting human approval.
* A list of **approved protocols** loaded into a service-level
  :class:`ProtocolRegistry`.
* A factory for building :class:`LlmJsonClient` (when configured).
* A sync snapshot accessor
  (:meth:`loaded_approved_snapshot`) that returns the current
  approved set as a tuple of :class:`BehaviorProtocol` so the
  :class:`SessionManager` can hand them to
  ``Lifeform.with_seed_protocols(...)``. From there the kernel
  session's stable :class:`ProtocolRegistryModule.load_protocol`
  auto-applies each protocol's compiled artifacts into the
  application owners AND keeps the protocol available for the
  online α/β PE-driven mixing across turns. This is the
  load-bearing wiring that makes "upload PDF → approve → AI
  behaves accordingly on the next session, and continues
  learning from PE during the session" hold end-to-end. Packet
  name: ``protocol-online-learning-active``.

  An earlier iteration of this packet went via a parallel
  ``DomainExperiencePackage`` injection path; that path was
  removed when the unified ``seed_protocols`` channel landed,
  because going through ``ProtocolRegistryModule.load_protocol``
  produces identical store mutations *and* enables α/β learning
  on the seeded protocols (the package-injection path bypassed
  the registry, so the loaded protocols were invisible to the
  online learning module).

What this does NOT own:

* Per-session :class:`ProtocolRegistryModule` α/β learning state.
  Each kernel session evolves its own weights from PE; that
  evolution is owned by the per-session module. Cross-session
  persistence of those weights goes through
  :class:`OwnerHydrationStore` (not this service).

Concurrency model:

* Mutating routes (submit / approve / reject / unload) go
  through an internal ``asyncio.Lock``.
* The snapshot accessor is sync (the underlying
  :class:`ProtocolRegistry` is RLock-protected, so calling it
  without awaiting our own asyncio lock is safe and avoids
  nested-lock concerns when ``SessionManager.create_session``
  reads it while holding its own lock).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lifeform_protocol_runtime import (
    DirectoryScanResult,
    DocumentChunk,
    LlmJsonClient,
    chunk_document,
    extract_protocol_candidate,
    extract_protocol_from_description,
    inject_protocol_from_payload,
    read_markdown,
    read_pdf,
    scan_directory_for_protocols,
)
from lifeform_service.protocol_persistence import ProtocolPersistenceStore
from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    ProtocolSourceKind,
    ReviewStatus,
)
from volvence_zero.protocol_runtime import ProtocolRegistry


_LOG = logging.getLogger("lifeform_service.protocol_uptake")


@dataclass
class _PendingEntry:
    """One candidate awaiting review."""

    candidate: BehaviorProtocolCandidate
    submitted_at_iso: str
    note: str = ""


@dataclass
class ProtocolUptakeConfig:
    """Construct-time configuration for :class:`ProtocolUptakeService`.

    Args:
        autoload_dir: optional directory scanned at startup. Files
            with ``.pdf`` / ``.md`` / ``.markdown`` / ``.txt`` are
            extracted into pending candidates. ``None`` disables
            autoload.
        autoload_force_approve: development override — if True,
            autoloaded candidates are auto-approved (review_status
            forced to ACTIVE) and pushed straight into the approved
            registry. Default False (operator must approve via API).
        llm_client_factory: callable returning an :class:`LlmJsonClient`
            or ``None``. Used by extraction routes (PDF / MD /
            task-description). Returning ``None`` makes those
            routes respond 503 ("LLM not configured"). API-injection
            routes don't need an LLM and remain available either way.
    """

    autoload_dir: Path | None = None
    autoload_force_approve: bool = False
    llm_client_factory: "callable" = None  # type: ignore[assignment]


class ProtocolUptakeService:
    """Service-level coordinator for the protocol uptake HTTP routes.

    Owns the in-memory pending + approved stores and exposes a small
    typed API used by route handlers. Threadsafe via asyncio.Lock.

    Optional persistence: when constructed with a
    :class:`ProtocolPersistenceStore` the service mirrors the
    in-memory ``approved`` registry to JSON files in
    ``persistence.approved_dir``. The library (disk) and the
    active set (in-memory registry) are intentionally **decoupled**:

    * Approving a candidate writes the protocol to disk AND loads
      it into the registry (current behaviour preserved + persisted).
    * On service restart, the library survives on disk; the
      registry starts empty. The chat-browser UI then surfaces the
      library and lets the operator multi-select which protocols
      to activate via :meth:`load_from_library`.
    * :meth:`unload_from_registry` removes a protocol from the
      active set but keeps the file on disk.
    * :meth:`delete_from_library` removes the file (and unloads
      from the registry if currently loaded).

    This split is the SSOT for the persistence decision the user
    made on 2026-05-22: "auto-discover from disk, multi-select in
    UI, then load" — explicit human activation per restart.
    """

    def __init__(
        self,
        *,
        config: ProtocolUptakeConfig,
        registry: ProtocolRegistry | None = None,
        persistence: ProtocolPersistenceStore | None = None,
    ) -> None:
        self._config = config
        self._registry = registry if registry is not None else ProtocolRegistry()
        self._persistence = persistence
        self._pending: dict[str, _PendingEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def registry(self) -> ProtocolRegistry:
        return self._registry

    @property
    def persistence(self) -> ProtocolPersistenceStore | None:
        return self._persistence

    @property
    def llm_client(self) -> LlmJsonClient | None:
        factory = self._config.llm_client_factory
        if factory is None:
            return None
        return factory()

    async def list_approved(self) -> tuple[BehaviorProtocol, ...]:
        async with self._lock:
            return self._registry.loaded_all()

    def loaded_approved_snapshot(self) -> tuple[BehaviorProtocol, ...]:
        """Return the currently-loaded (non-RETIRED) approved protocols.

        Sync on purpose: the underlying :class:`ProtocolRegistry`
        is RLock-protected, so reading it without taking our own
        ``asyncio.Lock`` is safe and lets
        :class:`SessionManager.create_session` call this from
        inside its own async lock without nesting.

        Empty registry returns ``()`` so the caller can use it
        unconditionally without an extra emptiness check. The
        :class:`SessionManager` forwards the result verbatim to
        :meth:`Lifeform.with_seed_protocols`, which in turn
        threads the protocols through to each new kernel
        session's stable :class:`ProtocolRegistryModule` for
        ``load_protocol`` (auto-applies hint / rule / knowledge
        / case to application owners) AND for online α/β PE-
        driven mixing.
        """

        return self._registry.loaded()

    async def list_pending(self) -> tuple[_PendingEntry, ...]:
        async with self._lock:
            return tuple(self._pending.values())

    # ------------------------------------------------------------------
    # Submit pending candidates
    # ------------------------------------------------------------------

    async def submit_candidate(
        self,
        candidate: BehaviorProtocolCandidate,
        *,
        note: str = "",
    ) -> str:
        async with self._lock:
            return self._submit_locked(candidate, note=note)

    def _submit_locked(
        self,
        candidate: BehaviorProtocolCandidate,
        *,
        note: str = "",
    ) -> str:
        import datetime as _dt
        pid = candidate.protocol.protocol_id
        if not pid.strip():
            raise ValueError("submit_candidate: protocol_id must be non-empty")
        self._pending[pid] = _PendingEntry(
            candidate=candidate,
            submitted_at_iso=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            note=note,
        )
        return pid

    # ------------------------------------------------------------------
    # Approve / reject
    # ------------------------------------------------------------------

    async def approve_pending(
        self,
        protocol_id: str,
        *,
        reviewer_id: str,
    ) -> BehaviorProtocol:
        from dataclasses import replace as _replace
        async with self._lock:
            entry = self._pending.get(protocol_id)
            if entry is None:
                raise KeyError(
                    f"approve_pending: no pending candidate {protocol_id!r}"
                )
            approved = _replace(
                entry.candidate.protocol,
                review_status=ReviewStatus.ACTIVE,
            )
            self._registry.load(approved)
            if self._persistence is not None:
                try:
                    self._persistence.write(approved)
                except OSError as exc:
                    # Disk write failed but the in-memory load already
                    # succeeded. Don't roll back the registry — the
                    # operator just approved this protocol and the
                    # current session must still see it. Surface the
                    # failure loudly so they can fix permissions /
                    # disk space before the next restart, when the
                    # persistence loss would actually bite.
                    _LOG.error(
                        "approve %s: persisted to in-memory registry but "
                        "disk write failed (%s). Protocol is active for "
                        "this run but will NOT survive restart until you "
                        "fix the underlying disk error and re-approve.",
                        protocol_id, exc,
                    )
            del self._pending[protocol_id]
            _LOG.info(
                "approved candidate %s by reviewer %s (persisted=%s)",
                protocol_id, reviewer_id, self._persistence is not None,
            )
            return approved

    async def reject_pending(
        self,
        protocol_id: str,
        *,
        reviewer_id: str,
        reason: str,
    ) -> None:
        async with self._lock:
            if protocol_id not in self._pending:
                raise KeyError(
                    f"reject_pending: no pending candidate {protocol_id!r}"
                )
            del self._pending[protocol_id]
            _LOG.info(
                "rejected candidate %s by reviewer %s reason=%r",
                protocol_id, reviewer_id, reason,
            )

    async def unload_protocol(self, protocol_id: str) -> bool:
        async with self._lock:
            return self._registry.unload(protocol_id)

    # ------------------------------------------------------------------
    # Library (disk-backed persisted approved protocols)
    # ------------------------------------------------------------------

    async def list_library(self) -> tuple[BehaviorProtocol, ...]:
        """Return every approved protocol on disk.

        When no :class:`ProtocolPersistenceStore` is wired, returns
        the empty tuple (library mode is opt-in via the service-level
        ``--protocol-approved-dir`` flag). The result is sorted by
        ``protocol_id`` for deterministic UI ordering.
        """

        if self._persistence is None:
            return ()
        async with self._lock:
            return self._persistence.list_all()

    async def load_from_library(self, protocol_id: str) -> BehaviorProtocol:
        """Load a persisted protocol into the in-memory active set.

        Reads ``approved_dir/<id>.json``, deserialises, and calls
        ``registry.load(...)``. Loading is idempotent — if the
        protocol is already in the registry the disk version
        overwrites the in-memory copy (handy when an external
        editor changed the JSON between activations).

        Raises:
            RuntimeError: no persistence store wired.
            KeyError: no file with that protocol_id on disk.
        """

        if self._persistence is None:
            raise RuntimeError(
                "load_from_library: no persistence store wired; pass "
                "--protocol-approved-dir at service start to enable "
                "the library mode."
            )
        async with self._lock:
            protocol = self._persistence.read(protocol_id)
            self._registry.load(protocol)
            _LOG.info(
                "loaded persisted protocol %s into in-memory registry",
                protocol_id,
            )
            return protocol

    async def unload_from_registry(self, protocol_id: str) -> bool:
        """Drop a protocol from the in-memory registry; keep the disk file.

        Idempotent: returns ``False`` when the protocol wasn't
        loaded. New sessions will no longer pick this protocol up
        via ``with_seed_protocols``; existing sessions keep the
        already-injected copy (per-session state is owned by the
        session's own :class:`ProtocolRegistryModule`).
        """

        async with self._lock:
            return self._registry.unload(protocol_id)

    async def delete_from_library(self, protocol_id: str) -> bool:
        """Delete the persisted JSON file AND unload from registry.

        This is the destructive variant — once called there is no
        way to bring this protocol back without re-uploading the
        original source document and re-approving. Returns
        ``True`` when something on disk was deleted (file existed),
        independent of whether the protocol was also loaded in
        the active set at the time of the call.

        Raises:
            RuntimeError: no persistence store wired.
        """

        if self._persistence is None:
            raise RuntimeError(
                "delete_from_library: no persistence store wired"
            )
        async with self._lock:
            self._registry.unload(protocol_id)
            removed = self._persistence.delete(protocol_id)
            if removed:
                _LOG.info(
                    "deleted persisted protocol %s from library and "
                    "registry",
                    protocol_id,
                )
            return removed

    def library_state_snapshot(self) -> tuple[tuple[BehaviorProtocol, bool], ...]:
        """Return ``((protocol, is_active), ...)`` for every library entry.

        Sync read used by the HTTP route handler that powers the
        chat-browser library panel. ``is_active`` is True when the
        protocol_id is also currently in the in-memory registry
        (non-RETIRED). When no persistence store is wired, returns
        an empty tuple.
        """

        if self._persistence is None:
            return ()
        active_ids = {p.protocol_id for p in self._registry.loaded()}
        return tuple(
            (proto, proto.protocol_id in active_ids)
            for proto in self._persistence.list_all()
        )

    # ------------------------------------------------------------------
    # Extraction adapters
    # ------------------------------------------------------------------

    def _require_llm(self) -> LlmJsonClient:
        client = self.llm_client
        if client is None:
            raise RuntimeError(
                "protocol uptake LLM not configured: set "
                "PROTOCOL_LLM_BASE_URL + PROTOCOL_LLM_API_KEY"
            )
        return client

    async def extract_from_pdf_bytes(
        self,
        pdf_bytes: bytes,
        *,
        filename: str,
        protocol_id_seed: str | None = None,
    ) -> BehaviorProtocolCandidate:
        client = self._require_llm()
        # Save to temp file because read_pdf operates on a path.
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            doc = read_pdf(tmp_path)
            chunks = chunk_document(
                doc.text, source_locator=filename or tmp_path
            )
            return extract_protocol_candidate(
                chunks,
                llm_client=client,
                source_locator=filename or tmp_path,
                source_kind=ProtocolSourceKind.PDF_UPTAKE,
                extractor_id="lifeform-service/uptake",
                protocol_id_seed=protocol_id_seed,
            )
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    async def extract_from_markdown_text(
        self,
        text: str,
        *,
        source_label: str,
        protocol_id_seed: str | None = None,
    ) -> BehaviorProtocolCandidate:
        client = self._require_llm()
        chunks = chunk_document(text, source_locator=source_label)
        return extract_protocol_candidate(
            chunks,
            llm_client=client,
            source_locator=source_label,
            source_kind=ProtocolSourceKind.MARKDOWN_UPTAKE,
            extractor_id="lifeform-service/uptake",
            protocol_id_seed=protocol_id_seed,
        )

    async def extract_from_description(
        self,
        description: str,
        *,
        protocol_id: str,
        advisor_name: str,
    ) -> BehaviorProtocolCandidate:
        client = self._require_llm()
        return extract_protocol_from_description(
            description,
            llm_client=client,
            protocol_id=protocol_id,
            advisor_name=advisor_name,
            extractor_id="lifeform-service/uptake",
        )

    async def inject_from_payload(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
    ) -> BehaviorProtocolCandidate:
        return inject_protocol_from_payload(
            payload,
            request_id=request_id,
            extractor_id="lifeform-service/uptake/api-injection",
        )

    # ------------------------------------------------------------------
    # Startup-time autoload
    # ------------------------------------------------------------------

    async def autoload_directory(self) -> tuple[DirectoryScanResult, ...]:
        """Scan ``config.autoload_dir`` and submit results.

        Returns the per-file scan results so the launcher can log
        them. Files yielding successful candidates land in
        ``pending``; failed files emit log warnings.

        If ``autoload_force_approve`` is True, successful
        candidates are immediately approved (DEV / smoke-test
        path).
        """

        if self._config.autoload_dir is None:
            return ()
        client = self.llm_client
        if client is None:
            _LOG.warning(
                "autoload_directory: PROTOCOL_AUTOLOAD_DIR is set but no LLM "
                "client is configured (PROTOCOL_LLM_BASE_URL / "
                "PROTOCOL_LLM_API_KEY); skipping autoload"
            )
            return ()
        results = scan_directory_for_protocols(
            self._config.autoload_dir,
            llm_client=client,
            extractor_id="lifeform-service/uptake/autoload",
        )
        for result in results:
            if result.status == "ok" and result.candidate is not None:
                await self.submit_candidate(
                    result.candidate, note=f"autoload from {result.source_path}"
                )
                if self._config.autoload_force_approve:
                    try:
                        await self.approve_pending(
                            result.candidate.protocol.protocol_id,
                            reviewer_id="autoload-force-approve",
                        )
                    except KeyError:
                        pass
            elif result.status == "error":
                _LOG.warning(
                    "autoload skipped %s: %s",
                    result.source_path, result.note,
                )
        return results


def candidate_to_json(entry: _PendingEntry) -> dict[str, Any]:
    p = entry.candidate.protocol
    return {
        "protocol_id": p.protocol_id,
        "advisor_name": p.advisor_name,
        "description": p.description,
        "version": p.version,
        "review_status": p.review_status.value,
        "boundary_count": len(p.boundary_contracts),
        "strategy_count": len(p.strategy_priors),
        "knowledge_seed_count": len(p.knowledge_seeds),
        "signature_case_count": len(p.signature_cases),
        "provenance": {
            "source_kind": entry.candidate.provenance.source_kind.value,
            "source_locator": entry.candidate.provenance.source_locator,
            "extracted_at_iso": entry.candidate.provenance.extracted_at_iso,
            "extractor_id": entry.candidate.provenance.extractor_id,
            "confidence": entry.candidate.provenance.confidence,
        },
        "submitted_at_iso": entry.submitted_at_iso,
        "note": entry.note,
        "requires_review": entry.candidate.requires_review,
    }


def protocol_to_json(p: BehaviorProtocol) -> dict[str, Any]:
    return {
        "protocol_id": p.protocol_id,
        "advisor_name": p.advisor_name,
        "description": p.description,
        "version": p.version,
        "review_status": p.review_status.value,
        "source_kind": p.source_kind.value,
        "source_locator": p.source_locator,
        "boundary_count": len(p.boundary_contracts),
        "strategy_count": len(p.strategy_priors),
        "knowledge_seed_count": len(p.knowledge_seeds),
        "signature_case_count": len(p.signature_cases),
        "revision_count": len(p.revision_log),
        "parent_protocol_id": p.parent_protocol_id,
    }


__all__ = [
    "ProtocolUptakeConfig",
    "ProtocolUptakeService",
    "candidate_to_json",
    "protocol_to_json",
]
