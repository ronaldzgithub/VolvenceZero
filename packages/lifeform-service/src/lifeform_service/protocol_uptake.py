"""Service-level state for the protocol uptake HTTP routes.

What this owns:

* A list of **pending candidates** (extracted from PDF / MD /
  task-description / API injection) awaiting human approval.
* A list of **approved protocols** loaded into a service-level
  :class:`ProtocolRegistry`.
* A factory for building :class:`LlmJsonClient` (when configured).
* A sync compile path
  (:meth:`compile_approved_to_domain_packages_snapshot`) that
  turns the current approved set into a tuple of
  :class:`DomainExperiencePackage` so the
  :class:`SessionManager` can inject them into each new
  :class:`Lifeform` via ``Lifeform.with_domain_experience(...)``.
  This is the load-bearing wiring that makes "upload PDF →
  approve → AI behaves accordingly on the next session" actually
  hold end-to-end. Packet name: ``protocol-uptake-to-session-injection``.

What this does NOT own:

* Per-session :class:`ProtocolRegistryModule` α/β learning state.
  v1 ships the static-injection path above; α/β reinforcement
  on protocol weights remains a per-session concern handled by
  ``ProtocolRegistryModule`` internals at runtime, fed by the
  PE loop. The service-level registry is the source of truth
  for *which protocols a session inherits at construction*, not
  for their runtime weight evolution.

Concurrency model:

* Mutating routes (submit / approve / reject / unload) go
  through an internal ``asyncio.Lock``.
* The compile-snapshot path is sync (the underlying
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
from volvence_zero.application.domain_experience import (
    DomainExperienceManifest,
    DomainExperiencePackage,
)
from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    ProtocolSourceKind,
    ReviewStatus,
)
from volvence_zero.protocol_runtime import (
    ProtocolApplicationArtifacts,
    ProtocolRegistry,
    compile_protocol_to_application_artifacts,
)


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
    """

    def __init__(
        self,
        *,
        config: ProtocolUptakeConfig,
        registry: ProtocolRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry if registry is not None else ProtocolRegistry()
        self._pending: dict[str, _PendingEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def registry(self) -> ProtocolRegistry:
        return self._registry

    @property
    def llm_client(self) -> LlmJsonClient | None:
        factory = self._config.llm_client_factory
        if factory is None:
            return None
        return factory()

    async def list_approved(self) -> tuple[BehaviorProtocol, ...]:
        async with self._lock:
            return self._registry.loaded_all()

    def compile_approved_to_domain_packages_snapshot(
        self,
    ) -> tuple[DomainExperiencePackage, ...]:
        """Compile every currently-approved (non-RETIRED) protocol into
        a :class:`DomainExperiencePackage`.

        Sync on purpose: the underlying :class:`ProtocolRegistry`
        is RLock-protected, so reading it without taking our own
        ``asyncio.Lock`` is safe and lets
        :class:`SessionManager.create_session` call this from
        inside its own async lock without nesting.

        One :class:`DomainExperiencePackage` per protocol; the
        manifest carries the protocol id / version / advisor name
        verbatim so application owners attribute the resulting
        ``BoundaryPriorHint`` / ``PlaybookRule`` /
        ``DomainKnowledgeRecord`` / ``CaseMemoryRecord`` entries
        back to a single uptake source.

        Empty registry returns ``()`` so the caller can use it
        unconditionally without an extra emptiness check.
        """

        approved = self._registry.loaded()
        if not approved:
            return ()
        packages: list[DomainExperiencePackage] = []
        for protocol in approved:
            artifacts = compile_protocol_to_application_artifacts(protocol)
            packages.append(_artifacts_to_domain_package(protocol, artifacts))
        return tuple(packages)

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
            del self._pending[protocol_id]
            _LOG.info(
                "approved candidate %s by reviewer %s", protocol_id, reviewer_id
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


def _artifacts_to_domain_package(
    protocol: BehaviorProtocol,
    artifacts: ProtocolApplicationArtifacts,
) -> DomainExperiencePackage:
    """Wrap one protocol's compiled artifacts in a typed package.

    Field renames from
    :class:`ProtocolApplicationArtifacts` to
    :class:`DomainExperiencePackage`:

    * ``boundary_prior_hints`` → ``boundary_hints``
    * ``domain_knowledge_records`` → ``knowledge_records``
    * ``case_memory_records`` → ``case_records``
    * ``playbook_rules`` passes through unchanged

    The synthesised ``DomainExperienceManifest`` carries the
    ``protocol_id`` / ``version`` so application owner audit
    and ``DomainExperienceValidationReport`` lineage point
    back to the originating uptake protocol.
    """

    manifest = DomainExperienceManifest(
        package_id=f"protocol-uptake:{protocol.protocol_id}",
        version=protocol.version or "1",
        display_name=protocol.advisor_name or protocol.protocol_id,
        domain_ids=("protocol-uptake",),
        target_contexts=("any",),
        evidence_level=protocol.review_status.value,
        owner="lifeform-service/protocol-uptake",
        description=(
            protocol.description
            or f"Approved protocol {protocol.protocol_id!r} compiled "
            "into a DomainExperiencePackage by ProtocolUptakeService."
        ),
    )
    return DomainExperiencePackage(
        manifest=manifest,
        boundary_hints=artifacts.boundary_prior_hints,
        playbook_rules=artifacts.playbook_rules,
        knowledge_records=artifacts.domain_knowledge_records,
        case_records=artifacts.case_memory_records,
    )


__all__ = [
    "ProtocolUptakeConfig",
    "ProtocolUptakeService",
    "candidate_to_json",
    "protocol_to_json",
]
