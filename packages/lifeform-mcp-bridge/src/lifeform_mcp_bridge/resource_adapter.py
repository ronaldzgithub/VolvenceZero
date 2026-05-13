"""Translate MCP ``resources/list`` -> ``IngestionEnvelope`` chunks.

Per ``docs/specs/mcp-bridge.md`` § "Resource translation":

* The bridge calls ``resources/list`` to discover resources, then
  ``resources/read`` for each one to fetch the text body.
* Each resource becomes an ``IngestionChunk`` (one chunk per
  resource for v0; future packets may split large markdown into
  paragraph-sized chunks via ``lifeform_ingestion.sources``).
* A binary / non-text mime type results in a chunk with
  ``parse_error="non_text_mime:<mime>"`` (silent drop is forbidden
  per the ingestion envelope contract).
* The whole batch is wrapped in one ``IngestionEnvelope`` per
  server, with ``source_kind=CORPUS`` and the manifest's
  ``default_compliance_profile``.

The envelope itself is NOT submitted here — the adapter returns it
to the caller, who routes it through ``BrainSession.run_turn(...
trigger_kind=INGESTION)`` (the canonical ingestion path). This keeps
the resource adapter pure: it produces an envelope, it never writes
to any owner store directly.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from lifeform_ingestion.envelope import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionProvenance,
    IngestionSourceKind,
)

from lifeform_mcp_bridge.client_pool import MCPClientPool
from lifeform_mcp_bridge.errors import (
    MCPBridgeError,
    MCPProtocolError,
)
from lifeform_mcp_bridge.safety_manifest import SafetyManifest, load_manifest
from lifeform_mcp_bridge.server_spec import MCPServerSpec


_LOG = logging.getLogger("lifeform_mcp_bridge.resource_adapter")
_SANITISE_RE = re.compile(r"[^A-Za-z0-9._:\-/]+")
_TEXT_MIME_PREFIXES = ("text/", "application/json", "application/yaml", "application/xml")


async def fetch_envelopes(
    *,
    pool: MCPClientPool,
    specs: Iterable[MCPServerSpec],
) -> tuple[IngestionEnvelope, ...]:
    """Build one ``IngestionEnvelope`` per spec from resources/list.

    Specs with ``enable_resources=False`` are skipped. Specs that
    expose zero resources are skipped (an empty envelope is
    rejected by ``IngestionEnvelope.__post_init__``). Order of the
    returned tuple matches the input ``specs`` order, with skipped
    specs simply absent.
    """
    envelopes: list[IngestionEnvelope] = []
    for spec in specs:
        if not spec.enable_resources:
            continue
        envelope = await _fetch_for_spec(pool=pool, spec=spec)
        if envelope is not None:
            envelopes.append(envelope)
    return tuple(envelopes)


async def _fetch_for_spec(
    *,
    pool: MCPClientPool,
    spec: MCPServerSpec,
) -> IngestionEnvelope | None:
    manifest = load_manifest(
        path=spec.safety_manifest_path,
        expected_server_name=spec.name,
    )
    client = await pool.ensure_started(spec)
    resources = await client.list_resources()
    if not resources:
        return None
    chunks: list[IngestionChunk] = []
    failures: list[str] = []
    for index, raw in enumerate(resources):
        chunk = await _fetch_one_resource(
            pool=pool,
            spec=spec,
            raw=raw,
            index=index,
        )
        chunks.append(chunk)
        if chunk.has_parse_error:
            failures.append(chunk.chunk_id)
    if not chunks:
        return None
    envelope_id = f"mcp:{spec.name}:{uuid.uuid4().hex}"
    return IngestionEnvelope(
        envelope_id=envelope_id,
        source_kind=IngestionSourceKind.CORPUS,
        chunks=tuple(chunks),
        provenance=IngestionProvenance(
            uploader=f"mcp:{spec.name}",
            upload_ts_ms=int(time.time() * 1000),
            source_uri=f"mcp://{spec.name}/resources",
            integrity_hash=_envelope_integrity(chunks),
        ),
        compliance_profile=_compliance_for(manifest=manifest),
        partial_failures=tuple(failures),
    )


async def _fetch_one_resource(
    *,
    pool: MCPClientPool,
    spec: MCPServerSpec,
    raw: Mapping[str, Any],
    index: int,
) -> IngestionChunk:
    uri = raw.get("uri")
    if not isinstance(uri, str) or not uri.strip():
        # Server returned a malformed entry — keep the chunk as a
        # parse_error so we don't silently drop. chunk_id has to be
        # unique within the envelope.
        return IngestionChunk(
            chunk_id=f"mcp:{spec.name}:malformed:{index}",
            text="",
            locator=f"index={index}",
            confidence=0.0,
            parse_error=f"missing_uri:raw={raw!r}",
        )
    name = raw.get("name") or uri
    chunk_id = _safe_chunk_id(server_name=spec.name, name=str(name), index=index)
    mime = str(raw.get("mimeType", "text/plain"))
    if not _is_textual_mime(mime):
        return IngestionChunk(
            chunk_id=chunk_id,
            text="",
            locator=f"uri={uri}",
            confidence=0.0,
            parse_error=f"non_text_mime:{mime}",
        )
    try:
        client = pool.client_for(spec.name)
        result = await client.read_resource(uri=uri)
    except MCPBridgeError as exc:
        return IngestionChunk(
            chunk_id=chunk_id,
            text="",
            locator=f"uri={uri}",
            confidence=0.0,
            parse_error=f"read_failed:{type(exc).__name__}:{exc}",
        )
    body = _extract_text_body(result)
    if body is None:
        return IngestionChunk(
            chunk_id=chunk_id,
            text="",
            locator=f"uri={uri}",
            confidence=0.0,
            parse_error="no_text_payload",
        )
    return IngestionChunk(
        chunk_id=chunk_id,
        text=body,
        locator=f"uri={uri}",
        confidence=1.0,
    )


def _extract_text_body(result: Mapping[str, Any]) -> str | None:
    """MCP ``resources/read`` returns ``{"contents": [{"text": "...", ...}, ...]}``.

    For v0 we concatenate all text segments separated by a blank
    line. Binary blobs (``blob`` field) are skipped (still no
    silent drop — caller surfaces a parse_error if every segment
    was binary).
    """
    contents = result.get("contents")
    if not isinstance(contents, list):
        return None
    parts: list[str] = []
    for entry in contents:
        if not isinstance(entry, Mapping):
            continue
        text = entry.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    if not parts:
        return None
    return "\n\n".join(parts)


def _is_textual_mime(mime: str) -> bool:
    lower = mime.lower()
    if any(lower.startswith(prefix) for prefix in _TEXT_MIME_PREFIXES):
        return True
    return lower in {"application/markdown"}


def _safe_chunk_id(*, server_name: str, name: str, index: int) -> str:
    """Make ``mcp:<server>:<name>:<index>`` safe and unique."""
    sanitised = _SANITISE_RE.sub("_", name).strip("_")[:80] or "resource"
    return f"mcp:{server_name}:{sanitised}:{index}"


def _envelope_integrity(chunks: Iterable[IngestionChunk]) -> str:
    """Stable SHA256 across the ordered chunk texts.

    Used by the audit log to detect when a server changes its
    resource set across sessions; the hash IS the integrity_hash on
    the IngestionProvenance record.
    """
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(chunk.text.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _compliance_for(*, manifest: SafetyManifest) -> IngestionComplianceProfile:
    if manifest.resources_default_compliance_profile == "consultative":
        return IngestionComplianceProfile.CONSULTATIVE
    return IngestionComplianceProfile.FORCED


__all__ = [
    "fetch_envelopes",
]
