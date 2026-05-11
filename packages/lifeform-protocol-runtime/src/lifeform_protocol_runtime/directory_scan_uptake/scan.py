"""Packet 7.2: directory scan adapter.

Walks a directory recursively, picks up files with supported
extensions (``.pdf`` / ``.md`` / ``.txt``), runs each through
the appropriate uptake adapter, and emits a list of
``BehaviorProtocolCandidate`` instances along with per-file
status (success / skipped / error).

Backwards-compat: this is a thin wrapper around the
``document_uptake`` and ``task_description_uptake`` modules,
so any improvement to those propagates here automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from volvence_zero.behavior_protocol import (
    BehaviorProtocolCandidate,
    ProtocolSourceKind,
)

from lifeform_protocol_runtime.document_uptake.extraction import (
    LlmJsonClient,
    extract_protocol_candidate,
)
from lifeform_protocol_runtime.document_uptake.ingestion import (
    chunk_document,
    read_markdown,
    read_pdf,
)


_SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class DirectoryScanResult:
    """Per-file scan outcome.

    Always carries ``source_path``; ``candidate`` is None on
    failure. ``status`` is one of ``"ok"`` / ``"skipped"`` / ``"error"``.
    """

    source_path: str
    status: str
    candidate: BehaviorProtocolCandidate | None
    note: str = ""


def scan_directory_for_protocols(
    directory: str | Path,
    *,
    llm_client: LlmJsonClient,
    extractor_id: str = "lifeform-protocol-runtime/directory-scan",
    chunk_size: int = 2048,
    overlap: int = 0,
    recursive: bool = True,
) -> tuple[DirectoryScanResult, ...]:
    """Walk ``directory`` and convert each supported file to a candidate.

    PDFs go through :func:`document_uptake.read_pdf`; ``.md``
    and ``.txt`` go through :func:`document_uptake.read_markdown`
    (which is robust to plain text). Each file gets its own
    candidate; failed files emit a result with ``status="error"``
    so callers can audit-log the failure without aborting the
    full scan.
    """

    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise ValueError(
            f"scan_directory_for_protocols: {directory!r} is not a directory"
        )

    pattern = "**/*" if recursive else "*"
    results: list[DirectoryScanResult] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTS:
            results.append(
                DirectoryScanResult(
                    source_path=str(path),
                    status="skipped",
                    candidate=None,
                    note=f"unsupported extension {ext!r}",
                )
            )
            continue
        try:
            if ext == ".pdf":
                doc = read_pdf(path)
                source_kind = ProtocolSourceKind.PDF_UPTAKE
            else:
                doc = read_markdown(path)
                source_kind = ProtocolSourceKind.MARKDOWN_UPTAKE
            chunks = chunk_document(
                doc.text,
                source_locator=str(path),
                max_tokens=chunk_size,
                overlap_tokens=overlap,
            )
            if not chunks:
                results.append(
                    DirectoryScanResult(
                        source_path=str(path),
                        status="skipped",
                        candidate=None,
                        note="empty document after chunking",
                    )
                )
                continue
            candidate = extract_protocol_candidate(
                chunks,
                llm_client=llm_client,
                source_locator=str(path),
                source_kind=source_kind,
                extractor_id=extractor_id,
                protocol_id_seed=path.stem,
            )
            results.append(
                DirectoryScanResult(
                    source_path=str(path),
                    status="ok",
                    candidate=candidate,
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            results.append(
                DirectoryScanResult(
                    source_path=str(path),
                    status="error",
                    candidate=None,
                    note=f"{type(exc).__name__}: {exc}",
                )
            )

    return tuple(results)


__all__ = ["DirectoryScanResult", "scan_directory_for_protocols"]
