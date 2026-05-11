"""Document ingestion + chunking (packet 2.2).

Pure-Python PDF / Markdown reader plus deterministic text chunker.
No LLM dependency — this layer is the only place outside the
caller that touches the filesystem.

Determinism guarantees (chunk_document):

* Same input + same params → same output (no random splits).
* Token approximation uses character count / 4 by default
  (rough English heuristic; downstream extractors that need
  exact tokens can re-chunk with a tokenizer).
* Boundaries prefer paragraph breaks > sentence breaks > word
  breaks > character splits, in that order.
* Each chunk records its ``source_offset`` (character index in
  the original text) for audit trail back to the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?。!?])\s+")
_WORD_BREAK = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentText:
    """Raw text + metadata for a single source document."""

    source_locator: str
    text: str
    page_count: int
    description: str = ""

    def __post_init__(self) -> None:
        if not self.source_locator.strip():
            raise ValueError("DocumentText.source_locator must be non-empty")
        if self.page_count < 0:
            raise ValueError(
                "DocumentText.page_count must be >= 0; got "
                f"{self.page_count!r}"
            )


@dataclass(frozen=True)
class DocumentChunk:
    """One chunk of a document, ready to feed an LLM extractor."""

    source_locator: str
    chunk_index: int
    source_offset: int
    text: str

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError(
                "DocumentChunk.chunk_index must be >= 0; got "
                f"{self.chunk_index!r}"
            )
        if self.source_offset < 0:
            raise ValueError(
                "DocumentChunk.source_offset must be >= 0; got "
                f"{self.source_offset!r}"
            )
        if not self.text:
            raise ValueError("DocumentChunk.text must be non-empty")


def read_pdf(path: str | Path) -> DocumentText:
    """Read a PDF file and return its concatenated text.

    Uses ``pypdf`` (pure-Python). Pages are joined by ``\\n\\n``
    so paragraph chunking can detect page boundaries as natural
    break points.
    """

    path = Path(path)
    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    text = "\n\n".join(page_texts).strip()
    return DocumentText(
        source_locator=str(path),
        text=text,
        page_count=len(reader.pages),
        description=f"pdf:{path.name}",
    )


def read_markdown(path: str | Path) -> DocumentText:
    """Read a Markdown / plain-text file as UTF-8 text."""

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    return DocumentText(
        source_locator=str(path),
        text=text,
        page_count=1,
        description=f"markdown:{path.name}",
    )


def _approx_tokens(text: str) -> int:
    """Rough token count: 1 token ~= 4 characters.

    Conservative for Chinese text (which has ~1 char/token in BPE)
    and slightly under for English. Good enough for chunk
    boundary decisions; downstream LLM call should re-count if
    exact budgeting matters.
    """

    return max(1, len(text) // 4)


def chunk_document(
    text: str,
    *,
    source_locator: str,
    max_tokens: int = 2048,
    overlap_tokens: int = 0,
) -> tuple[DocumentChunk, ...]:
    """Split ``text`` into chunks of roughly ``max_tokens`` tokens each.

    Boundary preference (in order):
    1. Paragraph break (``\\n\\n``)
    2. Sentence break (after ``. ! ? 。 ！ ？``)
    3. Word break (whitespace)
    4. Hard character split (last resort)

    ``overlap_tokens`` re-includes the tail of the previous
    chunk at the start of the next, useful for extraction tasks
    that need cross-boundary context. Default 0 (no overlap).
    """

    if not text.strip():
        return ()
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0; got {max_tokens!r}")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            "overlap_tokens must be in [0, max_tokens); got "
            f"{overlap_tokens!r}"
        )

    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunks: list[DocumentChunk] = []
    pos = 0
    chunk_idx = 0
    while pos < len(text):
        end_target = min(pos + max_chars, len(text))
        if end_target == len(text):
            chunk_text = text[pos:end_target].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        source_locator=source_locator,
                        chunk_index=chunk_idx,
                        source_offset=pos,
                        text=chunk_text,
                    )
                )
            break

        # Find a good split point at-or-before end_target.
        slice_text = text[pos:end_target]
        split_at = _find_split(slice_text)
        if split_at <= 0:
            # Couldn't find a clean split; hard cut at end_target.
            split_at = len(slice_text)

        chunk_text = slice_text[:split_at].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    source_locator=source_locator,
                    chunk_index=chunk_idx,
                    source_offset=pos,
                    text=chunk_text,
                )
            )
            chunk_idx += 1

        # Advance: respect overlap if requested.
        next_pos = pos + split_at - overlap_chars
        if next_pos <= pos:
            # Avoid infinite loop on degenerate overlap settings.
            next_pos = pos + split_at
        pos = next_pos

    return tuple(chunks)


def _find_split(text: str) -> int:
    """Find the best split index in ``text`` (preferring late breaks)."""

    matches = list(_PARAGRAPH_BREAK.finditer(text))
    if matches:
        return matches[-1].end()
    matches = list(_SENTENCE_BREAK.finditer(text))
    if matches:
        return matches[-1].end()
    matches = list(_WORD_BREAK.finditer(text))
    if matches:
        return matches[-1].end()
    return len(text)
