"""Web source adapter (Gap 3 slice 2b).

Turns a fetched web page (or in-memory HTML) into an
``IngestionEnvelope``: readable main-content text is extracted, then
chunked with the same paragraph-aware chunker the plain-text adapter
uses, with per-chunk ``url+offset`` locators.

Design rules (spec: ``docs/specs/runtime-ingestion.md``):

* **requests + readability-lxml only.** No Playwright, no browser
  auto-install. Both deps are the ``lifeform-ingestion[web]`` extra
  and are imported lazily so a thin install pays no cost; a missing
  dep raises a typed error naming the exact install incantation.
* **robots.txt is honored explicitly.** Before any page GET the
  adapter fetches ``<origin>/robots.txt`` and evaluates it with
  ``urllib.robotparser``. A disallow raises; an unreachable robots
  endpoint (other than a clean 404/410 "no robots file") also raises
  — we do not silently assume permission.
* **Bounded network behaviour.** Timeout defaults to 10s, one retry,
  response size capped. Non-HTML/plain content types are refused —
  content-type sniffing is not the adapter's job.
* **Parse failures are explicit.** When readability extracts nothing
  usable the adapter raises ``WebIngestionError`` (an empty envelope
  is forbidden by contract); it never silently drops content. The
  ``extractor="stdlib"`` escape hatch is an explicit, caller-chosen
  fallback (whole-page text via ``html.parser``), never an automatic
  one — a missing readability install must surface, not degrade.
* **SSRF discipline (first pass).** Only ``http`` / ``https`` URLs
  with a hostname and no userinfo are accepted. Hosts that need
  stricter egress policy should enforce it at the network layer.
* **No kernel imports.** Like every source adapter this module is a
  pure chunker; the only consumer of its output is
  ``IngestionPipeline`` driving ``LifeformSession.run_turn``.
"""

from __future__ import annotations

import hashlib
import html.parser
import time
import urllib.parse
import urllib.robotparser
from typing import Any

from lifeform_ingestion.envelope import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionProvenance,
    IngestionSourceKind,
)
from lifeform_ingestion.sources.plain_text import (
    DEFAULT_MAX_CHUNK_CHARS,
    chunk_plain_text,
)


DEFAULT_TIMEOUT_S: float = 10.0
DEFAULT_RETRIES: int = 1
DEFAULT_MAX_CONTENT_BYTES: int = 2 * 1024 * 1024  # 2 MiB
DEFAULT_USER_AGENT: str = "lifeform-ingestion/0.1 (+runtime-ingestion)"

_ACCEPTED_CONTENT_TYPES: tuple[str, ...] = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)


class WebIngestionError(ValueError):
    """Raised on fetch / robots / content-type / extraction failures.

    Subclasses ``ValueError`` like the other source-adapter errors so
    callers with a generic ``ValueError`` branch keep working, while
    web-specific callers can catch this precisely.
    """


def _load_requests() -> Any:
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on install
        raise WebIngestionError(
            "Web ingestion requires the requests dependency. Install with "
            "'pip install lifeform-ingestion[web]' or add requests>=2.31 "
            "to your environment."
        ) from exc
    return requests


def _load_readability() -> Any:
    try:
        import readability  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on install
        raise WebIngestionError(
            "Web ingestion main-content extraction requires the "
            "readability-lxml dependency. Install with "
            "'pip install lifeform-ingestion[web]' or add "
            "readability-lxml>=0.8 to your environment. (To extract "
            "whole-page text without readability, pass "
            "extractor='stdlib' explicitly.)"
        ) from exc
    return readability


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Collect visible text from HTML, skipping script/style/noscript.

    Block-level closes emit paragraph breaks so the downstream
    paragraph-aware chunker sees natural boundaries.
    """

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})
    _BLOCK_TAGS = frozenset(
        {
            "p",
            "div",
            "section",
            "article",
            "li",
            "br",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "tr",
            "blockquote",
            "pre",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self._pieces.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._pieces.append(data)

    def text(self) -> str:
        raw = " ".join(self._pieces)
        paragraphs = [
            " ".join(part.split()) for part in raw.split("\n\n")
        ]
        return "\n\n".join(p for p in paragraphs if p)


def _strip_tags(html_text: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html_text)
    return extractor.text()


def _extract_readable_text(html_text: str, *, extractor: str) -> tuple[str, str]:
    """Return ``(title, text)`` for the page.

    ``extractor="readability"`` (default) isolates the main article
    content; ``extractor="stdlib"`` is the explicit whole-page
    fallback. Anything else is a caller error.
    """
    if extractor == "stdlib":
        return ("", _strip_tags(html_text))
    if extractor != "readability":
        raise WebIngestionError(
            f"Unknown extractor {extractor!r}; expected 'readability' or 'stdlib'."
        )
    readability_mod = _load_readability()
    try:
        document = readability_mod.Document(html_text)
        title = document.short_title() or ""
        summary_html = document.summary(html_partial=True)
    except Exception as exc:  # noqa: BLE001 - re-raised typed with context
        raise WebIngestionError(
            f"readability failed to extract main content: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return (title, _strip_tags(summary_html))


def _validate_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise WebIngestionError(
            f"envelope_from_url: unsupported scheme {parsed.scheme!r} in "
            f"{url!r}; only http/https are allowed."
        )
    if not parsed.hostname:
        raise WebIngestionError(
            f"envelope_from_url: {url!r} has no hostname."
        )
    if parsed.username or parsed.password:
        raise WebIngestionError(
            f"envelope_from_url: {url!r} embeds userinfo credentials; "
            f"refusing (SSRF discipline)."
        )
    return parsed


def _check_robots(
    parsed: urllib.parse.ParseResult,
    *,
    requests_mod: Any,
    timeout_s: float,
    user_agent: str,
) -> None:
    """Explicit robots.txt gate before the page GET.

    404/410 mean "no robots policy" (allowed). A disallow verdict or
    an unreachable robots endpoint raises — assuming permission when
    the policy cannot be read would silently bypass the invariant.
    """
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        response = requests_mod.get(
            robots_url,
            timeout=timeout_s,
            headers={"User-Agent": user_agent},
        )
    except Exception as exc:  # noqa: BLE001 - re-raised typed with context
        raise WebIngestionError(
            f"robots.txt check failed for {robots_url!r}: "
            f"{type(exc).__name__}: {exc}. Refusing to fetch without a "
            f"readable robots policy."
        ) from exc
    if response.status_code in (404, 410):
        return
    if response.status_code != 200:
        raise WebIngestionError(
            f"robots.txt check for {robots_url!r} returned HTTP "
            f"{response.status_code}; refusing to fetch without a readable "
            f"robots policy."
        )
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(response.text.splitlines())
    if not parser.can_fetch(user_agent, parsed.geturl()):
        raise WebIngestionError(
            f"robots.txt at {robots_url!r} disallows fetching "
            f"{parsed.geturl()!r} for user agent {user_agent!r}."
        )


def _fetch_with_retry(
    url: str,
    *,
    requests_mod: Any,
    timeout_s: float,
    retries: int,
    user_agent: str,
) -> Any:
    last_error: Exception | None = None
    for _attempt in range(retries + 1):
        try:
            response = requests_mod.get(
                url,
                timeout=timeout_s,
                headers={"User-Agent": user_agent},
            )
        except Exception as exc:  # noqa: BLE001 - retried then re-raised typed
            last_error = exc
            continue
        if response.status_code == 200:
            return response
        last_error = WebIngestionError(
            f"GET {url!r} returned HTTP {response.status_code}"
        )
    raise WebIngestionError(
        f"envelope_from_url: fetching {url!r} failed after "
        f"{retries + 1} attempt(s): {type(last_error).__name__}: {last_error}"
    ) from last_error


def envelope_from_html(
    html_text: str,
    *,
    source_uri: str,
    uploader: str = "system",
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    extractor: str = "readability",
) -> IngestionEnvelope:
    """Build an envelope from already-fetched HTML.

    Pure offline path (no network): service hosts that fetch pages
    themselves (or tests) call this directly. Raises
    ``WebIngestionError`` when extraction yields no usable text —
    an empty envelope is forbidden by contract.
    """
    if not html_text.strip():
        raise WebIngestionError(
            f"envelope_from_html: input for {source_uri!r} is empty."
        )
    title, text = _extract_readable_text(html_text, extractor=extractor)
    if not text.strip():
        raise WebIngestionError(
            f"envelope_from_html: {source_uri!r} produced no extractable "
            f"text (extractor={extractor!r}); nothing to ingest."
        )
    if title:
        text = f"{title}\n\n{text}"
    pieces = chunk_plain_text(text, max_chunk_chars=max_chunk_chars)
    if not pieces:
        raise WebIngestionError(
            f"envelope_from_html: chunker returned no chunks for {source_uri!r}"
        )
    integrity_hash = hashlib.sha256(html_text.encode("utf-8")).hexdigest()
    if envelope_id is None:
        envelope_id = f"web:{integrity_hash[:12]}"
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)
    chunks = tuple(
        IngestionChunk(
            chunk_id=f"{envelope_id}:chunk:{index:04d}",
            text=segment,
            locator=f"url={source_uri},offset={start}-{end}",
            confidence=1.0,
        )
        for index, (segment, start, end) in enumerate(pieces)
    )
    provenance = IngestionProvenance(
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        source_uri=source_uri,
        integrity_hash=integrity_hash,
    )
    return IngestionEnvelope(
        envelope_id=envelope_id,
        source_kind=IngestionSourceKind.WEB,
        chunks=chunks,
        provenance=provenance,
        compliance_profile=compliance_profile,
        partial_failures=(),
    )


def envelope_from_url(
    url: str,
    *,
    uploader: str = "system",
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    extractor: str = "readability",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    respect_robots: bool = True,
    max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES,
    user_agent: str = DEFAULT_USER_AGENT,
) -> IngestionEnvelope:
    """Fetch ``url`` and build an envelope from its readable content.

    ``respect_robots=True`` (default) performs the explicit
    robots.txt check before the page GET; passing ``False`` is only
    appropriate for origins the operator controls (e.g. an internal
    wiki) and is an explicit caller decision, never a fallback.
    """
    parsed = _validate_url(url)
    if timeout_s <= 0:
        raise WebIngestionError(
            f"envelope_from_url: timeout_s must be > 0, got {timeout_s!r}"
        )
    if retries < 0:
        raise WebIngestionError(
            f"envelope_from_url: retries must be >= 0, got {retries!r}"
        )
    requests_mod = _load_requests()
    if respect_robots:
        _check_robots(
            parsed,
            requests_mod=requests_mod,
            timeout_s=timeout_s,
            user_agent=user_agent,
        )
    response = _fetch_with_retry(
        url,
        requests_mod=requests_mod,
        timeout_s=timeout_s,
        retries=retries,
        user_agent=user_agent,
    )
    content_type = (
        response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    )
    if content_type not in _ACCEPTED_CONTENT_TYPES:
        raise WebIngestionError(
            f"envelope_from_url: {url!r} returned content-type "
            f"{content_type!r}; only {_ACCEPTED_CONTENT_TYPES} are ingestable."
        )
    if len(response.content) > max_content_bytes:
        raise WebIngestionError(
            f"envelope_from_url: {url!r} response is "
            f"{len(response.content)} bytes, exceeds max_content_bytes="
            f"{max_content_bytes}. Raise the budget explicitly if intended."
        )
    body_text = response.text
    if content_type == "text/plain":
        # Plain-text pages skip HTML extraction entirely but keep the
        # WEB source kind + url locators for provenance.
        if not body_text.strip():
            raise WebIngestionError(
                f"envelope_from_url: {url!r} returned an empty plain-text body."
            )
        integrity_hash = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
        resolved_envelope_id = envelope_id or f"web:{integrity_hash[:12]}"
        resolved_ts = (
            upload_ts_ms if upload_ts_ms is not None else int(time.time() * 1000.0)
        )
        pieces = chunk_plain_text(body_text, max_chunk_chars=max_chunk_chars)
        chunks = tuple(
            IngestionChunk(
                chunk_id=f"{resolved_envelope_id}:chunk:{index:04d}",
                text=segment,
                locator=f"url={url},offset={start}-{end}",
                confidence=1.0,
            )
            for index, (segment, start, end) in enumerate(pieces)
        )
        return IngestionEnvelope(
            envelope_id=resolved_envelope_id,
            source_kind=IngestionSourceKind.WEB,
            chunks=chunks,
            provenance=IngestionProvenance(
                uploader=uploader,
                upload_ts_ms=resolved_ts,
                source_uri=url,
                integrity_hash=integrity_hash,
            ),
            compliance_profile=compliance_profile,
            partial_failures=(),
        )
    return envelope_from_html(
        body_text,
        source_uri=url,
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        envelope_id=envelope_id,
        compliance_profile=compliance_profile,
        max_chunk_chars=max_chunk_chars,
        extractor=extractor,
    )


__all__ = [
    "DEFAULT_MAX_CONTENT_BYTES",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_USER_AGENT",
    "WebIngestionError",
    "envelope_from_html",
    "envelope_from_url",
]
