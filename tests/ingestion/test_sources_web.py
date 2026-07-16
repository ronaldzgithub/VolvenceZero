"""Web source adapter tests (Gap 3 slice 2b).

Network is never touched: the fetch path injects a fake ``requests``
module via monkeypatching ``_load_requests``. The readability path
skips cleanly when the ``[web]`` extra is not installed; the stdlib
extractor path (explicit caller-chosen fallback) is always covered.
"""

from __future__ import annotations

import importlib.util

import pytest

from lifeform_ingestion import (
    IngestionComplianceProfile,
    IngestionSourceKind,
    WebIngestionError,
    envelope_from_html,
    envelope_from_url,
)
from lifeform_ingestion.sources import web as web_module


_HTML = """
<html>
  <head><title>Grief pacing</title><style>body {color: red}</style></head>
  <body>
    <script>var tracker = true;</script>
    <article>
      <h1>Supporting someone in grief</h1>
      <p>Acknowledge the felt experience first.</p>
      <p>Add structure gradually so the response does not skip past distress.</p>
    </article>
  </body>
</html>
"""


def _readability_installed() -> bool:
    return importlib.util.find_spec("readability") is not None


# ---------------------------------------------------------------------------
# envelope_from_html
# ---------------------------------------------------------------------------


def test_envelope_from_html_stdlib_extractor() -> None:
    envelope = envelope_from_html(
        _HTML, source_uri="https://example.org/grief", extractor="stdlib"
    )
    assert envelope.source_kind is IngestionSourceKind.WEB
    assert envelope.compliance_profile is IngestionComplianceProfile.FORCED
    assert envelope.partial_failures == ()
    text = " ".join(chunk.text for chunk in envelope.chunks)
    assert "Acknowledge the felt experience first." in text
    # Script / style bodies never leak into ingestable text.
    assert "tracker" not in text
    assert "color: red" not in text
    assert all(
        chunk.locator.startswith("url=https://example.org/grief,offset=")
        for chunk in envelope.chunks
    )


def test_envelope_from_html_empty_input_raises() -> None:
    with pytest.raises(WebIngestionError, match="empty"):
        envelope_from_html("  ", source_uri="https://example.org", extractor="stdlib")


def test_envelope_from_html_no_extractable_text_raises() -> None:
    with pytest.raises(WebIngestionError, match="no extractable"):
        envelope_from_html(
            "<html><body><script>only()</script></body></html>",
            source_uri="https://example.org/empty",
            extractor="stdlib",
        )


def test_envelope_from_html_unknown_extractor_raises() -> None:
    with pytest.raises(WebIngestionError, match="Unknown extractor"):
        envelope_from_html(
            _HTML, source_uri="https://example.org", extractor="playwright"
        )


def test_envelope_from_html_readability_extractor() -> None:
    if not _readability_installed():
        pytest.skip("readability-lxml not installed (lifeform-ingestion[web])")
    envelope = envelope_from_html(_HTML, source_uri="https://example.org/grief")
    text = " ".join(chunk.text for chunk in envelope.chunks)
    assert "Acknowledge the felt experience first." in text
    assert "tracker" not in text


# ---------------------------------------------------------------------------
# URL validation (SSRF discipline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, match",
    [
        ("ftp://example.org/file", "unsupported scheme"),
        ("file:///etc/passwd", "unsupported scheme"),
        ("https://", "no hostname"),
        ("https://user:pass@example.org/", "userinfo"),
    ],
)
def test_envelope_from_url_rejects_bad_urls(url: str, match: str) -> None:
    with pytest.raises(WebIngestionError, match=match):
        envelope_from_url(url)


# ---------------------------------------------------------------------------
# Fetch path (fake requests module — no network)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        content_type: str = "text/html",
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = {"Content-Type": content_type}


class _FakeRequests:
    """Route-table fake for ``requests``; records every GET."""

    def __init__(self, routes: dict[str, object]) -> None:
        self._routes = routes
        self.calls: list[str] = []

    def get(self, url: str, *, timeout: float, headers: dict) -> _FakeResponse:
        self.calls.append(url)
        outcome = self._routes.get(url)
        if outcome is None:
            return _FakeResponse(status_code=404)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, list):
            # Sequential outcomes for retry tests.
            next_outcome = outcome.pop(0)
            if isinstance(next_outcome, Exception):
                raise next_outcome
            return next_outcome
        assert isinstance(outcome, _FakeResponse)
        return outcome


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeRequests) -> None:
    monkeypatch.setattr(web_module, "_load_requests", lambda: fake)


_PAGE_URL = "https://example.org/article"
_ROBOTS_URL = "https://example.org/robots.txt"


def test_envelope_from_url_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRequests(
        {
            _ROBOTS_URL: _FakeResponse(status_code=404),
            _PAGE_URL: _FakeResponse(text=_HTML, content_type="text/html"),
        }
    )
    _install(monkeypatch, fake)
    envelope = envelope_from_url(_PAGE_URL, extractor="stdlib")
    assert envelope.source_kind is IngestionSourceKind.WEB
    assert envelope.provenance.source_uri == _PAGE_URL
    # robots checked BEFORE the page GET.
    assert fake.calls[0] == _ROBOTS_URL
    assert fake.calls[1] == _PAGE_URL


def test_envelope_from_url_robots_disallow_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRequests(
        {
            _ROBOTS_URL: _FakeResponse(
                text="User-agent: *\nDisallow: /article",
                content_type="text/plain",
            ),
            _PAGE_URL: _FakeResponse(text=_HTML),
        }
    )
    _install(monkeypatch, fake)
    with pytest.raises(WebIngestionError, match="disallows"):
        envelope_from_url(_PAGE_URL, extractor="stdlib")
    # The page itself was never fetched.
    assert _PAGE_URL not in fake.calls


def test_envelope_from_url_robots_unreachable_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRequests({_ROBOTS_URL: ConnectionError("boom")})
    _install(monkeypatch, fake)
    with pytest.raises(WebIngestionError, match="robots.txt check failed"):
        envelope_from_url(_PAGE_URL, extractor="stdlib")


def test_envelope_from_url_respect_robots_false_is_explicit_optout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRequests({_PAGE_URL: _FakeResponse(text=_HTML)})
    _install(monkeypatch, fake)
    envelope = envelope_from_url(_PAGE_URL, extractor="stdlib", respect_robots=False)
    assert envelope.chunks
    assert fake.calls == [_PAGE_URL]


def test_envelope_from_url_retries_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRequests(
        {
            _ROBOTS_URL: _FakeResponse(status_code=404),
            _PAGE_URL: [ConnectionError("flaky"), _FakeResponse(text=_HTML)],
        }
    )
    _install(monkeypatch, fake)
    envelope = envelope_from_url(_PAGE_URL, extractor="stdlib")
    assert envelope.chunks
    assert fake.calls.count(_PAGE_URL) == 2


def test_envelope_from_url_fails_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRequests(
        {
            _ROBOTS_URL: _FakeResponse(status_code=404),
            _PAGE_URL: [ConnectionError("down"), ConnectionError("still down")],
        }
    )
    _install(monkeypatch, fake)
    with pytest.raises(WebIngestionError, match="failed after 2 attempt"):
        envelope_from_url(_PAGE_URL, extractor="stdlib")


def test_envelope_from_url_rejects_non_text_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRequests(
        {
            _ROBOTS_URL: _FakeResponse(status_code=404),
            _PAGE_URL: _FakeResponse(
                text='{"not": "html"}', content_type="application/json"
            ),
        }
    )
    _install(monkeypatch, fake)
    with pytest.raises(WebIngestionError, match="content-type"):
        envelope_from_url(_PAGE_URL, extractor="stdlib")


def test_envelope_from_url_rejects_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRequests(
        {
            _ROBOTS_URL: _FakeResponse(status_code=404),
            _PAGE_URL: _FakeResponse(text="<p>" + "x" * 4096 + "</p>"),
        }
    )
    _install(monkeypatch, fake)
    with pytest.raises(WebIngestionError, match="max_content_bytes"):
        envelope_from_url(
            _PAGE_URL, extractor="stdlib", max_content_bytes=1024
        )


def test_envelope_from_url_plain_text_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRequests(
        {
            _ROBOTS_URL: _FakeResponse(status_code=404),
            _PAGE_URL: _FakeResponse(
                text="First paragraph.\n\nSecond paragraph.",
                content_type="text/plain",
            ),
        }
    )
    _install(monkeypatch, fake)
    envelope = envelope_from_url(_PAGE_URL)
    assert envelope.source_kind is IngestionSourceKind.WEB
    assert len(envelope.chunks) == 2
    assert envelope.chunks[0].text == "First paragraph."
