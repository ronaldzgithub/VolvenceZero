"""SSRF-bound HTTP client for the L0 crawler.

This is the **only** HTTP-issuing module in the figure vertical. The
client wraps :mod:`requests` with five SSRF gates and a small set of
crawler-specific helpers (304 sentinel, body cap, retry).

Five SSRF gates (all enforced inside :meth:`get`):

1. URL scheme must be http or https.
2. Hostname must be in :attr:`ScopePolicy.allowed_hosts`.
3. URL path must match a prefix in
   :attr:`ScopePolicy.allowed_path_prefixes` (default ``("/",)`` per
   host).
4. Redirects are followed at most one hop, and the redirect target
   is re-validated against the scope policy. Any redirect to a host
   outside scope -> :class:`ScopeRejection`.
5. Response body is streamed with a hard size cap
   (:attr:`ScopePolicy.max_body_bytes`); over-cap responses raise
   :class:`BodyTooLarge`.

Other features:

* 304 Not Modified -> returns :data:`NOT_MODIFIED` sentinel (callers
  treat as "no new bytes" without an exception).
* Network errors / 5xx / 429 -> retried up to ``retries`` times with
  exponential backoff (default 3 retries, 1s base).
* ``etag`` / ``last_modified`` kwargs populate ``If-None-Match`` /
  ``If-Modified-Since`` request headers when ``incremental`` is
  enabled by the scope policy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from lifeform_domain_figure.crawl.scope_policy import ScopePolicy, ScopeRole


class ScopeRejection(ValueError):
    """Raised when an URL is rejected by the scope policy."""


class BodyTooLarge(ValueError):
    """Raised when a response body exceeds ``ScopePolicy.max_body_bytes``."""


class FetchError(RuntimeError):
    """Raised on terminal HTTP / network failure after retries."""


@dataclass(frozen=True)
class HTTPResponse:
    """A successfully fetched HTTP response."""

    url_final: str
    http_status: int
    content_type: str
    body: bytes
    etag: str
    last_modified: str


class _NotModifiedSentinel:
    """Singleton sentinel returned when the server responds 304."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "<NOT_MODIFIED>"


NOT_MODIFIED = _NotModifiedSentinel()


def _build_session(retries: int) -> Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class BaseHTTPClient:
    """SSRF-bound HTTP client for crawler use.

    Single-threaded; share at the scheduler level. Pass a custom
    :class:`Session` for testing (e.g., one with patched ``get``).
    """

    def __init__(
        self,
        *,
        scope: ScopePolicy,
        timeout_s: float = 30.0,
        retries: int = 3,
        session: Session | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError(f"BaseHTTPClient.timeout_s must be > 0; got {timeout_s!r}")
        self._scope = scope
        self._timeout = timeout_s
        self._retries = retries
        self._session = session or _build_session(retries)
        self._sleep = sleep_fn or time.sleep

    @property
    def scope(self) -> ScopePolicy:
        return self._scope

    def close(self) -> None:
        self._session.close()

    def _check_scope(
        self, url: str, *, required_role: ScopeRole | None = None
    ) -> None:
        if not self._scope.is_in_scope(url, required_role=required_role):
            raise ScopeRejection(
                self._scope.reason_out_of_scope(url, required_role=required_role)
                or "url out of scope"
            )

    def _stream_body(self, response: Response) -> bytes:
        cap = self._scope.max_body_bytes
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > cap:
                raise BodyTooLarge(
                    f"response body exceeded scope.max_body_bytes={cap} "
                    f"for url={response.url!r}"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def get(
        self,
        url: str,
        *,
        etag: str = "",
        last_modified: str = "",
        accept: str = "",
        required_role: ScopeRole | None = None,
    ) -> HTTPResponse | _NotModifiedSentinel:
        """Issue an HTTP GET to ``url`` honouring all SSRF gates.

        Returns :data:`NOT_MODIFIED` when the server responds 304 (only
        possible when the scope policy allows incremental and the
        caller passed an ``etag`` or ``last_modified``).

        ``required_role`` (debt #26 closure): when supplied, the host
        MUST carry that :class:`ScopeRole` in
        :attr:`ScopePolicy.host_roles`. Backwards compatible: callers
        passing ``required_role=None`` (the default) get the legacy
        allowlist-only check. L0 corpus fetchers and D4 metadata
        clients pass their respective roles to prevent cross-role
        SSRF.
        """

        self._check_scope(url, required_role=required_role)
        headers: dict[str, str] = {"User-Agent": self._scope.user_agent}
        if accept:
            headers["Accept"] = accept
        if self._scope.incremental:
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified
        try:
            response = self._session.get(
                url,
                headers=headers,
                timeout=self._timeout,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise FetchError(f"network error fetching url={url!r}: {exc}") from exc

        if response.status_code in (301, 302, 303, 307, 308):
            target = response.headers.get("Location", "")
            response.close()
            if not target:
                raise FetchError(
                    f"redirect from url={url!r} missing Location header"
                )
            if not urlparse(target).scheme:
                base = urlparse(url)
                target = f"{base.scheme}://{base.netloc}{target}"
            try:
                self._check_scope(target, required_role=required_role)
            except ScopeRejection as exc:
                raise ScopeRejection(
                    f"redirect target {target!r} from {url!r} rejected: {exc}"
                ) from exc
            try:
                response = self._session.get(
                    target,
                    headers=headers,
                    timeout=self._timeout,
                    stream=True,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise FetchError(
                    f"network error following redirect to {target!r}: {exc}"
                ) from exc

        if response.status_code == 304:
            response.close()
            return NOT_MODIFIED
        if response.status_code >= 400:
            response.close()
            raise FetchError(
                f"http_status={response.status_code} for url={url!r}"
            )
        try:
            body = self._stream_body(response)
        finally:
            response.close()
        content_type = response.headers.get("Content-Type", "").strip()
        return HTTPResponse(
            url_final=str(response.url),
            http_status=int(response.status_code),
            content_type=content_type,
            body=body,
            etag=response.headers.get("ETag", ""),
            last_modified=response.headers.get("Last-Modified", ""),
        )


__all__ = [
    "BaseHTTPClient",
    "BodyTooLarge",
    "FetchError",
    "HTTPResponse",
    "NOT_MODIFIED",
    "ScopeRejection",
]
