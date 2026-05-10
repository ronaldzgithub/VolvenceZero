"""HTTP backbone + on-disk cache for D4 metadata clients.

The four live metadata clients (OpenAlex / Wikidata / Crossref / SEP)
share two pieces of infrastructure:

1. :class:`MetadataHTTPClient` — a thin wrapper around
   :class:`lifeform_domain_figure.crawl.http_client.BaseHTTPClient`
   that fixes ``required_role=ScopeRole.METADATA_FETCH`` on every
   call. Reusing the L0 client means the SSRF five-gate stack +
   role enforcement + retry + body cap apply uniformly to metadata
   API calls.

2. :class:`MetadataCache` — a content-addressable on-disk JSON
   cache so repeated lookups (e.g., the same author / DOI / QID
   referenced by multiple sources) hit local disk instead of the
   API. Layout::

       root/
         metadata_cache/
           {provider}/
             {key_sha256}/
               body.json     # raw response payload (UTF-8 JSON or HTML)
               meta.json     # {"key": "...", "fetched_at_iso": "...", "content_type": "..."}

   ``key`` is the lookup identifier (OpenAlex author id / Wikidata
   QID / Crossref DOI / SEP slug). ``key_sha256`` is the cache file
   key so arbitrary identifier strings (DOIs contain ``/``) land in
   safe filesystem paths.

The cache TTL defaults to ``DEFAULT_TTL_SECONDS`` (24h); operators
override per :class:`MetadataCache` instance. TTL=0 disables
expiration (useful for offline-after-first-fetch evidence runs).

Both pieces are independently injectable; verifiers / tests pass
their own ``MetadataHTTPClient`` (with a mocked HTTP backend) and
``MetadataCache`` (against ``tmp_path``) without standing up real
infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    HTTPResponse,
    NOT_MODIFIED,
)
from lifeform_domain_figure.crawl.scope_policy import (
    ScopePolicy,
    ScopeRole,
    default_metadata_scope_policy,
)


DEFAULT_TTL_SECONDS = 24 * 3600


@dataclass(frozen=True)
class MetadataResponse:
    """A successful metadata fetch carrying body bytes + content-type."""

    body: bytes
    content_type: str
    fetched_at_iso: str
    from_cache: bool = False

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, errors="replace")

    def json(self) -> object:
        return json.loads(self.text())


class MetadataHTTPClient:
    """Metadata-role HTTP wrapper for the four D4 clients.

    Construct one instance per scope policy / process; share across
    clients. The wrapper does NOT manage retries beyond what
    :class:`BaseHTTPClient` does; nor does it cache (use
    :class:`MetadataCache`).
    """

    def __init__(
        self,
        *,
        scope: ScopePolicy | None = None,
        http_client: BaseHTTPClient | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        if http_client is not None and scope is not None:
            raise ValueError(
                "MetadataHTTPClient: pass at most one of {scope, http_client}; "
                "they conflict (http_client carries its own scope)."
            )
        if http_client is not None:
            self._http_client = http_client
        else:
            effective_scope = scope or default_metadata_scope_policy()
            self._http_client = BaseHTTPClient(
                scope=effective_scope, timeout_s=timeout_s
            )

    @property
    def http_client(self) -> BaseHTTPClient:
        return self._http_client

    def get(self, url: str, *, accept: str = "application/json") -> MetadataResponse:
        """Issue a metadata-role GET; returns body + content-type.

        Raises :class:`ScopeRejection` if the URL is out of scope or
        the host lacks the ``METADATA_FETCH`` role; raises
        :class:`FetchError` on terminal HTTP / network failure.
        """

        from datetime import datetime, timezone

        response = self._http_client.get(
            url,
            accept=accept,
            required_role=ScopeRole.METADATA_FETCH,
        )
        if response is NOT_MODIFIED:
            raise RuntimeError(
                "MetadataHTTPClient.get: server returned 304 but no etag was "
                "supplied; this should be unreachable for metadata clients "
                "(which do not opt into incremental fetching)."
            )
        assert isinstance(response, HTTPResponse)
        return MetadataResponse(
            body=response.body,
            content_type=response.content_type,
            fetched_at_iso=datetime.now(timezone.utc).isoformat(),
        )

    def close(self) -> None:
        self._http_client.close()


def _key_sha(provider: str, key: str) -> str:
    payload = f"{provider}\n{key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MetadataCache:
    """Content-addressable JSON cache for metadata API lookups."""

    def __init__(
        self,
        *,
        root: Path,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError(
                f"MetadataCache.root must be a Path; got {type(root).__name__}"
            )
        if ttl_seconds < 0:
            raise ValueError(
                f"MetadataCache.ttl_seconds must be >= 0; got {ttl_seconds!r}"
            )
        self._root = root
        self._cache_root = root / "metadata_cache"
        self._ttl = ttl_seconds

    @property
    def root(self) -> Path:
        return self._root

    def _entry_dir(self, provider: str, key: str) -> Path:
        return self._cache_root / provider / _key_sha(provider, key)

    def get(self, provider: str, key: str) -> MetadataResponse | None:
        """Return a cached response if present and not expired; else ``None``."""

        entry_dir = self._entry_dir(provider, key)
        body_path = entry_dir / "body.json"
        meta_path = entry_dir / "meta.json"
        if not body_path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self._ttl > 0:
            from datetime import datetime
            try:
                fetched_at = datetime.fromisoformat(meta["fetched_at_iso"])
            except (KeyError, ValueError):
                return None
            now = time.time()
            fetched_ts = fetched_at.timestamp()
            if now - fetched_ts > self._ttl:
                return None
        body = body_path.read_bytes()
        return MetadataResponse(
            body=body,
            content_type=str(meta.get("content_type", "")),
            fetched_at_iso=str(meta.get("fetched_at_iso", "")),
            from_cache=True,
        )

    def put(self, provider: str, key: str, response: MetadataResponse) -> Path:
        """Persist ``response`` for ``(provider, key)``. Returns body path."""

        entry_dir = self._entry_dir(provider, key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        body_path = entry_dir / "body.json"
        meta_path = entry_dir / "meta.json"
        body_path.write_bytes(response.body)
        meta_path.write_text(
            json.dumps(
                {
                    "provider": provider,
                    "key": key,
                    "fetched_at_iso": response.fetched_at_iso,
                    "content_type": response.content_type,
                    "byte_len": len(response.body),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return body_path

    def fetch_or_get(
        self,
        provider: str,
        key: str,
        url: str,
        client: MetadataHTTPClient,
        *,
        accept: str = "application/json",
    ) -> MetadataResponse:
        """Cache-aware fetch: hit cache, else GET ``url`` and persist."""

        cached = self.get(provider, key)
        if cached is not None:
            return cached
        response = client.get(url, accept=accept)
        self.put(provider, key, response)
        return response


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "MetadataCache",
    "MetadataHTTPClient",
    "MetadataResponse",
]
