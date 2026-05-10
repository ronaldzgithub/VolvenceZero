"""Per-host robots.txt registry for the L0 crawler.

Behaviour:

* Per-host cache with TTL (default 1 hour).
* Cache fetch goes through :class:`BaseHTTPClient` so robots.txt
  fetches honour the same SSRF allowlist as document fetches.
* On any fetch failure (network error, 4xx other than 404, body cap,
  scope rejection) the registry is **fail-closed**: it returns
  ``False`` from :meth:`is_allowed` for that host until the cache
  entry expires. This is the conservative posture: when we cannot
  read the rules, we do not crawl.
* A 404 robots.txt is treated as "no rules" -> ``True`` for any URL
  (per RFC 9309 Section 2.3).

The user_agent for matching is taken from the
:class:`ScopePolicy.user_agent` attached to the HTTP client, so the
crawler identifies itself consistently.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib import robotparser
from urllib.parse import urlparse

from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    BodyTooLarge,
    FetchError,
    HTTPResponse,
    NOT_MODIFIED,
    ScopeRejection,
)


DEFAULT_TTL_SECONDS = 3600


@dataclass
class _RobotsCacheEntry:
    parser: robotparser.RobotFileParser | None
    fetched_at: float
    fail_closed: bool
    reason: str


class RobotsRegistry:
    """Per-host robots.txt cache + check.

    Not threadsafe; the L0 scheduler is single-threaded.
    """

    def __init__(
        self,
        *,
        http_client: BaseHTTPClient,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                f"RobotsRegistry.ttl_seconds must be > 0; got {ttl_seconds!r}"
            )
        self._http_client = http_client
        self._ttl = ttl_seconds
        self._clock = clock or time.monotonic
        self._cache: dict[str, _RobotsCacheEntry] = {}

    def _entry_for(self, host_origin: str, scheme: str) -> _RobotsCacheEntry:
        now = self._clock()
        cached = self._cache.get(host_origin)
        if cached is not None and (now - cached.fetched_at) < self._ttl:
            return cached
        robots_url = f"{scheme}://{host_origin}/robots.txt"
        parser: robotparser.RobotFileParser | None = None
        fail_closed = False
        reason = ""
        try:
            response = self._http_client.get(robots_url, accept="text/plain")
        except ScopeRejection as exc:
            fail_closed = True
            reason = f"scope rejection on robots.txt: {exc}"
        except BodyTooLarge as exc:
            fail_closed = True
            reason = f"robots.txt body cap exceeded: {exc}"
        except FetchError as exc:
            message = str(exc)
            if "http_status=404" in message:
                parser = robotparser.RobotFileParser()
                parser.parse([])
            else:
                fail_closed = True
                reason = f"robots.txt fetch failed: {message}"
        else:
            if response is NOT_MODIFIED:
                if cached is not None and cached.parser is not None:
                    parser = cached.parser
                else:
                    fail_closed = True
                    reason = "robots.txt 304 with no prior parser cache"
            else:
                assert isinstance(response, HTTPResponse)
                try:
                    text = response.body.decode("utf-8", errors="replace")
                except Exception as exc:
                    fail_closed = True
                    reason = f"robots.txt decode failure: {exc}"
                else:
                    parser = robotparser.RobotFileParser()
                    parser.parse(text.splitlines())
        entry = _RobotsCacheEntry(
            parser=parser,
            fetched_at=now,
            fail_closed=fail_closed,
            reason=reason,
        )
        self._cache[host_origin] = entry
        return entry

    def is_allowed(self, url: str) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for ``url`` against cached robots.

        ``reason`` is only populated when ``allowed`` is ``False``,
        capturing why (fail-closed cache state OR explicit
        ``Disallow`` rule match).
        """

        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        if not host or scheme not in ("http", "https"):
            return False, f"invalid url for robots check: {url!r}"
        if parsed.port:
            host_origin = f"{host}:{parsed.port}"
        else:
            host_origin = host
        entry = self._entry_for(host_origin, scheme)
        if entry.fail_closed:
            return False, entry.reason
        if entry.parser is None:
            return False, "robots.txt parser unavailable (no cache, no fetch)"
        user_agent = self._http_client.scope.user_agent
        allowed = entry.parser.can_fetch(user_agent, url)
        if not allowed:
            return False, f"robots.txt disallow for user_agent={user_agent!r} url={url!r}"
        return True, ""

    def known_hosts(self) -> tuple[str, ...]:
        return tuple(self._cache.keys())


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "RobotsRegistry",
]
