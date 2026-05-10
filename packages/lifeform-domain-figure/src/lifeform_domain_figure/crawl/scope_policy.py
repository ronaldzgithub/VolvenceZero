"""SSRF-bound scope policy for the L0 crawler.

The scope policy is the FIRST of five SSRF gates (the rest live in
:class:`BaseHTTPClient`). It enumerates an explicit allowlist of
hosts and per-host path prefixes; any URL whose host or path is not
in the allowlist is rejected before a socket is opened.

Each host carries a set of :class:`ScopeRole` tags (debt #26 closure)
declaring whether the host is for corpus byte download, metadata
JSON / HTML fetch, or both. Callers may pass ``required_role`` to
:meth:`is_in_scope` to require role-matched access; this prevents L0
corpus fetchers from accidentally hitting metadata endpoints, and
metadata clients from accidentally hitting corpus archives.

Only the five known figure-vertical archive hosts (CORPUS_FETCH role)
and four metadata API hosts (METADATA_FETCH role) are baked into
:data:`DEFAULT_HOSTS`. Adding more hosts is a deliberate config
change — operators construct a fresh :class:`ScopePolicy` with a
wider allowlist; the curator review step makes that change visible.

The policy also caps per-response body size and per-host page count
so a misconfigured archive cannot exhaust local disk or hammer one
host indefinitely. ``user_agent`` must be supplied so the crawler
identifies itself in HTTP traffic and so robots.txt rule matching
works against a known agent string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse


class ScopeRole(str, Enum):
    """Closed vocabulary of host roles in the figure vertical's scope policy.

    A host MAY carry multiple roles (e.g., archive.org could be
    tagged both as CORPUS_FETCH and METADATA_FETCH if we ever needed
    to call its metadata API for OCR file listing AND fetch corpus
    bytes from the same domain). The role check is "required role
    is in the host's role set"; absence of role on the host means
    role-checked access is denied.
    """

    CORPUS_FETCH = "corpus_fetch"
    METADATA_FETCH = "metadata_fetch"


DEFAULT_CORPUS_HOSTS: frozenset[str] = frozenset(
    {
        "einsteinpapers.press.princeton.edu",
        "en.wikisource.org",
        "de.wikisource.org",
        "fr.wikisource.org",
        "zh.wikisource.org",
        "www.gutenberg.org",
        "gutenberg.org",
        "archive.org",
        "ia801.us.archive.org",
        "ia902.us.archive.org",
        "ctext.org",
    }
)
"""Baked allowlist of figure-vertical CORPUS_FETCH hosts."""

DEFAULT_METADATA_HOSTS: frozenset[str] = frozenset(
    {
        "api.openalex.org",
        "www.wikidata.org",
        "query.wikidata.org",
        "api.crossref.org",
        "plato.stanford.edu",
    }
)
"""Baked allowlist of figure-vertical METADATA_FETCH hosts (debt #26)."""

DEFAULT_HOSTS: frozenset[str] = DEFAULT_CORPUS_HOSTS | DEFAULT_METADATA_HOSTS
"""Union of corpus + metadata hosts for callers needing both roles.

Kept under the historical :data:`DEFAULT_HOSTS` name so existing
``default_scope_policy()`` callers see no behavioural change at the
allowlist level. Role-checked access still requires
:attr:`ScopePolicy.host_roles` to declare the role per host.
"""

ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

DEFAULT_USER_AGENT = "VolvenceZero-FigureCrawler/0.1 (+contact: ops@volvence-zero.local)"

DEFAULT_MAX_BODY_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PAGES_PER_HOST = 500


def _empty_path_prefixes() -> dict[str, tuple[str, ...]]:
    return {}


def _empty_host_roles() -> dict[str, frozenset[ScopeRole]]:
    return {}


@dataclass(frozen=True)
class ScopePolicy:
    """Immutable scope policy consulted before every fetch.

    * ``allowed_hosts`` — hostnames the crawler / metadata clients may
      contact. Empty tuple is forbidden (operator must consciously
      enumerate hosts).
    * ``user_agent`` — passed in HTTP ``User-Agent`` header and used
      by :class:`RobotsRegistry` for rule matching.
    * ``allowed_path_prefixes`` — optional per-host path prefix
      allowlist (mapping host -> tuple of prefixes that must match the
      URL path). Hosts absent from this mapping default to ``("/",)``
      i.e. all paths allowed.
    * ``host_roles`` — optional per-host :class:`ScopeRole` tags.
      Empty default = no role-checked access (legacy behaviour);
      callers wanting role enforcement populate this dict and pass
      ``required_role`` to :meth:`is_in_scope`.
    * ``max_pages_per_host`` — hard cap on per-host visits per
      :class:`CrawlScheduler` run.
    * ``max_body_bytes`` — per-response body size cap.
    * ``incremental`` — whether the HTTP client honors
      ``If-None-Match`` / ``If-Modified-Since``.
    """

    allowed_hosts: frozenset[str]
    user_agent: str
    allowed_path_prefixes: dict[str, tuple[str, ...]] = field(
        default_factory=_empty_path_prefixes
    )
    host_roles: dict[str, frozenset[ScopeRole]] = field(
        default_factory=_empty_host_roles
    )
    max_pages_per_host: int = DEFAULT_MAX_PAGES_PER_HOST
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    incremental: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_hosts, frozenset):
            raise TypeError(
                "ScopePolicy.allowed_hosts must be a frozenset[str]; "
                f"got {type(self.allowed_hosts).__name__}"
            )
        if not self.allowed_hosts:
            raise ValueError(
                "ScopePolicy.allowed_hosts must be non-empty; refusing to "
                "construct a policy that allows zero hosts (which would also "
                "deny everything; that is a misconfiguration)."
            )
        for host in self.allowed_hosts:
            if not isinstance(host, str) or not host.strip():
                raise ValueError(
                    f"ScopePolicy.allowed_hosts contains non-string or blank entry: {host!r}"
                )
        if not isinstance(self.user_agent, str) or not self.user_agent.strip():
            raise ValueError("ScopePolicy.user_agent must be a non-empty string")
        if self.max_pages_per_host <= 0:
            raise ValueError(
                f"ScopePolicy.max_pages_per_host must be > 0; got {self.max_pages_per_host!r}"
            )
        if self.max_body_bytes <= 0:
            raise ValueError(
                f"ScopePolicy.max_body_bytes must be > 0; got {self.max_body_bytes!r}"
            )
        for host, prefixes in self.allowed_path_prefixes.items():
            if host not in self.allowed_hosts:
                raise ValueError(
                    f"ScopePolicy.allowed_path_prefixes references host "
                    f"{host!r} not present in allowed_hosts"
                )
            if not isinstance(prefixes, tuple) or not prefixes:
                raise ValueError(
                    f"ScopePolicy.allowed_path_prefixes[{host!r}] must be a "
                    f"non-empty tuple of path prefixes; got {prefixes!r}"
                )
        for host, roles in self.host_roles.items():
            if host not in self.allowed_hosts:
                raise ValueError(
                    f"ScopePolicy.host_roles references host {host!r} not "
                    f"present in allowed_hosts"
                )
            if not isinstance(roles, frozenset) or not roles:
                raise ValueError(
                    f"ScopePolicy.host_roles[{host!r}] must be a non-empty "
                    f"frozenset[ScopeRole]; got {roles!r}"
                )

    def scheme_allowed(self, url: str) -> bool:
        """Return True iff ``url`` uses a permitted scheme."""

        parsed = urlparse(url)
        return parsed.scheme.lower() in ALLOWED_SCHEMES

    def is_in_scope(
        self, url: str, *, required_role: ScopeRole | None = None
    ) -> bool:
        """Return True iff ``url`` passes scheme + host + path-prefix (+ role) gates.

        When ``required_role`` is supplied, the host MUST also carry
        that role in :attr:`host_roles`; absence-of-role-mapping for
        the host means role-required access is denied.
        """

        parsed = urlparse(url)
        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            return False
        host = (parsed.hostname or "").lower()
        if not host or host not in self.allowed_hosts:
            return False
        prefixes = self.allowed_path_prefixes.get(host, ("/",))
        path = parsed.path or "/"
        if not any(path.startswith(prefix) for prefix in prefixes):
            return False
        if required_role is not None:
            roles = self.host_roles.get(host, frozenset())
            if required_role not in roles:
                return False
        return True

    def reason_out_of_scope(
        self, url: str, *, required_role: ScopeRole | None = None
    ) -> str:
        """Return a human-readable reason ``url`` was rejected.

        Returns ``""`` when ``url`` is in scope. Used by the scheduler
        to populate :attr:`CrawlResult.error` for SKIPPED_SCOPE
        outcomes.
        """

        parsed = urlparse(url)
        if parsed.scheme.lower() not in ALLOWED_SCHEMES:
            return f"scheme={parsed.scheme!r} not in {sorted(ALLOWED_SCHEMES)!r}"
        host = (parsed.hostname or "").lower()
        if not host:
            return "missing host"
        if host not in self.allowed_hosts:
            return f"host={host!r} not in scope.allowed_hosts"
        prefixes = self.allowed_path_prefixes.get(host, ("/",))
        path = parsed.path or "/"
        if not any(path.startswith(prefix) for prefix in prefixes):
            return f"path={path!r} not in scope.allowed_path_prefixes[{host!r}]={prefixes!r}"
        if required_role is not None:
            roles = self.host_roles.get(host, frozenset())
            if required_role not in roles:
                return (
                    f"host={host!r} role mapping={sorted(r.value for r in roles)!r} "
                    f"missing required_role={required_role.value!r}"
                )
        return ""


def _corpus_role_map(hosts: frozenset[str]) -> dict[str, frozenset[ScopeRole]]:
    return {h: frozenset({ScopeRole.CORPUS_FETCH}) for h in hosts}


def _metadata_role_map(hosts: frozenset[str]) -> dict[str, frozenset[ScopeRole]]:
    return {h: frozenset({ScopeRole.METADATA_FETCH}) for h in hosts}


def default_scope_policy(user_agent: str = DEFAULT_USER_AGENT) -> ScopePolicy:
    """Construct a :class:`ScopePolicy` for L0 corpus crawl (CORPUS_FETCH role)."""

    return ScopePolicy(
        allowed_hosts=DEFAULT_CORPUS_HOSTS,
        user_agent=user_agent,
        host_roles=_corpus_role_map(DEFAULT_CORPUS_HOSTS),
    )


def default_metadata_scope_policy(
    user_agent: str = DEFAULT_USER_AGENT,
) -> ScopePolicy:
    """Construct a :class:`ScopePolicy` for D4 metadata clients (METADATA_FETCH role)."""

    return ScopePolicy(
        allowed_hosts=DEFAULT_METADATA_HOSTS,
        user_agent=user_agent,
        host_roles=_metadata_role_map(DEFAULT_METADATA_HOSTS),
    )


def default_combined_scope_policy(
    user_agent: str = DEFAULT_USER_AGENT,
) -> ScopePolicy:
    """Construct a :class:`ScopePolicy` carrying both corpus + metadata roles.

    Useful for callers (e.g., L2 verifier driver script) that need to
    invoke both L0 corpus fetchers and D4 metadata clients with one
    HTTP client. Each host still carries only its own role; role
    enforcement remains intact.
    """

    return ScopePolicy(
        allowed_hosts=DEFAULT_HOSTS,
        user_agent=user_agent,
        host_roles=_corpus_role_map(DEFAULT_CORPUS_HOSTS)
        | _metadata_role_map(DEFAULT_METADATA_HOSTS),
    )


__all__ = [
    "ALLOWED_SCHEMES",
    "DEFAULT_CORPUS_HOSTS",
    "DEFAULT_HOSTS",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_PAGES_PER_HOST",
    "DEFAULT_METADATA_HOSTS",
    "DEFAULT_USER_AGENT",
    "ScopePolicy",
    "ScopeRole",
    "default_combined_scope_policy",
    "default_metadata_scope_policy",
    "default_scope_policy",
]
