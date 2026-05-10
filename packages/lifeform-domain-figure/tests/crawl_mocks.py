"""Shared HTTP mocking helpers for the L0 crawler test suite.

Builds a fake :class:`requests.Session` whose ``get`` returns a
pre-canned response (status / headers / body). Used in place of any
real HTTP traffic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class FakeResponse:
    """Minimal stand-in for :class:`requests.Response` used by tests."""

    status_code: int
    headers: dict
    body: bytes
    url: str

    def iter_content(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]

    def close(self) -> None:
        return None


def make_response(
    *,
    status_code: int = 200,
    body: bytes = b"",
    content_type: str = "application/octet-stream",
    etag: str = "",
    last_modified: str = "",
    location: str = "",
    url: str = "",
) -> FakeResponse:
    headers: dict[str, str] = {"Content-Type": content_type}
    if etag:
        headers["ETag"] = etag
    if last_modified:
        headers["Last-Modified"] = last_modified
    if location:
        headers["Location"] = location
    return FakeResponse(
        status_code=status_code,
        headers=headers,
        body=body,
        url=url,
    )


class FakeSession:
    """Fake :class:`requests.Session` whose ``get`` consults a callable.

    ``handler(url, headers, kwargs)`` returns a :class:`FakeResponse`
    or raises ``requests.RequestException`` to simulate a network
    error.
    """

    def __init__(self, handler: Callable[..., FakeResponse]) -> None:
        self._handler = handler
        self.calls: list[tuple[str, dict]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict | None = None,
        timeout: float | None = None,
        stream: bool = True,
        allow_redirects: bool = True,
    ) -> FakeResponse:
        self.calls.append((url, dict(headers or {})))
        return self._handler(url, dict(headers or {}))

    def close(self) -> None:
        return None


__all__ = ["FakeResponse", "FakeSession", "make_response"]
