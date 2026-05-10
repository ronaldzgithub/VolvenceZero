"""Per-host token-bucket rate limiter for the L0 crawler.

Token bucket math:

* Each host has a bucket of capacity ``burst`` tokens that refills at
  ``rate_per_second`` tokens/second.
* :meth:`acquire` consumes one token if available and returns
  ``(True, 0.0)``; otherwise it returns
  ``(False, sleep_seconds)`` where ``sleep_seconds`` is the wait
  required for one token to refill. The caller decides whether to
  sleep + retry or to defer the request.

The default ``0.5`` requests/second + ``burst=5`` is intentionally
conservative for archive-friendly behaviour (one request every two
seconds, with a small initial burst). Operators can pass a custom
``rate_per_second`` / ``burst`` per :class:`ScopePolicy` rollout.

A ``clock`` callable is injectable so the unit tests can drive
deterministic refill math without sleeping; default is
``time.monotonic``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


DEFAULT_RATE_PER_SECOND = 0.5
DEFAULT_BURST = 5


@dataclass
class _BucketState:
    tokens: float
    last_refill_ts: float


class TokenBucketRateLimiter:
    """Per-host token-bucket rate limiter.

    Not threadsafe; callers using one instance from multiple threads
    must wrap their own lock. The L0 scheduler is single-threaded so
    this matches the consumer.
    """

    def __init__(
        self,
        *,
        rate_per_second: float = DEFAULT_RATE_PER_SECOND,
        burst: int = DEFAULT_BURST,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if rate_per_second <= 0.0:
            raise ValueError(
                f"TokenBucketRateLimiter.rate_per_second must be > 0; got {rate_per_second!r}"
            )
        if burst <= 0:
            raise ValueError(
                f"TokenBucketRateLimiter.burst must be > 0; got {burst!r}"
            )
        self._rate = rate_per_second
        self._burst = burst
        self._clock = clock or time.monotonic
        self._buckets: dict[str, _BucketState] = {}

    def _refill(self, host: str, now: float) -> _BucketState:
        bucket = self._buckets.get(host)
        if bucket is None:
            bucket = _BucketState(tokens=float(self._burst), last_refill_ts=now)
            self._buckets[host] = bucket
            return bucket
        elapsed = max(0.0, now - bucket.last_refill_ts)
        if elapsed > 0.0:
            bucket.tokens = min(float(self._burst), bucket.tokens + elapsed * self._rate)
            bucket.last_refill_ts = now
        return bucket

    def acquire(self, host: str) -> tuple[bool, float]:
        """Try to consume one token for ``host``.

        Returns ``(True, 0.0)`` on success and ``(False, sleep_s)``
        when the caller must wait that many seconds for the next
        token. The wait calculation uses the configured rate; bucket
        state is NOT mutated on a False return.
        """

        if not isinstance(host, str) or not host.strip():
            raise ValueError("TokenBucketRateLimiter.acquire: host must be a non-empty string")
        now = self._clock()
        bucket = self._refill(host, now)
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0.0
        deficit = 1.0 - bucket.tokens
        sleep_s = deficit / self._rate
        return False, sleep_s

    def known_hosts(self) -> tuple[str, ...]:
        """Return the hosts the limiter has seen, in insertion order."""

        return tuple(self._buckets.keys())


__all__ = [
    "DEFAULT_BURST",
    "DEFAULT_RATE_PER_SECOND",
    "TokenBucketRateLimiter",
]
