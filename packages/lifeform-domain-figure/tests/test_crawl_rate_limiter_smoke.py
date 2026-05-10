"""Smoke tests for the L0 token-bucket rate limiter."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.crawl.rate_limiter import TokenBucketRateLimiter


class _ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_burst_consumed_then_blocks() -> None:
    clock = _ManualClock()
    limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst=3, clock=clock)
    for _ in range(3):
        ok, sleep_s = limiter.acquire("host-a")
        assert ok
        assert sleep_s == 0.0
    ok, sleep_s = limiter.acquire("host-a")
    assert not ok
    assert sleep_s == pytest.approx(1.0, rel=1e-6)


def test_refill_after_clock_advance() -> None:
    clock = _ManualClock()
    limiter = TokenBucketRateLimiter(rate_per_second=2.0, burst=1, clock=clock)
    ok, _ = limiter.acquire("host-a")
    assert ok
    ok, sleep_s = limiter.acquire("host-a")
    assert not ok
    assert sleep_s == pytest.approx(0.5, rel=1e-6)
    clock.advance(0.5)
    ok, _ = limiter.acquire("host-a")
    assert ok


def test_per_host_isolation() -> None:
    clock = _ManualClock()
    limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst=1, clock=clock)
    ok_a, _ = limiter.acquire("host-a")
    ok_b, _ = limiter.acquire("host-b")
    assert ok_a and ok_b
    ok_a2, _ = limiter.acquire("host-a")
    assert not ok_a2


def test_known_hosts_ordering() -> None:
    clock = _ManualClock()
    limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst=2, clock=clock)
    limiter.acquire("host-c")
    limiter.acquire("host-a")
    limiter.acquire("host-b")
    assert limiter.known_hosts() == ("host-c", "host-a", "host-b")


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(rate_per_second=0)
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(burst=0)
    limiter = TokenBucketRateLimiter()
    with pytest.raises(ValueError):
        limiter.acquire("")
