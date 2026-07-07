#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Proxy bootstrap for CompanionBench scripts.

The Windows bench box often reaches OpenRouter / DashScope through a local
TUN/proxy client. urllib honours the standard ``*_PROXY`` variables, so the
scripts only need to set them once before constructing clients or subprocesses.
"""

from __future__ import annotations

import os
import socket
import sys

DEFAULT_PROXY_URL = "http://127.0.0.1:7897"
_LOCAL_NO_PROXY = "127.0.0.1,localhost,::1"


def _tcp_open(host: str, port: int, *, timeout_s: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _merge_no_proxy(existing: str) -> str:
    values = [part.strip() for part in existing.split(",") if part.strip()]
    for part in _LOCAL_NO_PROXY.split(","):
        if part not in values:
            values.append(part)
    return ",".join(values)


def configure_companion_bench_proxy() -> str:
    """Configure proxy env vars and return the proxy URL, or ``""``.

    Precedence:
    1. ``COMPANION_BENCH_PROXY_URL`` explicitly chooses a proxy.
    2. Existing ``HTTPS_PROXY`` / ``HTTP_PROXY`` is respected.
    3. If ``COMPANION_BENCH_AUTO_PROXY`` is not disabled and
       ``127.0.0.1:7897`` is reachable, use it automatically.
    """

    explicit = os.environ.get("COMPANION_BENCH_PROXY_URL", "").strip()
    existing = (
        os.environ.get("HTTPS_PROXY", "").strip()
        or os.environ.get("https_proxy", "").strip()
        or os.environ.get("HTTP_PROXY", "").strip()
        or os.environ.get("http_proxy", "").strip()
    )
    proxy_url = explicit or existing
    if not proxy_url and os.environ.get("COMPANION_BENCH_AUTO_PROXY", "1").strip() not in {
        "0",
        "false",
        "False",
    }:
        if _tcp_open("127.0.0.1", 7897):
            proxy_url = DEFAULT_PROXY_URL
    if not proxy_url:
        return ""

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ[key] = proxy_url
    no_proxy = _merge_no_proxy(os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", ""))
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    print(f"[proxy] using {proxy_url} (NO_PROXY={no_proxy})", file=sys.stderr, flush=True)
    return proxy_url


__all__ = ["configure_companion_bench_proxy"]
