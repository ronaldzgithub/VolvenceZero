#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Connectivity preflight for the hosted Companion Bench ablation.

Loads ``.local/llm.env`` and sends ONE tiny chat-completion to each endpoint —
the DashScope Qwen substrate and the two OpenRouter judge models — to confirm
the keys work and the model slugs resolve BEFORE spending on a full run.

Never prints key values. Prints only endpoint / model / status / a short reply
snippet. Exit 0 iff all probes succeed.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".local" / "llm.env"


def _load_env() -> None:
    if not ENV_FILE.exists():
        raise SystemExit(f"error: {ENV_FILE} not found")
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _probe(*, label: str, base_url: str, api_key: str, model: str,
           extra_headers: dict[str, str] | None = None) -> bool:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(extra_headers or {})
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        dt = time.time() - t0
        text = ""
        choices = raw.get("choices") or []
        if choices:
            text = str(choices[0].get("message", {}).get("content", ""))[:40]
        print(f"  OK   {label:<28} model={model:<28} {dt:5.1f}s reply={text!r}")
        return True
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        print(f"  FAIL {label:<28} model={model:<28} HTTP {exc.code}: {detail}", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001 -- surface any connectivity failure
        print(f"  FAIL {label:<28} model={model:<28} {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    _load_env()
    dash_key = os.environ.get("PROTOCOL_LLM_API_KEY", "").strip()
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    dash_base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    or_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    substrate_model = os.environ.get("ABLATION_SUBSTRATE_MODEL", "qwen-turbo")
    user_sim = os.environ.get("ABLATION_USER_SIM_MODEL", "openai/gpt-5-mini")
    arc = os.environ.get("ABLATION_ARC_MODEL", "anthropic/claude-3.7-sonnet")

    if not dash_key:
        print("error: PROTOCOL_LLM_API_KEY empty in .local/llm.env", file=sys.stderr)
        return 2
    if not or_key:
        print("error: OPENROUTER_API_KEY empty in .local/llm.env", file=sys.stderr)
        return 2

    print("preflight: probing substrate + judge endpoints (1 tiny call each)...")
    or_headers = {"HTTP-Referer": "https://volvence.zero", "X-Title": "companion-ablation"}
    results = [
        _probe(label="DashScope substrate", base_url=dash_base, api_key=dash_key, model=substrate_model),
        _probe(label="OpenRouter user-sim/perturn", base_url=or_base, api_key=or_key, model=user_sim, extra_headers=or_headers),
        _probe(label="OpenRouter arc judge", base_url=or_base, api_key=or_key, model=arc, extra_headers=or_headers),
    ]
    ok = all(results)
    print(f"\npreflight: {'ALL OK' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
