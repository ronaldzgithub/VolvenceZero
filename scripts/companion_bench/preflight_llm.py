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

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".local" / "llm.env"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from proxy_support import configure_companion_bench_proxy  # noqa: E402
from p1_readiness import (  # noqa: E402
    P1ReadinessError,
    build_run_manifest,
    fingerprint_weights,
    require_accelerator,
    require_commands,
    require_non_qwen_models,
    require_ports_free,
    resolve_weights_root,
    write_run_manifest,
    write_track_fingerprints,
)


def _require_refh_embedder_deps() -> None:
    """Fail before paid scoring if the selected retrieval embedder cannot load."""

    embedder = os.environ.get("REFH_EMBEDDER", "bge-m3").strip() or "bge-m3"
    if embedder == "hashing":
        return
    if embedder != "bge-m3":
        raise P1ReadinessError(
            f"unsupported REFH_EMBEDDER={embedder!r} for P1; expected bge-m3 or hashing"
        )
    if importlib.util.find_spec("sentence_transformers") is None:
        raise P1ReadinessError(
            "REFH_EMBEDDER=bge-m3 requires sentence-transformers; install "
            "the companion-ref-harness embed extra or set REFH_EMBEDDER=hashing "
            "for a dependency-free local smoke run"
        )


def _load_env(*, required: bool) -> None:
    if not ENV_FILE.exists():
        if required:
            raise SystemExit(f"error: {ENV_FILE} not found")
        return
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
    except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"  FAIL {label:<28} model={model:<28} {type(exc).__name__}: {exc}", file=sys.stderr)
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="skip paid API probes")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("VZ_SUBSTRATE_MODEL_ID", ""),
        help="local frozen Qwen model id",
    )
    parser.add_argument(
        "--weights-path",
        default=os.environ.get("VZ_SUBSTRATE_WEIGHTS_PATH", ""),
        help="explicit local directory containing weight files",
    )
    parser.add_argument(
        "--substrate-device",
        default=os.environ.get("VZ_SUBSTRATE_DEVICE", "cuda"),
        help="substrate torch device to require: cuda (default) / mps / cpu",
    )
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "companion-ablation" / "local",
    )
    parser.add_argument("--skip-port-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _load_env(required=not args.offline)
    configure_companion_bench_proxy()
    model_id = args.model_id or os.environ.get("VZ_SUBSTRATE_MODEL_ID", "").strip()
    if not model_id:
        print("error: set VZ_SUBSTRATE_MODEL_ID or pass --model-id", file=sys.stderr)
        return 2

    or_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    user_sim = os.environ.get("ABLATION_USER_SIM_MODEL", "openai/gpt-5-mini")
    perturn = os.environ.get("ABLATION_PERTURN_MODEL", user_sim)
    arc = os.environ.get("ABLATION_ARC_MODEL", "anthropic/claude-3.7-sonnet")

    try:
        require_non_qwen_models(
            (("user-sim", user_sim), ("per-turn judge", perturn), ("arc judge", arc))
        )
        require_commands()
        _require_refh_embedder_deps()
        if not args.skip_port_check:
            require_ports_free()
        gpu_name = require_accelerator(args.substrate_device)
        weights_root = resolve_weights_root(model_id, args.weights_path or None)
        fingerprint = fingerprint_weights(model_id, weights_root)
        manifest = build_run_manifest(
            repo_root=REPO_ROOT,
            substrate=fingerprint,
            user_sim_model=user_sim,
            perturn_model=perturn,
            arc_model=arc,
        )
        write_run_manifest(manifest, args.artifact_dir / "run_manifest.json")
        write_track_fingerprints(
            fingerprint,
            args.artifact_dir,
            (
                "raw",
                "ref-harness",
                "memory-only",
                "rag",
                "camel",
                "volvence-cold",
                "volvence",
                "pe-off",
                "eta-off",
                "active-learning-off",
                "lora-adapter",
            ),
        )
    except (P1ReadinessError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        "local preflight: ALL OK "
        f"(gpu={gpu_name!r}, weights_sha256={fingerprint.weights_sha256})"
    )
    if args.offline:
        print("API preflight: SKIPPED (--offline)")
        return 0

    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not or_key:
        print("error: OPENROUTER_API_KEY empty in .local/llm.env", file=sys.stderr)
        return 2

    print("API preflight: probing cross-family user-sim and judges...")
    or_headers = {"HTTP-Referer": "https://volvence.zero", "X-Title": "companion-ablation"}
    results = [
        _probe(label="OpenRouter user-sim/perturn", base_url=or_base, api_key=or_key, model=user_sim, extra_headers=or_headers),
        _probe(label="OpenRouter per-turn judge", base_url=or_base, api_key=or_key, model=perturn, extra_headers=or_headers),
        _probe(label="OpenRouter arc judge", base_url=or_base, api_key=or_key, model=arc, extra_headers=or_headers),
    ]
    ok = all(results)
    print(f"\nAPI preflight: {'ALL OK' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
