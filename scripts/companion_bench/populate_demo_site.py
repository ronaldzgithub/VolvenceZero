#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Populate the public site with full demo data using the deterministic-fake pipeline.

This is the placeholder substitute for the real
``score_reference_systems.py`` run when API keys / API budget are not yet
available. It runs the deterministic-fake pipeline (no network, no API
spend) for ~8 mock submissions across the full 24 public scenarios, then
pipes the artifacts through ``build_site.py`` so every page on the static
site (leaderboard, per-submission detail, pairwise compare, scenarios)
has fully-populated content shaped exactly like a real release-tier run.

The resulting ``aggregate_results.json`` is patched with ``demo: true`` so
the site renders the "Demo data" banner. Real reference-system scoring
uses ``score_reference_systems.py`` with real API keys and produces
``demo: false``.

Usage::

    python scripts/companion_bench/populate_demo_site.py
    python scripts/companion_bench/populate_demo_site.py --site-dir site --keep-artifacts
"""

from __future__ import annotations

import argparse
import importlib.resources as res
import json
import logging
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Mapping

from companion_bench.spec import load_scenarios_dir
from companion_bench.submission import (
    SubmissionAttestation,
    SubmissionManifest,
    dry_run_with_fakes,
    write_submission_summary,
)
from companion_bench.sut_client import EchoFakeSUTClient, SUTResponse
from companion_bench.user_simulator import DeterministicFakeUtteranceClient


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LOG = logging.getLogger("populate_demo_site")


# Eight mock submissions covering all three categories. Echo prefix +
# truncation knobs deliberately spread the scores so the leaderboard /
# pairwise viewer have visibly distinct rows.
DEMO_SUBMISSIONS: tuple[dict, ...] = (
    {
        "submission_id": "demo-2026q2-openai-gpt-5",
        "system_name": "OpenAI GPT-5 (demo)",
        "model_identifier": "openai/gpt-5",
        "leaderboard_category": "closed-api",
        "echo_prefix": "I hear you said: ",
        "noise_seed": 0,
    },
    {
        "submission_id": "demo-2026q2-anthropic-claude-opus-4-6",
        "system_name": "Anthropic Claude Opus 4.6 (demo)",
        "model_identifier": "anthropic/claude-opus-4.6",
        "leaderboard_category": "closed-api",
        "echo_prefix": "Thanks for sharing that. ",
        "noise_seed": 1,
    },
    {
        "submission_id": "demo-2026q2-google-gemini-3-pro",
        "system_name": "Google Gemini 3 Pro (demo)",
        "model_identifier": "google/gemini-3-pro",
        "leaderboard_category": "closed-api",
        "echo_prefix": "Acknowledged. ",
        "noise_seed": 2,
    },
    {
        "submission_id": "demo-2026q2-deepseek-v3",
        "system_name": "DeepSeek V3 (demo)",
        "model_identifier": "deepseek/deepseek-v3",
        "leaderboard_category": "open-weight",
        "echo_prefix": "Got it. You said: ",
        "noise_seed": 3,
    },
    {
        "submission_id": "demo-2026q2-qwen-2-5-72b",
        "system_name": "Qwen 2.5 72B Instruct (demo)",
        "model_identifier": "qwen/qwen2.5-72b-instruct",
        "leaderboard_category": "open-weight",
        "echo_prefix": "Reading you. ",
        "noise_seed": 4,
    },
    {
        "submission_id": "demo-2026q2-mistral-large",
        "system_name": "Mistral Large (demo)",
        "model_identifier": "mistral/mistral-large",
        "leaderboard_category": "closed-api",
        "echo_prefix": "Of course. ",
        "noise_seed": 5,
    },
    {
        "submission_id": "demo-2026q2-lifeform-companion",
        "system_name": "VolvenceZero Lifeform companion (demo)",
        "model_identifier": "lifeform-companion",
        "leaderboard_category": "bespoke",
        "echo_prefix": "I remember. ",
        "noise_seed": 6,
    },
    {
        "submission_id": "demo-2026q2-lifeform-raw",
        "system_name": "VolvenceZero Lifeform raw (demo)",
        "model_identifier": "lifeform-raw",
        "leaderboard_category": "bespoke",
        "echo_prefix": "",
        "noise_seed": 7,
    },
)


class _PrefixedEchoSUTClient(EchoFakeSUTClient):
    """EchoFakeSUTClient with a configurable prefix so different submissions
    produce visibly different transcripts (and therefore distinct judge scores).
    """

    def __init__(self, *, model: str, prefix: str) -> None:
        super().__init__(model=model)
        self._prefix = prefix

    def chat(self, *, messages: list[dict[str, str]], session_id: str,
             user_id: str | None, max_tokens: int | None,
             temperature: float | None) -> SUTResponse:
        resp = super().chat(
            messages=messages, session_id=session_id, user_id=user_id,
            max_tokens=max_tokens, temperature=temperature,
        )
        return SUTResponse(
            text=self._prefix + resp.text,
            model_id=resp.model_id,
            response_headers=resp.response_headers,
            usage_prompt_tokens=resp.usage_prompt_tokens,
            usage_completion_tokens=resp.usage_completion_tokens,
            raw=resp.raw,
        )


def _build_manifest(entry: Mapping[str, Any]) -> SubmissionManifest:
    return SubmissionManifest(
        submission_id=entry["submission_id"],
        system_name=entry["system_name"],
        model_identifier=entry["model_identifier"],
        base_url="http://localhost",
        api_key_env="UNSET",
        system_prompt="(demo placeholder)",
        generation_config={"temperature": 0.0, "max_tokens": 512},
        attestation=SubmissionAttestation(True, True, True, True),
        leaderboard_category=entry["leaderboard_category"],
    )


def _run_one(entry: Mapping[str, Any], specs, artifact_dir: pathlib.Path) -> None:
    submission_dir = artifact_dir / entry["submission_id"]
    submission_dir.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(entry)
    sut = _PrefixedEchoSUTClient(model=f"demo/{entry['submission_id']}",
                                  prefix=entry["echo_prefix"])
    user_backend = DeterministicFakeUtteranceClient()
    result = dry_run_with_fakes(
        manifest=manifest,
        specs=specs,
        sut_client=sut,
        user_backend=user_backend,
        paraphrase_seeds=(0,),
        artifact_dir=submission_dir,
    )
    write_submission_summary(result, submission_dir / "summary.json")
    _LOG.info("[%s] arcs=%d", entry["submission_id"], len(result.arc_bundles))


def _patch_demo_flag(site_dir: pathlib.Path) -> None:
    """Mark the leaderboard payload as demo so the site shows the banner."""
    agg_path = site_dir / "data" / "aggregate_results.json"
    if not agg_path.exists():
        return
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    data["demo"] = True
    agg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    pair_path = site_dir / "data" / "pairwise.json"
    if pair_path.exists():
        d = json.loads(pair_path.read_text(encoding="utf-8"))
        d["demo"] = True
        pair_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="populate_demo_site")
    p.add_argument("--site-dir", type=pathlib.Path,
                   default=REPO_ROOT / "site")
    p.add_argument("--artifact-dir", type=pathlib.Path,
                   default=REPO_ROOT / "artifacts" / "companion-bench" / "demo",
                   help="Where to put the demo bundles. Cleared at start.")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="Do NOT delete the artifact dir at start (useful for debugging).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.keep_artifacts and args.artifact_dir.exists():
        shutil.rmtree(args.artifact_dir)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = load_scenarios_dir(public_dir, include_held_out=False)
    _LOG.info("loaded %d public scenarios", len(specs))

    for entry in DEMO_SUBMISSIONS:
        _run_one(entry, specs, args.artifact_dir)

    # Pipe through build_site so all the site/data/ JSONs regen.
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "companion_bench"))
    import build_site  # noqa: WPS433
    rc = build_site.main([
        "--artifact-dir", str(args.artifact_dir),
        "--site-dir", str(args.site_dir),
    ])
    if rc != 0:
        return rc
    _patch_demo_flag(args.site_dir)
    _LOG.info("done — site demo data populated under %s", args.site_dir / "data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
