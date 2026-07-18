# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""companion-trajgen CLI.

``companion-trajgen generate`` runs the batch pipeline over the Companion
Bench public scenario set and writes canonical trajectory JSON + a manifest
under ``--out-dir`` (``train/`` and ``val/`` subdirectories, split by whole
scenario family).
"""

from __future__ import annotations

import argparse
import os
import sys

from companion_trajgen.pipeline import (
    DEFAULT_VAL_FAMILIES,
    generate_dataset,
    summarize_result,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="companion-trajgen",
        description="Synthetic labelled-trajectory generation for the "
        "Relationship Representation Standard.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="batch-generate trajectories from public scenarios"
    )
    generate.add_argument("--out-dir", required=True, help="output directory")
    generate.add_argument(
        "--mode",
        choices=("fsm", "llm"),
        default="fsm",
        help="fsm = deterministic zero-cost; llm = real simulator + SUT",
    )
    generate.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="paraphrase seeds per scenario (default: scenario's own count)",
    )
    generate.add_argument(
        "--val-families",
        default=",".join(DEFAULT_VAL_FAMILIES),
        help="comma-separated scenario families held for validation "
        f"(default: {','.join(DEFAULT_VAL_FAMILIES)})",
    )
    # llm-mode wiring (same procurement conventions as companion-bench runs)
    generate.add_argument("--sut-base-url", default="", help="[llm] SUT base URL")
    generate.add_argument("--sut-model", default="", help="[llm] SUT model id")
    generate.add_argument(
        "--sut-api-key-env",
        default="TRAJGEN_SUT_API_KEY",
        help="[llm] env var holding the SUT API key",
    )
    generate.add_argument("--sim-base-url", default="", help="[llm] simulator base URL")
    generate.add_argument("--sim-model", default="", help="[llm] simulator model id")
    generate.add_argument(
        "--sim-api-key-env",
        default="TRAJGEN_SIM_API_KEY",
        help="[llm] env var holding the simulator API key",
    )
    return parser


def _require_llm_arg(value: str, flag: str) -> str:
    if not value.strip():
        raise SystemExit(f"--mode llm requires {flag}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "generate":
        sut_client = None
        user_backend = None
        user_simulator_model = "deterministic-fake"
        if args.mode == "llm":
            from companion_bench.sut_client import OpenAIChatClient
            from companion_bench.user_simulator import OpenAIUtteranceClient

            sut_base_url = _require_llm_arg(args.sut_base_url, "--sut-base-url")
            sut_model = _require_llm_arg(args.sut_model, "--sut-model")
            sim_base_url = _require_llm_arg(args.sim_base_url, "--sim-base-url")
            sim_model = _require_llm_arg(args.sim_model, "--sim-model")
            sut_api_key = os.environ.get(args.sut_api_key_env, "")
            sim_api_key = os.environ.get(args.sim_api_key_env, "")
            if not sut_api_key or not sim_api_key:
                raise SystemExit(
                    f"--mode llm requires ${args.sut_api_key_env} and "
                    f"${args.sim_api_key_env} to be set"
                )
            sut_client = OpenAIChatClient(
                base_url=sut_base_url, api_key=sut_api_key, model=sut_model
            )
            user_backend = OpenAIUtteranceClient(
                base_url=sim_base_url, api_key=sim_api_key, model=sim_model
            )
            user_simulator_model = sim_model

        result = generate_dataset(
            out_dir=args.out_dir,
            mode=args.mode,
            seeds_per_scenario=args.seeds,
            val_families=tuple(
                family.strip() for family in args.val_families.split(",") if family.strip()
            ),
            sut_client=sut_client,
            user_backend=user_backend,
            user_simulator_model=user_simulator_model,
        )
        summary = summarize_result(result)
        print(
            f"[companion-trajgen] wrote {summary['total']} trajectories "
            f"(train={summary['train']}, val={summary['val']}) "
            f"manifest={result.manifest_path}"
        )
        return 0

    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
