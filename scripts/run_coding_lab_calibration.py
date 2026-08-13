#!/usr/bin/env python3
"""Run the coding-lab Packet 0 calibration (environment + oracle teeth).

Machinery calibration (default, no network, no brain):

    .venv/bin/python scripts/run_coding_lab_calibration.py --hand scripted

Frozen-hand calibration (requires an OpenAI-compatible endpoint; the
band verdict from this mode is the one Packet 2 prereg consumes):

    .venv/bin/python scripts/run_coding_lab_calibration.py \
        --hand api \
        --api-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --api-model qwen3-coder-next \
        --api-key-env DASHSCOPE_API_KEY

Exit codes: 0 = all verdicts True; 2 = at least one verdict False
(report still written); any exception = infrastructure failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "packages" / "lifeform-domain-coding" / "src"))

from lifeform_domain_coding.lab import (  # noqa: E402
    APIHandConfig,
    CalibrationConfig,
    run_calibration,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("scripted", "api"), default="scripted")
    parser.add_argument("--run-id", default=None, help="default: coding_lab_calibration_<unix>")
    parser.add_argument(
        "--output-root",
        default=str(_REPO_ROOT / "artifacts" / "coding_lab"),
    )
    parser.add_argument("--env-seed", type=int, default=20260812)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--episodes-per-chain", type=int, default=8)
    parser.add_argument("--band-low", type=float, default=0.2)
    parser.add_argument("--band-high", type=float, default=0.8)
    parser.add_argument(
        "--conventions",
        default="",
        help="逗号分隔的 house 约定 id（难度旋钮），如 convention_export_all。",
    )
    parser.add_argument("--heldout-variants", type=int, default=2)
    parser.add_argument("--scripted-hand-seed", type=int, default=11)
    parser.add_argument("--scripted-invariant-rate", type=float, default=0.25)
    parser.add_argument("--scripted-acceptance-rate", type=float, default=0.25)
    parser.add_argument("--api-base-url", default="")
    parser.add_argument("--api-model", default="")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument(
        "--api-extra-body-json",
        default="",
        help="JSON object merged into every request (e.g. OpenRouter provider pin)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing run dir: chains with rows.json load as-is; "
        "an interrupted chain is wiped and rerun whole.",
    )
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    api_config = None
    if args.hand == "api":
        if not args.api_base_url or not args.api_model:
            raise SystemExit("--hand api requires --api-base-url and --api-model")
        extra_body = json.loads(args.api_extra_body_json) if args.api_extra_body_json else {}
        if not isinstance(extra_body, dict):
            raise SystemExit("--api-extra-body-json must be a JSON object")
        api_config = APIHandConfig(
            base_url=args.api_base_url,
            model=args.api_model,
            api_key_env=args.api_key_env,
            temperature=args.api_temperature,
            extra_body=extra_body,
        )
    from lifeform_domain_coding.lab.episode import EpisodeBudget

    config = CalibrationConfig(
        run_id=args.run_id or f"coding_lab_calibration_{int(time.time())}",
        output_root=pathlib.Path(args.output_root),
        env_seed=args.env_seed,
        chains=args.chains,
        episodes_per_chain=args.episodes_per_chain,
        hand_kind=args.hand,
        scripted_hand_seed=args.scripted_hand_seed,
        scripted_invariant_sabotage_rate=args.scripted_invariant_rate,
        scripted_acceptance_sabotage_rate=args.scripted_acceptance_rate,
        api_hand_config=api_config,
        band_low=args.band_low,
        band_high=args.band_high,
        convention_ids=tuple(
            item.strip() for item in args.conventions.split(",") if item.strip()
        ),
        heldout_variants=args.heldout_variants,
        budget=EpisodeBudget(max_steps=args.max_steps, max_wall_seconds=args.max_wall_seconds),
        resume=args.resume,
    )
    report = asyncio.run(run_calibration(config))
    verdicts = report["verdicts"]
    print(json.dumps({"run_id": report["run_id"], "verdicts": verdicts}, ensure_ascii=False))
    print(f"report: {pathlib.Path(args.output_root) / report['run_id'] / 'report.json'}")
    return 0 if all(verdicts.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
