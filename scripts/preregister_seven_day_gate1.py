#!/usr/bin/env python3
"""Create an immutable hardware/model-specific Gate 1 preregistration."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from volvence_zero.agent.gate1_seven_day_preregistration import (
    build_gate1_seven_day_preregistration,
    write_gate1_seven_day_preregistration,
)
from volvence_zero.substrate import fingerprint_model_weight_files


def _model_contract(
    *,
    model_id: str,
    model_family: str,
    max_new_tokens: int,
    temperature: float | None = None,
) -> dict[str, object]:
    model_root = Path(snapshot_download(model_id, local_files_only=True))
    contract: dict[str, object] = {
        "model_id": model_id,
        "model_family": model_family,
        "weights_sha256": fingerprint_model_weight_files(model_root),
        "local_files_only": True,
        "frozen": True,
        "max_new_tokens": max_new_tokens,
    }
    if temperature is not None:
        contract.update(
            {
                "temperature": temperature,
                "top_p": 1.0,
                "rendering_contract": (
                    "typed-FSM substantive draft plus deterministic local-LLM "
                    "style rendering"
                ),
            }
        )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at-unix-ms", type=int, required=True)
    parser.add_argument(
        "--device", choices=("mps", "cuda", "cuda:0"), required=True
    )
    parser.add_argument(
        "--sut-model-id", default="HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    parser.add_argument("--sut-model-family", default="smollm")
    parser.add_argument("--sut-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--simulator-model-id", default="Qwen/Qwen2.5-1.5B-Instruct"
    )
    parser.add_argument("--simulator-model-family", default="qwen")
    parser.add_argument("--simulator-max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    payload = build_gate1_seven_day_preregistration(
        repo_root=args.repo_root.resolve(),
        created_at_unix_ms=args.created_at_unix_ms,
        execution_device=args.device,
        sut_model=_model_contract(
            model_id=args.sut_model_id,
            model_family=args.sut_model_family,
            max_new_tokens=args.sut_max_new_tokens,
        ),
        simulator_model=_model_contract(
            model_id=args.simulator_model_id,
            model_family=args.simulator_model_family,
            max_new_tokens=args.simulator_max_new_tokens,
            temperature=0.0,
        ),
    )
    digest = write_gate1_seven_day_preregistration(
        payload=payload, output_path=args.output
    )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
