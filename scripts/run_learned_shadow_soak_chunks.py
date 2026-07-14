#!/usr/bin/env python3
"""Chunked learned-shadow soak runner for Windows-native stability.

Continuous 500-turn runs can trigger native torch/transformers crashes on some
Windows CUDA stacks. This runner executes smaller independent chunks and emits
an aggregate artifact. It is directional evidence only: ACTIVE promotion still
requires one continuous real-trace artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=25)
    parser.add_argument("--turns-per-chunk", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/learned_shadow_soak_chunked"),
    )
    parser.add_argument("--substrate-mode", default="synthetic", choices=["synthetic", "hf"])
    parser.add_argument("--substrate-device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_records: list[dict[str, object]] = []
    for index in range(args.chunks):
        chunk_dir = args.output_dir / f"chunk-{index:03d}"
        command = [
            sys.executable,
            "-u",
            "scripts/run_learned_shadow_soak.py",
            "--turns",
            str(args.turns_per_chunk),
            "--sample-every",
            str(max(1, args.turns_per_chunk // 2)),
            "--checkpoint-every",
            str(args.turns_per_chunk),
            "--temporal-latent-dim",
            "16",
            "--backend-combo",
            "runtime+ssl+internal-rl+cms-torch",
            "--substrate-mode",
            args.substrate_mode,
            "--substrate-device",
            args.substrate_device,
            "--output-dir",
            str(chunk_dir),
        ]
        print(f"[chunked-soak] chunk {index + 1}/{args.chunks}: {' '.join(command)}")
        rc = subprocess.run(command, check=False).returncode
        artifact_path = chunk_dir / "learned_shadow_soak.json"
        record: dict[str, object] = {
            "chunk_index": index,
            "returncode": rc,
            "artifact_path": str(artifact_path),
            "artifact_present": artifact_path.is_file(),
        }
        if artifact_path.is_file():
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            record["turn_count"] = payload.get("turn_count", 0)
            record["real_trace_turns"] = payload.get("real_trace_turns", 0)
            record["mean_turn_seconds"] = payload.get("mean_turn_seconds", 0.0)
            record["strict_eta_gate_passed"] = (
                payload.get("strict_eta_gate", {}).get("gate_passed", False)
            )
        chunk_records.append(record)
        if rc != 0:
            break

    successful = [row for row in chunk_records if row.get("artifact_present")]
    aggregate = {
        "schema_version": "learned-shadow-soak-chunked.v1",
        "artifact_kind": "learned_shadow_soak_chunked",
        "continuous": False,
        "promotion_evidence": False,
        "note": (
            "Chunked Windows lane. Useful for directional stability and evidence "
            "surface coverage; not a substitute for continuous real-trace promotion evidence."
        ),
        "requested_turns": args.chunks * args.turns_per_chunk,
        "completed_turns": sum(int(row.get("turn_count", 0)) for row in successful),
        "completed_real_trace_turns": sum(
            int(row.get("real_trace_turns", 0)) for row in successful
        ),
        "chunk_count": len(chunk_records),
        "successful_chunk_count": len(successful),
        "chunks": chunk_records,
    }
    out_path = args.output_dir / "learned_shadow_soak_chunked.json"
    out_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[chunked-soak] aggregate written to {out_path}")
    return 0 if len(successful) == args.chunks else 1


if __name__ == "__main__":
    raise SystemExit(main())
