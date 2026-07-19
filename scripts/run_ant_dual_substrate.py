"""Workstream G1 demo -> research/ant/results/dual_substrate.json.

    python scripts/run_ant_dual_substrate.py

Shows the SAME kernel driving a non-language ant body and a language companion,
side by side. The deliverable is the printed table + JSON proving one
metacontroller produces z_t for both bodies.
"""

from __future__ import annotations

import asyncio
import argparse
from dataclasses import asdict
from pathlib import Path

from volvence_ant.evidence import collect_ant_provenance, write_ant_artifact_bundle
from volvence_ant.viz import run_formal_dual_substrate_demo

_RESULTS_DIR = Path("research/ant/results")
_REPO_ROOT = Path(__file__).resolve().parents[1]


async def main(*, model_id: str, model_source: str | None, turns: int) -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = await run_formal_dual_substrate_demo(
        hf_model_id=model_id,
        hf_model_source=model_source,
        turns=turns,
    )
    payload = {
        "artifact_kind": "digital-ant-formal-real-hf-dual-substrate",
        "experiment": "g1_dual_substrate",
        "description": report.description,
        "same_kernel_class": report.same_kernel_class,
        "same_code_dim": report.same_code_dim,
        "temporal_latent_dim": report.temporal_latent_dim,
        "ant": asdict(report.ant),
        "companion": asdict(report.companion),
        "turns": report.turns,
        "hf_runtime_origin": report.hf_runtime_origin,
        "hook_fire_rate": report.hook_fire_rate,
        "fallback_rate": report.fallback_rate,
        "verdict": (
            "PASS"
            if report.same_kernel_class
            and report.hook_fire_rate >= 0.75
            and report.fallback_rate == 0.0
            else "BLOCK"
        ),
    }
    manifest = write_ant_artifact_bundle(
        artifact_path=_RESULTS_DIR / "dual_substrate.json",
        payload=payload,
        provenance=collect_ant_provenance(
            repo_root=_REPO_ROOT,
            seeds=(0,),
            config={"model_id": model_id, "model_source": model_source, "turns": turns},
            model_fingerprint=f"{model_id}:{report.hf_runtime_origin}",
        ),
        repo_root=_REPO_ROOT,
    )
    print("=== G1: one kernel, two bodies ===")
    for probe in (report.ant, report.companion):
        print(
            f"  [{probe.body:>18}] substrate={probe.substrate_model_id:<20} "
            f"z_t(dim={probe.code_dim})={tuple(round(c, 3) for c in probe.code)} "
            f"-> {probe.output_kind}: {probe.output_summary}"
        )
    print(f"  same_kernel_class={report.same_kernel_class} same_code_dim={report.same_code_dim}")
    print(f"  verdict={payload['verdict']} manifest={manifest}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source")
    parser.add_argument("--turns", type=int, default=4)
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                model_id=args.model_id,
                model_source=args.model_source,
                turns=args.turns,
            )
        )
    )
