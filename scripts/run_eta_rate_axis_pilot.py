"""Stage-1 feasibility pilot: does the KL rate axis wake up as data grows?

The 2026-08-01 kill-eta run left the rate axis unexplained -- it barely moved
with alpha over the 7 hardcoded routes, so the frozen controller never had to
trade information for action accuracy. Gate 1 of the LLM-transfer ladder asks a
directional question the full 200-route sweep is too expensive to iterate on:
*as the seeded corpus grows, does spearman(alpha, rate) trend toward the
pre-registered <= -0.8 and does the rate span widen past the 7-route baseline?*

This script answers that cheaply by running the frozen arm only, a short alpha
grid, one seed, and a few updates across a ladder of route counts. It is
explicitly **not** the pre-registered Gate-1 verdict (that is
``run_eta_rate_distortion.py`` with the frozen preregistration at full scale);
its output is stamped feasibility-only and must not be cited as the gate result.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from companion_test_plan_common import (
    MPSAvailability,
    exclusive_mps_lock,
    mps_payload,
    require_mps,
)
from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    generate_eta_proof_corpus,
)
from volvence_zero.agent.eta_rate_distortion_evidence import (
    run_eta_rate_distortion_evidence,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-rate-axis-pilot.v1"

_SOURCE_FILES = (
    "packages/vz-temporal/src/volvence_zero/internal_rl/proof_environment.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_rate_distortion_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "scripts/run_eta_rate_axis_pilot.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-1 rate-axis feasibility pilot (frozen arm only)."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--route-counts",
        type=int,
        nargs="+",
        default=(6, 20, 40),
        help="Train route counts to sweep; heldout is scaled to a third.",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=(0.03, 0.3, 3.0)
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--updates", type=int, default=15)
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    args = parser.parse_args()
    if args.seeds < 1 or args.updates < 1:
        parser.error("--seeds and --updates must be >= 1")

    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty Stage-1 pilot directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ETAOpenWeightRuntimeConfig(
        device=args.device, model_dtype="float32"
    )
    uses_mps = args.device.startswith("mps")

    rungs: list[dict[str, object]] = []
    with ExitStack() as stack:
        mps: MPSAvailability | None = None
        if uses_mps:
            stack.enter_context(
                exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID)
            )
            mps = require_mps()
        for train_routes in args.route_counts:
            heldout_routes = max(4, train_routes // 3)
            corpus = generate_eta_proof_corpus(
                seed=args.corpus_seed,
                objective_count=args.objective_count,
                train_route_count=train_routes,
                heldout_route_count=heldout_routes,
            )
            started = time.perf_counter()
            report = run_eta_rate_distortion_evidence(
                alpha_grid=tuple(args.alphas),
                seed_schedule=tuple(range(args.seeds)),
                n_z=args.n_z,
                updates_per_run=args.updates,
                open_weight_config=config,
                arms=("frozen",),
                corpus=corpus,
            )
            elapsed = time.perf_counter() - started
            frozen_axis = next(
                r for r in report.rate_axis_responses if r.arm == "frozen"
            )
            rung = {
                "train_routes": report.train_route_count,
                "heldout_routes": report.heldout_route_count,
                "train_step_count": report.train_step_count,
                "spearman_alpha_rate": round(
                    frozen_axis.spearman_alpha_rate, 4
                ),
                "rate_span": round(frozen_axis.rate_span, 4),
                "rate_min": round(frozen_axis.rate_min, 4),
                "rate_max": round(frozen_axis.rate_max, 4),
                "curve": [
                    {
                        "alpha": row.alpha,
                        "rate": round(row.rate_mean, 4),
                        "distortion": round(row.distortion_mean, 4),
                        "heldout_distortion": round(
                            row.heldout_distortion_mean, 4
                        ),
                    }
                    for row in sorted(
                        report.curves, key=lambda r: r.alpha
                    )
                    if row.arm == "frozen"
                ],
                "elapsed_seconds": round(elapsed, 1),
            }
            rungs.append(rung)
            _write_json(output_dir / "rungs_partial.json", rungs)
            print(json.dumps(rung, indent=2))

    manifest = {
        "schema_version": "eta-rate-axis-pilot.v1",
        "experiment_id": "eta-stage1-rate-axis-pilot",
        "claim_scope": "feasibility-only-not-preregistered",
        "verdict_authoritative": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "device": args.device,
        "alphas": list(args.alphas),
        "seeds": args.seeds,
        "updates": args.updates,
        "n_z": args.n_z,
        "objective_count": args.objective_count,
        "corpus_seed": args.corpus_seed,
        "rungs": rungs,
        "gate1_reference": {
            "spearman_alpha_rate_max": -0.8,
            "rate_span_min": 0.30,
            "note": (
                "Directional read only. A trend of spearman falling toward "
                "-0.8 and rate_span widening with route count supports the "
                "data-mechanism hypothesis; the authoritative Gate-1 verdict "
                "requires the full preregistered 200-route frozen sweep."
            ),
        },
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_attestation": mps_payload(mps) if mps is not None else "not-required",
        "source_files": {
            name: _sha256(_REPO_ROOT / name) for name in _SOURCE_FILES
        },
    }
    _write_json(output_dir / "pilot_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "route_counts": [r["train_routes"] for r in rungs],
                "spearman_trend": [r["spearman_alpha_rate"] for r in rungs],
                "rate_span_trend": [r["rate_span"] for r in rungs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
