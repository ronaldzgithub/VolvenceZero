"""Stage 4 shadow sweep: run all probes with real_inputs() against TinyLlama.

Usage:
    PYTHONPATH=src python scripts/run_shadow_sweep.py [--model MODEL_ID]

Generates:
    experiments/shadow_sweep_<timestamp>.json — full results
    experiments/STAGE4_SHADOW_SWEEP.md — human-readable report

Requires the process-level ModelCache (S4.1) for any reasonable runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from volvence_labs.framework.probe import get_registry  # noqa: E402
from volvence_labs.framework.scheduler.runner import run_experiment  # noqa: E402
from volvence_labs.framework.wiring import get_profile  # noqa: E402
import volvence_labs.probes  # noqa: F401, E402  registers all probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--profile", default="shadow")
    parser.add_argument("--probes", nargs="*", default=None,
                        help="Subset of probes (default: all with real_inputs)")
    args = parser.parse_args()

    knobs = {"use_real_model": True, "model_id": args.model}
    profile = get_profile(args.profile)
    registry = get_registry()

    if args.probes:
        target_probes = args.probes
    else:
        target_probes = []
        for pid in registry.all_ids():
            cls = registry.get(pid)
            inst = cls()
            if hasattr(inst, "real_inputs"):
                target_probes.append(pid)
        target_probes.sort()

    print(f"== Stage 4 Shadow Sweep ==")
    print(f"Model:   {args.model}")
    print(f"Profile: {args.profile} ({len(profile.cells)} cells × {len(profile.seeds)} seeds = {len(profile.cells) * len(profile.seeds)} units/probe)")
    print(f"Probes:  {len(target_probes)}")
    print(f"Total:   {len(target_probes) * len(profile.cells) * len(profile.seeds)} units")
    print("---")
    sys.stdout.flush()

    results = []
    sweep_t0 = time.time()
    for pid in target_probes:
        t0 = time.time()
        try:
            report = run_experiment(pid, profile, knob_overrides=knobs)
            elapsed = time.time() - t0
            ok = sum(1 for u in report.units if u.ok)
            total = len(report.units)
            gate_passed = report.gate.passed if report.gate else False
            gate_reason = report.gate.reason if report.gate else "(no gate)"
            results.append({
                "probe_id": pid,
                "status": "OK",
                "ok_units": ok,
                "total_units": total,
                "gate_passed": gate_passed,
                "gate_reason": gate_reason,
                "elapsed_sec": round(elapsed, 1),
            })
            verdict = "PASS" if gate_passed else "FAIL"
            print(f"  {pid:32s} {ok}/{total} units  gate={verdict}  {elapsed:.0f}s  {gate_reason[:60]}")
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "probe_id": pid,
                "status": "ERROR",
                "error": str(e)[:200],
                "elapsed_sec": round(elapsed, 1),
            })
            print(f"  {pid:32s} ERROR: {str(e)[:80]}  {elapsed:.0f}s")
        sys.stdout.flush()

    total_elapsed = time.time() - sweep_t0
    n_pass = sum(1 for r in results if r.get("gate_passed"))
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print("---")
    print(f"Done: {n_pass}/{len(results)} gates PASS, {n_ok}/{len(results)} no-error")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Persist results
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = REPO_ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"shadow_sweep_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "model": args.model,
            "profile": args.profile,
            "total_elapsed_sec": round(total_elapsed, 1),
            "n_pass": n_pass,
            "n_ok": n_ok,
            "n_total": len(results),
            "results": results,
        }, f, indent=2)
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
