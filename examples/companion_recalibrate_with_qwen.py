"""Slice 2a phase 1.5: re-calibrate companion regime classifier on Qwen.

Phase 1 wired ``Qwen 2.5 0.5B Instruct`` into ``companion`` vertical
and the comparison demo confirmed the substrate switch reaches the
regime layer. But neither substrate triggered ``emotional_support``
or ``repair_and_deescalation`` even on canonical prompts (sleep
struggle / explicit clinical complaint). Root cause: the shipped
``companion-regime.bs`` was trained on synthetic activations, so its
``selection_weights`` are tuned for a different signal distribution
than what real Qwen actually emits.

This script re-runs ``run_super_loop_async`` with the real Qwen
substrate threaded through, so the resulting weights match the
real-substrate distribution. Output:

* ``companion-temporal.qwen.snap`` \u2014 metacontroller snapshot
* ``companion-regime.qwen.bs``    \u2014 regime selection_weights

By DEFAULT writes to a tmp dir + prints absolute paths so you can
review before deciding to overwrite the shipped
``companion-temporal.snap`` / ``companion-regime.bs`` artifacts.
Pass ``--inplace`` to overwrite the shipped versions directly.

**Import discipline:** mirrors the other ``examples/`` scripts \u2014
public ``lifeform-*`` wheel imports + a small stdlib whitelist.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import tempfile
import time

# Force UTF-8 stdout for Windows GBK consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# ---------------------------------------------------------------------------
# Public wheel imports only.
# ---------------------------------------------------------------------------

from lifeform_domain_emogpt import (
    DEFAULT_REAL_MODEL_SOURCE,
    bootstraps_dir as companion_bootstraps_dir,
    build_companion_lifeform_with_real_substrate,
    scenarios_dir as companion_scenarios_dir,
)
from lifeform_evolution import (
    load_scenario_pack_dir,
    save_regime_bootstrap,
    save_snapshot,
)
from lifeform_evolution.super_loop import run_super_loop_async


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_header(text: str) -> None:
    print()
    print("=" * 72)
    print(f"  {text}")
    print("=" * 72)


def _print_section(tag: str, text: str) -> None:
    print(f"  [{tag}] {text}")


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------


async def recalibrate(
    *,
    rounds: int,
    output_dir: pathlib.Path,
    diversity_threshold: float,
    diversity_lr: float,
    model_source: str,
    device: str,
    local_files_only: bool,
) -> None:
    _print_header(f"Phase 1: load {model_source} + companion scenarios")
    started_load = time.monotonic()
    bundle = build_companion_lifeform_with_real_substrate(
        model_source=model_source,
        device=device,
        local_files_only=local_files_only,
        # Hard-fail the recalibration if Qwen can't be loaded \u2014 a
        # degraded calibration would just produce another set of
        # synthetic-looking weights and be misleading.
        fallback_to_builtin=False,
    )
    if not bundle.is_real_substrate:
        raise RuntimeError(
            f"recalibration aborted: substrate fell back to "
            f"{bundle.runtime_origin!r}; run with HF cache primed "
            f"(see companionship_real_substrate_demo.py)."
        )
    _print_section(
        "substrate", f"loaded {bundle.status_label} in {time.monotonic() - started_load:.1f}s"
    )

    scenarios = load_scenario_pack_dir(companion_scenarios_dir())
    _print_section("scenarios", f"loaded {len(scenarios)} from companion vertical")

    _print_header(
        f"Phase 2: run_super_loop_async (rounds={rounds}) with real substrate"
    )
    print(
        "  Each round runs every scenario through the kernel + collects SSL "
        f"trace.\n  Device={device}. Total ETA depends on model and device: "
        f"{rounds} x ~60s.\n"
    )
    started_loop = time.monotonic()
    report = await run_super_loop_async(
        rounds=rounds,
        scenarios=scenarios,
        diversity_threshold=diversity_threshold,
        diversity_lr=diversity_lr,
        substrate_runtime=bundle.runtime,
    )
    elapsed_loop = time.monotonic() - started_loop
    _print_section("super_loop", f"completed {rounds} rounds in {elapsed_loop:.1f}s")

    _print_header("Phase 3: per-round summary")
    print(
        "    "
        f"{'round':<6}{'match':>7}{'misclass':>10}{'switch_freq':>14}"
    )
    print("    " + "-" * 40)
    for r in report.rounds:
        switch_freq = r.ssl.switch_frequency_last
        switch_str = f"{switch_freq:.3f}" if switch_freq is not None else "n/a"
        print(
            "    "
            f"{r.round_index:<6}"
            f"{r.regime_match_rate:>7.0%}"
            f"{r.misclassified_turn_count:>10}"
            f"{switch_str:>14}"
        )

    # Verdicts: did regime match improve? Did temporal evolve?
    _print_header("Phase 4: verdicts")
    for name, value in report.verdicts.items():
        _print_section("verdict", f"{name}={value}")

    _print_header("Phase 5: predicted regime distribution (final round)")
    final_round = report.rounds[-1]
    for regime_id, count in final_round.predicted_regime_counts:
        _print_section("predicted", f"{regime_id:<28}{count}")

    _print_header("Phase 6: save artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_path = output_dir / "companion-temporal.qwen.snap"
    regime_path = output_dir / "companion-regime.qwen.bs"
    save_snapshot(
        report.final_temporal_snapshot,
        temporal_path,
        metadata={
            "model_id": bundle.model_id,
            "model_source": model_source,
            "device": device,
            "rounds": rounds,
            "calibrated_at_ms": int(time.time() * 1000),
        },
    )
    save_regime_bootstrap(
        report.final_regime_bootstrap,
        regime_path,
        metadata={
            "model_id": bundle.model_id,
            "model_source": model_source,
            "device": device,
            "rounds": rounds,
            "match_rate_baseline": report.rounds[0].regime_match_rate,
            "match_rate_final": report.rounds[-1].regime_match_rate,
        },
    )
    _print_section("saved", f"temporal -> {temporal_path}")
    _print_section("saved", f"regime   -> {regime_path}")
    _print_section(
        "selection_weights",
        ", ".join(
            f"{rid}={w:.2f}"
            for rid, w in report.final_regime_bootstrap.selection_weights
        ),
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="number of super-loop rounds (default 4: 1 baseline + 3 trained)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="where to write recalibrated artifacts (default: a tmp dir)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help=(
            "OVERWRITE the shipped companion-temporal.snap / "
            "companion-regime.bs artifacts. Use with care; commits to "
            "real product behavior change."
        ),
    )
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=0.50,
        help="diversity-aware calibrator threshold (default 0.50)",
    )
    parser.add_argument(
        "--diversity-lr",
        type=float,
        default=0.30,
        help="diversity-aware calibrator step size (default 0.30)",
    )
    parser.add_argument("--model-source", default=DEFAULT_REAL_MODEL_SOURCE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="refuse network downloads and use only the local HF cache",
    )
    args = parser.parse_args()

    if args.inplace:
        if args.output_dir:
            print(
                "FAIL: --inplace and --output-dir are mutually exclusive",
                file=sys.stderr,
            )
            return 2
        # Use a tmp dir for the .qwen.* names, then we copy if
        # results are good. Direct overwrite happens AFTER manual
        # review of the verdicts.
        print(
            "WARNING: --inplace will overwrite the shipped companion "
            "artifacts AFTER this run completes successfully.\n"
            "Press Ctrl-C now if that's not what you want.\n",
            file=sys.stderr,
        )
        time.sleep(3)
        out = companion_bootstraps_dir()
    elif args.output_dir:
        out = pathlib.Path(args.output_dir)
    else:
        out = pathlib.Path(tempfile.mkdtemp(prefix="companion-qwen-recal-"))

    asyncio.run(
        recalibrate(
            rounds=args.rounds,
            output_dir=out,
            diversity_threshold=args.diversity_threshold,
            diversity_lr=args.diversity_lr,
            model_source=args.model_source,
            device=args.device,
            local_files_only=args.local_files_only,
        )
    )

    if args.inplace:
        # Move the .qwen.* artifacts on top of the shipped ones.
        import shutil

        shipped_temporal = out / "companion-temporal.snap"
        shipped_regime = out / "companion-regime.bs"
        new_temporal = out / "companion-temporal.qwen.snap"
        new_regime = out / "companion-regime.qwen.bs"
        shutil.copy2(new_temporal, shipped_temporal)
        shutil.copy2(new_regime, shipped_regime)
        print()
        print(f"INPLACE: {shipped_temporal} and {shipped_regime} updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
