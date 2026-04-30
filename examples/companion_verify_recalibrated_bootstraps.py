"""Verify a freshly-recalibrated companion bootstrap pair.

Loads the artifacts produced by ``companion_recalibrate_with_qwen.py``
(``companion-temporal.qwen.snap`` + ``companion-regime.qwen.bs``),
injects them into a real-Qwen companion lifeform, and runs a tiny
4-prompt diagnostic across the canonical companion regimes:

* `hello` -> light first-contact
* `task` -> concrete request (problem_solving expected)
* `emotion` -> low-mood disclosure (emotional_support expected)
* `rupture` -> rupture signal (repair_and_deescalation expected)
* `repair` -> apology / step-back (repair_and_deescalation OR emotional_support)

Prints the regime that landed for each, the 12-axis InterlocutorState
reduced to 3 key axes (rapport / resistance / trust), and a
**regime-distribution entropy** in nats so the caller can decide
whether the recalibration solved the monoculture problem from
phase 1.5.

Usage:

    python examples/companion_verify_recalibrated_bootstraps.py \\
        --artifact-dir /path/to/recal/output

If ``--artifact-dir`` is omitted the script looks for the most
recently-modified ``companion-qwen-recal-*`` tmp dir and uses that.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import pathlib
import sys
import tempfile
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from lifeform_domain_emogpt import build_companion_lifeform_with_real_substrate
from lifeform_evolution import load_regime_bootstrap
from lifeform_evolution.snapshot_io import load_snapshot


_PROBES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "hello",
        "Hey - it's been a while.",
        ("acquaintance_building", "casual_social"),
    ),
    (
        "task",
        "Can you help me draft a polite email declining a meeting invite?",
        ("problem_solving", "guided_exploration"),
    ),
    (
        "emotion",
        "Honestly I've been struggling with sleep, low-energy mornings, "
        "and I keep circling around freelancing but I'm scared.",
        ("emotional_support", "guided_exploration"),
    ),
    (
        "rupture",
        "Wait - that just felt clinical and procedural. I'm not asking "
        "you to optimise me.",
        ("repair_and_deescalation", "emotional_support"),
    ),
    (
        "repair",
        "OK. Sorry. Can we just back up. I just needed to say it out loud.",
        ("repair_and_deescalation", "emotional_support"),
    ),
)


def _find_latest_artifact_dir() -> pathlib.Path | None:
    tmp_root = pathlib.Path(tempfile.gettempdir())
    candidates = sorted(
        (p for p in tmp_root.iterdir() if p.is_dir()
         and p.name.startswith("companion-qwen-recal-")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _entropy_nats(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for k, v in counts.items():
        if v <= 0:
            continue
        p = v / total
        h -= p * math.log(p)
    return h


async def verify(artifact_dir: pathlib.Path) -> int:
    temporal_path = artifact_dir / "companion-temporal.qwen.snap"
    regime_path = artifact_dir / "companion-regime.qwen.bs"
    if not temporal_path.is_file() or not regime_path.is_file():
        print(f"FAIL: artifact files missing under {artifact_dir}",
              file=sys.stderr)
        return 1
    print(f"  [artifact-dir] {artifact_dir}")
    regime_artifact = load_regime_bootstrap(regime_path)
    temporal_artifact = load_snapshot(temporal_path)
    print("  [selection_weights]")
    for rid, w in regime_artifact.bootstrap.selection_weights:
        print(f"    {rid:<28}{w:>5.2f}")

    print("  [substrate] loading Qwen 2.5 0.5B Instruct ...")
    bundle = build_companion_lifeform_with_real_substrate(
        fallback_to_builtin=False
    )
    if not bundle.is_real_substrate:
        print(f"FAIL: substrate degraded ({bundle.runtime_origin})",
              file=sys.stderr)
        return 1
    lf = bundle.lifeform.with_regime_bootstrap(regime_artifact.bootstrap)
    lf = lf.with_temporal_bootstrap(temporal_artifact.snapshot)
    session = lf.create_session(session_id="companion-verify")

    print()
    header = (
        f"  {'phase':<10}{'regime':<26}"
        f"{'rapp':>6}{'res':>6}{'trust':>8}  hit?"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    landed_regimes: list[str] = []
    hits = 0
    for label, text, expected in _PROBES:
        await session.run_turn(text)
        snap = session.latest_active_snapshots.get("regime")
        regime = (
            snap.value.active_regime.regime_id if snap is not None else "?"
        )
        landed_regimes.append(regime)
        st = session.interlocutor_state
        hit = "Y" if regime in expected else "."
        if regime in expected:
            hits += 1
        print(
            f"  {label:<10}{regime:<26}"
            f"{st.rapport_warmth:>6.2f}{st.resistance_level:>6.2f}"
            f"{st.trust_signal:>+8.2f}  {hit}"
        )

    counts = Counter(landed_regimes)
    print()
    print(f"  hits: {hits}/{len(_PROBES)}")
    print(f"  regime distribution: {dict(counts)}")
    print(f"  entropy: {_entropy_nats(counts):.3f} nats")
    print(f"  unique regimes selected: {len(counts)}")
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifact-dir", type=str, default="",
        help="dir holding companion-temporal.qwen.snap + .bs",
    )
    args = parser.parse_args()
    if args.artifact_dir:
        artifact_dir = pathlib.Path(args.artifact_dir)
    else:
        latest = _find_latest_artifact_dir()
        if latest is None:
            print("FAIL: no companion-qwen-recal-* dir found in tmp; "
                  "pass --artifact-dir explicitly.", file=sys.stderr)
            return 1
        artifact_dir = latest
    return asyncio.run(verify(artifact_dir))


if __name__ == "__main__":
    raise SystemExit(_cli())
