"""Side-by-side companion demo: synthetic substrate vs real Qwen substrate.

This is the **slice 2a phase 1 diagnostic**: run THE SAME 5-turn
companion script under two substrate configurations and print
the trajectory tables side by side. The point is to make
visible whether switching the substrate from synthetic to a
real open-weight LLM (Qwen 2.5 0.5B Instruct, ~500 MB on disk)
unblocks the regime / InterlocutorState differentiation that
``companionship_end_to_end_demo.py`` showed flatlined under
synthetic.

Pipeline (same as the original companion demo, but doubled):

* Build TWO companion lifeforms, one synthetic, one real.
* Run the SAME canned 5-turn script through each.
* Sample regime + 12-axis InterlocutorState at every turn for
  both.
* Print TWO trajectory tables and a diff column showing the
  per-axis movement difference.

**Import discipline:** same as the other examples \u2014 no
``volvence_zero.*`` imports at top level. Real substrate is
loaded transparently inside ``lifeform-domain-emogpt`` /
``lifeform-affordance`` etc.

Run:

    python examples/companionship_real_substrate_demo.py

First run downloads ~500 MB Qwen weights (one-off,
``~/.cache/huggingface/``); subsequent runs are fast.

If the host has no outbound network or transformers is missing,
the helper falls back to a built-in random-weights GPT-2 (the
demo prints ``DEGRADED`` next to the affected column so the
output is honest about the mode in use).
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

# Force UTF-8 stdout for Windows GBK consoles. No-op on POSIX.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# ---------------------------------------------------------------------------
# Public wheel imports only.
# ---------------------------------------------------------------------------

from lifeform_domain_emogpt import (
    build_companion_lifeform,
    build_companion_lifeform_with_real_substrate,
)
from lifeform_ingestion import IngestionPipeline, envelope_from_text
from lifeform_thinking import (
    ThinkingWiringLevel,
    build_default_thinking_adapter,
)


# ---------------------------------------------------------------------------
# Demo script (same content as companionship_end_to_end_demo.py)
# ---------------------------------------------------------------------------


_BACKGROUND_MEMO = """The interlocutor's name is Mei. She is a 34-year-old
graphic designer based in Shanghai, currently between full-time roles. She
has mentioned in past conversations that she is working through a slow
transition out of agency life into something more independent, and that
she values being heard more than being given advice. She has a cat named
Tofu and lives alone. She tends to disclose carefully and pulls back fast
when conversations turn clinical.
"""


_SCRIPT: tuple[tuple[str, str], ...] = (
    (
        "turn-1-hello",
        "Hey - it's been a while. I wasn't sure if I'd come back, but here I am.",
    ),
    (
        "turn-2-disclose",
        "Honestly I've been struggling with sleep, low-energy mornings, "
        "and I keep circling around this idea of going freelance but I'm "
        "scared. I don't know how to talk about it without spiraling.",
    ),
    (
        "turn-3-guided",
        "I think I want to make the change but I keep imagining the worst "
        "outcome. Can we slow down and look at what's actually scary?",
    ),
    (
        "turn-4-rupture",
        "Wait - that just felt clinical and procedural. I'm not asking "
        "you to optimise me. I want to feel heard, not workshopped.",
    ),
    (
        "turn-5-repair",
        "OK. Sorry. Can we just back up. I don't even need an answer right "
        "now - I just needed to say it out loud.",
    ),
)


# ---------------------------------------------------------------------------
# Trajectory record
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    label: str
    regime: str
    engagement: float
    focus: float
    rapport: float
    resistance: float
    openness: float
    trust: float
    confidence: float


@dataclass
class SubstrateRunResult:
    name: str
    status_label: str
    is_real: bool
    trajectory: list[TurnRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_to_record(label: str, session: Any) -> TurnRecord:
    state = session.interlocutor_state
    regime_id = ""
    snap = session.latest_active_snapshots.get("regime")
    if snap is not None:
        active = getattr(snap.value, "active_regime", None)
        if active is not None:
            regime_id = str(getattr(active, "regime_id", ""))
    return TurnRecord(
        label=label,
        regime=regime_id or "<none>",
        engagement=state.engagement_intensity,
        focus=state.task_focus_level,
        rapport=state.rapport_warmth,
        resistance=state.resistance_level,
        openness=state.openness_to_guidance,
        trust=state.trust_signal,
        confidence=state.readout_confidence,
    )


async def _run_script_through_lifeform(
    lifeform: Any, *, session_id: str
) -> list[TurnRecord]:
    """Run the canned script through a lifeform and snapshot per turn.

    Includes a single ingestion turn at the start (background memo)
    + 5 natural turns. Mirrors the companion demo's discipline so
    the two substrate runs see IDENTICAL inputs.
    """
    session = lifeform.create_session(session_id=session_id)
    pipeline = IngestionPipeline()
    envelope = envelope_from_text(
        _BACKGROUND_MEMO,
        source_uri="memo://mei-background.txt",
        uploader="demo-runner",
    )
    await pipeline.process_envelope(
        env=envelope, session=session, end_scene_after=False
    )
    trajectory: list[TurnRecord] = [
        _snapshot_to_record("after-ingest", session)
    ]
    for label, text in _SCRIPT:
        await session.run_turn(text)
        trajectory.append(_snapshot_to_record(label, session))
    await session.end_scene(reason="real-substrate-demo-end")
    trajectory.append(_snapshot_to_record("after-close", session))
    return trajectory


async def _run_synthetic() -> SubstrateRunResult:
    """Synthetic substrate path: same as the existing companion demo."""
    print("\n[synthetic] building lifeform (no model download) ...")
    lifeform = build_companion_lifeform().with_thinking_adapter_factory(
        lambda: build_default_thinking_adapter(
            wiring_level=ThinkingWiringLevel.SHADOW,
        ),
    )
    print("[synthetic] running 5-turn script ...")
    trajectory = await _run_script_through_lifeform(
        lifeform, session_id="real-vs-synthetic-synth"
    )
    return SubstrateRunResult(
        name="synthetic",
        status_label="synthetic-default",
        is_real=False,
        trajectory=trajectory,
    )


async def _run_real() -> SubstrateRunResult:
    """Real Qwen substrate path. First call may take ~60s for download."""
    print(
        "\n[real] building lifeform on Qwen 2.5 0.5B Instruct "
        "(first run may download ~500 MB) ..."
    )
    bundle = build_companion_lifeform_with_real_substrate(
        # Force a real load with no fallback so the demo fails loud
        # if the model can't be reached \u2014 a degraded run is worse
        # than a clear error for slice 2a's diagnostic intent.
        # If you want fallback-on-error, flip this to True.
        fallback_to_builtin=False,
    )
    lifeform_with_thinking = bundle.lifeform.with_thinking_adapter_factory(
        lambda: build_default_thinking_adapter(
            wiring_level=ThinkingWiringLevel.SHADOW,
        ),
    )
    print(
        f"[real] substrate ready: status={bundle.status_label!r} "
        f"is_real={bundle.is_real_substrate}"
    )
    print(
        "[real] running 5-turn script "
        "(each turn ~3-10s on CPU; 5 turns + ingestion ~30-60s) ..."
    )
    trajectory = await _run_script_through_lifeform(
        lifeform_with_thinking, session_id="real-vs-synthetic-real"
    )
    return SubstrateRunResult(
        name="real",
        status_label=bundle.status_label,
        is_real=bundle.is_real_substrate,
        trajectory=trajectory,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_run_table(result: SubstrateRunResult) -> None:
    title_suffix = "" if result.is_real else "  [DEGRADED]"
    print(f"\n  {result.name.upper()} ({result.status_label}){title_suffix}")
    print(
        "    "
        f"{'phase':<22}"
        f"{'regime':<26}"
        f"{'eng':>6}"
        f"{'focus':>7}"
        f"{'rapp':>7}"
        f"{'res':>7}"
        f"{'open':>7}"
        f"{'trust':>8}"
    )
    print("    " + "-" * 90)
    for record in result.trajectory:
        regime_short = record.regime[:24]
        print(
            "    "
            f"{record.label:<22}"
            f"{regime_short:<26}"
            f"{record.engagement:>6.2f}"
            f"{record.focus:>7.2f}"
            f"{record.rapport:>7.2f}"
            f"{record.resistance:>7.2f}"
            f"{record.openness:>7.2f}"
            f"{record.trust:>+8.2f}"
        )


def _print_diff_table(
    synth: SubstrateRunResult, real: SubstrateRunResult
) -> None:
    """Print per-phase delta (real - synthetic) for the 6 visible axes.

    Uses the SAME phase labels in both runs (the demo's contract).
    A row of large absolute deltas means the substrate switch
    measurably changed the InterlocutorState; a row of near-zero
    deltas means the kernel was insensitive to substrate features
    at that turn.
    """
    print("\n  DIFF (real - synthetic) per phase")
    print(
        "    "
        f"{'phase':<22}"
        f"{'regime_real':<22}"
        f"{'regime_synth':<22}"
        f"{'rapp':>7}"
        f"{'res':>7}"
        f"{'trust':>8}"
        f"{'open':>7}"
    )
    print("    " + "-" * 100)
    by_label_real = {r.label: r for r in real.trajectory}
    for synth_row in synth.trajectory:
        real_row = by_label_real.get(synth_row.label)
        if real_row is None:
            continue
        print(
            "    "
            f"{synth_row.label:<22}"
            f"{real_row.regime[:20]:<22}"
            f"{synth_row.regime[:20]:<22}"
            f"{real_row.rapport - synth_row.rapport:>+7.2f}"
            f"{real_row.resistance - synth_row.resistance:>+7.2f}"
            f"{real_row.trust - synth_row.trust:>+8.2f}"
            f"{real_row.openness - synth_row.openness:>+7.2f}"
        )


def _summarise_diff(
    synth: SubstrateRunResult, real: SubstrateRunResult
) -> dict[str, float]:
    """Compute summary deltas the integration test asserts on.

    Returns dict of {axis: max_abs_diff_across_phases}. Useful as
    the canonical "did the substrate matter?" gate.
    """
    by_label_real = {r.label: r for r in real.trajectory}
    aggregates: dict[str, float] = {
        "engagement": 0.0,
        "focus": 0.0,
        "rapport": 0.0,
        "resistance": 0.0,
        "openness": 0.0,
        "trust": 0.0,
    }
    regimes_diff = 0
    for synth_row in synth.trajectory:
        real_row = by_label_real.get(synth_row.label)
        if real_row is None:
            continue
        if real_row.regime != synth_row.regime:
            regimes_diff += 1
        for axis in aggregates:
            delta = abs(getattr(real_row, axis) - getattr(synth_row, axis))
            if delta > aggregates[axis]:
                aggregates[axis] = delta
    aggregates["regime_phase_disagreement_count"] = float(regimes_diff)
    return aggregates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> dict[str, Any]:
    print("=" * 72)
    print("  Companion vertical: synthetic vs real substrate diagnostic")
    print("=" * 72)
    synth = await _run_synthetic()
    real = await _run_real()

    print("\n" + "=" * 72)
    print("  Per-substrate trajectory")
    print("=" * 72)
    _print_run_table(synth)
    _print_run_table(real)
    _print_diff_table(synth, real)
    summary = _summarise_diff(synth, real)
    print("\n  Summary deltas (max |real - synthetic| across phases)")
    for axis, value in summary.items():
        print(f"    {axis:<38}{value:>+7.2f}")
    print("\n  Reading guide:")
    print(
        "    - regime_phase_disagreement_count > 0 means the regime")
    print("      classifier landed in different bins for the two substrates")
    print("      on at least one phase. That's the canonical proof the")
    print("      substrate switch reaches the regime layer.")
    print("    - Any axis with max-diff > 0.10 means the 12-axis readout is")
    print("      meaningfully content-aware under the real substrate. <0.05")
    print("      means the kernel did not propagate substrate features for")
    print("      that axis on these prompts.")
    return {
        "synth": synth,
        "real": real,
        "summary": summary,
    }


def _cli() -> int:
    result = asyncio.run(main())
    if not result["real"].is_real:
        print(
            "\nFAIL: real substrate did not load (degraded mode). "
            "Set HF_TOKEN or check network.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
