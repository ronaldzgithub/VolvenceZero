"""Diagnostic probe for ``score_regimes`` per-feature behaviour on real Qwen.

Phase 1.7 fixed the ``selection_weights`` cap so it acts as a soft prior
again. But verify-script hits stayed 1/5 - non-task prompts (emotion /
rupture / repair) still routed to ``problem_solving``. This probe pulls
the issue ONE LEVEL DEEPER: it dumps each candidate's RAW
``base_score`` (the 0-1 weighted sum BEFORE selection_weights /
strategy_priors / fast_bias) per prompt, so we can see exactly which
regime WOULD have won under uniform priors.

If emotional_support's base_score is BELOW problem_solving's on the
"I have been struggling with sleep" prompt, the ``score_regimes``
formula's per-feature weights are wrong for real Qwen's residual
distribution. THAT becomes phase 1.8's adjustment target.

Run:

    python examples/companion_score_regimes_probe.py

Prints a per-prompt table of all 6 regimes' raw base_scores ranked
descending, plus the input substrate features (semantic_*_pull
proxies via dual_track drives) so the operator can correlate.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# ---------------------------------------------------------------------------
# This probe is a DIAGNOSTIC, not product code. It reaches into the
# kernel's ``score_regimes`` directly to get base_scores under
# uniform priors. That's intentional: instrumentation crosses the
# normal abstraction boundary by design. Application-layer code
# stays in ``examples/companionship_*`` which respects the boundary.
# ---------------------------------------------------------------------------

from volvence_zero.regime import REGIME_TEMPLATES, score_regimes
from lifeform_domain_emogpt import build_companion_lifeform_with_real_substrate


_PROMPTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("hello", "Hey - it's been a while.",
     ("acquaintance_building", "casual_social")),
    ("task", "Can you help me draft a polite email declining a meeting invite?",
     ("problem_solving", "guided_exploration")),
    ("emotion",
     "Honestly I've been struggling with sleep, low-energy mornings, "
     "and I keep circling around freelancing but I'm scared.",
     ("emotional_support", "guided_exploration")),
    ("rupture",
     "Wait - that just felt clinical and procedural. I'm not asking "
     "you to optimise me.",
     ("repair_and_deescalation", "emotional_support")),
    ("repair",
     "OK. Sorry. Can we just back up. I just needed to say it out loud.",
     ("repair_and_deescalation", "emotional_support")),
)


def _summarise_dual_track(snap: Any) -> str:
    """Compact single-line summary of the substrate signals as
    surfaced through dual_track. We dump these so the operator can
    see WHY a given regime won, not just THAT it did.
    """
    if snap is None:
        return "<no dual_track>"
    v = snap.value
    world = v.world_track
    self_ = v.self_track
    return (
        f"world(t={world.tension_level:.2f},code={tuple(round(c,2) for c in world.controller_code)},"
        f"hint={world.abstract_action_hint or '-'}) "
        f"self(t={self_.tension_level:.2f},code={tuple(round(c,2) for c in self_.controller_code)},"
        f"hint={self_.abstract_action_hint or '-'}) "
        f"cross={v.cross_track_tension:.2f}"
    )


async def main() -> int:
    print("Loading Qwen 0.5B Instruct...")
    bundle = build_companion_lifeform_with_real_substrate(
        fallback_to_builtin=False,
    )
    if not bundle.is_real_substrate:
        print(f"FAIL: substrate degraded ({bundle.runtime_origin})",
              file=sys.stderr)
        return 1
    print(f"substrate: {bundle.status_label}\n")

    session = bundle.lifeform.create_session(session_id="score-regimes-probe")
    # Uniform priors so base_score wins / loses on its own merit.
    uniform_weights = {t.regime_id: 1.0 for t in REGIME_TEMPLATES}
    flat_strategy = {t.regime_id: 0.0 for t in REGIME_TEMPLATES}
    flat_historical = {t.regime_id: 0.5 for t in REGIME_TEMPLATES}

    for label, text, expected in _PROMPTS:
        await session.run_turn(text)
        snaps = session.latest_active_snapshots
        memory_value = snaps.get("memory")
        memory_value = memory_value.value if memory_value is not None else None
        dual_track_snap = snaps.get("dual_track")
        dual_track_value = (
            dual_track_snap.value if dual_track_snap is not None else None
        )
        evaluation_snap = snaps.get("evaluation")
        evaluation_value = (
            evaluation_snap.value if evaluation_snap is not None else None
        )
        pe_snap = snaps.get("prediction_error")
        pe_value = pe_snap.value if pe_snap is not None else None

        # Call score_regimes with uniform priors so we see the
        # base_score winner with no calibrator influence.
        ranked = score_regimes(
            memory_snapshot=memory_value,
            dual_track_snapshot=dual_track_value,
            evaluation_snapshot=evaluation_value,
            prediction_error_snapshot=pe_value,
            historical_effectiveness=flat_historical,
            strategy_priors=flat_strategy,
            selection_weights=uniform_weights,
        )
        print("=" * 72)
        print(f"  {label}: expected {expected}")
        print(f"  prompt: {text[:80]!r}")
        print(f"  dual_track: {_summarise_dual_track(dual_track_snap)}")
        if evaluation_value is not None:
            metrics = {
                s.metric_name: s.value
                for s in evaluation_value.turn_scores
            }
            print(
                f"  eval: warmth={metrics.get('warmth', 0):.2f} "
                f"support={metrics.get('support_presence', 0):.2f} "
                f"task_pressure={metrics.get('task_pressure', 0):.2f} "
                f"cross_stab={metrics.get('cross_track_stability', 0):.2f} "
                f"info={metrics.get('info_integration', 0):.2f}"
            )
        print("  base_scores (uniform priors, ranked):")
        won = ranked[0][0]
        for regime_id, score in ranked:
            tick = " <-- WON" if regime_id == won else ""
            hit = " (in expected)" if regime_id in expected else ""
            print(f"    {regime_id:<28}{score:>6.3f}{tick}{hit}")
        gap_top1_top2 = ranked[0][1] - ranked[1][1]
        print(f"  margin top1-top2: {gap_top1_top2:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
