"""Verify combined effect of phase B (larger substrate) + phase A
(LLM-driven SemanticProposalRuntime) on the companion vertical.

Runs an 8-prompt diagnostic that interleaves regime-shaping turns
(``hello`` / ``task`` / ``emotion`` / ``rupture`` / ``repair``)
with commitment-shaping turns (``commit_create`` / ``commit_complete``
/ ``commit_block``). Both phases share one loaded Qwen model in
RAM \u2014 substrate residual capture and LLM commitment classification
read the same weights.

The new evidence this verify adds vs.
``companion_verify_recalibrated_bootstraps.py``:

* **commitment lifecycle counters**. With the NoOp runtime the
  commitment owner only ever sees OBSERVE proposals \u2014 no
  ``advocacy_proposed_count``, no ``outcome_completed_count``, no
  ``outcome_rejected_count``. With the LLM runtime the user-side
  commitment classification flows into the AAC lifecycle and these
  counters become non-zero, which is the load-bearing observable
  for slice 2a phase A.
* **per-turn proposal trace**. Each turn we print whichever
  commitment-slot proposal the LLM produced (or "OBSERVE" /
  "[NoOp]" for non-LLM lifeforms). That trace is the contract
  between the LLM runtime and the rest of the kernel \u2014 if the
  LLM is mis-classifying we see it directly here, not only in the
  downstream regime distribution.

Default model: ``Qwen/Qwen2.5-0.5B-Instruct`` because this verify
runs on tight-RAM diagnostic boxes; the runtime defaults in
``real_substrate.py`` are 1.5B for production. Override with
``--model-source`` when running on a host with more headroom.

Usage:

    python examples/companion_verify_llm_semantic_runtime.py
    python examples/companion_verify_llm_semantic_runtime.py \\
        --model-source Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from lifeform_domain_emogpt import build_companion_lifeform_with_real_substrate


# (label, user_text, expected_regime_set, kind)
# kind in {"regime", "commitment"} \u2014 lets us isolate the metric
# the prompt is meant to probe so the report reads cleanly.
_PROBES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "hello",
        "Hey - it's been a while.",
        ("acquaintance_building", "casual_social"),
        "regime",
    ),
    (
        "task",
        "Can you help me draft a polite email declining a meeting invite?",
        ("problem_solving", "guided_exploration"),
        "regime",
    ),
    (
        "commit_create",
        "I want to commit to writing for thirty minutes every morning starting tomorrow.",
        (),
        "commitment",
    ),
    (
        "emotion",
        "Honestly I've been struggling with sleep, low-energy mornings, "
        "and I keep circling around freelancing but I'm scared.",
        ("emotional_support", "guided_exploration"),
        "regime",
    ),
    (
        "commit_complete",
        "Quick update - I did my thirty minutes of writing today, finally.",
        (),
        "commitment",
    ),
    (
        "rupture",
        "Wait - that just felt clinical and procedural. I'm not asking "
        "you to optimise me.",
        ("repair_and_deescalation", "emotional_support"),
        "regime",
    ),
    (
        "commit_block",
        "Sorry I didn't actually keep up with the daily writing this week.",
        (),
        "commitment",
    ),
    (
        "repair",
        "OK. Sorry. Can we just back up. I just needed to say it out loud.",
        ("repair_and_deescalation", "emotional_support"),
        "regime",
    ),
)


def _entropy_nats(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        if v <= 0:
            continue
        p = v / total
        h -= p * math.log(p)
    return h


def _commitment_proposal_label(session) -> str:
    """Pull the *current-turn* commitment proposal label, if any.

    The kernel's per-turn semantic processing surfaces the runtime's
    proposal batch in the ``debug_state`` of the commitment slot.
    We don't depend on internal layout of the snapshot value beyond
    the description string; missing data degrades gracefully to a
    placeholder.
    """
    snap = session.latest_active_snapshots.get("commitment")
    if snap is None:
        return "[no-snapshot]"
    description = getattr(snap.value, "description", "") or ""
    if "llm-detected-new-commitment" in description.lower() or "llm-detected" in description.lower():
        return "llm"
    if "lifecycle entries" in description.lower():
        return "active"
    return "[observed]"


async def verify(
    *,
    model_source: str,
    model_id: str,
    use_llm: bool,
    fallback: bool,
) -> int:
    print()
    print("=" * 70)
    print(f"  PHASE B+A VERIFY  |  use_llm_semantic_runtime={use_llm}")
    print(f"  model_source: {model_source}")
    print("=" * 70)
    bundle = build_companion_lifeform_with_real_substrate(
        model_source=model_source,
        model_id=model_id,
        use_llm_semantic_runtime=use_llm,
        fallback_to_builtin=fallback,
        local_files_only=False,
    )
    print(f"  status: {bundle.status_label}")
    print(f"  is_real_substrate: {bundle.is_real_substrate}")
    print(f"  llm_semantic_runtime: {bundle.llm_semantic_runtime is not None}")
    if not bundle.is_real_substrate:
        print(f"  WARN: substrate degraded ({bundle.runtime_origin}); "
              f"results below are from random-weights fallback.",
              file=sys.stderr)
    session = bundle.lifeform.create_session(session_id="companion-verify-llm")
    print()
    header = (
        f"  {'phase':<16}{'kind':<11}{'regime':<26}"
        f"{'rapp':>6}{'res':>6}{'trust':>8}  {'commit-prop':<14} hit?"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    landed_regimes: list[str] = []
    regime_hits = 0
    regime_total = 0
    commitment_proposals_per_kind: list[tuple[str, str]] = []
    for label, text, expected, kind in _PROBES:
        await session.run_turn(text)
        snap = session.latest_active_snapshots.get("regime")
        regime = (
            snap.value.active_regime.regime_id if snap is not None else "?"
        )
        landed_regimes.append(regime)
        st = session.interlocutor_state
        commit_prop = _commitment_proposal_label(session)
        commitment_proposals_per_kind.append((kind, commit_prop))
        hit = "."
        if kind == "regime":
            regime_total += 1
            if regime in expected:
                regime_hits += 1
                hit = "Y"
        print(
            f"  {label:<16}{kind:<11}{regime:<26}"
            f"{st.rapport_warmth:>6.2f}{st.resistance_level:>6.2f}"
            f"{st.trust_signal:>+8.2f}  {commit_prop:<14} {hit}"
        )

    counts = Counter(landed_regimes)

    commitment_snap = session.latest_active_snapshots.get("commitment")
    print()
    print(f"  regime hits (regime-kind probes only): "
          f"{regime_hits}/{regime_total}")
    print(f"  regime distribution: {dict(counts)}")
    print(f"  unique regimes selected: {len(counts)}")
    print(f"  regime distribution entropy: {_entropy_nats(counts):.3f} nats")
    print()
    if commitment_snap is not None:
        cs = commitment_snap.value
        print(f"  commitment.active_commitments: "
              f"{len(getattr(cs, 'active_commitments', ()))}")
        print(f"  commitment.lifecycle_entries:  "
              f"{len(getattr(cs, 'lifecycle_entries', ()))}")
        print(f"  commitment.advocacy_proposed_count: "
              f"{getattr(cs, 'advocacy_proposed_count', 0)}")
        print(f"  commitment.advocacy_ready_count:    "
              f"{getattr(cs, 'advocacy_ready_count', 0)}")
        print(f"  commitment.outcome_progressed_count:"
              f" {getattr(cs, 'outcome_progressed_count', 0)}")
        print(f"  commitment.outcome_completed_count: "
              f"{getattr(cs, 'outcome_completed_count', 0)}")
        print(f"  commitment.outcome_rejected_count:  "
              f"{getattr(cs, 'outcome_rejected_count', 0)}")
        print(f"  commitment.outcome_stalled_count:   "
              f"{getattr(cs, 'outcome_stalled_count', 0)}")
    else:
        print("  WARN: no commitment snapshot exposed", file=sys.stderr)

    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-source", type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model source. Default Qwen 0.5B for tight-RAM hosts; "
             "set to Qwen/Qwen2.5-1.5B-Instruct on hosts with >=6 GB free.",
    )
    parser.add_argument(
        "--model-id", type=str, default="qwen2.5-0.5b-instruct",
        help="Internal model id label. Should match the model_source.",
    )
    parser.add_argument(
        "--no-llm-semantic", action="store_true",
        help="Disable LLM-driven SemanticProposalRuntime (NoOp baseline).",
    )
    parser.add_argument(
        "--fallback", action="store_true",
        help="Allow falling back to builtin random-weights GPT-2 if HF "
             "load fails. Off by default: this is a verify, we want to "
             "fail loud rather than report degraded numbers as real.",
    )
    args = parser.parse_args()
    return asyncio.run(
        verify(
            model_source=args.model_source,
            model_id=args.model_id,
            use_llm=not args.no_llm_semantic,
            fallback=args.fallback,
        )
    )


if __name__ == "__main__":
    raise SystemExit(_cli())
