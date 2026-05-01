"""Probe what the LLM classifies each turn as.

The phase B+A verify shows ``outcome_rejected_count: 6`` against
just ONE explicit ``commit_block`` probe. That implies the LLM is
classifying emotional / regime / hello content as BLOCK (or
CREATE / COMPLETE), not as OBSERVE \u2014 i.e. the noise lives at
the classification layer, not the OBSERVE-pollution layer the
``min_proposal_confidence=0.40`` filter targets.

This script bypasses the kernel pipeline and asks the LLM provider
directly, dumping per-turn raw output + parsed label so we can
decide between (a) prompt engineering, (b) model upgrade,
(c) prior-commitment-state conditioning.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from lifeform_domain_emogpt.real_substrate import (
    build_companion_lifeform_with_real_substrate,
)
from volvence_zero.semantic_state.llm_runtime import (
    _COMMITMENT_PROMPT,
    _parse_commitment_label,
)


_PROBES: tuple[tuple[str, str, str], ...] = (
    ("hello", "Hey - it's been a while.", "observe"),
    ("task", "Can you help me draft a polite email declining a meeting invite?", "observe"),
    ("commit_create", "I want to commit to writing for thirty minutes every morning starting tomorrow.", "create"),
    ("emotion",
     "Honestly I've been struggling with sleep, low-energy mornings, "
     "and I keep circling around freelancing but I'm scared.", "observe"),
    ("commit_complete", "Quick update - I did my thirty minutes of writing today, finally.", "complete"),
    ("rupture",
     "Wait - that just felt clinical and procedural. I'm not asking "
     "you to optimise me.", "observe"),
    ("commit_block", "Sorry I didn't actually keep up with the daily writing this week.", "block"),
    ("repair", "OK. Sorry. Can we just back up. I just needed to say it out loud.", "observe"),
)


def main() -> int:
    bundle = build_companion_lifeform_with_real_substrate(
        model_source="Qwen/Qwen2.5-0.5B-Instruct",
        model_id="qwen2.5-0.5b-instruct",
        use_llm_semantic_runtime=True,
        fallback_to_builtin=False,
        local_files_only=True,
    )
    provider = bundle.llm_semantic_runtime._provider
    print()
    print("  probing LLM commitment classifier on Qwen 0.5B")
    print(f"  {'phase':<16}{'expected':<10}{'parsed':<10}  raw")
    print("  " + "-" * 88)
    correct = 0
    for label, text, expected in _PROBES:
        prompt = _COMMITMENT_PROMPT.format(user_input=text[:600])
        raw = provider.generate(prompt=prompt, max_new_tokens=8, temperature=0.0)
        parsed = _parse_commitment_label(raw)
        parsed_label = parsed.value if parsed else "FALLBACK"
        if parsed_label == expected:
            correct += 1
        # raw is a single token-or-two; print as-is.
        print(f"  {label:<16}{expected:<10}{parsed_label:<10}  {raw!r}")
    print()
    print(f"  accuracy: {correct}/{len(_PROBES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
