"""Phase 2 W2.0c (debt #10B) — slow real-Qwen smoke for the
``--use-llm-semantic-runtime`` longitudinal path.

This test exercises the SAME wiring as
``test_llm_semantic_runtime_evidence_chain.py`` but through the
``lifeform-bench`` CLI on the actual Qwen 2.5 1.5B Instruct model,
not a fake provider. It is the regression guard for chunk 5's
evidence-run script: if this test passes locally then
``examples/run_cross_session_probe_llm.py`` will produce a
non-degenerate ``cross_session_probe_llm.json`` artifact.

Skipped by default because:

* Loading Qwen 1.5B in bfloat16 takes ~15 s on a warm cache and
  several minutes on a cold cache (~3 GB download).
* Each turn under the LLM semantic runtime adds ~1-2 short
  generations (~16 tokens each) on top of the residual capture,
  so a 2-round run with 3 turns/round runs in ~3-5 min on CPU.

Enable explicitly when you have Qwen cached locally:

    $env:VZ_RUN_LLM_SMOKE = "1"
    python -m pytest tests/lifeform_e2e/test_longitudinal_with_llm_runtime_smoke.py -q
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest


_REASON = (
    "Set VZ_RUN_LLM_SMOKE=1 to enable. Requires Qwen/Qwen2.5-1.5B-Instruct "
    "in HF cache (or network access for first download); each test takes "
    "~3-5 min on CPU."
)

_SHOULD_RUN = os.environ.get("VZ_RUN_LLM_SMOKE") == "1"


@pytest.mark.skipif(not _SHOULD_RUN, reason=_REASON)
def test_longitudinal_companion_with_llm_runtime_produces_records(
    tmp_path: pathlib.Path,
) -> None:
    """End-to-end smoke: 2 rounds of the low-mood-disclosure scenario
    through ``lifeform-bench --vertical companion --longitudinal-rounds 2
    --use-llm-semantic-runtime``. Exit code must be 0 (or 1 only because
    of the legitimate-skip family pass that depends on regime accuracy
    on tiny scenario sets), and the longitudinal artifact must show
    ``tom_records_total_last > 0`` proving the EQ chain stayed wired
    across both rounds.
    """
    from lifeform_evolution.cli import main as bench_main

    artifact_path = tmp_path / "cross_session_probe_llm_smoke.json"
    exit_code = bench_main(
        [
            "--vertical",
            "companion",
            "--longitudinal-rounds",
            "2",
            "--longitudinal-report",
            "--longitudinal-json",
            str(artifact_path),
            "--use-llm-semantic-runtime",
            # Keep the per-turn regime-match bar low; this test is
            # about EQ activation, not regime accuracy.
            "--min-regime-match-rate",
            "0.0",
            # Stick with the default low-mood-disclosure scenario
            # — it is the smallest scenario that exercises the full
            # disclosure / pressure / repair surface so the LLM
            # extractor has something to bite on.
        ]
    )

    # Exit code can legitimately be 1 if the longitudinal il_rapport
    # gate fails (debt #10C is still open per the plan), so we do
    # not assert exit_code == 0 here. We assert the *artifact*
    # which is the load-bearing observable.
    assert exit_code in (0, 1), (
        f"unexpected exit_code {exit_code} (only 0 or 1 acceptable; "
        "0 = full pass, 1 = il_rapport_trend gate not yet met "
        "which is debt #10C, not 10B)"
    )
    assert artifact_path.exists(), (
        f"longitudinal artifact missing at {artifact_path}; the "
        "CLI did not write one — wiring may be broken"
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]
    assert scenarios, "longitudinal artifact has no scenarios"
    first = scenarios[0]
    assert first["rounds"] == 2

    # Load-bearing observable: at least the LAST round produced
    # ToM records. We allow round-0 (cold start) to be 0 because
    # the LLM extractor needs upstream evidence (interlocutor +
    # role) to anchor its proposals.
    tom_last = first.get("tom_records_total_last")
    assert tom_last is not None and tom_last > 0, (
        "real-Qwen longitudinal smoke produced 0 ToM records in "
        f"the final round (tom_records_total_last={tom_last!r}); "
        "the LLM semantic runtime is wired but Qwen is not "
        "extracting any ToM observations — likely a prompt / "
        "parse-format regression"
    )

    # Round-by-round visibility — useful for debugging when the
    # smoke fails. Not gated.
    per_round_tom = first.get("per_round_tom_records_total") or []
    print(
        f"[smoke] per_round_tom_records_total = {per_round_tom!r} "
        f"trend={first.get('tom_records_total_trend')!r}"
    )
    per_round_cg = first.get("per_round_common_ground_dyad_atoms_total") or []
    print(
        f"[smoke] per_round_common_ground_dyad_atoms_total = "
        f"{per_round_cg!r} trend="
        f"{first.get('common_ground_dyad_atoms_total_trend')!r}"
    )


