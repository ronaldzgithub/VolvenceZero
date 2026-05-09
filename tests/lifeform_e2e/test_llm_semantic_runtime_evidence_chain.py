"""Phase 2 W2.0c (debt #10B) — EQ owner activation evidence chain.

Two paired tests pin the load-bearing observable for known-debt #10B
item 2: under the default ``NoOpSemanticProposalRuntime`` the four
about-other ToM owners and the common-ground owner stay empty
(``f3.tom_records_total == 0`` / ``f3.common_ground_dyad_atoms_total
== 0``); under an ``LLMSemanticProposalRuntime`` they flip non-zero.
The 0 -> > 0 transition is the runtime-gated signal that proves the
EQ owner chain is wired correctly end-to-end through the lifeform
benchmark surface.

The fake text provider returns deterministic JSON for each of the
three downstream consumers:

* ``LLMToMProposalRuntime`` calls with ``_TOM_PROMPT`` (contains
  the literal ``target_slot``)
* ``LLMCommonGroundProposalRuntime`` calls with ``_COMMON_GROUND_PROMPT``
  (contains the literal ``scope_kind``)
* ``LLMSemanticProposalRuntime`` (commitment classifier) calls with
  ``_COMMITMENT_PROMPT`` (contains the literal ``operation`` and a
  JSON object schema, NOT a JSON array)

Dispatching on prompt content keeps the fake provider a single
class instead of three; the prompts themselves are the contract
boundary so this is robust to refactors that don't change the
prompts' top-level structure.

These tests run in seconds and do NOT load Qwen / require torch /
require an HF cache. The slow real-Qwen smoke lives in
``test_longitudinal_with_llm_runtime_smoke.py`` and is skipped
unless ``VZ_RUN_LLM_SMOKE=1``.
"""

from __future__ import annotations

import asyncio

import pytest

from lifeform_evolution.benchmark import (
    BenchmarkReport,
    ScriptedScenario,
    ScriptedTurn,
    run_benchmark_async,
)
from lifeform_evolution.family_report import FamilyId, compute_family_report


_FAKE_TOM_PAYLOAD = (
    "["
    '{"target_slot": "belief_about_other", '
    '"summary": "user thinks the deadline moved", '
    '"detail": "user mentioned the new schedule", '
    '"evidence": "deadline moved", '
    '"confidence": 0.8, '
    '"control_signal": 0.3},'
    '{"target_slot": "intent_about_other", '
    '"summary": "user wants concise replies", '
    '"detail": "user prefers short answers under load", '
    '"evidence": "be brief", '
    '"confidence": 0.7, '
    '"control_signal": 0.25},'
    '{"target_slot": "feeling_about_other", '
    '"summary": "user feels heavy", '
    '"detail": "user described low energy", '
    '"evidence": "feeling stuck", '
    '"confidence": 0.75, '
    '"control_signal": 0.4},'
    '{"target_slot": "preference_about_other", '
    '"summary": "user prefers being heard first", '
    '"detail": "user wants empathy before solutions", '
    '"evidence": "just heard", '
    '"confidence": 0.7, '
    '"control_signal": 0.3}'
    "]"
)


_FAKE_COMMON_GROUND_PAYLOAD = (
    "["
    '{"scope_kind": "dyad", '
    '"scope_id": "self:primary", '
    '"summary": "We agreed to slow the pace.", '
    '"accepted_by_ids": ["self", "primary"], '
    '"evidence": "let us slow down", '
    '"confidence": 0.78, '
    '"recursion_depth": 1, '
    '"control_signal": 0.35}'
    "]"
)


_FAKE_COMMITMENT_PAYLOAD = (
    '{"operation": "observe", '
    '"alignment_evidence": "no commitment-shaped move in this turn", '
    '"confidence": 0.6}'
)


class _FakeRoutingProvider:
    """Deterministic text provider that dispatches on prompt content.

    The three downstream LLM-driven runtimes (commitment classifier,
    ToM extractor, common-ground extractor) ship distinct prompt
    templates. We discriminate by literal substrings of those
    templates (``operation`` / ``target_slot`` / ``scope_kind``)
    and return the matching payload. ``call_log`` lets the test
    surface which prompts actually reached the provider — useful
    for diagnosing wiring regressions.
    """

    def __init__(self) -> None:
        self.call_log: list[str] = []

    def generate(
        self, *, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0
    ) -> str:
        del max_new_tokens, temperature
        if "scope_kind" in prompt:
            tag = "common_ground"
            payload = _FAKE_COMMON_GROUND_PAYLOAD
        elif "target_slot" in prompt:
            tag = "tom"
            payload = _FAKE_TOM_PAYLOAD
        elif "operation" in prompt:
            tag = "commitment"
            payload = _FAKE_COMMITMENT_PAYLOAD
        else:
            tag = "unknown"
            payload = ""
        self.call_log.append(tag)
        return payload


def _scenario() -> ScriptedScenario:
    return ScriptedScenario(
        scenario_id="llm-semantic-runtime-evidence-chain",
        description=(
            "Three short turns covering disclosure, request, and a low-key "
            "preference cue so the ToM / common-ground extractors have "
            "input. The expected_regime_in is intentionally permissive — "
            "this test is about EQ owner activation, not regime accuracy."
        ),
        turns=(
            ScriptedTurn(user_input="Hey - things have been heavy lately."),
            ScriptedTurn(
                user_input="Can we slow down a bit and talk before planning?"
            ),
            ScriptedTurn(
                user_input="I prefer being heard first before getting any steps."
            ),
        ),
    )


def _run_with_runtime(*, runtime: object | None) -> BenchmarkReport:
    from lifeform_domain_emogpt import build_companion_lifeform
    from volvence_zero.memory import build_default_memory_store

    shared_store = build_default_memory_store()
    lifeform = build_companion_lifeform(
        memory_store=shared_store,
        semantic_proposal_runtime=runtime,
    )
    return asyncio.run(
        run_benchmark_async(scenario=_scenario(), lifeform=lifeform)
    )


def test_eq_owner_chain_inert_with_noop_semantic_runtime() -> None:
    """Matched control: ``NoOpSemanticProposalRuntime`` keeps EQ
    owner records at 0. This is the legacy default behaviour and
    proves the chain is not "always on" — the LLM runtime is the
    activator.
    """
    from volvence_zero.semantic_state import NoOpSemanticProposalRuntime

    bench = _run_with_runtime(runtime=NoOpSemanticProposalRuntime())

    assert bench.tom_records_total == 0, (
        "NoOp semantic runtime must keep ToM owner records empty; "
        f"got tom_records_total={bench.tom_records_total} (chain "
        f"unexpectedly activated without an LLM runtime)"
    )
    assert bench.common_ground_dyad_atoms_total == 0, (
        "NoOp semantic runtime must keep LLM-derived common-ground "
        "dyad atoms empty (upstream-derived atoms can still appear "
        "but the typed cold-start companion path produces 0); "
        f"got common_ground_dyad_atoms_total="
        f"{bench.common_ground_dyad_atoms_total}"
    )

    family = compute_family_report(bench=bench)
    f3 = family.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    tom_metric = next(m for m in f3.metrics if m.metric_id == "f3.tom_records_total")
    cg_metric = next(
        m for m in f3.metrics if m.metric_id == "f3.common_ground_dyad_atoms_total"
    )
    assert tom_metric.value == 0.0
    assert cg_metric.value == 0.0
    # Family pass must NOT depend on these diagnostic metrics — they
    # have no threshold, so absent / zero values do not flip the
    # family to FAIL on legacy harnesses.
    assert tom_metric.threshold is None
    assert cg_metric.threshold is None

    # Wave E1 (debt #10B item 3): NoOp semantic runtime ->
    # ``proposal_attempts_total == 0``. The diagnostic counters
    # distinguish "the LLM was never called" (this branch) from
    # "the LLM was called but parse failed" (a real Qwen failure
    # mode). Both surface as tom_records_total == 0 without these
    # counters.
    assert bench.tom_proposal_attempts_total == 0, (
        "NoOp semantic runtime should not call any LLM provider; "
        f"got tom_proposal_attempts_total="
        f"{bench.tom_proposal_attempts_total}"
    )
    assert bench.common_ground_proposal_attempts_total == 0
    assert bench.tom_proposal_last_parse_status == "no_call"
    assert bench.common_ground_proposal_last_parse_status == "no_call"


def test_eq_owner_chain_activates_with_llm_semantic_runtime() -> None:
    """Treatment arm: an ``LLMSemanticProposalRuntime`` (with a fake
    JSON-emitting provider) activates the four ToM owners + the
    common-ground LLM proposal source, and the activation is
    visible at the ``BenchmarkReport`` + ``FamilyReport`` surfaces.
    """
    from volvence_zero.semantic_state.llm_runtime import (
        LLMSemanticProposalRuntime,
    )

    fake = _FakeRoutingProvider()
    bench = _run_with_runtime(
        runtime=LLMSemanticProposalRuntime(provider=fake)
    )

    # Activation evidence #1: at least one ToM owner produced records.
    # We do not assert a specific count because de-duplication +
    # confidence gating can collapse repeated identical proposals
    # across the three turns into a smaller set; the load-bearing
    # observable is "> 0".
    assert bench.tom_records_total > 0, (
        "LLM semantic runtime should activate at least one ToM "
        "owner; got tom_records_total=0 even with the fake provider "
        f"wired (call_log={fake.call_log!r})"
    )
    # Activation evidence #2: at least one common-ground dyad atom
    # appeared. The companion vertical defaults to a single
    # interlocutor (``primary``) so the dyad is always
    # (self, primary); the fake provider emits exactly one matching
    # atom per turn, but per-turn de-dup may collapse them.
    assert bench.common_ground_dyad_atoms_total > 0, (
        "LLM common-ground proposal runtime should produce >= 1 "
        "dyad atom on the companion vertical; got "
        f"common_ground_dyad_atoms_total=0 "
        f"(call_log={fake.call_log!r})"
    )

    # Activation evidence #3: the family report mirrors the bench.
    family = compute_family_report(bench=bench)
    f3 = family.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    tom_metric = next(m for m in f3.metrics if m.metric_id == "f3.tom_records_total")
    cg_metric = next(
        m for m in f3.metrics if m.metric_id == "f3.common_ground_dyad_atoms_total"
    )
    assert tom_metric.value == float(bench.tom_records_total)
    assert cg_metric.value == float(bench.common_ground_dyad_atoms_total)

    # Wave E1 (debt #10B item 3) activation evidence #4: the typed
    # diagnostic counters surface on the bench report and the
    # family report. With a JSON-emitting fake provider, attempts
    # > 0 AND parsed_ok > 0 AND parse_errors == 0 (the fake always
    # returns well-formed JSON).
    assert bench.tom_proposal_attempts_total > 0, (
        "Diagnostic counter should record ToM LLM provider calls; "
        f"got tom_proposal_attempts_total=0 with call_log={fake.call_log!r}"
    )
    assert bench.tom_proposal_parsed_ok_total > 0, (
        "Diagnostic counter should record successful ToM parse "
        f"calls; got tom_proposal_parsed_ok_total=0 with "
        f"attempts_total={bench.tom_proposal_attempts_total}"
    )
    assert bench.tom_proposal_parse_errors_total == 0, (
        "Fake provider always returns well-formed JSON; "
        f"unexpected parse errors: {bench.tom_proposal_parse_errors_total}"
    )
    assert bench.common_ground_proposal_attempts_total > 0, (
        "Diagnostic counter should record common-ground LLM provider "
        f"calls; got common_ground_proposal_attempts_total=0 with "
        f"call_log={fake.call_log!r}"
    )
    assert bench.tom_proposal_last_parse_status == "ok", (
        "Last parse status should be ``ok`` with a JSON-emitting "
        f"fake provider; got {bench.tom_proposal_last_parse_status!r}"
    )

    # Option B / debt #10B item 3 follow-up: Layer 1 (per-session
    # runtimes) means the typed accumulator is monotonically
    # cumulative across the 3 scripted turns of this scenario instead
    # of resetting every turn under the old per-turn-rebuild path.
    # Pre-fix this would be 1 (last turn only) for both attempts
    # totals; post-fix it is exactly 3 (one provider call per turn,
    # courtesy of the per-turn cache inside ``LLMToMProposalRuntime``
    # / ``LLMCommonGroundProposalRuntime``).
    expected_turns = len(_scenario().turns)
    assert bench.tom_proposal_attempts_total == expected_turns, (
        f"Layer 1 (per-session ToM runtime) should produce one call "
        f"per turn cumulative; got {bench.tom_proposal_attempts_total} "
        f"vs expected {expected_turns}. A value of 1 indicates a "
        f"regression to the per-turn-rebuild path."
    )
    assert bench.common_ground_proposal_attempts_total == expected_turns, (
        f"Layer 1 (per-session common-ground runtime) should produce "
        f"one call per turn cumulative; got "
        f"{bench.common_ground_proposal_attempts_total} vs expected "
        f"{expected_turns}. A value of 1 indicates a regression to "
        f"the per-turn-rebuild path."
    )

    tom_attempts_metric = next(
        m for m in f3.metrics if m.metric_id == "f3.tom_proposal_attempts_total"
    )
    tom_parsed_ok_metric = next(
        m for m in f3.metrics if m.metric_id == "f3.tom_proposal_parsed_ok_total"
    )
    cg_attempts_metric = next(
        m
        for m in f3.metrics
        if m.metric_id == "f3.common_ground_proposal_attempts_total"
    )
    assert tom_attempts_metric.value == float(bench.tom_proposal_attempts_total)
    assert tom_parsed_ok_metric.value == float(bench.tom_proposal_parsed_ok_total)
    assert cg_attempts_metric.value == float(
        bench.common_ground_proposal_attempts_total
    )

    # Sanity: the fake provider was actually called for both ToM
    # and common-ground prompts. If only commitment prompts
    # arrived, the auto-wiring in ``final_wiring.py`` would be
    # broken and the assertions above would still pass via dumb
    # luck on a single turn. We pin the wiring explicitly here.
    assert "tom" in fake.call_log, (
        "fake provider received no ToM prompt; "
        f"call_log={fake.call_log!r} — auto-wired LLMToMProposalRuntime "
        "may have been dropped"
    )
    assert "common_ground" in fake.call_log, (
        "fake provider received no common-ground prompt; "
        f"call_log={fake.call_log!r} — auto-wired "
        "LLMCommonGroundProposalRuntime may have been dropped"
    )


def test_eq_owner_chain_treatment_strictly_dominates_control() -> None:
    """The activation difference is a one-way step: the LLM arm
    must produce STRICTLY MORE records than the NoOp arm. This
    keeps the test robust to baseline drift (e.g. if companion
    vertical ever defaults to wiring a non-LLM proposal source
    that produces small counts, the gap should still be positive).
    """
    from volvence_zero.semantic_state import NoOpSemanticProposalRuntime
    from volvence_zero.semantic_state.llm_runtime import (
        LLMSemanticProposalRuntime,
    )

    control = _run_with_runtime(runtime=NoOpSemanticProposalRuntime())
    treatment = _run_with_runtime(
        runtime=LLMSemanticProposalRuntime(provider=_FakeRoutingProvider())
    )

    assert treatment.tom_records_total > control.tom_records_total, (
        "LLM treatment arm should produce more ToM records than "
        f"NoOp control; got treatment={treatment.tom_records_total} "
        f"control={control.tom_records_total}"
    )
    assert (
        treatment.common_ground_dyad_atoms_total
        > control.common_ground_dyad_atoms_total
    ) or (
        # Fallback: companion vertical may produce upstream-derived
        # dyad atoms in BOTH arms (typed BELIEF -> common-ground).
        # In that case we still need treatment >= control + 1 atom
        # from the LLM proposal source.
        treatment.common_ground_dyad_atoms_total
        >= control.common_ground_dyad_atoms_total + 1
    ), (
        "LLM treatment arm should produce at least one more "
        "common-ground dyad atom than NoOp control; got "
        f"treatment={treatment.common_ground_dyad_atoms_total} "
        f"control={control.common_ground_dyad_atoms_total}"
    )


def test_use_llm_semantic_runtime_requires_longitudinal_rounds() -> None:
    """argparse-level validation: --use-llm-semantic-runtime without
    --longitudinal-rounds > 0 must exit non-zero. Pinning this here
    keeps the chunk-5 evidence script honest — a missing flag is
    much cheaper to catch at parse time than after Qwen loads.
    """
    from lifeform_evolution.cli import main as bench_main

    with pytest.raises(SystemExit) as exc:
        bench_main(
            [
                "--vertical",
                "companion",
                "--use-llm-semantic-runtime",
            ]
        )
    assert exc.value.code != 0


def test_use_llm_semantic_runtime_requires_companion_vertical() -> None:
    """argparse-level validation: --use-llm-semantic-runtime with
    --vertical coding must exit non-zero. Coding does not currently
    expose a real-substrate builder that takes memory_store; debt
    #10A's coding follow-up is the prerequisite for this gate.
    """
    from lifeform_evolution.cli import main as bench_main

    with pytest.raises(SystemExit) as exc:
        bench_main(
            [
                "--vertical",
                "coding",
                "--longitudinal-rounds",
                "2",
                "--use-llm-semantic-runtime",
            ]
        )
    assert exc.value.code != 0


@pytest.mark.parametrize("metric_id", ("f3.tom_records_total", "f3.common_ground_dyad_atoms_total"))
def test_diagnostic_metrics_never_have_thresholds(metric_id: str) -> None:
    """These metrics are diagnostic only — they must NEVER gain a
    threshold (which would flip every legacy NoOp harness to FAIL).
    Any future change adding a threshold needs to update both this
    test and ``docs/known-debts.md`` debt #10B.
    """
    from volvence_zero.semantic_state import NoOpSemanticProposalRuntime

    bench = _run_with_runtime(runtime=NoOpSemanticProposalRuntime())
    family = compute_family_report(bench=bench)
    f3 = family.family(FamilyId.F3_RELATIONSHIP_CONTINUITY)
    metric = next(m for m in f3.metrics if m.metric_id == metric_id)
    assert metric.threshold is None, (
        f"{metric_id} must stay diagnostic (threshold=None) so the "
        f"NoOp default does not gate F3 family pass. Got "
        f"threshold={metric.threshold!r}."
    )
