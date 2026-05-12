"""Wave F closure: synthesizer.synthesize wires L1 / L3 / L4 enforcement.

Three load-bearing assertions:

1. **L4 pre-check**: when the bundle's coverage_map classifies the
   query as out-of-domain or boundary-blocked AND the policy is
   STRICT_REFUSE, the synthesizer must short-circuit with the
   refusal text before the LLM runs.
2. **L1 style hint surfacing**: the synthesizer's rationale_tags
   must carry an ``l1_style_prior=...`` tag identifying the figure
   whose voice prior conditioned the turn.
3. **L3 post-verify**: the synthesizer's rationale_tags must
   carry an ``l3_grounded_verify=...`` tag whose verdict is the
   GroundedDecoder's output for the generated text.

We do not require an actual LLM here — we stub
:class:`LLMResponseSynthesizer.synthesize` with a deterministic
echo so the enforcer wiring is observable. The persona-LoRA
activation path is covered separately in the substrate tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from volvence_zero.agent.response import AgentResponse, ResponseContext

from lifeform_domain_figure import (
    FigureBundleInputs,
    FigureCorpusSourceBundle,
    build_einstein_profile,
    build_figure_artifact_bundle,
    build_figure_ingestion_envelope,
    synthetic_einstein_corpus,
)
from lifeform_expression.llm_synthesizer import LifeformLLMResponseSynthesizer


# ---------------------------------------------------------------------------
# Stubs: a fake LLM runtime + synthesizer subclass that side-steps the kernel
# substrate plumbing so we can drive .synthesize() without HF dependencies.
# ---------------------------------------------------------------------------


@dataclass
class _FakeRuntime:
    model_id: str = "fake-runtime"

    def generate(self, **_kwargs):
        from volvence_zero.substrate.residual_contracts import GenerationResult

        return GenerationResult(
            text="Spatially separated subsystems each carry their own definite physical state.",
            token_count=8,
            capture=None,
            description="fake generation",
        )


class _StubSynthesizer(LifeformLLMResponseSynthesizer):
    """Drop-in for tests that bypasses the kernel super().synthesize()."""

    def __init__(self, *, figure_bundle=None, fake_text=None) -> None:
        super().__init__(runtime=_FakeRuntime(), figure_bundle=figure_bundle)
        self._fake_text = fake_text or "Reality is independent of observation."

    def _render_response(
        self,
        *,
        context: ResponseContext,
        assembly,
        prompt: str,
        chat_messages,
    ) -> AgentResponse:
        del context, assembly, prompt, chat_messages
        return AgentResponse(
            text=self._fake_text,
            regime_id="test-regime",
            abstract_action="answer",
            rationale="stub",
            rationale_tags=("stub",),
        )


# Override the LLMResponseSynthesizer.synthesize call chain just enough that
# the figure-vertical L1/L3/L4 wrapping path runs without spinning up a real
# substrate. We replace the inherited synthesize-side prompt assembly with a
# direct return.
def _patch_super_synthesize(monkeypatch):
    from volvence_zero.agent import response as response_mod

    def _stub_synthesize(self, *, context, assembly=None):
        return AgentResponse(
            text=self._fake_text if hasattr(self, "_fake_text") else "stub",
            regime_id="test-regime",
            abstract_action="answer",
            rationale="stub super",
            rationale_tags=("stub_super",),
        )

    monkeypatch.setattr(
        response_mod.LLMResponseSynthesizer,
        "synthesize",
        _stub_synthesize,
        raising=True,
    )


# ---------------------------------------------------------------------------
# Bundle fixtures
# ---------------------------------------------------------------------------


def _einstein_bundle():
    profile = build_einstein_profile()
    papers, letters, lectures, notebooks = synthetic_einstein_corpus()
    envelopes = build_figure_ingestion_envelope(
        FigureCorpusSourceBundle(
            figure_id="einstein",
            papers=papers,
            letters=letters,
            lectures=lectures,
            notebooks=notebooks,
        ),
        uploader="contract-test:enforcers",
    ).envelopes
    return build_figure_artifact_bundle(
        FigureBundleInputs(profile=profile, envelopes=envelopes)
    )


def _context(text: str) -> ResponseContext:
    return ResponseContext(
        regime_id="default",
        regime_name="default",
        regime_switched=False,
        abstract_action="answer",
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="continue",
        user_input=text,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_synthesize_without_bundle_is_unchanged(monkeypatch) -> None:
    """When no figure bundle is bound, no L1/L3/L4 tags are emitted."""

    _patch_super_synthesize(monkeypatch)
    synth = _StubSynthesizer(figure_bundle=None, fake_text="hello")
    response = synth.synthesize(context=_context("anything"))
    tags = set(response.rationale_tags)
    l_tags = {t for t in tags if t.startswith("l1_") or t.startswith("l3_") or t.startswith("l4_")}
    assert l_tags == set(), (
        f"no L1/L3/L4 tags expected without bundle, got {l_tags!r}"
    )


def test_synthesize_with_bundle_emits_l1_and_l3_tags(monkeypatch) -> None:
    """Bundle bound + in-scope query: L1 style + L3 verify both surface."""

    _patch_super_synthesize(monkeypatch)
    bundle = _einstein_bundle()
    synth = _StubSynthesizer(
        figure_bundle=bundle,
        fake_text=(
            "Spatially separated subsystems each carry their own definite "
            "physical state in a complete physical theory."
        ),
    )
    # Use an in-scope query so L4 doesn't short-circuit.
    response = synth.synthesize(
        context=_context(
            "Are spatially separated subsystems each in a definite physical state?"
        )
    )
    tags = set(response.rationale_tags)
    assert any(t.startswith("l1_style_prior=") for t in tags), (
        f"expected an l1_style_prior tag, got {tags!r}"
    )
    assert any(t.startswith("l3_grounded_verify=") for t in tags), (
        f"expected an l3_grounded_verify tag, got {tags!r}"
    )


def test_synthesize_short_circuits_on_out_of_scope_query(monkeypatch) -> None:
    """STRICT_REFUSE policy + out-of-domain query → response.text is the
    reviewer-curated refusal text and no LLM-side text appears.

    Uses an out-of-domain query (sourdough bread) verified against
    the synthetic Einstein corpus's coverage classifier — see
    ``debug_coverage.py``. Tiramisu / mascarpone happen to land
    inside the philosophy_of_science centroid because of hashing-
    embedding aliasing on a small corpus, so we pick a query
    whose hashed tokens unambiguously miss every documented domain.
    """

    _patch_super_synthesize(monkeypatch)
    bundle = _einstein_bundle()
    synth = _StubSynthesizer(
        figure_bundle=bundle,
        fake_text="THIS-SHOULD-NEVER-APPEAR-because-LLM-was-skipped",
    )
    response = synth.synthesize(
        context=_context("How do I make sourdough bread from scratch?")
    )
    # The refusal text from ScopeRefuser config; the LLM text must NOT
    # appear in the response.
    assert "sourdough" not in response.text.lower()
    assert "THIS-SHOULD-NEVER-APPEAR" not in response.text
    assert response.abstract_action == "refuse_out_of_scope"
    assert "l4_scope_refusal" in response.rationale_tags


def test_synthesize_in_scope_query_does_not_refuse(monkeypatch) -> None:
    """In-scope query → response.text is the LLM stub output (not a refusal)."""

    _patch_super_synthesize(monkeypatch)
    bundle = _einstein_bundle()
    fake_text = "Reality is independent of observation as Einstein argued."
    synth = _StubSynthesizer(figure_bundle=bundle, fake_text=fake_text)
    response = synth.synthesize(
        context=_context(
            "Are spatially separated subsystems each in a definite physical state?"
        )
    )
    assert response.text == fake_text
    assert response.abstract_action != "refuse_out_of_scope"


def test_synthesize_records_history_only_after_enforcement_passes(monkeypatch) -> None:
    """When L4 short-circuits with a refusal, the recorded history must
    capture the refusal text — not the would-be LLM output. This keeps the
    conversation buffer consistent with what the user actually sees."""

    _patch_super_synthesize(monkeypatch)
    bundle = _einstein_bundle()
    synth = _StubSynthesizer(
        figure_bundle=bundle,
        fake_text="ghost-llm-output-not-returned",
    )
    synth.synthesize(
        context=_context("How do I make sourdough bread from scratch?")
    )
    history = synth._snapshot_history()  # noqa: SLF001 — internal access for test
    assert history, "refusal turn should still be recorded in history"
    last_user, last_assistant = history[-1]
    assert "sourdough" in last_user.lower()
    assert "ghost-llm-output-not-returned" not in last_assistant


@pytest.mark.parametrize(
    "no_bundle_synth_kwargs",
    [{}, {"figure_bundle": None}],
)
def test_synthesize_legacy_path_preserves_plan_rationale_tag(
    monkeypatch, no_bundle_synth_kwargs
) -> None:
    """Legacy callers (bundle=None) must still receive the prompt-plan
    rationale tag — Wave F enforcement does not displace it."""

    _patch_super_synthesize(monkeypatch)
    synth = _StubSynthesizer(fake_text="legacy", **no_bundle_synth_kwargs)
    response = synth.synthesize(context=_context("hi"))
    tags = set(response.rationale_tags)
    assert any(t.startswith("plan=") for t in tags), (
        f"plan tag must survive Wave F wiring; tags={tags!r}"
    )
