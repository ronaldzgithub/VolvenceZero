"""GroundedResponseSynthesizer — the lifeform's voice.

Subclasses the kernel ``ResponseSynthesizer`` ABC. Composes a ``PromptPlan``
from the planner, then renders. The renderer is rule-based by default and
deterministic for replay; subclass and override ``_render`` to plug in an
LLM grounded by the same plan.

Design notes:

* The kernel's ``synthesize`` signature only exposes ``context`` and
  ``assembly``. We **do not** reach into kernel internals or ``BrainSession``
  to grab additional state — staying inside the contract is the whole point
  of this layer (R8).

* The base ``ResponseSynthesizer`` already does extensive heuristic
  templating; we delegate to it for fallback paths (REFER-OUT, judgment
  process, missing assembly) and only build our own structured renderer on
  top.

* The plan is exposed in ``AgentResponse.rationale`` as a structured tail
  so the lifeform-evolution layer can audit which plan drove each turn.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

from lifeform_core import VitalsSnapshot
from volvence_zero.agent.response import (
    AgentResponse,
    ResponseContext,
    ResponseSynthesizer,
)
from volvence_zero.application.runtime import ResponseAssemblySnapshot
from volvence_zero.interlocutor import InterlocutorState

from lifeform_expression.prompt_planner import (
    PromptPlan,
    PromptPlanner,
    SectionId,
    TurnIntent,
)


VitalsSnapshotProvider = Callable[[], VitalsSnapshot | None]
InterlocutorStateProvider = Callable[[], InterlocutorState | None]


class GroundedResponseSynthesizer(ResponseSynthesizer):
    """ResponseSynthesizer driven by a structured PromptPlan.

    Optional ``vitals_snapshot_provider`` is a zero-arg callable returning
    the current ``VitalsSnapshot`` (or ``None``) for the calling session.
    The lifeform layer wires a per-session provider via closure so that
    drives produced by ``VitalsModule`` reach the planner \u2014 which adds a
    ``CONTINUITY_NOTE`` section and a ``vitals_pressure=...`` rationale
    tag whenever the slow-scale PE is above threshold. Without a provider
    the synthesizer behaves exactly as before, so this is fully
    backwards-compatible.
    """

    def __init__(
        self,
        *,
        planner: PromptPlanner | None = None,
        vitals_snapshot_provider: VitalsSnapshotProvider | None = None,
        interlocutor_state_provider: InterlocutorStateProvider | None = None,
    ) -> None:
        self._planner = planner or PromptPlanner()
        self._vitals_snapshot_provider = vitals_snapshot_provider
        self._interlocutor_state_provider = interlocutor_state_provider

    @property
    def planner(self) -> PromptPlanner:
        return self._planner

    @property
    def vitals_snapshot_provider(self) -> VitalsSnapshotProvider | None:
        return self._vitals_snapshot_provider

    @property
    def interlocutor_state_provider(self) -> InterlocutorStateProvider | None:
        return self._interlocutor_state_provider

    def with_vitals_provider(
        self, provider: VitalsSnapshotProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Return a clone of this synthesizer bound to the given vitals provider.

        The lifeform layer calls this once per ``LifeformSession`` so each
        session has its own synthesizer instance reading its own
        ``VitalsModule``. The original synthesizer (and its planner) are
        kept intact so the Brain-level default does not get mutated.
        Existing interlocutor provider (if any) is preserved.
        """
        return GroundedResponseSynthesizer(
            planner=self._planner,
            vitals_snapshot_provider=provider,
            interlocutor_state_provider=self._interlocutor_state_provider,
        )

    def with_interlocutor_provider(
        self, provider: InterlocutorStateProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Return a clone bound to the given interlocutor-state provider.

        Mirrors ``with_vitals_provider``: the lifeform layer wires
        a per-session closure to ``LifeformSession.interlocutor_state``
        so the planner sees fresh 12-axis readouts every turn. The
        Brain-level default synthesizer is untouched. Existing
        vitals provider (if any) is preserved on the clone.
        """
        return GroundedResponseSynthesizer(
            planner=self._planner,
            vitals_snapshot_provider=self._vitals_snapshot_provider,
            interlocutor_state_provider=provider,
        )

    # Intents we delegate to the base kernel renderer because the base
    # already produces strong, regime-faithful text for them (and tests pin
    # the judgment-process / refer-out shapes). Direct-answer also delegates
    # because there is no useful section structure to add on top of the
    # kernel's regime-tail templates.
    _DELEGATE_TO_BASE: frozenset[TurnIntent] = frozenset(
        {TurnIntent.REFER_OUT, TurnIntent.JUDGMENT_PROCESS, TurnIntent.DIRECT_ANSWER}
    )

    def synthesize(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None = None,
    ) -> AgentResponse:
        vitals = self._read_vitals_snapshot()
        interlocutor_state = self._read_interlocutor_state()
        plan = self._planner.plan(
            context=context,
            assembly=assembly,
            vitals=vitals,
            interlocutor_state=interlocutor_state,
        )

        if plan.intent in self._DELEGATE_TO_BASE or assembly is None:
            base = super().synthesize(context=context, assembly=assembly)
            return _attach_plan_rationale(base, plan)

        text = self._render(
            context=context, assembly=assembly, plan=plan, vitals=vitals
        )
        if not text.strip():
            base = super().synthesize(context=context, assembly=assembly)
            return _attach_plan_rationale(base, plan)

        regime_id = assembly.regime_id if assembly is not None else context.regime_id
        regime_name = assembly.regime_name if assembly is not None else context.regime_name
        abstract_action = assembly.abstract_action if assembly is not None else context.abstract_action
        rationale = _build_rationale(
            regime_name=regime_name,
            regime_id=regime_id,
            abstract_action=abstract_action,
            context=context,
            assembly=assembly,
            plan=plan,
            vitals=vitals,
        )
        return AgentResponse(
            text=text,
            regime_id=regime_id,
            abstract_action=abstract_action,
            rationale=rationale,
        )

    def _read_vitals_snapshot(self) -> VitalsSnapshot | None:
        """Pull the current vitals snapshot (None when no provider bound)."""
        if self._vitals_snapshot_provider is None:
            return None
        return self._vitals_snapshot_provider()

    def _read_interlocutor_state(self) -> InterlocutorState | None:
        """Pull the current 12-axis interlocutor readout (None when unbound).

        Returning ``None`` is a normal cold-start signal: the
        planner treats it as "no modulation". Callers are expected
        to inspect ``readout_confidence`` themselves; here we just
        forward whatever the provider hands back without any
        interpretation \u2014 the planner is the policy owner.
        """
        if self._interlocutor_state_provider is None:
            return None
        return self._interlocutor_state_provider()

    # ------------------------------------------------------------------
    # Rendering hooks — subclass to swap in an LLM
    # ------------------------------------------------------------------

    def _render(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        plan: PromptPlan,
        vitals: VitalsSnapshot | None = None,
    ) -> str:
        sentences: list[str] = []
        for section in plan.sections:
            rendered = self._render_section(
                section=section,
                context=context,
                assembly=assembly,
                plan=plan,
                vitals=vitals,
            )
            if rendered:
                sentences.append(rendered)
        return " ".join(_dedupe_sentences(sentences)).strip()

    def _render_section(
        self,
        *,
        section: SectionId,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        plan: PromptPlan,
        vitals: VitalsSnapshot | None = None,
    ) -> str:
        if section is SectionId.ACKNOWLEDGE_PRESSURE:
            return self._render_acknowledge(context=context, assembly=assembly)
        if section is SectionId.REGIME_FRAME:
            return self._render_regime_frame(context=context, assembly=assembly)
        if section is SectionId.OPEN_LOOP_HANDOFF:
            return self._render_open_loop(context=context, assembly=assembly)
        if section is SectionId.CLARIFICATION:
            return self._render_clarification(context=context, assembly=assembly, plan=plan)
        if section is SectionId.NEXT_STEP:
            return self._render_next_step(context=context, assembly=assembly)
        if section is SectionId.BOUNDARY_DISCLAIMER:
            return self._render_boundary(context=context, assembly=assembly)
        if section is SectionId.REFLECTION_HOOK:
            return self._render_reflection(context=context)
        if section is SectionId.CONTINUITY_NOTE:
            return self._render_continuity(
                context=context, assembly=assembly, vitals=vitals
            )
        return ""

    # ------------------------------------------------------------------
    # Per-section text — keep deterministic so replay tests are stable
    # ------------------------------------------------------------------

    @staticmethod
    def _render_acknowledge(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        if context.regime_id == "repair_and_deescalation":
            return (
                "I want to slow this down so we can stay grounded together. "
                "I am not going to push past what just happened."
            )
        if context.regime_id == "emotional_support":
            return "I am hearing weight in this and I want to stay with it before we move."
        if assembly.continuum_target_position >= 0.7:
            return "I want to acknowledge the pressure here before we narrow anything."
        return "I am not going to rush past what you just said."

    @staticmethod
    def _render_regime_frame(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        regime = assembly.regime_id or context.regime_id or ""
        if regime == "emotional_support":
            return "I will stay supportive first and not jump into solving."
        if regime == "guided_exploration":
            return "I would rather explore this with you step by step than guess at one answer."
        if regime == "problem_solving":
            return "I see a structured path here we can walk together."
        if regime == "repair_and_deescalation":
            return "I will keep my responses steady and avoid escalating tone."
        if regime == "acquaintance_building":
            return "I want to keep this warm rather than transactional."
        if regime == "casual_social":
            return "I can stay in steady, low-pressure mode."
        return "I will stay context-aware and keep usefulness and continuity in view."

    @staticmethod
    def _render_open_loop(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        if assembly.case_hit_count or assembly.playbook_rule_count:
            return (
                "There is at least one thread we left open before; I will "
                "thread it forward instead of restarting from scratch."
            )
        return "I will keep prior open threads in view rather than dropping them."

    @staticmethod
    def _render_clarification(
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        plan: PromptPlan,
    ) -> str:
        if plan.question_budget <= 0:
            return "I want to keep this bounded; if anything is missing I will hold off rather than over-commit."
        return "Before we go further, can you say a little more about what specifically would feel useful right now?"

    @staticmethod
    def _render_next_step(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        regime = assembly.regime_id or context.regime_id or ""
        if regime in {"emotional_support", "repair_and_deescalation"}:
            return "We can stay here together, and when you are ready, name one small next step that would feel manageable."
        if regime == "problem_solving":
            return "Concretely, the smallest useful next step is to name the constraint that matters most."
        if regime == "guided_exploration":
            return "Pick one of the threads we just surfaced and we can go a little deeper on it."
        return "From here we can take one small, concrete step that keeps things moving without forcing the pace."

    @staticmethod
    def _render_boundary(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        if assembly.refer_out_required:
            return (
                "I should also flag that I am staying high-level here; for anything "
                "with material risk, please bring in a qualified professional."
            )
        if assembly.clarification_required:
            return "I will keep this bounded until one missing detail is clarified."
        if assembly.citation_mode == "required":
            return "I am keeping any factual reference clearly sourced and non-definitive."
        if assembly.required_disclaimer_phrases:
            joined = "; ".join(assembly.required_disclaimer_phrases)
            return f"Note: {joined}."
        return ""

    @staticmethod
    def _render_reflection(*, context: ResponseContext) -> str:
        if context.reflection_writeback_applied:
            return f"I am also carrying forward {context.reflection_lesson_count} reflected lesson(s) from the slow loop."
        if context.primary_reflection_lesson is not None:
            return "I am letting the slower reflective layer shape this rather than treating it as a no-op."
        if context.primary_reflection_tension is not None:
            return "I am keeping an eye on the tension that is still open instead of smoothing past it."
        return ""

    @staticmethod
    def _render_continuity(
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        vitals: VitalsSnapshot | None = None,
    ) -> str:
        # When the always-on drive layer is signalling pressure, the
        # continuity note acknowledges the elapsed silence directly rather
        # than the generic "carrying continuity" line. We name the most
        # affected drive so reflection / evaluation can audit which drive
        # actually drove the response shape.
        if vitals is not None and vitals.above_proactive_threshold:
            out_of_band = [d for d in vitals.drive_levels if d.out_of_band]
            if out_of_band:
                worst = max(out_of_band, key=lambda d: d.pe_contribution)
                return (
                    f"I want to acknowledge that some time has passed; my "
                    f"{worst.name.replace('_', ' ')} level dropped while we "
                    f"were quiet, and I am noticing it now rather than acting "
                    f"as if no gap occurred."
                )
        if assembly.prompt_residue_summary:
            return "I am also carrying continuity from our recent thread."
        return "I am keeping continuity in view across our turns."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedupe_sentences(sentences: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving order — defensive against the case
    where two adjacent sections render similar lines for the same regime.
    """
    seen: set[str] = set()
    out: list[str] = []
    for sentence in sentences:
        normalised = " ".join(sentence.split())
        if not normalised:
            continue
        if normalised in seen:
            continue
        seen.add(normalised)
        out.append(normalised)
    return out


def _attach_plan_rationale(response: AgentResponse, plan: PromptPlan) -> AgentResponse:
    """Append the plan summary to a response rendered by the base kernel renderer.

    Used on the ``DELEGATE_TO_BASE`` path (refer-out / judgment-process /
    direct-answer) so even those responses carry the planner's rationale
    tags forward \u2014 most importantly ``vitals_pressure=...`` and any
    explicit cross-cutting tags the planner emitted. Without this, vitals
    state would only reach reflection / evaluation when the renderer is
    NOT the kernel's base, which makes the "always-on drive" signal
    silently disappear in roughly half of common turns.
    """
    rationale = response.rationale
    if rationale and not rationale.endswith("."):
        rationale += "."
    plan_tag = (
        f" Plan: intent={plan.intent.value}; "
        f"sections={','.join(s.value for s in plan.sections)}; "
        f"q={plan.question_budget}."
    )
    extra_tags = [
        tag for tag in plan.rationale_tags
        if tag.startswith("vitals_pressure=")
        or tag.startswith("interlocutor_conf=")
        or tag.startswith("il_")
    ]
    extra = (" " + " ".join(extra_tags) + ".") if extra_tags else ""
    return AgentResponse(
        text=response.text,
        regime_id=response.regime_id,
        abstract_action=response.abstract_action,
        rationale=(rationale + plan_tag + extra).strip(),
    )


def _build_rationale(
    *,
    regime_name: str,
    regime_id: Optional[str],
    abstract_action: Optional[str],
    context: ResponseContext,
    assembly: ResponseAssemblySnapshot,
    plan: PromptPlan,
    vitals: VitalsSnapshot | None = None,
) -> str:
    parts: list[str] = [f"regime={regime_id or 'none'}"]
    if abstract_action:
        parts.append(f"temporal={abstract_action}")
    parts.append(f"switch_gate={context.temporal_switch_gate:.2f}")
    parts.append(f"intent={plan.intent.value}")
    parts.append("sections=" + ",".join(s.value for s in plan.sections))
    parts.append(f"q={plan.question_budget}")
    if assembly.knowledge_hit_count:
        parts.append(f"knowledge_hits={assembly.knowledge_hit_count}")
    if assembly.case_hit_count:
        parts.append(f"case_hits={assembly.case_hit_count}")
    if assembly.playbook_rule_count:
        parts.append(f"playbook_rules={assembly.playbook_rule_count}")
    parts.append(f"risk={assembly.risk_band.value}")
    if vitals is not None and vitals.above_proactive_threshold:
        out_of_band = ",".join(
            d.name for d in vitals.drive_levels if d.out_of_band
        )
        if out_of_band:
            parts.append(f"vitals_pressure={out_of_band}")
            parts.append(f"vitals_total_pe={vitals.total_pe:.2f}")
    # Forward Gap 9 slice 2c interlocutor-state tags so reflection /
    # evaluation can audit which user-state axes shaped this turn.
    for tag in plan.rationale_tags:
        if tag.startswith("interlocutor_conf=") or tag.startswith("il_"):
            parts.append(tag)
    return f"GroundedResponseSynthesizer from {regime_name}; " + ", ".join(parts) + "."
