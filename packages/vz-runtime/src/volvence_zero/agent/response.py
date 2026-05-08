from __future__ import annotations

from dataclasses import dataclass, field

from volvence_zero.application.runtime import ResponseAssemblySnapshot, ResponseMode
from volvence_zero.regime.contracts import ExpressionBrief
from volvence_zero.social_cognition import PRIMARY_INTERLOCUTOR_ID, SELF_INTERLOCUTOR_ID


@dataclass(frozen=True)
class AgentResponse:
    """Final per-turn expression artefact.

    ``rationale_tags`` is the structured (typed) audit surface for the
    response. Each entry is a stable ``"key=value"`` token (or a bare
    flag like ``"reflection_writeback=applied"``). Consumers that need
    to gate behaviour on what shaped a turn must read this tuple
    rather than substring-match the ``rationale`` string. See
    ``docs/specs/expression-layer.md``.

    The ``rationale`` string remains the human-readable summary and
    will continue to embed the same tokens for log readability, but
    its exact phrasing is **not** a contract.
    """

    text: str
    regime_id: str | None
    abstract_action: str | None
    rationale: str
    rationale_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepairExpressionAdvisory:
    """One-turn expression advisory derived from ``rupture_state``.

    This is not a new rupture owner. Runtime builds it from the owner
    snapshot after propagation so expression can adjust wording without
    reading SHADOW snapshots or re-interpreting rupture evidence.

    ``kind_label`` is the canonical human-readable phrase published by
    ``rupture_state`` (single source of truth, W3 SSOT). The lifeform
    expression layer renders this verbatim; it does NOT maintain a
    parallel RuptureKind -> string map.
    """

    rupture_kind: str
    confidence: float
    signal_strength: float
    description: str
    kind_label: str = ""


@dataclass(frozen=True)
class ResponseContext:
    regime_id: str | None
    regime_name: str
    regime_switched: bool
    abstract_action: str | None
    alert_count: int
    temporal_switch_gate: float
    temporal_is_switching: bool
    reflection_lesson_count: int
    reflection_tension_count: int
    reflection_writeback_applied: bool
    primary_reflection_lesson: str | None
    primary_reflection_tension: str | None
    joint_schedule_action: str
    user_input: str = ""
    active_speaker_id: str = PRIMARY_INTERLOCUTOR_ID
    addressee_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,)
    subject_ids: tuple[str, ...] = (PRIMARY_INTERLOCUTOR_ID,)
    audience_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,)
    repair_advisory: RepairExpressionAdvisory | None = None
    # W3 of ssot-cleanup-p0-p4: per-regime expression brief threaded
    # from ``RegimeIdentity.expression_brief`` so the synthesizer
    # does not branch on ``regime_id`` to pick variant text.
    regime_expression_brief: ExpressionBrief = field(
        default_factory=ExpressionBrief
    )
    # Optional prior conversational turns (oldest-first), each entry
    # is a ``(user_text, assistant_text)`` pair. Expression layer is
    # the owner; runtime / cognition modules MUST NOT consume this
    # to reconstruct social facts or memory state. It exists purely
    # so the LLM expression path can keep continuity for small
    # base models that need raw chat context.
    prior_turns: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GenerationConstraints:
    response_mode: str
    answer_depth_limit: str
    citation_mode: str
    max_questions: int
    required_disclaimer_phrases: tuple[str, ...] = ()
    ordering_bias: tuple[str, ...] = ()
    prompt_residue_summary: str = ""
    prompt_residue_ratio: float = 0.0
    continuum_target_position: float = 0.5
    ordering_driver: str = "playbook-only"
    decoding_profile: str = "balanced"
    question_budget: int | None = None


class ResponseSynthesizer:
    """Expression-layer synthesizer over structured runtime state.

    Base class uses fixed templates. Subclass ``LLMResponseSynthesizer``
    replaces templates with real LLM generation.
    """

    def _render_judgment_process_response(
        self,
        *,
        assembly: ResponseAssemblySnapshot,
    ) -> str:
        speech_plan = assembly.speech_plan
        if speech_plan is None:
            return (
                "I am responding to your request to see the basis for my answer. "
                "I infer that you need a reply that shows how the current conversation is shaping the next move. "
                "So I will state the cue, the need I infer, and the adjustment before offering support."
            )
        text = (
            f"{speech_plan.cue} "
            f"{speech_plan.inferred_need} "
            f"{speech_plan.response_adjustment}"
        )
        return text.strip()

    def _render_speech_plan_response(
        self,
        *,
        assembly: ResponseAssemblySnapshot,
    ) -> str:
        """Generic speech-plan renderer for non-``judgment-process`` intents.

        Until the kernel grows a learned per-intent renderer (or an LLM-backed
        ``GroundedResponseSynthesizer`` is plugged in via lifeform-expression),
        this lifts the structured ``cue`` / ``inferred_need`` /
        ``response_adjustment`` script directly into prose. That is enough to
        make the six product regimes feel distinct rather than collapsing
        into the judgment-process template.
        """
        speech_plan = assembly.speech_plan
        if speech_plan is None:
            return ""
        text = (
            f"{speech_plan.cue} "
            f"{speech_plan.inferred_need} "
            f"{speech_plan.response_adjustment}"
        )
        return text.strip()

    def synthesize(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None = None,
    ) -> AgentResponse:
        # Per-lesson and per-tension UX text now lives in the lifeform
        # layer (``lifeform_expression.reflection_hints``) so the
        # kernel does not own how reflection ids sound to the user.
        # The kernel still surfaces *that* a lesson / tension exists
        # (counts + flags below); only the rendered prose moved.

        transition_hint = ""
        if context.temporal_is_switching or context.temporal_switch_gate >= 0.65:
            transition_hint = " I am deliberately shifting the internal control path rather than answering on autopilot."
        elif context.joint_schedule_action in {"ssl-only", "ssl-only-pe"}:
            transition_hint = " I am keeping the current frame stable long enough to consolidate it before widening."
        elif context.joint_schedule_action in {"full-cycle", "full-cycle-pe"}:
            transition_hint = " I have already run a deeper internal cycle, so I can shape this reply more deliberately."

        regime_shift_hint = ""
        if context.regime_switched:
            regime_shift_hint = " I am also changing the interaction frame instead of forcing the previous one to fit."

        memory_hint = ""
        if assembly is not None and assembly.prompt_residue_summary:
            memory_hint = " I am carrying forward continuity from the current interaction state."

        reflection_hint = ""
        if context.reflection_writeback_applied:
            reflection_hint = (
                f" I am also carrying forward {context.reflection_lesson_count} reflected lesson"
                f"{'' if context.reflection_lesson_count == 1 else 's'} from the slow loop."
            )
        elif context.reflection_lesson_count:
            reflection_hint = (
                f" I can already feel {context.reflection_lesson_count} slower reflection"
                f"{'' if context.reflection_lesson_count == 1 else 's'} shaping how I respond."
            )

        knowledge_hint = ""
        if assembly is not None and assembly.knowledge_hit_count:
            knowledge_hint = (
                f" I am grounding this reply with {assembly.knowledge_hit_count} relevant background cue"
                f"{'' if assembly.knowledge_hit_count == 1 else 's'}."
            )

        case_hint = ""
        if assembly is not None and assembly.case_hit_count:
            case_hint = (
                f" I am also checking {assembly.case_hit_count} similar interaction pattern"
                f"{'' if assembly.case_hit_count == 1 else 's'} for pacing rather than copying them literally."
            )

        playbook_hint = ""
        if assembly is not None and assembly.playbook_rule_count:
            playbook_hint = (
                f" I also have {assembly.playbook_rule_count} pacing cue"
                f"{'' if assembly.playbook_rule_count == 1 else 's'} shaping the order of the reply."
            )

        boundary_hint = ""
        if assembly is not None and assembly.refer_out_required:
            boundary_hint = (
                " I should stay within high-level support and encourage appropriate professional follow-up "
                "instead of sounding definitive."
            )
        elif assembly is not None and assembly.clarification_required:
            boundary_hint = (
                " I should keep this bounded and ask for any missing local or contextual detail before acting "
                "over-certain."
            )
        elif assembly is not None and assembly.citation_mode == "required":
            boundary_hint = " I should keep any factual guidance sourced, bounded, and clearly non-definitive."

        tension_hint = ""
        if context.primary_reflection_tension is not None:
            tension_hint = (
                " I want to keep the open tensions in view rather than smoothing past them."
            )

        lesson_hint = ""
        if context.primary_reflection_lesson is not None:
            lesson_hint = (
                " I am letting the slower reflective layer shape this reply."
            )

        response_mode = assembly.response_mode.value if assembly is not None else None
        effective_regime_id = assembly.regime_id if assembly is not None else context.regime_id
        effective_regime_name = assembly.regime_name if assembly is not None else context.regime_name
        effective_abstract_action = assembly.abstract_action if assembly is not None else context.abstract_action
        if assembly is not None and assembly.expression_intent == "judgment-process":
            text = self._render_judgment_process_response(assembly=assembly)
            rationale_parts = [f"regime={effective_regime_id or 'none'}", "expression=judgment-process"]
            if effective_abstract_action:
                rationale_parts.append(f"temporal={effective_abstract_action}")
            rationale_parts.append(f"switch_gate={context.temporal_switch_gate:.2f}")
            rationale_parts.append(f"question_budget={assembly.max_questions}")
            rationale = ", ".join(rationale_parts)
            return AgentResponse(
                text=text,
                regime_id=effective_regime_id,
                abstract_action=effective_abstract_action,
                rationale=f"Synthesized from {effective_regime_name}; {rationale}.",
                rationale_tags=tuple(rationale_parts),
            )

        # Other structured intents — render directly from the speech plan so
        # each regime gets a visibly different turn shape rather than falling
        # through to the generic regime-tail templates below.
        _STRUCTURED_INTENTS = {
            "support-first",
            "support-before-decision",
            "repair-first",
            "structure-first",
            "warmth-first",
            "clarify-first",
            "refer-out",
        }
        if (
            assembly is not None
            and assembly.expression_intent in _STRUCTURED_INTENTS
            and assembly.speech_plan is not None
        ):
            text = self._render_speech_plan_response(assembly=assembly)
            if text:
                rationale_parts = [
                    f"regime={effective_regime_id or 'none'}",
                    f"expression={assembly.expression_intent}",
                ]
                if effective_abstract_action:
                    rationale_parts.append(f"temporal={effective_abstract_action}")
                rationale_parts.append(f"switch_gate={context.temporal_switch_gate:.2f}")
                rationale_parts.append(f"question_budget={assembly.max_questions}")
                rationale = ", ".join(rationale_parts)
                return AgentResponse(
                    text=text,
                    regime_id=effective_regime_id,
                    abstract_action=effective_abstract_action,
                    rationale=f"Synthesized from {effective_regime_name}; {rationale}.",
                    rationale_tags=tuple(rationale_parts),
                )
        if response_mode == ResponseMode.REFER_OUT.value:
            text = (
                "I want to keep this careful and high-level. "
                "I can help you orient the next step, but I should avoid sounding definitive here."
            )
        elif response_mode == ResponseMode.CLARIFY.value:
            text = (
                "I can help, but I should keep this bounded until one key detail is clarified. "
                "That way I do not over-commit too early."
            )
        elif effective_regime_id == "repair_and_deescalation":
            text = (
                "I want to slow this down a little and make sure I respond in a steady, repairing way. "
                "We can handle the immediate issue, but I want to keep the interaction safe and grounded."
            )
        elif effective_regime_id == "emotional_support":
            text = (
                "I am hearing emotional weight in this, so I want to stay supportive first and not rush past it. "
                "We can still move toward something useful together."
            )
        elif effective_regime_id == "problem_solving":
            text = (
                "I see a concrete problem-solving path here. "
                "I can help structure the next steps clearly and keep the solution actionable."
            )
        elif effective_regime_id == "guided_exploration":
            text = (
                "This feels like a place for guided exploration rather than a rushed answer. "
                "I can help us narrow the space step by step."
            )
        elif effective_regime_id == "acquaintance_building":
            text = (
                "I want to keep this warm and relational rather than treating it like a cold transaction. "
                "We can build clarity without losing that sense of connection."
            )
        elif effective_regime_id == "casual_social":
            text = (
                "I can keep this steady, natural, and continuous rather than over-formalizing it too early. "
                "That gives us room to stay useful without losing flow."
            )
        else:
            text = (
                "I can stay with the current context and respond in a way that keeps both usefulness and continuity in view."
            )

        if context.alert_count:
            text += " I also notice some internal caution signals, so I will stay measured rather than over-commit."
        text += transition_hint
        text += regime_shift_hint
        text += memory_hint
        text += reflection_hint
        text += knowledge_hint
        text += case_hint
        text += playbook_hint
        text += lesson_hint
        text += tension_hint
        text += boundary_hint

        rationale_parts = [f"regime={effective_regime_id or 'none'}"]
        if effective_abstract_action:
            rationale_parts.append(f"temporal={effective_abstract_action}")
        rationale_parts.append(f"switch_gate={context.temporal_switch_gate:.2f}")
        rationale_parts.append(f"joint={context.joint_schedule_action}")
        if context.alert_count:
            rationale_parts.append(f"alerts={context.alert_count}")
        if assembly is not None and assembly.knowledge_hit_count:
            rationale_parts.append(f"knowledge_hits={assembly.knowledge_hit_count}")
        if assembly is not None and assembly.case_hit_count:
            rationale_parts.append(f"case_hits={assembly.case_hit_count}")
        if assembly is not None and assembly.playbook_rule_count:
            rationale_parts.append(f"playbook_rules={assembly.playbook_rule_count}")
        rationale_parts.append(f"risk={(assembly.risk_band.value if assembly is not None else 'low')}")
        if context.reflection_lesson_count:
            rationale_parts.append(f"reflection_lessons={context.reflection_lesson_count}")
        if context.primary_reflection_lesson is not None:
            rationale_parts.append(f"primary_lesson={context.primary_reflection_lesson}")
        if context.primary_reflection_tension is not None:
            rationale_parts.append(f"primary_tension={context.primary_reflection_tension}")
        if context.reflection_writeback_applied:
            rationale_parts.append("reflection_writeback=applied")
        rationale = ", ".join(rationale_parts)

        return AgentResponse(
            text=text,
            regime_id=effective_regime_id,
            abstract_action=effective_abstract_action,
            rationale=f"Synthesized from {effective_regime_name}; {rationale}.",
            rationale_tags=tuple(rationale_parts),
        )


class LLMResponseSynthesizer(ResponseSynthesizer):
    """Expression-layer synthesizer backed by a real LLM.

    Uses ``OpenWeightResidualRuntime.generate()`` to produce text,
    with system prompt assembled from live cognitive state via
    ``volvence_zero.agent.prompts.build_system_prompt``.
    """

    def __init__(
        self,
        *,
        runtime: object,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> None:
        self._runtime = runtime
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    @staticmethod
    def _decoding_profile_for_assembly(assembly: ResponseAssemblySnapshot) -> str:
        if assembly.ordering_driver in {"continuum-support-first", "continuum-support-clarify"}:
            return "support-first"
        if assembly.ordering_driver == "continuum-clarify-first":
            return "clarify-first"
        if assembly.ordering_driver == "continuum-structure-first":
            return "structure-first"
        if assembly.continuum_target_position >= 0.70:
            return "support-first"
        if assembly.continuum_target_position >= 0.52:
            return "clarify-first"
        return "structure-first"

    def synthesize(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None = None,
    ) -> AgentResponse:
        from volvence_zero.agent.prompts import build_chat_messages, build_system_prompt

        if assembly is None:
            return super().synthesize(context=context, assembly=assembly)

        system_prompt = build_system_prompt(
            assembly=assembly,
            context=context,
        )
        chat_messages = build_chat_messages(
            assembly=assembly,
            context=context,
        )

        user_input = context.user_input
        if not user_input:
            return super().synthesize(context=context, assembly=assembly)

        control_params = assembly.control_code if assembly is not None else ()
        control_scale = assembly.control_scale if assembly is not None and control_params else (
            context.temporal_switch_gate * 0.15 if control_params else 0.0
        )
        if assembly is not None and control_params:
            continuum_control_gain = 1.0 + abs(assembly.continuum_target_position - 0.5) * 0.6
            control_scale = min(0.38, control_scale * continuum_control_gain)
        decoding_profile = self._decoding_profile_for_assembly(assembly) if assembly is not None else "balanced"
        constraints = (
            GenerationConstraints(
                response_mode=assembly.response_mode.value,
                answer_depth_limit=assembly.answer_depth_limit,
                citation_mode=assembly.citation_mode,
                max_questions=assembly.max_questions,
                required_disclaimer_phrases=assembly.required_disclaimer_phrases,
                ordering_bias=assembly.ordering_plan,
                prompt_residue_summary=assembly.prompt_residue_summary,
                prompt_residue_ratio=assembly.prompt_residue_ratio,
                continuum_target_position=assembly.continuum_target_position,
                ordering_driver=assembly.ordering_driver,
                decoding_profile=decoding_profile,
                question_budget=(
                    assembly.speech_plan.question_budget
                    if assembly.speech_plan is not None
                    else assembly.max_questions
                ),
            )
            if assembly is not None
            else None
        )

        result = self._runtime.generate(
            prompt=user_input,
            system_context=system_prompt,
            chat_messages=chat_messages,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            control_parameters=control_params,
            control_scale=control_scale,
            generation_constraints=constraints,
        )

        generated_text = result.text.strip()
        if not generated_text:
            return super().synthesize(context=context, assembly=assembly)

        rationale_parts = [
            f"regime={(assembly.regime_id if assembly is not None else context.regime_id) or 'none'}",
            f"model={self._runtime.model_id}",
            f"tokens={result.token_count}",
        ]
        if assembly is not None and assembly.abstract_action:
            rationale_parts.append(f"temporal={assembly.abstract_action}")
        elif context.abstract_action:
            rationale_parts.append(f"temporal={context.abstract_action}")
        rationale_parts.append(f"switch_gate={context.temporal_switch_gate:.2f}")
        if assembly is not None and assembly.knowledge_hit_count:
            rationale_parts.append(f"knowledge_hits={assembly.knowledge_hit_count}")
        if assembly is not None and assembly.case_hit_count:
            rationale_parts.append(f"case_hits={assembly.case_hit_count}")
        if assembly is not None and assembly.playbook_rule_count:
            rationale_parts.append(f"playbook_rules={assembly.playbook_rule_count}")
            rationale_parts.append(f"ordering_driver={assembly.ordering_driver}")
            rationale_parts.append(f"continuum_target={assembly.continuum_target_position:.2f}")
            rationale_parts.append(f"decoding_profile={decoding_profile}")
        rationale_parts.append(f"risk={(assembly.risk_band.value if assembly is not None else 'low')}")

        return AgentResponse(
            text=generated_text,
            regime_id=assembly.regime_id if assembly is not None else context.regime_id,
            abstract_action=assembly.abstract_action if assembly is not None else context.abstract_action,
            rationale=f"LLM generated; {', '.join(rationale_parts)}.",
            rationale_tags=tuple(rationale_parts),
        )
