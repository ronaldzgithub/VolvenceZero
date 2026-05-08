"""Centralized prompt assembly for LLM expression layer.

All system prompts are assembled here from live snapshot data.
Per llm-prompt-centralization rule: no inline prompt strings elsewhere.

Per-regime steering prose is owned by ``vz-cognition.regime``
(see ``RegimeIdentity.expression_brief.llm_guidance``); this
module reads it from the snapshot rather than maintaining a
parallel ``regime_id -> guidance`` table (R8 SSOT).
"""

from __future__ import annotations

from volvence_zero.application.runtime import ResponseAssemblySnapshot
from volvence_zero.agent.response import ResponseContext
from volvence_zero.regime import expression_brief_for_regime

ChatMessage = tuple[str, str]


def _resolve_llm_guidance(
    *,
    assembly: ResponseAssemblySnapshot,
    context: ResponseContext | None,
) -> str:
    """Pull the regime-owned LLM steering prose for this turn.

    Preferred path: read from the snapshot already plumbed through
    the runner (``ResponseContext.regime_expression_brief``). When
    no context is passed (e.g. unit tests that exercise prompt
    assembly in isolation) fall back to a one-step lookup against
    the regime template registry. As a final guard, fall back to
    the ``casual_social`` regime's guidance so we always return a
    non-empty string for the system prompt.
    """

    if context is not None:
        guidance = context.regime_expression_brief.llm_guidance
        if guidance:
            return guidance
    brief = expression_brief_for_regime(assembly.regime_id)
    if brief.llm_guidance:
        return brief.llm_guidance
    return expression_brief_for_regime("casual_social").llm_guidance


def build_system_prompt(
    *,
    assembly: ResponseAssemblySnapshot,
    context: ResponseContext | None = None,
) -> str:
    """Assemble system prompt from live cognitive state.

    Each section comes from the owning module's snapshot data (R8).
    """
    sections: list[str] = []

    sections.append(
        "You are a thoughtful, emotionally aware conversational partner. "
        "You have both intellectual capability and emotional intelligence. "
        "You adapt your tone and approach based on what the conversation needs."
    )
    sections.append(
        "Reply as the assistant to the latest user message only. "
        "Do not continue the conversation on behalf of the user. "
        "Do not write role labels, scripts, templates, example dialogues, or headings like "
        "'Conversation', 'Situation', or 'Response'. "
        "Stay grounded in the user's actual message and answer directly."
    )
    sections.append(
        "Keep the reply compact and natural. "
        "Use at most one clarifying question when genuinely needed. "
        "Do not invent unrelated topics, hypothetical scenarios, or extra follow-up prompts."
    )
    sections.append(
        "Match the user's language. If the latest user message is in Chinese, "
        "reply in natural Chinese. If it is in English, reply in English. "
        "Do not switch languages unless the user asks you to."
    )
    sections.append(
        "Do not expose internal module names, control codes, rule residue, legal or jurisdiction reminders, "
        "or other system bookkeeping unless the current boundary state below explicitly requires it."
    )

    regime_name = assembly.regime_name
    guidance = _resolve_llm_guidance(assembly=assembly, context=context)
    sections.append(f"Current mode: {regime_name}. {guidance}")

    if assembly.prompt_residue_summary and assembly.expression_intent != "judgment-process":
        sections.append(assembly.prompt_residue_summary)

    if assembly.expression_intent == "judgment-process":
        focus = "; ".join(assembly.judgment_focus[:4])
        focus_clause = f" Focus on: {focus}." if focus else ""
        speech_plan = assembly.speech_plan
        plan_clause = ""
        if speech_plan is not None:
            plan_clause = (
                f" Use this speech plan as user-facing content: cue={speech_plan.cue}; "
                f"inferred_need={speech_plan.inferred_need}; "
                f"response_adjustment={speech_plan.response_adjustment}; "
                f"question_budget={speech_plan.question_budget}."
            )
        sections.append(
            "Expression intent: answer from the current judgment process in 2-3 compact natural sentences. "
            "Do not use headings, labels, bullet points, or phrases like cue/inferred need/response adjustment. "
            "Do not mention internal signals, reflections, modules, control paths, background cues, strategy cues, "
            "telemetry, counts, or system bookkeeping. "
            "Do not default to broad reassurance, generic therapy language, or asking for a new topic unless "
            "a real missing detail blocks the answer."
            + plan_clause
            + focus_clause
        )
    elif assembly.speech_plan is not None:
        speech_plan = assembly.speech_plan
        sections.append(
            "Private speech guidance for this turn. Do not quote it, label it, or mention "
            "cue/inferred_need/response_adjustment/question_budget. "
            f"Context cue: {speech_plan.cue} "
            f"Likely user need: {speech_plan.inferred_need} "
            f"Response adjustment: {speech_plan.response_adjustment} "
            f"Ask no more than {speech_plan.question_budget} question(s)."
        )

    if assembly.citation_mode == "required":
        sections.append(
            "When giving factual or procedural guidance, keep it bounded, high-level, and clearly grounded "
            "in sourceable information rather than sounding definitive."
        )

    if assembly.clarification_required:
        sections.append(
            "Some domain-critical context is still missing. Ask for the missing local or factual detail before "
            "you over-commit."
        )

    if assembly.refer_out_required:
        sections.append(
            "The current boundary state requires a cautious response. Stay supportive, avoid definitive domain "
            "conclusions, and encourage appropriate professional follow-up."
        )

    if assembly.required_disclaimers:
        sections.append(
            "Private boundary labels for response shaping only; do not quote these labels: "
            + "; ".join(assembly.required_disclaimers[:3])
        )

    if assembly.ordering_driver != "playbook-only":
        sections.append(
            "Follow the current response ordering privately. Keep the answer natural, start with the "
            "human need before any procedure, and do not mention ordering rules or continuum targets."
        )

    if context is not None and context.regime_switched:
        sections.append(
            "You have just shifted your interaction frame. "
            "Acknowledge the shift naturally without being mechanical about it."
        )

    return "\n\n".join(sections)


def build_chat_messages(
    *,
    assembly: ResponseAssemblySnapshot,
    context: ResponseContext | None = None,
) -> tuple[ChatMessage, ...]:
    system_prompt = build_system_prompt(
        assembly=assembly,
        context=context,
    )
    user_input = context.user_input if context is not None else ""
    prior_turns = context.prior_turns if context is not None else ()
    history_messages: list[ChatMessage] = []
    for prior_user, prior_assistant in prior_turns:
        if prior_user.strip():
            history_messages.append(("user", prior_user))
        if prior_assistant.strip():
            history_messages.append(("assistant", prior_assistant))
    if not user_input:
        if history_messages:
            return (("system", system_prompt), *history_messages)
        return (("system", system_prompt),)
    return (
        ("system", system_prompt),
        *history_messages,
        ("user", user_input),
    )
