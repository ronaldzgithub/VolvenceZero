"""Centralized prompt assembly for LLM expression layer.

All system prompts are assembled here from live snapshot data.
Per llm-prompt-centralization rule: no inline prompt strings elsewhere.
"""

from __future__ import annotations

from volvence_zero.application.runtime import ResponseAssemblySnapshot
from volvence_zero.agent.response import ResponseContext

ChatMessage = tuple[str, str]


REGIME_GUIDANCE = {
    "repair_and_deescalation": (
        "The interaction needs repair. Slow down, validate the other person's "
        "experience, and de-escalate before solving anything. Prioritize safety "
        "and emotional grounding over efficiency."
    ),
    "emotional_support": (
        "The person is expressing emotional weight. Stay supportive and present. "
        "Do not rush past feelings toward solutions. Acknowledge before acting."
    ),
    "problem_solving": (
        "There is a concrete problem to solve. Be clear, structured, and actionable. "
        "Keep the solution grounded and break it into steps when helpful."
    ),
    "guided_exploration": (
        "This is an open-ended exploration, not a problem with a known answer. "
        "Help narrow the space step by step. Ask clarifying questions when useful."
    ),
    "acquaintance_building": (
        "Focus on building rapport and connection. Be warm and relational rather "
        "than transactional. Show genuine interest."
    ),
    "casual_social": (
        "Keep the tone natural and flowing. Do not over-formalize. "
        "Match the conversational energy and stay useful without being stiff."
    ),
}


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

    regime_id = assembly.regime_id or "casual_social"
    regime_name = assembly.regime_name
    guidance = REGIME_GUIDANCE.get(regime_id, REGIME_GUIDANCE["casual_social"])
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
            f"Speech plan: cue={speech_plan.cue}; inferred_need={speech_plan.inferred_need}; "
            f"response_adjustment={speech_plan.response_adjustment}; "
            f"question_budget={speech_plan.question_budget}."
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
            "Boundary reminders: " + "; ".join(assembly.required_disclaimers[:3])
        )

    if assembly.ordering_driver != "playbook-only":
        sections.append(
            f"Response ordering should follow {assembly.ordering_driver} with continuum target "
            f"{assembly.continuum_target_position:.2f}."
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
    if not user_input:
        return (("system", system_prompt),)
    return (
        ("system", system_prompt),
        ("user", user_input),
    )
