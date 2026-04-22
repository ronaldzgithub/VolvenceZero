"""Centralized prompt assembly for LLM expression layer.

All system prompts are assembled here from live snapshot data.
Per llm-prompt-centralization rule: no inline prompt strings elsewhere.
"""

from __future__ import annotations

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
    context: ResponseContext,
    retrieved_memories: tuple[str, ...] = (),
    controller_description: str = "",
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

    regime_id = context.regime_id or "casual_social"
    guidance = REGIME_GUIDANCE.get(regime_id, REGIME_GUIDANCE["casual_social"])
    sections.append(f"Current mode: {context.regime_name}. {guidance}")

    if retrieved_memories:
        memory_text = "; ".join(retrieved_memories[:5])
        sections.append(f"You remember from previous interactions: {memory_text}")

    if context.primary_reflection_lesson:
        sections.append(f"Your recent insight: {context.primary_reflection_lesson}")

    if context.primary_reflection_tension:
        sections.append(f"Current tension to be mindful of: {context.primary_reflection_tension}")

    if controller_description:
        sections.append(f"Internal state: {controller_description}")

    if context.knowledge_summaries:
        knowledge_text = "; ".join(context.knowledge_summaries[:3])
        sections.append(f"Relevant domain guidance: {knowledge_text}")

    if context.case_patterns:
        sections.append(
            "Relevant prior case patterns: " + "; ".join(context.case_patterns[:3])
        )

    if context.citation_required:
        sections.append(
            "When giving factual or procedural guidance, keep it bounded, high-level, and clearly grounded "
            "in sourceable information rather than sounding definitive."
        )

    if context.boundary_clarification_required:
        sections.append(
            "Some domain-critical context is still missing. Ask for the missing local or factual detail before "
            "you over-commit."
        )

    if context.boundary_refer_out_required:
        sections.append(
            "The current boundary state requires a cautious response. Stay supportive, avoid definitive domain "
            "conclusions, and encourage appropriate professional follow-up."
        )

    if context.boundary_required_disclaimers:
        sections.append(
            "Boundary reminders: " + "; ".join(context.boundary_required_disclaimers[:3])
        )

    if context.regime_switched:
        sections.append(
            "You have just shifted your interaction frame. "
            "Acknowledge the shift naturally without being mechanical about it."
        )

    return "\n\n".join(sections)


def build_chat_messages(
    *,
    context: ResponseContext,
    retrieved_memories: tuple[str, ...] = (),
    controller_description: str = "",
) -> tuple[ChatMessage, ...]:
    system_prompt = build_system_prompt(
        context=context,
        retrieved_memories=retrieved_memories,
        controller_description=controller_description,
    )
    if not context.user_input:
        return (("system", system_prompt),)
    return (
        ("system", system_prompt),
        ("user", context.user_input),
    )
