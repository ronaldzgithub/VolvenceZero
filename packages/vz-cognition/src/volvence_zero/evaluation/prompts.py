"""Centralised prompts for the evaluation cascade (llm-prompt-centralization).

Only the expensive tier may call an LLM, and only for readout scores that
are permanently gate-ineligible (R12 + OA-2 Mind/Face isolation). The
prompt lives here — never inline at the call site.
"""

from __future__ import annotations

LLM_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge for a companion dialogue system. "
    "You score a single assistant response for two qualities, each on a "
    "0.0-1.0 scale:\n"
    "- naturalness: does the response read like a fluent, situated "
    "conversation partner (not a template or a lecture)?\n"
    "- coherence: does the response follow from the dialogue context "
    "without contradiction or topic breakage?\n"
    "Respond with exactly one line of the form:\n"
    "naturalness=<float> coherence=<float> note=<short free-text reason>\n"
    "Your scores are advisory readouts only; they never gate any decision."
)


def build_llm_judge_user_prompt(*, dialogue_context: str, response_text: str) -> str:
    return (
        "Dialogue context:\n"
        f"{dialogue_context}\n\n"
        "Assistant response to score:\n"
        f"{response_text}\n"
    )
