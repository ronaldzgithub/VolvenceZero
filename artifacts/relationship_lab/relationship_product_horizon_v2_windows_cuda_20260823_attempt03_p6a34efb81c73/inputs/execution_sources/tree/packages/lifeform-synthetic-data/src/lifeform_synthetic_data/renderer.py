"""Expression-only trajectory rendering over stable text-slot IDs."""

from __future__ import annotations

from .contracts import ExperienceTrajectory
from .llm import (
    BudgetLedger,
    JsonCompletion,
    JsonCompletionClient,
    LLMRenderError,
    LLMResponseError,
)
from .prompt_manager import RenderPrompt, build_render_prompt
from .world import replace_rendered_text


class RenderContractError(LLMResponseError):
    """The model returned JSON that violates the stable-slot contract."""


def render_trajectory(
    trajectory: ExperienceTrajectory,
    *,
    client: JsonCompletionClient,
    budget: BudgetLedger,
    max_output_tokens: int,
) -> tuple[ExperienceTrajectory, JsonCompletion, RenderPrompt]:
    prompt = build_render_prompt(trajectory)
    reservation = budget.reserve(
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        max_output_tokens=max_output_tokens,
    )
    try:
        completion = client.complete_json(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )
    except LLMRenderError:
        budget.release(reservation)
        raise
    budget.settle(reservation, completion)
    rendered_slots = validate_render_completion(
        trajectory=trajectory,
        completion=completion,
    )
    rendered = replace_rendered_text(
        trajectory,
        rendered_slots=rendered_slots,
        model_id=completion.model_id,
        prompt_hash=prompt.prompt_hash,
    )
    return rendered, completion, prompt


def validate_render_completion(
    *,
    trajectory: ExperienceTrajectory,
    completion: JsonCompletion,
) -> tuple[tuple[str, str], ...]:
    payload = completion.payload()
    if set(payload) != {"trajectory_id", "slots"}:
        raise RenderContractError("renderer root must contain exactly trajectory_id and slots")
    trajectory_id = payload["trajectory_id"]
    if trajectory_id != trajectory.trajectory_id:
        raise RenderContractError("renderer changed trajectory_id")
    raw_slots = payload["slots"]
    if not isinstance(raw_slots, list):
        raise RenderContractError("renderer slots must be an array")
    slots: list[tuple[str, str]] = []
    for index, raw_slot in enumerate(raw_slots):
        if not isinstance(raw_slot, dict):
            raise RenderContractError(f"renderer slots[{index}] must be an object")
        if set(raw_slot) != {"turn_id", "text"}:
            raise RenderContractError(f"renderer slots[{index}] must contain exactly turn_id and text")
        turn_id = raw_slot["turn_id"]
        text = raw_slot["text"]
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise RenderContractError(f"renderer slots[{index}].turn_id must be non-empty")
        if not isinstance(text, str) or not text.strip():
            raise RenderContractError(f"renderer slots[{index}].text must be non-empty")
        slots.append((turn_id, text))
    return tuple(slots)


__all__ = [
    "RenderContractError",
    "render_trajectory",
    "validate_render_completion",
]
