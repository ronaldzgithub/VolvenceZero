"""Centralized prompt and renderer-schema loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .canonical import stable_hash
from .contracts import ExperienceTrajectory
from .world import text_slot_request

_REQUEST_MARKER = "{{REQUEST_JSON}}"


@dataclass(frozen=True)
class RenderPrompt:
    system_prompt: str
    user_prompt: str
    prompt_hash: str
    request_json: str


def build_render_prompt(trajectory: ExperienceTrajectory) -> RenderPrompt:
    root = Path(__file__).parent
    system_path = root / "prompts" / "render_slots.system.md"
    user_path = root / "prompts" / "render_slots.user.md"
    schema_path = root / "schemas" / "render-slots.schema.json"
    system_template = _read_non_empty(system_path)
    user_template = _read_non_empty(user_path)
    if user_template.count(_REQUEST_MARKER) != 1:
        raise ValueError("render_slots.user.md must contain exactly one {{REQUEST_JSON}} marker")
    schema = _load_json_object(schema_path)
    request = {
        "trajectory_id": trajectory.trajectory_id,
        "scenario_ref": trajectory.scenario_ref,
        "variation_seed": trajectory.provenance.seed,
        "slots": list(text_slot_request(trajectory)),
        "output_schema": schema,
    }
    request_json = json.dumps(
        request,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    user_prompt = user_template.replace(_REQUEST_MARKER, request_json)
    return RenderPrompt(
        system_prompt=system_template,
        user_prompt=user_prompt,
        prompt_hash=stable_hash(
            {
                "system": system_template,
                "user_template": user_template,
                "schema": schema,
            }
        ),
        request_json=request_json,
    )


def _read_non_empty(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt asset is empty: {path}")
    return text


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON schema asset: {path}") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"JSON schema asset root must be an object: {path}")
    return decoded


__all__ = ["RenderPrompt", "build_render_prompt"]
