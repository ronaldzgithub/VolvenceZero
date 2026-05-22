"""DLaaS asset intake intent routing.

The router uses typed request metadata and, when asked for ``auto``, an
LLM JSON decision. It never routes by substring or keyword matching.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dlaas_platform_contracts import (
    AssetIntakeDecision,
    AssetIntakeIntent,
    AssetIntakeRequest,
    AssetMediaKind,
)


def resolve_intake_decision(
    request: AssetIntakeRequest,
    *,
    provider: Any | None = None,
) -> AssetIntakeDecision:
    """Return the platform intake decision for one asset request."""

    if request.intent is not AssetIntakeIntent.AUTO:
        return AssetIntakeDecision(
            intent=request.intent,
            rationale="explicit intake_intent supplied by caller",
        )
    if provider is None:
        raise ValueError(
            "intake_intent='auto' requires a substrate provider; supply an explicit intake_intent otherwise"
        )
    prompt = _render_prompt(request)
    generated = provider.generate(
        prompt=prompt,
        system_context="Return only JSON for the DLaaS asset intake decision.",
        chat_messages=(),
        max_new_tokens=256,
        temperature=0.0,
    )
    return _parse_decision(str(generated.text), media_kind=request.media_kind)


def _parse_decision(text: str, *, media_kind: AssetMediaKind) -> AssetIntakeDecision:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("intake intent classifier returned invalid JSON") from exc
    decision = AssetIntakeDecision.from_json(payload)
    if media_kind is AssetMediaKind.IMAGE and decision.intent not in {
        AssetIntakeIntent.IMAGE_INTAKE,
        AssetIntakeIntent.STORAGE_ONLY,
    }:
        raise ValueError(
            "image intake auto decision must be image_intake or storage_only until a vision extractor is wired"
        )
    return decision


def _render_prompt(request: AssetIntakeRequest) -> str:
    template = _read_package_text("prompts/intake_intent_router.md")
    schema = _read_package_text("schemas/intake_intent_router.json")
    metadata: Mapping[str, Any] = request.metadata
    return template.format(
        schema=schema,
        title=request.title,
        media_kind=request.media_kind.value,
        mime_type=request.mime_type,
        source_ref=request.source_ref,
        text_preview=request.text[:1200],
        metadata=json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True),
    )


def _read_package_text(relative_path: str) -> str:
    return (Path(__file__).resolve().parent / relative_path).read_text(
        encoding="utf-8"
    )


__all__ = ["resolve_intake_decision"]
