"""SHELL affordance descriptor for the L0 visual presence layer.

Registers ``talking_head_video.v1`` as a deploy-side embodiment
capability following the SHELL kind defined in
``docs/specs/affordance.md`` (parallel to ``text_streaming`` and
``voice``).

This descriptor describes **what the shell can render**, not what the
kernel should produce. The kernel still emits text only; the
``dlaas-platform-api`` ``OutputAct`` wrapper attaches the
``capability="talking_head_video"`` payload when the active shell
advertises it, and degrades to plain ``text`` (with ``degraded=True``
and ``original_capability="talking_head_video"``) when it does not.
See ``docs/specs/dlaas-platform.md`` §"OutputAct 包装".

Two consent grants are required so an operator can grant likeness
rendering without granting voice cloning (or vice versa):

* ``likeness.render`` — face / motion model permitted to drive
  user-visible frames.
* ``voice.clone`` — TTS may use a per-figure voice clone id.

The descriptor sets ``irreversible=True`` because once a likeness
frame leaves the lab to a third-party SFU it cannot be unsent. Audit
is required and a per-app webhook fires
``presence.consent_revoked`` when the operator pulls likeness via
``DELETE /v1/personas/:id`` on the rendering plane.

Two regimes are pre-blocked: ``minor_user`` (no likeness rendering
for under-13 users by default; an integrator can override per-tenant)
and ``crisis`` (drop to text during active rupture so the user gets a
calmer surface).
"""

from __future__ import annotations

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceSafety,
)


CONSENT_LIKENESS_RENDER = "likeness.render"
"""Operator-issued grant: this figure's face may drive user-visible frames."""

CONSENT_VOICE_CLONE = "voice.clone"
"""Operator-issued grant: TTS may use a per-figure voice clone."""


_PRESENCE_BLOCKED_REGIMES: tuple[str, ...] = (
    "minor_user",
    "crisis",
)


TALKING_HEAD_VIDEO_DESCRIPTOR = AffordanceDescriptor(
    name="talking_head_video.v1",
    kind=AffordanceKind.SHELL,
    version="0.1.0",
    display_name="Talking head video stream",
    description=(
        "Render the lifeform's response as a synchronized audio + video "
        "talking-head stream of the bound FigurePresenceArtifact, via "
        "the apps/presence-service rendering plane."
    ),
    when_to_use=(
        "Use when the active shell.embodiment list advertises "
        "talking_head_video, the bound figure has a non-None "
        "FigurePresenceArtifact with a still-valid likeness consent, "
        "and the current regime is neither minor_user nor crisis."
    ),
    when_not_to_use=(
        "Do not use when the shell does not advertise this capability "
        "(degrade to text), when consent has been revoked on the "
        "rendering plane, when the figure has no presence artifact "
        "attached, or when the active regime is minor_user or crisis."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "persona_id": {"type": "string"},
            "session_token": {"type": "string"},
            "scope_key": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["persona_id", "session_token", "scope_key"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "engine": {
                "type": "string",
                "enum": ["server_photoreal", "client_3d"],
            },
            "transport": {
                "type": "string",
                "enum": ["webrtc", "ws", "sse"],
            },
            "degraded": {"type": "boolean"},
            "degraded_reason": {"type": "string"},
            "first_frame_at": {"type": "string"},
        },
        "required": ["session_id", "engine", "transport", "degraded"],
        "additionalProperties": False,
    },
    cost_model=AffordanceCost(
        latency_class=AffordanceLatencyClass.FAST,
        monetary_class=AffordanceMonetaryClass.MEDIUM,
        rate_limit_per_minute=120,
    ),
    safety_model=AffordanceSafety(
        # The user does not need to confirm every utter() — confirming
        # consent happened off-stack at session-open time. Per-utterance
        # confirmation would make the conversation unusable.
        requires_user_confirmation=False,
        # A frame leaving the lab to a third-party SFU cannot be unsent.
        irreversible=True,
        # Both grants are required: rendering the face and using a
        # cloned voice are independently sensitive.
        requires_consent_grant=(CONSENT_LIKENESS_RENDER, CONSENT_VOICE_CLONE),
        blocked_in_regimes=_PRESENCE_BLOCKED_REGIMES,
        audit_required=True,
    ),
    preconditions=("scene.is_open", "presence.persona_active"),
    affordance_tags=("presence", "shell", "video", "likeness"),
    examples=(
        # apps/presence-service mints session_token; the shell calls
        # /v1/sessions/:id/utter with text from the kernel response.
        "talking_head_video.v1(persona_id='figure-presence:einstein:abcd1234', "
        "session_token='vps_sess_...', scope_key='einstein:userN:default', "
        "text='Lasst uns einen Gedanken durchdenken...')",
    ),
    source_path="lifeform_domain_figure.presence_affordance:TALKING_HEAD_VIDEO_DESCRIPTOR",
)


PRESENCE_AFFORDANCE_DESCRIPTORS: tuple[AffordanceDescriptor, ...] = (
    TALKING_HEAD_VIDEO_DESCRIPTOR,
)


__all__ = [
    "CONSENT_LIKENESS_RENDER",
    "CONSENT_VOICE_CLONE",
    "PRESENCE_AFFORDANCE_DESCRIPTORS",
    "TALKING_HEAD_VIDEO_DESCRIPTOR",
]
