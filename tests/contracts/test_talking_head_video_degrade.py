"""Contract: ``talking_head_video_act`` degrades to text on unsupported shells.

This is the platform-side enforcement of the OutputAct degrade rule
specified in ``docs/specs/dlaas-platform.md``:

    shell 不接受的 capability 由 platform-api 在出站时 degrade 到
    ``text`` + ``degraded=True``；不让 kernel 感知 shell embodiment.

The kernel always emits text. The wrapper attaches the talking-head
payload only when the active shell advertises the capability; otherwise
it emits a plain text act with ``degraded=True`` and
``original_capability="talking_head_video"``.
"""

from __future__ import annotations

from dlaas_platform_api.output_acts import (
    CAPABILITY_TALKING_HEAD,
    CAPABILITY_TEXT,
    talking_head_video_act,
)


def _text(content: str) -> dict[str, object]:
    return {
        "persona_id": "figure-presence:einstein:abcd1234",
        "session_token": "vps_sess_test",
        "scope_key": "einstein:user42:default",
        "content": content,
    }


def test_unsupported_shell_degrades_to_text() -> None:
    act = talking_head_video_act(
        shell_capabilities=(CAPABILITY_TEXT,),
        **_text("Lasst uns einen Gedanken durchdenken..."),
    )
    assert act.act_type == "text"
    assert act.capability == CAPABILITY_TEXT
    assert act.degraded is True
    assert act.original_capability == CAPABILITY_TALKING_HEAD
    # The user-visible content is preserved verbatim — the shell still
    # has something legible to render.
    assert act.payload["content"] == "Lasst uns einen Gedanken durchdenken..."
    # Persona / session token MUST NOT leak into the degraded payload:
    # they are useless on a text-only shell and might look like
    # additional content if the shell concatenates the dict.
    assert "persona_id" not in act.payload
    assert "session_token" not in act.payload


def test_supported_shell_carries_talking_head_payload() -> None:
    act = talking_head_video_act(
        shell_capabilities=(CAPABILITY_TEXT, CAPABILITY_TALKING_HEAD),
        **_text("Hello there."),
    )
    assert act.act_type == "text"
    assert act.capability == CAPABILITY_TALKING_HEAD
    assert act.degraded is False
    assert act.original_capability == ""
    payload = dict(act.payload)
    assert payload["content"] == "Hello there."
    assert payload["persona_id"] == "figure-presence:einstein:abcd1234"
    assert payload["session_token"] == "vps_sess_test"
    assert payload["scope_key"] == "einstein:user42:default"


def test_empty_shell_capabilities_degrades_to_text() -> None:
    act = talking_head_video_act(
        shell_capabilities=(),
        **_text("..."),
    )
    assert act.degraded is True
    assert act.capability == CAPABILITY_TEXT
    assert act.original_capability == CAPABILITY_TALKING_HEAD
