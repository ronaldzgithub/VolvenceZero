"""Unit tests for the OpenAI-compat post-turn observability hook seam.

The router calls ``app['openai_compat_on_turn']`` after a successful
lifeform turn so a full-stack DLaaS host can record a cognition snapshot,
aligning OpenAI-path observability with the native dispatch. The seam must
be:

* additive — absent hook = no-op (bare lifeform-service host);
* decoupled — invoked with primitives (ai_id / session / session_id), so
  the adapter never imports the recorder;
* honest — skipped when no ``ai_id`` is bound (nothing to attribute to);
* safe — a hook that raises must NEVER break the chat completion.
"""

from __future__ import annotations

from types import SimpleNamespace

from lifeform_openai_compat.router import _ON_TURN_APP_KEY, _invoke_on_turn_hook
from lifeform_openai_compat.session_bridge import (
    LifeformCompletionResult,
    SessionResolution,
)


def _result(*, session: object | None = SimpleNamespace()) -> LifeformCompletionResult:
    return LifeformCompletionResult(
        response=SimpleNamespace(system_fingerprint="lifeform:test"),
        resolution=SessionResolution(session_id="sess-1", kind="explicit"),
        active_regime=None,
        active_abstract_action=None,
        pe_magnitude=0.0,
        rationale_tags=(),
        session=session,
    )


def _request(app: dict, metadata: dict) -> object:
    return SimpleNamespace(app=app)


def _parsed(metadata: dict) -> object:
    return SimpleNamespace(metadata=metadata)


def test_hook_called_with_bound_ai_id() -> None:
    calls: list[dict] = []

    def hook(request, *, ai_id, session, session_id):  # noqa: ANN001
        calls.append(
            {"ai_id": ai_id, "session": session, "session_id": session_id}
        )

    session = SimpleNamespace(tag="the-session")
    app = {_ON_TURN_APP_KEY: hook}
    _invoke_on_turn_hook(
        request=_request(app, {}),
        parsed=_parsed({"dlaas.ai_id": "ai_demo"}),
        result=_result(session=session),
    )
    assert calls == [
        {"ai_id": "ai_demo", "session": session, "session_id": "sess-1"}
    ]


def test_hook_skipped_without_ai_id() -> None:
    calls: list[int] = []
    app = {_ON_TURN_APP_KEY: lambda *a, **k: calls.append(1)}
    _invoke_on_turn_hook(
        request=_request(app, {}),
        parsed=_parsed({}),  # no dlaas.ai_id
        result=_result(),
    )
    assert calls == []


def test_no_hook_registered_is_noop() -> None:
    # No exception even though there is no hook and no ai_id.
    _invoke_on_turn_hook(
        request=_request({}, {}),
        parsed=_parsed({"dlaas.ai_id": "ai_demo"}),
        result=_result(),
    )


def test_hook_skipped_when_session_missing() -> None:
    calls: list[int] = []
    app = {_ON_TURN_APP_KEY: lambda *a, **k: calls.append(1)}
    _invoke_on_turn_hook(
        request=_request(app, {}),
        parsed=_parsed({"dlaas.ai_id": "ai_demo"}),
        result=_result(session=None),
    )
    assert calls == []


def test_hook_exception_is_swallowed() -> None:
    def boom(request, *, ai_id, session, session_id):  # noqa: ANN001
        raise RuntimeError("sink down")

    app = {_ON_TURN_APP_KEY: boom}
    # Must not raise.
    _invoke_on_turn_hook(
        request=_request(app, {}),
        parsed=_parsed({"dlaas.ai_id": "ai_demo"}),
        result=_result(),
    )
