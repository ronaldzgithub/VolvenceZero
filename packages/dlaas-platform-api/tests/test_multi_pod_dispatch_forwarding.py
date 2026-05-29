"""Test that the api takes the forward path when the launcher is multi-pod."""

from __future__ import annotations

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from dlaas_platform_contracts import InteractionEnvelope, InteractionType
from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY

from dlaas_platform_api.app import _dispatch_envelope_to_instance


def _app_with(launcher) -> web.Application:
    app = web.Application()
    app[INSTANCE_MANAGER_APP_KEY] = launcher
    return app


def _envelope() -> InteractionEnvelope:
    return InteractionEnvelope(
        contract_id="ctr_1",
        session_id="s1",
        end_user_ref="alice",
        interaction_type=InteractionType.CHAT,
        human_brief="hello",
    )


class _ForwardingLauncher:
    """Has forward_interaction (the multi-process discriminator)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def forward_interaction(self, *, ai_id, envelope):
        self.calls.append(ai_id)
        # Raise a RuntimeError so the handler returns 502 WITHOUT reaching
        # the audit/usage stores — this isolates the "did we take the
        # forward branch?" assertion from the full app scaffolding.
        raise RuntimeError("forwarded-to-pod")


class _NonForwardingLauncher:
    """No forward_interaction -> api should NOT take the forward path."""

    def get(self, ai_id):  # present so it is not the discriminator
        raise AssertionError("local path should resolve via _resolve_session_manager")


async def test_forward_branch_taken_when_launcher_supports_it() -> None:
    launcher = _ForwardingLauncher()
    req = make_mocked_request(
        "POST", "/dlaas/instances/ai_1/interactions", app=_app_with(launcher)
    )

    resp = await _dispatch_envelope_to_instance(req, "ai_1", _envelope())
    # 502 from the forward branch's RuntimeError handler proves the
    # envelope was routed to forward_interaction (not the local path).
    assert resp.status == 502
    assert launcher.calls == ["ai_1"]


async def test_no_forward_attr_skips_remote_path() -> None:
    # A launcher without forward_interaction must NOT be treated as
    # multi-pod; the handler proceeds to the local resolution path.
    req = make_mocked_request(
        "POST",
        "/dlaas/instances/ai_1/interactions",
        app=_app_with(_NonForwardingLauncher()),
    )
    # Local path: launcher does not conform to LauncherProtocol (missing
    # overview/status/wake/sleep) so _resolve_session_manager falls back
    # to app["session_manager"], which is absent -> KeyError surfaces.
    # We only assert the forward branch was skipped (no AssertionError
    # from a forward call), which manifests as a non-502 outcome.
    raised = False
    try:
        await _dispatch_envelope_to_instance(req, "ai_1", _envelope())
    except KeyError:
        raised = True
    assert raised
