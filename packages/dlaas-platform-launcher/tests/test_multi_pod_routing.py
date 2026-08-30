"""Tests for multi-pod routing: protocol, remote proxy, launcher (P2)."""

from __future__ import annotations

import pytest

from dlaas_platform_launcher import (
    ExplicitSessionForwardingLauncherProtocol,
    InstanceManager,
    InstanceNotFound,
    LauncherProtocol,
    MultiPodLauncher,
    OperationsForwardingLauncherProtocol,
    RemoteInstanceManager,
    RuntimePod,
    VerticalBrainForwardingLauncherProtocol,
)


# --- LauncherProtocol conformance -----------------------------------


def test_instance_manager_conforms_to_protocol() -> None:
    mgr = InstanceManager(vertical_resolver=lambda _n: None)
    assert isinstance(mgr, LauncherProtocol)


def test_multi_pod_launcher_conforms_to_protocol() -> None:
    launcher = MultiPodLauncher()
    assert isinstance(launcher, LauncherProtocol)
    assert isinstance(launcher, ExplicitSessionForwardingLauncherProtocol)
    assert isinstance(launcher, OperationsForwardingLauncherProtocol)
    assert isinstance(launcher, VerticalBrainForwardingLauncherProtocol)


# --- RemoteInstanceManager (fake transport) -------------------------


class _FakeEnvelope:
    def __init__(self, text: str) -> None:
        self._text = text

    def to_json(self) -> dict:
        return {"human_brief": self._text}


def _transport_recording(responses):
    calls = []

    async def transport(method, url, json_body):
        calls.append((method, url, json_body))
        return responses.get(url, (200, {"status": "ok"}))

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


async def test_remote_forward_interaction_routes_url() -> None:
    transport = _transport_recording(
        {"http://pod/dlaas/instances/ai_1/interactions": (200, {"reply": "hi"})}
    )
    proxy = RemoteInstanceManager(base_url="http://pod", transport=transport)
    body = await proxy.forward_interaction(ai_id="ai_1", envelope=_FakeEnvelope("x"))
    assert body == {"reply": "hi"}
    assert transport.calls[0][1].endswith("/dlaas/instances/ai_1/interactions")


async def test_remote_forward_404_raises_not_found() -> None:
    transport = _transport_recording(
        {"http://pod/dlaas/instances/ai_1/interactions": (404, {})}
    )
    proxy = RemoteInstanceManager(base_url="http://pod", transport=transport)
    with pytest.raises(InstanceNotFound):
        await proxy.forward_interaction(ai_id="ai_1", envelope=_FakeEnvelope("x"))


async def test_remote_forward_500_raises_runtime() -> None:
    transport = _transport_recording(
        {"http://pod/dlaas/instances/ai_1/interactions": (500, {"detail": "boom"})}
    )
    proxy = RemoteInstanceManager(base_url="http://pod", transport=transport)
    with pytest.raises(RuntimeError):
        await proxy.forward_interaction(ai_id="ai_1", envelope=_FakeEnvelope("x"))


async def test_remote_session_and_operations_forwarding_routes_urls() -> None:
    session_url = "http://pod/dlaas/v1/instances/ai_1/sessions"
    operations_url = (
        "http://pod/dlaas/v1/instances/ai_1/sessions/session_1/"
        "operations/context-packs"
    )
    transport = _transport_recording(
        {
            session_url: (201, {"created": True}),
            operations_url: (201, {"context_pack_id": "pack-1"}),
        }
    )
    proxy = RemoteInstanceManager(base_url="http://pod", transport=transport)
    assert isinstance(proxy, ExplicitSessionForwardingLauncherProtocol)
    assert isinstance(proxy, OperationsForwardingLauncherProtocol)
    assert await proxy.forward_session_create(
        ai_id="ai_1",
        payload={"session_id": "session_1"},
    ) == (201, {"created": True})
    assert await proxy.forward_operations_request(
        ai_id="ai_1",
        session_id="session_1",
        operation="context-packs",
        payload={"schema_version": "operations-context-request.v1"},
    ) == (201, {"context_pack_id": "pack-1"})
    assert [call[1] for call in transport.calls] == [session_url, operations_url]


async def test_remote_uniform_brain_forwarding_routes_url() -> None:
    brain_url = (
        "http://pod/dlaas/v1/instances/ai_1/sessions/session_1/"
        "brain/context-packs"
    )
    transport = _transport_recording(
        {brain_url: (201, {"context_pack_id": "pack-1"})}
    )
    proxy = RemoteInstanceManager(base_url="http://pod", transport=transport)
    assert isinstance(proxy, VerticalBrainForwardingLauncherProtocol)
    assert await proxy.forward_brain_request(
        ai_id="ai_1",
        session_id="session_1",
        operation="context-packs",
        payload={"request_id": "request-1"},
    ) == (201, {"context_pack_id": "pack-1"})
    assert transport.calls[0][1] == brain_url


async def test_remote_operations_forwarding_rejects_unknown_operation() -> None:
    proxy = RemoteInstanceManager(
        base_url="http://pod",
        transport=_transport_recording({}),
    )
    with pytest.raises(ValueError, match="unsupported Operations Brain operation"):
        await proxy.forward_operations_request(
            ai_id="ai_1",
            session_id="session_1",
            operation="arbitrary-path",
            payload={},
        )


# --- MultiPodLauncher routing ---------------------------------------


class _FakePodManager:
    def __init__(self, pod_id: str) -> None:
        self.pod_id = pod_id
        self.acquired: list[str] = []
        self.forwarded: list[str] = []
        self.sessions: list[str] = []
        self.operations: list[tuple[str, str]] = []

    async def acquire(self, *, ai_id, runtime_template_id, **kwargs):
        self.acquired.append(ai_id)
        return {"ok": True, "pod": self.pod_id}

    async def forward_interaction(self, *, ai_id, envelope):
        self.forwarded.append(ai_id)
        return {"pod": self.pod_id, "ai_id": ai_id}

    async def forward_session_create(self, *, ai_id, payload):
        self.sessions.append(ai_id)
        return 201, {"pod": self.pod_id, "session_id": payload["session_id"]}

    async def forward_operations_request(
        self,
        *,
        ai_id,
        session_id,
        operation,
        payload,
    ):
        del payload
        self.operations.append((ai_id, operation))
        return 201, {"pod": self.pod_id, "session_id": session_id}

    async def forward_brain_request(
        self,
        *,
        ai_id,
        session_id,
        operation,
        payload,
    ):
        del payload
        self.operations.append((ai_id, f"brain:{operation}"))
        return 201, {"pod": self.pod_id, "session_id": session_id}

    async def wake(self, *, ai_id, **kwargs):
        return {"woke": ai_id, "pod": self.pod_id}

    async def sleep(self, *, ai_id, **kwargs):
        return {"slept": ai_id, "pod": self.pod_id}


def _launcher_two_pods(capacity: int = 1):
    launcher = MultiPodLauncher()
    a = _FakePodManager("pod-a")
    b = _FakePodManager("pod-b")
    launcher.register_pod(RuntimePod(runtime_pod_id="pod-a", capacity=capacity), a)
    launcher.register_pod(RuntimePod(runtime_pod_id="pod-b", capacity=capacity), b)
    return launcher, a, b


async def test_acquire_places_and_forward_routes_consistently() -> None:
    launcher, a, b = _launcher_two_pods()
    await launcher.acquire(ai_id="ai_1", runtime_template_id="companion")
    await launcher.acquire(ai_id="ai_2", runtime_template_id="companion")
    # capacity-1 pods -> different pods.
    assert {a.acquired and "ai_1" in a.acquired, b.acquired and "ai_2" in b.acquired}

    r1 = await launcher.forward_interaction(
        ai_id="ai_1", envelope=_FakeEnvelope("hi")
    )
    # ai_1 forwards to the SAME pod it was placed on (sticky).
    owner = launcher.router.resolve("ai_1").runtime_pod_id
    assert r1["pod"] == owner


async def test_forward_unplaced_raises() -> None:
    launcher, _, _ = _launcher_two_pods()
    with pytest.raises(InstanceNotFound):
        await launcher.forward_interaction(
            ai_id="ghost", envelope=_FakeEnvelope("x")
        )


async def test_session_and_operations_follow_sticky_pod_placement() -> None:
    launcher, _, _ = _launcher_two_pods()
    await launcher.acquire(ai_id="ai_1", runtime_template_id="operations")
    owner = launcher.router.resolve("ai_1").runtime_pod_id
    session_status, session_body = await launcher.forward_session_create(
        ai_id="ai_1",
        payload={"session_id": "session_1"},
    )
    operations_status, operations_body = await launcher.forward_operations_request(
        ai_id="ai_1",
        session_id="session_1",
        operation="context-packs",
        payload={},
    )
    assert session_status == operations_status == 201
    assert session_body["pod"] == operations_body["pod"] == owner


async def test_uniform_brain_follows_sticky_pod_placement() -> None:
    launcher, _, _ = _launcher_two_pods()
    await launcher.acquire(ai_id="ai_1", runtime_template_id="coding")
    owner = launcher.router.resolve("ai_1").runtime_pod_id
    status, body = await launcher.forward_brain_request(
        ai_id="ai_1",
        session_id="session_1",
        operation="context-packs",
        payload={},
    )
    assert status == 201
    assert body["pod"] == owner


def test_get_not_supported() -> None:
    launcher, _, _ = _launcher_two_pods()
    with pytest.raises(NotImplementedError):
        launcher.get("ai_1")


async def test_status_and_overview_placement_derived() -> None:
    launcher, _, _ = _launcher_two_pods()
    await launcher.acquire(ai_id="ai_1", runtime_template_id="companion")
    status = launcher.status("ai_1")
    assert status.ai_id == "ai_1"
    overview = launcher.overview()
    assert any(item["ai_id"] == "ai_1" for item in overview)


async def test_sleep_release_frees_placement() -> None:
    launcher, _, _ = _launcher_two_pods(capacity=1)
    await launcher.acquire(ai_id="ai_1", runtime_template_id="companion")
    await launcher.sleep(ai_id="ai_1", release_instance=True)
    assert not launcher.has("ai_1")
