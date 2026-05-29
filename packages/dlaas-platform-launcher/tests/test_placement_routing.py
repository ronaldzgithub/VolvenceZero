"""Tests for multi-pod placement routing (debt #17)."""

from __future__ import annotations

import pytest

from dlaas_platform_launcher import (
    AiIdPlacementRouter,
    MultiPodLauncher,
    PlacementCapacityError,
    PlacementNotFound,
    RuntimePod,
)


def _router_with_two_pods(capacity: int = 2) -> AiIdPlacementRouter:
    router = AiIdPlacementRouter()
    router.register_pod(RuntimePod(runtime_pod_id="pod-a", capacity=capacity))
    router.register_pod(RuntimePod(runtime_pod_id="pod-b", capacity=capacity))
    return router


def test_placement_is_sticky() -> None:
    router = _router_with_two_pods()
    first = router.place(ai_id="ai_1")
    second = router.place(ai_id="ai_1")
    assert first.runtime_pod_id == second.runtime_pod_id
    assert second.placement_generation == 1


def test_least_loaded_spreads_across_pods() -> None:
    router = _router_with_two_pods()
    p1 = router.place(ai_id="ai_1")
    p2 = router.place(ai_id="ai_2")
    assert p1.runtime_pod_id != p2.runtime_pod_id


def test_capacity_enforced() -> None:
    router = _router_with_two_pods(capacity=1)
    router.place(ai_id="ai_1")
    router.place(ai_id="ai_2")
    with pytest.raises(PlacementCapacityError):
        router.place(ai_id="ai_3")


def test_release_frees_capacity() -> None:
    router = _router_with_two_pods(capacity=1)
    router.place(ai_id="ai_1")
    router.place(ai_id="ai_2")
    router.release(ai_id="ai_1")
    # Now there is room again; re-place bumps generation.
    placed = router.place(ai_id="ai_3")
    assert placed.runtime_pod_id in {"pod-a", "pod-b"}


def test_resolve_unplaced_raises() -> None:
    router = _router_with_two_pods()
    with pytest.raises(PlacementNotFound):
        router.resolve("ghost")


def test_substrate_profile_constrains_placement() -> None:
    router = AiIdPlacementRouter()
    router.register_pod(
        RuntimePod(runtime_pod_id="vllm-pod", substrate_profile="vllm-shared")
    )
    router.register_pod(
        RuntimePod(runtime_pod_id="tf-pod", substrate_profile="shared-frozen")
    )
    placed = router.place(ai_id="ai_1", substrate_profile="vllm-shared")
    assert placed.runtime_pod_id == "vllm-pod"
    with pytest.raises(PlacementCapacityError):
        router.place(ai_id="ai_2", substrate_profile="no-such-profile")


def test_migrate_bumps_generation_and_moves() -> None:
    router = _router_with_two_pods()
    placed = router.place(ai_id="ai_1")
    other = "pod-b" if placed.runtime_pod_id == "pod-a" else "pod-a"
    migrated = router.migrate(ai_id="ai_1", target_pod_id=other)
    assert migrated.runtime_pod_id == other
    assert migrated.placement_generation == placed.placement_generation + 1


class _FakeManager:
    def __init__(self, pod_id: str) -> None:
        self.pod_id = pod_id
        self.acquired: list[str] = []
        self.released: list[str] = []

    async def acquire(self, *, ai_id, runtime_template_id, **kwargs):
        self.acquired.append(ai_id)
        return f"sm:{self.pod_id}:{ai_id}"

    def get(self, ai_id):
        return f"sm:{self.pod_id}:{ai_id}"

    def record_interaction(self, ai_id):
        pass

    async def release(self, ai_id):
        self.released.append(ai_id)


async def test_multipod_launcher_routes_to_owning_pod() -> None:
    launcher = MultiPodLauncher()
    mgr_a = _FakeManager("pod-a")
    mgr_b = _FakeManager("pod-b")
    launcher.register_pod(RuntimePod(runtime_pod_id="pod-a", capacity=1), mgr_a)
    launcher.register_pod(RuntimePod(runtime_pod_id="pod-b", capacity=1), mgr_b)

    sm1 = await launcher.acquire(ai_id="ai_1", runtime_template_id="companion")
    sm2 = await launcher.acquire(ai_id="ai_2", runtime_template_id="companion")
    # Two ai_ids on capacity-1 pods land on different pods.
    assert {sm1.split(":")[1], sm2.split(":")[1]} == {"pod-a", "pod-b"}

    # get() routes back to the owning pod consistently.
    assert launcher.get("ai_1") == sm1
    assert launcher.has("ai_1")

    await launcher.release("ai_1")
    assert not launcher.has("ai_1")
