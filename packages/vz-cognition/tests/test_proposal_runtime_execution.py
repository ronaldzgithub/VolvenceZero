"""Execution-boundary tests for cognition LLM proposal runtimes."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
from volvence_zero.social import (
    LLMCommonGroundProposalRuntime,
    LLMToMProposalRuntime,
)
from volvence_zero.substrate.runtime_execution import run_runtime_call


class _BlockingSharedState:
    def __init__(self) -> None:
        self.first_call_entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.call_count = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.thread_ids: set[int] = set()


class _SharedRuntimeOwner:
    pass


class _BlockingProvider:
    """A distinct provider wrapper bound to one shared execution owner."""

    def __init__(self, *, state: _BlockingSharedState, owner: object) -> None:
        self._state = state
        self._owner = owner

    @property
    def runtime_execution_owner(self) -> object:
        return self._owner

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        del prompt, max_new_tokens, temperature
        state = self._state
        with state._lock:
            state.call_count += 1
            state.active_calls += 1
            state.thread_ids.add(threading.get_ident())
            state.max_active_calls = max(
                state.max_active_calls,
                state.active_calls,
            )
            state.first_call_entered.set()
        try:
            if not state.release.wait(timeout=5.0):
                raise TimeoutError("test provider was not released")
            return "[]"
        finally:
            with state._lock:
                state.active_calls -= 1


def test_shared_proposal_provider_stays_off_loop_and_single_flight() -> None:
    """Semantic, ToM and common-ground calls share one serialized gate."""

    state = _BlockingSharedState()
    shared_owner = _SharedRuntimeOwner()
    providers = tuple(
        _BlockingProvider(state=state, owner=shared_owner) for _ in range(5)
    )
    assert len({id(provider) for provider in providers}) == 5
    assert all(
        provider.runtime_execution_owner is shared_owner for provider in providers
    )
    semantic_runtimes = tuple(
        LLMSemanticProposalRuntime(provider=provider)
        for provider in providers[:3]
    )
    tom_runtime = LLMToMProposalRuntime(provider=providers[3])
    common_ground_runtime = LLMCommonGroundProposalRuntime(provider=providers[4])

    def release_after_observing_first_call() -> None:
        if not state.first_call_entered.wait(timeout=2.0):
            return
        time.sleep(0.2)
        state.release.set()

    releaser = threading.Thread(
        target=release_after_observing_first_call,
        daemon=True,
    )
    releaser.start()

    async def exercise() -> tuple[object, ...]:
        async def wait_for_first_call() -> None:
            while not state.first_call_entered.is_set():
                await asyncio.sleep(0)

        semantic_tasks = tuple(
            asyncio.create_task(
                runtime.propose_async(
                    target_slot="commitment",
                    user_input=f"NPC {index} will inspect the north wall.",
                    substrate_snapshot=None,
                    memory_snapshot=None,
                    previous_snapshot=None,
                    turn_index=index,
                )
            )
            for index, runtime in enumerate(semantic_runtimes, start=1)
        )
        tasks = (
            *semantic_tasks,
            asyncio.create_task(
                tom_runtime.propose_async(
                    target_slot="belief_about_other",
                    user_input="Lan believes the north wall is leaking.",
                    substrate_snapshot=None,
                    memory_snapshot=None,
                    previous_snapshot=None,
                    turn_index=1,
                )
            ),
            asyncio.create_task(
                common_ground_runtime.propose_async(
                    user_input="We all saw water at the north wall.",
                    turn_index=1,
                )
            ),
        )
        await asyncio.wait_for(wait_for_first_call(), timeout=1.0)
        assert not state.release.is_set(), (
            "the asyncio loop did not regain control while synchronous "
            "proposal generation was blocked"
        )
        return await asyncio.gather(*tasks)

    try:
        results = asyncio.run(exercise())
    finally:
        state.release.set()
        releaser.join(timeout=1.0)

    assert len(results) == 5
    assert state.call_count == 5
    assert state.max_active_calls == 1
    assert len(state.thread_ids) == 1


def test_serial_owner_executor_recovers_after_failure_without_logging_secret(
    caplog,
) -> None:
    owner = _SharedRuntimeOwner()
    thread_ids: list[int] = []

    def fail_once() -> str:
        thread_ids.append(threading.get_ident())
        raise RuntimeError("provider-secret=must-not-enter-log")

    def succeed() -> str:
        thread_ids.append(threading.get_ident())
        return "recovered"

    async def exercise() -> str:
        with pytest.raises(RuntimeError, match="provider-secret"):
            await run_runtime_call(
                runtime=owner,
                operation=fail_once,
                operation_kind="semantic_proposal:commitment",
            )
        return await run_runtime_call(
            runtime=owner,
            operation=succeed,
            operation_kind="semantic_proposal:commitment",
        )

    assert asyncio.run(exercise()) == "recovered"
    assert len(set(thread_ids)) == 1
    assert "operation_kind=semantic_proposal:commitment" in caplog.text
    assert "cause_type=RuntimeError" in caplog.text
    assert "failure_fingerprint=" in caplog.text
    assert "provider-secret" not in caplog.text
