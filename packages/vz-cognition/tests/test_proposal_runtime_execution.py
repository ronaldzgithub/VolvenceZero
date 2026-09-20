"""Execution-boundary tests for cognition LLM proposal runtimes."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
from volvence_zero.social import (
    LLMCommonGroundProposalRuntime,
    LLMToMProposalRuntime,
)
from volvence_zero.substrate.runtime_execution import run_runtime_call
from volvence_zero.substrate.text_generation import HFTextGenerationProvider


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


class _OverlapDetectingTokenizer:
    """Fail loudly if two wrappers enter one tokenizer at the same time."""

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0
        self.call_count = 0
        self.fail_next = False

    def use(self) -> str:
        with self._state_lock:
            self.active_calls += 1
            self.max_active_calls = max(
                self.max_active_calls,
                self.active_calls,
            )
            overlapping = self.active_calls > 1
        try:
            if overlapping:
                raise RuntimeError("fast-tokenizer-overlap")
            time.sleep(0.03)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("tokenizer-secret=must-not-enter-log")
            self.call_count += 1
            return "tokenized"
        finally:
            with self._state_lock:
                self.active_calls -= 1


class _ModelResource:
    pass


class _HFResourceRuntime:
    def __init__(self, *, model: object, tokenizer: object) -> None:
        self._model = model
        self._tokenizer = tokenizer

    @property
    def runtime_execution_owner(self) -> object:
        return self._model

    @property
    def runtime_tokenizer_owner(self) -> object:
        return self._tokenizer


class _NestedProvider:
    """Provider that relies on recursive resource discovery via its runtime."""

    def __init__(self, *, runtime: _HFResourceRuntime) -> None:
        self._runtime = runtime

    @property
    def runtime_execution_owner(self) -> object:
        return self._runtime


class _DirectHFProbeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_active = False
        self.tokenizer_active = 0
        self.tokenizer_max_active = 0
        self.tokenizer_during_model = False
        self.fail_next_generation = False


class _FakeFastTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, *, state: _DirectHFProbeState) -> None:
        self._state = state

    def _tokenizer_call(self) -> None:
        with self._state.lock:
            self._state.tokenizer_active += 1
            self._state.tokenizer_max_active = max(
                self._state.tokenizer_max_active,
                self._state.tokenizer_active,
            )
            if self._state.model_active:
                self._state.tokenizer_during_model = True
        try:
            time.sleep(0.01)
        finally:
            with self._state.lock:
                self._state.tokenizer_active -= 1

    def apply_chat_template(self, *args, **kwargs) -> str:
        del args, kwargs
        self._tokenizer_call()
        return "rendered"

    def __call__(self, *args, **kwargs):
        del args, kwargs
        import torch

        self._tokenizer_call()
        return type(
            "FakeEncoding",
            (),
            {"input_ids": torch.tensor([[1, 2]])},
        )()

    def decode(self, *args, **kwargs) -> str:
        del args, kwargs
        self._tokenizer_call()
        return "generated"


class _FakeHFModel:
    def __init__(self, *, state: _DirectHFProbeState) -> None:
        self._state = state

    def generate(self, input_ids, **kwargs):
        del kwargs
        import torch

        with self._state.lock:
            if self._state.model_active:
                raise RuntimeError("hf-model-overlap")
            self._state.model_active = True
        try:
            time.sleep(0.04)
            if self._state.fail_next_generation:
                self._state.fail_next_generation = False
                raise RuntimeError("direct-provider-secret")
            return torch.cat((input_ids, torch.tensor([[3]])), dim=1)
        finally:
            with self._state.lock:
                self._state.model_active = False


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


def test_distinct_wrappers_recursively_share_model_owner_thread() -> None:
    model = _ModelResource()
    tokenizer = _OverlapDetectingTokenizer()
    providers = tuple(
        _NestedProvider(
            runtime=_HFResourceRuntime(model=model, tokenizer=tokenizer)
        )
        for _ in range(4)
    )
    thread_ids: set[int] = set()
    active_calls = 0
    max_active_calls = 0
    state_lock = threading.Lock()

    def operation() -> str:
        nonlocal active_calls, max_active_calls
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            thread_ids.add(threading.get_ident())
        try:
            time.sleep(0.03)
            return "generated"
        finally:
            with state_lock:
                active_calls -= 1

    async def exercise() -> tuple[str, ...]:
        return tuple(
            await asyncio.gather(
                *(
                    run_runtime_call(
                        runtime=provider,
                        operation=operation,
                        operation_kind="recursive_owner_test",
                    )
                    for provider in providers
                )
            )
        )

    assert asyncio.run(exercise()) == ("generated",) * 4
    assert max_active_calls == 1
    assert len(thread_ids) == 1


def test_shared_tokenizer_is_serialized_across_distinct_model_owners() -> None:
    tokenizer = _OverlapDetectingTokenizer()
    providers = tuple(
        _NestedProvider(
            runtime=_HFResourceRuntime(
                model=_ModelResource(),
                tokenizer=tokenizer,
            )
        )
        for _ in range(4)
    )

    async def exercise() -> tuple[str, ...]:
        return tuple(
            await asyncio.gather(
                *(
                    run_runtime_call(
                        runtime=provider,
                        operation=tokenizer.use,
                        operation_kind="tokenizer_gate_test",
                    )
                    for provider in providers
                )
            )
        )

    assert asyncio.run(exercise()) == ("tokenized",) * 4
    assert tokenizer.call_count == 4
    assert tokenizer.max_active_calls == 1


def test_shared_tokenizer_gate_recovers_after_failure_without_secret(
    caplog,
) -> None:
    tokenizer = _OverlapDetectingTokenizer()
    tokenizer.fail_next = True
    provider = _NestedProvider(
        runtime=_HFResourceRuntime(
            model=_ModelResource(),
            tokenizer=tokenizer,
        )
    )

    async def exercise() -> str:
        with pytest.raises(RuntimeError, match="tokenizer-secret"):
            await run_runtime_call(
                runtime=provider,
                operation=tokenizer.use,
                operation_kind="tokenizer_recovery_test",
            )
        return await run_runtime_call(
            runtime=provider,
            operation=tokenizer.use,
            operation_kind="tokenizer_recovery_test",
        )

    assert asyncio.run(exercise()) == "tokenized"
    assert tokenizer.call_count == 1
    assert tokenizer.max_active_calls == 1
    assert "operation_kind=tokenizer_recovery_test" in caplog.text
    assert "cause_type=RuntimeError" in caplog.text
    assert "failure_fingerprint=" in caplog.text
    assert "tokenizer-secret" not in caplog.text


def test_direct_hf_providers_guard_full_tokenizer_generation_interval() -> None:
    state = _DirectHFProbeState()
    tokenizer = _FakeFastTokenizer(state=state)
    model = _FakeHFModel(state=state)
    providers = tuple(
        HFTextGenerationProvider(
            model=model,
            tokenizer=tokenizer,
            use_chat_template=True,
        )
        for _ in range(4)
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = tuple(
            executor.map(
                lambda provider: provider.generate(prompt="hello"),
                providers,
            )
        )

    assert results == ("generated",) * 4
    assert state.tokenizer_max_active == 1
    assert state.tokenizer_during_model is False

    state.fail_next_generation = True
    with pytest.raises(RuntimeError, match="direct-provider-secret"):
        providers[0].generate(prompt="fail once")
    assert providers[1].generate(prompt="recover") == "generated"


def test_live_llm_apprenticeship_extraction_uses_owner_thread_and_heartbeat() -> None:
    from volvence_zero.apprenticeship import (
        ApprenticeshipAlignmentModule,
        LLMGuidanceConstraintExtractor,
    )

    class _ApprenticeshipProvider:
        def __init__(self, *, owner: object) -> None:
            self._owner = owner
            self._state_lock = threading.Lock()
            self.active_calls = 0
            self.max_active_calls = 0
            self.thread_ids: set[int] = set()

        @property
        def runtime_execution_owner(self) -> object:
            return self._owner

        def generate(
            self,
            *,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.0,
        ) -> str:
            del prompt, max_new_tokens, temperature
            with self._state_lock:
                self.active_calls += 1
                self.max_active_calls = max(
                    self.max_active_calls,
                    self.active_calls,
                )
                self.thread_ids.add(threading.get_ident())
            try:
                time.sleep(0.04)
                return (
                    '[{"statement":"acknowledge before solving",'
                    '"level":"abstract","polarity":1,'
                    '"target_key":"acknowledge first",'
                    '"confidence":0.9}]'
                )
            finally:
                with self._state_lock:
                    self.active_calls -= 1

    provider = _ApprenticeshipProvider(owner=_SharedRuntimeOwner())
    extractor = LLMGuidanceConstraintExtractor(provider)
    event_loop_thread_id = threading.get_ident()

    async def exercise() -> tuple[object, object, int]:
        heartbeat_ticks = 0
        stop_heartbeat = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal heartbeat_ticks
            while not stop_heartbeat.is_set():
                heartbeat_ticks += 1
                await asyncio.sleep(0)

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            first, second = await asyncio.gather(
                ApprenticeshipAlignmentModule(
                    apprenticeship=True,
                    extractor=extractor,
                ).process_standalone(
                    apprenticeship=True,
                    guidance_text="First teaching turn.",
                    turn_index=1,
                ),
                ApprenticeshipAlignmentModule(
                    apprenticeship=True,
                    extractor=extractor,
                ).process_standalone(
                    apprenticeship=True,
                    guidance_text="Second teaching turn.",
                    turn_index=2,
                ),
            )
        finally:
            stop_heartbeat.set()
            await heartbeat_task
        return first, second, heartbeat_ticks

    first, second, heartbeat_ticks = asyncio.run(exercise())

    assert first.value.guidance_constraints[0].statement == (
        "acknowledge before solving"
    )
    assert second.value.guidance_constraints[0].source_turn == 2
    assert heartbeat_ticks > 1
    assert provider.max_active_calls == 1
    assert len(provider.thread_ids) == 1
    assert event_loop_thread_id not in provider.thread_ids
