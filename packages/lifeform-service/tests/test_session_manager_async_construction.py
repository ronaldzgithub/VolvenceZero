from __future__ import annotations

import asyncio
import threading
import time

import pytest

from lifeform_service import SessionManager
from lifeform_service import SessionAlreadyExistsError
from lifeform_service.templates import (
    ContentAddressedTemplateBinding,
    TemplateContext,
)


class _BlockingLifeform:
    def __init__(
        self,
        *,
        session_delay_seconds: float,
        fail_on_create: bool = False,
    ) -> None:
        self._session_delay_seconds = session_delay_seconds
        self._fail_on_create = fail_on_create
        self.shutdown_calls = 0

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    def create_session(self, *, session_id: str) -> object:
        time.sleep(self._session_delay_seconds)
        if self._fail_on_create:
            raise RuntimeError("session construction failed")
        return object()


class _BlockingContentAddressedAdapter:
    def __init__(self, lifeform: _BlockingLifeform) -> None:
        self._lifeform = lifeform
        self.calls = 0

    def build_session_context_from_content_addressed_template(self, **_kwargs):
        self.calls += 1
        time.sleep(0.12)
        return self._lifeform, TemplateContext(payload={})


async def test_fresh_session_construction_does_not_starve_event_loop() -> None:
    factory_calls = 0

    def blocking_factory(_runtime) -> _BlockingLifeform:
        nonlocal factory_calls
        factory_calls += 1
        time.sleep(0.12)
        return _BlockingLifeform(session_delay_seconds=0.12)

    manager = SessionManager(
        lifeform_factory=blocking_factory,
        vertical_name="blocking-test",
        idle_eviction_seconds=None,
    )
    stop_ticker = asyncio.Event()
    tick_count = 0

    async def ticker() -> None:
        nonlocal tick_count
        while not stop_ticker.is_set():
            tick_count += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    session = await manager.create_session(session_id="fresh-session")
    stop_ticker.set()
    await ticker_task

    assert session is await manager.get_session("fresh-session")
    assert factory_calls == 1
    assert tick_count >= 5


async def test_content_addressed_resolve_and_adapter_do_not_starve_loop(
    tmp_path,
    monkeypatch,
) -> None:
    digest = "a" * 64
    templates_root = tmp_path / "novel-worlds"
    blob_path = templates_root / "blobs" / f"{digest}.json"
    blob_path.parent.mkdir(parents=True)
    blob_path.write_text("{}", encoding="utf-8")
    binding = ContentAddressedTemplateBinding(
        template_id="scene-before",
        template_uri=f"novel-worlds/blobs/{digest}.json",
        template_bundle_sha256=digest,
        template_source_sha256="b" * 64,
    )
    lifeform = _BlockingLifeform(session_delay_seconds=0.01)
    adapter = _BlockingContentAddressedAdapter(lifeform)
    original_resolve = ContentAddressedTemplateBinding.resolve_under

    def blocking_resolve(self, *args, **kwargs):
        time.sleep(0.12)
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(ContentAddressedTemplateBinding, "resolve_under", blocking_resolve)
    manager = SessionManager(
        lifeform_factory=lambda _runtime: lifeform,
        vertical_name="content-addressed-test",
        template_adapter=adapter,
        templates_root_dir=templates_root,
        idle_eviction_seconds=None,
    )
    tick_count = 0
    stop_ticker = asyncio.Event()

    async def ticker() -> None:
        nonlocal tick_count
        while not stop_ticker.is_set():
            tick_count += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    await manager.create_session(
        session_id="content-addressed",
        template_binding=binding,
    )
    stop_ticker.set()
    await ticker_task

    assert adapter.calls == 1
    assert tick_count >= 5


async def test_cancelled_waiter_leaves_single_flight_to_commit() -> None:
    factory_started = threading.Event()
    release_factory = threading.Event()
    factory_calls = 0

    def blocking_factory(_runtime) -> _BlockingLifeform:
        nonlocal factory_calls
        factory_calls += 1
        factory_started.set()
        assert release_factory.wait(timeout=2)
        return _BlockingLifeform(session_delay_seconds=0.01)

    manager = SessionManager(
        lifeform_factory=blocking_factory,
        vertical_name="cancel-test",
        idle_eviction_seconds=None,
    )
    first = asyncio.create_task(manager.create_session(session_id="same-session"))
    assert await asyncio.to_thread(factory_started.wait, 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    joined = asyncio.create_task(manager.create_session(session_id="same-session"))
    release_factory.set()
    with pytest.raises(SessionAlreadyExistsError):
        await joined

    committed = await manager.get_session("same-session")
    assert committed is not None
    assert factory_calls == 1


async def test_failed_creation_cleans_lifeform_and_allows_retry() -> None:
    built: list[_BlockingLifeform] = []

    def factory(_runtime) -> _BlockingLifeform:
        lifeform = _BlockingLifeform(
            session_delay_seconds=0.01,
            fail_on_create=not built,
        )
        built.append(lifeform)
        return lifeform

    manager = SessionManager(
        lifeform_factory=factory,
        vertical_name="failure-cleanup-test",
        idle_eviction_seconds=None,
    )
    with pytest.raises(RuntimeError, match="session construction failed"):
        await manager.create_session(session_id="retryable-session")

    recovered = await manager.create_session(session_id="retryable-session")
    assert recovered is await manager.get_session("retryable-session")
    assert len(built) == 2
    assert built[0].shutdown_calls == 1
    assert built[1].shutdown_calls == 0
