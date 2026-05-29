"""Unit tests for the bounded LRU PEFT adapter cache.

Uses a fake :class:`PeftAdapterOps` so the LRU / hit-miss / eviction
logic and the R2 disable-on-exit invariant are tested without torch.
"""

from __future__ import annotations

from volvence_zero.substrate import (
    PeftAdapterCache,
    adapter_name_for,
    peft_cache_max,
)


class _FakePeftModel:
    def __init__(self) -> None:
        self.adapters: set[str] = set()
        self.active: str | None = None
        self.enabled = False
        self.events: list[tuple[str, str]] = []


class _FakeOps:
    def __init__(self) -> None:
        self.model = _FakePeftModel()
        self.create_calls = 0
        self.load_calls = 0

    def create(self, base_model, checkpoint_dir, name):
        self.create_calls += 1
        self.model.adapters.add(name)
        self.model.events.append(("create", name))
        return self.model

    def load(self, peft_model, checkpoint_dir, name):
        self.load_calls += 1
        peft_model.adapters.add(name)
        peft_model.events.append(("load", name))

    def set_active(self, peft_model, name):
        peft_model.active = name
        peft_model.events.append(("set_active", name))

    def enable(self, peft_model):
        peft_model.enabled = True
        peft_model.events.append(("enable", ""))

    def disable(self, peft_model):
        peft_model.enabled = False
        peft_model.events.append(("disable", ""))

    def delete(self, peft_model, name):
        peft_model.adapters.discard(name)
        peft_model.events.append(("delete", name))


def test_first_activation_is_miss_and_creates() -> None:
    ops = _FakeOps()
    cache = PeftAdapterCache(ops=ops, max_adapters=4)
    with cache.activate(base_model=object(), checkpoint_dir="/ck/a"):
        pass
    assert ops.create_calls == 1
    assert ops.load_calls == 0
    assert cache.misses == 1
    assert cache.hits == 0
    assert cache.resident_count == 1


def test_repeat_activation_is_hit_no_reload() -> None:
    ops = _FakeOps()
    cache = PeftAdapterCache(ops=ops, max_adapters=4)
    with cache.activate(base_model=object(), checkpoint_dir="/ck/a"):
        pass
    with cache.activate(base_model=object(), checkpoint_dir="/ck/a"):
        pass
    assert ops.create_calls == 1
    assert ops.load_calls == 0
    assert cache.hits == 1
    assert cache.misses == 1


def test_second_distinct_adapter_uses_load_not_create() -> None:
    ops = _FakeOps()
    cache = PeftAdapterCache(ops=ops, max_adapters=4)
    with cache.activate(base_model=object(), checkpoint_dir="/ck/a"):
        pass
    with cache.activate(base_model=object(), checkpoint_dir="/ck/b"):
        pass
    assert ops.create_calls == 1
    assert ops.load_calls == 1
    assert cache.resident_count == 2


def test_lru_eviction_drops_oldest() -> None:
    ops = _FakeOps()
    cache = PeftAdapterCache(ops=ops, max_adapters=2)
    with cache.activate(base_model=object(), checkpoint_dir="/ck/a"):
        pass
    with cache.activate(base_model=object(), checkpoint_dir="/ck/b"):
        pass
    # Touch a so b becomes the LRU victim candidate, then add c.
    with cache.activate(base_model=object(), checkpoint_dir="/ck/a"):
        pass
    with cache.activate(base_model=object(), checkpoint_dir="/ck/c"):
        pass
    assert cache.resident_count == 2
    deleted = [name for kind, name in ops.model.events if kind == "delete"]
    assert deleted == [adapter_name_for("/ck/b")]


def test_disable_called_on_exit_for_r2() -> None:
    ops = _FakeOps()
    cache = PeftAdapterCache(ops=ops, max_adapters=2)
    with cache.activate(base_model=object(), checkpoint_dir="/ck/a"):
        assert ops.model.enabled is True
    assert ops.model.enabled is False


def test_disable_called_even_on_exception() -> None:
    ops = _FakeOps()
    cache = PeftAdapterCache(ops=ops, max_adapters=2)
    try:
        with cache.activate(base_model=object(), checkpoint_dir="/ck/a"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert ops.model.enabled is False


def test_peft_cache_max_env(monkeypatch) -> None:
    monkeypatch.delenv("VZ_LORA_CACHE_MAX", raising=False)
    assert peft_cache_max() == 4
    monkeypatch.setenv("VZ_LORA_CACHE_MAX", "8")
    assert peft_cache_max() == 8
