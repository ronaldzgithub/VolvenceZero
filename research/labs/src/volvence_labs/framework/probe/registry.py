"""Probe registry: 通过 id 查找 probe 类。

用法：
    from volvence_labs.framework.probe import register_probe, BaseProbe

    @register_probe
    class MyProbe(BaseProbe):
        id = "my-probe-v1"
        ...

registry 在模块导入时累积；volvence_labs.probes 的 __init__ 负责把所有内置
probe 模块 import 一遍，从而触发注册。
"""

from __future__ import annotations

from typing import Iterable

from .types import BaseProbe


class ProbeRegistry:
    def __init__(self) -> None:
        self._probes: dict[str, type[BaseProbe]] = {}

    def register(self, cls: type[BaseProbe]) -> type[BaseProbe]:
        pid = getattr(cls, "id", None)
        if not pid or not isinstance(pid, str):
            raise ValueError(f"probe class {cls!r} must set a non-empty id")
        if pid in self._probes and self._probes[pid] is not cls:
            raise ValueError(f"duplicate probe id: {pid!r}")
        self._probes[pid] = cls
        return cls

    def get(self, probe_id: str) -> type[BaseProbe]:
        if probe_id not in self._probes:
            raise KeyError(
                f"unknown probe id: {probe_id!r} "
                f"(have {sorted(self._probes)})"
            )
        return self._probes[probe_id]

    def all_ids(self) -> list[str]:
        return sorted(self._probes)

    def all_items(self) -> Iterable[tuple[str, type[BaseProbe]]]:
        return sorted(self._probes.items())


_REGISTRY = ProbeRegistry()


def get_registry() -> ProbeRegistry:
    return _REGISTRY


def register_probe(cls: type[BaseProbe]) -> type[BaseProbe]:
    return _REGISTRY.register(cls)
