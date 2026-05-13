"""BenchmarkMetricDescriptor schema-driven extraction contract test (B3).

实现 [`docs/specs/evaluation-cascade.md`](../../docs/specs/evaluation-cascade.md)
§B3 ``metric_means`` schema-driven 接入:

- ``RuntimeModule.declare_benchmark_metrics()`` 是 classmethod，默认返回 ``()``
- ``BenchmarkMetricDescriptor`` schema 字段稳定
- key 全局唯一性约束（用所有 RuntimeModule 子类的 declare 集合作 union 时
  无重复）
- 添加 declaration 不改 benchmark 代码即可让新 readout 流入对照表
  （契约面 — 实际 benchmark 接入是 follow-up）
"""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.evaluation.backbone import EvaluationModule
from volvence_zero.runtime.kernel import (
    BenchmarkMetricDescriptor,
    RuntimeModule,
)


def test_benchmark_metric_descriptor_fields_frozen() -> None:
    expected = ("key", "extractor_path", "description", "declared_by_owner")
    actual = tuple(f.name for f in dataclasses.fields(BenchmarkMetricDescriptor))
    assert actual == expected


def test_benchmark_metric_descriptor_is_frozen() -> None:
    """frozen dataclass: declarations must be tamper-proof."""
    d = BenchmarkMetricDescriptor(
        key="x",
        extractor_path="snapshot.value.x",
        description="...",
        declared_by_owner="ModuleX",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.key = "y"  # type: ignore[misc]


def test_runtime_module_declare_benchmark_metrics_default_empty() -> None:
    """B3 backward compat: every existing RuntimeModule subclass returns
    an empty tuple by default — current metric_means behaviour unchanged."""

    class _NoDeclareModule(RuntimeModule):
        slot_name = "dummy_no_metric"
        owner = "_NoDeclareModule"
        value_type = dict

        async def process(self, upstream):  # pragma: no cover
            raise NotImplementedError

    assert _NoDeclareModule.declare_benchmark_metrics() == ()


def test_evaluation_module_declare_benchmark_metrics_returns_tuple() -> None:
    """Sanity: real production module returns tuple type for schema."""
    metrics = EvaluationModule.declare_benchmark_metrics()
    assert isinstance(metrics, tuple)


def test_owner_subclass_can_override_declare_benchmark_metrics() -> None:
    """B3: subclasses can declare metrics; they appear in the union."""

    class _ModuleWithMetric(RuntimeModule):
        slot_name = "dummy_with_metric"
        owner = "_ModuleWithMetric"
        value_type = dict

        async def process(self, upstream):  # pragma: no cover
            raise NotImplementedError

        @classmethod
        def declare_benchmark_metrics(cls) -> tuple[BenchmarkMetricDescriptor, ...]:
            return (
                BenchmarkMetricDescriptor(
                    key="dummy_metric_a",
                    extractor_path="value.field_a",
                    description="test metric A",
                    declared_by_owner=cls.owner,
                ),
            )

    decls = _ModuleWithMetric.declare_benchmark_metrics()
    assert len(decls) == 1
    assert decls[0].key == "dummy_metric_a"
    assert decls[0].declared_by_owner == "_ModuleWithMetric"


def test_global_key_uniqueness_across_real_modules() -> None:
    """Among all RuntimeModule subclasses present in the kernel package,
    declared metric keys must not collide.

    The check uses module-class discovery via subclass walk. Subclasses
    declared in tests / lifeform-* layers are intentionally not enumerated
    here; this contract is the kernel SSOT only.
    """
    # Walk all known RuntimeModule subclasses recursively.
    seen: set[type] = set()
    to_visit: list[type] = list(RuntimeModule.__subclasses__())
    while to_visit:
        cls = to_visit.pop()
        if cls in seen:
            continue
        seen.add(cls)
        to_visit.extend(cls.__subclasses__())

    keys_to_owners: dict[str, list[str]] = {}
    for cls in seen:
        try:
            decls = cls.declare_benchmark_metrics()
        except Exception:  # noqa: BLE001  — schema problem
            # Fail-loudly on misbehaving declarations:
            raise
        for d in decls:
            keys_to_owners.setdefault(d.key, []).append(d.declared_by_owner)

    duplicates = {k: owners for k, owners in keys_to_owners.items() if len(owners) > 1}
    assert not duplicates, (
        "BenchmarkMetricDescriptor.key collision across owners:\n"
        + "\n".join(
            f"  - {k!r}: declared by {owners}" for k, owners in duplicates.items()
        )
    )
