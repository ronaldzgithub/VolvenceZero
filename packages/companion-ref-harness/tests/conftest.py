# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Pytest configuration for the companion-ref-harness test suite.

Enables ``pytest-asyncio`` auto mode so every ``async def test_*``
runs without needing ``@pytest.mark.asyncio`` on each function.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every async test as ``asyncio`` automatically.

    We do this in code (not via ``[tool.pytest.ini_options]
    asyncio_mode = "auto"`` in pyproject) so the package-level config
    does not leak into the rest of the monorepo's test suite when
    these tests are collected from the repo root.
    """
    for item in items:
        if isinstance(item, pytest.Function):
            if _is_async_test(item):
                item.add_marker(pytest.mark.asyncio)


def _is_async_test(item: pytest.Function) -> bool:
    func = item.obj
    import inspect
    return inspect.iscoroutinefunction(func)
