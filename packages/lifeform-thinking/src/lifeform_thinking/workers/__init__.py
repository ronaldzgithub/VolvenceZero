"""Thinking workers \u2014 read-only async functions producing ThinkingArtifacts.

Each worker:

* receives the ``ThinkingTask`` it was scheduled for plus a frozen
  snapshot mapping restricted to the task's fingerprint scope
* returns a ``ThinkingArtifact`` with ``status = COMPLETED`` and a
  typed payload OR raises (the scheduler wraps the exception into
  a ``FAILED`` artifact)
* must NOT call any kernel-owner mutation API \u2014 contract tests in
  ``tests/contracts/test_thinking_worker_read_only.py`` grep this
  package for forbidden imports / attribute writes
"""

from lifeform_thinking.workers.mid_reflection import (
    MidReflectionPayload,
    mid_reflection_worker,
)

__all__ = [
    "MidReflectionPayload",
    "mid_reflection_worker",
]
