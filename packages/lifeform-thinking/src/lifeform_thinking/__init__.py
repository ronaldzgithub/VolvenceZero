"""Lifeform middle-frequency thinking scheduler (Gap 4 slice 2b).

Public API:

* ``ThinkingScheduler`` / ``ThinkingLoopSnapshot`` / ``ThinkingWiringLevel``
* ``FingerprintScope`` / ``compute_fingerprint`` / ``fingerprints_match``
* ``mid_reflection_worker`` + ``MidReflectionPayload``

Immutable envelopes (``ThinkingTask`` / ``ThinkingArtifact`` / status
enums) are re-exported from ``vz-contracts`` for convenience; they
are the cross-wheel contract surface.

See ``docs/specs/thinking-loop.md`` for the spec and
``docs/implementation/13_emogpt_prd_alignment_upgrade.md`` Gap 4 for
the phased rollout plan.
"""

from volvence_zero.thinking import (
    APPLIABLE_THINKING_TASK_STATUSES,
    TERMINAL_THINKING_TASK_STATUSES,
    ThinkingArtifact,
    ThinkingDepth,
    ThinkingPurpose,
    ThinkingTask,
    ThinkingTaskStatus,
)

from lifeform_thinking.adapter import (
    CONSUMER_SELF_TEMPORAL,
    CONSUMER_WORLD_TEMPORAL,
    MID_REFLECTION_SCOPE,
    ThinkingAdapter,
    ThinkingAdapterSnapshot,
    build_default_thinking_adapter,
)
from lifeform_thinking.fingerprint import (
    FingerprintScope,
    compute_fingerprint,
    fingerprints_match,
)
from lifeform_thinking.scheduler import (
    ThinkingLoopSnapshot,
    ThinkingScheduler,
    ThinkingWiringLevel,
    WorkerFunc,
)
from lifeform_thinking.workers import (
    MidReflectionPayload,
    mid_reflection_worker,
)

__all__ = [
    "APPLIABLE_THINKING_TASK_STATUSES",
    "CONSUMER_SELF_TEMPORAL",
    "CONSUMER_WORLD_TEMPORAL",
    "FingerprintScope",
    "MID_REFLECTION_SCOPE",
    "MidReflectionPayload",
    "TERMINAL_THINKING_TASK_STATUSES",
    "ThinkingAdapter",
    "ThinkingAdapterSnapshot",
    "ThinkingArtifact",
    "ThinkingDepth",
    "ThinkingLoopSnapshot",
    "ThinkingPurpose",
    "ThinkingScheduler",
    "ThinkingTask",
    "ThinkingTaskStatus",
    "ThinkingWiringLevel",
    "WorkerFunc",
    "build_default_thinking_adapter",
    "compute_fingerprint",
    "fingerprints_match",
    "mid_reflection_worker",
]
