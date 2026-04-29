"""Follow-up scheduling for the lifeform.

Reads ``open_loop`` and ``commitment`` snapshots from the kernel and
produces ``FollowupItem`` records — purely advisory. Whether to actually
re-engage the user is a UX decision (CLI prompt, push notification, scheduled
job) made by the surrounding product code. The lifeform NEVER auto-emits
turns to the kernel on its own — that would violate the snapshot contract by
making the lifeform a second owner of conversation initiation.

The kernel side already produces ``commitment.honored_commitment_refs`` and
``open_loop.unresolved_loops`` etc.; this manager only looks at the public
snapshot fields, never inside the owners.
"""

from __future__ import annotations

from typing import Any

from lifeform_core.types import FollowupItem


class FollowupManager:
    """Maintains a pending followup queue derived from kernel snapshots."""

    def __init__(
        self,
        *,
        default_due_delay_ticks: int = 90,
        max_pending: int = 32,
    ) -> None:
        self._default_due_delay = default_due_delay_ticks
        self._max_pending = max_pending
        self._pending: list[FollowupItem] = []
        self._counter = 0
        self._seen_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Ingestion (snapshot-driven, never owner-driven)
    # ------------------------------------------------------------------

    def ingest_open_loops(
        self,
        *,
        unresolved_loops: tuple[Any, ...],
        current_tick: int,
    ) -> tuple[FollowupItem, ...]:
        """Convert ``open_loop.unresolved_loops`` into ``FollowupItem``s.

        Each entry is keyed by its identifier (we accept either dataclass
        instances with an ``id`` / ``loop_id`` attribute or string keys). New
        keys produce new pending items; already-tracked keys are skipped to
        avoid duplicates across turns.
        """
        new_items: list[FollowupItem] = []
        for entry in unresolved_loops:
            key = self._key_for(entry, prefix="loop")
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            self._counter += 1
            description = _entry_description(entry, fallback="open loop")
            item = FollowupItem(
                followup_id=f"fu-{self._counter:05d}",
                source="open_loop",
                description=description,
                due_at_tick=current_tick + self._default_due_delay,
                priority=0.5,
                metadata={"key": key},
            )
            self._pending.append(item)
            new_items.append(item)
        self._enforce_capacity()
        return tuple(new_items)

    def ingest_at_risk_commitments(
        self,
        *,
        at_risk_refs: tuple[Any, ...],
        current_tick: int,
        priority: float = 0.7,
    ) -> tuple[FollowupItem, ...]:
        new_items: list[FollowupItem] = []
        for entry in at_risk_refs:
            key = self._key_for(entry, prefix="commit")
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            self._counter += 1
            description = _entry_description(entry, fallback="at-risk commitment")
            item = FollowupItem(
                followup_id=f"fu-{self._counter:05d}",
                source="commitment",
                description=description,
                due_at_tick=current_tick + max(1, self._default_due_delay // 2),
                priority=priority,
                metadata={"key": key},
            )
            self._pending.append(item)
            new_items.append(item)
        self._enforce_capacity()
        return tuple(new_items)

    def ingest_commitment_lifecycle(
        self,
        *,
        lifecycle_entries: tuple[Any, ...],
        current_tick: int,
        defer_only_delay_multiplier: float = 2.5,
    ) -> tuple[FollowupItem, ...]:
        """Policy-aware ingestion of commitment lifecycle entries (Gap 7).

        Reads each entry's typed ``followup_policy`` and routes it to a
        follow-up item with cadence matching the policy:

        - ``GENTLE_CHECKIN`` (default): due after the standard half-delay
          used for at-risk commitments so the lifeform proactively
          re-engages.
        - ``DEFER_ONLY``: due after ``defer_only_delay_multiplier`` * the
          default delay so the lifeform does NOT badger the user about a
          commitment they explicitly pushed back against; priority is
          also lowered so the queue's capacity enforcer sheds these first
          under pressure.

        We intentionally do NOT fabricate a follow-up for every lifecycle
        entry \u2014 only records whose alignment state has reached one of
        the "needs user attention" markers (REJECT, MODIFY, or explicitly
        READY/PROPOSED without alignment). AGREE + completed commitments
        do not generate follow-ups from this path (they are handled by
        ``honored_commitment_refs`` cleanup upstream).

        Each entry is keyed by its ``record_id`` so repeated calls across
        turns dedupe against ``_seen_keys``; this keeps the lifeform from
        enqueueing the same "user said reject on X" follow-up once per
        turn.
        """
        new_items: list[FollowupItem] = []
        # Defer an import of the enum to avoid a hard import of the
        # kernel from this lifeform-side module; we only need value
        # comparisons, and defensive access to ``.value`` handles both
        # enum instances and their string values without hasattr abuse.
        base_delay = max(1, self._default_due_delay // 2)
        defer_delay = max(
            base_delay + 1,
            int(self._default_due_delay * defer_only_delay_multiplier),
        )
        for entry in lifecycle_entries:
            record_id = getattr(entry, "record_id", None)
            if not isinstance(record_id, str) or not record_id:
                continue
            alignment_value = _enum_value(
                getattr(entry, "alignment_state", None),
                default="unknown",
            )
            advocacy_value = _enum_value(
                getattr(entry, "advocacy_state", None),
                default="not_ready",
            )
            needs_surface = (
                alignment_value in {"reject", "modify"}
                or advocacy_value in {"ready", "proposed"}
            )
            if not needs_surface:
                continue
            policy_value = _enum_value(
                getattr(entry, "followup_policy", None),
                default="gentle_checkin",
            )
            key = f"commit-lifecycle::{record_id}::{policy_value}::{alignment_value}"
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            self._counter += 1
            if policy_value == "defer_only":
                due_delay = defer_delay
                priority = 0.25
                description = (
                    f"Deferred follow-up for commitment {record_id} "
                    f"(policy=defer_only, alignment={alignment_value})."
                )
            else:
                due_delay = base_delay
                priority = 0.65 if alignment_value == "reject" else 0.55
                description = (
                    f"Gentle check-in for commitment {record_id} "
                    f"(policy=gentle_checkin, alignment={alignment_value})."
                )
            item = FollowupItem(
                followup_id=f"fu-{self._counter:05d}",
                source="commitment-lifecycle",
                description=description,
                due_at_tick=current_tick + due_delay,
                priority=priority,
                metadata={
                    "key": key,
                    "record_id": record_id,
                    "policy": policy_value,
                    "alignment": alignment_value,
                    "advocacy": advocacy_value,
                },
            )
            self._pending.append(item)
            new_items.append(item)
        self._enforce_capacity()
        return tuple(new_items)

    def ingest_proactive_drive_pressure(
        self,
        *,
        total_pe: float,
        out_of_band_drive_names: tuple[str, ...],
        current_tick: int,
        priority: float = 0.55,
        followup_id: str | None = None,
    ) -> FollowupItem:
        """Surface a vitals-driven proactive followup.

        Produced by ``VitalsModule.consider_proactive_followup`` when the
        slow-scale PE crosses the configured threshold. The lifeform layer
        does NOT auto-execute the follow-up \u2014 it goes into the same pending
        queue as open-loop / commitment items, leaving "actually re-engage
        the user" as a UX decision in the surrounding product.

        Each call produces a fresh item; deduping uses the followup_id (or a
        time-bucketed default) so repeated crossings within the same
        cooldown produce one entry, not a flood. The cooldown is enforced
        upstream by ``VitalsModule``; this method just records what arrives.
        """
        self._counter += 1
        fid = followup_id or f"vitals-{current_tick}-{self._counter:05d}"
        descriptor = (
            ", ".join(out_of_band_drive_names)
            if out_of_band_drive_names
            else "drive deviation"
        )
        item = FollowupItem(
            followup_id=fid,
            source="vitals",
            description=f"Proactive check-in: {descriptor} pressure rising",
            due_at_tick=current_tick,
            priority=priority,
            metadata={
                "total_pe": f"{total_pe:.4f}",
                "drives_out_of_band": ",".join(out_of_band_drive_names),
            },
        )
        self._pending.append(item)
        self._enforce_capacity()
        return item

    def ingest_scene_close(
        self,
        *,
        scene_id: str,
        open_loops: tuple[str, ...],
        current_tick: int,
    ) -> FollowupItem | None:
        """Optional bookkeeping followup at scene end.

        If a scene closes with unresolved open loops, register one consolidated
        followup to nudge re-engagement next scene. Idempotent per scene.
        """
        if not open_loops:
            return None
        key = f"scene-end::{scene_id}"
        if key in self._seen_keys:
            return None
        self._seen_keys.add(key)
        self._counter += 1
        description = f"Scene {scene_id} closed with {len(open_loops)} unresolved open loop(s)."
        item = FollowupItem(
            followup_id=f"fu-{self._counter:05d}",
            source="scene-end",
            description=description,
            due_at_tick=current_tick + self._default_due_delay,
            priority=0.6,
            metadata={"scene_id": scene_id, "open_loop_count": str(len(open_loops))},
        )
        self._pending.append(item)
        self._enforce_capacity()
        return item

    # ------------------------------------------------------------------
    # Read / dequeue
    # ------------------------------------------------------------------

    @property
    def pending(self) -> tuple[FollowupItem, ...]:
        return tuple(self._pending)

    def due_now(self, *, current_tick: int) -> tuple[FollowupItem, ...]:
        return tuple(item for item in self._pending if item.due_at_tick <= current_tick)

    def acknowledge(self, followup_id: str) -> bool:
        """Drop a pending followup once the product layer has acted on it."""
        for index, item in enumerate(self._pending):
            if item.followup_id == followup_id:
                del self._pending[index]
                return True
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enforce_capacity(self) -> None:
        if len(self._pending) <= self._max_pending:
            return
        # Drop the oldest, lowest-priority items first.
        self._pending.sort(key=lambda item: (item.priority, -item.due_at_tick), reverse=True)
        self._pending = self._pending[: self._max_pending]

    @staticmethod
    def _key_for(entry: Any, *, prefix: str) -> str:
        if isinstance(entry, str):
            return f"{prefix}::{entry}"
        for attr in ("loop_id", "commitment_ref", "id", "ref"):
            value = getattr(entry, attr, None)
            if value:
                return f"{prefix}::{value}"
        # Fall back to a stable repr — guarantees idempotent deduping even
        # when the kernel's owner produces opaque tuple entries.
        return f"{prefix}::{repr(entry)}"


def _entry_description(entry: Any, *, fallback: str) -> str:
    if isinstance(entry, str):
        return entry
    for attr in ("description", "summary", "text", "loop_id", "commitment_ref"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value:
            return value
    return fallback


def _enum_value(maybe_enum: Any, *, default: str) -> str:
    """Extract the string value of an enum / string / None defensively.

    Ordering matters: ``str, Enum`` subclasses satisfy ``isinstance(x,
    str)`` BUT str-ify to their repr (``"FollowupPolicy.GENTLE_CHECKIN"``)
    rather than their value. So we must check the ``.value`` attribute
    first and only fall back to ``str`` for genuine plain strings.
    """
    if maybe_enum is None:
        return default
    value = getattr(maybe_enum, "value", None)
    if isinstance(value, str):
        return value
    if isinstance(maybe_enum, str):
        return maybe_enum
    return default
