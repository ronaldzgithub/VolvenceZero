"""Scene lifecycle for the lifeform layer.

A *scene* is the lifeform unit at which the kernel's ``begin_new_context``
boundary fires (R6 — slow reflection). The kernel does not own scenes;
the lifeform decides:

* When a scene opens (first turn after init / after a previous scene closed)
* When a scene closes (idle threshold reached, explicit user farewell, or
  caller invokes ``end_scene()``)

Each scene close triggers:

1. ``runner.begin_new_context(reason=...)`` on the kernel runner — kernel
   enqueues the session-post slow loop job.
2. Optionally, ``await runner.drain_session_post_slow_loop()`` so the slow
   loop produces evidence before the next scene opens.

The SceneManager also captures a compact ``Scene`` record per scene so the
lifeform-evolution layer can reason about scene-level outcomes.
"""

from __future__ import annotations

from dataclasses import replace

from lifeform_core.types import Scene, SceneStatus, TickEvent, TickKind


class SceneManager:
    """Owns the open scene + history of closed scenes.

    Methods are sync; the actual kernel ``begin_new_context`` / drain are
    invoked from ``LifeformSession`` because they need access to the
    underlying ``AgentSessionRunner``.
    """

    def __init__(self, *, idle_close_after_system_ticks: int | None = 60) -> None:
        """Args:
            idle_close_after_system_ticks: if set, the SceneManager will mark
                an open scene as eligible for closing when this many SYSTEM
                ticks have passed since the last turn. ``None`` disables
                automatic close — caller must invoke ``end_scene()`` explicitly.
        """
        self._idle_close_after = idle_close_after_system_ticks
        self._open_scene: Scene | None = None
        self._closed: list[Scene] = []
        self._scene_counter = 0

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @property
    def open_scene(self) -> Scene | None:
        return self._open_scene

    @property
    def closed_scenes(self) -> tuple[Scene, ...]:
        return tuple(self._closed)

    @property
    def has_open_scene(self) -> bool:
        return self._open_scene is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open_scene_now(self, *, current_tick: int) -> Scene:
        if self._open_scene is not None:
            return self._open_scene
        self._scene_counter += 1
        scene = Scene(
            scene_id=f"scene-{self._scene_counter:05d}",
            started_at_tick=current_tick,
        )
        self._open_scene = scene
        return scene

    def record_turn(self, *, current_tick: int) -> Scene:
        """Increment the open scene's turn count.

        Auto-opens a scene if none exists. Returns the now-current scene.
        """
        if self._open_scene is None:
            self.open_scene_now(current_tick=current_tick)
        assert self._open_scene is not None
        new_scene = replace(
            self._open_scene,
            turn_count=self._open_scene.turn_count + 1,
            last_turn_at_tick=current_tick,
        )
        self._open_scene = new_scene
        return new_scene

    def close_open_scene(
        self,
        *,
        current_tick: int,
        open_loops: tuple[str, ...] = (),
        commitments: tuple[str, ...] = (),
    ) -> Scene | None:
        """Close the open scene and append it to history. Returns the closed
        scene record, or ``None`` if there was nothing open.
        """
        scene = self._open_scene
        if scene is None:
            return None
        closed = replace(
            scene,
            closed_at_tick=current_tick,
            open_loops_at_close=open_loops,
            commitments_at_close=commitments,
        )
        self._closed.append(closed)
        self._open_scene = None
        return closed

    # ------------------------------------------------------------------
    # Tick-driven idle detection
    # ------------------------------------------------------------------

    def on_tick(self, event: TickEvent) -> bool:
        """Return True iff the open scene should be closed for idle-timeout.

        Pure read: does not actually close. ``LifeformSession`` is the one
        that owns the close-and-drain action because it needs the kernel
        runner reference.
        """
        if event.kind is not TickKind.SYSTEM:
            return False
        if self._idle_close_after is None:
            return False
        scene = self._open_scene
        if scene is None or scene.last_turn_at_tick is None:
            return False
        ticks_idle = event.tick_index - scene.last_turn_at_tick
        return ticks_idle >= self._idle_close_after

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def status_summary(self) -> dict[str, int | str | None]:
        scene = self._open_scene
        return {
            "scene_count_total": self._scene_counter,
            "scene_count_closed": len(self._closed),
            "open_scene_id": scene.scene_id if scene else None,
            "open_scene_status": (scene.status.value if scene else SceneStatus.CLOSED.value),
            "open_scene_turn_count": scene.turn_count if scene else 0,
        }
