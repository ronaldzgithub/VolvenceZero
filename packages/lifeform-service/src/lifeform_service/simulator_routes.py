"""HTTP routes that drive a ``companion_bench`` user simulator.

Three endpoints, all scoped to a live ``lifeform-service`` session:

* ``GET /v1/scenarios`` — list public companion-bench scenarios so the
  chat UI can populate its scenario dropdown.
* ``POST /v1/sessions/{sid}/simulator/init`` — bind a scenario to the
  session and emit the deterministic per-session-turn schedule.
* ``POST /v1/sessions/{sid}/simulator/next-user-turn`` — advance the
  simulator one tick, returning the next synthetic user utterance plus
  the matching FSM step (when one fires).

Why this lives in ``lifeform-service`` (not ``companion-bench``)
================================================================

``companion-bench`` is the Apache 2.0 vendor-neutral benchmark wheel.
By contract it MUST NOT import any internal wheel
(see ``tests/contracts/test_companion_bench_no_internal_imports.py``).
The simulator-driving glue we need here references both
``companion_bench.spec`` / ``companion_bench.user_simulator`` (OK,
Apache 2.0 public API) *and* the proprietary ``lifeform_service`` /
``volvence_zero`` runtime — so it can only live on this side of the
license / SSOT boundary.

The state owned here (per-session :class:`_SimulatorState`) is the
single source of truth for "where is the synthetic user in the scripted
arc"; UI / CLI consume it exclusively through the three routes. No
other module reaches into ``_SIM_CACHE`` directly.
"""

from __future__ import annotations

import importlib.resources as res
import json
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from companion_bench.spec import ScenarioSpec, load_scenarios_dir
from companion_bench.user_simulator import (
    DeterministicFakeUtteranceClient,
    GeneratedUserTurn,
    TurnContext,
    UserSimulator,
    UtteranceClient,
)

from lifeform_service.session_manager import (
    SessionManager,
    SessionNotFoundError,
)


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------


def _public_scenarios_dir() -> pathlib.Path:
    """Locate the public scenario YAMLs shipped with ``companion-bench``.

    Mirrors :func:`companion_bench.cli._public_dir` so we don't pull
    that private CLI helper across the wheel boundary.
    """

    return pathlib.Path(
        str(res.files("companion_bench") / "scenarios" / "public")
    )


def _load_public_scenarios() -> tuple[ScenarioSpec, ...]:
    return load_scenarios_dir(_public_scenarios_dir(), include_held_out=False)


def _scenario_to_listing(spec: ScenarioSpec) -> dict[str, Any]:
    return {
        "scenario_id": spec.scenario_id,
        "family": spec.family.value,
        "language": spec.language,
        "arc_length_sessions": spec.arc_length_sessions,
        "paraphrase_seed_count": spec.paraphrase_seed_count,
        "session_turn_range": list(spec.session_turn_range),
        "inter_session_gap_days": list(spec.inter_session_gap_days),
        "persona": spec.user_simulator.persona,
        "goal_count": len(spec.user_simulator.goals),
        "fsm_step_count": len(spec.user_simulator.fsm),
    }


# ---------------------------------------------------------------------------
# Per-session simulator state
# ---------------------------------------------------------------------------


class ArcExhaustedError(Exception):
    """Raised when ``next_user_turn`` is called after the arc completes."""


class SimulatorBackendError(Exception):
    """Raised when the LLM utterance backend fails (HTTP / parse / etc)."""


@dataclass(frozen=True)
class _ScheduleEntry:
    """One row of the deterministic per-arc schedule.

    ``gap_days`` is the gap **before** this turn fires (0 within a
    session; the configured inter-session gap on each session's first
    turn).
    """

    session_index: int
    turn_index: int
    gap_days: int


@dataclass
class _SimulatorState:
    """Per-session record. Mutable cursor; everything else is frozen."""

    scenario_id: str
    paraphrase_seed: int
    spec: ScenarioSpec
    simulator: UserSimulator
    schedule: tuple[_ScheduleEntry, ...]
    cursor: int = 0
    # Set when the cache transfers state across an arc-session boundary
    # (see ``resume_from_session_id``). Surfaced in ``/init`` responses
    # so the UI can confirm the resume actually landed.
    resumed_from_session_id: str | None = None
    # Defensive guard: only one synthetic user emit per arc-session can
    # be "pending" an assistant reply. The cache uses this to refuse a
    # double-emit if the UI forgets to wait for /turns to land.
    pending_assistant_for_cursor: int | None = field(default=None)


def _build_schedule(spec: ScenarioSpec, paraphrase_seed: int) -> tuple[
    _ScheduleEntry, ...
]:
    """Deterministically expand a :class:`ScenarioSpec` into a flat schedule.

    ``session_turn_range`` is a ``[min, max]`` inclusive pair; we sample
    one count per session using a PRNG seeded by
    ``(scenario_id, paraphrase_seed)``. This keeps every (scenario, seed)
    pair byte-identical across runs even though the spec itself leaves
    the per-session length open.
    """

    rng = random.Random(f"{spec.scenario_id}|{paraphrase_seed}|schedule")
    lo, hi = spec.session_turn_range
    rows: list[_ScheduleEntry] = []
    for s_idx in range(1, spec.arc_length_sessions + 1):
        turn_count = rng.randint(lo, hi)
        gap_days = (
            0 if s_idx == 1 else spec.inter_session_gap_days[s_idx - 2]
        )
        for t_idx in range(1, turn_count + 1):
            rows.append(
                _ScheduleEntry(
                    session_index=s_idx,
                    turn_index=t_idx,
                    gap_days=gap_days if t_idx == 1 else 0,
                )
            )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class _SimulatorCache:
    """In-memory owner of every active :class:`_SimulatorState`.

    Keyed by ``session_id`` (the HTTP session the simulator is bound
    to). The cache is the single source of truth for simulator state;
    every consumer goes through the routes.
    """

    def __init__(
        self,
        *,
        utterance_backend: UtteranceClient,
        scenarios: tuple[ScenarioSpec, ...],
    ) -> None:
        self._backend = utterance_backend
        self._scenarios: dict[str, ScenarioSpec] = {
            spec.scenario_id: spec for spec in scenarios
        }
        self._states: dict[str, _SimulatorState] = {}

    @property
    def utterance_backend(self) -> UtteranceClient:
        return self._backend

    def list_scenarios(self) -> tuple[ScenarioSpec, ...]:
        return tuple(self._scenarios.values())

    def get_scenario(self, scenario_id: str) -> ScenarioSpec | None:
        return self._scenarios.get(scenario_id)

    def get(self, session_id: str) -> _SimulatorState | None:
        return self._states.get(session_id)

    def evict(self, session_id: str) -> None:
        self._states.pop(session_id, None)

    def init(
        self,
        *,
        session_id: str,
        scenario_id: str,
        paraphrase_seed: int,
        resume_from_session_id: str | None,
        recent_assistant_text: str,
    ) -> _SimulatorState:
        spec = self._scenarios.get(scenario_id)
        if spec is None:
            raise KeyError(scenario_id)
        if paraphrase_seed < 0 or paraphrase_seed >= spec.paraphrase_seed_count:
            raise ValueError(
                f"paraphrase_seed must be in "
                f"[0, {spec.paraphrase_seed_count}); got {paraphrase_seed}"
            )
        if resume_from_session_id is not None:
            prior = self._states.get(resume_from_session_id)
            if prior is None:
                raise KeyError(resume_from_session_id)
            if (
                prior.scenario_id != scenario_id
                or prior.paraphrase_seed != paraphrase_seed
            ):
                raise ValueError(
                    "resume_from_session_id binds a different "
                    "(scenario_id, paraphrase_seed) pair; refusing to "
                    "transfer state across mismatched arcs"
                )
            # Drain any outstanding assistant reply before transferring.
            if recent_assistant_text.strip():
                prior.simulator.append_assistant(recent_assistant_text.strip())
                prior.pending_assistant_for_cursor = None
            transferred = _SimulatorState(
                scenario_id=prior.scenario_id,
                paraphrase_seed=prior.paraphrase_seed,
                spec=prior.spec,
                simulator=prior.simulator,
                schedule=prior.schedule,
                cursor=prior.cursor,
                resumed_from_session_id=resume_from_session_id,
                pending_assistant_for_cursor=prior.pending_assistant_for_cursor,
            )
            self._states[session_id] = transferred
            # Drop the old binding so the simulator can never be in two
            # places at once (SSOT).
            del self._states[resume_from_session_id]
            return transferred

        simulator = UserSimulator(
            spec=spec,
            paraphrase_seed=paraphrase_seed,
            backend=self._backend,
        )
        state = _SimulatorState(
            scenario_id=scenario_id,
            paraphrase_seed=paraphrase_seed,
            spec=spec,
            simulator=simulator,
            schedule=_build_schedule(spec, paraphrase_seed),
            cursor=0,
            resumed_from_session_id=None,
            pending_assistant_for_cursor=None,
        )
        self._states[session_id] = state
        return state

    def next_user_turn(
        self,
        *,
        session_id: str,
        recent_assistant_text: str,
    ) -> tuple[GeneratedUserTurn, _ScheduleEntry, str, int]:
        """Advance the simulator one tick.

        Returns ``(generated, entry, arc_position, next_gap_days)``
        where ``arc_position`` is one of ``"in_session"`` /
        ``"session_end"`` / ``"arc_end"`` and ``next_gap_days`` is the
        gap before the next session (``0`` when ``arc_position`` is
        ``"in_session"`` or ``"arc_end"``).
        """

        state = self._states.get(session_id)
        if state is None:
            raise KeyError(session_id)
        if state.cursor >= len(state.schedule):
            raise ArcExhaustedError(
                "simulator arc already exhausted; no further turns"
            )
        if recent_assistant_text.strip():
            state.simulator.append_assistant(recent_assistant_text.strip())
            state.pending_assistant_for_cursor = None
        entry = state.schedule[state.cursor]
        try:
            generated = state.simulator.next_turn(
                TurnContext(
                    session_index=entry.session_index,
                    turn_index=entry.turn_index,
                    inter_session_gap_days=entry.gap_days,
                )
            )
        except Exception as exc:
            # The UserSimulator's backend.complete() may raise any
            # error type (urllib HTTPError, RuntimeError from our
            # OpenAiUtteranceClient, network timeouts...); we narrow
            # to a typed wrapper so the route can map to 502 without
            # masking arc-exhausted ArcExhaustedError above.
            raise SimulatorBackendError(str(exc)) from exc
        state.pending_assistant_for_cursor = state.cursor
        state.cursor += 1
        if state.cursor >= len(state.schedule):
            arc_position = "arc_end"
            next_gap_days = 0
        else:
            next_entry = state.schedule[state.cursor]
            if next_entry.session_index != entry.session_index:
                arc_position = "session_end"
                next_gap_days = next_entry.gap_days
            else:
                arc_position = "in_session"
                next_gap_days = 0
        return generated, entry, arc_position, next_gap_days


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


_APP_KEY = "simulator_cache"


def install_simulator_cache(
    app: web.Application,
    *,
    utterance_backend: UtteranceClient | None,
) -> _SimulatorCache:
    """Build a cache and attach it to ``app`` under a stable key.

    When ``utterance_backend`` is ``None`` we install the deterministic
    fake from ``companion_bench`` so the surface is always available;
    operators set ``PROTOCOL_LLM_API_KEY`` to switch on a real backend.
    """

    backend: UtteranceClient = (
        utterance_backend
        if utterance_backend is not None
        else DeterministicFakeUtteranceClient()
    )
    scenarios = _load_public_scenarios()
    cache = _SimulatorCache(
        utterance_backend=backend,
        scenarios=scenarios,
    )
    app[_APP_KEY] = cache
    return cache


def get_simulator_cache(app: web.Application) -> _SimulatorCache:
    cache = app.get(_APP_KEY)
    if cache is None:
        raise RuntimeError(
            "simulator_cache not installed; "
            "call install_simulator_cache(app, ...) at startup"
        )
    return cache


def evict_simulator_cache_entry(app: web.Application, session_id: str) -> None:
    """Drop a session's simulator state (called on session close)."""

    cache = app.get(_APP_KEY)
    if cache is None:
        return
    cache.evict(session_id)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


def _json_error(
    *, status: int, error: str, detail: str = ""
) -> web.Response:
    body: dict[str, Any] = {"error": error}
    if detail:
        body["detail"] = detail
    return web.json_response(body, status=status)


async def _require_json(request: web.Request) -> dict[str, Any]:
    raw = await request.text()
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise web.HTTPBadRequest(reason=f"invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(reason="invalid_json: body must be object")
    return payload


async def handle_list_scenarios(request: web.Request) -> web.Response:
    """``GET /v1/scenarios`` — list public companion-bench scenarios.

    Query params:

    * ``family`` — optional ``F1``..``F6`` filter
    * ``language`` — optional ``en`` / ``zh`` / ``bilingual`` filter
    """

    cache = get_simulator_cache(request.app)
    family = request.query.get("family", "").strip().upper()
    language = request.query.get("language", "").strip().lower()
    rows: list[dict[str, Any]] = []
    for spec in cache.list_scenarios():
        if family and spec.family.value != family:
            continue
        if language and spec.language != language:
            continue
        rows.append(_scenario_to_listing(spec))
    rows.sort(key=lambda r: r["scenario_id"])
    return web.json_response({"scenarios": rows})


async def handle_simulator_init(request: web.Request) -> web.Response:
    """``POST /v1/sessions/{sid}/simulator/init``."""

    session_id = request.match_info["session_id"]
    cache = get_simulator_cache(request.app)
    manager: SessionManager = request.app["session_manager"]
    # Verify the binding session exists (loud failure if the UI mis-typed).
    try:
        await manager.get_session(session_id)
    except SessionNotFoundError as exc:
        return _json_error(
            status=404, error="session_not_found", detail=str(exc)
        )
    payload = await _require_json(request)
    scenario_id = payload.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        return _json_error(
            status=400,
            error="invalid_scenario_id",
            detail="scenario_id is required and must be a non-empty string",
        )
    paraphrase_seed_raw = payload.get("paraphrase_seed", 0)
    if not isinstance(paraphrase_seed_raw, int) or isinstance(
        paraphrase_seed_raw, bool
    ):
        return _json_error(
            status=400,
            error="invalid_paraphrase_seed",
            detail="paraphrase_seed must be an integer",
        )
    resume_from = payload.get("resume_from_session_id")
    if resume_from is not None and not isinstance(resume_from, str):
        return _json_error(
            status=400,
            error="invalid_resume_from_session_id",
            detail="resume_from_session_id must be a string when provided",
        )
    recent_assistant_text = payload.get("recent_assistant_text", "")
    if not isinstance(recent_assistant_text, str):
        return _json_error(
            status=400,
            error="invalid_recent_assistant_text",
            detail="recent_assistant_text must be a string when provided",
        )
    try:
        state = cache.init(
            session_id=session_id,
            scenario_id=scenario_id.strip(),
            paraphrase_seed=int(paraphrase_seed_raw),
            resume_from_session_id=resume_from.strip() if resume_from else None,
            recent_assistant_text=recent_assistant_text,
        )
    except KeyError as exc:
        return _json_error(
            status=404,
            error="unknown_scenario_or_session",
            detail=str(exc),
        )
    except ValueError as exc:
        return _json_error(
            status=400,
            error="invalid_simulator_init",
            detail=str(exc),
        )

    schedule_payload: list[dict[str, Any]] = []
    # Collapse the flat schedule into per-session summaries for the UI.
    grouped: dict[int, list[_ScheduleEntry]] = {}
    for entry in state.schedule:
        grouped.setdefault(entry.session_index, []).append(entry)
    for s_idx in sorted(grouped):
        rows = grouped[s_idx]
        schedule_payload.append(
            {
                "session_index": s_idx,
                "turn_count": len(rows),
                "gap_days": rows[0].gap_days,
            }
        )

    return web.json_response(
        {
            "session_id": session_id,
            "scenario_id": state.scenario_id,
            "paraphrase_seed": state.paraphrase_seed,
            "arc_length_sessions": state.spec.arc_length_sessions,
            "schedule": schedule_payload,
            "identity": {
                "name": state.simulator.identity.name,
                "occupation": state.simulator.identity.occupation,
                "contextual_detail": state.simulator.identity.contextual_detail,
            },
            "resumed_from_session_id": state.resumed_from_session_id,
            "cursor": state.cursor,
            "language": state.spec.language,
            "persona": state.spec.user_simulator.persona,
        }
    )


async def handle_simulator_next_user_turn(
    request: web.Request,
) -> web.Response:
    """``POST /v1/sessions/{sid}/simulator/next-user-turn``."""

    session_id = request.match_info["session_id"]
    cache = get_simulator_cache(request.app)
    payload = await _require_json(request)
    recent_assistant_text = payload.get("recent_assistant_text", "")
    if not isinstance(recent_assistant_text, str):
        return _json_error(
            status=400,
            error="invalid_recent_assistant_text",
            detail="recent_assistant_text must be a string when provided",
        )
    try:
        generated, entry, arc_position, next_gap_days = cache.next_user_turn(
            session_id=session_id,
            recent_assistant_text=recent_assistant_text,
        )
    except KeyError:
        return _json_error(
            status=404,
            error="simulator_not_bound",
            detail=(
                "no simulator is bound to this session; "
                "call POST /v1/sessions/{sid}/simulator/init first"
            ),
        )
    except ArcExhaustedError as exc:
        return _json_error(
            status=409,
            error="arc_exhausted",
            detail=str(exc),
        )
    except SimulatorBackendError as exc:
        return _json_error(
            status=502,
            error="simulator_backend_failed",
            detail=str(exc),
        )

    fsm_step_payload: dict[str, Any] | None = None
    if generated.fsm_step is not None:
        fsm_step_payload = {
            "action": generated.fsm_step.action,
            "payload": generated.fsm_step.payload,
            "session": generated.fsm_step.session,
            "turn": generated.fsm_step.turn,
        }
    return web.json_response(
        {
            "session_id": session_id,
            "user_text": generated.text,
            "fsm_step": fsm_step_payload,
            "session_index": entry.session_index,
            "turn_index": entry.turn_index,
            "gap_days": entry.gap_days,
            "arc_position": arc_position,
            "next_gap_days": next_gap_days,
            "paraphrase_seed": generated.paraphrase_seed,
            "rng_draw_count": generated.rng_draw_count,
            "identity_name": generated.identity.name,
        }
    )


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_simulator_routes(app: web.Application) -> None:
    """Wire the three simulator endpoints onto ``app.router``."""

    app.router.add_get("/v1/scenarios", handle_list_scenarios)
    app.router.add_post(
        "/v1/sessions/{session_id}/simulator/init",
        handle_simulator_init,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/simulator/next-user-turn",
        handle_simulator_next_user_turn,
    )


__all__ = [
    "ArcExhaustedError",
    "SimulatorBackendError",
    "evict_simulator_cache_entry",
    "get_simulator_cache",
    "install_simulator_cache",
    "register_simulator_routes",
]
