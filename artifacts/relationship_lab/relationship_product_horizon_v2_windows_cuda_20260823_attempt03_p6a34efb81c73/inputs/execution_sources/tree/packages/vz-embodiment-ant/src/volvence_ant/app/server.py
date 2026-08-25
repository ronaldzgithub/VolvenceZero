"""aiohttp API and static host for the digital-ant realtime app."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from aiohttp import web

from volvence_ant.app.contracts import (
    APP_SCHEMA_VERSION,
    AppArm,
    AppCommand,
    AppCommandKind,
    AppDisturbance,
    AppDisturbanceKind,
    AppExperimentConfig,
    AppMode,
    AppObjective,
    WorldObjectKind,
)
from volvence_ant.app.runner import AntAppManager
from volvence_ant.evidence.ecology_checkpoint import (
    load_promoted_ecology_checkpoint,
)


_MANAGER_KEY: web.AppKey[AntAppManager] = web.AppKey("digital-ant-manager", AntAppManager)
_WEB_ROOT_KEY: web.AppKey[Path | None] = web.AppKey("digital-ant-web-root", Path | None)


def _reject_unknown_fields(payload: dict[str, object], *, allowed: frozenset[str]) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError("unsupported request fields: " + ", ".join(sorted(unknown)))


def _config_from_json(payload: dict[str, object]) -> AppExperimentConfig:
    _reject_unknown_fields(
        payload,
        allowed=frozenset(
            {
                "mode",
                "arm",
                "objective",
                "seed",
                "n_ants",
                "temporal_latent_dim",
                "tick_interval_ms",
                "max_ticks",
                "autostart",
                "food_x",
                "food_y",
                "motor_turn_gain",
                "motor_turn_bias",
                "motor_switch_tick",
                "motor_switched_turn_gain",
                "motor_switched_turn_bias",
            }
        ),
    )
    return AppExperimentConfig(
        mode=AppMode(str(payload.get("mode", AppMode.SOLO.value))),
        arm=AppArm(str(payload.get("arm", AppArm.LEARNED.value))),
        objective=AppObjective(str(payload.get("objective", AppObjective.FORAGING.value))),
        seed=int(payload.get("seed", 0)),
        n_ants=int(payload.get("n_ants", 1)),
        temporal_latent_dim=int(payload.get("temporal_latent_dim", 16)),
        tick_interval_ms=int(payload.get("tick_interval_ms", 150)),
        max_ticks=(None if payload.get("max_ticks", 1000) is None else int(payload.get("max_ticks", 1000))),
        autostart=bool(payload.get("autostart", True)),
        food_x=float(payload.get("food_x", 6.0)),
        food_y=float(payload.get("food_y", 0.0)),
        motor_turn_gain=float(payload.get("motor_turn_gain", 1.0)),
        motor_turn_bias=float(payload.get("motor_turn_bias", 0.0)),
        motor_switch_tick=(None if payload.get("motor_switch_tick") is None else int(payload["motor_switch_tick"])),
        motor_switched_turn_gain=float(payload.get("motor_switched_turn_gain", 1.0)),
        motor_switched_turn_bias=float(payload.get("motor_switched_turn_bias", 0.0)),
    )


def _command_from_json(payload: dict[str, object]) -> AppCommand:
    _reject_unknown_fields(payload, allowed=frozenset({"command_id", "kind", "value"}))
    raw_value = payload.get("value")
    return AppCommand(
        command_id=str(payload.get("command_id") or uuid4().hex),
        kind=AppCommandKind(str(payload["kind"])),
        value=None if raw_value is None else float(raw_value),
    )


def _disturbance_from_json(payload: dict[str, object]) -> AppDisturbance:
    _reject_unknown_fields(
        payload,
        allowed=frozenset(
            {
                "event_id",
                "kind",
                "requested_tick",
                "body_id",
                "food_index",
                "x",
                "y",
                "magnitude",
                "turn_gain",
                "turn_bias",
                "object_id",
                "object_kind",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "radius",
                "strength",
                "decay",
                "remaining",
                "angle",
                "length",
                "harm_threshold",
                "delta_x",
                "delta_y",
            }
        ),
    )

    def optional_int(key: str) -> int | None:
        value = payload.get(key)
        return None if value is None else int(value)

    def optional_float(key: str) -> float | None:
        value = payload.get(key)
        return None if value is None else float(value)

    return AppDisturbance(
        event_id=str(payload.get("event_id") or uuid4().hex),
        kind=AppDisturbanceKind(str(payload["kind"])),
        requested_tick=optional_int("requested_tick"),
        body_id=optional_int("body_id"),
        food_index=int(payload.get("food_index", 0)),
        x=optional_float("x"),
        y=optional_float("y"),
        magnitude=optional_float("magnitude"),
        turn_gain=optional_float("turn_gain"),
        turn_bias=optional_float("turn_bias"),
        object_id=(None if payload.get("object_id") is None else str(payload["object_id"])),
        object_kind=(None if payload.get("object_kind") is None else WorldObjectKind(str(payload["object_kind"]))),
        start_x=optional_float("start_x"),
        start_y=optional_float("start_y"),
        end_x=optional_float("end_x"),
        end_y=optional_float("end_y"),
        radius=optional_float("radius"),
        strength=optional_float("strength"),
        decay=optional_float("decay"),
        remaining=optional_float("remaining"),
        angle=optional_float("angle"),
        length=optional_float("length"),
        harm_threshold=optional_float("harm_threshold"),
        delta_x=optional_float("delta_x"),
        delta_y=optional_float("delta_y"),
    )


async def _json_body(request: web.Request) -> dict[str, object]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="request body must be a JSON object")
    return payload


def _manager(request: web.Request) -> AntAppManager:
    return request.app[_MANAGER_KEY]


def _run(request: web.Request):
    run_id = request.match_info["run_id"]
    try:
        return _manager(request).get_run(run_id)
    except KeyError as exc:
        raise web.HTTPNotFound(text=str(exc)) from exc


async def create_run(request: web.Request) -> web.Response:
    payload = await _json_body(request)
    try:
        config = _config_from_json(payload)
        run = await _manager(request).create_run(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    return web.json_response({"run_id": run.run_id, "status": asdict(run.status())}, status=201)


async def get_status(request: web.Request) -> web.Response:
    return web.json_response(asdict(_run(request).status()))


async def apply_command(request: web.Request) -> web.Response:
    payload = await _json_body(request)
    try:
        status = await _run(request).apply_command(_command_from_json(payload))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    return web.json_response(asdict(status))


async def queue_disturbance(request: web.Request) -> web.Response:
    payload = await _json_body(request)
    try:
        record = await _run(request).queue_disturbance(_disturbance_from_json(payload))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    return web.json_response(asdict(record), status=202)


async def get_replay(request: web.Request) -> web.Response:
    return web.json_response(_run(request).replay_payload())


async def health(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "service": "digital-ant-app",
            "schema_version": APP_SCHEMA_VERSION,
        }
    )


async def stream_events(request: web.Request) -> web.StreamResponse:
    run = _run(request)
    try:
        last_sequence = max(
            int(request.query.get("after", "0")),
            int(request.headers.get("Last-Event-ID", "0")),
        )
    except ValueError as exc:
        raise web.HTTPBadRequest(text="after must be an integer") from exc
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)
    try:
        while True:
            events = await run.wait_for_events(last_sequence)
            if not events:
                if run.terminal:
                    break
                await response.write(b": keepalive\n\n")
                continue
            for event in events:
                body = (f"id: {event.sequence}\nevent: {event.kind.value}\ndata: {event.payload_json}\n\n").encode(
                    "utf-8"
                )
                await response.write(body)
                last_sequence = event.sequence
            if run.terminal and last_sequence >= run.latest_sequence:
                break
    except (ConnectionResetError, BrokenPipeError):
        return response
    return response


async def static_app(request: web.Request) -> web.StreamResponse:
    web_root = request.app[_WEB_ROOT_KEY]
    if web_root is None:
        return web.json_response(
            {
                "service": "digital-ant-app",
                "message": "frontend build not found; run the Vite dev server",
            }
        )
    relative = request.match_info.get("path", "")
    candidate = (web_root / relative).resolve()
    if web_root.resolve() not in candidate.parents and candidate != web_root.resolve():
        raise web.HTTPForbidden(text="invalid static path")
    if candidate.is_file():
        return web.FileResponse(candidate)
    return web.FileResponse(web_root / "index.html")


async def _manager_context(app: web.Application) -> AsyncIterator[None]:
    yield
    await app[_MANAGER_KEY].close()


def default_web_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[3] / "web" / "dist"
    return candidate if (candidate / "index.html").is_file() else None


def create_app(
    *,
    manager: AntAppManager | None = None,
    web_root: Path | None = None,
) -> web.Application:
    app = web.Application()
    app[_MANAGER_KEY] = manager or AntAppManager()
    app[_WEB_ROOT_KEY] = web_root if web_root is not None else default_web_root()
    app.cleanup_ctx.append(_manager_context)
    app.router.add_get("/api/v1/health", health)
    app.router.add_post("/api/v1/runs", create_run)
    app.router.add_get("/api/v1/runs/{run_id}/status", get_status)
    app.router.add_get("/api/v1/runs/{run_id}/events", stream_events)
    app.router.add_post("/api/v1/runs/{run_id}/commands", apply_command)
    app.router.add_post("/api/v1/runs/{run_id}/disturbances", queue_disturbance)
    app.router.add_get("/api/v1/runs/{run_id}/replay", get_replay)
    app.router.add_get("/{path:.*}", static_app)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Digital-ant realtime experiment app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--web-root", type=Path)
    parser.add_argument(
        "--evidence-artifact",
        type=Path,
        help="read-only formal evidence artifact used only for PASS/BLOCK display",
    )
    parser.add_argument(
        "--ecology-checkpoint-report",
        type=Path,
        help=(
            "local PASS ecology checkpoint report; its canonical JSON archive "
            "is integrity-checked and restored only through owner APIs"
        ),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[5]
    ecology_checkpoint = (
        load_promoted_ecology_checkpoint(
            report_path=args.ecology_checkpoint_report.resolve(),
            repo_root=repo_root,
        )
        if args.ecology_checkpoint_report is not None
        else None
    )
    manager = (
        AntAppManager.from_evidence_artifact(
            str(args.evidence_artifact),
            ecology_checkpoint=ecology_checkpoint,
        )
        if args.evidence_artifact is not None
        else AntAppManager(ecology_checkpoint=ecology_checkpoint)
    )
    web.run_app(
        create_app(manager=manager, web_root=args.web_root),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()


__all__ = ["create_app", "default_web_root", "main"]
