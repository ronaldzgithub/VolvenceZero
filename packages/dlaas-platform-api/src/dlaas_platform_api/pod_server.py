"""Single-pod DLaaS server (one substrate / one GPU child process).

In multi-pod mode the :class:`PodProcessSupervisor` spawns one of these
per GPU. Each pod runs a full DLaaS app bound to ONE substrate (built
from a substrate profile) plus a thin ``acquire`` route the parent's
:class:`RemoteInstanceManager` calls to register an ai_id on the pod.

The pod's own launcher is a plain :class:`InstanceManager` (no
``forward_interaction``), so the pod dispatches interactions locally —
there is no forwarding loop.

Run standalone:

    python -m dlaas_platform_api.pod_server --port 9101 \
        --substrate-profile shared-frozen-persona-lora \
        --model-id Qwen/Qwen2.5-1.5B-Instruct --device cuda
"""

from __future__ import annotations

import argparse

from aiohttp import web

from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY

from dlaas_platform_api.substrate_profiles import (
    SubstrateProfile,
    build_runtime_for_profile,
)


async def _handle_pod_acquire(request: web.Request) -> web.Response:
    """Register an ai_id on this pod's InstanceManager (parent RPC)."""

    ai_id = request.match_info.get("ai_id", "")
    if not ai_id:
        return web.json_response(
            {"status": "error", "error": "invalid_ai_id"}, status=400
        )
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body
        payload = {}
    runtime_template_id = str(payload.get("runtime_template_id", "") or "")
    if not runtime_template_id:
        return web.json_response(
            {"status": "error", "error": "missing_runtime_template_id"},
            status=400,
        )
    launcher = request.app[INSTANCE_MANAGER_APP_KEY]
    try:
        await launcher.acquire(
            ai_id=ai_id,
            runtime_template_id=runtime_template_id,
            tenant_id=str(payload.get("tenant_id", "") or ""),
            scope_strategy=str(payload.get("scope_strategy", "") or ""),
        )
    except LookupError as exc:
        return web.json_response(
            {"status": "error", "error": "vertical_not_registered", "detail": str(exc)},
            status=503,
        )
    return web.json_response({"status": "ok", "ai_id": ai_id})


def build_pod_app(
    *,
    profile: SubstrateProfile,
    db_path: str = ":memory:",
) -> web.Application:
    """Build a single-substrate DLaaS app for one pod from ``profile``."""

    from dlaas_platform_api.app import build_dlaas_app

    runtime = build_runtime_for_profile(profile)
    app = build_dlaas_app(db_path=db_path, substrate_runtime=runtime)
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/acquire", _handle_pod_acquire
    )
    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dlaas_platform_api.pod_server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--substrate-profile", required=True)
    parser.add_argument("--mode", default="shared_frozen")
    parser.add_argument("--runtime-backend", default="transformers")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--db-path", default=":memory:")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - process entry
    args = _parse_args(argv)
    profile = SubstrateProfile(
        substrate_profile_id=args.substrate_profile,
        mode=args.mode,
        runtime_backend=args.runtime_backend,
        model_id=args.model_id,
        device=args.device,
    )
    app = build_pod_app(profile=profile, db_path=args.db_path)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - process entry
    main()


__all__ = ["build_pod_app", "main"]
