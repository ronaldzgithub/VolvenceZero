"""HTTP proxy to a remote DLaaS runtime pod (multi-process / multi-GPU).

In multi-pod mode each ai_id lives in a child pod process that runs its
own substrate + :class:`InstanceManager` and exposes the DLaaS HTTP
surface. :class:`RemoteInstanceManager` is the parent-side proxy the
:class:`MultiPodLauncher` holds per pod: it forwards adopt/wake/sleep
and the interaction envelope to the pod over HTTP.

Transport is injected (``transport``) so the routing/forwarding logic is
unit-testable without a live socket; the default transport uses
``aiohttp.ClientSession``. The transport is an async callable
``(method, url, json_body) -> (status_code, json_dict)``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from dlaas_platform_launcher.instance_manager import InstanceNotFound

Transport = Callable[[str, str, "dict[str, Any] | None"], Awaitable[tuple[int, dict]]]


async def _aiohttp_transport(
    method: str, url: str, json_body: dict | None
) -> tuple[int, dict]:  # pragma: no cover - requires network
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(method, url, json=json_body) as resp:
            try:
                body = await resp.json()
            except Exception:  # noqa: BLE001 - non-JSON error body
                body = {"status": "error", "detail": await resp.text()}
            return resp.status, body


class RemoteInstanceManager:
    """Parent-side HTTP proxy for one runtime pod."""

    def __init__(
        self,
        *,
        base_url: str,
        transport: Transport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("RemoteInstanceManager.base_url must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._transport = transport or _aiohttp_transport

    @property
    def base_url(self) -> str:
        return self._base_url

    async def acquire(
        self,
        *,
        ai_id: str,
        runtime_template_id: str,
        **kwargs: Any,
    ) -> dict:
        status, body = await self._transport(
            "POST",
            f"{self._base_url}/dlaas/v1/instances/{ai_id}/acquire",
            {"runtime_template_id": runtime_template_id, **kwargs},
        )
        if status >= 500:
            raise RuntimeError(
                f"pod acquire failed for ai_id={ai_id!r}: {body}"
            )
        return body

    async def forward_interaction(self, *, ai_id: str, envelope: Any) -> dict:
        payload = envelope.to_json() if hasattr(envelope, "to_json") else envelope
        status, body = await self._transport(
            "POST",
            f"{self._base_url}/dlaas/instances/{ai_id}/interactions",
            payload,
        )
        if status == 404:
            raise InstanceNotFound(ai_id)
        if status >= 500:
            raise RuntimeError(
                f"pod interaction forward failed for ai_id={ai_id!r}: {body}"
            )
        return body

    async def wake(self, *, ai_id: str, **kwargs: Any) -> dict:
        status, body = await self._transport(
            "POST",
            f"{self._base_url}/dlaas/v1/instances/{ai_id}/wake",
            dict(kwargs),
        )
        if status >= 500:
            raise RuntimeError(f"pod wake failed for ai_id={ai_id!r}: {body}")
        return body

    async def sleep(self, *, ai_id: str, **kwargs: Any) -> dict:
        status, body = await self._transport(
            "POST",
            f"{self._base_url}/dlaas/v1/instances/{ai_id}/sleep",
            dict(kwargs),
        )
        if status >= 500:
            raise RuntimeError(f"pod sleep failed for ai_id={ai_id!r}: {body}")
        return body

    async def status(self, ai_id: str) -> dict:
        status, body = await self._transport(
            "GET",
            f"{self._base_url}/dlaas/v1/instances/{ai_id}",
            None,
        )
        if status == 404:
            raise InstanceNotFound(ai_id)
        return body

    async def overview(self) -> list[dict]:
        status, body = await self._transport(
            "GET", f"{self._base_url}/dlaas/v1/instances", None
        )
        if status >= 500:
            raise RuntimeError(f"pod overview failed: {body}")
        instances = body.get("instances") if isinstance(body, dict) else None
        return list(instances or [])


__all__ = ["RemoteInstanceManager", "Transport"]
