"""Loopback-only JSON server for Research Lab snapshots and exact commands."""

from __future__ import annotations

import hmac
import json
import re
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from .collector import ResearchLabCollector
from .commands import PortalCommandError, ResearchLabCommandService

_TASK_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_MAX_REQUEST_BYTES = 16 * 1024


def create_server(
    collector: ResearchLabCollector,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    command_service: ResearchLabCommandService | None = None,
    allowed_origins: tuple[str, ...] = (),
    csrf_token: str | None = None,
) -> ThreadingHTTPServer:
    if host not in _LOOPBACK_HOSTS:
        raise ValueError("Research Lab API may only bind to a loopback host")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    normalized_origins = tuple(_normalize_origin(origin) for origin in allowed_origins)
    if command_service is not None and not normalized_origins:
        raise ValueError("mutation mode requires at least one explicit loopback UI origin")
    token = csrf_token or (secrets.token_urlsafe(32) if command_service is not None else None)
    if command_service is not None and (token is None or len(token) < 32):
        raise ValueError("mutation mode requires a CSRF token of at least 32 characters")
    handler = _handler_factory(
        collector,
        command_service=command_service,
        allowed_origins=frozenset(normalized_origins),
        csrf_token=token,
    )
    return ThreadingHTTPServer((host, port), handler)


def serve(
    collector: ResearchLabCollector,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    command_service: ResearchLabCommandService | None = None,
    allowed_origins: tuple[str, ...] = (),
) -> None:
    server = create_server(
        collector,
        host=host,
        port=port,
        command_service=command_service,
        allowed_origins=allowed_origins,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _handler_factory(
    collector: ResearchLabCollector,
    *,
    command_service: ResearchLabCommandService | None,
    allowed_origins: frozenset[str],
    csrf_token: str | None,
) -> type[BaseHTTPRequestHandler]:
    class ResearchLabHandler(BaseHTTPRequestHandler):
        server_version = "VolvenceResearchLab/1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib protocol hook
            path = urlsplit(self.path).path
            if path == "/healthz":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "mode": "controlled_mutation" if command_service is not None else "read_only",
                    },
                )
                return
            if path == "/api/v1/session":
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "schema_version": "volvence-research-lab-session.v1",
                        "mutations_enabled": command_service is not None,
                        "csrf_token": csrf_token,
                        "supported_actions": (
                            list(command_service.supported_actions) if command_service is not None else []
                        ),
                    },
                )
                return
            snapshot = collector.collect()
            if path == "/api/v1/snapshot":
                self._write_json(HTTPStatus.OK, snapshot.to_jsonable())
                return
            prefix = "/api/v1/tasks/"
            if path.startswith(prefix):
                task_id = unquote(path[len(prefix) :])
                if not _TASK_ID.fullmatch(task_id):
                    self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_task_id"})
                    return
                item = snapshot.get_task(task_id)
                if item is None:
                    self._write_json(HTTPStatus.NOT_FOUND, {"error": "task_not_found"})
                    return
                item_payload = next(value for value in snapshot.to_jsonable()["items"] if value["task_id"] == task_id)
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "schema_version": snapshot.schema_version,
                        "generated_at": snapshot.generated_at,
                        "revision": snapshot.revision,
                        "item": item_payload,
                    },
                )
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib protocol hook
            if command_service is None:
                self._write_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "read_only", "message": "mutation endpoints are not enabled"},
                    extra_headers={"Allow": "GET"},
                )
                return
            path = urlsplit(self.path).path
            commands = {
                "/api/v1/a0/review": command_service.review_a0,
                "/api/v1/reconcile": command_service.reconcile,
                "/api/v1/candidates/import": command_service.import_candidate,
                "/api/v1/a1/authorize-shadow": command_service.authorize_shadow,
                "/api/v1/a2/authorize-active": command_service.authorize_active,
                "/api/v1/rollback": command_service.rollback,
            }
            command = commands.get(path)
            if command is None:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            security_error = self._mutation_security_error()
            if security_error is not None:
                self._write_json(HTTPStatus.FORBIDDEN, security_error)
                return
            try:
                payload = self._read_json_payload()
                result = command(payload)
            except PortalCommandError as exc:
                self._write_json(
                    HTTPStatus(exc.status_code),
                    {"error": exc.code, "message": exc.message},
                )
                return
            except Exception as exc:  # request-process fault boundary
                self.log_error("Research Lab mutation failed: %r", exc)
                self._write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "internal_error", "message": "mutation failed at the local request boundary"},
                )
                return
            self._write_json(HTTPStatus.OK, result)

        def _mutation_security_error(self) -> dict[str, str] | None:
            origin = self.headers.get("Origin")
            if origin not in allowed_origins:
                return {
                    "error": "origin_forbidden",
                    "message": "mutation Origin does not match an explicitly configured local UI",
                }
            submitted = self.headers.get("X-Research-Lab-CSRF")
            if csrf_token is None or submitted is None or not hmac.compare_digest(submitted, csrf_token):
                return {"error": "csrf_forbidden", "message": "CSRF token is missing or invalid"}
            return None

        def _read_json_payload(self) -> dict[str, object]:
            content_type = self.headers.get("Content-Type", "")
            if content_type.partition(";")[0].strip().lower() != "application/json":
                raise PortalCommandError(
                    "unsupported_media_type",
                    "mutation requests must use application/json",
                    status_code=415,
                )
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length is not None else -1
            except ValueError as exc:
                raise PortalCommandError("invalid_request", "invalid Content-Length", status_code=400) from exc
            if not 0 < length <= _MAX_REQUEST_BYTES:
                raise PortalCommandError(
                    "invalid_request_size",
                    f"mutation body must contain 1 to {_MAX_REQUEST_BYTES} bytes",
                    status_code=413,
                )
            try:
                value = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PortalCommandError("invalid_json", "mutation body is not valid JSON", status_code=400) from exc
            if not isinstance(value, dict):
                raise PortalCommandError("invalid_request", "mutation body must be a JSON object", status_code=400)
            return value

        def _write_json(
            self,
            status: HTTPStatus,
            payload: object,
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            super().log_message(format, *args)

    return ResearchLabHandler


def _normalize_origin(origin: str) -> str:
    value = origin.rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Research Lab UI origin must be an exact loopback origin: {origin!r}")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"Research Lab UI origin has an invalid port: {origin!r}") from exc
    return value
