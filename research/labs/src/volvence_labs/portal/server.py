"""Loopback-only JSON server for immutable Research Lab snapshots."""

from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from .collector import ResearchLabCollector

_TASK_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def create_server(
    collector: ResearchLabCollector,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    if host not in _LOOPBACK_HOSTS:
        raise ValueError("Research Lab read-only API may only bind to a loopback host")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    handler = _handler_factory(collector)
    return ThreadingHTTPServer((host, port), handler)


def serve(
    collector: ResearchLabCollector,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = create_server(collector, host=host, port=port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _handler_factory(collector: ResearchLabCollector) -> type[BaseHTTPRequestHandler]:
    class ResearchLabHandler(BaseHTTPRequestHandler):
        server_version = "VolvenceResearchLab/1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib protocol hook
            path = urlsplit(self.path).path
            if path == "/healthz":
                self._write_json(HTTPStatus.OK, {"status": "ok", "mode": "read_only"})
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
            self._write_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {"error": "read_only", "message": "mutation endpoints are not enabled in this milestone"},
                extra_headers={"Allow": "GET"},
            )

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
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            super().log_message(format, *args)

    return ResearchLabHandler
