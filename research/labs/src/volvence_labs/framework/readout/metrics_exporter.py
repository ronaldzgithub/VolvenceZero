"""Prometheus metrics exporter for Volvence Labs.

Exposes probe run metrics via HTTP for Prometheus scraping.
Start with: volvence-labs metrics-server --port 9090

Metrics exported:
- volvence_probes_registered: gauge of registered probe count
- volvence_runs_total: counter of completed runs (by probe_id, cell, outcome)
- volvence_runs_failed_total: counter of failed runs
- volvence_run_duration_seconds: histogram of run durations
- volvence_metric_value: gauge of latest metric values (by probe_id, cell, metric_name)
- volvence_gate_decision: gauge of latest gate decision (1=approve, 0=hold, -1=reject)
"""

from __future__ import annotations

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from ..probe import get_registry
from ..snapshot import CASStore, RunLog, default_paths


# --- Registry and metrics ---

_registry = CollectorRegistry()

PROBES_REGISTERED = Gauge(
    "volvence_probes_registered",
    "Number of registered probes",
    registry=_registry,
)

RUNS_TOTAL = Counter(
    "volvence_runs_total",
    "Total completed probe runs",
    ["probe_id", "cell", "outcome"],
    registry=_registry,
)

RUNS_FAILED = Counter(
    "volvence_runs_failed_total",
    "Total failed probe runs",
    ["probe_id", "cell"],
    registry=_registry,
)

RUN_DURATION = Histogram(
    "volvence_run_duration_seconds",
    "Run duration in seconds",
    ["probe_id", "cell"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600],
    registry=_registry,
)

METRIC_VALUE = Gauge(
    "volvence_metric_value",
    "Latest metric value from probe readouts",
    ["probe_id", "cell", "metric_name", "seed"],
    registry=_registry,
)

GATE_DECISION = Gauge(
    "volvence_gate_decision",
    "Latest gate decision (1=approve, 0=hold, -1=reject)",
    ["probe_id"],
    registry=_registry,
)


def _load_run_history(root: Optional[str] = None) -> None:
    """Load existing run history into Prometheus metrics."""
    import volvence_labs.probes  # noqa: F401 — ensure probes registered

    registry = get_registry()
    PROBES_REGISTERED.set(len(registry.all_ids()))

    paths = default_paths(root)
    store = CASStore(paths)
    log = RunLog(paths, store)

    records = log.list(limit=5000)
    for rec in records:
        probe_id = rec.probe_id
        cell = rec.ablation_cell
        outcome = "ok" if rec.ok else "fail"

        RUNS_TOTAL.labels(probe_id=probe_id, cell=cell, outcome=outcome).inc()
        if not rec.ok:
            RUNS_FAILED.labels(probe_id=probe_id, cell=cell).inc()

        RUN_DURATION.labels(probe_id=probe_id, cell=cell).observe(rec.elapsed_s)

        # Load readout metrics
        try:
            readouts = store.get_obj(rec.readouts_sha)
            metrics = readouts.get("metrics", {})
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    METRIC_VALUE.labels(
                        probe_id=probe_id,
                        cell=cell,
                        metric_name=metric_name,
                        seed=str(rec.seed),
                    ).set(value)
        except Exception:
            pass

    log.close()
    store.close()


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves Prometheus metrics."""

    def do_GET(self):
        if self.path == "/metrics":
            output = generate_latest(_registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default access logs
        pass


def start_metrics_server(
    *,
    host: str = "0.0.0.0",
    port: int = 9090,
    root: Optional[str] = None,
) -> None:
    """Start the Prometheus metrics HTTP server (blocking).

    Loads existing run history, then serves /metrics endpoint.
    """
    print(f"[metrics] loading run history...")
    _load_run_history(root)
    print(f"[metrics] loaded. serving on http://{host}:{port}/metrics")

    server = HTTPServer((host, port), _MetricsHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[metrics] shutting down")
        server.shutdown()


def record_run(
    probe_id: str,
    cell: str,
    ok: bool,
    elapsed_s: float,
    metrics: Optional[dict] = None,
    seed: int = 0,
) -> None:
    """Record a single run completion into Prometheus metrics.

    Call this from the runner after each unit completes.
    """
    outcome = "ok" if ok else "fail"
    RUNS_TOTAL.labels(probe_id=probe_id, cell=cell, outcome=outcome).inc()
    if not ok:
        RUNS_FAILED.labels(probe_id=probe_id, cell=cell).inc()
    RUN_DURATION.labels(probe_id=probe_id, cell=cell).observe(elapsed_s)

    if metrics:
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                METRIC_VALUE.labels(
                    probe_id=probe_id,
                    cell=cell,
                    metric_name=metric_name,
                    seed=str(seed),
                ).set(value)


def record_gate_decision(probe_id: str, decision: str) -> None:
    """Record a gate decision. decision: 'approve'|'hold'|'reject'."""
    value_map = {"approve": 1, "hold": 0, "reject": -1}
    GATE_DECISION.labels(probe_id=probe_id).set(value_map.get(decision, 0))
