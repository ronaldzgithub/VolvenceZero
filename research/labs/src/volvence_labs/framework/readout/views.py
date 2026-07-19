"""SQL-driven views over the RunLog + CAS readouts.

All views return plain dicts/lists (JSON-serializable) so they can be consumed
by the dashboard or any downstream tool.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from ..snapshot import CASStore, LabsPaths, RunLog, default_paths


def runs_view(
    *,
    probe_id: Optional[str] = None,
    limit: int = 200,
    root: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List runs with basic metadata."""
    paths = default_paths(root)
    store = CASStore(paths)
    log = RunLog(paths, store)
    records = log.list(probe_id=probe_id, limit=limit)
    result = []
    for r in records:
        result.append({
            "run_id": r.run_id,
            "probe_id": r.probe_id,
            "cell": r.ablation_cell,
            "wiring": r.wiring,
            "seed": r.seed,
            "created_at": r.created_at,
            "readouts_sha": r.readouts_sha,
        })
    log.close()
    store.close()
    return result


def metrics_pivot(
    probe_id: str,
    *,
    root: Optional[str] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Pivot metrics by cell. Returns {cell: [{seed, metric_name: value, ...}]}."""
    paths = default_paths(root)
    store = CASStore(paths)
    log = RunLog(paths, store)
    records = log.list(probe_id=probe_id, limit=500)

    pivot: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        try:
            readouts = store.get_obj(r.readouts_sha)
        except (KeyError, json.JSONDecodeError):
            continue
        metrics = readouts.get("metrics", {})
        entry = {"seed": r.seed, "run_id": r.run_id, **metrics}
        pivot.setdefault(r.ablation_cell, []).append(entry)

    log.close()
    store.close()
    return pivot


def ablation_diff(
    probe_id: str,
    *,
    baseline_cell: str = "baseline",
    target_cell: str = "probe_on",
    root: Optional[str] = None,
) -> dict[str, Any]:
    """Compare metrics between two cells (mean ± std across seeds).

    Returns {metric_name: {baseline_mean, baseline_std, target_mean, target_std, diff_mean}}.
    """
    pivot = metrics_pivot(probe_id, root=root)
    baseline_entries = pivot.get(baseline_cell, [])
    target_entries = pivot.get(target_cell, [])

    if not baseline_entries or not target_entries:
        return {"error": "insufficient data", "baseline_n": len(baseline_entries), "target_n": len(target_entries)}

    # Collect all metric names from both cells.
    all_metrics: set[str] = set()
    for e in baseline_entries + target_entries:
        all_metrics.update(k for k in e if k not in ("seed", "run_id"))

    import math

    result: dict[str, Any] = {}
    for m in sorted(all_metrics):
        b_vals = [e[m] for e in baseline_entries if m in e]
        t_vals = [e[m] for e in target_entries if m in e]
        if not b_vals or not t_vals:
            continue
        b_mean = sum(b_vals) / len(b_vals)
        t_mean = sum(t_vals) / len(t_vals)
        b_std = math.sqrt(sum((x - b_mean) ** 2 for x in b_vals) / len(b_vals)) if len(b_vals) > 1 else 0.0
        t_std = math.sqrt(sum((x - t_mean) ** 2 for x in t_vals) / len(t_vals)) if len(t_vals) > 1 else 0.0
        result[m] = {
            "baseline_mean": b_mean,
            "baseline_std": b_std,
            "target_mean": t_mean,
            "target_std": t_std,
            "diff_mean": t_mean - b_mean,
        }

    return result
