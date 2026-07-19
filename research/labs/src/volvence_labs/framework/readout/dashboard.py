"""Static HTML dashboard generator.

Produces a self-contained HTML file with:
- Runs table (sortable)
- Metrics pivot per probe
- Ablation diff summary
- Simple inline sparklines (pure JS, no dependencies)
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any, Optional

from .views import ablation_diff, metrics_pivot, runs_view
from ..probe import get_registry


def generate_dashboard(
    *,
    out_path: Optional[str] = None,
    root: Optional[str] = None,
) -> str:
    """Generate static HTML dashboard. Returns the HTML string."""
    # Ensure probes are registered.
    import volvence_labs.probes  # noqa: F401

    registry = get_registry()
    probe_ids = registry.all_ids()

    runs = runs_view(root=root, limit=500)
    probes_data: dict[str, Any] = {}
    for pid in probe_ids:
        pivot = metrics_pivot(pid, root=root)
        diff = ablation_diff(pid, root=root)
        if pivot or diff:
            probes_data[pid] = {"pivot": pivot, "diff": diff}

    html_content = _render_html(runs, probes_data, probe_ids)

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_content, encoding="utf-8")

    return html_content


def _render_html(
    runs: list[dict],
    probes_data: dict[str, Any],
    probe_ids: list[str],
) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    runs_json = json.dumps(runs[:100], indent=None, ensure_ascii=False)
    probes_json = json.dumps(probes_data, indent=None, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>volvence-labs dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; background: #fafafa; }}
h1 {{ color: #333; }}
h2 {{ color: #555; margin-top: 2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.85em; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.metric-pos {{ color: #2a7; }}
.metric-neg {{ color: #c44; }}
.sparkline {{ display: inline-block; height: 20px; }}
.summary {{ background: #fff; border: 1px solid #ddd; padding: 1em; border-radius: 4px; margin: 1em 0; }}
</style>
</head>
<body>
<h1>volvence-labs readout dashboard</h1>
<p>Generated: {ts} | Runs: {len(runs)} | Probes: {len(probe_ids)}</p>

<h2>Recent Runs</h2>
<table id="runs-table">
<thead><tr><th>run_id</th><th>probe</th><th>cell</th><th>seed</th><th>wiring</th></tr></thead>
<tbody>
{"".join(_run_row(r) for r in runs[:50])}
</tbody>
</table>

<h2>Ablation Diffs</h2>
{"".join(_probe_section(pid, probes_data.get(pid, {})) for pid in probe_ids if pid in probes_data)}

<script>
const runsData = {runs_json};
const probesData = {probes_json};
</script>
</body>
</html>"""


def _run_row(r: dict) -> str:
    return (
        f"<tr><td>{html.escape(r['run_id'][:40])}</td>"
        f"<td>{html.escape(r['probe_id'])}</td>"
        f"<td>{html.escape(r['cell'])}</td>"
        f"<td>{r['seed']}</td>"
        f"<td>{html.escape(r['wiring'])}</td></tr>\n"
    )


def _probe_section(pid: str, data: dict) -> str:
    diff = data.get("diff", {})
    if not diff or "error" in diff:
        return f"<div class='summary'><h3>{html.escape(pid)}</h3><p>Insufficient data</p></div>\n"

    rows = ""
    for metric, stats in diff.items():
        d = stats.get("diff_mean", 0)
        cls = "metric-pos" if d > 0 else "metric-neg" if d < 0 else ""
        rows += (
            f"<tr><td>{html.escape(metric)}</td>"
            f"<td>{stats['baseline_mean']:.4f} ± {stats['baseline_std']:.4f}</td>"
            f"<td>{stats['target_mean']:.4f} ± {stats['target_std']:.4f}</td>"
            f"<td class='{cls}'>{d:+.4f}</td></tr>\n"
        )

    return f"""<div class='summary'>
<h3>{html.escape(pid)}</h3>
<table>
<thead><tr><th>Metric</th><th>Baseline (mean±std)</th><th>Probe On (mean±std)</th><th>Diff</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
"""
