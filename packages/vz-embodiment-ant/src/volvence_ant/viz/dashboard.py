"""Live and replay dashboard driven only by immutable ``AntStepRecord`` values."""

from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock, Thread
from typing import Iterable

from volvence_ant.runtime import AntStepRecord


def _frame(label: str, record: AntStepRecord) -> dict:
    payload = asdict(record)
    payload["label"] = label
    return payload


_DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Digital Ant Evidence</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}
header{padding:14px 18px;border-bottom:1px solid #444}
main{display:grid;grid-template-columns:2fr 1fr;gap:16px;padding:16px}
canvas{width:100%;background:#181818;border:1px solid #444}
#metrics{white-space:pre-wrap;font-family:ui-monospace,monospace}
.pass{color:#65c466}.block{color:#e06c75}
</style></head><body>
<header><strong>Digital Ant — snapshot-driven live evidence</strong>
<span id="status"> waiting</span></header>
<main><canvas id="world" width="900" height="650"></canvas>
<section><h3>Kernel telemetry</h3><div id="metrics"></div></section></main>
<script>
const colors=["#61afef","#e06c75","#98c379","#c678dd","#e5c07b"];
async function refresh(){
 const r=await fetch("/frames"); const data=await r.json();
 const c=document.querySelector("#world"),x=c.getContext("2d");
 x.clearRect(0,0,c.width,c.height); x.strokeStyle="#444";
 x.beginPath();x.moveTo(c.width/2,0);x.lineTo(c.width/2,c.height);
 x.moveTo(0,c.height/2);x.lineTo(c.width,c.height/2);x.stroke();
 const groups={}; for(const f of data.frames)(groups[f.label]??=[]).push(f);
 Object.entries(groups).forEach(([label,fs],i)=>{
   x.strokeStyle=colors[i%colors.length];x.beginPath();
   fs.forEach((f,j)=>{const px=c.width/2+f.x*25,py=c.height/2-f.y*25;
     j?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
   const f=fs[fs.length-1];if(f){x.fillStyle=colors[i%colors.length];
     x.fillText(label,c.width/2+f.x*25+5,c.height/2-f.y*25)}
 });
 const last=data.frames[data.frames.length-1];
 if(last)document.querySelector("#metrics").textContent=
  `arm: ${last.label}\\ntick: ${last.tick}\\naction: ${last.abstract_action}`+
  `\\nβ: ${last.switch_gate.toFixed(3)}\\nz: [${last.code.map(v=>v.toFixed(2)).join(", ")}]`+
  `\\nPE: ${last.pe_magnitude.toFixed(3)}\\nreward: ${last.signed_reward.toFixed(3)}`+
  `\\ncredit: ${last.cumulative_credit.toFixed(3)}\\nwriteback: ${last.bounded_writeback_applied}`+
  `\\nmemory entries: ${last.memory_entries_total}\\nCMS observations: ${last.cms_total_observations}`+
  `\\nschedule: ${last.joint_schedule_action}`+
  `\\nwiring: ${last.backend_wiring.map(v=>v.join("=")).join("\\n        ")}`;
 document.querySelector("#status").textContent=` · ${data.frames.length} frames`;
 if(!data.complete)setTimeout(refresh,150);
} refresh();
</script></body></html>"""


class LiveAntDashboard:
    """Threaded localhost dashboard; publishing never mutates kernel owners."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._frames: list[dict] = []
        self._lock = Lock()
        self._complete = False
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/frames":
                    with dashboard._lock:
                        body = json.dumps(
                            {
                                "frames": tuple(dashboard._frames),
                                "complete": dashboard._complete,
                            }
                        ).encode()
                    content_type = "application/json"
                else:
                    body = _DASHBOARD_HTML.encode()
                    content_type = "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self.url = f"http://{host}:{self._server.server_port}/"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def publish(self, label: str, record: AntStepRecord) -> None:
        with self._lock:
            self._frames.append(_frame(label, record))

    def finish(self) -> None:
        with self._lock:
            self._complete = True

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def export_replay(self, path: Path) -> Path:
        with self._lock:
            payload = {
                "schema_version": "digital-ant-replay.v1",
                "frames": tuple(self._frames),
                "complete": self._complete,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def write_replay_dashboard(
    *, tracks: dict[str, Iterable[AntStepRecord]], out_path: Path
) -> Path:
    frames = [
        _frame(label, record)
        for label, records in tracks.items()
        for record in records
    ]
    replay_json = json.dumps(
        {"frames": frames, "complete": True}, separators=(",", ":")
    )
    html = _DASHBOARD_HTML.replace(
        'const r=await fetch("/frames"); const data=await r.json();',
        f"const data={replay_json};",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


__all__ = ["LiveAntDashboard", "write_replay_dashboard"]
