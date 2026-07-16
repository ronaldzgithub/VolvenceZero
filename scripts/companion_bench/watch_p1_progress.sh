#!/usr/bin/env bash
# Live P1 same-substrate scoring dashboard (run in a second terminal).
#
# Usage:
#   bash scripts/companion_bench/watch_p1_progress.sh
#   bash scripts/companion_bench/watch_p1_progress.sh artifacts/companion-ablation/20260715T050841Z
#   bash scripts/companion_bench/watch_p1_progress.sh artifacts/companion-ablation/20260715T050841Z --follow
#   bash scripts/companion_bench/watch_p1_progress.sh artifacts/companion-ablation/20260715T050841Z --tail pe-off

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_DIR="${1:-}"
FOLLOW=0
TAIL_TRACK=""

shift_args=("$@")
if [[ -n "$ARTIFACT_DIR" ]]; then
  shift_args=("${@:2}")
fi

while [[ ${#shift_args[@]} -gt 0 ]]; do
  case "${shift_args[0]}" in
    --follow|-f)
      FOLLOW=1
      shift_args=("${shift_args[@]:1}")
      ;;
    --tail)
      [[ ${#shift_args[@]} -ge 2 ]] || { echo "usage: --tail <track-id>" >&2; exit 2; }
      TAIL_TRACK="${shift_args[1]}"
      shift_args=("${shift_args[@]:2}")
      ;;
    -h|--help)
      sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      if [[ -z "$ARTIFACT_DIR" ]]; then
        ARTIFACT_DIR="${shift_args[0]}"
        shift_args=("${shift_args[@]:1}")
      else
        echo "error: unknown argument: ${shift_args[0]}" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$ARTIFACT_DIR" ]]; then
  base="${REPO_ROOT}/artifacts/companion-ablation"
  [[ -d "$base" ]] || { echo "error: no artifact dir and ${base} missing" >&2; exit 2; }
  ARTIFACT_DIR="$(python - "$base" <<'PY'
import pathlib, sys
base = pathlib.Path(sys.argv[1])
print(max(base.iterdir(), key=lambda p: p.stat().st_mtime).resolve())
PY
)"
fi
if [[ "$ARTIFACT_DIR" != /* ]]; then
  ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_DIR}"
fi

SCORES="${ARTIFACT_DIR}/scores"
[[ -d "$SCORES" ]] || { echo "error: scores dir not found: ${SCORES}" >&2; exit 2; }

render_dashboard() {
  python - "$SCORES" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

scores = pathlib.Path(sys.argv[1])
expected = 30
rows = []
for track_dir in sorted(p for p in scores.iterdir() if p.is_dir()):
    tid = track_dir.name
    bundles = len(list((track_dir / "arcs").glob("*.bundle.json"))) if (track_dir / "arcs").is_dir() else 0
    summary_arc = None
    summary_path = track_dir / "summary.json"
    if summary_path.is_file():
        summary_arc = json.loads(summary_path.read_text(encoding="utf-8")).get("arc_count")
    fail_path = track_dir / "arcs" / "arc_failure.jsonl"
    failures = sum(1 for _ in fail_path.open(encoding="utf-8")) if fail_path.is_file() else 0
    progress = track_dir / "progress.jsonl"
    last = None
    if progress.is_file():
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = json.loads(line)
    status = "pending"
    if bundles >= expected and summary_arc == expected:
        status = "done"
    elif bundles > 0 or last:
        status = "running"
    bar_len = 20
    filled = min(bar_len, int(bar_len * bundles / expected)) if expected else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    detail = ""
    if last:
        ev = last.get("event", "?")
        sid = last.get("scenario_id", "?")
        scored = last.get("scored", "?")
        total = last.get("total", expected)
        if ev == "arc_ok":
            detail = f"last ok {sid} final={last.get('final', '?'):.1f} ({scored}/{total})"
        elif ev == "arc_failed":
            detail = f"last FAIL {sid} {last.get('exception_type', '?')} ({scored}/{total})"
        elif ev == "arc_resumed":
            detail = f"resumed {sid} ({scored}/{total})"
        else:
            detail = f"last {ev} {sid}"
    rows.append((tid, status, bundles, summary_arc, failures, bar, detail))

now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
print(f"P1 progress  artifact={scores.parent}")
print(f"updated      {now}")
print()
print(f"{'track':<28} {'status':<8} {'bundles':>7} {'summary':>7} {'fail':>5}  bar")
print("-" * 88)
for tid, status, bundles, summary_arc, failures, bar, detail in rows:
    summary_s = str(summary_arc) if summary_arc is not None else "-"
    print(f"{tid:<28} {status:<8} {bundles:>3}/{expected:<3} {summary_s:>7} {failures:>5}  [{bar}]")
    if detail:
        print(f"  {detail}")
done = sum(1 for r in rows if r[1] == "done")
print()
print(f"tracks done: {done}/{len(rows)}  (expected {expected} arcs per track, seed 0)")
verdict = scores.parent / "verdict_p1.json"
print(f"verdict_p1.json: {'yes' if verdict.is_file() else 'not yet'}")
PY
}

if [[ -n "$TAIL_TRACK" ]]; then
  progress="${SCORES}/abl-${TAIL_TRACK}/progress.jsonl"
  [[ -f "$progress" ]] || progress="${SCORES}/${TAIL_TRACK}/progress.jsonl"
  [[ -f "$progress" ]] || { echo "error: no progress.jsonl for track ${TAIL_TRACK}" >&2; exit 2; }
  echo "tailing ${progress}"
  tail -f "$progress"
  exit 0
fi

if [[ "$FOLLOW" -eq 1 ]]; then
  while true; do
    clear 2>/dev/null || true
    render_dashboard
    echo
    echo "refresh every 30s  (Ctrl-C to stop; --tail pe-off for live arc log)"
    sleep 30
  done
else
  render_dashboard
fi
