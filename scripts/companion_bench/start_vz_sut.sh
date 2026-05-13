#!/usr/bin/env bash
# Start / stop a local VolvenceZero companion-bench SUT process.
#
# Used by `run_companion_bench_smoke.sh` to background a `lifeform-serve`
# process on port 8000 with `--enable-openai-compat` so companion-bench
# can hit it via the OpenAI-compatible chat-completions surface.
#
# Subcommands:
#   start   — launch lifeform-serve in background, wait for /v1/health, exit 0
#             when ready (or 1 on timeout). PID written to .vz_sut.pid.
#   stop    — read PID and SIGTERM the process; remove .vz_sut.pid.
#   status  — print PID and last-known health state.
#
# Env vars (optional):
#   VZ_SUT_PORT          (default 8000)
#   VZ_SUT_VERTICAL      (default companion)
#   VZ_SUT_SUBSTRATE     (default synthetic)
#   VZ_SUT_HEALTH_TIMEOUT_S  (default 30)
#   VZ_SUT_LOG           (default artifacts/companion_bench_smoke/vz_sut.log)
#
# Refs:
#   docs/external/companion-bench-openrouter-setup.md
#   packages/lifeform-service/src/lifeform_service/cli.py (--enable-openai-compat)

set -euo pipefail

PORT="${VZ_SUT_PORT:-8000}"
VERTICAL="${VZ_SUT_VERTICAL:-companion}"
SUBSTRATE="${VZ_SUT_SUBSTRATE:-synthetic}"
HEALTH_TIMEOUT="${VZ_SUT_HEALTH_TIMEOUT_S:-30}"
LOG_FILE="${VZ_SUT_LOG:-artifacts/companion_bench_smoke/vz_sut.log}"
PID_FILE=".vz_sut.pid"

cmd_start() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "VZ SUT already running (pid $PID, port $PORT)"
      exit 0
    fi
    rm -f "$PID_FILE"
  fi

  mkdir -p "$(dirname "$LOG_FILE")"
  echo "Starting lifeform-serve on port $PORT (vertical=$VERTICAL substrate=$SUBSTRATE)..."
  nohup lifeform-serve \
    --vertical "$VERTICAL" \
    --substrate-mode "$SUBSTRATE" \
    --enable-openai-compat \
    --port "$PORT" \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  PID=$(cat "$PID_FILE")
  echo "Started pid=$PID, log=$LOG_FILE"

  echo "Waiting for /v1/health (timeout ${HEALTH_TIMEOUT}s)..."
  WAITED=0
  while [ "$WAITED" -lt "$HEALTH_TIMEOUT" ]; do
    if curl -sf "http://127.0.0.1:${PORT}/v1/health" >/dev/null 2>&1; then
      echo "VZ SUT healthy on http://127.0.0.1:${PORT}/v1"
      exit 0
    fi
    sleep 1
    WAITED=$((WAITED + 1))
  done

  echo "ERROR: VZ SUT did not become healthy within ${HEALTH_TIMEOUT}s"
  echo "Last log lines:"
  tail -n 30 "$LOG_FILE" || true
  exit 1
}

cmd_stop() {
  if [ ! -f "$PID_FILE" ]; then
    echo "No PID file ($PID_FILE); nothing to stop"
    exit 0
  fi
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping VZ SUT (pid $PID)..."
    kill -TERM "$PID" || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      kill -KILL "$PID" || true
    fi
  else
    echo "Process pid $PID already gone"
  fi
  rm -f "$PID_FILE"
  echo "Stopped"
}

cmd_status() {
  if [ ! -f "$PID_FILE" ]; then
    echo "Not running (no $PID_FILE)"
    exit 1
  fi
  PID=$(cat "$PID_FILE")
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "PID file present but process pid=$PID is dead"
    exit 1
  fi
  echo "Running pid=$PID port=$PORT"
  if curl -sf "http://127.0.0.1:${PORT}/v1/health" >/dev/null 2>&1; then
    echo "Health: ok"
  else
    echo "Health: unreachable"
    exit 2
  fi
}

case "${1:-}" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  *)
    echo "Usage: $0 {start|stop|status}"
    exit 1
    ;;
esac
