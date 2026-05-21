#!/usr/bin/env bash
# Stop the browser chat service started by start_browser_chat_qwen.sh.
#
# Usage:
#   bash stop_browser_chat_qwen.sh
#   PORT=8766 bash stop_browser_chat_qwen.sh
#
# Windows: .\stop_browser_chat_qwen.ps1  (or $env:PORT='8766')

set -euo pipefail

PORT="${PORT:-8765}"

if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof is required to find the chat service process." >&2
  exit 1
fi

PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"

if [[ -z "$PIDS" ]]; then
  echo "No browser chat service is listening on port ${PORT}."
  exit 0
fi

echo "Stopping browser chat service on port ${PORT}: ${PIDS}"
kill $PIDS

sleep 1

REMAINING_PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
if [[ -n "$REMAINING_PIDS" ]]; then
  echo "Service is still listening on port ${PORT}; forcing stop: ${REMAINING_PIDS}"
  kill -9 $REMAINING_PIDS
fi

echo "Browser chat service stopped."
