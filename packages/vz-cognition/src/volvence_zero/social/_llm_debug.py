"""Opt-in observability sink for LLM-backed social proposal runtimes.

Both ``LLMToMProposalRuntime`` and ``LLMCommonGroundProposalRuntime``
generate from a text provider, then strict-parse a JSON payload. When
parse fails (or yields zero typed proposals) the runtime returns ``()``
silently — by design, because malformed LLM output is not a kernel
fault.

That silence is the right product behaviour, but it makes diagnosing a
0-records evidence run impossible without re-running the model and
catching every call by hand. This module provides an optional sink so a
diagnostic run can capture the raw provider output + parse outcome to a
JSONL file, while the default code path stays a no-op.

Activation contract (env-gated, no constructor surface change):

* Set ``VZ_LLM_PROPOSAL_DEBUG_LOG=<path>`` before launching the host
  process. The path's parent directory must exist; we **fail loudly**
  if it does not (or if the file cannot be opened for append). This
  matches the no-swallow-errors rule: silent observability that
  silently breaks is worse than no observability.
* Each call to ``log_proposal_attempt(...)`` appends one JSON record
  per LLM call. Records are flushed immediately so a crash mid-run
  still leaves the partial trace on disk.
* When the env var is unset or empty, ``make_attempt_logger`` returns
  ``None`` and the runtime skips the call entirely (no string format,
  no IO).

Schema (one JSON record per LLM call):

* ``ts`` — ISO-8601 UTC timestamp
* ``runtime_id`` — owning runtime's ``runtime_id`` (e.g. ``social-tom-llm-structured``)
* ``target_slot`` — for ToM, the requested slot; ``None`` for
  common-ground (which doesn't slot-dispatch)
* ``turn_index`` — caller-supplied turn id
* ``prompt`` — the FULL formatted prompt sent to the provider; we do
  not truncate because the diagnostic value of seeing exactly which
  user-input variant produced a malformed reply outweighs disk cost
  on a debug-only path
* ``raw_output`` — the FULL raw string from ``provider.generate(...)``,
  pre-parse
* ``parse_status`` — one of ``"ok"`` / ``"parse_error"`` /
  ``"empty_or_rejected"``
* ``parsed_count`` — number of typed proposals that survived schema
  validation (0 when ``parse_status != "ok"``)
* ``parse_error`` — short error class name when ``parse_status ==
  "parse_error"``; ``None`` otherwise
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_ENV_VAR = "VZ_LLM_PROPOSAL_DEBUG_LOG"


AttemptLogger = Callable[[dict[str, object]], None]


def make_attempt_logger() -> AttemptLogger | None:
    """Return a JSONL-append callable bound to ``VZ_LLM_PROPOSAL_DEBUG_LOG``.

    Returns ``None`` when the env var is unset / empty so the default
    runtime path stays zero-overhead. Raises ``FileNotFoundError`` when
    the env var is set but its parent directory is missing — silent
    observability that silently breaks is worse than no observability.
    """
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        return None
    target = Path(raw).expanduser()
    parent = target.parent
    if not parent.exists():
        raise FileNotFoundError(
            f"{_ENV_VAR}={raw!s} parent directory does not exist: "
            f"{parent}. Create it before launching the diagnostic run."
        )

    def _append(record: dict[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")
            fh.flush()

    return _append


def log_proposal_attempt(
    logger: AttemptLogger | None,
    *,
    runtime_id: str,
    target_slot: str | None,
    turn_index: int,
    prompt: str,
    raw_output: str,
    parsed_count: int,
    parse_status: str,
    parse_error: str | None,
) -> None:
    """Append one diagnostic record to the configured JSONL sink.

    No-op when ``logger`` is ``None`` so this can be called
    unconditionally from the runtime's hot path.
    """
    if logger is None:
        return
    record: dict[str, object] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "runtime_id": runtime_id,
        "target_slot": target_slot,
        "turn_index": turn_index,
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_count": parsed_count,
        "parse_status": parse_status,
        "parse_error": parse_error,
    }
    logger(record)


__all__ = [
    "AttemptLogger",
    "log_proposal_attempt",
    "make_attempt_logger",
]
