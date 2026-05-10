"""Load / construct :class:`EvaluationSnapshot` for the figure CLI.

The CLI's ``bake-steering`` and ``bake-lora`` subcommands route a
:class:`ModificationProposal` through the OFFLINE gate, which reads
:class:`EvaluationSnapshot.alerts` + ``contract_integrity`` /
``rollback_resilience`` / ``fallback_reliance`` scores. Operators
need a way to supply that snapshot from disk **and** a way to run
the CLI in development mode without hand-writing JSON every time.

Two entry points:

* :func:`load_evaluation_snapshot` — JSON path → typed snapshot.
  Schema mismatches fail loud (no-swallow): a missing field is the
  intended failure mode that tells the operator "the gate did not
  receive the eval data you expected".
* :func:`default_clean_snapshot` — the same constant snapshot the
  test suite uses (see
  ``tests/test_persona_lora_apply_smoke.py::_clean_evaluation_snapshot``).
  Surfaced via the ``--evaluation-snapshot default-clean`` literal so
  operators have to opt in explicitly; this prevents production
  promotions from silently coasting on the developer default.

JSON shape::

    {
        "description": "<free text>",
        "turn_scores": [
            {
                "family": "behavior",
                "metric_name": "contract_integrity",
                "value": 0.99,
                "confidence": 0.95,
                "evidence": "<text>"
            },
            ...
        ],
        "session_scores": [...],   // same shape, optional
        "alerts": ["<legacy alert text>", ...]  // optional
    }
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from volvence_zero.evaluation.types import EvaluationScore, EvaluationSnapshot


DEFAULT_CLEAN_LITERAL = "default-clean"


def load_evaluation_snapshot(spec: str) -> EvaluationSnapshot:
    """Resolve ``spec`` to a typed snapshot.

    ``spec`` is exactly the value passed to ``--evaluation-snapshot``:

    * the literal :data:`DEFAULT_CLEAN_LITERAL` → returns
      :func:`default_clean_snapshot` (no I/O);
    * any other value → treated as a path to a JSON file produced
      by the operator pipeline.
    """

    if not spec.strip():
        raise ValueError(
            "load_evaluation_snapshot: --evaluation-snapshot must be "
            "non-empty (use 'default-clean' for the developer default)"
        )
    if spec == DEFAULT_CLEAN_LITERAL:
        return default_clean_snapshot()
    path = pathlib.Path(spec)
    if not path.is_file():
        raise FileNotFoundError(
            f"load_evaluation_snapshot: no file at {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _snapshot_from_payload(payload)


def default_clean_snapshot() -> EvaluationSnapshot:
    """The clean snapshot operators use for developer-mode demos.

    Mirrors the ``_clean_evaluation_snapshot`` helper in
    :mod:`tests.test_persona_lora_apply_smoke` byte-for-byte so the
    CLI's developer mode and the wheel's smoke tests produce
    interchangeable evidence (R8: one snapshot definition, two
    consumers).
    """

    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                "behavior", "contract_integrity", 0.99, 0.95,
                "all contracts honored",
            ),
            EvaluationScore(
                "behavior", "rollback_resilience", 0.99, 0.95,
                "rollback drill clean",
            ),
            EvaluationScore(
                "behavior", "fallback_reliance", 0.10, 0.95,
                "no fallback",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="default-clean offline-gate snapshot",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _snapshot_from_payload(payload: Any) -> EvaluationSnapshot:
    if not isinstance(payload, dict):
        raise ValueError(
            f"load_evaluation_snapshot: top-level JSON must be an "
            f"object, got {type(payload).__name__}"
        )
    description = str(payload.get("description", ""))
    turn_scores = tuple(
        _score_from_payload(entry) for entry in payload.get("turn_scores", ())
    )
    session_scores = tuple(
        _score_from_payload(entry)
        for entry in payload.get("session_scores", ())
    )
    alerts_raw = payload.get("alerts", ())
    alerts = tuple(str(a) for a in alerts_raw)
    return EvaluationSnapshot(
        turn_scores=turn_scores,
        session_scores=session_scores,
        alerts=alerts,
        description=description,
    )


def _score_from_payload(entry: Any) -> EvaluationScore:
    if not isinstance(entry, dict):
        raise ValueError(
            f"_score_from_payload: each score entry must be an "
            f"object, got {type(entry).__name__}"
        )
    return EvaluationScore(
        family=str(entry["family"]),
        metric_name=str(entry["metric_name"]),
        value=float(entry["value"]),
        confidence=float(entry["confidence"]),
        evidence=str(entry["evidence"]),
    )


__all__ = [
    "DEFAULT_CLEAN_LITERAL",
    "default_clean_snapshot",
    "load_evaluation_snapshot",
]
