"""Adapter: lifeform-evolution trace records → vz-substrate ``TrainingTrace``.

The trace collector emits flat ``TraceTurnRecord`` rows for portability. The
``vz-temporal`` SSL trainer expects ``TrainingTrace`` dataclasses with per-token
``TraceStep`` containing residual activations. This module bridges the two.

Why ``build_training_trace`` rather than carrying real residuals through the
ndjson schema? Two reasons:

1. ``TraceTurnRecord`` is meant to be a **portable, human-readable** archive
   (one JSON line per turn). Embedding raw float residuals there would blow
   the schema up by orders of magnitude and break json-line tooling.
2. ``vz-substrate.build_training_trace`` synthesises deterministic per-token
   residuals from the source text, which is exactly the substrate of the
   tokens we want to train on. For real-substrate runs (HF backend), a future
   helper can capture per-step residuals into a sidecar parquet file and
   round-trip them — but that is M2 of "downward growth"; this module is
   M1 (close the SSL loop on synthetic substrate).

Public API:

* ``trace_records_to_training_dataset(records)`` — pure conversion.
* ``trace_records_from_ndjson(path)`` — read back records the trace collector
  wrote, so this module can also work on traces collected in a previous run.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable

from volvence_zero.substrate import (
    TrainingTrace,
    TrainingTraceDataset,
    build_training_trace,
)

from lifeform_evolution.trace_collector import TraceTurnRecord


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def trace_records_to_training_dataset(
    records: Iterable[TraceTurnRecord],
    *,
    layer_count: int = 3,
) -> TrainingTraceDataset:
    """Build a ``TrainingTraceDataset`` from a sequence of trace rows.

    Each record becomes one ``TrainingTrace`` whose ``source_text`` is the
    concatenation of the user input and the assistant response — that is what
    the substrate would have processed when generating the turn.
    """
    dataset = TrainingTraceDataset()
    for record in records:
        trace = trace_record_to_training_trace(record, layer_count=layer_count)
        dataset.add_trace(trace)
    return dataset


def trace_record_to_training_trace(
    record: TraceTurnRecord,
    *,
    layer_count: int = 3,
) -> TrainingTrace:
    """Convert a single ``TraceTurnRecord`` to a ``TrainingTrace``.

    Trace ID encoding lets downstream training reports be attributed back to
    the originating turn:

        ``<scenario_id>::<session_id>::turn<NN>``
    """
    source_text = _compose_source_text(record)
    trace_id = (
        f"{record.scenario_id}::{record.session_id}::turn{record.turn_index:03d}"
    )
    return build_training_trace(
        trace_id=trace_id,
        source_text=source_text,
        layer_count=layer_count,
    )


def _compose_source_text(record: TraceTurnRecord) -> str:
    user_part = (record.user_input or "").strip()
    response_part = (record.response_text or "").strip()
    if user_part and response_part:
        return f"{user_part} {response_part}"
    return user_part or response_part or "<empty>"


# ---------------------------------------------------------------------------
# Reading collector output back from disk
# ---------------------------------------------------------------------------


def trace_records_from_ndjson(path: str | pathlib.Path) -> tuple[TraceTurnRecord, ...]:
    """Read records the ``TraceCollector`` wrote.

    The collector writes a frozen-dataclass JSON encoding; we re-construct
    ``TraceTurnRecord`` instances by mapping field-by-field. Forward-compat:
    unknown extra keys are dropped silently.
    """
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Trace file not found: {file_path}")

    field_names = {field for field in TraceTurnRecord.__dataclass_fields__.keys()}
    records: list[TraceTurnRecord] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            kwargs = {k: v for k, v in payload.items() if k in field_names}
            kwargs.setdefault("metadata", {})
            records.append(TraceTurnRecord(**kwargs))
    return tuple(records)
