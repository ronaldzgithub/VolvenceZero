"""Packet 7.1: free-text task description uptake adapter.

Reads short task description strings (e.g. "Tonally cheerful
sales assistant for booking spas") and converts them to
``BehaviorProtocolCandidate`` via a single LLM call. Unlike
DocumentUptake (multi-chunk PDFs), TaskDescription operates
on one short prompt.

Public API:

* :func:`extract_protocol_from_description` — given a string and
  an LLM client, return a ``BehaviorProtocolCandidate``.
"""

from __future__ import annotations

from lifeform_protocol_runtime.task_description_uptake.extraction import (
    extract_protocol_from_description,
)

__all__ = ["extract_protocol_from_description"]
