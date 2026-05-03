"""Dialogue benchmark facade.

Slice A.1 (2026-05-02): the previously single ~8k-line
``volvence_zero.agent.dialogue_benchmark`` module is moved into this
subpackage as a self-contained ``_legacy`` module. This package exposes
the same public API surface so external imports keep working through
the new ``volvence_zero.agent.dialogue`` path.

Subsequent slices (A.2 ... A.6) will incrementally split ``_legacy``
into sibling files by capability domain:

* ``types``        — dataclass / Enum / Protocol contract surface
* ``simulators``   — DeterministicUserSimulator / TranscriptOnlyUserSimulator
                      / OpenDialogueREPLReader and their builders
* ``scenarios``    — proof cases, open scenarios, case variants,
                      paraphrase families, paper-suite manifest config
* ``case_reports`` — build_dialogue_case_report / build_open_dialogue_case_report
                      and their per-turn / per-window helpers
* ``runners``      — build_standard_dialogue_runner / async run_*_benchmark
* ``paper_suite``  — expert-review packets, human-rating aggregation,
                      evidence bundle, comprehensive checkpointing

Until those slices land, ``_legacy`` is the single source of truth and
this ``__init__`` simply re-exports it without modification.
"""

from __future__ import annotations

from volvence_zero.agent.dialogue._legacy import *  # noqa: F401,F403

# Private helpers that external callers (notably
# ``volvence_zero.agent.__init__`` and the dialogue benchmark test
# suite) re-export by their underscore-prefixed name. ``import *`` skips
# these by Python convention so they are imported explicitly.
from volvence_zero.agent.dialogue._legacy import (  # noqa: F401
    _case_summary_metrics,
    _empty_emergence_dashboard,
    _open_case_summary_metrics,
)
