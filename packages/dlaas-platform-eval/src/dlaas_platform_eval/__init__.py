"""DLaaS platform-tier eval gate (Slice 6).

Public surface:

* :func:`attach_eval_routes` — wires every audience / exam / license
  endpoint onto an aiohttp app.
* :class:`DefaultRubricGrader` — deterministic fail-closed scorer
  used when no LLM judge backend is configured. Returns a flat
  rubric breakdown; aggregate falls below the pass threshold by
  default so license gates fail safe until a real grader is wired.

Slice 6 strictly readouts: no learning signal flows back from these
endpoints into the kernel. The license gate blocks
``template.status → published`` but never mutates kernel state.
"""

from __future__ import annotations

from dlaas_platform_eval.grader import (
    DefaultRubricGrader,
    GradedSubmission,
    RubricGrader,
)
from dlaas_platform_eval.routes import EVAL_BUNDLE_APP_KEY, attach_eval_routes

__all__ = (
    "DefaultRubricGrader",
    "EVAL_BUNDLE_APP_KEY",
    "GradedSubmission",
    "RubricGrader",
    "attach_eval_routes",
)
