"""Family-memorial figure profile loader.

The family-memorial product (in VolvenceDeploy ``apps/family-memorial/``)
takes a small JSON descriptor — display name, relation, lifespan, free-
form bio — and bakes a per-memorial ``FigureArtifactBundle``. The
corpus itself is the family's uploaded interviews / letters / photos,
NOT a reviewer-curated primary source archive. So the
:class:`HistoricalFigureProfile` we feed to ``build_figure_artifact_bundle``
is **minimal-by-design**: a name, a lifespan, a non-empty boundary
prior declaring "out-of-scope topics", and otherwise empty
reviewer-curated lists. The actual factual content (knowledge seeds,
signature cases) is supplied through the curated corpus path at L1.

This keeps the L4 coverage map honest: family memorials have no
reviewer-curated "what this person wrote about" prior — they have
ONLY what the family uploaded. If the descendant asks about Wikipedia
trivia not in the family corpus, L4 must refuse. The boundary prior
encodes that contract.

Schema (the JSON file the bake-worker writes):

```json
{
  "figure_id": "family_<memorialId>",
  "display_name": "外公",
  "relation": "外公",
  "birth_year": 1925,
  "death_year": 2008,
  "bio": "Free-form short narrative used as L4 background context.",
  "time_window_id": "family-<memorialId>-1925-2008"
}
```

Returned profile has:

* ``profile_id = figure_id``
* ``figure_name = display_name``
* ``figure_lifespan = (birth_year, death_year)`` —
  birth_year defaults to ``death_year`` when omitted (an "unknown
  birth year" memorial still satisfies the ``died >= born`` check).
* ``boundary_priors`` — exactly one boundary, ``out_of_scope_topics``
  set to a small list of well-known categories the bake-worker should
  not let through (politics, religion, medical advice, etc.).
* ``domain_coverage_seed`` — ``("family_personal_narrative",)`` only;
  no Wikipedia-style breadth.
* ``time_windows`` — a single window matching the lifespan when both
  years are provided.

Everything else is empty by design — the corpus is where the
substance lives.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from lifeform_domain_figure.profile import (
    FigureBoundaryPrior,
    HistoricalFigureProfile,
    TimeWindowedView,
)


_DEFAULT_OUT_OF_SCOPE: tuple[str, ...] = (
    # Anything outside the family's own narrative; the bake worker is
    # not authorised to surface political / medical / financial /
    # legal / dangerous-advice content for a private memorial.
    "current-events",
    "political-commentary",
    "medical-advice",
    "legal-advice",
    "financial-advice",
    "dangerous-activity-instructions",
    # Hard product rule: memorials are for one specific deceased
    # person; impersonating other relatives is out of scope even when
    # those relatives appear by name in the corpus.
    "impersonating-living-people",
    "impersonating-other-deceased",
)


def build_family_profile_from_json(
    payload: dict[str, Any],
) -> HistoricalFigureProfile:
    """Build a minimal :class:`HistoricalFigureProfile` from a JSON dict.

    Validates required keys and produces a profile whose
    ``boundary_priors`` carry the family-memorial "out-of-scope"
    declaration. ``HistoricalFigureProfile.__post_init__`` performs
    the final structural validation; this loader only fills in the
    minimum.
    """

    figure_id = _required_str(payload, "figure_id")
    display_name = _required_str(payload, "display_name")
    death_year = _required_int(payload, "death_year")
    birth_year_raw = payload.get("birth_year")
    if birth_year_raw is None:
        birth_year = death_year
    else:
        birth_year = int(birth_year_raw)
    if death_year < birth_year:
        raise ValueError(
            "build_family_profile_from_json: death_year "
            f"({death_year}) must be >= birth_year ({birth_year})"
        )
    bio = str(payload.get("bio", "") or "")
    relation = str(payload.get("relation", "") or "family-member")
    time_window_id = str(payload.get("time_window_id", "") or "")
    out_of_scope = tuple(
        payload.get("out_of_scope_topics") or _DEFAULT_OUT_OF_SCOPE
    )
    if not out_of_scope:
        raise ValueError(
            "build_family_profile_from_json: out_of_scope_topics must "
            "be a non-empty tuple; family memorials require an explicit "
            "L4 boundary."
        )

    boundary = FigureBoundaryPrior(
        boundary_id=f"{figure_id}_family_scope_boundary",
        regime_id=None,
        trigger_reasons=(
            "question_outside_family_corpus",
            "request_outside_family_attested_domain",
        ),
        answer_depth_limit_hint="refuse",
        clarification_required=False,
        refer_out_required=False,
        blocked_topics=out_of_scope,
        required_disclaimers=(
            "answer-only-from-family-uploaded-materials",
        ),
        confidence=1.0,
        description=(
            "Family memorial boundary: the memorial may speak only "
            "about topics attested by the family's uploaded corpus. "
            "Anything else is L4 OUT_OF_DOMAIN."
        ),
        out_of_scope_topics=out_of_scope,
    )

    time_windows: tuple[TimeWindowedView, ...] = ()
    if time_window_id:
        time_windows = (
            TimeWindowedView(
                window_id=time_window_id,
                year_start=birth_year,
                year_end=death_year,
                description=f"Family memorial lifespan for {display_name}.",
            ),
        )

    return HistoricalFigureProfile(
        profile_id=figure_id,
        figure_name=display_name,
        figure_lifespan=(birth_year, death_year),
        version="0.1.0",
        reviewed_by="family-attestation",
        source_uri=f"family-memorial://{figure_id}",
        description=(
            bio
            or f"Family memorial for {display_name} ({relation}); corpus "
            f"is family-attested."
        ),
        # L4 reads ``domain_coverage_seed`` at compile time. We
        # deliberately list a SINGLE narrow category so the coverage
        # map does not grant accidental breadth.
        domain_coverage_seed=("family_personal_narrative",),
        knowledge_seeds=(),
        signature_cases=(),
        strategy_priors=(),
        boundary_priors=(boundary,),
        drive_priors=(),
        time_windows=time_windows,
        target_contexts=("figure-companion", "family-memorial"),
    )


def load_family_profile_file(path: str | pathlib.Path) -> HistoricalFigureProfile:
    """Load + parse a family-memorial profile JSON file.

    The bake-worker writes the JSON next to the cleaning store; the
    CLI reads it via the ``--profile-file`` flag. We deliberately
    accept either a single profile object or a one-entry list (the
    bake-worker could write either; downstream consumers should not
    have to guess).
    """

    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"load_family_profile_file: {p} does not exist or is not a file"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                f"load_family_profile_file: {p} contains a list with "
                f"len={len(raw)}; family-memorial profile files must "
                "carry exactly one profile."
            )
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError(
            f"load_family_profile_file: {p} top-level must be an object."
        )
    return build_family_profile_from_json(raw)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"build_family_profile_from_json: required string field "
            f"{key!r} is missing or empty."
        )
    return value.strip()


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise ValueError(
            f"build_family_profile_from_json: required field {key!r} "
            "is missing."
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"build_family_profile_from_json: field {key!r} must be an int."
        ) from exc


__all__ = (
    "build_family_profile_from_json",
    "load_family_profile_file",
)
