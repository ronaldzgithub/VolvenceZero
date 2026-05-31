"""Myriad dynamic figure-profile loader (D-myriad-1).

The ``myriad`` product (VolvenceDeploy ``apps/myriad/``) lets an operator
spin up *many* lightweight figure companions from a small JSON descriptor
without authoring a reviewer-curated Python profile per persona. Unlike
:mod:`lifeform_domain_figure.profiles.family` — whose corpus is a private
family archive and whose boundary refuses everything outside the family
narrative — a myriad persona is a **generic configurable figure**: the
descriptor carries the persona's display name, an optional lifespan, the
domain-coverage seeds that scope what it may speak about, and an explicit
out-of-scope boundary. The factual substance still arrives through the
curated-corpus path at L1; the profile is minimal-by-design so the L4
coverage map stays honest.

This loader is the bake-CLI counterpart to the reviewer-curated
``einstein`` / ``lu_xun`` factories: where those ship as code, a myriad
persona is loaded at bake time from a JSON file the operator (or the
myriad app) writes. ``profile_id`` MUST be ``myriad_<slug>`` so the CLI
``--figure myriad_<slug>`` and the on-disk profile stay in lock-step.

Schema (the JSON file the operator / myriad app writes)::

    {
      "figure_id": "myriad_<slug>",
      "display_name": "Marie Curie",
      "birth_year": 1867,
      "death_year": 1934,
      "bio": "Free-form short narrative used as L4 background context.",
      "domain_coverage_seed": ["radioactivity", "chemistry"],
      "out_of_scope_topics": ["contemporary-politics", "medical-advice"],
      "time_window_id": "myriad-<slug>-1867-1934"
    }

``birth_year`` / ``death_year`` / ``time_window_id`` are optional — a
myriad persona need not be a deceased real person, so a living or wholly
fictional configuration may omit the lifespan. When both years are given
the loader emits a single :class:`TimeWindowedView`. ``out_of_scope_topics``
defaults to a conservative generic block-list when omitted.
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


_PROFILE_PREFIX = "myriad_"

_DEFAULT_OUT_OF_SCOPE: tuple[str, ...] = (
    # Generic guardrails for an operator-configured persona; the operator
    # can override with an explicit list, but it can never be empty (L4
    # requires an honest boundary).
    "current-events",
    "political-commentary",
    "medical-advice",
    "legal-advice",
    "financial-advice",
    "dangerous-activity-instructions",
    "impersonating-living-people",
)

_DEFAULT_DOMAIN_COVERAGE: tuple[str, ...] = ("myriad_persona_narrative",)


def build_myriad_profile_from_json(
    payload: dict[str, Any],
) -> HistoricalFigureProfile:
    """Build a minimal :class:`HistoricalFigureProfile` from a JSON dict.

    Validates required keys, then produces a profile whose
    ``boundary_priors`` carry the persona's out-of-scope declaration and
    whose ``domain_coverage_seed`` scopes the L4 coverage map. Final
    structural validation is delegated to
    :meth:`HistoricalFigureProfile.__post_init__`.
    """

    figure_id = _required_str(payload, "figure_id")
    if not figure_id.startswith(_PROFILE_PREFIX):
        raise ValueError(
            f"build_myriad_profile_from_json: figure_id {figure_id!r} must "
            f"start with {_PROFILE_PREFIX!r}; the CLI gates on this prefix "
            "to route myriad personas through this loader."
        )
    display_name = _required_str(payload, "display_name")

    death_year_raw = payload.get("death_year")
    birth_year_raw = payload.get("birth_year")
    # ``HistoricalFigureProfile.figure_lifespan`` is a required (int, int)
    # pair. A myriad persona need not be a deceased real person, so when
    # the descriptor omits the lifespan we use the ``(0, 0)`` sentinel
    # ("unspecified window") which satisfies the ``died >= born`` check
    # and produces no ``TimeWindowedView``.
    lifespan: tuple[int, int]
    time_windows: tuple[TimeWindowedView, ...] = ()
    if death_year_raw is None and birth_year_raw is None:
        lifespan = (0, 0)
    else:
        death_year = (
            int(death_year_raw)
            if death_year_raw is not None
            else int(birth_year_raw)  # type: ignore[arg-type]
        )
        birth_year = (
            int(birth_year_raw)
            if birth_year_raw is not None
            else death_year
        )
        if death_year < birth_year:
            raise ValueError(
                "build_myriad_profile_from_json: death_year "
                f"({death_year}) must be >= birth_year ({birth_year})"
            )
        lifespan = (birth_year, death_year)
        time_window_id = str(payload.get("time_window_id", "") or "")
        if time_window_id:
            time_windows = (
                TimeWindowedView(
                    window_id=time_window_id,
                    year_start=birth_year,
                    year_end=death_year,
                    description=(
                        f"Myriad persona active window for {display_name}."
                    ),
                ),
            )

    bio = str(payload.get("bio", "") or "")
    out_of_scope = tuple(
        payload.get("out_of_scope_topics") or _DEFAULT_OUT_OF_SCOPE
    )
    if not out_of_scope:
        raise ValueError(
            "build_myriad_profile_from_json: out_of_scope_topics must be a "
            "non-empty tuple; a myriad persona requires an explicit L4 "
            "boundary."
        )
    domain_coverage = tuple(
        payload.get("domain_coverage_seed") or _DEFAULT_DOMAIN_COVERAGE
    )
    if not domain_coverage:
        raise ValueError(
            "build_myriad_profile_from_json: domain_coverage_seed must be a "
            "non-empty tuple."
        )

    boundary = FigureBoundaryPrior(
        boundary_id=f"{figure_id}_myriad_scope_boundary",
        regime_id=None,
        trigger_reasons=(
            "question_outside_configured_domain",
            "request_outside_persona_attested_domain",
        ),
        answer_depth_limit_hint="refuse",
        clarification_required=False,
        refer_out_required=False,
        blocked_topics=out_of_scope,
        required_disclaimers=(
            "answer-only-within-configured-persona-scope",
        ),
        confidence=1.0,
        description=(
            "Myriad persona boundary: the persona may speak only about its "
            "configured domain-coverage seeds and the curated corpus. "
            "Anything else is L4 OUT_OF_DOMAIN."
        ),
        out_of_scope_topics=out_of_scope,
    )

    return HistoricalFigureProfile(
        profile_id=figure_id,
        figure_name=display_name,
        figure_lifespan=lifespan,
        version="0.1.0",
        reviewed_by="myriad-operator-attestation",
        source_uri=f"myriad://{figure_id}",
        description=(
            bio
            or f"Myriad persona {display_name}; corpus is operator-curated."
        ),
        domain_coverage_seed=domain_coverage,
        knowledge_seeds=(),
        signature_cases=(),
        strategy_priors=(),
        boundary_priors=(boundary,),
        drive_priors=(),
        time_windows=time_windows,
        target_contexts=("figure-companion", "myriad"),
    )


def load_myriad_profile_file(
    path: str | pathlib.Path,
) -> HistoricalFigureProfile:
    """Load + parse a myriad persona profile JSON file.

    Accepts either a single profile object or a one-entry list (the
    operator tool could write either), mirroring
    :func:`lifeform_domain_figure.profiles.family.load_family_profile_file`.
    """

    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"load_myriad_profile_file: {p} does not exist or is not a file"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                f"load_myriad_profile_file: {p} contains a list with "
                f"len={len(raw)}; myriad persona profile files must carry "
                "exactly one profile."
            )
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError(
            f"load_myriad_profile_file: {p} top-level must be an object."
        )
    return build_myriad_profile_from_json(raw)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"build_myriad_profile_from_json: required string field "
            f"{key!r} is missing or empty."
        )
    return value.strip()


__all__ = (
    "build_myriad_profile_from_json",
    "load_myriad_profile_file",
)
