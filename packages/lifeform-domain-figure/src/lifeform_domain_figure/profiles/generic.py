"""Generic persona profile loader (D-myriad-1 / D-MY-1 closure).

Any persona slug — Myriad historical figures (``libai``, ``dufu``),
digital-employee personas, future operator-defined figures — can be
baked from a single JSON descriptor without authoring a reviewer-
curated Python profile per persona. The descriptor schema is the one
the Myriad app already ships under ``apps/myriad/seed/figures/<slug>/
profile.json`` in VolvenceDeploy, so the deploy-side persona bake
worker can mount that directory unchanged.

Schema (JSON object; ``slug`` + ``display_name`` are required, the
rest is optional with documented mapping)::

    {
      "slug": "libai",                          # REQUIRED, path-safe id
      "display_name": "李白",                    # REQUIRED
      "courtesy_name": "太白",                   # optional, folded into description
      "era": "tang",                            # optional, domain seed + description
      "vocation": "poet",                       # optional, domain seed + description
      "lifespan": "701-762",                    # optional "BORN-DIED"; omit -> (0, 0)
      "regime_tags": ["scholarly", "artistic"], # optional, domain seeds + case scope
      "recommended_regime": "scholarly",        # optional, strategy prior regime
      "domain_coverage_seed": ["tang_poetry"],  # optional explicit override of the
                                                # derived (vocation + era + regime_tags)
                                                # domain seeds
      "drive_priors": {"freedom": 0.85},        # optional name -> target in [0, 1]
      "signature_cases": [                      # optional; each REQUIRES id + summary
        {"id": "drunken-verse", "summary": "...", "regime_tags": ["artistic"]}
      ],
      "boundary_priors": {                      # REQUIRED with non-empty topics: the
        "out_of_scope_topics": ["modern technology"]  # L4 contract needs an honest
      },                                        # boundary for every persona
      "style_prior": {                          # optional, compiled into one strategy
        "register": "literary",                 # prior whose description carries the
        "verbosity": "concise",                 # terms into the synthetic corpus ->
        "imagery": "moon, wine, river"          # style prior pipeline
      },
      "evidence_sources": [                     # optional; each REQUIRES title
        {"title": "《李太白集》", "provenance": "Public-domain."}
      ],
      "time_windows": [                         # entries WITHOUT integer year_start +
        {"slug": "youth", "label": "青年游侠"}    # year_end are presentation metadata
      ]                                         # only and are NOT compiled into
    }                                           # TimeWindowedView (no fabricated years)

Mapping decisions (documented, deliberate):

* ``domain_coverage_seed`` derives from explicit override, else
  ``vocation`` + ``era`` + ``regime_tags``. At least one must be
  present — the L4 coverage map refuses to build without an
  in-domain centroid.
* ``voice_hint`` is presence-plane metadata and is ignored here
  (the figure profile owns cognition / boundary state, not TTS).
* Myriad ``time_windows`` entries carry only ``slug`` + ``label``;
  ``TimeWindowedView`` requires integer years. Entries that also
  declare ``year_start`` / ``year_end`` compile into windows;
  label-only entries are skipped (we refuse to fabricate years).
* ``evidence_sources`` become ``FigureKnowledgeSeed`` records with
  ``evidence_strength="secondary"`` — they are catalogue pointers,
  not reviewed primary-source claims.

Unlike :mod:`lifeform_domain_figure.profiles.myriad` (which requires
the ``myriad_<slug>`` prefix and its own minimal schema), this loader
accepts bare slugs and the richer Myriad seed schema; both coexist —
the CLI routes ``myriad_*`` / ``family_*`` ids to their dedicated
loaders and everything else here.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from lifeform_domain_figure.profile import (
    FigureBoundaryPrior,
    FigureDrivePrior,
    FigureKnowledgeSeed,
    FigureSignatureCase,
    FigureStrategyPrior,
    HistoricalFigureProfile,
    TimeWindowedView,
)
from lifeform_domain_figure.profiles.einstein import build_einstein_profile
from lifeform_domain_figure.profiles.family import load_family_profile_file
from lifeform_domain_figure.profiles.lu_xun import build_lu_xun_profile
from lifeform_domain_figure.profiles.myriad import load_myriad_profile_file


# Slugs become on-disk bundle directory names (<root>/<figure_id>/...),
# so the character set is restricted to path-safe identifier characters.
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_LIFESPAN_RE = re.compile(r"^\s*(\d{1,4})\s*-\s*(\d{1,4})\s*$")

# Built-in reviewer-curated profiles ship as Python factories; every
# other slug requires a JSON descriptor.
_BUILTIN_PROFILE_FACTORIES = {
    "einstein": build_einstein_profile,
    "lu_xun": build_lu_xun_profile,
}


def build_generic_profile_from_json(
    payload: dict[str, Any],
    *,
    expected_slug: str | None = None,
) -> HistoricalFigureProfile:
    """Build a :class:`HistoricalFigureProfile` from a Myriad-style dict.

    Fails loudly (``ValueError``) on missing / malformed required
    fields; final structural validation is delegated to
    :meth:`HistoricalFigureProfile.__post_init__`.
    """

    slug = _required_str(payload, "slug")
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"build_generic_profile_from_json: slug {slug!r} is not a "
            "path-safe identifier (expected [A-Za-z0-9][A-Za-z0-9_-]*)."
        )
    if expected_slug is not None and slug != expected_slug:
        raise ValueError(
            f"build_generic_profile_from_json: profile declares slug "
            f"{slug!r} but the caller expected {expected_slug!r}; the "
            "profile file and the requested figure id must stay in "
            "lock-step."
        )
    display_name = _required_str(payload, "display_name")
    courtesy_name = _optional_str(payload, "courtesy_name")
    era = _optional_str(payload, "era")
    vocation = _optional_str(payload, "vocation")
    lifespan = _parse_lifespan(payload.get("lifespan"), slug=slug)
    regime_tags = _str_tuple(payload.get("regime_tags"), field="regime_tags")
    recommended_regime = _optional_str(payload, "recommended_regime")

    domain_coverage = _domain_coverage(
        payload, slug=slug, vocation=vocation, era=era, regime_tags=regime_tags
    )
    primary_domain = domain_coverage[0]

    out_of_scope = _required_out_of_scope(payload, slug=slug)
    boundary = FigureBoundaryPrior(
        boundary_id=f"{slug}-persona-scope-boundary",
        regime_id=None,
        trigger_reasons=(
            "question_outside_configured_domain",
            "request_outside_persona_documented_coverage",
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
            f"Generic persona boundary for {display_name}: the persona "
            "may speak only about its configured domain coverage and "
            "curated corpus. Anything else is L4 OUT_OF_DOMAIN."
        ),
        out_of_scope_topics=out_of_scope,
    )

    return HistoricalFigureProfile(
        profile_id=slug,
        figure_name=display_name,
        figure_lifespan=lifespan,
        version=_optional_str(payload, "version") or "0.1.0",
        reviewed_by="generic-persona-profile-json",
        source_uri=f"persona://{slug}",
        description=_build_description(
            payload,
            display_name=display_name,
            courtesy_name=courtesy_name,
            era=era,
            vocation=vocation,
            lifespan=lifespan,
        ),
        domain_coverage_seed=domain_coverage,
        knowledge_seeds=_knowledge_seeds(
            payload, slug=slug, primary_domain=primary_domain
        ),
        signature_cases=_signature_cases(
            payload,
            slug=slug,
            primary_domain=primary_domain,
            regime_tags=regime_tags,
        ),
        strategy_priors=_strategy_priors(
            payload,
            slug=slug,
            primary_domain=primary_domain,
            regime_tags=regime_tags,
            recommended_regime=recommended_regime,
        ),
        boundary_priors=(boundary,),
        drive_priors=_drive_priors(payload, slug=slug),
        time_windows=_time_windows(payload, slug=slug, display_name=display_name),
        target_contexts=("figure-companion", "generic-persona"),
    )


def load_generic_profile_file(
    path: str | pathlib.Path,
    *,
    expected_slug: str | None = None,
) -> HistoricalFigureProfile:
    """Load + parse a generic persona profile JSON file."""

    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"load_generic_profile_file: {p} does not exist or is not a file"
        )
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"load_generic_profile_file: {p} top-level must be an object."
        )
    return build_generic_profile_from_json(raw, expected_slug=expected_slug)


def load_profile(
    slug: str,
    profile_path: str | pathlib.Path | None = None,
) -> HistoricalFigureProfile:
    """Resolve a figure slug to a :class:`HistoricalFigureProfile`.

    Resolution order:

    1. Built-in reviewer-curated slugs (``einstein`` / ``lu_xun``)
       come from their shipped Python factories; a ``profile_path``
       is rejected so a stray JSON can never shadow the reviewed
       profile.
    2. ``family_*`` / ``myriad_*`` ids route to their dedicated
       JSON loaders (schemas differ from the generic one).
    3. Every other slug requires ``profile_path`` pointing at a
       generic persona JSON whose ``slug`` matches.
    """

    if not slug.strip():
        raise ValueError("load_profile: slug must be non-empty")
    builtin = _BUILTIN_PROFILE_FACTORIES.get(slug)
    if builtin is not None:
        if profile_path is not None:
            raise ValueError(
                f"load_profile: slug {slug!r} is a built-in reviewer-"
                "curated profile; refusing to override it with a JSON "
                f"file ({profile_path})."
            )
        return builtin()
    if profile_path is None:
        raise ValueError(
            f"load_profile: slug {slug!r} is not a built-in profile; "
            "profile_path to a persona JSON descriptor is required."
        )
    if slug.startswith("family_"):
        profile = load_family_profile_file(profile_path)
    elif slug.startswith("myriad_"):
        profile = load_myriad_profile_file(profile_path)
    else:
        return load_generic_profile_file(profile_path, expected_slug=slug)
    if profile.profile_id != slug:
        raise ValueError(
            f"load_profile: profile file {profile_path} declares "
            f"profile_id {profile.profile_id!r} but the caller asked "
            f"for {slug!r}; the two must stay in lock-step."
        )
    return profile


# ---------------------------------------------------------------------------
# Field mappers
# ---------------------------------------------------------------------------


def _build_description(
    payload: dict[str, Any],
    *,
    display_name: str,
    courtesy_name: str,
    era: str,
    vocation: str,
    lifespan: tuple[int, int],
) -> str:
    explicit = _optional_str(payload, "bio") or _optional_str(
        payload, "description"
    )
    parts: list[str] = []
    headline = display_name
    if courtesy_name:
        headline = f"{display_name} (courtesy name {courtesy_name})"
    qualifiers: list[str] = []
    if vocation:
        qualifiers.append(vocation)
    if era:
        qualifiers.append(f"of the {era} era")
    if lifespan != (0, 0):
        qualifiers.append(f"lifespan {lifespan[0]}-{lifespan[1]}")
    if qualifiers:
        parts.append(f"{headline}, {', '.join(qualifiers)}.")
    else:
        parts.append(f"{headline}.")
    if explicit:
        parts.append(explicit)
    parts.append(
        "Generic persona profile compiled from an operator-curated JSON "
        "descriptor; factual substance arrives through the corpus path."
    )
    return " ".join(parts)


def _domain_coverage(
    payload: dict[str, Any],
    *,
    slug: str,
    vocation: str,
    era: str,
    regime_tags: tuple[str, ...],
) -> tuple[str, ...]:
    explicit = _str_tuple(
        payload.get("domain_coverage_seed"), field="domain_coverage_seed"
    )
    ordered: list[str] = list(explicit)
    for candidate in (vocation, era, *regime_tags):
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    if not ordered:
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} declares "
            "no domain_coverage_seed, vocation, era, or regime_tags; the "
            "L4 coverage map requires at least one in-domain seed."
        )
    return tuple(ordered)


def _required_out_of_scope(
    payload: dict[str, Any], *, slug: str
) -> tuple[str, ...]:
    boundary_raw = payload.get("boundary_priors")
    if not isinstance(boundary_raw, dict):
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} is "
            "missing the required 'boundary_priors' object; every "
            "persona must declare an explicit L4 boundary."
        )
    topics = _str_tuple(
        boundary_raw.get("out_of_scope_topics"),
        field="boundary_priors.out_of_scope_topics",
    )
    if not topics:
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} declares "
            "no boundary_priors.out_of_scope_topics; the list must be "
            "non-empty (L4 requires an honest boundary)."
        )
    return topics


def _knowledge_seeds(
    payload: dict[str, Any],
    *,
    slug: str,
    primary_domain: str,
) -> tuple[FigureKnowledgeSeed, ...]:
    raw = payload.get("evidence_sources")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} field "
            "'evidence_sources' must be a list of objects."
        )
    seeds: list[FigureKnowledgeSeed] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"build_generic_profile_from_json: evidence_sources[{index}] "
                f"for persona {slug!r} must be an object."
            )
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(
                f"build_generic_profile_from_json: evidence_sources[{index}] "
                f"for persona {slug!r} is missing the required 'title'."
            )
        provenance = str(entry.get("provenance", "") or "").strip()
        seeds.append(
            FigureKnowledgeSeed(
                seed_id=f"evidence-{index:02d}",
                domain=primary_domain,
                title=title.strip(),
                summary=provenance or f"Catalogued evidence source: {title.strip()}.",
                snippet=title.strip(),
                evidence_locator=f"profile:{slug}:evidence:{index:02d}",
                confidence=0.7,
                # Catalogue pointers, not reviewed primary-source
                # claims — see module docstring mapping decisions.
                evidence_strength="secondary",
                topic_tags=("evidence-catalogue",),
            )
        )
    return tuple(seeds)


def _signature_cases(
    payload: dict[str, Any],
    *,
    slug: str,
    primary_domain: str,
    regime_tags: tuple[str, ...],
) -> tuple[FigureSignatureCase, ...]:
    raw = payload.get("signature_cases")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} field "
            "'signature_cases' must be a list of objects."
        )
    cases: list[FigureSignatureCase] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"build_generic_profile_from_json: signature_cases[{index}] "
                f"for persona {slug!r} must be an object."
            )
        case_id = entry.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(
                f"build_generic_profile_from_json: signature_cases[{index}] "
                f"for persona {slug!r} is missing the required 'id'."
            )
        summary = entry.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                f"build_generic_profile_from_json: signature_cases[{index}] "
                f"({case_id!r}) for persona {slug!r} is missing the "
                "required 'summary'."
            )
        case_regimes = _str_tuple(
            entry.get("regime_tags"), field=f"signature_cases[{index}].regime_tags"
        ) or regime_tags or ("general",)
        cases.append(
            FigureSignatureCase(
                case_id=case_id.strip(),
                domain=primary_domain,
                problem_pattern=f"persona-signature:{case_id.strip()}",
                user_state_pattern="dialogue-engages-signature-theme",
                risk_markers=("risk-low",),
                track_tags=("self",),
                regime_tags=case_regimes,
                intervention_ordering=(
                    "recall_documented_pattern",
                    "respond_in_persona_voice",
                ),
                outcome_label="stable",
                description=summary.strip(),
                confidence=0.8,
                relevance_score=0.8,
            )
        )
    return tuple(cases)


def _strategy_priors(
    payload: dict[str, Any],
    *,
    slug: str,
    primary_domain: str,
    regime_tags: tuple[str, ...],
    recommended_regime: str,
) -> tuple[FigureStrategyPrior, ...]:
    style_raw = payload.get("style_prior")
    if style_raw is None and not recommended_regime:
        return ()
    if style_raw is not None and not isinstance(style_raw, dict):
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} field "
            "'style_prior' must be an object."
        )
    style = style_raw or {}
    register = str(style.get("register", "") or "").strip()
    verbosity = str(style.get("verbosity", "") or "").strip()
    imagery = str(style.get("imagery", "") or "").strip()
    description_bits = ["Respond in the persona's documented register"]
    if register:
        description_bits.append(f"({register})")
    if imagery:
        description_bits.append(f"drawing on documented imagery: {imagery}")
    description = " ".join(description_bits) + "."
    return (
        FigureStrategyPrior(
            rule_id=f"{slug}-style-register",
            problem_pattern="general-persona-dialogue",
            recommended_regime=recommended_regime or None,
            recommended_ordering=(
                "ground_in_documented_material",
                "answer_in_persona_register",
            ),
            recommended_pacing=verbosity or "measured",
            avoid_patterns=(
                "invent-undocumented-biography",
                "speak-outside-coverage",
            ),
            applicability_scope=regime_tags or (primary_domain,),
            confidence=0.7,
            description=description,
        ),
    )


def _drive_priors(
    payload: dict[str, Any], *, slug: str
) -> tuple[FigureDrivePrior, ...]:
    raw = payload.get("drive_priors")
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} field "
            "'drive_priors' must be an object of name -> target."
        )
    drives: list[FigureDrivePrior] = []
    for name, target_raw in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"build_generic_profile_from_json: drive_priors for persona "
                f"{slug!r} contains an empty drive name."
            )
        try:
            target = float(target_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"build_generic_profile_from_json: drive_priors[{name!r}] "
                f"for persona {slug!r} must be a number, got {target_raw!r}."
            ) from exc
        if not 0.0 <= target <= 1.0:
            raise ValueError(
                f"build_generic_profile_from_json: drive_priors[{name!r}] "
                f"for persona {slug!r} must be in [0, 1], got {target!r}."
            )
        drives.append(
            FigureDrivePrior(
                name=name.strip(),
                target=target,
                homeostatic_band=(
                    round(max(0.0, target - 0.2), 6),
                    round(min(1.0, target + 0.1), 6),
                ),
                decay_per_tick=0.003,
                pe_weight=0.8,
                initial_level=target,
                recharge_per_turn=0.01,
            )
        )
    return tuple(drives)


def _time_windows(
    payload: dict[str, Any], *, slug: str, display_name: str
) -> tuple[TimeWindowedView, ...]:
    raw = payload.get("time_windows")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} field "
            "'time_windows' must be a list of objects."
        )
    windows: list[TimeWindowedView] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"build_generic_profile_from_json: time_windows[{index}] "
                f"for persona {slug!r} must be an object."
            )
        year_start = entry.get("year_start")
        year_end = entry.get("year_end")
        if year_start is None or year_end is None:
            # Label-only window (Myriad presentation metadata): no
            # integer years -> not compiled; we refuse to fabricate
            # a date range. See module docstring mapping decisions.
            continue
        window_id = entry.get("slug")
        if not isinstance(window_id, str) or not window_id.strip():
            raise ValueError(
                f"build_generic_profile_from_json: time_windows[{index}] "
                f"for persona {slug!r} declares years but no 'slug'."
            )
        label = str(entry.get("label", "") or "").strip()
        windows.append(
            TimeWindowedView(
                window_id=window_id.strip(),
                year_start=int(year_start),
                year_end=int(year_end),
                description=label or f"Time window {window_id} for {display_name}.",
            )
        )
    return tuple(windows)


def _parse_lifespan(
    raw: Any, *, slug: str
) -> tuple[int, int]:
    if raw is None:
        return (0, 0)
    if not isinstance(raw, str):
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} field "
            f"'lifespan' must be a 'BORN-DIED' string, got {raw!r}."
        )
    match = _LIFESPAN_RE.match(raw)
    if match is None:
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} lifespan "
            f"{raw!r} is malformed; expected 'BORN-DIED' (e.g. '701-762')."
        )
    born, died = int(match.group(1)), int(match.group(2))
    if died < born:
        raise ValueError(
            f"build_generic_profile_from_json: persona {slug!r} lifespan "
            f"{raw!r} has died < born."
        )
    return (born, died)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"build_generic_profile_from_json: required string field "
            f"{key!r} is missing or empty."
        )
    return value.strip()


def _optional_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            f"build_generic_profile_from_json: field {key!r} must be a "
            f"string when present, got {type(value).__name__}."
        )
    return value.strip()


def _str_tuple(raw: Any, *, field: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item.strip() for item in raw
    ):
        raise ValueError(
            f"build_generic_profile_from_json: field {field!r} must be a "
            "list of non-empty strings when present."
        )
    return tuple(item.strip() for item in raw)


__all__ = (
    "build_generic_profile_from_json",
    "load_generic_profile_file",
    "load_profile",
)
