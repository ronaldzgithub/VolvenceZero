"""Reviewed historical-figure profile schema.

This package deliberately starts from reviewed, structured artifacts.
It does NOT inspect arbitrary primary-source text to decide behaviour:
extraction can be done by a separate LLM structured-output pass or by
human review, then this module compiles the result into the
``FigureArtifactBundle`` that downstream owners consume.

Why this is its own schema (not an extension of ``CharacterSoulProfile``):

* Real-person profiles encode ``version_window`` (early- vs late-figure)
  because positions can shift over decades; fictional characters
  generally do not need that.
* ``evidence_strength`` is mandatory rather than optional — every
  reviewed claim must declare whether it rests on first-hand writing,
  second-hand correspondence, or contemporaneous witness.
* ``domain_coverage_seed`` is the explicit "what this figure wrote
  about" prior. The figure vertical's L4 not-known refusal contract
  reads this seed plus the corpus coverage map; fictional characters
  do not have a comparable epistemic edge.
* ``HistoricalFigureProfile`` does NOT inherit / wrap
  ``CharacterSoulProfile`` because the two schemas evolve at
  different rates and cross-coupling would force shared churn on
  unrelated changes (R8).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureKnowledgeSeed:
    """Reviewed factual / value / world-model statement for the figure.

    Mirrors :class:`lifeform_domain_character.CharacterKnowledgeSeed`
    but adds a mandatory ``evidence_strength`` because every claim
    about a real person must declare its evidentiary basis.
    """

    seed_id: str
    domain: str
    title: str
    summary: str
    snippet: str
    evidence_locator: str
    confidence: float
    evidence_strength: str
    topic_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FigureSignatureCase:
    """A documented situation-position-outcome pattern from primary sources."""

    case_id: str
    domain: str
    problem_pattern: str
    user_state_pattern: str
    risk_markers: tuple[str, ...]
    track_tags: tuple[str, ...]
    regime_tags: tuple[str, ...]
    intervention_ordering: tuple[str, ...]
    outcome_label: str
    description: str
    confidence: float
    relevance_score: float = 0.75
    escalation_observed: bool = False
    repair_observed: bool = False


@dataclass(frozen=True)
class FigureStrategyPrior:
    """A reviewed pacing or ordering prior derived from multiple cases."""

    rule_id: str
    problem_pattern: str
    recommended_regime: str | None
    recommended_ordering: tuple[str, ...]
    recommended_pacing: str
    avoid_patterns: tuple[str, ...]
    applicability_scope: tuple[str, ...]
    confidence: float
    description: str
    knowledge_weight_hint: float = 0.45
    experience_weight_hint: float = 0.65


@dataclass(frozen=True)
class FigureBoundaryPrior:
    """A reviewed boundary or anti-pattern for the figure vertical.

    Includes the L4-relevant ``out_of_scope_topics`` list so that the
    coverage map can lift them as hard refusal hints during compile,
    rather than the runtime owner reconstructing them from text.
    """

    boundary_id: str
    regime_id: str | None
    trigger_reasons: tuple[str, ...]
    answer_depth_limit_hint: str
    clarification_required: bool
    refer_out_required: bool
    blocked_topics: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    confidence: float
    description: str
    out_of_scope_topics: tuple[str, ...] = ()


@dataclass(frozen=True)
class FigureDrivePrior:
    """One drive channel for the figure's always-on pressure profile."""

    name: str
    target: float
    homeostatic_band: tuple[float, float]
    decay_per_tick: float
    pe_weight: float
    initial_level: float = 0.5
    recharge_per_turn: float = 0.0
    recharge_per_regime: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class TimeWindowedView:
    """A time-bounded view onto a figure's positions / drives.

    Real-person positions can shift across a lifetime (early Einstein
    accepted quanta, late Einstein resisted Copenhagen). Each window
    is a frozen reviewed artifact that pins drives / cases / strategies
    to a specific date range; the runtime selects the active window via
    DLaaS ``TemplateSpec.figure_time_window``.

    The default window is ``(0, 0)`` which means "no time-window
    selection" — the profile's own knowledge / case / strategy / boundary
    lists are used unchanged.
    """

    window_id: str
    year_start: int
    year_end: int
    description: str
    knowledge_seed_overrides: tuple[FigureKnowledgeSeed, ...] = ()
    signature_case_overrides: tuple[FigureSignatureCase, ...] = ()
    strategy_prior_overrides: tuple[FigureStrategyPrior, ...] = ()
    boundary_prior_overrides: tuple[FigureBoundaryPrior, ...] = ()
    drive_prior_overrides: tuple[FigureDrivePrior, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("TimeWindowedView.window_id", self.window_id)
        if self.year_end < self.year_start:
            raise ValueError(
                f"TimeWindowedView.year_end ({self.year_end}) must be >= "
                f"year_start ({self.year_start}) for window "
                f"{self.window_id!r}"
            )


@dataclass(frozen=True)
class HistoricalFigureProfile:
    """Reviewed artifact that compiles into the figure vertical bundle.

    Compared to :class:`lifeform_domain_character.CharacterSoulProfile`,
    every field carries explicit evidentiary or temporal scope so the
    L3 / L4 contract can be enforced without text heuristics.
    """

    profile_id: str
    figure_name: str
    figure_lifespan: tuple[int, int]
    version: str
    reviewed_by: str
    source_uri: str
    description: str
    domain_coverage_seed: tuple[str, ...]
    knowledge_seeds: tuple[FigureKnowledgeSeed, ...]
    signature_cases: tuple[FigureSignatureCase, ...]
    strategy_priors: tuple[FigureStrategyPrior, ...]
    boundary_priors: tuple[FigureBoundaryPrior, ...]
    drive_priors: tuple[FigureDrivePrior, ...] = ()
    time_windows: tuple[TimeWindowedView, ...] = ()
    target_contexts: tuple[str, ...] = ("figure-companion", "primary-source-grounded")

    def __post_init__(self) -> None:
        _require_non_empty("HistoricalFigureProfile.profile_id", self.profile_id)
        _require_non_empty("HistoricalFigureProfile.figure_name", self.figure_name)
        _require_non_empty("HistoricalFigureProfile.version", self.version)
        _require_non_empty("HistoricalFigureProfile.reviewed_by", self.reviewed_by)
        _require_non_empty("HistoricalFigureProfile.source_uri", self.source_uri)
        if not self.boundary_priors:
            raise ValueError(
                "HistoricalFigureProfile.boundary_priors must be non-empty: "
                "every figure must declare at least one explicit boundary"
            )
        if not self.domain_coverage_seed:
            raise ValueError(
                "HistoricalFigureProfile.domain_coverage_seed must be "
                "non-empty: the L4 not-known refusal contract reads this "
                "seed at compile time"
            )
        born, died = self.figure_lifespan
        if died < born:
            raise ValueError(
                f"HistoricalFigureProfile.figure_lifespan invalid: "
                f"{self.figure_lifespan!r}"
            )
        _check_unique(
            "knowledge_seeds.seed_id",
            tuple(seed.seed_id for seed in self.knowledge_seeds),
        )
        _check_unique(
            "signature_cases.case_id",
            tuple(case.case_id for case in self.signature_cases),
        )
        _check_unique(
            "strategy_priors.rule_id",
            tuple(rule.rule_id for rule in self.strategy_priors),
        )
        _check_unique(
            "boundary_priors.boundary_id",
            tuple(boundary.boundary_id for boundary in self.boundary_priors),
        )
        _check_unique(
            "drive_priors.name",
            tuple(drive.name for drive in self.drive_priors),
        )
        _check_unique(
            "time_windows.window_id",
            tuple(view.window_id for view in self.time_windows),
        )

    def select_window(self, window_id: str | None) -> "HistoricalFigureProfile":
        """Return a profile view with the named time window applied.

        Returns ``self`` unchanged when ``window_id`` is ``None`` or
        empty. Otherwise, locates the matching :class:`TimeWindowedView`
        and produces a new frozen profile whose lists are the union of
        base entries with that view's overrides (overrides win on
        matching ids).
        """

        if not window_id:
            return self
        matches = tuple(view for view in self.time_windows if view.window_id == window_id)
        if not matches:
            raise ValueError(
                f"HistoricalFigureProfile.select_window: window_id "
                f"{window_id!r} not found among {[v.window_id for v in self.time_windows]!r}"
            )
        view = matches[0]
        return HistoricalFigureProfile(
            profile_id=self.profile_id,
            figure_name=self.figure_name,
            figure_lifespan=self.figure_lifespan,
            version=f"{self.version}+window:{window_id}",
            reviewed_by=self.reviewed_by,
            source_uri=self.source_uri,
            description=f"{self.description} [time window: {view.description}]",
            domain_coverage_seed=self.domain_coverage_seed,
            knowledge_seeds=_merge_by_id(
                self.knowledge_seeds, view.knowledge_seed_overrides, "seed_id"
            ),
            signature_cases=_merge_by_id(
                self.signature_cases, view.signature_case_overrides, "case_id"
            ),
            strategy_priors=_merge_by_id(
                self.strategy_priors, view.strategy_prior_overrides, "rule_id"
            ),
            boundary_priors=_merge_by_id(
                self.boundary_priors, view.boundary_prior_overrides, "boundary_id"
            ),
            drive_priors=_merge_by_id(
                self.drive_priors, view.drive_prior_overrides, "name"
            ),
            time_windows=self.time_windows,
            target_contexts=self.target_contexts,
        )


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _check_unique(field_name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique, got {values!r}")


def _merge_by_id(
    base: tuple, overrides: tuple, id_field: str
) -> tuple:
    """Merge ``overrides`` into ``base`` keyed on ``id_field``.

    Override entries replace base entries with the same id; remaining
    base entries pass through. Result preserves base order, with new
    override-only entries appended in their declared order.
    """

    overrides_by_id = {getattr(item, id_field): item for item in overrides}
    out = []
    seen: set[str] = set()
    for item in base:
        item_id = getattr(item, id_field)
        if item_id in overrides_by_id:
            out.append(overrides_by_id[item_id])
        else:
            out.append(item)
        seen.add(item_id)
    for item in overrides:
        item_id = getattr(item, id_field)
        if item_id not in seen:
            out.append(item)
    return tuple(out)


__all__ = [
    "FigureBoundaryPrior",
    "FigureDrivePrior",
    "FigureKnowledgeSeed",
    "FigureSignatureCase",
    "FigureStrategyPrior",
    "HistoricalFigureProfile",
    "TimeWindowedView",
]
