"""ScopeRefuser — L4 not-known refusal enforcement.

Reads :class:`lifeform_domain_figure.FigureCoverageMap.classify_query`
(via duck typing) and, when the query falls outside in-domain or
matches a profile boundary's out-of-scope topic, emits a typed
:class:`ScopeRefusalDirective` so the response synthesizer can:

* refuse outright (``strict_refuse``),
* answer with a soft disclaimer attached (``soft_disclaim``), or
* pass through unchanged (``passthrough``) — used when the figure
  vertical is wired in shadow mode and the runtime is collecting
  evidence rather than enforcing.

This module is the L4 counterpart to :mod:`grounded_decoder`. The
two together implement the figure vertical's "answer only what is
documented; refuse what is not" contract without any keyword
heuristics — both delegate the actual decision to the corpus-derived
artifacts produced by ``lifeform-domain-figure``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class CoveragePolicy(str, Enum):
    """How the runtime treats out-of-scope queries.

    Mirrors the ``CoveragePolicy`` field on
    :class:`dlaas_platform_contracts.TemplateSpec` (planned for P4.1).
    The names are deliberately self-documenting — no
    ``BOUNDARY`` enum integer values that callers have to look up.
    """

    STRICT_REFUSE = "strict_refuse"
    SOFT_DISCLAIM = "soft_disclaim"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class ScopeRefusalDirective:
    """Result of :meth:`ScopeRefuser.evaluate`.

    The response synthesizer reads this directive instead of
    re-deriving any judgement on its own. If ``should_refuse`` is
    True the synthesizer must emit ``refusal_text`` (and skip
    further generation). If ``disclaimers`` is non-empty the
    synthesizer must include them in the rendered response.
    ``rationale`` is for the audit log.
    """

    should_refuse: bool
    refusal_text: str
    disclaimers: tuple[str, ...]
    coverage_decision: str
    rationale: str


@dataclass(frozen=True)
class ScopeRefuserConfig:
    """Tunable knobs for :class:`ScopeRefuser`.

    The strict-refuse refusal text is reviewer-curated rather than
    runtime-generated so the L4 contract has a stable, citation-
    free output for the case where no in-domain answer exists.
    """

    policy: CoveragePolicy = CoveragePolicy.SOFT_DISCLAIM
    out_of_domain_refusal: str = (
        "I'm sorry — that topic falls outside what this figure documented "
        "in their primary sources. I can't answer it for them."
    )
    boundary_refusal: str = (
        "That request runs into one of the figure's documented boundaries "
        "and I won't speak for them on it."
    )
    out_of_domain_disclaimer: str = (
        "Note: this question lies outside the figure's documented coverage; "
        "the response is informed by adjacent context only."
    )
    boundary_disclaimer: str = (
        "Note: this topic touches one of the figure's reviewer-declared "
        "out-of-scope areas; treat the response as best-effort framing."
    )


class _CoverageMapLike(Protocol):
    """Duck-typed coverage map contract that ScopeRefuser consumes.

    The optional ``retrieval_index`` kwarg is the debt #39
    retrieval-augmented floor: when the static centroid path would
    mark an in-corpus query OUT_OF_DOMAIN (the Wave K Einstein
    relativity / postulate / theory bug), the coverage map gets a
    second chance to lift it to IN_DOMAIN by cosine-matching real
    corpus chunks. The Protocol advertises that the kwarg is
    accepted; :class:`ScopeRefuser` only passes it through when the
    constructor was given a non-None ``retrieval_index``, so
    consumers / mocks that don't care about the floor keep working
    without having to widen their ``classify_query`` signature.
    """

    def classify_query(self, query: str, **kwargs: Any) -> Any: ...


class ScopeRefuser:
    """L4 enforcer that classifies queries and emits a typed directive."""

    def __init__(
        self,
        coverage_map: _CoverageMapLike,
        *,
        config: ScopeRefuserConfig | None = None,
        retrieval_index: Any = None,
    ) -> None:
        self._coverage_map = coverage_map
        self._config = config or ScopeRefuserConfig()
        self._retrieval_index = retrieval_index

    @property
    def config(self) -> ScopeRefuserConfig:
        return self._config

    def evaluate(self, *, query: str) -> ScopeRefusalDirective:
        """Classify the query and translate the decision into a directive.

        The actual classification is performed by the coverage map
        (duck-typed via :class:`_CoverageMapLike`); this method only
        translates the result into the runtime-facing
        :class:`ScopeRefusalDirective` shape, applying the active
        :class:`CoveragePolicy`.

        When the refuser was constructed with a non-None
        ``retrieval_index`` the kwarg is forwarded to
        :meth:`classify_query` so the coverage map's
        retrieval-augmented floor (debt #39) can lift in-corpus
        queries that miss every static centroid. When the refuser
        was constructed without a retrieval_index, the call shape
        stays byte-for-byte identical to the pre-wiring path so
        existing callers / lightweight test mocks do not need to
        widen their signature.
        """

        if self._retrieval_index is not None:
            classification = self._coverage_map.classify_query(
                query, retrieval_index=self._retrieval_index
            )
        else:
            classification = self._coverage_map.classify_query(query)
        decision = _decision_value(classification)
        rationale = _decision_rationale(classification)
        if decision == "in_domain":
            return ScopeRefusalDirective(
                should_refuse=False,
                refusal_text="",
                disclaimers=(),
                coverage_decision=decision,
                rationale=rationale,
            )
        if decision == "boundary_blocked":
            return self._directive_for_boundary(decision, rationale)
        return self._directive_for_out_of_domain(decision, rationale)

    def _directive_for_boundary(
        self, decision: str, rationale: str
    ) -> ScopeRefusalDirective:
        policy = self._config.policy
        if policy is CoveragePolicy.STRICT_REFUSE:
            return ScopeRefusalDirective(
                should_refuse=True,
                refusal_text=self._config.boundary_refusal,
                disclaimers=(),
                coverage_decision=decision,
                rationale=rationale,
            )
        if policy is CoveragePolicy.SOFT_DISCLAIM:
            return ScopeRefusalDirective(
                should_refuse=False,
                refusal_text="",
                disclaimers=(self._config.boundary_disclaimer,),
                coverage_decision=decision,
                rationale=rationale,
            )
        return ScopeRefusalDirective(
            should_refuse=False,
            refusal_text="",
            disclaimers=(),
            coverage_decision=decision,
            rationale=f"{rationale} | passthrough policy active",
        )

    def _directive_for_out_of_domain(
        self, decision: str, rationale: str
    ) -> ScopeRefusalDirective:
        policy = self._config.policy
        if policy is CoveragePolicy.STRICT_REFUSE:
            return ScopeRefusalDirective(
                should_refuse=True,
                refusal_text=self._config.out_of_domain_refusal,
                disclaimers=(),
                coverage_decision=decision,
                rationale=rationale,
            )
        if policy is CoveragePolicy.SOFT_DISCLAIM:
            return ScopeRefusalDirective(
                should_refuse=False,
                refusal_text="",
                disclaimers=(self._config.out_of_domain_disclaimer,),
                coverage_decision=decision,
                rationale=rationale,
            )
        return ScopeRefusalDirective(
            should_refuse=False,
            refusal_text="",
            disclaimers=(),
            coverage_decision=decision,
            rationale=f"{rationale} | passthrough policy active",
        )


def _decision_value(classification: Any) -> str:
    """Extract the string decision from a coverage classification.

    Accepts either the typed :class:`CoverageDecision` enum (whose
    ``.value`` is a string) or any duck-typed object with a
    ``.decision`` attribute. We keep this single ``getattr`` /
    ``isinstance`` dance here so the rest of the file stays narrow
    on the contract surface.
    """

    decision_attr = classification.decision
    if hasattr(decision_attr, "value"):
        return decision_attr.value
    if isinstance(decision_attr, str):
        return decision_attr
    raise TypeError(
        f"ScopeRefuser._decision_value: unexpected classification.decision "
        f"type {type(decision_attr).__name__!r}; expected enum or str."
    )


def _decision_rationale(classification: Any) -> str:
    rationale = getattr(classification, "rationale", "")
    return rationale if isinstance(rationale, str) else ""


__all__ = [
    "CoveragePolicy",
    "ScopeRefuser",
    "ScopeRefuserConfig",
    "ScopeRefusalDirective",
]
