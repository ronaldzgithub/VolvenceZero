"""Contrast pair schema for the L2 stance-fidelity contract.

A :class:`FigureContrastPair` records the figure's documented
position vs a named contemporary's position on the same axis, with
reviewer-curated paraphrases of each. The pair is the load-bearing
evidence for steering-vector training (P5.2): the bake job learns
a residual direction that pushes the substrate toward the figure's
side and away from the named opponent's side.

This module is **schema only** — the bake job lives in
:mod:`lifeform_domain_figure.steering_bake`. Splitting the data
declaration from the training pipeline keeps the imports light when
a caller only needs to inspect or extend the contrast corpus.

All paraphrases shipped with the wheel are reviewer-paraphrased
synthetic original text: NO verbatim quotes from any opponent's
primary sources are included (their estate's IP is just as binding
as the figure's). Real production deployments add their own
reviewed pairs; the schema is identical.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureContrastPair:
    """One reviewed (figure stance, opponent stance) contrast pair.

    Fields:

    * ``pair_id``           — canonical citation key
                              (``"einstein-vs-bohr:locality"``).
    * ``axis``              — short label for the disagreement axis
                              (``"locality"``, ``"determinism"``,
                              ``"completeness"``).
    * ``figure_stance``     — reviewer-paraphrased text of the
                              figure's documented position.
    * ``opponent_id``       — normalised id of the named opponent
                              (``"bohr"``, ``"heisenberg"``).
    * ``opponent_stance``   — reviewer-paraphrased text of the
                              opponent's documented counter-position.
    * ``evidence_locator``  — path or citation back to the reviewed
                              source documents that justify both
                              paraphrases.
    * ``confidence``        — reviewer-assigned confidence in the
                              paraphrase fidelity (in ``[0, 1]``).
    * ``description``       — short human-readable description of
                              the disagreement (used as the
                              steering vector's audit label).
    """

    pair_id: str
    axis: str
    figure_stance: str
    opponent_id: str
    opponent_stance: str
    evidence_locator: str
    confidence: float
    description: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("pair_id", self.pair_id),
            ("axis", self.axis),
            ("figure_stance", self.figure_stance),
            ("opponent_id", self.opponent_id),
            ("opponent_stance", self.opponent_stance),
            ("evidence_locator", self.evidence_locator),
            ("description", self.description),
        ):
            if not value or not value.strip():
                raise ValueError(
                    f"FigureContrastPair.{field_name} must be non-empty "
                    f"(pair_id={self.pair_id!r})"
                )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"FigureContrastPair.confidence must be in [0,1], "
                f"got {self.confidence!r}"
            )


@dataclass(frozen=True)
class FigureContrastSet:
    """A reviewed bundle of contrast pairs for one figure.

    The set is keyed by ``figure_id`` so the figure vertical can hold
    several axes per figure without a separate registration step. The
    set is frozen and integrity-hashable so steering bakes produced
    from it are reproducible byte-for-byte (R15 rollback contract).
    """

    figure_id: str
    pairs: tuple[FigureContrastPair, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError("FigureContrastSet.figure_id must be non-empty")
        if not self.pairs:
            raise ValueError(
                "FigureContrastSet.pairs must be non-empty; refusing to "
                "build a steering set with no contrast evidence."
            )
        seen: set[str] = set()
        for pair in self.pairs:
            if pair.pair_id in seen:
                raise ValueError(
                    f"FigureContrastSet has duplicate pair_id "
                    f"{pair.pair_id!r}"
                )
            seen.add(pair.pair_id)


def build_einstein_contrast_set() -> FigureContrastSet:
    """Reviewer-paraphrased Einstein-vs-contemporary contrast set.

    Three documented axes of disagreement:

    1. Locality / separability (vs Bohr).
    2. Determinism (vs Born).
    3. Completeness of the quantum description (vs Heisenberg).

    Every text is **synthetic original** reviewer paraphrase; no
    verbatim quotes from primary sources of any party are shipped.
    """

    return FigureContrastSet(
        figure_id="einstein",
        pairs=(
            FigureContrastPair(
                pair_id="einstein-vs-bohr:locality",
                axis="locality",
                figure_stance=(
                    "Spatially separated systems should each carry their "
                    "own definite physical state; any complete theory "
                    "must reflect that without nonlocal influences."
                ),
                opponent_id="bohr",
                opponent_stance=(
                    "The very notion of an independent state for a "
                    "subsystem already assumes a measurement context; "
                    "the formalism is complete in the Copenhagen sense."
                ),
                evidence_locator="profile:einstein:contrast:locality",
                confidence=0.92,
                description=(
                    "Documented disagreement on whether spatially "
                    "separated subsystems must carry locally separable "
                    "states (Solvay 1927/1930, EPR 1935 and reactions)."
                ),
            ),
            FigureContrastPair(
                pair_id="einstein-vs-born:determinism",
                axis="determinism",
                figure_stance=(
                    "A deeper theory should describe definite causal "
                    "connections; probabilistic predictions are an "
                    "intermediate stage rather than the final word."
                ),
                opponent_id="born",
                opponent_stance=(
                    "The probabilistic interpretation is not provisional; "
                    "it is the fundamental form physical theories should "
                    "take, with no underlying determinism waiting to be "
                    "discovered."
                ),
                evidence_locator="profile:einstein:contrast:determinism",
                confidence=0.88,
                description=(
                    "Documented disagreement (Einstein-Born "
                    "correspondence) on whether probabilistic predictions "
                    "are a fundamental or intermediate description."
                ),
            ),
            FigureContrastPair(
                pair_id="einstein-vs-heisenberg:completeness",
                axis="completeness",
                figure_stance=(
                    "Quantum mechanics predicts experiments accurately "
                    "and is provisionally correct as a statistical "
                    "theory, yet a more complete deeper theory remains "
                    "to be sought."
                ),
                opponent_id="heisenberg",
                opponent_stance=(
                    "There is no deeper theory to find; the uncertainty "
                    "relations are constitutive of physical description, "
                    "not symptoms of an incomplete model."
                ),
                evidence_locator="profile:einstein:contrast:completeness",
                confidence=0.86,
                description=(
                    "Documented disagreement on whether the quantum "
                    "formalism is the complete description of reality "
                    "or only an intermediate stage."
                ),
            ),
        ),
        description=(
            "Reviewer-paraphrased Einstein contrast set across "
            "locality / determinism / completeness axes."
        ),
    )


__all__ = [
    "FigureContrastPair",
    "FigureContrastSet",
    "build_einstein_contrast_set",
]
