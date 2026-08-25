"""Interval valuation, option value, and value of information.

Three things are computed here, and they are deliberately one
computation rather than three:

* **What each option is worth** — as an interval, never a point.
* **What each option is worth *because it keeps choices open*** — the
  reason a reversible "wait three months" can beat an irreversible
  option with a higher headline number.
* **Which unresolved unknown is worth chasing next** — the value of
  information.

The last two are the same quantity seen from different ends. An
option's option-value is largely the information it buys; an unknown's
VOI is how much the ranking would move if it were resolved. Computing
them from one model is not tidiness — it is the reason the model can be
wrong in a checkable way. Two separately-tuned heuristics could always
be made to agree on the fourth-act transcript.

Four rules the arithmetic here enforces, each of which exists because
the obvious alternative produces confident nonsense:

1. **Intervals, never points.** A single expected value invites "option
   A is best" when the ranges overlap completely.

2. **Unknowns widen; haircuts shift.** A three-year lock-up is a known
   discount — it moves the interval down. "Nobody has read the equity
   documents" is *not* a discount: resolving it could land anywhere, so
   it widens the interval. Collapsing an unresolved question into a
   midpoint is exactly the false precision that makes a decision aid
   dangerous.

3. **Aggregate comonotonically.** Summing dimension intervals as if
   their extremes were independent narrows the total, which makes
   separation claims easier. We do not know the correlation, so we take
   the width-maximising assumption: the conservatism points away from
   claiming one option beats another.

4. **Separation is a claim, not a comparison.** ``leader`` is only
   populated when the top interval clears the runner-up's. Overlapping
   intervals yield no leader at all — the honest reading is about
   robustness and reversibility, not "highest EV".

One scale decision worth stating rather than leaving implicit: option
value is deliberately small next to the dimension values, so it can
decide ``most_robust_ref`` under overlap but can never push one option's
interval clear of another's. Optionality is a defensible reason to
prefer a choice when the numbers cannot separate; it is not a reason to
override numbers that can. If the two ever need to compete on equal
terms, that is a modelling decision to make explicitly — not something
to arrive at by scaling a bonus up until it wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class Interval:
    """A closed range of value. ``base`` is the working estimate."""

    low: float
    base: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.base <= self.high:
            raise ValueError(
                f"Interval requires low <= base <= high, got "
                f"({self.low}, {self.base}, {self.high})"
            )

    @property
    def width(self) -> float:
        return self.high - self.low

    def shifted(self, factor: float) -> "Interval":
        """Apply a known haircut. Scales the whole interval down."""
        factor = _clamp01(factor)
        return Interval(self.low * factor, self.base * factor, self.high * factor)

    def widened(self, spread: float) -> "Interval":
        """Widen for an unresolved question.

        ``base`` does not move: an unknown is not evidence that the
        working estimate was wrong, only that it is unsupported.
        """
        spread = max(0.0, spread)
        return Interval(self.low - spread, self.base, self.high + spread)

    def plus(self, other: "Interval") -> "Interval":
        return Interval(
            self.low + other.low, self.base + other.base, self.high + other.high
        )

    def scaled(self, weight: float) -> "Interval":
        weight = max(0.0, weight)
        return Interval(self.low * weight, self.base * weight, self.high * weight)

    def clears(self, other: "Interval") -> bool:
        """Strictly above ``other`` with no overlap."""
        return self.low > other.high


@dataclass(frozen=True)
class DimensionEstimate:
    """One option scored on one dimension.

    ``option_ref`` / ``dimension_ref`` / ``evidence_refs`` point at the
    owners that hold the corresponding content. Nothing here carries
    prose — same ownership rule as the workspace records.

    ``haircuts`` are known discounts in [0,1] (0.7 = "worth 70% of face
    because of the lock-up"). ``unresolved_refs`` name unknowns that
    make this estimate unsupported; each widens the interval instead of
    moving it.
    """

    option_ref: str
    dimension_ref: str
    interval: Interval
    evidence_refs: tuple[str, ...] = ()
    haircuts: tuple[tuple[str, float], ...] = ()
    unresolved_refs: tuple[str, ...] = ()
    # How far a single unresolved question could move this estimate.
    # Expressed in the same units as the interval.
    unresolved_spread: float = 0.0

    def resolved_interval(
        self, *, assume_resolved: frozenset[str] = frozenset()
    ) -> Interval:
        """Interval after haircuts, widened for still-open unknowns.

        ``assume_resolved`` is how the VOI probe asks "what would this
        look like if we knew?" — a resolved unknown stops widening.
        """
        interval = self.interval
        for _, factor in self.haircuts:
            interval = interval.shifted(factor)
        open_unknowns = [
            ref for ref in self.unresolved_refs if ref not in assume_resolved
        ]
        if open_unknowns and self.unresolved_spread > 0.0:
            interval = interval.widened(self.unresolved_spread * len(open_unknowns))
        return interval

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence_refs)


@dataclass(frozen=True)
class OptionValuation:
    option_ref: str
    value: Interval
    option_value: float
    total: Interval
    # Unknowns still widening this option's interval.
    open_unknown_refs: tuple[str, ...] = ()
    unsupported_dimension_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnknownValue:
    """What resolving one unknown would do to the ranking."""

    unknown_ref: str
    # True when the leading option differs between the optimistic and
    # pessimistic resolutions — i.e. the answer could change the choice.
    can_flip_leader: bool
    # How much of the overlap between the two leading options this
    # unknown accounts for — i.e. how far answering it moves the field
    # toward being decidable at all.
    width_share: float

    @property
    def is_worth_asking(self) -> bool:
        return self.can_flip_leader or self.width_share > 0.0


@dataclass(frozen=True)
class ValuationResult:
    options: tuple[OptionValuation, ...] = ()
    unknowns: tuple[UnknownValue, ...] = ()
    # Only set when the top interval strictly clears the runner-up.
    leader_ref: str | None = None
    # Set when no leader separates but one option is most reversible /
    # keeps the most choices open. This is the honest answer under
    # overlap, and the rendering contract keys off which of the two is
    # populated.
    most_robust_ref: str | None = None
    rationale: str = ""

    @property
    def separated(self) -> bool:
        return self.leader_ref is not None

    def next_unknown_to_resolve(self) -> str | None:
        """The unknown whose answer would move the ranking most.

        Returns ``None`` when nothing left to learn could change the
        outcome — the termination condition. Without it the system asks
        forever, which reads as interrogation rather than help.
        """
        candidates = [u for u in self.unknowns if u.is_worth_asking]
        if not candidates:
            return None
        best = max(
            candidates, key=lambda u: (u.can_flip_leader, u.width_share, u.unknown_ref)
        )
        return best.unknown_ref


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _aggregate(
    estimates: list[DimensionEstimate],
    weights: dict[str, float],
    *,
    assume_resolved: frozenset[str] = frozenset(),
) -> Interval:
    """Weighted comonotonic sum across dimensions.

    Comonotonic — extremes added directly — because we do not know how
    the dimensions covary and the alternative (independence) narrows the
    result, which makes it easier to claim one option beats another.
    Conservatism belongs on the side of not making that claim.
    """
    total = Interval(0.0, 0.0, 0.0)
    for estimate in estimates:
        weight = weights.get(estimate.dimension_ref, 1.0)
        total = total.plus(
            estimate.resolved_interval(assume_resolved=assume_resolved).scaled(weight)
        )
    return total


def _contender_overlap(valuations: dict[str, Interval]) -> float:
    """How much the two leading intervals still overlap.

    This is the quantity that has to reach zero before a comparative
    claim is possible, so it is the right thing to measure information
    against: an answer is valuable to the extent it moves the top two
    apart.

    The obvious cheaper measure — the width of the leading interval — is
    blind in a way that matters. An unknown that widens a *challenger*
    leaves the leader's width untouched and scores zero, even when
    resolving it is the only thing standing between us and a decision.
    """
    if len(valuations) < 2:
        only = next(iter(valuations.values()), None)
        return 0.0 if only is None else only.width
    ordered = sorted(valuations.values(), key=lambda item: item.base, reverse=True)
    top, second = ordered[0], ordered[1]
    return max(0.0, min(top.high, second.high) - max(top.low, second.low))


def _leader(valuations: list[tuple[str, Interval]]) -> str | None:
    """Top option only if its interval strictly clears every other."""
    if len(valuations) < 2:
        return valuations[0][0] if valuations else None
    ordered = sorted(valuations, key=lambda item: item[1].base, reverse=True)
    top_ref, top = ordered[0]
    if all(top.clears(other) for _, other in ordered[1:]):
        return top_ref
    return None


def evaluate_options(
    estimates: tuple[DimensionEstimate, ...],
    *,
    weights: tuple[tuple[str, float], ...] = (),
    reversibility: tuple[tuple[str, float], ...] = (),
    resolves: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> ValuationResult:
    """Value every option, price its optionality, and rank the unknowns.

    ``weights`` are the user-confirmed dimension weights (the semantic
    fact lives in ``goal_value``; only the numbers are passed here).
    ``reversibility`` is per option in [0,1] — 1.0 meaning the choice
    can be undone at no cost. ``resolves`` maps an option to the
    unknowns that taking it would answer.

    Option value is ``reversibility × (information the option buys)``.
    That product is the formal content of "wait three months": the
    option is cheap to undo *and* it resolves the things the ranking
    hangs on. An irreversible option that answers everything gets no
    credit here — you cannot use what you learn.
    """
    weight_map = dict(weights)
    reversibility_map = dict(reversibility)
    resolves_map = dict(resolves)

    by_option: dict[str, list[DimensionEstimate]] = {}
    for estimate in estimates:
        by_option.setdefault(estimate.option_ref, []).append(estimate)
    if not by_option:
        return ValuationResult(rationale="valuation: no estimates")

    all_unknowns: set[str] = set()
    for estimate in estimates:
        all_unknowns.update(estimate.unresolved_refs)

    base_values = {
        option_ref: _aggregate(items, weight_map)
        for option_ref, items in by_option.items()
    }
    baseline_leader = _leader(list(base_values.items()))

    # --- value of information -------------------------------------------------
    # For each unknown: recompute the ranking as if it were answered. A
    # question whose answer cannot change which option leads is not worth
    # asking, however uncertain it feels.
    baseline_overlap = _contender_overlap(base_values)
    unknown_values: list[UnknownValue] = []
    for unknown in sorted(all_unknowns):
        resolved = {
            option_ref: _aggregate(
                items, weight_map, assume_resolved=frozenset({unknown})
            )
            for option_ref, items in by_option.items()
        }
        resolved_leader = _leader(list(resolved.items()))
        can_flip = resolved_leader != baseline_leader
        after = _contender_overlap(resolved)
        share = (
            0.0
            if baseline_overlap <= 0.0
            else _clamp01((baseline_overlap - after) / baseline_overlap)
        )
        unknown_values.append(
            UnknownValue(
                unknown_ref=unknown, can_flip_leader=can_flip, width_share=round(share, 4)
            )
        )

    information_value = {
        value.unknown_ref: (1.0 if value.can_flip_leader else value.width_share)
        for value in unknown_values
    }

    # --- option value ---------------------------------------------------------
    option_valuations: list[OptionValuation] = []
    for option_ref, items in by_option.items():
        reversibility_score = _clamp01(reversibility_map.get(option_ref, 0.0))
        bought = sum(
            information_value.get(ref, 0.0)
            for ref in resolves_map.get(option_ref, ())
        )
        option_value = reversibility_score * bought
        value = base_values[option_ref]
        open_unknowns = sorted(
            {ref for item in items for ref in item.unresolved_refs}
        )
        unsupported = sorted(
            item.dimension_ref for item in items if not item.has_evidence
        )
        option_valuations.append(
            OptionValuation(
                option_ref=option_ref,
                value=value,
                option_value=round(option_value, 4),
                # Option value is additive on the working estimate and on
                # the upside; it does not raise the floor, because the
                # choices an option preserves are worth nothing in the
                # branch where the option itself goes badly.
                total=Interval(
                    value.low, value.base + option_value, value.high + option_value
                ),
                open_unknown_refs=tuple(open_unknowns),
                unsupported_dimension_refs=tuple(unsupported),
            )
        )

    totals = [(item.option_ref, item.total) for item in option_valuations]
    leader_ref = _leader(totals)
    most_robust_ref = None
    if leader_ref is None:
        # No separation. The defensible statement is about which option
        # keeps the most choices open, not which one "wins".
        most_robust_ref = max(
            option_valuations,
            key=lambda item: (
                item.option_value,
                _clamp01(reversibility_map.get(item.option_ref, 0.0)),
                -item.total.width,
                item.option_ref,
            ),
        ).option_ref
    rationale = (
        f"valuation: {len(option_valuations)} option(s), "
        f"{len(unknown_values)} unknown(s), "
        f"leader={leader_ref or 'none(intervals overlap)'}"
    )
    return ValuationResult(
        options=tuple(option_valuations),
        unknowns=tuple(unknown_values),
        leader_ref=leader_ref,
        most_robust_ref=most_robust_ref,
        rationale=rationale,
    )
