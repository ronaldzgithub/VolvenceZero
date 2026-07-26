"""Audit harness for the panorama participation gate.

Replays :mod:`volvence_zero.regime.panorama_corpus` against a gate
implementation and reports **ceiling breaches and floor misses
separately**.

That separation is the whole point. Collapsing them into one accuracy
number (or an F1) hides the asymmetry the gate is built around:

* A *floor miss* — the gate stayed quieter than the situation
  warranted — costs the user one structured turn. The next turn can
  still open. Recoverable.
* A *ceiling breach* — the gate expanded a decision panorama into a
  turn that did not want one — is a relational harm, and the user
  usually will not say so. It just reads as cold.

So the two rates are reported side by side and never summed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace

from volvence_zero.regime.hint_readout import (
    build_hint_readout_context,
    readout_participation_hint,
)
from volvence_zero.regime.identity import ParticipationLevel
from volvence_zero.regime.panorama_corpus import (
    PANORAMA_CORPUS,
    PanoramaScenario,
    ScenarioSnapshots,
    build_snapshots,
    level_rank,
)

# A gate under audit: given a scenario and its materialised snapshots,
# return the panorama level it would publish.
GateFn = Callable[[PanoramaScenario, ScenarioSnapshots], ParticipationLevel]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    family: str
    topic: str
    expected_min: str
    expected_max: str
    actual: str
    # "ok" | "over" (ceiling breach) | "under" (floor miss)
    verdict: str
    note: str


@dataclass(frozen=True)
class FamilyRates:
    family: str
    total: int
    over_count: int
    under_count: int
    ok_count: int

    @property
    def over_rate(self) -> float:
        return round(self.over_count / self.total, 4) if self.total else 0.0

    @property
    def under_rate(self) -> float:
        return round(self.under_count / self.total, 4) if self.total else 0.0


@dataclass(frozen=True)
class GateAuditReport:
    gate_name: str
    cases: tuple[CaseResult, ...]
    families: tuple[FamilyRates, ...]

    @property
    def over_count(self) -> int:
        return sum(1 for case in self.cases if case.verdict == "over")

    @property
    def under_count(self) -> int:
        return sum(1 for case in self.cases if case.verdict == "under")

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_name": self.gate_name,
            # Deliberately two separate top-level numbers. Do not add a
            # combined score here; see the module docstring.
            "ceiling_breaches": self.over_count,
            "floor_misses": self.under_count,
            "total_cases": len(self.cases),
            "families": [
                {
                    "family": family.family,
                    "total": family.total,
                    "ceiling_breaches": family.over_count,
                    "floor_misses": family.under_count,
                    "ok": family.ok_count,
                    "ceiling_breach_rate": family.over_rate,
                    "floor_miss_rate": family.under_rate,
                }
                for family in self.families
            ],
            "cases": [asdict(case) for case in self.cases],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _classify(scenario: PanoramaScenario, actual: ParticipationLevel) -> str:
    if level_rank(actual) > level_rank(scenario.expected_max):
        return "over"
    if level_rank(actual) < level_rank(scenario.expected_min):
        return "under"
    return "ok"


def audit_gate(
    gate: GateFn,
    *,
    gate_name: str,
    corpus: Iterable[PanoramaScenario] = PANORAMA_CORPUS,
) -> GateAuditReport:
    cases: list[CaseResult] = []
    for scenario in corpus:
        snapshots = build_snapshots(scenario)
        actual = gate(scenario, snapshots)
        cases.append(
            CaseResult(
                case_id=scenario.case_id,
                family=scenario.family,
                topic=scenario.topic,
                expected_min=scenario.expected_min.value,
                expected_max=scenario.expected_max.value,
                actual=actual.value,
                verdict=_classify(scenario, actual),
                note=scenario.note,
            )
        )
    families: list[FamilyRates] = []
    for family in ("negative", "boundary", "positive"):
        subset = [case for case in cases if case.family == family]
        if not subset:
            continue
        families.append(
            FamilyRates(
                family=family,
                total=len(subset),
                over_count=sum(1 for case in subset if case.verdict == "over"),
                under_count=sum(1 for case in subset if case.verdict == "under"),
                ok_count=sum(1 for case in subset if case.verdict == "ok"),
            )
        )
    return GateAuditReport(
        gate_name=gate_name, cases=tuple(cases), families=tuple(families)
    )


# ---------------------------------------------------------------------------
# Gate adapters
# ---------------------------------------------------------------------------


def v1_gate(
    scenario: PanoramaScenario, snapshots: ScenarioSnapshots
) -> ParticipationLevel:
    """The shipped Gap-8 slice-2 readout, unchanged.

    This is the P0 baseline: it reads only the attention / controller
    profile and never sees the decision-structure signals.
    """
    context = build_hint_readout_context(
        regime_id=scenario.regime_id,
        turns_in_current_regime=scenario.turns_in_current_regime,
        candidates=snapshots.candidates,
        memory=snapshots.memory,
        dual_track=snapshots.dual_track,
        evaluation=snapshots.evaluation,
        prediction_error=None,
    )
    return readout_participation_hint(context, panorama_gate="v1").panorama_level


def build_v2_context(
    scenario: PanoramaScenario, snapshots: ScenarioSnapshots
):
    """Build the readout context the v2 gate sees, semantic owners included."""
    return build_hint_readout_context(
        regime_id=scenario.regime_id,
        turns_in_current_regime=scenario.turns_in_current_regime,
        candidates=snapshots.candidates,
        memory=snapshots.memory,
        dual_track=snapshots.dual_track,
        evaluation=snapshots.evaluation,
        prediction_error=None,
        plan_intent=snapshots.plan_intent,
        goal_value=snapshots.goal_value,
        open_loop=snapshots.open_loop,
        belief_assumption=snapshots.belief_assumption,
        commitment=snapshots.commitment,
        boundary_consent=snapshots.boundary_consent,
        previous_panorama_level=scenario.previous_panorama_level,
    )


def v2_gate(
    scenario: PanoramaScenario, snapshots: ScenarioSnapshots
) -> ParticipationLevel:
    """The decision-structure gate.

    Same context builder, same discretisation, same publisher. The only
    thing that changed is which features the panorama decision reads.
    """
    context = build_v2_context(scenario, snapshots)
    return readout_participation_hint(context, panorama_gate="v2").panorama_level


# ---------------------------------------------------------------------------
# Adversarial probes
#
# The corpus and the gate were written by the same hand, so "v2 passes
# the corpus" on its own is close to worthless. These two probes are
# what make the result falsifiable:
#
# * ``ablation_report`` pins one feature to 1.0 — i.e. makes the gate
#   blind to it — and re-audits. If the corpus still passes, that
#   feature is decoration and the four-way conjunction is a story.
# * ``feature_correlations`` checks the four are not measuring the same
#   thing. A geometric mean over collinear axes is not a conjunction of
#   four conditions, it is one condition raised to a power.
# ---------------------------------------------------------------------------

DECISION_FEATURES = (
    "option_multiplicity",
    "irreversibility",
    "ranking_instability",
    "unknown_dominance",
)


def ablation_report(feature: str) -> GateAuditReport:
    """Audit v2 with ``feature`` pinned to 1.0 (the gate cannot see it)."""
    if feature not in DECISION_FEATURES:
        raise ValueError(f"unknown decision feature: {feature!r}")

    def blinded(
        scenario: PanoramaScenario, snapshots: ScenarioSnapshots
    ) -> ParticipationLevel:
        context = replace(build_v2_context(scenario, snapshots), **{feature: 1.0})
        return readout_participation_hint(
            context, panorama_gate="v2"
        ).panorama_level

    return audit_gate(blinded, gate_name=f"v2-ablate-{feature}")


def feature_matrix() -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Per-case decision-structure feature values, in corpus order."""
    rows: list[tuple[str, tuple[float, ...]]] = []
    for scenario in PANORAMA_CORPUS:
        context = build_v2_context(scenario, build_snapshots(scenario))
        rows.append(
            (
                scenario.case_id,
                tuple(getattr(context, name) for name in DECISION_FEATURES),
            )
        )
    return tuple(rows)


def _pearson(left: list[float], right: list[float]) -> float:
    n = len(left)
    if n < 2:
        return 0.0
    mean_l = sum(left) / n
    mean_r = sum(right) / n
    cov = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
    var_l = sum((a - mean_l) ** 2 for a in left)
    var_r = sum((b - mean_r) ** 2 for b in right)
    if var_l <= 0.0 or var_r <= 0.0:
        return 0.0
    return round(cov / ((var_l * var_r) ** 0.5), 4)


def feature_correlations() -> tuple[tuple[str, str, float], ...]:
    """Pairwise Pearson correlation of the four features over the corpus."""
    rows = feature_matrix()
    columns = {
        name: [row[1][index] for row in rows]
        for index, name in enumerate(DECISION_FEATURES)
    }
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(DECISION_FEATURES):
        for right in DECISION_FEATURES[i + 1 :]:
            pairs.append((left, right, _pearson(columns[left], columns[right])))
    return tuple(pairs)
