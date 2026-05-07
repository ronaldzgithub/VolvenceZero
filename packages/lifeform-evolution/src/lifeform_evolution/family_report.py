"""Six-family evaluation grouping over a ``BenchmarkReport``.

R12 says the system must be evaluated across six families:

* **F1 \u2014 task capability**: did it actually help the user?
* **F2 \u2014 interaction quality**: was the interaction comfortable?
* **F3 \u2014 relationship continuity**: does it carry forward across scenes?
* **F4 \u2014 learning quality**: is adaptation correct and stable?
* **F5 \u2014 abstraction quality**: are the controllers meaningful?
* **F6 \u2014 safety and boundedness**: does adaptation stay inside the rails?

The kernel's ``vz-cognition.evaluation.backbone`` already produces all the
underlying owner-side signals; this module is the **product-layer reporter**
that pulls those signals out of a ``BenchmarkReport`` and presents them
grouped by family. The point is not to invent new metrics \u2014 it is to make
the existing evidence answer the R12 question of whether a "digital
organism" was evaluated, not just a "helpful assistant".

Pure / read-only: takes a frozen ``BenchmarkReport`` and returns a frozen
``FamilyReport``. No kernel mutation, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from lifeform_evolution.benchmark import BenchmarkReport, TurnReport


class FamilyId(str, Enum):
    F1_TASK_CAPABILITY = "F1"
    F2_INTERACTION_QUALITY = "F2"
    F3_RELATIONSHIP_CONTINUITY = "F3"
    F4_LEARNING_QUALITY = "F4"
    F5_ABSTRACTION_QUALITY = "F5"
    F6_SAFETY_BOUNDEDNESS = "F6"


_FAMILY_NAMES: dict[FamilyId, str] = {
    FamilyId.F1_TASK_CAPABILITY: "task capability",
    FamilyId.F2_INTERACTION_QUALITY: "interaction quality",
    FamilyId.F3_RELATIONSHIP_CONTINUITY: "relationship continuity",
    FamilyId.F4_LEARNING_QUALITY: "learning quality",
    FamilyId.F5_ABSTRACTION_QUALITY: "abstraction quality",
    FamilyId.F6_SAFETY_BOUNDEDNESS: "safety and boundedness",
}


@dataclass(frozen=True)
class FamilyMetric:
    """One metric inside a family."""

    metric_id: str
    name: str
    value: float
    threshold: float | None
    higher_is_better: bool = True
    note: str = ""

    @property
    def passed(self) -> bool:
        if self.threshold is None:
            return True
        if self.higher_is_better:
            return self.value >= self.threshold
        return self.value <= self.threshold


@dataclass(frozen=True)
class FamilyEvaluation:
    """Aggregated evaluation for one family."""

    family_id: FamilyId
    family_name: str
    metrics: tuple[FamilyMetric, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for m in self.metrics if m.passed)

    @property
    def total_count(self) -> int:
        return len(self.metrics)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / max(1, self.total_count)

    @property
    def family_passed(self) -> bool:
        # A family passes when ALL its metrics with thresholds pass. Metrics
        # without thresholds are reported but do not gate.
        return all(m.passed for m in self.metrics)


@dataclass(frozen=True)
class FamilyReport:
    """All six families evaluated for a single scenario."""

    scenario_id: str
    families: tuple[FamilyEvaluation, ...] = field(default_factory=tuple)

    @property
    def overall_passed(self) -> bool:
        return all(fam.family_passed for fam in self.families)

    @property
    def overall_pass_rate(self) -> float:
        total = sum(fam.total_count for fam in self.families)
        passed = sum(fam.passed_count for fam in self.families)
        return passed / max(1, total)

    def family(self, family_id: FamilyId) -> FamilyEvaluation:
        for fam in self.families:
            if fam.family_id == family_id:
                return fam
        raise KeyError(f"family {family_id!r} not present in report {self.scenario_id!r}")


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


_DEFAULT_RESPONSE_LENGTH_MAX = 1200  # F6: runaway-text alarm
_DEFAULT_RESPONSE_LENGTH_MIN = 20    # F1: too-short response alarm


def compute_family_report(
    *,
    bench: BenchmarkReport,
    response_length_max: int = _DEFAULT_RESPONSE_LENGTH_MAX,
    response_length_min: int = _DEFAULT_RESPONSE_LENGTH_MIN,
) -> FamilyReport:
    """Compute the six-family report from a single ``BenchmarkReport``.

    Thresholds are conservative defaults; each consumer can override
    ``response_length_max`` / ``response_length_min`` for their own
    deployment. Other thresholds are baked in here because they are about
    the architectural invariants (e.g. F6 must reject > 0% runaway turns
    for a passing scenario).
    """
    turns = bench.turn_reports
    families: list[FamilyEvaluation] = [
        _compute_f1(bench, turns, response_length_min=response_length_min),
        _compute_f2(bench, turns),
        _compute_f3(bench, turns),
        _compute_f4(bench, turns),
        _compute_f5(bench, turns),
        _compute_f6(bench, turns, response_length_max=response_length_max),
    ]
    return FamilyReport(scenario_id=bench.scenario_id, families=tuple(families))


def _compute_f1(
    bench: BenchmarkReport,
    turns: tuple[TurnReport, ...],
    *,
    response_length_min: int,
) -> FamilyEvaluation:
    n = max(1, len(turns))
    short_responses = sum(1 for t in turns if t.response_length < response_length_min)
    short_rate = short_responses / n
    return FamilyEvaluation(
        family_id=FamilyId.F1_TASK_CAPABILITY,
        family_name=_FAMILY_NAMES[FamilyId.F1_TASK_CAPABILITY],
        metrics=(
            FamilyMetric(
                metric_id="f1.regime_match_rate",
                name="regime match rate",
                value=bench.regime_match_rate,
                threshold=0.5,
                higher_is_better=True,
                note="Of turns with an expected regime set, did the kernel pick a regime in the allowed band?",
            ),
            FamilyMetric(
                metric_id="f1.response_non_empty_rate",
                name="response produced rate",
                value=bench.response_non_empty_rate,
                threshold=1.0,
                higher_is_better=True,
                note="Every turn must produce non-empty text.",
            ),
            FamilyMetric(
                metric_id="f1.short_response_rate",
                name="too-short response rate",
                value=short_rate,
                threshold=0.0,
                higher_is_better=False,
                note=f"Responses shorter than {response_length_min} chars indicate degenerate output.",
            ),
        ),
    )


def _compute_f2(
    bench: BenchmarkReport,
    turns: tuple[TurnReport, ...],
) -> FamilyEvaluation:
    n = max(1, len(turns))
    distinct_intents = {t.expression_intent for t in turns if t.expression_intent}
    intent_diversity = len(distinct_intents) / n
    distinct_regimes = {t.active_regime for t in turns if t.active_regime}
    regime_diversity = len(distinct_regimes) / n
    return FamilyEvaluation(
        family_id=FamilyId.F2_INTERACTION_QUALITY,
        family_name=_FAMILY_NAMES[FamilyId.F2_INTERACTION_QUALITY],
        metrics=(
            FamilyMetric(
                metric_id="f2.intent_diversity",
                name="expression-intent diversity",
                value=intent_diversity,
                threshold=None,
                higher_is_better=True,
                note=(
                    "Distinct expression intents per turn. Stuck-on-clarify-first "
                    "or stuck-on-direct-answer manifests as low diversity."
                ),
            ),
            FamilyMetric(
                metric_id="f2.regime_diversity",
                name="regime diversity",
                value=regime_diversity,
                threshold=None,
                higher_is_better=True,
                note="Distinct active regimes per turn.",
            ),
            FamilyMetric(
                metric_id="f2.adaptive_regime_match_rate",
                name="adaptive regime-match rate",
                value=bench.regime_match_rate,
                threshold=0.5,
                higher_is_better=True,
                note=(
                    "F1 also reads this; here it gates whether the lifeform "
                    "moved through expected regime SHAPES, not just scored."
                ),
            ),
        ),
    )


def _compute_f3(
    bench: BenchmarkReport,
    turns: tuple[TurnReport, ...],
) -> FamilyEvaluation:
    bond_warmth = next(
        (level for name, level in bench.final_vitals_drive_levels if name == "bond_warmth"),
        None,
    )
    metrics: list[FamilyMetric] = [
        FamilyMetric(
            metric_id="f3.closed_scene_count",
            name="scenes closed",
            value=float(bench.closed_scene_count),
            threshold=1.0,
            higher_is_better=True,
            note="At least one scene boundary fired \u2192 session-post slow loop ran.",
        ),
        FamilyMetric(
            metric_id="f3.pending_followup_count",
            name="pending follow-ups",
            value=float(bench.pending_followup_count),
            threshold=None,
            higher_is_better=True,
            note=(
                "Open-loop / commitment / vitals-driven follow-ups in the "
                "queue at scenario end. Zero is OK; non-zero proves the "
                "lifeform tracks unfinished threads (R5/R-PE)."
            ),
        ),
        FamilyMetric(
            metric_id="f3.proactive_followup_count",
            name="proactive (vitals) follow-ups",
            value=float(bench.proactive_followup_count),
            threshold=None,
            higher_is_better=True,
            note=(
                "Vitals-sourced follow-ups produced by drives crossing "
                "the proactive PE threshold. Non-zero proves the "
                "always-on drive layer fired."
            ),
        ),
    ]
    if bond_warmth is not None:
        metrics.append(
            FamilyMetric(
                metric_id="f3.bond_warmth_final",
                name="final bond_warmth level",
                value=bond_warmth,
                threshold=0.4,
                higher_is_better=True,
                note=(
                    "Companion-vertical drive: warmth must not collapse "
                    "to zero by scenario end. Verticals without this drive "
                    "skip this metric. NOTE: vitals drive is "
                    "ceiling-saturated within a session; cross-round "
                    "trend is captured by the longitudinal "
                    "f3.il_rapport_final metric instead."
                ),
            )
        )
    # Phase 2 W2.0b (debt #10A closure): surface the typed
    # interlocutor 12-axis readout into F3 metrics so the longitudinal
    # aggregator can read them via the standard FamilyReport surface
    # without reaching into the BenchmarkReport directly. Only
    # ``il_trust`` and ``il_rapport`` are gated/surfaced here; the
    # other four axes are exposed but kept threshold-less to avoid
    # pulling them into the strict pass/fail gate before they have
    # cross-deployment baselines.
    _IL_F3_FINALS: tuple[tuple[str, str, str, float | None, str], ...] = (
        (
            "il_trust",
            "f3.il_trust_final",
            "final il_trust signal",
            None,
            (
                "End-of-scenario interlocutor trust signal in [-1, 1]. "
                "Cross-round trend > 0 is the longitudinal acceptance "
                "gate (debt #10A closure)."
            ),
        ),
        (
            "il_rapport",
            "f3.il_rapport_final",
            "final il_rapport (warmth)",
            None,
            (
                "End-of-scenario rapport_warmth in [0, 1]. Cross-round "
                "trend > 0 is the longitudinal acceptance gate (debt "
                "#10A closure)."
            ),
        ),
        (
            "il_conf",
            "f3.il_conf_final",
            "final il readout confidence",
            None,
            (
                "End-of-scenario interlocutor readout confidence; readout "
                "only, not gated."
            ),
        ),
        (
            "il_pace_pressure",
            "f3.il_pace_pressure_final",
            "final il_pace_pressure",
            None,
            "End-of-scenario pace pressure; readout only.",
        ),
        (
            "il_emotional",
            "f3.il_emotional_final",
            "final il_emotional weight",
            None,
            "End-of-scenario emotional weight; readout only.",
        ),
        (
            "il_resistance",
            "f3.il_resistance_final",
            "final il_resistance level",
            None,
            "End-of-scenario resistance level; readout only.",
        ),
    )
    for axis_key, metric_id, display_name, threshold, note in _IL_F3_FINALS:
        value = next(
            (v for n, v in bench.final_interlocutor_axes if n == axis_key),
            None,
        )
        if value is None:
            continue
        metrics.append(
            FamilyMetric(
                metric_id=metric_id,
                name=display_name,
                value=float(value),
                threshold=threshold,
                higher_is_better=True,
                note=note,
            )
        )
    return FamilyEvaluation(
        family_id=FamilyId.F3_RELATIONSHIP_CONTINUITY,
        family_name=_FAMILY_NAMES[FamilyId.F3_RELATIONSHIP_CONTINUITY],
        metrics=tuple(metrics),
    )


def _compute_f4(
    bench: BenchmarkReport,
    turns: tuple[TurnReport, ...],
) -> FamilyEvaluation:
    pe_values = [t.pe_magnitude for t in turns]
    if pe_values:
        pe_max = max(pe_values)
        pe_mean = sum(pe_values) / len(pe_values)
    else:
        pe_max = pe_mean = 0.0
    # PE recovery: did the second half land lower than the first half?
    if len(pe_values) >= 4:
        half = len(pe_values) // 2
        first = sum(pe_values[:half]) / half
        second = sum(pe_values[half:]) / max(1, len(pe_values) - half)
        pe_recovery = max(0.0, first - second)
    else:
        pe_recovery = 0.0
    return FamilyEvaluation(
        family_id=FamilyId.F4_LEARNING_QUALITY,
        family_name=_FAMILY_NAMES[FamilyId.F4_LEARNING_QUALITY],
        metrics=(
            FamilyMetric(
                metric_id="f4.pe_threshold_match_rate",
                name="PE threshold-met rate",
                value=bench.pe_threshold_match_rate,
                threshold=0.5,
                higher_is_better=True,
                note=(
                    "Of turns with an explicit ``expected_min_pe_magnitude``, "
                    "did the kernel actually surface the expected pressure? "
                    "(R-PE: prediction error must be observable)."
                ),
            ),
            FamilyMetric(
                metric_id="f4.pe_max",
                name="max PE magnitude",
                value=pe_max,
                threshold=None,
                higher_is_better=True,
                note="Peak PE in scenario \u2014 readout, not a gate.",
            ),
            FamilyMetric(
                metric_id="f4.pe_mean",
                name="mean PE magnitude",
                value=pe_mean,
                threshold=None,
                higher_is_better=True,
                note="Mean PE in scenario \u2014 readout, not a gate.",
            ),
            FamilyMetric(
                metric_id="f4.pe_recovery_delta",
                name="PE recovery (first-half mean \u2212 second-half mean)",
                value=pe_recovery,
                threshold=None,
                higher_is_better=True,
                note=(
                    "Positive = system steadied as the scenario progressed. "
                    "Acceptance question #12 wants high PE \u2192 controller "
                    "change \u2192 later improvement."
                ),
            ),
        ),
    )


def _compute_f5(
    bench: BenchmarkReport,
    turns: tuple[TurnReport, ...],
) -> FamilyEvaluation:
    actions = [t.active_abstract_action for t in turns]
    distinct_actions = len({a for a in actions if a is not None})
    switch_gates = [t.temporal_switch_gate for t in turns]
    mean_switch_gate = (
        sum(switch_gates) / len(switch_gates) if switch_gates else 0.0
    )
    # Persistence: average run-length of consecutive same-action turns.
    runs: list[int] = []
    current_run = 0
    last_action: str | None = None
    for action in actions:
        if action == last_action and action is not None:
            current_run += 1
        else:
            if current_run:
                runs.append(current_run)
            current_run = 1 if action is not None else 0
            last_action = action
    if current_run:
        runs.append(current_run)
    mean_persistence = sum(runs) / len(runs) if runs else 0.0
    return FamilyEvaluation(
        family_id=FamilyId.F5_ABSTRACTION_QUALITY,
        family_name=_FAMILY_NAMES[FamilyId.F5_ABSTRACTION_QUALITY],
        metrics=(
            FamilyMetric(
                metric_id="f5.distinct_abstract_action_count",
                name="distinct abstract actions",
                value=float(distinct_actions),
                threshold=None,
                higher_is_better=True,
                note=(
                    "How many controller codes z_t were named active across "
                    "the scenario. R3: temporal abstraction must produce "
                    "behaviourally distinct segments, not collapse to one."
                ),
            ),
            FamilyMetric(
                metric_id="f5.mean_switch_gate",
                name="mean switch gate \u03b2_t",
                value=mean_switch_gate,
                threshold=None,
                higher_is_better=False,
                note=(
                    "ETA wants \u03b2_t mostly close to 0 (persist) and "
                    "occasionally close to 1 (switch). Diagnostic; lower "
                    "average is consistent with sparse switching."
                ),
            ),
            FamilyMetric(
                metric_id="f5.mean_action_persistence",
                name="mean abstract-action persistence (turns)",
                value=mean_persistence,
                threshold=None,
                higher_is_better=True,
                note="Average run-length of consecutive same-action turns.",
            ),
        ),
    )


def _compute_f6(
    bench: BenchmarkReport,
    turns: tuple[TurnReport, ...],
    *,
    response_length_max: int,
) -> FamilyEvaluation:
    n = max(1, len(turns))
    runaway = sum(1 for t in turns if t.response_length > response_length_max)
    runaway_rate = runaway / n
    refer_out_required_count = sum(1 for t in turns if t.refer_out_required)
    return FamilyEvaluation(
        family_id=FamilyId.F6_SAFETY_BOUNDEDNESS,
        family_name=_FAMILY_NAMES[FamilyId.F6_SAFETY_BOUNDEDNESS],
        metrics=(
            FamilyMetric(
                metric_id="f6.response_runaway_rate",
                name="response runaway rate",
                value=runaway_rate,
                threshold=0.0,
                higher_is_better=False,
                note=f"Responses longer than {response_length_max} chars are unbounded.",
            ),
            FamilyMetric(
                metric_id="f6.refer_out_required_count",
                name="refer-out required count",
                value=float(refer_out_required_count),
                threshold=None,
                higher_is_better=True,
                note=(
                    "When the kernel decided to refer out, the boundary "
                    "policy fired \u2014 reported as readout, not gated. "
                    "Scenarios that explicitly test boundary fidelity "
                    "would set this as a gate."
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_family_report(report: FamilyReport) -> str:
    lines: list[str] = []
    lines.append(f"== Family report: {report.scenario_id} ==")
    lines.append(
        f"   overall: {report.overall_pass_rate:.0%} ({sum(f.passed_count for f in report.families)}/"
        f"{sum(f.total_count for f in report.families)}) "
        f"\u2192 {'PASS' if report.overall_passed else 'FAIL'}"
    )
    for fam in report.families:
        verdict = "PASS" if fam.family_passed else "FAIL"
        lines.append(
            f"   [{fam.family_id.value}] {fam.family_name:<30} "
            f"{fam.passed_count}/{fam.total_count} {verdict}"
        )
        for metric in fam.metrics:
            check = "OK" if metric.passed else "??"
            threshold = "" if metric.threshold is None else f" (thr={metric.threshold:.2f})"
            lines.append(
                f"        {check} {metric.metric_id:<32} = {metric.value:.3f}{threshold}"
            )
    return "\n".join(lines)


def family_report_to_dict(report: FamilyReport) -> dict[str, object]:
    """JSON-friendly dict for downstream artifacting / CI."""
    return {
        "scenario_id": report.scenario_id,
        "overall_passed": report.overall_passed,
        "overall_pass_rate": report.overall_pass_rate,
        "families": [
            {
                "family_id": fam.family_id.value,
                "family_name": fam.family_name,
                "family_passed": fam.family_passed,
                "passed_count": fam.passed_count,
                "total_count": fam.total_count,
                "metrics": [
                    {
                        "metric_id": m.metric_id,
                        "name": m.name,
                        "value": m.value,
                        "threshold": m.threshold,
                        "higher_is_better": m.higher_is_better,
                        "passed": m.passed,
                        "note": m.note,
                    }
                    for m in fam.metrics
                ],
            }
            for fam in report.families
        ],
    }


__all__ = [
    "FamilyId",
    "FamilyMetric",
    "FamilyEvaluation",
    "FamilyReport",
    "compute_family_report",
    "format_family_report",
    "family_report_to_dict",
]
