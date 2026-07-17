"""Latent-semantic grounding evidence readout (experiment 1 of
``docs/specs/semantic-grounding-evidence.md``).

Answers: do latent action families correspond to real semantic dynamics
(discrimination / lead / transfer), or are they surface-wording clusters?

Boundaries (R8 / R12 / R15):

* Read-only, out-of-turn analysis over already-published snapshots
  (``temporal_abstraction`` / ``prediction_error`` / semantic owners /
  ``common_ground``). No owner state is reconstructed or mutated.
* Semantic deltas come exclusively from typed owner snapshot diffs;
  never from transcript text.
* The report is a non-gating reference artifact. It does not enter any
  reward path and does not change ``evaluate_learned_active_candidate``
  gates (promotion-gate integration requires its own convergence packet).

Because soak artifacts do not retain per-turn snapshot sequences, this
module ships a small capture helper (`SemanticGroundingTurnCapture`)
that extracts a compact per-turn evidence record from each
``AgentTurnResult.active_snapshots``. The analysis then runs offline
over the captured sequence (in memory or via JSON round-trip).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from volvence_zero.temporal_types import abstract_action_family_id

SEMANTIC_GROUNDING_REPORT_SCHEMA_VERSION = "semantic-grounding-report.v1"
SEMANTIC_GROUNDING_TURNS_SCHEMA_VERSION = "semantic-grounding-turns.v1"

#: Typed owner-snapshot fields whose per-turn diff forms the semantic
#: delta vector. Every axis reads an exact published field (schema drift
#: fails loudly via AttributeError). Order is frozen: it defines the
#: vector layout of ``SemanticGroundingTurnEvidence.semantic_delta``.
SEMANTIC_DELTA_AXES: tuple[str, ...] = (
    "relationship_state.trust_level",
    "relationship_state.repair_pressure",
    "relationship_state.emotional_load",
    "relationship_state.unresolved_tension_count",
    "relationship_state.recent_repair_count",
    "commitment.active_commitments_count",
    "commitment.at_risk_commitments_count",
    "commitment.outcome_completed_count",
    "commitment.alignment_reject_count",
    "open_loop.unresolved_loops_count",
    "open_loop.closure_refs_count",
    "open_loop.closure_pressure",
    "common_ground.dyad_atoms_count",
    "common_ground.group_atoms_count",
    "user_model.stable_preferences_count",
    "user_model.sensitive_boundaries_count",
    "boundary_consent.granted_consents_count",
    "boundary_consent.denied_boundaries_count",
    "goal_value.explicit_goals_count",
    "goal_value.value_conflict",
    "belief_assumption.beliefs_count",
    "belief_assumption.contradiction_refs_count",
    "plan_intent.plan_revision_count",
    "plan_intent.completed_plan_refs_count",
    "execution_result.completed_actions_count",
    "execution_result.failed_actions_count",
)

_SlotReader = Callable[[Any], dict[str, float]]

#: Per-slot typed feature readers. Direct attribute access on the
#: published snapshot value; a missing field means contract drift and
#: must raise, not default (no-swallow rule).
_SLOT_FEATURE_READERS: dict[str, _SlotReader] = {
    "relationship_state": lambda v: {
        "trust_level": float(v.trust_level),
        "repair_pressure": float(v.repair_pressure),
        "emotional_load": float(v.emotional_load),
        "unresolved_tension_count": float(v.unresolved_tension_count),
        "recent_repair_count": float(v.recent_repair_count),
    },
    "commitment": lambda v: {
        "active_commitments_count": float(len(v.active_commitments)),
        "at_risk_commitments_count": float(len(v.at_risk_commitments)),
        "outcome_completed_count": float(v.outcome_completed_count),
        "alignment_reject_count": float(v.alignment_reject_count),
    },
    "open_loop": lambda v: {
        "unresolved_loops_count": float(len(v.unresolved_loops)),
        "closure_refs_count": float(len(v.closure_refs)),
        "closure_pressure": float(v.closure_pressure),
    },
    "common_ground": lambda v: {
        "dyad_atoms_count": float(len(v.dyad_atoms)),
        "group_atoms_count": float(len(v.group_atoms)),
    },
    "user_model": lambda v: {
        "stable_preferences_count": float(len(v.stable_preferences)),
        "sensitive_boundaries_count": float(len(v.sensitive_boundaries)),
    },
    "boundary_consent": lambda v: {
        "granted_consents_count": float(len(v.granted_consents)),
        "denied_boundaries_count": float(len(v.denied_boundaries)),
    },
    "goal_value": lambda v: {
        "explicit_goals_count": float(len(v.explicit_goals)),
        "value_conflict": float(v.value_conflict),
    },
    "belief_assumption": lambda v: {
        "beliefs_count": float(len(v.beliefs)),
        "contradiction_refs_count": float(len(v.contradiction_refs)),
    },
    "plan_intent": lambda v: {
        "plan_revision_count": float(v.plan_revision_count),
        "completed_plan_refs_count": float(len(v.completed_plan_refs)),
    },
    "execution_result": lambda v: {
        "completed_actions_count": float(len(v.completed_actions)),
        "failed_actions_count": float(len(v.failed_actions)),
    },
}


class SemanticGroundingError(RuntimeError):
    """Raised when grounding evidence input violates the contract."""


@dataclass(frozen=True)
class SemanticGroundingTurnEvidence:
    """Compact per-turn record extracted from published snapshots."""

    turn_index: int
    case_id: str
    active_abstract_action: str
    action_family: str
    switch_gate: float
    is_switching: bool
    newly_closed_segment_ids: tuple[str, ...]
    pe_magnitude: float
    pe_segment_id: str
    semantic_delta: tuple[float, ...]
    covered_slots: tuple[str, ...]

    @property
    def has_semantic_delta(self) -> bool:
        return any(value != 0.0 for value in self.semantic_delta)


class SemanticGroundingTurnCapture:
    """Stateful extractor: feed each turn's ``active_snapshots``.

    Holds the previous turn's feature vector (for the diff) and the set
    of already-seen segment ids (``closed_segments`` accumulates in the
    temporal snapshot, we only report newly closed ones).
    """

    def __init__(self) -> None:
        self._previous_features: dict[str, float] | None = None
        self._seen_segment_ids: set[str] = set()
        self._turns: list[SemanticGroundingTurnEvidence] = []

    @property
    def turns(self) -> tuple[SemanticGroundingTurnEvidence, ...]:
        return tuple(self._turns)

    def observe_turn(
        self,
        *,
        turn_index: int,
        active_snapshots: Mapping[str, Any],
        case_id: str = "default",
    ) -> SemanticGroundingTurnEvidence:
        features, covered_slots = _semantic_features(active_snapshots)
        if self._previous_features is None:
            delta = tuple(0.0 for _ in SEMANTIC_DELTA_AXES)
        else:
            delta = tuple(
                features[axis] - self._previous_features[axis]
                for axis in SEMANTIC_DELTA_AXES
            )
        self._previous_features = features

        temporal_snapshot = active_snapshots.get("temporal_abstraction")
        if temporal_snapshot is None:
            raise SemanticGroundingError(
                "active_snapshots is missing 'temporal_abstraction'; grounding "
                "evidence requires the temporal owner snapshot every turn."
            )
        temporal_value = temporal_snapshot.value
        action_id = temporal_value.active_abstract_action
        family = abstract_action_family_id(action_id) or action_id

        new_segment_ids: list[str] = []
        for closure in temporal_value.closed_segments:
            if closure.segment_id not in self._seen_segment_ids:
                self._seen_segment_ids.add(closure.segment_id)
                new_segment_ids.append(closure.segment_id)

        pe_snapshot = active_snapshots.get("prediction_error")
        if pe_snapshot is None:
            pe_magnitude = 0.0
            pe_segment_id = ""
        else:
            pe_value = pe_snapshot.value
            pe_magnitude = float(pe_value.error.magnitude)
            pe_segment_id = pe_value.action_context.segment_id

        evidence = SemanticGroundingTurnEvidence(
            turn_index=turn_index,
            case_id=case_id,
            active_abstract_action=action_id,
            action_family=family,
            switch_gate=float(temporal_value.controller_state.switch_gate),
            is_switching=bool(temporal_value.controller_state.is_switching),
            newly_closed_segment_ids=tuple(new_segment_ids),
            pe_magnitude=pe_magnitude,
            pe_segment_id=pe_segment_id,
            semantic_delta=delta,
            covered_slots=covered_slots,
        )
        self._turns.append(evidence)
        return evidence


def _semantic_features(
    active_snapshots: Mapping[str, Any],
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Read typed feature values for every axis; absent slots yield 0.0
    for their axes and are excluded from coverage."""

    features: dict[str, float] = {axis: 0.0 for axis in SEMANTIC_DELTA_AXES}
    covered: list[str] = []
    for slot, reader in _SLOT_FEATURE_READERS.items():
        snapshot = active_snapshots.get(slot)
        if snapshot is None:
            continue
        covered.append(slot)
        for name, value in reader(snapshot.value).items():
            features[f"{slot}.{name}"] = value
    return features, tuple(covered)


# ---------------------------------------------------------------------------
# JSON round-trip for captured turn evidence
# ---------------------------------------------------------------------------


def turn_evidence_to_payload(
    turns: Sequence[SemanticGroundingTurnEvidence],
) -> dict[str, Any]:
    return {
        "schema_version": SEMANTIC_GROUNDING_TURNS_SCHEMA_VERSION,
        "artifact_kind": "semantic_grounding_turns",
        "axes": list(SEMANTIC_DELTA_AXES),
        "turns": [asdict(turn) for turn in turns],
    }


def turn_evidence_from_payload(
    payload: Mapping[str, Any],
) -> tuple[SemanticGroundingTurnEvidence, ...]:
    schema = payload["schema_version"]
    if schema != SEMANTIC_GROUNDING_TURNS_SCHEMA_VERSION:
        raise SemanticGroundingError(
            f"turn-evidence schema_version mismatch: got {schema!r}, expected "
            f"{SEMANTIC_GROUNDING_TURNS_SCHEMA_VERSION!r}."
        )
    axes = tuple(payload["axes"])
    if axes != SEMANTIC_DELTA_AXES:
        raise SemanticGroundingError(
            "turn-evidence axes do not match the current SEMANTIC_DELTA_AXES "
            "layout; re-capture the turns with this code version instead of "
            "silently re-interpreting the vector."
        )
    turns: list[SemanticGroundingTurnEvidence] = []
    for entry in payload["turns"]:
        turns.append(
            SemanticGroundingTurnEvidence(
                turn_index=int(entry["turn_index"]),
                case_id=str(entry["case_id"]),
                active_abstract_action=str(entry["active_abstract_action"]),
                action_family=str(entry["action_family"]),
                switch_gate=float(entry["switch_gate"]),
                is_switching=bool(entry["is_switching"]),
                newly_closed_segment_ids=tuple(entry["newly_closed_segment_ids"]),
                pe_magnitude=float(entry["pe_magnitude"]),
                pe_segment_id=str(entry["pe_segment_id"]),
                semantic_delta=tuple(
                    float(v) for v in entry["semantic_delta"]
                ),
                covered_slots=tuple(entry["covered_slots"]),
            )
        )
    return tuple(turns)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingThresholds:
    """Coverage + verdict thresholds (spec defaults)."""

    min_closed_segments: int = 50
    min_reused_families: int = 3
    min_nonzero_delta_ratio: float = 0.3
    shuffle_count: int = 200
    shuffle_seed: int = 7
    d2_max_lag: int = 3
    d3_margin: float = 0.1
    d3_min_samples_per_signature: int = 2


@dataclass(frozen=True)
class ShuffledControlSummary:
    real_value: float
    shuffled_mean: float
    shuffled_p95: float
    shuffle_count: int
    passed: bool
    description: str


@dataclass(frozen=True)
class D2LeadSummary:
    peak_lag: int
    peak_value: float
    shuffled_p95: float
    coupling_ok: bool
    lead_ok: bool
    passed: bool
    lag_correlations: tuple[tuple[int, float], ...]
    description: str


@dataclass(frozen=True)
class D3TransferSummary:
    same_family_similarity: float
    cross_family_similarity: float
    margin: float
    evaluable: bool
    passed: bool
    same_family_pair_count: int
    cross_family_pair_count: int
    description: str


@dataclass(frozen=True)
class FamilySignature:
    family: str
    sample_count: int
    case_count: int
    centroid: tuple[float, ...]


@dataclass(frozen=True)
class GroundingCoverage:
    turn_count: int
    closed_segment_count: int
    family_count: int
    reused_family_count: int
    nonzero_delta_ratio: float
    meets_thresholds: bool
    description: str


@dataclass(frozen=True)
class SemanticGroundingReport:
    schema_version: str
    artifact_kind: str
    non_gating: bool
    axes: tuple[str, ...]
    thresholds: GroundingThresholds
    coverage: GroundingCoverage
    d1_discrimination: ShuffledControlSummary
    d2_lead: D2LeadSummary
    d3_transfer: D3TransferSummary
    family_signatures: tuple[FamilySignature, ...]
    verdict: str
    description: str

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["axes"] = list(self.axes)
        return payload


def build_semantic_grounding_report(
    turns: Sequence[SemanticGroundingTurnEvidence],
    *,
    thresholds: GroundingThresholds | None = None,
) -> SemanticGroundingReport:
    """Compute D1/D2/D3 with shuffled controls over captured turns."""

    resolved = thresholds or GroundingThresholds()
    if not turns:
        raise SemanticGroundingError(
            "build_semantic_grounding_report requires at least one captured "
            "turn; got an empty sequence."
        )

    standardized = _standardize_deltas([t.semantic_delta for t in turns])
    families = [t.action_family for t in turns]

    coverage = _coverage(turns, resolved)
    rng = random.Random(resolved.shuffle_seed)

    d1 = _d1_discrimination(
        standardized, families, rng=rng, shuffle_count=resolved.shuffle_count
    )
    d2 = _d2_lead(
        turns,
        rng=rng,
        shuffle_count=resolved.shuffle_count,
        max_lag=resolved.d2_max_lag,
    )
    d3, signatures = _d3_transfer(
        turns,
        standardized,
        margin=resolved.d3_margin,
        min_samples=resolved.d3_min_samples_per_signature,
    )

    if not coverage.meets_thresholds:
        verdict = "insufficient-coverage"
    elif d1.passed and d2.passed and d3.passed:
        verdict = "retain"
    elif d1.passed:
        verdict = "weak"
    else:
        verdict = "fail"

    return SemanticGroundingReport(
        schema_version=SEMANTIC_GROUNDING_REPORT_SCHEMA_VERSION,
        artifact_kind="semantic_grounding_report",
        non_gating=True,
        axes=SEMANTIC_DELTA_AXES,
        thresholds=resolved,
        coverage=coverage,
        d1_discrimination=d1,
        d2_lead=d2,
        d3_transfer=d3,
        family_signatures=signatures,
        verdict=verdict,
        description=(
            "Latent-semantic grounding readout (non-gating). Verdict "
            f"{verdict!r}: D1 passed={d1.passed}, D2 passed={d2.passed}, "
            f"D3 passed={d3.passed}, coverage met="
            f"{coverage.meets_thresholds}. A 'fail' verdict is a kill "
            "signal for the 'emergent abstraction is grounded' claim and "
            "must be reported as-is."
        ),
    )


def _standardize_deltas(
    deltas: Sequence[tuple[float, ...]],
) -> list[tuple[float, ...]]:
    """Per-axis z-standardization so count axes do not dominate float axes.

    Zero-variance axes stay zero (uninformative for separation).
    """

    n_axes = len(SEMANTIC_DELTA_AXES)
    n = len(deltas)
    means = [sum(d[i] for d in deltas) / n for i in range(n_axes)]
    stds = []
    for i in range(n_axes):
        variance = sum((d[i] - means[i]) ** 2 for d in deltas) / n
        stds.append(math.sqrt(variance))
    standardized: list[tuple[float, ...]] = []
    for d in deltas:
        standardized.append(
            tuple(
                (d[i] - means[i]) / stds[i] if stds[i] > 0.0 else 0.0
                for i in range(n_axes)
            )
        )
    return standardized


def _coverage(
    turns: Sequence[SemanticGroundingTurnEvidence],
    thresholds: GroundingThresholds,
) -> GroundingCoverage:
    closed_segments = sum(len(t.newly_closed_segment_ids) for t in turns)
    families = {t.action_family for t in turns}
    family_cases: dict[str, set[str]] = {}
    for t in turns:
        family_cases.setdefault(t.action_family, set()).add(t.case_id)
    reused = sum(1 for cases in family_cases.values() if len(cases) >= 2)
    nonzero_ratio = (
        sum(1 for t in turns if t.has_semantic_delta) / len(turns)
    )
    meets = (
        closed_segments >= thresholds.min_closed_segments
        and reused >= thresholds.min_reused_families
        and nonzero_ratio >= thresholds.min_nonzero_delta_ratio
    )
    return GroundingCoverage(
        turn_count=len(turns),
        closed_segment_count=closed_segments,
        family_count=len(families),
        reused_family_count=reused,
        nonzero_delta_ratio=round(nonzero_ratio, 6),
        meets_thresholds=meets,
        description=(
            f"{len(turns)} turns, {closed_segments} closed segments, "
            f"{len(families)} families ({reused} reused across cases), "
            f"nonzero-delta ratio {nonzero_ratio:.3f}."
        ),
    )


def _separation(
    deltas: Sequence[tuple[float, ...]],
    labels: Sequence[str],
) -> float:
    """Between-group / within-group mean squared distance ratio."""

    n = len(deltas)
    n_axes = len(SEMANTIC_DELTA_AXES)
    overall = tuple(sum(d[i] for d in deltas) / n for i in range(n_axes))
    groups: dict[str, list[tuple[float, ...]]] = {}
    for d, label in zip(deltas, labels):
        groups.setdefault(label, []).append(d)

    between = 0.0
    within = 0.0
    for members in groups.values():
        size = len(members)
        centroid = tuple(
            sum(m[i] for m in members) / size for i in range(n_axes)
        )
        between += size * _sq_distance(centroid, overall)
        within += sum(_sq_distance(m, centroid) for m in members)
    between /= n
    within /= n
    if within <= 0.0:
        return 0.0
    return between / within


def _sq_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _d1_discrimination(
    standardized: Sequence[tuple[float, ...]],
    families: Sequence[str],
    *,
    rng: random.Random,
    shuffle_count: int,
) -> ShuffledControlSummary:
    real = _separation(standardized, families)
    shuffled_values: list[float] = []
    labels = list(families)
    for _ in range(shuffle_count):
        rng.shuffle(labels)
        shuffled_values.append(_separation(standardized, labels))
    shuffled_mean = (
        sum(shuffled_values) / len(shuffled_values) if shuffled_values else 0.0
    )
    shuffled_p95 = _percentile(shuffled_values, 0.95)
    distinct_families = len(set(families))
    passed = distinct_families >= 2 and real > shuffled_p95 and real > 0.0
    return ShuffledControlSummary(
        real_value=round(real, 6),
        shuffled_mean=round(shuffled_mean, 6),
        shuffled_p95=round(shuffled_p95, 6),
        shuffle_count=shuffle_count,
        passed=passed,
        description=(
            "D1 discrimination: between/within separation of semantic delta "
            f"vectors grouped by family. real={real:.4f}, shuffled "
            f"p95={shuffled_p95:.4f}, families={distinct_families}."
        ),
    )


def _lag_correlation(
    switch_series: Sequence[int],
    semantic_series: Sequence[int],
    lag: int,
) -> float:
    n = len(switch_series)
    hits = 0
    for t in range(n):
        u = t + lag
        if 0 <= u < n and switch_series[t] == 1 and semantic_series[u] == 1:
            hits += 1
    total_switch = sum(switch_series)
    total_semantic = sum(semantic_series)
    if total_switch == 0 or total_semantic == 0:
        return 0.0
    return hits / math.sqrt(total_switch * total_semantic)


def _d2_lead(
    turns: Sequence[SemanticGroundingTurnEvidence],
    *,
    rng: random.Random,
    shuffle_count: int,
    max_lag: int,
) -> D2LeadSummary:
    switch_series = [
        1 if (t.is_switching or t.newly_closed_segment_ids) else 0
        for t in turns
    ]
    semantic_series = [1 if t.has_semantic_delta else 0 for t in turns]

    lag_values: list[tuple[int, float]] = []
    for lag in range(-max_lag, max_lag + 1):
        lag_values.append(
            (lag, round(_lag_correlation(switch_series, semantic_series, lag), 6))
        )
    peak_lag, peak_value = max(lag_values, key=lambda item: item[1])

    n = len(switch_series)
    shuffled_peaks: list[float] = []
    for _ in range(shuffle_count):
        offset = rng.randrange(1, n) if n > 1 else 0
        shifted = switch_series[offset:] + switch_series[:offset]
        best = max(
            _lag_correlation(shifted, semantic_series, lag)
            for lag in range(-max_lag, max_lag + 1)
        )
        shuffled_peaks.append(best)
    shuffled_p95 = _percentile(shuffled_peaks, 0.95)

    coupling_ok = peak_value > shuffled_p95 and peak_value > 0.0
    lead_ok = peak_lag >= 0
    return D2LeadSummary(
        peak_lag=peak_lag,
        peak_value=peak_value,
        shuffled_p95=round(shuffled_p95, 6),
        coupling_ok=coupling_ok,
        lead_ok=lead_ok,
        passed=coupling_ok and lead_ok,
        lag_correlations=tuple(lag_values),
        description=(
            "D2 lead: switch events (beta closure / is_switching) vs semantic "
            f"delta events. peak lag={peak_lag} (>=0 means switch leads or is "
            f"synchronous), peak={peak_value:.4f}, shuffled-timing "
            f"p95={shuffled_p95:.4f}."
        ),
    )


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _d3_transfer(
    turns: Sequence[SemanticGroundingTurnEvidence],
    standardized: Sequence[tuple[float, ...]],
    *,
    margin: float,
    min_samples: int,
) -> tuple[D3TransferSummary, tuple[FamilySignature, ...]]:
    n_axes = len(SEMANTIC_DELTA_AXES)

    per_case_family: dict[tuple[str, str], list[tuple[float, ...]]] = {}
    for turn, delta in zip(turns, standardized):
        per_case_family.setdefault(
            (turn.case_id, turn.action_family), []
        ).append(delta)

    signatures: dict[tuple[str, str], tuple[float, ...]] = {}
    for key, members in per_case_family.items():
        if len(members) < min_samples:
            continue
        signatures[key] = tuple(
            sum(m[i] for m in members) / len(members) for i in range(n_axes)
        )

    same_sims: list[float] = []
    cross_sims: list[float] = []
    keys = sorted(signatures)
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            case_a, family_a = key_a
            case_b, family_b = key_b
            if case_a == case_b:
                continue
            sim = _cosine(signatures[key_a], signatures[key_b])
            if family_a == family_b:
                same_sims.append(sim)
            else:
                cross_sims.append(sim)

    evaluable = bool(same_sims) and bool(cross_sims)
    same_mean = sum(same_sims) / len(same_sims) if same_sims else 0.0
    cross_mean = sum(cross_sims) / len(cross_sims) if cross_sims else 0.0
    passed = evaluable and same_mean > cross_mean + margin

    family_totals: dict[str, list[tuple[str, tuple[float, ...]]]] = {}
    for (case_id, family), centroid in signatures.items():
        family_totals.setdefault(family, []).append((case_id, centroid))
    signature_records: list[FamilySignature] = []
    for family in sorted(family_totals):
        entries = family_totals[family]
        centroid = tuple(
            sum(c[i] for _, c in entries) / len(entries) for i in range(n_axes)
        )
        sample_count = sum(
            len(per_case_family[(case_id, family)]) for case_id, _ in entries
        )
        signature_records.append(
            FamilySignature(
                family=family,
                sample_count=sample_count,
                case_count=len(entries),
                centroid=tuple(round(v, 6) for v in centroid),
            )
        )

    summary = D3TransferSummary(
        same_family_similarity=round(same_mean, 6),
        cross_family_similarity=round(cross_mean, 6),
        margin=margin,
        evaluable=evaluable,
        passed=passed,
        same_family_pair_count=len(same_sims),
        cross_family_pair_count=len(cross_sims),
        description=(
            "D3 transfer: cross-case signature similarity. same-family "
            f"mean={same_mean:.4f} vs cross-family mean={cross_mean:.4f} "
            f"(margin {margin}); evaluable={evaluable} "
            f"({len(same_sims)} same-family pairs, {len(cross_sims)} "
            "cross-family pairs across distinct cases)."
        ),
    )
    return summary, tuple(signature_records)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


__all__ = [
    "SEMANTIC_DELTA_AXES",
    "SEMANTIC_GROUNDING_REPORT_SCHEMA_VERSION",
    "SEMANTIC_GROUNDING_TURNS_SCHEMA_VERSION",
    "D2LeadSummary",
    "D3TransferSummary",
    "FamilySignature",
    "GroundingCoverage",
    "GroundingThresholds",
    "SemanticGroundingError",
    "SemanticGroundingReport",
    "SemanticGroundingTurnCapture",
    "SemanticGroundingTurnEvidence",
    "ShuffledControlSummary",
    "build_semantic_grounding_report",
    "turn_evidence_from_payload",
    "turn_evidence_to_payload",
]
