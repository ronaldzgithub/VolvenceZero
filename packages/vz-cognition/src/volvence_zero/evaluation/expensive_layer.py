"""Expensive layer of the evaluation cascade (architecture-uplift A2 step 3).

Implements the schema + module skeleton defined in
[`docs/specs/evaluation-cascade.md`](../../../../../../../docs/specs/evaluation-cascade.md)
§A2.3.

Key invariants (spec §关键不变量 3):

- LLM-judge ``readouts`` are advisory only; ``is_gate_eligible`` is
  permanently ``False`` (enforced by ``tests/contracts/test_evaluation_cascade_readouts.py``)
- Plugged into the cascade via dependency on ``evaluation_mid``
- DISABLED by default; promotion gated by ETA strong-proof PASS

Phased implementation:

- **T9**: schemas + module skeleton + ``is_gate_eligible`` constant.
- **C4 (2026-07-16, this packet)**: the tier is no longer an empty shell:
  ``process()`` re-emits the mid tier's aggregated scores, derives a
  substrate-geometry readout (persona-vector style monitoring entrance,
  read-only over the published ``SubstrateSnapshot``), and publishes any
  LLM-judge readouts produced through the injectable
  ``LlmJudgeBackend`` seam (centralised prompt, zero LLM calls unless a
  backend is injected AND cases are submitted). Wiring stays DISABLED by
  default; promotion is still gated by ETA strong-proof PASS.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from volvence_zero.evaluation.cheap_layer import EvaluationCascadeRole
from volvence_zero.evaluation.mid_layer import MidLayerScore, MidLayerSnapshot
from volvence_zero.evaluation.prompts import (
    LLM_JUDGE_SYSTEM_PROMPT,
    build_llm_judge_user_prompt,
)
from volvence_zero.runtime.kernel import (
    RuntimeModule,
    Snapshot,
    WiringLevel,
)
from volvence_zero.substrate import SubstrateSnapshot

__all__ = [
    "HeadToHeadResult",
    "LlmJudgeReadout",
    "LlmJudgeBackend",
    "LlmJudgeCase",
    "ExpensiveLayerSnapshot",
    "ExpensiveLayerModule",
    "build_deterministic_head_to_head_snapshot",
    "build_substrate_geometry_scores",
    "run_llm_judge_readouts",
]


@dataclasses.dataclass(frozen=True)
class HeadToHeadResult:
    """DM-7 / EVO-6 head-to-head winrate (spec §A2.3)."""

    profile_a: str
    profile_b: str
    case_count: int
    winrate_a_vs_b: float
    confidence_interval_low: float
    confidence_interval_high: float
    judge_kind: str  # "deterministic" / "llm-readout"
    notes: str


@dataclasses.dataclass(frozen=True)
class LlmJudgeReadout:
    """LLM-judge readout (spec §A2.3).

    ``is_gate_eligible`` is permanently ``False`` (R12 + OA-2 isolation).
    Contract test ``test_llm_judge_readout_never_gate_eligible`` enforces
    this at module load.
    """

    case_id: str
    judge_model_id: str
    naturalness_score: float
    coherence_score: float
    note: str
    is_gate_eligible: bool = False


@dataclasses.dataclass(frozen=True)
class ExpensiveLayerSnapshot:
    """Expensive-tier evaluation snapshot."""

    generation_id: str
    head_to_head_results: tuple[HeadToHeadResult, ...]
    llm_judge_readouts: tuple[LlmJudgeReadout, ...]
    aggregated_scores: tuple[MidLayerScore, ...]
    description: str
    cascade_role: EvaluationCascadeRole = EvaluationCascadeRole.EXPENSIVE_LAYER


_EMPTY_EXPENSIVE_SNAPSHOT = ExpensiveLayerSnapshot(
    generation_id="",
    head_to_head_results=(),
    llm_judge_readouts=(),
    aggregated_scores=(),
    description="expensive_layer: upstream mid snapshot unavailable",
)


# ---------------------------------------------------------------------------
# LLM-judge seam (C4). Zero LLM calls unless a backend is injected.
# ---------------------------------------------------------------------------


class LlmJudgeBackend(Protocol):
    """Injectable LLM caller for judge readouts.

    Receives the centralised system + user prompts and returns the raw
    model reply. It must NOT be given any gate authority — readouts built
    from it are permanently ``is_gate_eligible=False``.
    """

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


@dataclasses.dataclass(frozen=True)
class LlmJudgeCase:
    """One dialogue case submitted for judge readout."""

    case_id: str
    dialogue_context: str
    response_text: str


def _parse_judge_reply(reply: str) -> tuple[float, float, str]:
    """Parse the judge protocol line ``naturalness=x coherence=y note=...``.

    Protocol parsing (exact API contract, not keyword-driven logic).
    Malformed replies fail loudly — an LLM failure must not silently
    become a missing readout (spec §F: expensive_layer LLM 失败 → 向上抛).
    """

    line = reply.strip().splitlines()[0] if reply.strip() else ""
    if not line.startswith("naturalness="):
        raise ValueError(f"malformed judge reply (missing naturalness=): {reply!r}")
    try:
        rest = line[len("naturalness="):]
        naturalness_text, rest = rest.split(" coherence=", 1)
        coherence_text, _, note = rest.partition(" note=")
        naturalness = float(naturalness_text)
        coherence = float(coherence_text)
    except (ValueError, IndexError) as error:
        raise ValueError(f"malformed judge reply: {reply!r}") from error
    if not (0.0 <= naturalness <= 1.0 and 0.0 <= coherence <= 1.0):
        raise ValueError(f"judge scores out of [0, 1]: {reply!r}")
    return naturalness, coherence, note.strip()


def run_llm_judge_readouts(
    *,
    backend: LlmJudgeBackend,
    judge_model_id: str,
    cases: Sequence[LlmJudgeCase],
) -> tuple[LlmJudgeReadout, ...]:
    """Produce gate-ineligible judge readouts via the centralised prompt."""

    readouts: list[LlmJudgeReadout] = []
    for case in cases:
        reply = backend.complete(
            system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
            user_prompt=build_llm_judge_user_prompt(
                dialogue_context=case.dialogue_context,
                response_text=case.response_text,
            ),
        )
        naturalness, coherence, note = _parse_judge_reply(reply)
        readouts.append(
            LlmJudgeReadout(
                case_id=case.case_id,
                judge_model_id=judge_model_id,
                naturalness_score=naturalness,
                coherence_score=coherence,
                note=note,
            )
        )
    return tuple(readouts)


# ---------------------------------------------------------------------------
# Substrate geometry readout (C4): persona-vector style monitoring entrance.
# ---------------------------------------------------------------------------


def _mean_activation(steps: Sequence[Any]) -> tuple[float, ...]:
    vectors = [
        activation.activation
        for step in steps
        for activation in step.residual_activations
        if activation.activation
    ]
    if not vectors:
        return ()
    width = min(len(vector) for vector in vectors)
    return tuple(
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(width)
    )


def _l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    width = min(len(a), len(b))
    if width == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(width))
    norm_a = _l2_norm(a[:width])
    norm_b = _l2_norm(b[:width])
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - dot / (norm_a * norm_b)))


def build_substrate_geometry_scores(
    substrate_snapshot: SubstrateSnapshot,
) -> tuple[MidLayerScore, ...]:
    """Persona-vector style geometry readout over the residual stream.

    Read-only over the substrate owner's published snapshot (R8): the
    tier does not touch the model. Two monitoring metrics:

    - ``persona_geometry_norm``: mean L2 norm of the residual mean
      activation over the turn (activation-magnitude drift entrance).
    - ``persona_geometry_drift``: cosine distance between the residual
      mean activation of the first and second half of the turn's
      residual sequence (within-turn persona drift entrance).
    """

    steps = substrate_snapshot.residual_sequence
    if not steps:
        return ()
    overall_mean = _mean_activation(steps)
    if not overall_mean:
        return ()
    midpoint = max(1, len(steps) // 2)
    first_mean = _mean_activation(steps[:midpoint])
    second_mean = _mean_activation(steps[midpoint:])
    drift = _cosine_distance(first_mean, second_mean) if second_mean else 0.0
    evidence = (
        f"residual_sequence steps={len(steps)} model={substrate_snapshot.model_id} "
        f"frozen={substrate_snapshot.is_frozen}"
    )
    return (
        MidLayerScore(
            family="abstraction",
            metric_name="persona_geometry_norm",
            value=round(_l2_norm(overall_mean), 6),
            confidence=0.6,
            baseline_label="",
            delta_vs_baseline=None,
            evidence=evidence,
        ),
        MidLayerScore(
            family="abstraction",
            metric_name="persona_geometry_drift",
            value=round(drift, 6),
            confidence=0.6,
            baseline_label="",
            delta_vs_baseline=None,
            evidence=evidence,
        ),
    )


def _metric_win_value(
    *,
    candidate_value: float,
    baseline_value: float,
    higher_is_better: bool,
    epsilon: float,
) -> float:
    delta = candidate_value - baseline_value
    if abs(delta) <= epsilon:
        return 0.5
    if higher_is_better:
        return 1.0 if delta > 0 else 0.0
    return 1.0 if delta < 0 else 0.0


def build_deterministic_head_to_head_snapshot(
    *,
    generation_id: str,
    baseline_label: str,
    per_profile_metric_means: Mapping[str, Mapping[str, float]],
    metric_names: tuple[str, ...],
    lower_is_better: tuple[str, ...] = (),
    epsilon: float = 1e-6,
) -> ExpensiveLayerSnapshot:
    """Build deterministic head-to-head readouts from metric means.

    This helper is intentionally metric-based and LLM-free. It is suitable for
    Phase 2 smoke / paper-suite aggregates where the benchmark already
    produced comparable metric means for each profile.
    """
    if not generation_id:
        raise ValueError("generation_id must be non-empty")
    baseline_metrics = per_profile_metric_means[baseline_label]
    lower_better = frozenset(lower_is_better)
    results: list[HeadToHeadResult] = []
    for profile_label, candidate_metrics in sorted(per_profile_metric_means.items()):
        if profile_label == baseline_label:
            continue
        wins: list[float] = []
        for metric_name in metric_names:
            if metric_name not in candidate_metrics or metric_name not in baseline_metrics:
                continue
            wins.append(
                _metric_win_value(
                    candidate_value=candidate_metrics[metric_name],
                    baseline_value=baseline_metrics[metric_name],
                    higher_is_better=metric_name not in lower_better,
                    epsilon=epsilon,
                )
            )
        if not wins:
            raise ValueError(
                f"No overlapping metric_names for profile {profile_label!r} "
                f"against baseline {baseline_label!r}."
            )
        winrate = round(sum(wins) / len(wins), 4)
        results.append(
            HeadToHeadResult(
                profile_a=profile_label,
                profile_b=baseline_label,
                case_count=len(wins),
                winrate_a_vs_b=winrate,
                confidence_interval_low=winrate,
                confidence_interval_high=winrate,
                judge_kind="deterministic",
                notes=(
                    f"Deterministic metric-means comparison over {len(wins)} "
                    f"metrics; lower_is_better={tuple(sorted(lower_better))}."
                ),
            )
        )
    return ExpensiveLayerSnapshot(
        generation_id=generation_id,
        head_to_head_results=tuple(results),
        llm_judge_readouts=(),
        aggregated_scores=(),
        description=(
            f"Deterministic head-to-head snapshot for {len(results)} profiles "
            f"against baseline={baseline_label}."
        ),
    )


class ExpensiveLayerModule(RuntimeModule[ExpensiveLayerSnapshot]):
    """Expensive-tier evaluation publisher.

    DISABLED by default; promotion gated by ETA strong-proof PASS per spec
    §迁移协议 Step 3.

    C4: the tier is real. It re-emits the mid tier's aggregated scores,
    adds the substrate geometry readout when a substrate snapshot is
    published, and publishes judge readouts submitted through
    ``submit_judge_cases``. Zero LLM calls unless a backend was injected
    AND cases were submitted (contract-tested).
    """

    slot_name = "evaluation_expensive"
    owner = "ExpensiveLayerModule"
    value_type = ExpensiveLayerSnapshot
    dependencies = ("evaluation_mid", "substrate")
    default_wiring_level = WiringLevel.DISABLED

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        llm_judge_backend: LlmJudgeBackend | None = None,
        judge_model_id: str = "",
        generation_id: str = "",
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        if llm_judge_backend is not None and not judge_model_id:
            raise ValueError(
                "judge_model_id must be provided when llm_judge_backend is injected"
            )
        self._llm_judge_backend = llm_judge_backend
        self._judge_model_id = judge_model_id
        self._generation_id = generation_id
        self._pending_judge_cases: list[LlmJudgeCase] = []

    def submit_judge_cases(self, cases: Sequence[LlmJudgeCase]) -> None:
        """Queue dialogue cases for judge readout on the next process()."""

        if self._llm_judge_backend is None:
            raise ValueError(
                "cannot submit judge cases: no llm_judge_backend injected"
            )
        self._pending_judge_cases.extend(cases)

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ExpensiveLayerSnapshot]:
        # fail-loudly on missing dependency (kernel UpstreamView guards this)
        mid_snapshot = upstream["evaluation_mid"]
        mid_value = (
            mid_snapshot.value
            if isinstance(mid_snapshot.value, MidLayerSnapshot)
            else None
        )
        if mid_value is None:
            return self.publish(_EMPTY_EXPENSIVE_SNAPSHOT)

        aggregated: list[MidLayerScore] = list(mid_value.aggregated_scores)
        substrate_snapshot = upstream.get("substrate")
        if substrate_snapshot is not None and isinstance(
            substrate_snapshot.value, SubstrateSnapshot
        ):
            aggregated.extend(
                build_substrate_geometry_scores(substrate_snapshot.value)
            )

        judge_readouts: tuple[LlmJudgeReadout, ...] = ()
        if self._llm_judge_backend is not None and self._pending_judge_cases:
            cases = tuple(self._pending_judge_cases)
            self._pending_judge_cases = []
            judge_readouts = run_llm_judge_readouts(
                backend=self._llm_judge_backend,
                judge_model_id=self._judge_model_id,
                cases=cases,
            )

        return self.publish(
            ExpensiveLayerSnapshot(
                generation_id=self._generation_id,
                head_to_head_results=(),
                llm_judge_readouts=judge_readouts,
                aggregated_scores=tuple(aggregated),
                description=(
                    "expensive_layer readouts: "
                    f"{len(aggregated)} aggregated scores "
                    f"({len(aggregated) - len(mid_value.aggregated_scores)} substrate-geometry), "
                    f"{len(judge_readouts)} LLM-judge readouts (gate-ineligible)."
                ),
            )
        )
