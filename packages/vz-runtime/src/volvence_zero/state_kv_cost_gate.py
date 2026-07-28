"""Cost/latency comparison gate: Prefix-KV (arm G) versus text carrier (arm B').

State KV design plan P3 exit condition (research/state_kv/
01_state_kv_complete_design_plan.md): identification evidence already showed
arm G carries the state without prompt bytes; this gate produces the *other*
half of the P3 verdict -- that the latent carrier is also cheaper than
rendering the same readout into the system prompt. Without this artifact the
"成本占优" half of the P3 claim is asserted, not computed.

Measurement rules, mirroring the identification runner's invariants:

1. **Cost is measured on what was actually sent.** A transparent proxy sits
   between the synthesizer and the runtime and records the exact chat payload
   each ``generate`` call received, plus wall-clock around the call.
   Recomputing "the prompt we would have built" would attest to something
   that never ran.
2. **Arm G's cost includes its prefix slots.** The prefix KV occupies
   attention slots exactly like prompt tokens do, so a comparison that
   ignored them would let a huge prefix masquerade as "free".
3. **A fake substrate cannot produce a latency verdict.** Payload sizes are
   real on any substrate (the synthesizer builds real prompts), but timing a
   trace-only fake proves nothing, so the latency claim tops out at
   ``insufficient_data`` without frozen weights -- the same ceiling the
   identification verdict applies.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import median

from volvence_zero.agent.response import ResponseContext
from volvence_zero.state_kv_identification import (
    PREFIX_ARM_LABEL,
    ClaimResult,
    ClaimState,
    IdentificationArm,
    ProbeCase,
    SubstrateEvidenceKind,
    arm_from_profile,
    context_for_arm,
)

__all__ = [
    "COST_GATE_SCHEMA_VERSION",
    "DEFAULT_COST_ARM_LABELS",
    "TEXT_CARRIER_ARM_LABEL",
    "ArmCostSummary",
    "CostGateVerdict",
    "MeasuredTurnCost",
    "MeasuringRuntimeProxy",
    "build_cost_gate_verdict",
    "run_cost_gate",
]

COST_GATE_SCHEMA_VERSION = "state-kv-cost-gate.v1"

TEXT_CARRIER_ARM_LABEL = "state-kv-arm-bprime"

# Candidate first, baseline second; row order is part of the artifact.
DEFAULT_COST_ARM_LABELS: tuple[str, str] = (
    PREFIX_ARM_LABEL,
    TEXT_CARRIER_ARM_LABEL,
)

_CLAIM_PAYLOAD = "claim_conditioning_payload_smaller"
_CLAIM_LATENCY = "claim_latency_not_worse"

COST_CLAIM_NAMES: tuple[str, ...] = (_CLAIM_PAYLOAD, _CLAIM_LATENCY)


@dataclass(frozen=True)
class MeasuredTurnCost:
    """Cost readout for one arm x user x probe turn, from the sent payload."""

    arm_label: str
    user_id: str
    probe_id: str
    prompt_tokens: int
    prompt_chars: int
    prefix_slots: int
    generated_tokens: int
    latency_ms: float

    @property
    def conditioning_cost_tokens(self) -> int:
        """Attention-slot cost of the whole conditioned prompt.

        Prompt tokens plus prefix slots: both occupy K/V positions the model
        attends over, so they are the comparable unit across carriers.
        """

        return self.prompt_tokens + self.prefix_slots

    @property
    def latency_per_token_ms(self) -> float:
        """Wall-clock per emitted token, the carrier-attributable unit.

        Raw per-turn wall-clock is dominated by how *long* an answer the arm
        chose to write (EOS varies under a fixed decode budget), which is
        behaviour, not carrier cost. Dividing by emitted tokens charges each
        arm its prefill and per-step decode cost per unit of output, which
        is what the carrier actually changes.
        """

        return self.latency_ms / max(1, self.generated_tokens)

    def as_json_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm_label,
            "user": self.user_id,
            "probe": self.probe_id,
            "prompt_tokens": self.prompt_tokens,
            "prompt_chars": self.prompt_chars,
            "prefix_slots": self.prefix_slots,
            "conditioning_cost_tokens": self.conditioning_cost_tokens,
            "generated_tokens": self.generated_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "latency_per_token_ms": round(self.latency_per_token_ms, 3),
        }


class MeasuringRuntimeProxy:
    """Transparent runtime proxy that measures each ``generate`` call.

    The proxy exposes exactly the surface ``LLMResponseSynthesizer`` consumes
    (``generate`` and ``model_id``) and forwards everything unchanged, so the
    measured run is byte-identical to an unmeasured one. Turn boundaries are
    declared by the driver via :meth:`begin_turn` / :meth:`finish_turn`;
    a synthesize call that skips the runtime entirely (fallback path) fails
    the turn loudly instead of silently reporting a zero-cost row.
    """

    def __init__(
        self,
        inner: object,
        *,
        token_counter: Callable[[Sequence[tuple[str, str]]], int],
    ) -> None:
        self._inner = inner
        self._token_counter = token_counter
        self._pending: dict[str, object] | None = None
        self._captured: MeasuredTurnCost | None = None

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def begin_turn(
        self,
        *,
        arm_label: str,
        user_id: str,
        probe_id: str,
        prefix_slots: int,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError(
                "cost gate turn overlap: begin_turn called before the "
                "previous turn was finished."
            )
        self._pending = {
            "arm_label": arm_label,
            "user_id": user_id,
            "probe_id": probe_id,
            "prefix_slots": prefix_slots,
        }
        self._captured = None

    def finish_turn(self) -> MeasuredTurnCost:
        if self._pending is None:
            raise RuntimeError("cost gate finish_turn called without begin_turn.")
        captured = self._captured
        self._pending = None
        self._captured = None
        if captured is None:
            raise RuntimeError(
                "cost gate turn produced no generate call: the synthesizer "
                "fell back without reaching the runtime, so this turn has no "
                "measurable cost and the row must not be fabricated."
            )
        return captured

    def generate(self, **kwargs: object) -> object:
        pending = self._pending
        if pending is None:
            raise RuntimeError(
                "cost gate runtime received a generate call outside a "
                "declared turn; measurements would be attributed to nothing."
            )
        if self._captured is not None:
            raise RuntimeError(
                "cost gate turn produced more than one generate call; "
                "one-turn-one-call is what makes the latency row meaningful."
            )
        chat_messages = kwargs.get("chat_messages")
        if not isinstance(chat_messages, Sequence) or not chat_messages:
            raise ValueError(
                "cost gate requires the synthesizer to pass chat_messages; "
                "without the exact sent payload the prompt cost cannot be "
                "measured honestly."
            )
        # ChatMessage is a (role, content) pair (agent/prompts.py).
        prompt_chars = sum(len(content) for _, content in chat_messages)
        started = time.perf_counter()
        result = self._inner.generate(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._captured = MeasuredTurnCost(
            arm_label=str(pending["arm_label"]),
            user_id=str(pending["user_id"]),
            probe_id=str(pending["probe_id"]),
            prompt_tokens=int(self._token_counter(chat_messages)),
            prompt_chars=prompt_chars,
            prefix_slots=int(pending["prefix_slots"]),  # type: ignore[arg-type]
            generated_tokens=int(result.token_count),
            latency_ms=latency_ms,
        )
        return result


@dataclass(frozen=True)
class ArmCostSummary:
    """Aggregates for one arm, recomputable from the turn table."""

    arm_label: str
    turn_count: int
    prefix_slots: int
    median_prompt_tokens: float
    median_conditioning_cost_tokens: float
    median_generated_tokens: float
    median_latency_ms: float
    median_latency_per_token_ms: float

    def as_json_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm_label,
            "turn_count": self.turn_count,
            "prefix_slots": self.prefix_slots,
            "median_prompt_tokens": self.median_prompt_tokens,
            "median_conditioning_cost_tokens": (
                self.median_conditioning_cost_tokens
            ),
            "median_generated_tokens": self.median_generated_tokens,
            "median_latency_ms": round(self.median_latency_ms, 3),
            "median_latency_per_token_ms": round(
                self.median_latency_per_token_ms, 3
            ),
        }


def _summarise_arm(
    arm_label: str, turns: Sequence[MeasuredTurnCost]
) -> ArmCostSummary:
    prefix_slots = {turn.prefix_slots for turn in turns}
    if len(prefix_slots) != 1:
        raise ValueError(
            f"arm {arm_label!r} recorded inconsistent prefix slot counts "
            f"{sorted(prefix_slots)}; one arm must run one carrier "
            "configuration."
        )
    return ArmCostSummary(
        arm_label=arm_label,
        turn_count=len(turns),
        prefix_slots=next(iter(prefix_slots)),
        median_prompt_tokens=median(turn.prompt_tokens for turn in turns),
        median_conditioning_cost_tokens=median(
            turn.conditioning_cost_tokens for turn in turns
        ),
        median_generated_tokens=median(
            turn.generated_tokens for turn in turns
        ),
        median_latency_ms=median(turn.latency_ms for turn in turns),
        median_latency_per_token_ms=median(
            turn.latency_per_token_ms for turn in turns
        ),
    )


def _turns_by_key(
    turns: Sequence[MeasuredTurnCost],
) -> dict[tuple[str, str], MeasuredTurnCost]:
    keyed: dict[tuple[str, str], MeasuredTurnCost] = {}
    for turn in turns:
        key = (turn.user_id, turn.probe_id)
        if key in keyed:
            raise ValueError(
                f"arm {turn.arm_label!r} measured {key} twice; the per-probe "
                "comparison needs exactly one row per user/probe."
            )
        keyed[key] = turn
    return keyed


def evaluate_payload_claim(
    *,
    candidate_turns: Sequence[MeasuredTurnCost],
    baseline_turns: Sequence[MeasuredTurnCost],
) -> ClaimResult:
    """Claim: on every probe, G's prompt + prefix slots < B''s prompt.

    Strict per-probe dominance rather than an aggregate mean: a carrier that
    is cheaper on average but more expensive on some probes has not shown a
    cost advantage, it has shown a trade-off.
    """

    candidate = _turns_by_key(candidate_turns)
    baseline = _turns_by_key(baseline_turns)
    if not candidate or set(candidate) != set(baseline):
        return ClaimResult(
            name=_CLAIM_PAYLOAD,
            state=ClaimState.INSUFFICIENT_DATA,
            detail="arms did not measure the same user/probe set",
        )
    violations: list[str] = []
    savings: list[float] = []
    for key in sorted(candidate):
        candidate_cost = candidate[key].conditioning_cost_tokens
        baseline_cost = baseline[key].conditioning_cost_tokens
        if candidate_cost >= baseline_cost:
            violations.append(
                f"{key[0]}/{key[1]}: {candidate_cost} >= {baseline_cost}"
            )
        elif baseline_cost > 0:
            savings.append(1.0 - candidate_cost / baseline_cost)
    if violations:
        return ClaimResult(
            name=_CLAIM_PAYLOAD,
            state=ClaimState.FAIL,
            detail=(
                "prefix carrier not cheaper on: " + "; ".join(violations)
            ),
        )
    mean_saving = sum(savings) / len(savings) if savings else 0.0
    return ClaimResult(
        name=_CLAIM_PAYLOAD,
        state=ClaimState.PASS,
        detail=(
            f"prefix carrier strictly cheaper on all {len(candidate)} "
            f"user/probe turns; mean attention-slot saving "
            f"{mean_saving:.1%} vs text carrier"
        ),
    )


def evaluate_latency_claim(
    *,
    candidate_turns: Sequence[MeasuredTurnCost],
    baseline_turns: Sequence[MeasuredTurnCost],
    substrate_kind: SubstrateEvidenceKind,
    tolerance: float,
    minimum_turns: int = 4,
) -> ClaimResult:
    """Claim: G's median per-emitted-token latency <= B''s x (1 + tolerance).

    Per emitted token, not per turn: with a fixed decode budget the arms stop
    at different EOS points, so raw per-turn wall-clock measures how long an
    answer the arm chose to write -- behaviour, not carrier cost. Per-token
    wall-clock charges each arm its prefill plus per-step decode cost per
    unit of output, which is the part the carrier changes. Medians rather
    than means because single-turn timing on a shared CPU is heavy-tailed;
    the tolerance is recorded in the artifact so a reader knows exactly what
    "not worse" meant.
    """

    if substrate_kind is not SubstrateEvidenceKind.FROZEN_WEIGHTS:
        return ClaimResult(
            name=_CLAIM_LATENCY,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=(
                f"substrate is {substrate_kind.value}; timing a non-frozen "
                "substrate says nothing about production latency"
            ),
        )
    if min(len(candidate_turns), len(baseline_turns)) < minimum_turns:
        return ClaimResult(
            name=_CLAIM_LATENCY,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=(
                f"needs at least {minimum_turns} turns per arm for a stable "
                f"median, got {len(candidate_turns)} vs {len(baseline_turns)}"
            ),
        )
    candidate_median = median(
        turn.latency_per_token_ms for turn in candidate_turns
    )
    baseline_median = median(
        turn.latency_per_token_ms for turn in baseline_turns
    )
    budget = baseline_median * (1.0 + tolerance)
    if candidate_median <= budget:
        return ClaimResult(
            name=_CLAIM_LATENCY,
            state=ClaimState.PASS,
            detail=(
                f"median {candidate_median:.1f}ms/token <= "
                f"{baseline_median:.1f}ms/token x (1 + {tolerance:.2f})"
            ),
        )
    return ClaimResult(
        name=_CLAIM_LATENCY,
        state=ClaimState.FAIL,
        detail=(
            f"median {candidate_median:.1f}ms/token exceeds the "
            f"{budget:.1f}ms/token budget ({baseline_median:.1f}ms/token x "
            f"(1 + {tolerance:.2f}))"
        ),
    )


@dataclass(frozen=True)
class CostGateVerdict:
    """The artifact: two claim states plus the turn table behind them."""

    schema_version: str
    gate_state: str
    substrate_kind: SubstrateEvidenceKind
    substrate_fingerprint: str
    candidate_arm_label: str
    baseline_arm_label: str
    latency_tolerance: float
    claims: tuple[ClaimResult, ...]
    arm_summaries: tuple[ArmCostSummary, ...]
    turn_table: tuple[MeasuredTurnCost, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def claim(self, name: str) -> ClaimResult:
        for result in self.claims:
            if result.name == name:
                return result
        raise KeyError(f"cost gate carries no claim named {name!r}")

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate_state": self.gate_state,
            "substrate_kind": self.substrate_kind.value,
            "substrate_fingerprint": self.substrate_fingerprint,
            "candidate_arm": self.candidate_arm_label,
            "baseline_arm": self.baseline_arm_label,
            "latency_tolerance": self.latency_tolerance,
            "claims": [claim.as_json_dict() for claim in self.claims],
            "arm_summaries": [
                summary.as_json_dict() for summary in self.arm_summaries
            ],
            "turn_table": [turn.as_json_dict() for turn in self.turn_table],
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(), ensure_ascii=False, indent=2, sort_keys=False
        )


def build_cost_gate_verdict(
    *,
    turns: Sequence[MeasuredTurnCost],
    substrate_kind: SubstrateEvidenceKind,
    substrate_fingerprint: str,
    candidate_arm_label: str = PREFIX_ARM_LABEL,
    baseline_arm_label: str = TEXT_CARRIER_ARM_LABEL,
    latency_tolerance: float = 0.10,
    extra_notes: Sequence[str] = (),
) -> CostGateVerdict:
    """Compute the cost gate verdict from measured turns.

    ``gate_state`` is ``pass`` only when both claims pass; a single
    ``insufficient_data`` claim keeps the gate at ``insufficient_data``, and
    any failed claim fails the gate. There is no partial credit: "cheaper but
    slower" is not the P3 cost conclusion.
    """

    if not substrate_fingerprint:
        raise ValueError(
            "cost gate requires a substrate fingerprint: without it the two "
            "arms cannot be shown to have shared one substrate."
        )
    if not 0.0 <= latency_tolerance < 1.0:
        raise ValueError(
            f"latency_tolerance must be in [0, 1), got {latency_tolerance}."
        )
    by_arm: dict[str, list[MeasuredTurnCost]] = {}
    for turn in turns:
        by_arm.setdefault(turn.arm_label, []).append(turn)
    unexpected = set(by_arm) - {candidate_arm_label, baseline_arm_label}
    if unexpected:
        raise ValueError(
            f"cost gate received turns for unexpected arms: {sorted(unexpected)}."
        )
    candidate_turns = tuple(by_arm.get(candidate_arm_label, ()))
    baseline_turns = tuple(by_arm.get(baseline_arm_label, ()))
    claims = (
        evaluate_payload_claim(
            candidate_turns=candidate_turns,
            baseline_turns=baseline_turns,
        ),
        evaluate_latency_claim(
            candidate_turns=candidate_turns,
            baseline_turns=baseline_turns,
            substrate_kind=substrate_kind,
            tolerance=latency_tolerance,
        ),
    )
    if any(claim.state is ClaimState.FAIL for claim in claims):
        gate_state = "fail"
    elif all(claim.state is ClaimState.PASS for claim in claims):
        gate_state = "pass"
    else:
        gate_state = "insufficient_data"
    arm_summaries = tuple(
        _summarise_arm(label, by_arm[label])
        for label in (candidate_arm_label, baseline_arm_label)
        if label in by_arm and by_arm[label]
    )
    return CostGateVerdict(
        schema_version=COST_GATE_SCHEMA_VERSION,
        gate_state=gate_state,
        substrate_kind=substrate_kind,
        substrate_fingerprint=substrate_fingerprint,
        candidate_arm_label=candidate_arm_label,
        baseline_arm_label=baseline_arm_label,
        latency_tolerance=latency_tolerance,
        claims=claims,
        arm_summaries=arm_summaries,
        turn_table=tuple(turns),
        notes=tuple(extra_notes),
    )


def _prefix_slots_for_arm(arm: IdentificationArm, *, prefix_slots: int) -> int:
    """Prefix slots charged to an arm's turns.

    Only the prefix carrier pays them; charging the text arm for slots it
    does not occupy would rig the comparison in G's favour.
    """

    return prefix_slots if arm.conditioning_mode == "prefix_kv" else 0


def run_cost_gate(
    *,
    cases: Sequence[ProbeCase],
    synthesizer: object,
    measuring_runtime: MeasuringRuntimeProxy,
    base_context: ResponseContext,
    substrate_kind: SubstrateEvidenceKind,
    substrate_fingerprint: str,
    prefix_slots: int,
    arm_labels: tuple[str, str] = DEFAULT_COST_ARM_LABELS,
    latency_tolerance: float = 0.10,
    extra_notes: Sequence[str] = (),
) -> CostGateVerdict:
    """Run both arms over the probe cases and compute the verdict.

    ``synthesizer`` must be wired to ``measuring_runtime`` (which wraps the
    real runtime); the driver declares turn boundaries so every measured row
    is attributable to one arm x user x probe.
    """

    if not cases:
        raise ValueError("cost gate needs at least one probe case.")
    if prefix_slots <= 0:
        raise ValueError(
            "cost gate requires the prefix artifact's slot count; without it "
            "arm G's attention cost would be understated as prompt-only."
        )
    candidate_label, baseline_label = arm_labels
    turns: list[MeasuredTurnCost] = []
    for label in (candidate_label, baseline_label):
        arm = arm_from_profile(label)
        arm_prefix_slots = _prefix_slots_for_arm(arm, prefix_slots=prefix_slots)
        for case in cases:
            context = context_for_arm(
                arm=arm,
                case=case,
                base_context=base_context,
            )
            measuring_runtime.begin_turn(
                arm_label=label,
                user_id=case.user_id,
                probe_id=case.probe_id,
                prefix_slots=arm_prefix_slots,
            )
            synthesizer.synthesize(context=context, assembly=case.assembly)
            turns.append(measuring_runtime.finish_turn())
    return build_cost_gate_verdict(
        turns=turns,
        substrate_kind=substrate_kind,
        substrate_fingerprint=substrate_fingerprint,
        candidate_arm_label=candidate_label,
        baseline_arm_label=baseline_label,
        latency_tolerance=latency_tolerance,
        extra_notes=extra_notes,
    )
