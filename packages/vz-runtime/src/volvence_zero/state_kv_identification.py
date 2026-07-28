"""Carrier-identification evidence runner (State KV, package 2).

Implements the runner side of ``docs/specs/state-kv-identification-evidence.md``:
package 1 froze the contracts (``prompt_state_delivery``, the three attestation
tags, the two ``*-pure`` profiles); this module turns them into an artifact a
third party can re-check -- ``verdict_identification.json``.

The whole point of this file is that **the claim is computed, not asserted**.
Every one of the four spec claims is a function of recorded per-turn
fingerprints, and each one can come back ``insufficient_data``. Three design
rules follow from that and are the reason this module is not a thin script:

1. **Attestation is read back from the turn, never from the request.** A turn's
   ``prompt_fp`` / ``prompt_state_sections`` / ``decode_fp`` and whether the
   residual was actually injected are parsed out of the emitted
   ``rationale_tags``. Recomputing "the prompt we would have sent" would attest
   to something that was never sent (spec §关键不变量 6).
2. **A fake substrate can never produce a retained claim.**
   :class:`SubstrateEvidenceKind` gates the verdict state machine: a trace-only
   runtime tops out at ``insufficient_data``. Without that gate the zero-cost
   P0-smoke run would be quotable as the real identification result.
3. **No judge, no identification number.** ``claim_identification_above_chance``
   requires blind-judge votes. When no judge is wired the claim is
   ``insufficient_data`` -- this module never synthesises a matching accuracy to
   fill the field.

Stage scope (spec §执行阶段): P0-smoke runs the four arms on a deterministic
fake substrate to prove ``prompt_fp`` equality and tag completeness at zero
cost. P1-directional swaps in frozen Qwen plus a cross-family blind judge by
replacing the runtime and passing a :class:`BlindMatchingJudge`; the claim logic
below does not change between the two.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol

from volvence_zero.agent.profile_registry import resolve_profile
from volvence_zero.agent.response import ResponseContext
from volvence_zero.application.runtime import ResponseAssemblySnapshot
from volvence_zero.personal_conditioning_contracts import (
    PersonalConditioningSnapshot,
)

__all__ = [
    "ALL_ARM_LABELS",
    "CONTROL_ARM_LABEL",
    "DEFAULT_CANDIDATE_ARM_LABEL",
    "IDENTIFICATION_SCHEMA_VERSION",
    "IDENTIFICATION_ARM_LABELS",
    "PREFIX_ARM_LABEL",
    "PREFIX_IDENTIFICATION_ARM_LABELS",
    "PURE_ARM_LABELS",
    "ArmObservation",
    "BlindMatchingJudge",
    "ClaimResult",
    "ClaimState",
    "IdentificationArm",
    "IdentificationTurn",
    "IdentificationVerdict",
    "MatchingReadout",
    "ProbeCase",
    "SubstrateEvidenceKind",
    "arm_from_profile",
    "bootstrap_matching_ci",
    "build_identification_verdict",
    "context_for_arm",
    "observe_arm",
    "run_identification_smoke",
]

IDENTIFICATION_SCHEMA_VERSION = "state-kv-identification.v1"

# Arm order is the spec's §四臂矩阵 order and is part of the artifact: a reader
# comparing two runs must see the same rows in the same places.
IDENTIFICATION_ARM_LABELS: tuple[str, ...] = (
    "state-kv-arm-a-pure",
    "state-kv-arm-e-pure",
    "state-kv-arm-bprime",
    "state-kv-arm-e",
)

PREFIX_ARM_LABEL = "state-kv-arm-g-prefix-pure"

# The P3 lane: the same four arms plus the prefix-KV carrier, so arm G is
# measured against the residual carrier and the text carrier in one run on one
# frozen substrate rather than across two runs that could differ in anything.
PREFIX_IDENTIFICATION_ARM_LABELS: tuple[str, ...] = (
    *IDENTIFICATION_ARM_LABELS,
    PREFIX_ARM_LABEL,
)

# Every arm this module knows how to order in an artifact. Row order is part of
# the artifact: a reader comparing two runs must see the same rows in the same
# places.
ALL_ARM_LABELS: tuple[str, ...] = PREFIX_IDENTIFICATION_ARM_LABELS

# The control arm whose prompt every candidate must match byte-for-byte, and
# the default candidate. Claim 1 is defined on the (control, candidate) pair;
# the remaining arms deliberately differ in prompt bytes.
PURE_ARM_LABELS: tuple[str, str] = (
    "state-kv-arm-a-pure",
    "state-kv-arm-e-pure",
)
CONTROL_ARM_LABEL = PURE_ARM_LABELS[0]
DEFAULT_CANDIDATE_ARM_LABEL = PURE_ARM_LABELS[1]

_CLAIM_PROMPT_IDENTITY = "claim_prompt_identity"
_CLAIM_OUTPUT_DIVERGENCE = "claim_output_divergence"
_CLAIM_IDENTIFICATION = "claim_identification_above_chance"
_CLAIM_CARRIER_CAUSALITY = "claim_carrier_causality"

CLAIM_NAMES: tuple[str, ...] = (
    _CLAIM_PROMPT_IDENTITY,
    _CLAIM_OUTPUT_DIVERGENCE,
    _CLAIM_IDENTIFICATION,
    _CLAIM_CARRIER_CAUSALITY,
)


class ClaimState(str, Enum):
    """Per-claim outcome.

    ``INSUFFICIENT_DATA`` is a first-class result, not an error: most of the
    zero-cost smoke run legitimately lands here, and collapsing it into
    ``FAIL`` would make "we could not measure this" indistinguishable from
    "we measured this and it did not hold".
    """

    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class VerdictState(str, Enum):
    """Spec §判据 overall state machine."""

    INSUFFICIENT_DATA = "insufficient_data"
    FAIL = "fail"
    WEAK_POSITIVE = "weak-positive"
    RETAIN_PROMPT_CLOSED = "retain-prompt-closed"
    RETAIN_STRICT = "retain-strict"


class SubstrateEvidenceKind(Enum):
    """What the substrate under test can support as evidence.

    ``TRACE_ONLY`` covers the synthetic runtime and any fake used for wiring
    smoke: it reports ``personal_conditioning_applied=False`` by contract, so
    no positive identification claim can rest on it.
    ``FROZEN_WEIGHTS`` is a real frozen model whose residual hook actually
    fires.
    """

    TRACE_ONLY = "trace-only"
    FROZEN_WEIGHTS = "frozen-weights"


class C5Grade(str, Enum):
    """Spec §C5 分档: how much of the sampling-layer carrier was closed."""

    DECODE_MATCHED = "decode-matched"
    DECODE_DIVERGENT = "decode-divergent"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Arm settings, derived from the profile registry (single source of truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentificationArm:
    """One arm's delivery settings, resolved from its profile."""

    label: str
    conditioning_active: bool
    conditioning_mode: str
    prompt_state_delivery: str


def arm_from_profile(label: str) -> IdentificationArm:
    """Resolve an arm's settings from the profile registry.

    The registry is the only place arm wiring is defined; re-declaring a
    ``label -> (wiring, mode, delivery)`` table here would let the runner drift
    from the profiles the rest of the system runs, and the drift would be
    invisible because both sides would "agree with the spec".
    """

    resolved = resolve_profile(label)
    overrides: Mapping[str, object] = resolved.merged_flag_overrides
    wiring = str(overrides.get("personal_conditioning", "WiringLevel.SHADOW"))
    mode = str(overrides.get("personal_conditioning_mode", "residual"))
    delivery = str(overrides.get("prompt_state_delivery", "text"))
    if mode not in ("residual", "text", "prefix_kv"):
        raise ValueError(
            f"profile {label!r} declares unknown personal_conditioning_mode "
            f"{mode!r}"
        )
    if delivery not in ("text", "suppressed"):
        raise ValueError(
            f"profile {label!r} declares unknown prompt_state_delivery "
            f"{delivery!r}"
        )
    return IdentificationArm(
        label=label,
        conditioning_active=wiring.endswith("ACTIVE"),
        conditioning_mode=mode,
        prompt_state_delivery=delivery,
    )


# ---------------------------------------------------------------------------
# Probe material
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeCase:
    """One user's state and assembly for one probe sentence.

    Supplied by the caller rather than rebuilt here: the smoke stage uses
    hand-written personas, while P1-directional feeds the 20-session
    longitudinal harness through the same seam. Keeping the material outside
    the runner is what makes the two stages comparable.
    """

    user_id: str
    probe_id: str
    user_input: str
    conditioning: PersonalConditioningSnapshot
    assembly: ResponseAssemblySnapshot

    def __post_init__(self) -> None:
        if not self.user_id or not self.probe_id:
            raise ValueError("ProbeCase requires a user_id and a probe_id.")
        if not self.user_input:
            raise ValueError("ProbeCase requires a probe sentence.")


def context_for_arm(
    *,
    arm: IdentificationArm,
    case: ProbeCase,
    base_context: ResponseContext,
) -> ResponseContext:
    """Apply one arm's delivery settings to a probe case.

    Mirrors ``session_observation``'s wiring decision -- SHADOW delivers
    nothing, ACTIVE delivers either the snapshot or its rendered statement,
    never both -- so an arm here receives exactly what the same arm would
    receive in a live session.
    """

    conditioning: PersonalConditioningSnapshot | None = None
    statement = ""
    statement_ref = ""
    if arm.conditioning_active and not case.conditioning.is_cold_start:
        if arm.conditioning_mode == "text":
            statement = case.conditioning.rendered_statement
            if not statement:
                raise ValueError(
                    f"arm {arm.label!r} delivers state as text but the "
                    f"snapshot for user {case.user_id!r} renders empty; a "
                    "text arm with nothing to say is not a control for a "
                    "residual arm that has something to inject."
                )
            statement_ref = (
                f"{case.conditioning.schema_version}:"
                f"{case.conditioning.confidence:.2f}:"
                f"{case.conditioning.source_fingerprint[:12]}"
            )
        else:
            conditioning = case.conditioning
    return replace(
        base_context,
        user_input=case.user_input,
        personal_conditioning=conditioning,
        personal_conditioning_statement=statement,
        personal_conditioning_statement_ref=statement_ref,
        prompt_state_delivery=arm.prompt_state_delivery,
        personal_conditioning_carrier=(
            "prefix_kv" if arm.conditioning_mode == "prefix_kv" else "residual"
        ),
    )


# ---------------------------------------------------------------------------
# Per-turn observation, read back out of the attestation tags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentificationTurn:
    """What one arm × user × probe turn attested to."""

    arm_label: str
    user_id: str
    probe_id: str
    prompt_fp: str
    prompt_state_sections: int
    decode_fp: str
    conditioning_applied: bool
    conditioning_delivered: bool
    text: str

    def as_json_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm_label,
            "user": self.user_id,
            "probe": self.probe_id,
            "prompt_fp": self.prompt_fp,
            "prompt_state_sections": self.prompt_state_sections,
            "decode_fp": self.decode_fp,
            "conditioning_applied": self.conditioning_applied,
            "conditioning_delivered": self.conditioning_delivered,
        }


def _single_tag(tags: Sequence[str], key: str) -> str:
    prefix = f"{key}="
    matches = [tag[len(prefix) :] for tag in tags if tag.startswith(prefix)]
    if len(matches) != 1:
        # The three attestation tags are contractually unconditional. A missing
        # or duplicated one means the attestation channel itself is broken, and
        # any claim computed from it would be unfalsifiable.
        raise ValueError(
            f"expected exactly one {key!r} attestation tag, got {matches!r}"
        )
    return matches[0]


def turn_from_response(
    *,
    arm: IdentificationArm,
    case: ProbeCase,
    response: object,
    conditioning_delivered: bool,
) -> IdentificationTurn:
    """Extract one turn's attestation from an ``AgentResponse``.

    Reads the emitted tags rather than the request, per spec §关键不变量 6.
    """

    tags = tuple(getattr(response, "rationale_tags", ()))
    applied_tags = [t for t in tags if t.startswith("personal_conditioning=")]
    return IdentificationTurn(
        arm_label=arm.label,
        user_id=case.user_id,
        probe_id=case.probe_id,
        prompt_fp=_single_tag(tags, "prompt_fp"),
        prompt_state_sections=int(_single_tag(tags, "prompt_state_sections")),
        decode_fp=_single_tag(tags, "decode_fp"),
        conditioning_applied=bool(applied_tags),
        conditioning_delivered=conditioning_delivered,
        text=str(getattr(response, "text", "")),
    )


@dataclass(frozen=True)
class ArmObservation:
    """All turns recorded for one arm."""

    arm: IdentificationArm
    turns: tuple[IdentificationTurn, ...]

    def turns_by_probe(self) -> dict[str, tuple[IdentificationTurn, ...]]:
        grouped: dict[str, list[IdentificationTurn]] = {}
        for turn in self.turns:
            grouped.setdefault(turn.probe_id, []).append(turn)
        return {probe: tuple(items) for probe, items in grouped.items()}


def observe_arm(
    *,
    arm: IdentificationArm,
    cases: Sequence[ProbeCase],
    synthesizer: object,
    base_context: ResponseContext,
) -> ArmObservation:
    """Run every probe case through one arm and record its attestation."""

    if not cases:
        raise ValueError(f"arm {arm.label!r} needs at least one probe case.")
    turns: list[IdentificationTurn] = []
    for case in cases:
        context = context_for_arm(arm=arm, case=case, base_context=base_context)
        response = synthesizer.synthesize(
            context=context, assembly=case.assembly
        )
        turns.append(
            turn_from_response(
                arm=arm,
                case=case,
                response=response,
                conditioning_delivered=(
                    context.personal_conditioning is not None
                    or bool(context.personal_conditioning_statement)
                ),
            )
        )
    return ArmObservation(arm=arm, turns=tuple(turns))


# ---------------------------------------------------------------------------
# Blind matching judge (P1-directional supplies the implementation)
# ---------------------------------------------------------------------------


class BlindMatchingJudge(Protocol):
    """Cross-family blind judge for the two-alternative matching task.

    The judge sees a response plus two candidate user summaries and nothing
    else -- no arm label, no prompt, no internal state (spec §判据 3). It is a
    Protocol because the smoke stage runs without one: the substrate family
    must not also judge (§关键不变量 5), so a real judge costs money and
    arrives with P1-directional.
    """

    @property
    def judge_model_id(self) -> str: ...

    def match(self, *, response_text: str, candidate_user_ids: Sequence[str]) -> str:
        """Return the ``user_id`` the response is attributed to."""


@dataclass(frozen=True)
class MatchingReadout:
    """Blind-matching accuracy for one arm. Read-only evidence (R12)."""

    arm_label: str
    correct: int
    total: int
    accuracy: float
    ci_low: float
    ci_high: float
    judge_model_id: str

    def as_json_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm_label,
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "judge_model_id": self.judge_model_id,
        }


def bootstrap_matching_ci(
    votes: Sequence[bool],
    *,
    seed: int,
    resamples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a two-alternative matching rate.

    Returns ``(accuracy, ci_low, ci_high)``. Seeded and resample-count
    explicit so a third party re-running the artifact gets the same interval;
    an unseeded CI would make the retain/weak-positive boundary irreproducible.
    """

    if not votes:
        raise ValueError("bootstrap CI requires at least one vote.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    if resamples <= 0:
        raise ValueError(f"resamples must be positive, got {resamples}.")
    observed = sum(1 for vote in votes if vote) / len(votes)
    rng = random.Random(seed)
    n = len(votes)
    rates: list[float] = []
    for _ in range(resamples):
        drawn = sum(1 for _ in range(n) if rng.choice(votes))
        rates.append(drawn / n)
    rates.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, int(tail * resamples) - 1)
    high_index = min(resamples - 1, int((1.0 - tail) * resamples))
    return observed, rates[low_index], rates[high_index]


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimResult:
    name: str
    state: ClaimState
    detail: str

    def as_json_dict(self) -> dict[str, object]:
        return {"claim": self.name, "state": self.state.value, "detail": self.detail}


def evaluate_prompt_identity(
    observations: Mapping[str, ArmObservation],
    *,
    candidate_arm_label: str = DEFAULT_CANDIDATE_ARM_LABEL,
) -> ClaimResult:
    """Claim 1: the two pure arms sent byte-identical prompts, with no state sections.

    Failure here voids the experiment (spec §判据 1), so it is evaluated first
    and its failure short-circuits the rest. ``candidate_arm_label`` selects
    which latent carrier is on trial: the residual arm by default, the
    prefix-KV arm on the P3 lane. The control arm never changes, so a claim
    computed for one carrier can be compared to the other.
    """

    pair = (CONTROL_ARM_LABEL, candidate_arm_label)
    missing = [label for label in pair if label not in observations]
    if missing:
        return ClaimResult(
            name=_CLAIM_PROMPT_IDENTITY,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=f"pure arms not observed: {', '.join(missing)}",
        )
    control, candidate = (observations[label] for label in pair)
    control_by_probe = control.turns_by_probe()
    candidate_by_probe = candidate.turns_by_probe()
    if set(control_by_probe) != set(candidate_by_probe):
        return ClaimResult(
            name=_CLAIM_PROMPT_IDENTITY,
            state=ClaimState.INSUFFICIENT_DATA,
            detail="pure arms did not run the same probe set",
        )
    mismatches: list[str] = []
    state_sections: list[str] = []
    for probe in sorted(control_by_probe):
        turns = control_by_probe[probe] + candidate_by_probe[probe]
        fingerprints = {turn.prompt_fp for turn in turns}
        if len(fingerprints) != 1:
            mismatches.append(f"{probe}:{sorted(fingerprints)}")
        for turn in turns:
            if turn.prompt_state_sections != 0:
                state_sections.append(
                    f"{turn.arm_label}/{turn.user_id}/{probe}="
                    f"{turn.prompt_state_sections}"
                )
    if mismatches or state_sections:
        details = []
        if mismatches:
            details.append(f"prompt_fp diverged on {', '.join(mismatches)}")
        if state_sections:
            details.append(
                f"state-derived sections present: {', '.join(state_sections)}"
            )
        return ClaimResult(
            name=_CLAIM_PROMPT_IDENTITY,
            state=ClaimState.FAIL,
            detail="; ".join(details),
        )
    turn_count = len(control.turns) + len(candidate.turns)
    return ClaimResult(
        name=_CLAIM_PROMPT_IDENTITY,
        state=ClaimState.PASS,
        detail=(
            f"{turn_count} turns across {len(control_by_probe)} probes share one "
            "prompt_fp per probe with prompt_state_sections=0"
        ),
    )


def evaluate_output_divergence(
    observations: Mapping[str, ArmObservation],
    *,
    substrate_kind: SubstrateEvidenceKind,
    candidate_arm_label: str = DEFAULT_CANDIDATE_ARM_LABEL,
) -> ClaimResult:
    """Claim 2: within the candidate arm, two users' outputs actually diverge.

    A trace-only substrate reports ``personal_conditioning_applied=False`` by
    contract, so it is recorded as ``insufficient_data`` -- never as success
    (spec §判据 2).
    """

    candidate_label = candidate_arm_label
    observation = observations.get(candidate_label)
    if observation is None:
        return ClaimResult(
            name=_CLAIM_OUTPUT_DIVERGENCE,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=f"{candidate_label} not observed",
        )
    unapplied = [
        f"{turn.user_id}/{turn.probe_id}"
        for turn in observation.turns
        if turn.conditioning_delivered and not turn.conditioning_applied
    ]
    if unapplied:
        return ClaimResult(
            name=_CLAIM_OUTPUT_DIVERGENCE,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=(
                f"substrate={substrate_kind.value} did not inject the residual "
                f"on {len(unapplied)} delivered turn(s) "
                f"({', '.join(unapplied[:4])}); injection cannot be inferred "
                "from having passed a snapshot"
            ),
        )
    identical: list[str] = []
    for probe, turns in sorted(observation.turns_by_probe().items()):
        if len({turn.text for turn in turns}) == 1 and len(turns) > 1:
            identical.append(probe)
    if identical:
        return ClaimResult(
            name=_CLAIM_OUTPUT_DIVERGENCE,
            state=ClaimState.FAIL,
            detail=(
                "candidate arm produced identical text across users on probes: "
                f"{', '.join(identical)}"
            ),
        )
    return ClaimResult(
        name=_CLAIM_OUTPUT_DIVERGENCE,
        state=ClaimState.PASS,
        detail=(
            f"{candidate_label} injected on every delivered turn and outputs "
            "differ across users on every probe"
        ),
    )


def evaluate_identification(
    matching: Mapping[str, MatchingReadout],
    *,
    candidate_arm_label: str = DEFAULT_CANDIDATE_ARM_LABEL,
) -> ClaimResult:
    """Claim 3: blind matching on the candidate arm beats chance.

    Without judge votes this is ``insufficient_data``. It is never estimated
    from internal state -- that would be scoring the hypothesis with itself.
    """

    candidate_label = candidate_arm_label
    readout = matching.get(candidate_label)
    if readout is None:
        return ClaimResult(
            name=_CLAIM_IDENTIFICATION,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=(
                "no blind-judge votes for "
                f"{candidate_label}; a cross-family judge is required "
                "(spec §关键不变量 5) and arrives with P1-directional"
            ),
        )
    if readout.ci_low > 0.5:
        return ClaimResult(
            name=_CLAIM_IDENTIFICATION,
            state=ClaimState.PASS,
            detail=(
                f"accuracy={readout.accuracy:.3f} "
                f"CI=({readout.ci_low:.3f}, {readout.ci_high:.3f}) "
                f"judge={readout.judge_model_id}"
            ),
        )
    return ClaimResult(
        name=_CLAIM_IDENTIFICATION,
        state=ClaimState.FAIL,
        detail=(
            f"accuracy={readout.accuracy:.3f} CI lower bound "
            f"{readout.ci_low:.3f} does not clear chance"
        ),
    )


def evaluate_carrier_causality(
    matching: Mapping[str, MatchingReadout],
    *,
    candidate_arm_label: str = DEFAULT_CANDIDATE_ARM_LABEL,
) -> ClaimResult:
    """Claim 4: the control arm collapses to chance and the carrier clears it.

    First half rules out prompt residue and lucky divergence; second half
    confirms the candidate latent carrier is carrying attributable state. On
    the original residual lane the text arm remains the positive-control
    carrier; on the prefix lane the prefix arm itself is the carrier under
    test, so requiring B-prime to clear would make a text-arm weakness veto a
    prefix result.
    """

    control_label = CONTROL_ARM_LABEL
    carrier_label = (
        "state-kv-arm-bprime"
        if candidate_arm_label == DEFAULT_CANDIDATE_ARM_LABEL
        else candidate_arm_label
    )
    control = matching.get(control_label)
    carrier = matching.get(carrier_label)
    if control is None or carrier is None:
        missing = [
            label
            for label, readout in (
                (control_label, control),
                (carrier_label, carrier),
            )
            if readout is None
        ]
        return ClaimResult(
            name=_CLAIM_CARRIER_CAUSALITY,
            state=ClaimState.INSUFFICIENT_DATA,
            detail=f"no blind-judge votes for: {', '.join(missing)}",
        )
    control_at_chance = control.ci_low <= 0.5 <= control.ci_high
    carrier_above_chance = carrier.ci_low > 0.5
    if control_at_chance and carrier_above_chance:
        return ClaimResult(
            name=_CLAIM_CARRIER_CAUSALITY,
            state=ClaimState.PASS,
            detail=(
                f"{control_label} CI covers chance "
                f"({control.ci_low:.3f}, {control.ci_high:.3f}); "
                f"{carrier_label} clears it ({carrier.ci_low:.3f})"
            ),
        )
    reasons = []
    if not control_at_chance:
        reasons.append(
            f"{control_label} did not collapse to chance "
            f"({control.ci_low:.3f}, {control.ci_high:.3f})"
        )
    if not carrier_above_chance:
        reasons.append(
            f"{carrier_label} did not clear chance "
            f"(CI low {carrier.ci_low:.3f})"
        )
    return ClaimResult(
        name=_CLAIM_CARRIER_CAUSALITY,
        state=ClaimState.FAIL,
        detail="; ".join(reasons),
    )


def grade_c5(
    observations: Mapping[str, ArmObservation],
    *,
    candidate_arm_label: str = DEFAULT_CANDIDATE_ARM_LABEL,
) -> tuple[C5Grade, str]:
    """Spec §C5 分档: did the sampling-layer carrier also match across users?"""

    candidate = observations.get(candidate_arm_label)
    if candidate is None:
        return C5Grade.UNKNOWN, "candidate arm not observed"
    divergent: list[str] = []
    for probe, turns in sorted(candidate.turns_by_probe().items()):
        if len({turn.decode_fp for turn in turns}) > 1:
            divergent.append(probe)
    if divergent:
        return (
            C5Grade.DECODE_DIVERGENT,
            f"decode_fp differs across users on probes: {', '.join(divergent)}",
        )
    return (
        C5Grade.DECODE_MATCHED,
        "decode_fp identical across users on every probe",
    )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentificationVerdict:
    """The artifact: four claim states, C5 grade, and the evidence behind them."""

    schema_version: str
    verdict_state: VerdictState
    substrate_kind: SubstrateEvidenceKind
    substrate_fingerprint: str
    claims: tuple[ClaimResult, ...]
    c5_grade: C5Grade
    c5_detail: str
    matching: tuple[MatchingReadout, ...]
    prompt_fp_table: tuple[dict[str, object], ...]
    judge_model_id: str
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Which latent carrier claims 1/2/3 were computed against. Without this in
    # the artifact, two runs with identical claim states would be
    # indistinguishable even though they tested different channels.
    candidate_arm_label: str = DEFAULT_CANDIDATE_ARM_LABEL

    def claim(self, name: str) -> ClaimResult:
        for result in self.claims:
            if result.name == name:
                return result
        raise KeyError(f"verdict carries no claim named {name!r}")

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "verdict_state": self.verdict_state.value,
            "substrate_kind": self.substrate_kind.value,
            "substrate_fingerprint": self.substrate_fingerprint,
            "candidate_arm": self.candidate_arm_label,
            "claims": [claim.as_json_dict() for claim in self.claims],
            "c5_grade": self.c5_grade.value,
            "c5_detail": self.c5_detail,
            "matching": [readout.as_json_dict() for readout in self.matching],
            "prompt_fp_table": list(self.prompt_fp_table),
            "judge_model_id": self.judge_model_id,
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(), ensure_ascii=False, indent=2, sort_keys=False
        )


def _resolve_verdict_state(
    *,
    claims: Mapping[str, ClaimResult],
    c5_grade: C5Grade,
    substrate_kind: SubstrateEvidenceKind,
) -> tuple[VerdictState, tuple[str, ...]]:
    """Spec §判据 state machine, with the fake-substrate ceiling applied.

    ``fail`` is reachable on any substrate: a failed prompt-identity or
    carrier-causality check is a negative result, and suppressing negatives on
    a fake substrate would be worse than reporting them. Every positive state
    requires ``FROZEN_WEIGHTS``.
    """

    notes: list[str] = []
    identity = claims[_CLAIM_PROMPT_IDENTITY]
    causality = claims[_CLAIM_CARRIER_CAUSALITY]
    if identity.state is ClaimState.FAIL or causality.state is ClaimState.FAIL:
        return VerdictState.FAIL, tuple(notes)
    if any(
        claims[name].state is not ClaimState.PASS for name in CLAIM_NAMES
    ):
        unmet = [
            name
            for name in CLAIM_NAMES
            if claims[name].state is not ClaimState.PASS
        ]
        notes.append(f"claims not established: {', '.join(unmet)}")
        return VerdictState.INSUFFICIENT_DATA, tuple(notes)
    if substrate_kind is not SubstrateEvidenceKind.FROZEN_WEIGHTS:
        # Reached only if a fake substrate ever reported injection *and* a
        # judge produced above-chance matching. Refusing here is the teeth:
        # a smoke run must not be quotable as the identification result.
        notes.append(
            f"all four claims held but substrate is {substrate_kind.value}; "
            "a retained verdict requires frozen weights whose residual hook "
            "actually fires"
        )
        return VerdictState.INSUFFICIENT_DATA, tuple(notes)
    if c5_grade is C5Grade.DECODE_MATCHED:
        return VerdictState.RETAIN_STRICT, tuple(notes)
    notes.append(
        "sampling-layer channel (C5) not closed: decode_fp differs across "
        "users, so the difference is not solely internal state"
    )
    return VerdictState.RETAIN_PROMPT_CLOSED, tuple(notes)


def build_identification_verdict(
    *,
    observations: Sequence[ArmObservation],
    substrate_kind: SubstrateEvidenceKind,
    substrate_fingerprint: str,
    matching: Sequence[MatchingReadout] = (),
    judge_model_id: str = "",
    extra_notes: Sequence[str] = (),
    candidate_arm_label: str = DEFAULT_CANDIDATE_ARM_LABEL,
) -> IdentificationVerdict:
    """Compute the verdict from recorded turns and (optional) judge votes."""

    if not substrate_fingerprint:
        raise ValueError(
            "identification verdict requires a substrate fingerprint: without "
            "it the four arms cannot be shown to share one frozen substrate "
            "(spec §关键不变量 2)."
        )
    by_label = {observation.arm.label: observation for observation in observations}
    if len(by_label) != len(observations):
        raise ValueError("each arm may be observed at most once per verdict.")
    matching_by_label = {readout.arm_label: readout for readout in matching}
    if len(matching_by_label) != len(matching):
        raise ValueError("each arm may carry at most one matching readout.")
    if matching and not judge_model_id:
        raise ValueError(
            "matching readouts require the judge model id: an unattributed "
            "accuracy cannot be checked against the cross-family rule."
        )

    if candidate_arm_label not in ALL_ARM_LABELS:
        raise ValueError(
            f"unknown candidate arm {candidate_arm_label!r}; expected one of "
            f"{ALL_ARM_LABELS}"
        )
    identity = evaluate_prompt_identity(
        by_label, candidate_arm_label=candidate_arm_label
    )
    if identity.state is ClaimState.FAIL:
        # Spec §判据 1: the experiment is void; do not report downstream claims
        # computed on prompts that were not in fact identical.
        voided = tuple(
            ClaimResult(
                name=name,
                state=ClaimState.INSUFFICIENT_DATA,
                detail="not evaluated: prompt identity failed, experiment void",
            )
            for name in CLAIM_NAMES[1:]
        )
        claims = (identity, *voided)
    else:
        claims = (
            identity,
            evaluate_output_divergence(
                by_label,
                substrate_kind=substrate_kind,
                candidate_arm_label=candidate_arm_label,
            ),
            evaluate_identification(
                matching_by_label, candidate_arm_label=candidate_arm_label
            ),
            evaluate_carrier_causality(
                matching_by_label, candidate_arm_label=candidate_arm_label
            ),
        )
    c5_grade, c5_detail = grade_c5(
        by_label, candidate_arm_label=candidate_arm_label
    )
    state, notes = _resolve_verdict_state(
        claims={claim.name: claim for claim in claims},
        c5_grade=c5_grade,
        substrate_kind=substrate_kind,
    )
    prompt_fp_table = tuple(
        turn.as_json_dict()
        for label in ALL_ARM_LABELS
        if label in by_label
        for turn in by_label[label].turns
    )
    return IdentificationVerdict(
        schema_version=IDENTIFICATION_SCHEMA_VERSION,
        verdict_state=state,
        substrate_kind=substrate_kind,
        substrate_fingerprint=substrate_fingerprint,
        claims=claims,
        c5_grade=c5_grade,
        c5_detail=c5_detail,
        matching=tuple(
            matching_by_label[label]
            for label in IDENTIFICATION_ARM_LABELS
            if label in matching_by_label
        ),
        prompt_fp_table=prompt_fp_table,
        judge_model_id=judge_model_id,
        notes=(*notes, *extra_notes),
        candidate_arm_label=candidate_arm_label,
    )


def run_identification_smoke(
    *,
    cases: Sequence[ProbeCase],
    synthesizer: object,
    base_context: ResponseContext,
    substrate_kind: SubstrateEvidenceKind,
    substrate_fingerprint: str,
    arm_labels: Sequence[str] = IDENTIFICATION_ARM_LABELS,
    judge: BlindMatchingJudge | None = None,
    bootstrap_seed: int = 20260726,
    candidate_arm_label: str = DEFAULT_CANDIDATE_ARM_LABEL,
) -> IdentificationVerdict:
    """Run every arm over every probe case and build the verdict.

    ``judge=None`` is the P0-smoke path: claims 3 and 4 come back
    ``insufficient_data`` rather than being filled with a placeholder number.
    """

    observations = [
        observe_arm(
            arm=arm_from_profile(label),
            cases=cases,
            synthesizer=synthesizer,
            base_context=base_context,
        )
        for label in arm_labels
    ]
    matching: list[MatchingReadout] = []
    if judge is not None:
        candidate_ids = sorted({case.user_id for case in cases})
        if len(candidate_ids) != 2:
            raise ValueError(
                "two-alternative matching requires exactly two users, got "
                f"{candidate_ids}"
            )
        for observation in observations:
            votes = [
                judge.match(
                    response_text=turn.text, candidate_user_ids=candidate_ids
                )
                == turn.user_id
                for turn in observation.turns
            ]
            accuracy, ci_low, ci_high = bootstrap_matching_ci(
                votes, seed=bootstrap_seed
            )
            matching.append(
                MatchingReadout(
                    arm_label=observation.arm.label,
                    correct=sum(1 for vote in votes if vote),
                    total=len(votes),
                    accuracy=accuracy,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    judge_model_id=judge.judge_model_id,
                )
            )
    return build_identification_verdict(
        observations=observations,
        substrate_kind=substrate_kind,
        substrate_fingerprint=substrate_fingerprint,
        matching=tuple(matching),
        judge_model_id=judge.judge_model_id if judge is not None else "",
        candidate_arm_label=candidate_arm_label,
    )
