"""Carrier-identification runner acceptance (State KV package 2).

These tests pin down the three properties that make
``verdict_identification.json`` worth publishing:

1. On the deterministic fake substrate the four arms really do run, claim 1
   (byte-identical pure-arm prompts) is *computed* and passes, and the overall
   verdict is ``insufficient_data`` -- the honest ceiling for a fake.
2. A fake substrate can never yield a retained verdict, even if every claim
   somehow passes. This is the anti-overclaim guard; without it the zero-cost
   smoke would be quotable as the real identification result.
3. Claims 3 and 4 stay ``insufficient_data`` when no blind judge ran, instead
   of being filled with a placeholder accuracy.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from volvence_zero.agent.response import LLMResponseSynthesizer, ResponseContext
from volvence_zero.application.runtime import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.state_kv_identification import (
    IDENTIFICATION_ARM_LABELS,
    IDENTIFICATION_SCHEMA_VERSION,
    PURE_ARM_LABELS,
    ArmObservation,
    C5Grade,
    ClaimState,
    IdentificationTurn,
    MatchingReadout,
    ProbeCase,
    SubstrateEvidenceKind,
    VerdictState,
    arm_from_profile,
    bootstrap_matching_ci,
    build_identification_verdict,
    context_for_arm,
    run_identification_smoke,
)

_PROBES = (
    ("p0", "我又搞砸了"),
    ("p1", "今天还是没睡好"),
)


class _FakeRuntime:
    """Deterministic fake substrate.

    ``applies_conditioning`` models the two substrate kinds: the trace-only
    synthetic runtime reports ``personal_conditioning_applied=False`` by
    contract, while a hook-bearing runtime reports ``True``. Output text is
    derived from the prompt bytes and (when applied) from the conditioning
    vector, so divergence in this fake reflects exactly the carriers that were
    open -- no hidden channel.
    """

    model_id = "fake-frozen-substrate"

    def __init__(self, *, applies_conditioning: bool) -> None:
        self._applies = applies_conditioning
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        conditioning = kwargs.get("personal_conditioning")
        applied = self._applies and conditioning is not None
        parts = [str(kwargs.get("system_context", "")), str(kwargs.get("prompt", ""))]
        if applied:
            parts.append(conditioning.source_fingerprint)
        # sha256, not ``hash()``: str hashing is salted per process, so a
        # ``hash()``-derived reply would make the artifact differ between runs
        # of the same "deterministic" fake.
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8]
        return SimpleNamespace(
            text=f"reply-{digest}",
            token_count=7,
            personal_conditioning_applied=applied,
        )


def _assembly(*, user: str) -> ResponseAssemblySnapshot:
    base = ResponseAssemblySnapshot(
        regime_id="steady",
        regime_name="Steady",
        abstract_action=None,
        response_mode=ResponseMode.SUPPORT,
        answer_depth_limit="high-level-only",
        citation_mode="none",
        clarification_required=False,
        refer_out_required=False,
        ordering_plan=(),
        knowledge_briefs=(),
        case_briefs=(),
        playbook_ordering=(),
        required_disclaimers=(),
        required_disclaimer_phrases=(),
        control_code=(),
        control_scale=0.0,
        max_questions=0,
        prompt_residue_summary="",
        prompt_residue_ratio=0.0,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="identification probe assembly",
    )
    # Divergent per-user state, exactly the material the pure arms must not
    # leak into the prompt.
    if user == "alice":
        return replace(
            base,
            regime_name="Steady Support",
            prompt_residue_summary=(
                "Carry forward continuity from prior context: her cat died."
            ),
            required_disclaimers=("grief-sensitive",),
            ordering_driver="continuum-support-first",
        )
    return replace(
        base,
        regime_name="Task Focus",
        prompt_residue_summary=(
            "Carry forward continuity from prior context: new job Monday."
        ),
        clarification_required=True,
        ordering_driver="continuum-structure-first",
    )


def _conditioning(*, fill: float, user: str) -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(fill for _ in PERSONAL_CONDITIONING_VECTOR_LABELS),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 2),),
        source_fingerprint=f"conditioning-{user}",
        confidence=0.7,
        is_cold_start=False,
        description=f"{user} probe state",
        rendered_statement=f"State readout for {user}: engagement {fill:.2f}.",
    )


def _cases() -> tuple[ProbeCase, ...]:
    cases: list[ProbeCase] = []
    for user, fill in (("alice", 0.8), ("bob", 0.2)):
        for probe_id, text in _PROBES:
            cases.append(
                ProbeCase(
                    user_id=user,
                    probe_id=probe_id,
                    user_input=text,
                    conditioning=_conditioning(fill=fill, user=user),
                    assembly=_assembly(user=user),
                )
            )
    return tuple(cases)


def _base_context() -> ResponseContext:
    return ResponseContext(
        regime_id="steady",
        regime_name="Steady",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="none",
    )


def _run(*, applies_conditioning: bool, judge: Any = None):
    synthesizer = LLMResponseSynthesizer(
        runtime=_FakeRuntime(applies_conditioning=applies_conditioning)
    )
    return run_identification_smoke(
        cases=_cases(),
        synthesizer=synthesizer,
        base_context=_base_context(),
        substrate_kind=(
            SubstrateEvidenceKind.FROZEN_WEIGHTS
            if applies_conditioning
            else SubstrateEvidenceKind.TRACE_ONLY
        ),
        substrate_fingerprint="fake-substrate-fp",
        judge=judge,
    )


# ---------------------------------------------------------------------------
# Arm settings come from the profile registry, not a local table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "active", "mode", "delivery"),
    [
        ("state-kv-arm-a-pure", False, "residual", "suppressed"),
        ("state-kv-arm-e-pure", True, "residual", "suppressed"),
        ("state-kv-arm-bprime", True, "text", "text"),
        ("state-kv-arm-e", True, "residual", "text"),
    ],
)
def test_arm_settings_are_resolved_from_the_profile_registry(
    label: str, active: bool, mode: str, delivery: str
) -> None:
    arm = arm_from_profile(label)
    assert (arm.conditioning_active, arm.conditioning_mode) == (active, mode)
    assert arm.prompt_state_delivery == delivery


def test_shadow_arm_delivers_no_state_through_either_path() -> None:
    case = _cases()[0]
    context = context_for_arm(
        arm=arm_from_profile("state-kv-arm-a-pure"),
        case=case,
        base_context=_base_context(),
    )
    assert context.personal_conditioning is None
    assert context.personal_conditioning_statement == ""
    assert context.prompt_state_delivery == "suppressed"


def test_text_arm_delivers_the_rendered_statement_only() -> None:
    case = _cases()[0]
    context = context_for_arm(
        arm=arm_from_profile("state-kv-arm-bprime"),
        case=case,
        base_context=_base_context(),
    )
    assert context.personal_conditioning is None
    assert context.personal_conditioning_statement
    assert context.personal_conditioning_statement_ref
    assert context.prompt_state_delivery == "text"


def test_text_arm_refuses_an_empty_rendered_statement() -> None:
    """A text arm with nothing to say is not a control for a residual arm."""

    case = _cases()[0]
    blank = replace(case, conditioning=replace(case.conditioning, rendered_statement=""))
    with pytest.raises(ValueError, match="renders empty"):
        context_for_arm(
            arm=arm_from_profile("state-kv-arm-bprime"),
            case=blank,
            base_context=_base_context(),
        )


# ---------------------------------------------------------------------------
# The smoke run: claim 1 computed and passing, verdict honestly capped
# ---------------------------------------------------------------------------


def test_smoke_run_proves_prompt_identity_on_the_pure_arms() -> None:
    verdict = _run(applies_conditioning=False)
    identity = verdict.claim("claim_prompt_identity")
    assert identity.state is ClaimState.PASS, identity.detail
    # 2 arms x 2 users x 2 probes
    assert "8 turns" in identity.detail


def test_smoke_run_records_every_arm_in_spec_order() -> None:
    verdict = _run(applies_conditioning=False)
    seen = []
    for row in verdict.prompt_fp_table:
        if row["arm"] not in seen:
            seen.append(row["arm"])
    assert tuple(seen) == IDENTIFICATION_ARM_LABELS
    assert verdict.schema_version == IDENTIFICATION_SCHEMA_VERSION
    assert len(verdict.prompt_fp_table) == 4 * len(_cases())


def test_trace_only_substrate_is_insufficient_not_success() -> None:
    """Passing a snapshot is not injecting it."""

    verdict = _run(applies_conditioning=False)
    divergence = verdict.claim("claim_output_divergence")
    assert divergence.state is ClaimState.INSUFFICIENT_DATA
    assert "did not inject the residual" in divergence.detail
    assert verdict.verdict_state is VerdictState.INSUFFICIENT_DATA


def test_missing_judge_leaves_identification_unmeasured() -> None:
    verdict = _run(applies_conditioning=False)
    for name in ("claim_identification_above_chance", "claim_carrier_causality"):
        claim = verdict.claim(name)
        assert claim.state is ClaimState.INSUFFICIENT_DATA
        assert "judge" in claim.detail
    assert verdict.matching == ()
    assert verdict.judge_model_id == ""


def test_injecting_substrate_diverges_outputs_across_users() -> None:
    verdict = _run(applies_conditioning=True)
    assert verdict.claim("claim_prompt_identity").state is ClaimState.PASS
    assert verdict.claim("claim_output_divergence").state is ClaimState.PASS
    # Still insufficient overall: no judge ran.
    assert verdict.verdict_state is VerdictState.INSUFFICIENT_DATA


def test_pure_arms_share_prompt_bytes_while_only_residual_carries_state() -> None:
    """The carrier-identification setup, checked on the recorded turns."""

    verdict = _run(applies_conditioning=True)
    by_arm: dict[str, list[dict[str, object]]] = {}
    for row in verdict.prompt_fp_table:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    pure_fps = {
        row["prompt_fp"]
        for label in PURE_ARM_LABELS
        for row in by_arm[label]
        if row["probe"] == "p0"
    }
    assert len(pure_fps) == 1
    text_fps = {
        row["prompt_fp"] for row in by_arm["state-kv-arm-bprime"] if row["probe"] == "p0"
    }
    # The text arm must be observably different, per user.
    assert len(text_fps) == 2
    assert not (pure_fps & text_fps)
    assert all(
        row["prompt_state_sections"] == 0
        for label in PURE_ARM_LABELS
        for row in by_arm[label]
    )


# ---------------------------------------------------------------------------
# Anti-overclaim guard
# ---------------------------------------------------------------------------


def _turn(
    *,
    arm: str,
    user: str,
    probe: str,
    prompt_fp: str,
    decode_fp: str = "decode-same",
    applied: bool = True,
    delivered: bool = True,
    text: str | None = None,
) -> IdentificationTurn:
    return IdentificationTurn(
        arm_label=arm,
        user_id=user,
        probe_id=probe,
        prompt_fp=prompt_fp,
        prompt_state_sections=0,
        decode_fp=decode_fp,
        conditioning_applied=applied,
        conditioning_delivered=delivered,
        text=text if text is not None else f"{arm}-{user}-{probe}",
    )


def _observation(label: str, **turn_kwargs: Any) -> ArmObservation:
    return ArmObservation(
        arm=arm_from_profile(label),
        turns=tuple(
            _turn(arm=label, user=user, probe="p0", prompt_fp="fp-shared", **turn_kwargs)
            for user in ("alice", "bob")
        ),
    )


def _perfect_matching(label: str) -> MatchingReadout:
    return MatchingReadout(
        arm_label=label,
        correct=20,
        total=20,
        accuracy=1.0,
        ci_low=0.9,
        ci_high=1.0,
        judge_model_id="judge-x",
    )


def _chance_matching(label: str) -> MatchingReadout:
    return MatchingReadout(
        arm_label=label,
        correct=10,
        total=20,
        accuracy=0.5,
        ci_low=0.3,
        ci_high=0.7,
        judge_model_id="judge-x",
    )


def test_fake_substrate_cannot_produce_a_retained_verdict() -> None:
    """All four claims passing on a fake still may not be quoted as retention."""

    verdict = build_identification_verdict(
        observations=[
            _observation(PURE_ARM_LABELS[0]),
            _observation(PURE_ARM_LABELS[1]),
            _observation("state-kv-arm-bprime"),
        ],
        substrate_kind=SubstrateEvidenceKind.TRACE_ONLY,
        substrate_fingerprint="fake-fp",
        matching=(
            _chance_matching(PURE_ARM_LABELS[0]),
            _perfect_matching(PURE_ARM_LABELS[1]),
            _perfect_matching("state-kv-arm-bprime"),
        ),
        judge_model_id="judge-x",
    )
    assert all(claim.state is ClaimState.PASS for claim in verdict.claims)
    assert verdict.verdict_state is VerdictState.INSUFFICIENT_DATA
    assert any("requires frozen weights" in note for note in verdict.notes)


def test_frozen_weights_with_matched_decode_reaches_retain_strict() -> None:
    verdict = build_identification_verdict(
        observations=[
            _observation(PURE_ARM_LABELS[0]),
            _observation(PURE_ARM_LABELS[1]),
            _observation("state-kv-arm-bprime"),
        ],
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        substrate_fingerprint="qwen-fp",
        matching=(
            _chance_matching(PURE_ARM_LABELS[0]),
            _perfect_matching(PURE_ARM_LABELS[1]),
            _perfect_matching("state-kv-arm-bprime"),
        ),
        judge_model_id="judge-x",
    )
    assert verdict.c5_grade is C5Grade.DECODE_MATCHED
    assert verdict.verdict_state is VerdictState.RETAIN_STRICT


def test_divergent_decode_config_downgrades_to_prompt_closed() -> None:
    candidate = ArmObservation(
        arm=arm_from_profile(PURE_ARM_LABELS[1]),
        turns=(
            _turn(
                arm=PURE_ARM_LABELS[1],
                user="alice",
                probe="p0",
                prompt_fp="fp-shared",
                decode_fp="decode-a",
            ),
            _turn(
                arm=PURE_ARM_LABELS[1],
                user="bob",
                probe="p0",
                prompt_fp="fp-shared",
                decode_fp="decode-b",
            ),
        ),
    )
    verdict = build_identification_verdict(
        observations=[_observation(PURE_ARM_LABELS[0]), candidate,
                      _observation("state-kv-arm-bprime")],
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        substrate_fingerprint="qwen-fp",
        matching=(
            _chance_matching(PURE_ARM_LABELS[0]),
            _perfect_matching(PURE_ARM_LABELS[1]),
            _perfect_matching("state-kv-arm-bprime"),
        ),
        judge_model_id="judge-x",
    )
    assert verdict.c5_grade is C5Grade.DECODE_DIVERGENT
    assert verdict.verdict_state is VerdictState.RETAIN_PROMPT_CLOSED
    assert any("C5" in note for note in verdict.notes)


def test_prompt_identity_failure_voids_downstream_claims() -> None:
    diverged = ArmObservation(
        arm=arm_from_profile(PURE_ARM_LABELS[1]),
        turns=(
            _turn(
                arm=PURE_ARM_LABELS[1],
                user="alice",
                probe="p0",
                prompt_fp="fp-other",
            ),
            _turn(
                arm=PURE_ARM_LABELS[1], user="bob", probe="p0", prompt_fp="fp-other"
            ),
        ),
    )
    verdict = build_identification_verdict(
        observations=[_observation(PURE_ARM_LABELS[0]), diverged],
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        substrate_fingerprint="qwen-fp",
        matching=(
            _chance_matching(PURE_ARM_LABELS[0]),
            _perfect_matching(PURE_ARM_LABELS[1]),
        ),
        judge_model_id="judge-x",
    )
    assert verdict.verdict_state is VerdictState.FAIL
    assert verdict.claim("claim_prompt_identity").state is ClaimState.FAIL
    for name in (
        "claim_output_divergence",
        "claim_identification_above_chance",
        "claim_carrier_causality",
    ):
        claim = verdict.claim(name)
        assert claim.state is ClaimState.INSUFFICIENT_DATA
        assert "experiment void" in claim.detail


def test_state_sections_in_a_pure_arm_fail_claim_one() -> None:
    leaking = ArmObservation(
        arm=arm_from_profile(PURE_ARM_LABELS[1]),
        turns=(
            replace(
                _turn(
                    arm=PURE_ARM_LABELS[1],
                    user="alice",
                    probe="p0",
                    prompt_fp="fp-shared",
                ),
                prompt_state_sections=2,
            ),
            _turn(
                arm=PURE_ARM_LABELS[1], user="bob", probe="p0", prompt_fp="fp-shared"
            ),
        ),
    )
    verdict = build_identification_verdict(
        observations=[_observation(PURE_ARM_LABELS[0]), leaking],
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        substrate_fingerprint="qwen-fp",
    )
    claim = verdict.claim("claim_prompt_identity")
    assert claim.state is ClaimState.FAIL
    assert "state-derived sections present" in claim.detail


def test_verdict_requires_a_substrate_fingerprint() -> None:
    with pytest.raises(ValueError, match="substrate fingerprint"):
        build_identification_verdict(
            observations=[_observation(PURE_ARM_LABELS[0])],
            substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
            substrate_fingerprint="",
        )


def test_matching_readout_requires_a_named_judge() -> None:
    with pytest.raises(ValueError, match="judge model id"):
        build_identification_verdict(
            observations=[_observation(PURE_ARM_LABELS[0])],
            substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
            substrate_fingerprint="qwen-fp",
            matching=(_chance_matching(PURE_ARM_LABELS[0]),),
        )


def test_identical_outputs_across_users_fail_divergence() -> None:
    flat = ArmObservation(
        arm=arm_from_profile(PURE_ARM_LABELS[1]),
        turns=(
            _turn(
                arm=PURE_ARM_LABELS[1],
                user="alice",
                probe="p0",
                prompt_fp="fp-shared",
                text="same",
            ),
            _turn(
                arm=PURE_ARM_LABELS[1],
                user="bob",
                probe="p0",
                prompt_fp="fp-shared",
                text="same",
            ),
        ),
    )
    verdict = build_identification_verdict(
        observations=[_observation(PURE_ARM_LABELS[0]), flat],
        substrate_kind=SubstrateEvidenceKind.FROZEN_WEIGHTS,
        substrate_fingerprint="qwen-fp",
    )
    claim = verdict.claim("claim_output_divergence")
    assert claim.state is ClaimState.FAIL
    assert "identical text across users" in claim.detail


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_is_seeded_and_reproducible() -> None:
    votes = [True] * 14 + [False] * 6
    first = bootstrap_matching_ci(votes, seed=7, resamples=500)
    second = bootstrap_matching_ci(votes, seed=7, resamples=500)
    assert first == second
    accuracy, low, high = first
    assert accuracy == pytest.approx(0.7)
    assert 0.0 <= low <= accuracy <= high <= 1.0


def test_bootstrap_ci_covers_chance_for_a_coin_flip() -> None:
    votes = [True, False] * 10
    _, low, high = bootstrap_matching_ci(votes, seed=11, resamples=500)
    assert low <= 0.5 <= high


def test_bootstrap_ci_clears_chance_for_a_perfect_judge() -> None:
    _, low, _ = bootstrap_matching_ci([True] * 20, seed=11, resamples=500)
    assert low > 0.5


def test_bootstrap_ci_rejects_empty_votes() -> None:
    with pytest.raises(ValueError, match="at least one vote"):
        bootstrap_matching_ci([], seed=1)


# ---------------------------------------------------------------------------
# Attestation channel integrity
# ---------------------------------------------------------------------------


def test_missing_attestation_tag_is_an_error_not_a_default() -> None:
    """A claim computed from an absent fingerprint would be unfalsifiable."""

    class _TaglessSynthesizer:
        def synthesize(self, *, context: Any, assembly: Any) -> Any:
            return SimpleNamespace(text="hi", rationale_tags=())

    with pytest.raises(ValueError, match="attestation tag"):
        run_identification_smoke(
            cases=_cases(),
            synthesizer=_TaglessSynthesizer(),
            base_context=_base_context(),
            substrate_kind=SubstrateEvidenceKind.TRACE_ONLY,
            substrate_fingerprint="fake-fp",
            arm_labels=(PURE_ARM_LABELS[0],),
        )
