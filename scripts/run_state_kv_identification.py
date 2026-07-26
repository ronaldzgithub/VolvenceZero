#!/usr/bin/env python3
"""Run the State KV carrier-identification arms and emit verdict_identification.json.

Stage P0-smoke of ``docs/specs/state-kv-identification-evidence.md`` §执行阶段:
four arms, two personas, K probe sentences, on a deterministic fake substrate.
Cost is zero and the expected verdict is ``insufficient_data`` -- the run exists
to prove that the pure arms' prompts are byte-identical and that all three
attestation tags are emitted on every turn, which is what P1-directional needs
before any money is spent on a frozen model plus a cross-family judge.

The personas here are hand-written probe material, not the longitudinal
harness: they carry divergent typed readouts and divergent assemblies so that a
prompt-carrier leak would show up as a ``prompt_fp`` mismatch. Swapping in the
20-session harness and a real runtime is a change of two arguments to
``run_identification_smoke``, not a change to the claim logic.

Usage:
    python scripts/run_state_kv_identification.py \
        --output artifacts/state_kv/verdict_identification.json
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.agent.response import (  # noqa: E402
    LLMResponseSynthesizer,
    ResponseContext,
)
from volvence_zero.application.runtime import (  # noqa: E402
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.personal_conditioning_contracts import (  # noqa: E402
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.personal_conditioning_rendering import (  # noqa: E402
    render_personal_conditioning_statement,
)
from volvence_zero.state_kv_identification import (  # noqa: E402
    IDENTIFICATION_ARM_LABELS,
    ProbeCase,
    SubstrateEvidenceKind,
    run_identification_smoke,
)

# Probe sentences: identical for both users, so any per-user difference in the
# response must come from state, not from what was asked.
PROBE_SENTENCES: tuple[tuple[str, str], ...] = (
    ("p0", "我又搞砸了"),
    ("p1", "今天还是没睡好"),
    ("p2", "你觉得我该继续吗"),
)

# Two divergent personas. The 16 coordinates are typed owner readouts; the
# values differ per persona so the residual carrier has something to carry and
# the rendered statement (arm B-prime) has the same information content.
PERSONAS: tuple[tuple[str, float, str, str], ...] = (
    (
        "persona-a",
        0.82,
        "Carry forward continuity from prior context: her cat died last week.",
        "continuum-support-first",
    ),
    (
        "persona-b",
        0.24,
        "Carry forward continuity from prior context: he starts a new job Monday.",
        "continuum-structure-first",
    ),
)


class DeterministicFakeSubstrate:
    """Trace-only fake: derives text from the prompt, never injects.

    Mirrors the synthetic runtime's contract
    (``personal_conditioning_applied=False``) so the smoke run exercises the
    honest path -- claim 2 must come back ``insufficient_data`` rather than
    being satisfied by a fake that pretends to inject.
    """

    model_id = "deterministic-fake-substrate"

    def __init__(self, *, applies_conditioning: bool = False) -> None:
        self._applies = applies_conditioning

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            f"{self.model_id}:applies={self._applies}".encode("utf-8")
        ).hexdigest()[:16]

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        conditioning = kwargs.get("personal_conditioning")
        applied = self._applies and conditioning is not None
        parts = [
            str(kwargs.get("system_context", "")),
            str(kwargs.get("prompt", "")),
        ]
        if applied:
            parts.append(conditioning.source_fingerprint)
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8]
        return SimpleNamespace(
            text=f"reply-{digest}",
            token_count=len(parts),
            personal_conditioning_applied=applied,
        )


def _assembly(*, residue: str, ordering_driver: str) -> ResponseAssemblySnapshot:
    return ResponseAssemblySnapshot(
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
        prompt_residue_summary=residue,
        prompt_residue_ratio=0.4,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="carrier-identification probe assembly",
        ordering_driver=ordering_driver,
    )


def _conditioning(*, user_id: str, fill: float) -> PersonalConditioningSnapshot:
    state_vector = tuple(
        min(1.0, max(0.0, fill + 0.01 * index))
        for index in range(len(PERSONAL_CONDITIONING_VECTOR_LABELS))
    )
    statement = render_personal_conditioning_statement(
        state_vector=state_vector,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        confidence=0.72,
        is_cold_start=False,
    )
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=state_vector,
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 3), ("relationship_state", 2)),
        source_fingerprint=hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16],
        confidence=0.72,
        is_cold_start=False,
        description=f"probe state for {user_id}",
        rendered_statement=statement,
    )


def build_probe_cases() -> tuple[ProbeCase, ...]:
    cases: list[ProbeCase] = []
    for user_id, fill, residue, ordering_driver in PERSONAS:
        assembly = _assembly(residue=residue, ordering_driver=ordering_driver)
        conditioning = _conditioning(user_id=user_id, fill=fill)
        for probe_id, sentence in PROBE_SENTENCES:
            cases.append(
                ProbeCase(
                    user_id=user_id,
                    probe_id=probe_id,
                    user_input=sentence,
                    conditioning=conditioning,
                    assembly=assembly,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "artifacts/state_kv/verdict_identification.json"),
        help="where to write verdict_identification.json",
    )
    parser.add_argument(
        "--inject",
        action="store_true",
        help=(
            "let the fake substrate report injection (exercises claim 2 on a "
            "fake; the verdict is still capped at insufficient_data)"
        ),
    )
    args = parser.parse_args(argv)

    runtime = DeterministicFakeSubstrate(applies_conditioning=args.inject)
    verdict = run_identification_smoke(
        cases=build_probe_cases(),
        synthesizer=LLMResponseSynthesizer(runtime=runtime),
        base_context=_base_context(),
        substrate_kind=SubstrateEvidenceKind.TRACE_ONLY,
        substrate_fingerprint=runtime.fingerprint,
        arm_labels=IDENTIFICATION_ARM_LABELS,
        judge=None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(verdict.to_json() + "\n", encoding="utf-8")

    print(f"verdict_state = {verdict.verdict_state.value}")
    for claim in verdict.claims:
        print(f"  {claim.name:38s} {claim.state.value:18s} {claim.detail}")
    print(f"  c5_grade{'':30s} {verdict.c5_grade.value:18s} {verdict.c5_detail}")
    for note in verdict.notes:
        print(f"  note: {note}")
    print(f"turns recorded: {len(verdict.prompt_fp_table)}")
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
