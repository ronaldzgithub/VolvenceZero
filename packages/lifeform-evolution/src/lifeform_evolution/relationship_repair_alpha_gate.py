"""Matched-control gate for closed-alpha relationship repair expression.

This is intentionally internal/scripted. It verifies the alpha invariant:
the same typed external rupture is observable in both arms, but only the
repair-alpha-enabled arm may turn the runtime advisory into expression
evidence (typed ``repair_alpha=<kind>`` / ``intent=repair-first`` /
``acknowledge_section=repair_alpha`` tags in
``AgentResponse.rationale_tags``).

The gate reads the structured ``rationale_tags`` tuple, NOT substring
matches against rendered text. Any phrasing change must therefore go
through the typed tag schema, which is the SSOT contract enforced by
``tests/lifeform_expression/test_rationale_tags.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any

from lifeform_core import DialogueExternalOutcomeKind, LifeformConfig
from lifeform_domain_emogpt import build_companion_lifeform
from lifeform_expression import GroundedResponseSynthesizer, PromptPlanner
from volvence_zero.brain import BrainConfig
from volvence_zero.memory import (
    StaticIdentityProvider,
    UserIdentity,
    list_durable_entries_for_scope,
)
from volvence_zero.rupture_state import RuptureStateSnapshot


@dataclasses.dataclass(frozen=True)
class RepairAlphaArmReport:
    arm: str
    repair_alpha_enabled: bool
    rupture_kind: str
    rupture_confidence: float
    repair_alpha_rationale_present: bool
    repair_first_intent_present: bool
    repair_alpha_phrase_present: bool
    durable_rupture_repair_count: int
    observed_repair_memory_count: int
    same_user_recall_count: int
    cross_user_leakage_count: int
    response_text: str
    response_rationale: str

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RepairAlphaGateReport:
    passed: bool
    treatment: RepairAlphaArmReport
    matched_control: RepairAlphaArmReport
    gate_items: tuple[tuple[str, bool, str], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "treatment": self.treatment.to_json(),
            "matched_control": self.matched_control.to_json(),
            "gate_items": [
                {"name": name, "passed": passed, "detail": detail}
                for name, passed, detail in self.gate_items
            ],
        }


def format_relationship_repair_alpha_report(report: RepairAlphaGateReport) -> str:
    status = "PASSED" if report.passed else "FAILED"
    lines = [f"Relationship repair alpha gate: {status}"]
    lines.append(
        "  treatment: "
        f"rupture={report.treatment.rupture_kind or '<none>'} "
        f"repair_alpha={report.treatment.repair_alpha_rationale_present} "
        f"observed_memory={report.treatment.observed_repair_memory_count} "
        f"same_user_recall={report.treatment.same_user_recall_count} "
        f"leakage={report.treatment.cross_user_leakage_count}"
    )
    lines.append(
        "  control: "
        f"rupture={report.matched_control.rupture_kind or '<none>'} "
        f"repair_alpha={report.matched_control.repair_alpha_rationale_present} "
        f"observed_memory={report.matched_control.observed_repair_memory_count}"
    )
    lines.append("  gate_items:")
    for name, passed, detail in report.gate_items:
        lines.append(f"    - {name}: {'ok' if passed else 'FAIL'} ({detail})")
    return "\n".join(lines)


def _build_lifeform(
    *,
    repair_alpha_enabled: bool,
    scope_root_dir: Path,
    user_id: str,
):
    synthesizer = GroundedResponseSynthesizer(
        planner=PromptPlanner(repair_alpha_enabled=repair_alpha_enabled)
    )
    config = LifeformConfig(
        brain_config=BrainConfig(memory_scope_root_dir=str(scope_root_dir))
    )
    identity = UserIdentity(user_id=user_id, scope_key=user_id)
    return build_companion_lifeform(
        config=config,
        response_synthesizer=synthesizer,
        identity_provider=StaticIdentityProvider(identity=identity),
    )


async def _run_arm(
    *,
    repair_alpha_enabled: bool,
    scope_root_dir: Path,
) -> RepairAlphaArmReport:
    user_id = "alice" if repair_alpha_enabled else "control"
    lifeform = _build_lifeform(
        repair_alpha_enabled=repair_alpha_enabled,
        scope_root_dir=scope_root_dir,
        user_id=user_id,
    )
    arm = "treatment" if repair_alpha_enabled else "matched_control"
    session = lifeform.create_session(session_id=f"repair-alpha-{arm}")

    await session.run_turn(
        "I am trying to talk about a scary transition without turning it into a project."
    )
    await session.run_turn(
        "That felt clinical and procedural. I am not asking you to optimise me."
    )
    session.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        confidence=0.95,
        turn_index=2,
        evidence_ref=f"repair-alpha:{arm}:over-directive",
        description="Scripted typed rupture for matched-control repair-alpha gate.",
    )
    result = await session.run_turn(
        "Can we back up and repair the tone before you suggest anything else?"
    )

    rupture_snapshot = session.latest_shadow_snapshots.get("rupture_state")
    rupture_value = rupture_snapshot.value if rupture_snapshot is not None else None
    if rupture_value is not None and not isinstance(
        rupture_value, RuptureStateSnapshot
    ):
        raise TypeError(
            "rupture_state shadow snapshot has unexpected payload "
            f"type {type(rupture_value).__name__}"
        )
    if rupture_value is None:
        rupture_kind = ""
        rupture_confidence = 0.0
    else:
        rupture_kind = (
            rupture_value.rupture_kind.value
            if rupture_value.rupture_kind is not None
            else ""
        )
        rupture_confidence = float(rupture_value.confidence)
    rationale_tags = tuple(result.response.rationale_tags)
    rationale = result.response.rationale
    text = result.response.text
    if repair_alpha_enabled and "repair_alpha=over_directive" in rationale_tags:
        session.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.OVER_DIRECTIVE,
            confidence=0.95,
            turn_index=3,
            evidence_ref=f"repair-alpha:{arm}:reviewed-over-directive",
            description="Review provenance for the rupture after repair expression.",
        )
        session.submit_dialogue_outcome(
            kind=DialogueExternalOutcomeKind.FELT_HEARD,
            confidence=0.9,
            turn_index=3,
            evidence_ref=f"repair-alpha:{arm}:felt-heard",
            description="Scripted user confirms the repair felt heard.",
        )
        await session.run_turn(
            "Yes, that repair felt more heard. Please remember that slower frame."
        )
    await session.end_scene(reason=f"repair-alpha-{arm}-end")
    durable_entries = list_durable_entries_for_scope(
        session.brain_session.runner.memory_store,
        user_scope=user_id,
    )
    rupture_entries = tuple(
        entry for entry in durable_entries if "rupture_repair" in entry.tags
    )
    observed_entries = tuple(
        entry for entry in rupture_entries if "repair_outcome:observed" in entry.tags
    )
    recall_lifeform = _build_lifeform(
        repair_alpha_enabled=repair_alpha_enabled,
        scope_root_dir=scope_root_dir,
        user_id=user_id,
    )
    recall_session = recall_lifeform.create_session(
        session_id=f"repair-alpha-{arm}-recall"
    )
    same_user_recall_entries = tuple(
        entry
        for entry in list_durable_entries_for_scope(
            recall_session.brain_session.runner.memory_store,
            user_scope=user_id,
        )
        if "rupture_repair" in entry.tags
    )
    bob_lifeform = _build_lifeform(
        repair_alpha_enabled=repair_alpha_enabled,
        scope_root_dir=scope_root_dir,
        user_id=f"{user_id}-bob",
    )
    bob_session = bob_lifeform.create_session(session_id=f"repair-alpha-{arm}-bob")
    cross_user_entries = tuple(
        entry
        for entry in list_durable_entries_for_scope(
            bob_session.brain_session.runner.memory_store,
            user_scope=user_id,
        )
        if "rupture_repair" in entry.tags
    )
    return RepairAlphaArmReport(
        arm=arm,
        repair_alpha_enabled=repair_alpha_enabled,
        rupture_kind=rupture_kind,
        rupture_confidence=rupture_confidence,
        repair_alpha_rationale_present="repair_alpha=over_directive" in rationale_tags,
        repair_first_intent_present="intent=repair-first" in rationale_tags,
        repair_alpha_phrase_present="acknowledge_section=repair_alpha" in rationale_tags,
        durable_rupture_repair_count=len(rupture_entries),
        observed_repair_memory_count=len(observed_entries),
        same_user_recall_count=len(same_user_recall_entries),
        cross_user_leakage_count=len(cross_user_entries),
        response_text=text,
        response_rationale=rationale,
    )


async def run_relationship_repair_alpha_gate_async(
    *,
    out_path: str | Path | None = None,
    scope_root_dir: str | Path | None = None,
) -> RepairAlphaGateReport:
    if scope_root_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return await run_relationship_repair_alpha_gate_async(
                out_path=out_path,
                scope_root_dir=Path(tmp),
            )
    scope_root = Path(scope_root_dir)
    treatment = await _run_arm(
        repair_alpha_enabled=True,
        scope_root_dir=scope_root,
    )
    control = await _run_arm(
        repair_alpha_enabled=False,
        scope_root_dir=scope_root,
    )

    treatment_rupture_ok = treatment.rupture_kind == "over_directive"
    control_rupture_ok = control.rupture_kind == "over_directive"
    treatment_expression_ok = (
        treatment.repair_alpha_rationale_present
        and treatment.repair_first_intent_present
        and treatment.repair_alpha_phrase_present
    )
    control_expression_ok = (
        not control.repair_alpha_rationale_present
        and not control.repair_first_intent_present
        and not control.repair_alpha_phrase_present
    )
    treatment_observed_memory_ok = treatment.observed_repair_memory_count >= 1
    control_observed_memory_ok = control.observed_repair_memory_count == 0
    treatment_recall_ok = treatment.same_user_recall_count >= 1
    treatment_leakage_ok = treatment.cross_user_leakage_count == 0
    gate_items = (
        (
            "treatment_observes_typed_rupture",
            treatment_rupture_ok,
            f"treatment rupture_kind={treatment.rupture_kind!r}",
        ),
        (
            "control_observes_same_typed_rupture",
            control_rupture_ok,
            f"control rupture_kind={control.rupture_kind!r}",
        ),
        (
            "treatment_uses_repair_alpha_expression",
            treatment_expression_ok,
            (
                "requires repair_alpha rationale, repair-first intent, "
                "and repair-alpha phrase."
            ),
        ),
        (
            "control_does_not_use_repair_alpha_expression",
            control_expression_ok,
            "control must not expose repair_alpha expression evidence.",
        ),
        (
            "treatment_writes_observed_repair_memory",
            treatment_observed_memory_ok,
            (
                "treatment observed repair memory count="
                f"{treatment.observed_repair_memory_count}"
            ),
        ),
        (
            "control_writes_no_observed_repair_memory",
            control_observed_memory_ok,
            (
                "control observed repair memory count="
                f"{control.observed_repair_memory_count}"
            ),
        ),
        (
            "treatment_same_user_recalls_repair_memory",
            treatment_recall_ok,
            f"same-user recall count={treatment.same_user_recall_count}",
        ),
        (
            "treatment_cross_user_leakage_absent",
            treatment_leakage_ok,
            f"cross-user leakage count={treatment.cross_user_leakage_count}",
        ),
    )
    report = RepairAlphaGateReport(
        passed=all(passed for _, passed, _ in gate_items),
        treatment=treatment,
        matched_control=control,
        gate_items=gate_items,
    )
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_json(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def run_relationship_repair_alpha_gate(
    *,
    out_path: str | Path | None = None,
    scope_root_dir: str | Path | None = None,
) -> RepairAlphaGateReport:
    return asyncio.run(
        run_relationship_repair_alpha_gate_async(
            out_path=out_path,
            scope_root_dir=scope_root_dir,
        )
    )


__all__ = (
    "format_relationship_repair_alpha_report",
    "RepairAlphaArmReport",
    "RepairAlphaGateReport",
    "run_relationship_repair_alpha_gate",
    "run_relationship_repair_alpha_gate_async",
)
