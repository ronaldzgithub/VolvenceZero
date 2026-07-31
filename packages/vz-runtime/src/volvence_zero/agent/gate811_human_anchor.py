"""Preregister the Gate 8/11 blinded longitudinal human anchor.

This module owns no runtime state.  It freezes an evaluation-only protocol
whose source arms remain owned by their original Gate 8 and Gate 11 evidence
bundles.  Human ratings are readouts and must never enter reward, credit, or
online adaptation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


GATE811_HUMAN_ANCHOR_SCHEMA_VERSION = "gate811-human-anchor-prereg.v1"
GATE811_HUMAN_ANCHOR_CODE_PATHS = (
    "packages/vz-runtime/src/volvence_zero/agent/gate811_human_anchor.py",
    "scripts/preregister_gate811_human_anchor.py",
)
GATE811_HUMAN_ANCHOR_CAPTURE_SEEDS = (1401, 1413, 1427)
GATE811_HUMAN_ANCHOR_FORMAL_SEEDS = (1501, 1513, 1527)
GATE811_HUMAN_ANCHOR_PILOT_SELECTION_SEED = 1451
GATE811_HUMAN_ANCHOR_BLINDING_SEED = 1459


@dataclass(frozen=True)
class GateHumanAnchorSourceBinding:
    gate_id: int
    artifact_root: str
    manifest_path: str
    manifest_sha256: str
    schema_version: str
    experimental_arm: str
    control_arm: str

    def __post_init__(self) -> None:
        if self.gate_id not in (8, 11):
            raise ValueError("human-anchor source gate must be 8 or 11")
        if len(self.manifest_sha256) != 64:
            raise ValueError("source manifest_sha256 must be a SHA-256 digest")
        if not self.experimental_arm or not self.control_arm:
            raise ValueError("source comparison arms must be non-empty")
        if self.experimental_arm == self.control_arm:
            raise ValueError("source comparison arms must differ")


@dataclass(frozen=True)
class GateHumanAnchorContrast:
    contrast_id: str
    gate_id: int
    experimental_arm: str
    control_arm: str
    primary_hypothesis: str

    def __post_init__(self) -> None:
        if not self.contrast_id.strip():
            raise ValueError("contrast_id must be non-empty")
        if self.gate_id not in (8, 11):
            raise ValueError("contrast gate must be 8 or 11")
        if self.experimental_arm == self.control_arm:
            raise ValueError("contrast arms must differ")
        if not self.primary_hypothesis.strip():
            raise ValueError("primary_hypothesis must be non-empty")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source_binding(
    *,
    repo_root: Path,
    gate_id: int,
    artifact_root: str,
    expected_schema: str,
    experimental_arm: str,
    control_arm: str,
) -> GateHumanAnchorSourceBinding:
    manifest_path = Path(artifact_root) / "manifest.yaml"
    absolute_manifest = repo_root / manifest_path
    payload = json.loads(absolute_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"Gate {gate_id} source schema drift")
    arms = payload.get("arm_schedule")
    if not isinstance(arms, list):
        raise ValueError(f"Gate {gate_id} source lacks arm_schedule")
    if experimental_arm not in arms or control_arm not in arms:
        raise ValueError(f"Gate {gate_id} source lacks frozen comparison arms")
    return GateHumanAnchorSourceBinding(
        gate_id=gate_id,
        artifact_root=artifact_root,
        manifest_path=str(manifest_path),
        manifest_sha256=_sha256_file(absolute_manifest),
        schema_version=expected_schema,
        experimental_arm=experimental_arm,
        control_arm=control_arm,
    )


def _code_manifest(repo_root: Path) -> dict[str, str]:
    return {
        relative: _sha256_file(repo_root / relative)
        for relative in GATE811_HUMAN_ANCHOR_CODE_PATHS
    }


def build_gate811_human_anchor_preregistration(
    *,
    repo_root: str | Path,
    created_at_unix_ms: int,
) -> dict[str, object]:
    """Build the frozen Gate 8/11 pilot and formal human-rating protocol."""

    if isinstance(created_at_unix_ms, bool) or created_at_unix_ms <= 0:
        raise ValueError("created_at_unix_ms must be positive")
    root = Path(repo_root)
    sources = (
        _load_source_binding(
            repo_root=root,
            gate_id=11,
            artifact_root=(
                "artifacts/gate11_per_user_continuity_v2_20260730"
            ),
            expected_schema="gate11-per-user-continuity.v2",
            experimental_arm="correct-user-state",
            control_arm="stateless",
        ),
        _load_source_binding(
            repo_root=root,
            gate_id=8,
            artifact_root="artifacts/gate8_wake_sleep_longitudinal_20260730",
            expected_schema="gate8-wake-sleep-longitudinal.v1",
            experimental_arm="sleep-consolidation",
            control_arm="no-sleep",
        ),
    )
    contrasts = (
        GateHumanAnchorContrast(
            contrast_id="gate11-correct-state-vs-stateless",
            gate_id=11,
            experimental_arm="correct-user-state",
            control_arm="stateless",
            primary_hypothesis=(
                "Correct-user-state longitudinal transcripts are preferred "
                "over matched stateless transcripts by blinded humans."
            ),
        ),
        GateHumanAnchorContrast(
            contrast_id="gate8-sleep-vs-no-sleep",
            gate_id=8,
            experimental_arm="sleep-consolidation",
            control_arm="no-sleep",
            primary_hypothesis=(
                "Sleep-consolidation longitudinal transcripts are preferred "
                "over matched no-sleep transcripts by blinded humans."
            ),
        ),
    )
    code_manifest = _code_manifest(root)
    code_tree_sha256 = hashlib.sha256(
        _canonical_bytes(code_manifest)
    ).hexdigest()
    return {
        "schema_version": GATE811_HUMAN_ANCHOR_SCHEMA_VERSION,
        "created_at_unix_ms": created_at_unix_ms,
        "code_manifest": code_manifest,
        "code_tree_sha256": code_tree_sha256,
        "source_bindings": [asdict(source) for source in sources],
        "contrasts": [asdict(contrast) for contrast in contrasts],
        "capture": {
            "capture_seeds": list(GATE811_HUMAN_ANCHOR_CAPTURE_SEEDS),
            "formal_seeds": list(GATE811_HUMAN_ANCHOR_FORMAL_SEEDS),
            "matched_variables": [
                "source_lineage",
                "persona",
                "user_turns",
                "event_arc",
                "turn_budget",
                "generation_seed",
                "model_and_adapter_fingerprint",
            ],
            "only_manipulated_variable": "comparison_arm",
            "session_count_per_transcript": 3,
            "total_turns_per_transcript": 30,
            "callback_opportunity_required": True,
            "emotional_event_required": True,
            "boundary_event_required": True,
            "fresh_transcript_capture_required": True,
            "historical_owner_metrics_are_not_human_ground_truth": True,
        },
        "blinding": {
            "pilot_selection_seed": (
                GATE811_HUMAN_ANCHOR_PILOT_SELECTION_SEED
            ),
            "orientation_and_order_seed": GATE811_HUMAN_ANCHOR_BLINDING_SEED,
            "pairwise_same_source_required": True,
            "internal_key_separate": True,
            "rater_visible_fields": [
                "pair_id",
                "transcript_a",
                "transcript_b",
                "rating_dimensions",
                "forced_preference",
            ],
            "hidden_fields": [
                "gate_id",
                "contrast_id",
                "arm_label",
                "system_identity",
                "model_identifier",
                "source_artifact_path",
                "owner_metric",
                "expected_winner",
            ],
            "project_members_may_rate": False,
            "llm_judge_is_human_anchor": False,
        },
        "rating": {
            "scale_min": 1,
            "scale_max": 7,
            "dimensions": [
                "rememberedness",
                "relationship_continuity",
                "boundary_respect",
            ],
            "forced_preference_prompt": (
                "Which transcript feels more like an ongoing relationship "
                "with someone who knows this person?"
            ),
            "abstention_allowed_only_for_malformed_pair": True,
        },
        "pilot": {
            "pairs_per_contrast": 24,
            "ratings_per_pair": 3,
            "minimum_unique_raters": 6,
            "pilot_rows_excluded_from_formal": True,
            "minimum_krippendorff_alpha_ordinal": 0.6,
            "failure_action": (
                "freeze pilot as non-claim evidence; revise rubric in a new "
                "schema before any formal capture"
            ),
        },
        "formal": {
            "target_power": 0.8,
            "familywise_alpha": 0.05,
            "multiplicity": "holm-two-contrasts",
            "minimum_pairs_per_contrast": 60,
            "maximum_pairs_per_contrast": 300,
            "ratings_per_pair": 3,
            "sample_size_rule": (
                "use pilot variance and preference rate with the frozen "
                "minimum effects; round up before formal capture"
            ),
            "minimum_preference_win_rate": 0.6,
            "preference_ci_rule": "two-sided-95%-wilson-lower>0.5",
            "minimum_composite_likert_delta": 0.35,
            "composite_ci_rule": (
                "rater-cluster-bootstrap-95%-lower>0"
            ),
            "boundary_noninferiority_margin": -0.25,
            "boundary_ci_rule": (
                "rater-cluster-bootstrap-95%-lower>=-0.25"
            ),
            "minimum_krippendorff_alpha_ordinal": 0.6,
            "gate_verdicts_are_independent": True,
            "both_gates_required_for_debt_51_closure": True,
        },
        "authorization": {
            "pilot_may_produce_human_anchored_claim": False,
            "formal_requires_frozen_power_analysis": True,
            "rating_is_evaluation_readout_only": True,
            "rating_may_enter_reward_or_credit": False,
            "production_promotion_authorized": False,
            "human_recruitment_required": True,
        },
    }


def validate_gate811_human_anchor_preregistration(
    payload: Mapping[str, object],
    *,
    repo_root: str | Path,
) -> None:
    """Fail loudly if source, code, arms, or frozen decisions drift."""

    created_at = payload.get("created_at_unix_ms")
    if isinstance(created_at, bool) or not isinstance(created_at, int):
        raise ValueError("human-anchor preregistration lacks timestamp")
    expected = build_gate811_human_anchor_preregistration(
        repo_root=repo_root,
        created_at_unix_ms=created_at,
    )
    if dict(payload) != expected:
        raise ValueError("Gate 8/11 human-anchor preregistration drift")


def write_gate811_human_anchor_preregistration(
    *,
    payload: Mapping[str, object],
    output_path: str | Path,
) -> dict[str, object]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_bytes(dict(payload))
    output.write_bytes(serialized)
    manifest = {
        "schema_version": GATE811_HUMAN_ANCHOR_SCHEMA_VERSION,
        "preregistration_path": str(output),
        "preregistration_sha256": hashlib.sha256(serialized).hexdigest(),
        "code_tree_sha256": payload["code_tree_sha256"],
        "production_promotion_authorized": False,
    }
    manifest_path = output.with_name(f"{output.stem}.manifest.json")
    manifest_path.write_bytes(_canonical_bytes(manifest))
    return manifest


__all__ = [
    "GATE811_HUMAN_ANCHOR_BLINDING_SEED",
    "GATE811_HUMAN_ANCHOR_CAPTURE_SEEDS",
    "GATE811_HUMAN_ANCHOR_CODE_PATHS",
    "GATE811_HUMAN_ANCHOR_FORMAL_SEEDS",
    "GATE811_HUMAN_ANCHOR_PILOT_SELECTION_SEED",
    "GATE811_HUMAN_ANCHOR_SCHEMA_VERSION",
    "GateHumanAnchorContrast",
    "GateHumanAnchorSourceBinding",
    "build_gate811_human_anchor_preregistration",
    "validate_gate811_human_anchor_preregistration",
    "write_gate811_human_anchor_preregistration",
]
