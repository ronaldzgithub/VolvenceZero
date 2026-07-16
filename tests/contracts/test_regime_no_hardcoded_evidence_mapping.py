"""C1 (#79/#80) SSOT contracts for the regime owner.

1. ``regime/identity.py`` must not hardcode regime-id string literals:
   evidence->prior and consolidation->prior mappings live in the
   template table (``metacontroller_evidence_affinity`` /
   ``consolidation_affinity``), so adding a regime only needs a
   template row.
2. ``score_regimes`` must not grow new evaluation-metric dependencies
   beyond the frozen allowlist (#80: evaluation is a readout, PE is the
   primary drive; removing the existing consumption entirely is gated
   on longitudinal evidence, but expansion is a regression today).
"""

from __future__ import annotations

import ast
from pathlib import Path


_REGIME_ROOT = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "vz-cognition"
    / "src"
    / "volvence_zero"
    / "regime"
)

_REGIME_IDS = {
    "casual_social",
    "acquaintance_building",
    "emotional_support",
    "guided_exploration",
    "problem_solving",
    "repair_and_deescalation",
}


def test_identity_module_has_no_hardcoded_regime_id_literals() -> None:
    """#79: the evidence/consolidation mappings must come from the
    template table, never from regime-id literals in identity.py."""

    text = (_REGIME_ROOT / "identity.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in _REGIME_IDS
        ):
            hits.append((node.lineno, node.value))
    assert not hits, (
        "regime-id string literals found in regime/identity.py "
        "(should live in templates.py affinity tables):\n"
        + "\n".join(f"  L{lineno}: {value!r}" for lineno, value in hits)
    )


# Frozen allowlist of evaluation metric names score_regimes may consume
# (#80). Growing this list is an SSOT regression: new regime-scoring
# signal must come from PE / dual_track / memory, or be justified in
# docs/specs/cognitive-regime.md before landing here.
_ALLOWED_EVALUATION_METRICS = {
    "info_integration",
    "task_pressure",
    "repair_pressure",
    "social_pressure",
    "decision_delegation_pressure",
    "semantic_surface_active",
    "warmth",
    "support_presence",
    "cross_track_stability",
    "support_before_decision_pressure",
}


def test_score_regimes_evaluation_metric_consumption_is_frozen() -> None:
    text = (_REGIME_ROOT / "scoring.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    consumed: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_metric"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    consumed.add(arg.value)
    unexpected = consumed - _ALLOWED_EVALUATION_METRICS
    assert not unexpected, (
        f"score_regimes consumes new evaluation metrics {sorted(unexpected)}; "
        "#80 requires new scoring signal to come from PE / dual_track / "
        "memory (evaluation stays a readout)."
    )


def test_every_template_signal_maps_to_known_regime() -> None:
    from volvence_zero.regime.templates import REGIME_TEMPLATES

    known_signals = {
        "self_axis",
        "world_axis",
        "shared_axis",
        "stabilize_axis",
        "sparse_switch",
        "posterior_guard",
        "replacement",
        "rollback_guard",
    }
    known_updates = {
        "increase_self_track_priority",
        "increase_world_track_priority",
    }
    for template in REGIME_TEMPLATES:
        for signal, delta in template.metacontroller_evidence_affinity:
            assert signal in known_signals, (template.regime_id, signal)
            assert -0.25 <= delta <= 0.25
        for update, multiplier in template.consolidation_affinity:
            assert update in known_updates, (template.regime_id, update)
            assert 0.0 < multiplier <= 2.0
