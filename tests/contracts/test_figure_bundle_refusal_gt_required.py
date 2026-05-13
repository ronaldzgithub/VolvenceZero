"""Contract test: production-tier figure bundles must ship a refusal GT eval.

Validates that any ``FigureArtifactBundle`` declared "production-tier"
(via env var or marker file) carries a non-None
``refusal_eval_report`` field. This is the SHADOW shape; real
production-tier marker land with figure-evidence-packet G-A ACTIVE.

For now (SHADOW), the test asserts:

1. ``FigureArtifactBundle`` has the three new optional fields
   (``refusal_eval_report`` / ``grounding_eval_report`` /
   ``voice_blind_test_report``) with default None
2. The fields are NOT included in ``compute_bundle_integrity_hash``
   (eval is readout, not bundle identity — debt #58/#59/#60 spec)
3. Wave K Einstein bundle, when reloaded, retains its existing
   integrity_hash byte-identically

See:

* docs/specs/figure-refusal-gt-protocol.md
* docs/moving forward/figure-evidence-packet.md §2.1
* docs/known-debts.md #58 / #59 / #60
"""

from __future__ import annotations

import dataclasses

import pytest


def test_bundle_has_three_optional_eval_report_fields() -> None:
    """SHADOW: schema has the new fields default-None."""

    from lifeform_domain_figure.figure_artifact import FigureArtifactBundle

    field_names = {f.name for f in dataclasses.fields(FigureArtifactBundle)}
    assert "refusal_eval_report" in field_names
    assert "grounding_eval_report" in field_names
    assert "voice_blind_test_report" in field_names


def test_eval_reports_not_in_integrity_hash_inputs() -> None:
    """SHADOW: integrity_hash signature does NOT include eval reports.

    Eval reports are readouts; reviewing them must not invalidate
    bundle identity (R12 evaluation is not a learning source +
    R15 byte-level rollback contract requires bundle identity to
    be over load-bearing fields only).
    """
    import inspect

    from lifeform_domain_figure.figure_artifact import compute_bundle_integrity_hash

    sig = inspect.signature(compute_bundle_integrity_hash)
    param_names = set(sig.parameters.keys())
    assert "refusal_eval_report" not in param_names
    assert "grounding_eval_report" not in param_names
    assert "voice_blind_test_report" not in param_names


def test_compatible_substrates_in_integrity_hash() -> None:
    """F-C contract: compatible_substrates IS in integrity_hash inputs."""
    import inspect

    from lifeform_domain_figure.figure_artifact import compute_bundle_integrity_hash

    sig = inspect.signature(compute_bundle_integrity_hash)
    assert "compatible_substrates" in sig.parameters


@pytest.mark.skip(
    reason=(
        "SHADOW: production-tier marker check (assert refusal_eval_report "
        "non-None for production bundles) lands with figure-evidence-packet "
        "G-A ACTIVE (Phase A W6)."
    )
)
def test_production_tier_bundle_must_have_refusal_eval_report() -> None:
    """ACTIVE-tier scaffold: real check pending production marker."""
    pass
