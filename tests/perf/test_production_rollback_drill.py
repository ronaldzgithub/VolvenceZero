"""F-D: production-grade rollback drill (debt #50).

The contract-level ``test_learned_baseline_rollback_drill.py`` (Wave
E3) validates rollback in synthetic / in-memory; this file extends
the drill to **real production**:

1. Real Qwen 1.5B substrate
2. ``PersonaLoRAPool.activate(figure_id, runtime=runtime)`` over the
   ``TransformersOpenWeightResidualRuntime`` (Wave D)
3. 10 turn generation → trigger rollback → 10 more turn → assert
   logits L1 distance to baseline < 1e-6 (byte-identical revert)
4. Cross-restart audit chain: bundle pickle reload → activate →
   rollback → audit log unchanged

Default-skipped (``@pytest.mark.hf @pytest.mark.perf`` double mark).
Run::

    pytest tests/perf/test_production_rollback_drill.py -m "perf and hf"

Debts: known-debts.md #50 (production rollback drill) + #20 closure
(LoRA hot-swap) + #41 (real Qwen PEFT bake)
Packets:
* docs/moving forward/cross-cutting-foundation-packet.md §2.4
* docs/specs/rollback-drill-cadence.md
"""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.perf, pytest.mark.hf]


LOGITS_L1_DISTANCE_TOLERANCE: float = 1e-6


def test_figure_lora_activation_rollback_byte_identical(
    asyncio_harness,  # noqa: ANN001
) -> None:
    """SHADOW scaffold."""

    pytest.skip(
        "F-D SHADOW scaffold: real Qwen 1.5B + LoRA activate → 10 turn "
        "→ rollback → byte-identical revert lands as F-D ACTIVE "
        "(Phase B W2-W3). Tolerance: logits L1 distance < "
        f"{LOGITS_L1_DISTANCE_TOLERANCE:.0e}. "
        "See docs/specs/rollback-drill-cadence.md."
    )


def test_cross_restart_audit_chain_unchanged() -> None:
    """SHADOW scaffold: bundle pickle reload → activate → rollback.

    Audit chain in ``data/figure_audit/<figure_id>/*.json`` must
    remain append-only across process restarts; rollback writes a
    fresh audit row, never mutates existing ones.
    """

    pytest.skip(
        "F-D SHADOW scaffold: cross-restart audit chain integrity "
        "test lands with F-D ACTIVE."
    )


def test_substrate_upgrade_rollback_drill_n_to_n_minus_1() -> None:
    """SHADOW scaffold: substrate upgrade N+1 → fall back to N.

    Depends on F-C ``SubstrateFingerprint``: drill validates that
    bundles baked against N can be re-loaded after rolling substrate
    weights from N+1 back to N. If no longer compatible (different
    weights_sha256), drill fails loudly with a re-bake suggestion.
    """

    pytest.skip(
        "F-D SHADOW scaffold: substrate upgrade rollback drill lands "
        "with F-D ACTIVE; depends on F-C SubstrateFingerprint."
    )
