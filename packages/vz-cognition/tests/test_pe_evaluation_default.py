from __future__ import annotations

import pytest

from volvence_zero.prediction.error import pe_evaluation_decoupled_active


def test_evaluation_decoupling_defaults_active(monkeypatch) -> None:
    monkeypatch.delenv("VZ_PE_EVALUATION_DECOUPLED", raising=False)
    assert pe_evaluation_decoupled_active()


def test_evaluation_decoupling_has_explicit_shadow_rollback(monkeypatch) -> None:
    monkeypatch.setenv("VZ_PE_EVALUATION_DECOUPLED", "SHADOW")
    assert not pe_evaluation_decoupled_active()


def test_evaluation_decoupling_rejects_unknown_gate_value(monkeypatch) -> None:
    monkeypatch.setenv("VZ_PE_EVALUATION_DECOUPLED", "maybe")
    with pytest.raises(ValueError, match="must be ACTIVE or SHADOW"):
        pe_evaluation_decoupled_active()
