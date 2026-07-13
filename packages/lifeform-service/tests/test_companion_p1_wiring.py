from __future__ import annotations

from types import SimpleNamespace

import lifeform_domain_emogpt
from lifeform_service import verticals


class _FakeLifeform:
    def __init__(self) -> None:
        self.affordance_registry_ready = False

    def ensure_affordance_registry(self) -> tuple[object, object]:
        self.affordance_registry_ready = True
        return object(), object()


def test_companion_and_cold_share_full_runtime_wiring(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return _FakeLifeform()

    semantic_runtime = object()
    monkeypatch.setattr(lifeform_domain_emogpt, "build_companion_lifeform", fake_build)
    monkeypatch.setattr(
        verticals,
        "_build_llm_semantic_runtime_from_runtime",
        lambda runtime: semantic_runtime,
    )

    runtime = SimpleNamespace()
    companion = verticals._try_companion()
    cold = verticals._try_uncalibrated_companion()
    assert companion is not None
    assert cold is not None

    trained_lifeform = companion.factory(runtime)
    cold_lifeform = cold.factory(runtime)

    assert trained_lifeform.affordance_registry_ready
    assert cold_lifeform.affordance_registry_ready
    assert calls[0]["semantic_proposal_runtime"] is semantic_runtime
    assert calls[1]["semantic_proposal_runtime"] is semantic_runtime
    assert "use_temporal_bootstrap" not in calls[0]
    assert "use_regime_bootstrap" not in calls[0]
    assert calls[1]["use_temporal_bootstrap"] is False
    assert calls[1]["use_regime_bootstrap"] is False


def test_companion_bootstrap_pair_is_required(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(lifeform_domain_emogpt, "bootstraps_dir", lambda: tmp_path)

    try:
        lifeform_domain_emogpt.require_companion_bootstraps()
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing companion bootstrap pair did not fail")

    assert "companion-temporal.snap" in message
    assert "companion-regime.bs" in message
