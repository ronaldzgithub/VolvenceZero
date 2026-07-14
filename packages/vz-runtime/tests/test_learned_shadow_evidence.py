"""CP-LSS-01 acceptance: unified learned-shadow evidence artifact.

Proves that one session under the frozen learned-shadow profile produces
SHADOW evidence from all four torch autograd owners (temporal SSL, temporal
runtime forward, internal RL, CMS band) with zero write-back, and that the
collector fails loudly when the profile is not actually SHADOW.
"""

from __future__ import annotations

import pytest

from volvence_zero.agent.learned_shadow_evidence import (
    LEARNED_SHADOW_EVIDENCE_SCHEMA_VERSION,
    LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
    LearnedShadowEvidenceError,
    build_learned_shadow_rollout_config,
    collect_learned_shadow_evidence,
)
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel
from volvence_zero.tensor_backend import is_torch_available

torch_only = pytest.mark.skipif(not is_torch_available(), reason="torch not installed")

_SMOKE_TURNS = (
    "Walk me through the harbor plan for tomorrow.",
    "The tide tables changed; adjust the schedule.",
    "Now summarize what we committed to.",
    "One more check: anything still open?",
)


def _learned_shadow_runner() -> AgentSessionRunner:
    return AgentSessionRunner(
        config=build_learned_shadow_rollout_config(),
        temporal_latent_dim=LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
        rare_heavy_enabled=False,
    )


def test_learned_shadow_profile_freezes_all_four_backends_to_shadow() -> None:
    config = build_learned_shadow_rollout_config()
    assert config.temporal_ssl_backend is WiringLevel.SHADOW
    assert config.temporal_runtime_backend is WiringLevel.SHADOW
    assert config.internal_rl_backend is WiringLevel.SHADOW
    assert config.cms_torch_backend is WiringLevel.SHADOW
    # Production defaults stay DISABLED; the profile is opt-in.
    defaults = FinalRolloutConfig()
    assert defaults.temporal_ssl_backend is WiringLevel.DISABLED
    assert defaults.temporal_runtime_backend is WiringLevel.DISABLED
    assert defaults.internal_rl_backend is WiringLevel.DISABLED
    assert defaults.cms_torch_backend is WiringLevel.DISABLED


@torch_only
async def test_learned_shadow_evidence_covers_all_four_owners_without_writeback() -> None:
    runner = _learned_shadow_runner()
    for text in _SMOKE_TURNS:
        await runner.run_turn(text)

    payload = collect_learned_shadow_evidence(runner)

    assert payload["schema_version"] == LEARNED_SHADOW_EVIDENCE_SCHEMA_VERSION
    assert payload["temporal_latent_dim"] == LEARNED_SHADOW_TEMPORAL_LATENT_DIM
    assert all(level == "shadow" for level in payload["backend_wiring"].values())

    for track in ("world", "self"):
        runtime = payload["temporal_runtime"][track]
        assert runtime["steps_compared"] >= 1
        assert runtime["torch_available"] is True
        assert runtime["within_tolerance"] is True, runtime["description"]
        # CP-06 (GAP-09): behaviour-level comparison dimensions.
        assert runtime["switch_decision_match"] is True, runtime["description"]
        assert runtime["family_selection_match"] is True, runtime["description"]

    ssl = payload["temporal_ssl"]
    assert ssl["torch_backend"] == "shadow"
    assert ssl["torch_wrote_back"] is False
    assert ssl["trained_steps"] >= 1
    # CP-05 (GAP-09): SHADOW exports a candidate checkpoint (not written to
    # the store) plus a forward-parity report on the untouched live params.
    candidate = ssl["candidate_checkpoint"]
    assert candidate["wrote_back"] is False
    assert candidate["candidate_fingerprint"]
    parity = ssl["forward_parity"]
    assert parity["within_tolerance"] is True, parity["description"]

    # CP-07 (GAP-09): reward composition purity readout.
    composition = payload["internal_rl"]["reward_composition"]
    assert 0.0 <= composition["pe_derived_abs_fraction"] <= 1.0
    assert composition["component_names"]

    rl = payload["internal_rl"]
    tracks = (
        (rl["task"], rl["relationship"]) if rl["kind"] == "dual-track" else (rl["task"],)
    )
    for track_payload in tracks:
        assert track_payload["torch_backend"] == "shadow"
        assert track_payload["torch_wrote_back"] is False
        assert track_payload["transition_count"] >= 1

    cms = payload["cms"]
    assert cms["backend"] == "shadow"
    assert cms["wrote_back"] is False
    # CP-08 (GAP-09): live-weight forward parity + anti-forgetting readouts
    # ride alongside the torch step scalars in one artifact.
    assert cms["forward_parity_within_tolerance"] is True
    assert 0.0 <= cms["new_knowledge_absorption"] <= 1.0
    assert 0.0 <= cms["old_knowledge_retention"] <= 1.0


async def test_collector_fails_loudly_on_default_disabled_profile() -> None:
    runner = AgentSessionRunner(
        temporal_latent_dim=LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
        rare_heavy_enabled=False,
    )
    await runner.run_turn("A single turn under production defaults.")
    with pytest.raises(LearnedShadowEvidenceError, match="temporal_ssl_backend"):
        collect_learned_shadow_evidence(runner)


async def test_cp02_ndim_disabled_baseline_stays_pure_and_stable() -> None:
    """CP-02 exit: n_z=16 with all backends DISABLED shows no torch involvement."""

    runner = AgentSessionRunner(
        temporal_latent_dim=LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
        rare_heavy_enabled=False,
    )
    codes = []
    for text in _SMOKE_TURNS[:2]:
        result = await runner.run_turn(text)
        state = result.active_snapshots["temporal_abstraction"].value.controller_state
        assert state.code_dim == LEARNED_SHADOW_TEMPORAL_LATENT_DIM
        assert all(0.0 <= value <= 1.0 for value in state.code)
        codes.append(state.code)
    assert runner.world_temporal_policy.latest_runtime_shadow_report is None
    assert runner.self_temporal_policy.latest_runtime_shadow_report is None
