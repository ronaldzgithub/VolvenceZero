from __future__ import annotations

import pytest

from volvence_zero.agent.capacity_ladder import (
    CapacityLadderArm,
    build_capacity_ladder_manifest,
)


def test_default_capacity_ladder_manifest_matches_plan_axes() -> None:
    manifest = build_capacity_ladder_manifest()
    # 4 n_z * 3 PE capacity * 3 COCOA hidden * 4 backend combos
    # * 2 trace lengths * 3 substrates * 3 seeds
    assert manifest.arm_count == 4 * 3 * 3 * 4 * 2 * 3 * 3
    assert manifest.schema_version == "capacity-ladder-manifest.v1"
    first = manifest.arms[0]
    assert first.temporal_latent_dim == 3
    assert first.pe_critic_capacity == 1
    assert first.cocoa_hidden == 8
    assert first.backend_combo == "runtime-only"
    assert first.trace_turns == 500
    assert first.substrate_label == "qwen-0.5b-screen"
    assert first.seed == 0


def test_capacity_ladder_can_build_small_p0_manifest() -> None:
    manifest = build_capacity_ladder_manifest(
        n_z_values=(3, 16),
        pe_critic_capacities=(1,),
        cocoa_hidden_values=(8,),
        backend_combos=("runtime-only",),
        trace_turns=(500,),
        substrates=("synthetic-p0",),
        seeds=(0,),
    )
    assert manifest.arm_count == 2
    assert {arm.temporal_latent_dim for arm in manifest.arms} == {3, 16}


def test_capacity_ladder_arm_validates_closed_axes() -> None:
    with pytest.raises(ValueError, match="pe_critic_capacity"):
        CapacityLadderArm(
            arm_id="bad",
            temporal_latent_dim=16,
            pe_critic_capacity=3,
            cocoa_hidden=8,
            backend_combo="runtime-only",
            trace_turns=500,
            substrate_label="synthetic",
            seed=0,
        )
