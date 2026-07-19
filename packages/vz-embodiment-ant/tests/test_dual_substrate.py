"""Workstream G1 test: one kernel, two bodies."""

from __future__ import annotations

from volvence_ant.viz import run_dual_substrate_demo


async def test_same_kernel_drives_both_bodies() -> None:
    report = await run_dual_substrate_demo(temporal_latent_dim=4, seed=0)
    assert report.same_kernel_class
    assert report.same_code_dim
    assert report.ant.code_dim == 4
    assert report.companion.code_dim == 4
    # the two bodies emit different output modalities from the same controller
    assert report.ant.output_kind == "motor-command"
    assert report.companion.output_kind == "text"
    assert report.ant.substrate_model_id != report.companion.substrate_model_id
