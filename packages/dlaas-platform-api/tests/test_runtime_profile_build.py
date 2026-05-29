"""Tests for profile-driven runtime build + backend detection (P2.4)."""

from __future__ import annotations

import pytest

from dlaas_platform_api.substrate_profiles import (
    BACKEND_VLLM,
    MODE_SHARED_FROZEN,
    MODE_SYNTHETIC,
    SubstrateProfile,
    build_runtime_for_profile,
    running_substrate_backend,
)


def test_synthetic_profile_builds_none() -> None:
    profile = SubstrateProfile(
        substrate_profile_id="synthetic-dev", mode=MODE_SYNTHETIC
    )
    assert build_runtime_for_profile(profile) is None


def test_shared_frozen_without_model_id_fails_loud() -> None:
    profile = SubstrateProfile(
        substrate_profile_id="shared-frozen", mode=MODE_SHARED_FROZEN
    )
    with pytest.raises(ValueError, match="model_id"):
        build_runtime_for_profile(profile)


def test_running_backend_none_and_synthetic() -> None:
    assert running_substrate_backend(None) == ""

    class _Synthetic:
        pass

    # A non-substrate object is treated as transformers only if it is a
    # real runtime; arbitrary objects fall through to transformers, so
    # we assert the None / explicit-synthetic contract here.
    from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime

    assert running_substrate_backend(SyntheticOpenWeightResidualRuntime()) == ""


def test_vllm_profile_carries_backend() -> None:
    profile = SubstrateProfile(
        substrate_profile_id="vllm",
        mode=MODE_SHARED_FROZEN,
        runtime_backend=BACKEND_VLLM,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
    )
    assert profile.runtime_backend == BACKEND_VLLM
    assert profile.to_json()["runtime_backend"] == BACKEND_VLLM
