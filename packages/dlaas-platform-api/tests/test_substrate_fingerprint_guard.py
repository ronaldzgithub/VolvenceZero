"""Tests for the adopt-time substrate fingerprint compatibility guard."""

from __future__ import annotations

from dlaas_platform_api.control_plane import _bundle_substrate_compatible
from volvence_zero.substrate import LEGACY_FINGERPRINT, SubstrateFingerprint


class _Bundle:
    def __init__(self, compatible) -> None:
        self.compatible_substrates = tuple(compatible)


class _Runtime:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


def test_empty_compatible_is_unconstrained() -> None:
    assert _bundle_substrate_compatible(
        bundle=_Bundle(()), runtime=_Runtime("Qwen/Qwen2.5-1.5B-Instruct")
    )


def test_synthetic_runtime_skips_check() -> None:
    fp = SubstrateFingerprint(model_id="Qwen/X", version="v1", weights_sha256="a" * 64)
    assert _bundle_substrate_compatible(bundle=_Bundle((fp,)), runtime=None)


def test_matching_model_id_is_compatible() -> None:
    fp = SubstrateFingerprint(
        model_id="Qwen/Qwen2.5-1.5B-Instruct", version="v2.5", weights_sha256="b" * 64
    )
    assert _bundle_substrate_compatible(
        bundle=_Bundle((fp,)),
        runtime=_Runtime("Qwen/Qwen2.5-1.5B-Instruct"),
    )


def test_mismatched_model_id_is_incompatible() -> None:
    fp = SubstrateFingerprint(
        model_id="Qwen/Qwen2.5-7B-Instruct", version="v2.5", weights_sha256="c" * 64
    )
    assert not _bundle_substrate_compatible(
        bundle=_Bundle((fp,)),
        runtime=_Runtime("Qwen/Qwen2.5-1.5B-Instruct"),
    )


def test_legacy_fingerprint_matches_anything() -> None:
    assert _bundle_substrate_compatible(
        bundle=_Bundle((LEGACY_FINGERPRINT,)),
        runtime=_Runtime("any/model"),
    )
