"""Held-out environment variant sealing (coding-lab Packet 0).

Generates structurally different environment variants, records their
recipe plus content hashes, and DISCARDS the generated trees. The
sealed manifest lets a future transfer packet regenerate the exact
variants while guaranteeing nobody peeked at them during main-lane
development: no tree ever lands under the run directory.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import time
from dataclasses import asdict, dataclass

from lifeform_domain_coding.lab.generation import (
    ALL_INVARIANT_IDS,
    GENERATOR_VERSION,
    EnvSpec,
    generate_environment,
)
from lifeform_domain_coding.lab.tasks import generate_task_chain

_PROBE_CHAIN_SEED = 1
_PROBE_CHAIN_LENGTH = 6


@dataclass(frozen=True)
class SealedVariant:
    """Recipe + hashes of one unopened held-out environment."""

    variant_id: str
    env_seed: int
    package_name: str
    param_offset: int
    invariant_ids: tuple[str, ...]
    generator_version: str
    tree_sha256: str
    probe_chain_sha256: str
    #: Default () keeps pre-2026-08-13 sealed manifests loadable.
    convention_ids: tuple[str, ...] = ()


def _variant_spec(base_spec: EnvSpec, index: int) -> EnvSpec:
    rotated = tuple(
        invariant_id
        for offset, invariant_id in enumerate(ALL_INVARIANT_IDS)
        if offset != index % len(ALL_INVARIANT_IDS)
    )
    return EnvSpec(
        env_seed=base_spec.env_seed + 7_777_777 * (index + 1),
        package_name=f"hv{index}_app",
        param_offset=1_000 + index * 17,
        invariant_ids=rotated,
        convention_ids=base_spec.convention_ids,
    )


def seal_heldout_variants(
    *,
    base_spec: EnvSpec,
    count: int,
    manifest_path: pathlib.Path,
) -> tuple[SealedVariant, ...]:
    """Seal ``count`` variants into ``manifest_path`` (trees discarded)."""

    if count < 1:
        raise ValueError("count must be >= 1")
    sealed: list[SealedVariant] = []
    for index in range(count):
        spec = _variant_spec(base_spec, index)
        with tempfile.TemporaryDirectory(prefix="coding-lab-heldout-") as scratch:
            environment = generate_environment(spec, pathlib.Path(scratch) / "tree")
            tree_hash = environment.tree_hash
        chain = generate_task_chain(spec, chain_seed=_PROBE_CHAIN_SEED, length=_PROBE_CHAIN_LENGTH)
        chain_blob = json.dumps(
            [
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "description": task.description,
                }
                for task in chain
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        sealed.append(
            SealedVariant(
                variant_id=f"heldout-{index}",
                env_seed=spec.env_seed,
                package_name=spec.package_name,
                param_offset=spec.param_offset,
                invariant_ids=spec.invariant_ids,
                generator_version=GENERATOR_VERSION,
                tree_sha256=tree_hash,
                probe_chain_sha256=hashlib.sha256(chain_blob.encode("utf-8")).hexdigest(),
                convention_ids=spec.convention_ids,
            )
        )
    manifest_path = pathlib.Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "sealed_at_unix": int(time.time()),
                "discipline": (
                    "Trees were generated in a temporary directory, hashed and discarded. "
                    "Do not regenerate these variants until the transfer packet opens them."
                ),
                "probe_chain_seed": _PROBE_CHAIN_SEED,
                "probe_chain_length": _PROBE_CHAIN_LENGTH,
                "variants": [asdict(variant) for variant in sealed],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(sealed)


def verify_sealed_variant(variant: SealedVariant) -> bool:
    """Regenerate ``variant`` in scratch space and compare hashes.

    Intended for the future transfer packet's opening ceremony (and for
    unit tests of the sealing mechanism itself).
    """

    spec = EnvSpec(
        env_seed=variant.env_seed,
        package_name=variant.package_name,
        param_offset=variant.param_offset,
        invariant_ids=tuple(variant.invariant_ids),
        convention_ids=tuple(variant.convention_ids),
    )
    with tempfile.TemporaryDirectory(prefix="coding-lab-heldout-verify-") as scratch:
        environment = generate_environment(spec, pathlib.Path(scratch) / "tree")
        return environment.tree_hash == variant.tree_sha256


__all__ = ["SealedVariant", "seal_heldout_variants", "verify_sealed_variant"]
