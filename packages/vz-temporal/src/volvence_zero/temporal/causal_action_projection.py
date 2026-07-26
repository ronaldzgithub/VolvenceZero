"""Generic actuator-subspace projection for the causal latent action head."""

from __future__ import annotations

from collections.abc import Sequence


def normalize_causal_action_head_input_mirror(
    permutation: tuple[int, ...] | None,
    signs: tuple[int, ...] | None,
    *,
    n_input: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate one signed, involutive permutation over encoder inputs."""

    if permutation is None and signs is None:
        return (), ()
    if permutation is None or signs is None:
        raise ValueError(
            "causal action head input mirror requires both permutation and signs"
        )
    if len(permutation) != n_input or len(signs) != n_input:
        raise ValueError(
            "causal action head input mirror dimension mismatch: "
            f"expected={n_input}, permutation={len(permutation)}, "
            f"signs={len(signs)}"
        )
    if (
        len(set(permutation)) != n_input
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < n_input
            for index in permutation
        )
    ):
        raise ValueError(
            "causal action head input mirror permutation must contain each "
            f"index in [0, {n_input}) exactly once, got {permutation!r}"
        )
    if any(
        isinstance(sign, bool)
        or not isinstance(sign, int)
        or sign not in (-1, 1)
        for sign in signs
    ):
        raise ValueError(
            "causal action head input mirror signs must be integers in {-1, 1}, "
            f"got {signs!r}"
        )
    for index, source in enumerate(permutation):
        if (
            permutation[source] != index
            or signs[index] * signs[source] != 1
        ):
            raise ValueError(
                "causal action head input mirror must be involutive, got "
                f"index={index}, source={source}, "
                f"roundtrip={permutation[source]}, "
                f"sign_product={signs[index] * signs[source]}"
            )
    return permutation, signs


def mirror_causal_action_head_input(
    values: Sequence[float],
    *,
    permutation: tuple[int, ...],
    signs: tuple[int, ...],
) -> tuple[float, ...]:
    """Apply a validated signed permutation to one encoder input vector."""

    if len(values) != len(permutation) or len(values) != len(signs):
        raise ValueError(
            "causal action head input mirror requires aligned vectors: "
            f"values={len(values)}, permutation={len(permutation)}, "
            f"signs={len(signs)}"
        )
    return tuple(
        float(signs[index]) * float(values[source])
        for index, source in enumerate(permutation)
    )


def normalize_causal_action_head_contrast_pairs(
    pairs: tuple[tuple[int, int], ...] | None,
    *,
    n_z: int,
    effective_dims: Sequence[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Validate disjoint opponent-coded latent coordinates."""

    if pairs is None:
        return ()
    seen: set[int] = set()
    supported = (
        set(range(n_z))
        if effective_dims is None
        else set(effective_dims)
    )
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(
                "causal action head contrast pairs must contain exactly "
                f"two z indices, got {pair!r}"
            )
        left, right = pair
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int)
            or not isinstance(right, int)
            or not 0 <= left < n_z
            or not 0 <= right < n_z
        ):
            raise ValueError(
                "causal action head contrast pairs must use integer z "
                f"indices within [0, {n_z}), got {pair!r}"
            )
        if left == right or left in seen or right in seen:
            raise ValueError(
                "causal action head contrast pairs must contain distinct, "
                f"disjoint indices, got {pairs!r}"
            )
        if left not in supported or right not in supported:
            raise ValueError(
                "causal action head contrast pairs must be contained in "
                f"effective_dims, got pair={pair!r}, "
                f"effective_dims={tuple(sorted(supported))!r}"
            )
        seen.update(pair)
    return pairs


def project_causal_action_head_vector(
    values: Sequence[float],
    *,
    contrast_pairs: tuple[tuple[int, int], ...],
) -> tuple[float, ...]:
    """Orthogonally remove the common mode of each opponent-coded pair."""

    projected = [float(value) for value in values]
    for left, right in contrast_pairs:
        contrast = 0.5 * (projected[left] - projected[right])
        projected[left] = contrast
        projected[right] = -contrast
    return tuple(projected)


def project_base_code_off_contrast(
    values: Sequence[float],
    *,
    contrast_pairs: tuple[tuple[int, int], ...],
) -> tuple[float, ...]:
    """Orthogonally remove the antisymmetric part of each opponent-coded pair.

    Complement of :func:`project_causal_action_head_vector`: the head keeps
    only the contrast, the base keeps only the common mode. Under exclusive
    steering the temporal owner applies this to the deterministic base policy
    mean so the state-conditioned head is the single learned writer of the
    actuator contrast axes, while the base retains the common mode (speed).
    """

    projected = [float(value) for value in values]
    for left, right in contrast_pairs:
        common = 0.5 * (projected[left] + projected[right])
        projected[left] = common
        projected[right] = common
    return tuple(projected)


def project_causal_action_head_mirror_equivariant(
    values: Sequence[float],
    mirrored_values: Sequence[float],
    *,
    contrast_pairs: tuple[tuple[int, int], ...],
) -> tuple[float, ...]:
    """Project one head output onto the reflection-equivariant subspace.

    Unpaired coordinates are reflection-invariant and therefore average the
    two lanes. Each opponent-coded pair is a reflected actuator axis: its
    coordinates swap under reflection, so only the antisymmetric response
    ``0.5 * (f(s) - f(mirror(s)))`` may survive.
    """

    if len(values) != len(mirrored_values):
        raise ValueError(
            "causal action head mirror outputs must have equal dimensions: "
            f"values={len(values)}, mirrored={len(mirrored_values)}"
        )
    projected = [
        0.5 * (float(value) + float(mirrored))
        for value, mirrored in zip(values, mirrored_values, strict=True)
    ]
    for left, right in contrast_pairs:
        direct_contrast = 0.5 * (
            float(values[left]) - float(values[right])
        )
        mirrored_contrast = 0.5 * (
            float(mirrored_values[left]) - float(mirrored_values[right])
        )
        equivariant = 0.5 * (direct_contrast - mirrored_contrast)
        projected[left] = equivariant
        projected[right] = -equivariant
    return tuple(projected)


def mirror_causal_action_head_output_gradient(
    values: Sequence[float],
    *,
    contrast_pairs: tuple[tuple[int, int], ...],
) -> tuple[float, ...]:
    """Transform an output gradient into the mirrored training lane."""

    mirrored = [float(value) for value in values]
    for left, right in contrast_pairs:
        mirrored[left] = float(values[right])
        mirrored[right] = float(values[left])
    return tuple(mirrored)


__all__ = [
    "mirror_causal_action_head_input",
    "mirror_causal_action_head_output_gradient",
    "normalize_causal_action_head_input_mirror",
    "normalize_causal_action_head_contrast_pairs",
    "project_base_code_off_contrast",
    "project_causal_action_head_mirror_equivariant",
    "project_causal_action_head_vector",
]
