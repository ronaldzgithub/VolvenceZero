"""Generic actuator-subspace projection for the causal latent action head."""

from __future__ import annotations

from collections.abc import Sequence


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


__all__ = [
    "normalize_causal_action_head_contrast_pairs",
    "project_causal_action_head_vector",
]
