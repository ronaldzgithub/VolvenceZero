"""Readout-only role clustering and rare-heavy bundle contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndividualRareHeavyRef:
    individual_id: int
    artifact_id: str
    artifact_digest: str
    provenance: str
    gate_verdict: str


@dataclass(frozen=True)
class ColonyRareHeavyBundle:
    schema_version: str
    pressure_label: str
    individuals: tuple[IndividualRareHeavyRef, ...]
    rollback_verified: bool


@dataclass(frozen=True)
class RoleProbe:
    individual_id: int
    trajectory_radius: float
    trail_reliance: float
    discovery_contribution: float
    patrol_contribution: float


@dataclass(frozen=True)
class RoleReadout:
    individual_id: int
    cluster_id: int
    role_label: str


def cluster_behavioral_roles(
    probes: tuple[RoleProbe, ...],
    *,
    seed: int = 0,
) -> tuple[RoleReadout, ...]:
    """Cluster held-out behavior; labels never feed back into training."""

    if len(probes) < 2:
        raise ValueError("role clustering requires at least two probes")
    features = np.asarray(
        [
            (
                probe.trajectory_radius,
                probe.trail_reliance,
                probe.discovery_contribution,
                probe.patrol_contribution,
            )
            for probe in probes
        ],
        dtype=float,
    )
    scale = np.maximum(features.std(axis=0), 1e-8)
    normalized = (features - features.mean(axis=0)) / scale
    rng = np.random.default_rng(seed)
    centers = normalized[rng.choice(len(probes), size=2, replace=False)].copy()
    assignments = np.zeros(len(probes), dtype=int)
    for _ in range(32):
        distances = np.linalg.norm(
            normalized[:, None, :] - centers[None, :, :],
            axis=2,
        )
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, assignments) and _ > 0:
            break
        assignments = updated
        for cluster_id in (0, 1):
            members = normalized[assignments == cluster_id]
            if len(members) == 0:
                raise ValueError("degenerate role clustering produced an empty cluster")
            centers[cluster_id] = members.mean(axis=0)
    explorer_cluster = int(
        np.argmax(
            [
                features[assignments == cluster_id, 0].mean()
                + features[assignments == cluster_id, 2].mean()
                for cluster_id in (0, 1)
            ]
        )
    )
    return tuple(
        RoleReadout(
            individual_id=probe.individual_id,
            cluster_id=int(cluster_id),
            role_label="explorer" if cluster_id == explorer_cluster else "patroller",
        )
        for probe, cluster_id in zip(probes, assignments, strict=True)
    )


__all__ = [
    "ColonyRareHeavyBundle",
    "IndividualRareHeavyRef",
    "RoleProbe",
    "RoleReadout",
    "cluster_behavioral_roles",
]
