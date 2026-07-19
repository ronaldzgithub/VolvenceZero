"""Phase 2: rare-heavy caste plasticity (offline role reprogramming).

The caste profile is produced by an OFFLINE optimisation over colony yield under
a given environmental pressure — the digital analogue of the gene-expression
program that sets an individual's role during a developmental window. It is an
immutable artifact consumed at runtime and NEVER produced at runtime (mirroring
rare-heavy artifact semantics: online-fast cannot trigger it).
"""

from __future__ import annotations

from volvence_ant.caste.role_reprogramming import (
    CasteProfile,
    EnvironmentPressure,
    ReprogrammingResult,
    reprogram_castes,
)
from volvence_ant.caste.rare_heavy_roles import (
    ColonyRareHeavyBundle,
    IndividualRareHeavyRef,
    RoleProbe,
    RoleReadout,
    cluster_behavioral_roles,
)

__all__ = [
    "CasteProfile",
    "ColonyRareHeavyBundle",
    "EnvironmentPressure",
    "IndividualRareHeavyRef",
    "ReprogrammingResult",
    "RoleProbe",
    "RoleReadout",
    "cluster_behavioral_roles",
    "reprogram_castes",
]
