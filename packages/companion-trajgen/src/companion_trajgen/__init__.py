# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""companion-trajgen: synthetic labelled-trajectory generation for the
Relationship Representation Standard.

See README.md and docs/external/relationship-representation-rfc-v0.md.
"""

from companion_trajgen.exporter import arc_record_to_trajectory, write_trajectory
from companion_trajgen.labeler import label_arc
from companion_trajgen.pipeline import generate_dataset, load_public_scenarios

__all__ = [
    "arc_record_to_trajectory",
    "generate_dataset",
    "label_arc",
    "load_public_scenarios",
    "write_trajectory",
]
