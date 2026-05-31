"""Vertical: repair30 field-service repair assistant (D23).

Public API:

* ``build_repair30_package`` — the repair30
  ``DomainExperiencePackage`` (diagnostic triage / safety-gate /
  parts-and-procedure / customer-comms regime priors).
* ``build_repair30_lifeform`` — convenience factory returning a
  ready-to-run ``Lifeform`` for the repair30 vertical.

Design posture (see also the repo rule ``first-principles-not-patches``):

The repair30 product previously had no dedicated boundary and woke
against the ``einstein-full`` figure vertical (deploy debt D23). That is
wrong: a field-service repair assistant is not a historical-figure
persona, and the einstein boundary refuses anything post-1955 / outside
physics — useless for an appliance repair flow. This wheel gives
repair30 its own boundary as *data* — a domain experience package — with
no prompt strings and no keyword->behaviour maps. Regime names are
runtime-state priors fed to the playbook / boundary owners, not labels
matched against text. v0 reuses the companion calibration basin (vitals
+ temporal + regime bootstraps); only the domain experience differs.
This is the documented per-vertical seam, so a future dedicated
repair30 super-loop bootstrap only changes this wheel, never the kernel.
"""

from __future__ import annotations

from lifeform_domain_repair30.builder import build_repair30_lifeform
from lifeform_domain_repair30.repair_pack import build_repair30_package

__all__ = (
    "build_repair30_lifeform",
    "build_repair30_package",
)
