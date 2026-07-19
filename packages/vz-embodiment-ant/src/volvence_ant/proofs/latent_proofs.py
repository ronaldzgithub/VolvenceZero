"""Authoritative, substrate-agnostic learning proofs (reused verbatim).

These are the SAME matched-control proofs the kernel uses as Phase-2 exit
evidence, accessed through the ``vz-runtime`` facade (not vz-temporal
internals):

- ``collect_internal_rl_no_optimize_proof`` — real PPO on the latent code
  ``z_t`` vs an identical no-optimize control on the same sparse/delayed-reward
  task. The learning claim holds only if optimize improves the return while the
  no-optimize control stays flat.
- ``collect_strict_eta_gate_evidence`` — the information-bottleneck ladder:
  higher alpha must increase switch sparsity monotonically while held-out
  action-family reuse does not degrade.

They run on the controller latent space, which is exactly what the digital-ant
project is validating is reusable independent of the substrate. Torch required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AntLatentProofReport:
    internal_rl_no_optimize_proof: dict[str, Any]
    strict_eta_gate: dict[str, Any]
    learning_is_real: bool
    eta_bottleneck_holds: bool
    description: str


def run_ant_latent_proofs(
    *, rl_iterations: int = 40, eta_epochs: int = 25
) -> AntLatentProofReport:
    """Run both latent-space proofs and summarize the verdicts.

    Raises ImportError if torch is unavailable (the proofs are torch-backed);
    the caller decides whether that is fatal for its lane.
    """

    from volvence_zero.agent.learned_shadow_evidence import (
        collect_internal_rl_no_optimize_proof,
        collect_strict_eta_gate_evidence,
    )

    rl_proof = collect_internal_rl_no_optimize_proof(iterations=rl_iterations)
    eta_gate = collect_strict_eta_gate_evidence(epochs=eta_epochs)

    learning_is_real = bool(
        rl_proof["full_improves"]
        and rl_proof["control_does_not_improve"]
        and rl_proof["full_beats_control"]
    )
    eta_holds = bool(eta_gate["gate_passed"])
    return AntLatentProofReport(
        internal_rl_no_optimize_proof=rl_proof,
        strict_eta_gate=eta_gate,
        learning_is_real=learning_is_real,
        eta_bottleneck_holds=eta_holds,
        description=(
            f"latent proofs: learning_is_real={learning_is_real} "
            f"(full_improves={rl_proof['full_improves']}, "
            f"control_flat={rl_proof['control_does_not_improve']}, "
            f"full_beats_control={rl_proof['full_beats_control']}); "
            f"eta_bottleneck_holds={eta_holds}"
        ),
    )
