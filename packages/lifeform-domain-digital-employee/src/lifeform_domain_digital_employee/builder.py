"""Convenience factory for the digital-employee org / twin lifeforms.

This mirrors :func:`lifeform_domain_emogpt.build_companion_lifeform` but
swaps the companion ``DomainExperiencePackage`` for the role-specialised
org / twin packs in this wheel. v0 intentionally reuses the companion
pre-trained calibration basin (vitals + temporal + regime bootstraps): the
kernel ships flat priors, and the companion bootstraps are a reasonable
warm start for a relationship-aware work agent. The *behavioural*
differentiation that makes org != twin != companion is carried by the
data-only domain experience package (knowledge / case / playbook /
boundary priors), which is the documented per-vertical seam.

Once a dedicated digital-employee super-loop produces org/twin-specific
bootstraps, only the ``_load_*_bootstrap`` calls below change; the rest of
the wiring (and the BFF, and the launcher) stays put.
"""

from __future__ import annotations

from typing import Any, Literal

from lifeform_domain_digital_employee.org_pack import (
    build_digital_employee_org_package,
)
from lifeform_domain_digital_employee.twin_pack import (
    build_digital_employee_twin_package,
)

Role = Literal["org", "twin"]


def build_digital_employee_lifeform(
    *,
    role: Role,
    config: object | None = None,
    use_temporal_bootstrap: bool = True,
    use_regime_bootstrap: bool = True,
    use_vitals_bootstrap: bool = True,
    substrate_runtime: Any = None,
    semantic_proposal_runtime: Any = None,
    memory_store: Any = None,
    response_synthesizer: Any = None,
    identity_provider: Any = None,
) -> Any:
    """Build a role-specialised digital-employee ``Lifeform``.

    Args:
        role: ``"org"`` for the company-level OrgAgent, ``"twin"`` for the
            per-employee EmployeeTwin. Selects which domain experience
            package is compiled into the kernel's application owners.
        substrate_runtime: optional shared ``OpenWeightResidualRuntime``;
            when supplied every session shares this one model instance
            (multi-tenant single-GPU path), identical to the companion
            factory's contract.

    The remaining args mirror
    :func:`lifeform_domain_emogpt.build_companion_lifeform` 1:1.
    """

    from dataclasses import replace as _replace

    from lifeform_core import Lifeform, LifeformConfig
    from volvence_zero.substrate import SemanticFeatureSurfaceSubstrateAdapter

    # Reuse the companion calibration basin (vitals + bootstraps) as the
    # v0 warm start. These imports stay local so importing this wheel does
    # not pull torch/transformers transitively (see emogpt __getattr__).
    from lifeform_domain_emogpt import (
        build_companion_vitals_bootstrap,
        load_companion_regime_bootstrap,
        load_companion_temporal_bootstrap,
    )

    if role == "org":
        package = build_digital_employee_org_package()
        surface_label = "digital-employee-org-semantic-surface"
    elif role == "twin":
        package = build_digital_employee_twin_package()
        surface_label = "digital-employee-twin-semantic-surface"
    else:  # pragma: no cover - typed Literal guards this at call sites
        raise ValueError(f"unknown digital-employee role: {role!r}")

    base_config = config if isinstance(config, LifeformConfig) else LifeformConfig()
    brain_overrides: dict[str, Any] = {"rare_heavy_enabled": False}
    if substrate_runtime is not None:
        brain_overrides["substrate_mode"] = "injected"
    base_config = _replace(
        base_config,
        brain_config=_replace(base_config.brain_config, **brain_overrides),
    )
    base_config = base_config.with_domain_experience((package,))

    if use_vitals_bootstrap:
        base_config = base_config.with_vitals(build_companion_vitals_bootstrap())

    temporal_bootstrap = (
        load_companion_temporal_bootstrap() if use_temporal_bootstrap else None
    )
    regime_bootstrap = (
        load_companion_regime_bootstrap() if use_regime_bootstrap else None
    )

    return Lifeform(
        base_config,
        temporal_bootstrap=temporal_bootstrap,
        regime_bootstrap=regime_bootstrap,
        substrate_runtime=substrate_runtime,
        substrate_adapter_factory=(
            None
            if substrate_runtime is not None
            else lambda user_input, turn_index: SemanticFeatureSurfaceSubstrateAdapter(
                source_text=user_input,
                model_id=f"{surface_label}:{turn_index}",
                fallback_active=1.0,
            )
        ),
        response_synthesizer=response_synthesizer,
        semantic_proposal_runtime=semantic_proposal_runtime,
        memory_store=memory_store,
        identity_provider=identity_provider,
    )
