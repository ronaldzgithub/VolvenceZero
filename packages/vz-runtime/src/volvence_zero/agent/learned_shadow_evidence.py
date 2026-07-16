"""Learned-shadow operator profile + unified evidence smoke artifact.

CP-LSS-01 (learned-shadow closure packet): per-owner SHADOW code for the four
torch autograd backends already exists (temporal SSL / temporal runtime
forward / internal RL / CMS band), but until this module there was no single
frozen operator profile and no one-command artifact that proves all four
owners produced SHADOW evidence on the same session without becoming live
writers.

Scope boundaries:

* This is the synthetic / CPU evidence lane only (evidence_program.md P0
  wiring tier). It proves wiring + no-side-effect SHADOW semantics; it does
  NOT constitute ACTIVE-promotion evidence, which requires the Linux CUDA
  real-trace lane per the AGI-uplift plan.
* The builder only READS owner-local evidence surfaces
  (``latest_runtime_shadow_report`` / ``latest_ssl_report`` /
  ``latest_internal_rl_report`` / ``latest_cms_backend_evidence``); it never
  re-runs training or mutates owner state.
* Missing evidence fails loudly (``LearnedShadowEvidenceError``) instead of
  exporting a partially-populated artifact.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel

LEARNED_SHADOW_EVIDENCE_SCHEMA_VERSION = "learned-shadow-evidence-smoke.v1"

#: Frozen learned-shadow operator profile: the controller capacity the plan
#: unlocks for evidence profiles (CP-02) without touching production defaults.
LEARNED_SHADOW_TEMPORAL_LATENT_DIM = 16

_SHADOW_BACKEND_FIELDS = (
    "temporal_ssl_backend",
    "temporal_runtime_backend",
    "internal_rl_backend",
    "cms_torch_backend",
)


class LearnedShadowEvidenceError(RuntimeError):
    """A required SHADOW evidence surface is missing or violates invariants."""


def build_learned_shadow_rollout_config() -> FinalRolloutConfig:
    """Return the frozen learned-shadow profile: all four torch backends SHADOW.

    Production defaults (all DISABLED) are intentionally untouched; operators
    opt into this profile explicitly. Rollback = construct a default
    ``FinalRolloutConfig()``.
    """

    return FinalRolloutConfig(
        temporal_ssl_backend=WiringLevel.SHADOW,
        temporal_runtime_backend=WiringLevel.SHADOW,
        internal_rl_backend=WiringLevel.SHADOW,
        cms_torch_backend=WiringLevel.SHADOW,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearnedShadowEvidenceError(message)


def _runtime_shadow_payload(report: Any, *, track_label: str) -> dict[str, Any]:
    _require(
        report is not None,
        f"{track_label} temporal policy published no runtime SHADOW report; "
        "was temporal_runtime_backend actually SHADOW and did a turn run?",
    )
    return {
        "steps_compared": report.steps_compared,
        "max_abs_diff_posterior_mean": report.max_abs_diff_posterior_mean,
        "max_abs_diff_z_tilde": report.max_abs_diff_z_tilde,
        "max_abs_diff_beta": report.max_abs_diff_beta,
        "max_abs_diff_applied": report.max_abs_diff_applied,
        "within_tolerance": report.within_tolerance,
        "tolerance": report.tolerance,
        "pure_latency_ms": report.pure_latency_ms,
        "torch_latency_ms": report.torch_latency_ms,
        "latency_ok": report.latency_ok,
        "promotable": report.promotable,
        "torch_available": report.torch_available,
        # CP-06 (GAP-09): behaviour-level comparison dimensions.
        "switch_decision_match": report.switch_decision_match,
        "family_selection_match": report.family_selection_match,
        "nearest_family_pure": report.nearest_family_pure,
        "nearest_family_torch": report.nearest_family_torch,
        "description": report.description,
    }


def _ssl_payload(report: Any) -> dict[str, Any]:
    _require(
        report is not None,
        "joint loop published no SSL training report; the schedule must run "
        "SSL at least once before exporting learned-shadow evidence.",
    )
    _require(
        report.torch_backend == WiringLevel.SHADOW.value,
        f"SSL torch backend evidence is {report.torch_backend!r}, expected "
        "'shadow'. Torch missing or the backend was not SHADOW.",
    )
    _require(
        not report.torch_wrote_back,
        "SSL torch backend wrote back under SHADOW; SHADOW must be side-effect free.",
    )
    return {
        "torch_backend": report.torch_backend,
        "torch_prediction_loss": report.torch_prediction_loss,
        "torch_kl_loss": report.torch_kl_loss,
        "torch_switch_sparsity": report.torch_switch_sparsity,
        "torch_parameters_changed": report.torch_parameters_changed,
        "torch_grad_norm": report.torch_grad_norm,
        "torch_wrote_back": report.torch_wrote_back,
        "pure_prediction_loss": report.prediction_loss,
        "pure_kl_loss": report.kl_loss,
        "trained_steps": report.trained_steps,
    }


def _internal_rl_track_payload(report: Any) -> dict[str, Any]:
    _require(
        report.torch_backend == WiringLevel.SHADOW.value,
        f"Internal RL torch backend evidence is {report.torch_backend!r}, "
        "expected 'shadow'. Torch missing, no transitions, or backend not SHADOW.",
    )
    _require(
        not report.torch_wrote_back,
        "Internal RL torch backend wrote back under SHADOW; SHADOW must be side-effect free.",
    )
    return {
        "torch_backend": report.torch_backend,
        "torch_parameters_changed": report.torch_parameters_changed,
        "torch_policy_loss": report.torch_policy_loss,
        "torch_value_loss": report.torch_value_loss,
        "torch_approx_kl": report.torch_approx_kl,
        "torch_wrote_back": report.torch_wrote_back,
        "transition_count": report.transition_count,
        "track": report.track.value,
    }


def _internal_rl_payload(report: Any) -> dict[str, Any]:
    _require(
        report is not None,
        "joint loop published no internal RL optimization report; run enough "
        "turns for a full cycle (rl_interval) before exporting evidence.",
    )
    # DualTrackOptimizationReport carries task/relationship OptimizationReports;
    # a bare OptimizationReport is a single-track run.
    if hasattr(report, "task_report"):
        return {
            "kind": "dual-track",
            "task": _internal_rl_track_payload(report.task_report),
            "relationship": _internal_rl_track_payload(report.relationship_report),
        }
    return {"kind": "single-track", "task": _internal_rl_track_payload(report)}


def _cms_payload(evidence: dict | None) -> dict[str, Any]:
    _require(
        evidence is not None,
        "CMS core published no backend evidence; at least one band update must "
        "run (a normal turn writes the online-fast band) before export.",
    )
    assert evidence is not None
    _require(
        evidence["backend"] == WiringLevel.SHADOW.value,
        f"CMS torch backend evidence is {evidence['backend']!r}, expected 'shadow'.",
    )
    _require(
        not evidence["wrote_back"],
        "CMS torch backend wrote back under SHADOW; SHADOW must be side-effect free.",
    )
    return dict(evidence)


def collect_strict_eta_gate_evidence(
    *,
    alphas: tuple[float, ...] = (0.0, 0.3, 1.0),
    epochs: int = 25,
    n_z: int = 8,
    seed: int = 1234,
) -> dict[str, Any]:
    """Run the CP-09 strict ETA suite and derive the promotion-gate boolean.

    Wraps ``run_strict_eta_evidence`` (owner: vz-temporal) so the learned-shadow
    lane and the soak artifact consume one shared payload instead of hardcoding
    ``strict_eta_gate_passed=False``. The gate passes only when all three
    directional claims hold on the controlled hierarchical suite:

    1. higher alpha increases switch sparsity (rate-distortion direction),
    2. sparsity is monotone non-decreasing across the alpha ladder,
    3. held-out action-family (code) reuse does not degrade under the
       bottleneck.

    Torch required; raises ImportError otherwise (the caller decides whether
    that is fatal for its lane). Synthetic-suite tier: this is the mechanism
    gate, not real-trace capability evidence.
    """

    from volvence_zero.temporal.torch_metacontroller import run_strict_eta_evidence

    evidence = run_strict_eta_evidence(
        alphas=alphas, epochs=epochs, n_z=n_z, seed=seed
    )
    gate_passed = (
        evidence.high_alpha_increases_sparsity
        and evidence.sparsity_monotone_nondecreasing
        and evidence.held_out_reuse_improves_with_bottleneck
    )
    return {
        "evidence_kind": "strict_eta_gate",
        "alphas": list(alphas),
        "epochs": epochs,
        "n_z": n_z,
        "seed": seed,
        "sparsity_by_alpha": [
            {"alpha": alpha, "switch_sparsity": sparsity}
            for alpha, sparsity in evidence.sparsity_by_alpha_standard_normal
        ],
        "high_alpha_increases_sparsity": evidence.high_alpha_increases_sparsity,
        "sparsity_monotone_nondecreasing": evidence.sparsity_monotone_nondecreasing,
        "held_out_reuse_alpha0": evidence.held_out_reuse_standard_normal_alpha0,
        "held_out_reuse_high_alpha": evidence.held_out_reuse_standard_normal_high_alpha,
        "held_out_reuse_improves_with_bottleneck": (
            evidence.held_out_reuse_improves_with_bottleneck
        ),
        "gate_passed": gate_passed,
        "description": evidence.description,
    }


def collect_learned_shadow_evidence(runner: Any) -> dict[str, Any]:
    """Assemble the unified learned-shadow evidence payload from a live runner.

    ``runner`` is an ``AgentSessionRunner`` that already executed at least one
    turn (and at least one full joint-loop cycle for internal RL evidence)
    under ``build_learned_shadow_rollout_config()``. Read-only: no owner state
    is mutated. Raises ``LearnedShadowEvidenceError`` when any of the four
    owners failed to produce SHADOW evidence or violated no-write-back.
    """

    config = runner.rollout_config
    for field_name in _SHADOW_BACKEND_FIELDS:
        level = getattr(config, field_name)
        _require(
            level is WiringLevel.SHADOW,
            f"learned-shadow evidence requires {field_name}=SHADOW, got {level.value!r}.",
        )
    code_dim = runner.temporal_latent_dim
    _require(
        code_dim > 3,
        f"learned-shadow evidence requires an ndim controller (n_z > 3), got {code_dim}.",
    )
    learned_core = runner.memory_store.learned_core
    _require(learned_core is not None, "memory store has no CMS learned core attached.")
    assert learned_core is not None

    payload: dict[str, Any] = {
        "schema_version": LEARNED_SHADOW_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": "learned_shadow_evidence_smoke",
        "evidence_tier": "p0-wiring-synthetic",
        "temporal_latent_dim": code_dim,
        "backend_wiring": {
            field_name: getattr(config, field_name).value
            for field_name in _SHADOW_BACKEND_FIELDS
        },
        "temporal_runtime": {
            "world": _runtime_shadow_payload(
                runner.world_temporal_policy.latest_runtime_shadow_report,
                track_label="world",
            ),
            "self": _runtime_shadow_payload(
                runner.self_temporal_policy.latest_runtime_shadow_report,
                track_label="self",
            ),
        },
        "temporal_ssl": _ssl_payload(runner.joint_loop.latest_ssl_report),
        "internal_rl": _internal_rl_payload(runner.joint_loop.latest_internal_rl_report),
        "cms": _cms_payload(learned_core.latest_cms_backend_evidence),
    }
    # M2 (#89 code side): the CMS torch-band promotion gate readout — settled
    # pure-vs-torch update-outcome comparisons, parity pass rate, and the
    # bounded anti-forgetting window — ships inside the same artifact so the
    # SHADOW->ACTIVE decision is one payload, not a downstream re-join.
    payload["cms"]["promotion_readout"] = asdict(
        learned_core.cms_backend_promotion_readout()
    )
    # CP-05 (GAP-09): SSL SHADOW candidate checkpoint + forward parity.
    candidate = runner.joint_loop.latest_torch_ssl_candidate
    if candidate is not None:
        from volvence_zero.runtime.kernel import stable_value_hash

        payload["temporal_ssl"]["candidate_checkpoint"] = {
            "trace_id": candidate.trace_id,
            "wrote_back": candidate.wrote_back,
            "candidate_fingerprint": stable_value_hash(
                (
                    candidate.candidate_encoder_parameters,
                    candidate.candidate_switch_parameters,
                    candidate.candidate_decoder_parameters,
                )
            ),
            "parameters_changed": candidate.parameters_changed,
        }
    parity = runner.joint_loop.latest_ssl_forward_parity
    if parity is not None:
        payload["temporal_ssl"]["forward_parity"] = _runtime_shadow_payload(
            parity, track_label="ssl-lane"
        )
    # CP-07 (GAP-09): reward composition purity readout.
    composition = runner.joint_loop.latest_reward_composition
    if composition is not None:
        payload["internal_rl"]["reward_composition"] = {
            "pe_derived_abs_fraction": composition["pe_derived_abs_fraction"],
            "reward_mode": composition["reward_mode"],
            "component_names": [name for name, _ in composition["components"]],
        }
    return payload


def collect_internal_rl_no_optimize_proof(
    *, iterations: int = 40
) -> dict[str, Any]:
    """Run the CP-07 offline proof episode: PPO vs no-optimize matched control.

    Wraps ``run_internal_rl_proof`` (owner: vz-temporal) so the learned-shadow
    soak carries the optimize-beats-no-optimize control evidence in the same
    artifact. Torch required; synthetic proof tier (not real-trace evidence).
    """

    from volvence_zero.internal_rl.torch_internal_rl import run_internal_rl_proof

    report = run_internal_rl_proof(iterations=iterations)
    return {
        "evidence_kind": "internal_rl_no_optimize_proof",
        "iterations": iterations,
        "full_mean_return_before": report.full.mean_return_before,
        "full_mean_return_after": report.full.mean_return_after,
        "full_return_improvement": report.full.return_improvement,
        "no_optimize_mean_return_after": report.no_optimize.mean_return_after,
        "no_optimize_return_improvement": report.no_optimize.return_improvement,
        "full_improves": report.full_improves,
        "control_does_not_improve": report.control_does_not_improve,
        "full_beats_control": report.full_beats_control,
        "description": report.description,
    }


__all__ = [
    "LEARNED_SHADOW_EVIDENCE_SCHEMA_VERSION",
    "LEARNED_SHADOW_TEMPORAL_LATENT_DIM",
    "LearnedShadowEvidenceError",
    "build_learned_shadow_rollout_config",
    "collect_internal_rl_no_optimize_proof",
    "collect_learned_shadow_evidence",
    "collect_strict_eta_gate_evidence",
]
