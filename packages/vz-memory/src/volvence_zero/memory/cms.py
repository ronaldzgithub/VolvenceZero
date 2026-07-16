"""CMS multi-timescale memory core (online-fast / session-medium /
background-slow bands with gradient-style updates and anti-forgetting).

The pure-data contracts, per-band MLP, and low-level math helpers live
in sibling modules (``cms_contracts``, ``cms_band_mlp``, ``cms_math``).
This module keeps the runtime class and re-exports the full public API
so existing ``from volvence_zero.memory.cms import X`` imports continue
to work without changes.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from volvence_zero.learned_update import (
    LEARNED_UPDATE_BASE_FEATURE_DIM,
    LEARNED_UPDATE_PE_AWARE_FEATURE_DIM,
    LEARNED_UPDATE_PE_FEATURE_DIM,
    LearnedUpdateDecision,
    LearnedUpdateRule,
    LearnedUpdateRuleState,
)
from volvence_zero.memory.cms_band_mlp import CMSBandMLP
from volvence_zero.memory.cms_contracts import (
    CMSBandState,
    CMSCheckpointState,
    CMSContinuumBand,
    CMSContinuumProfile,
    CMSContinuumReconstructionEdge,
    CMSHopeSelfModificationState,
    CMSState,
    CMSTowerConsolidationUpdate,
    CMSTowerLevelState,
    CMSTowerProfile,
    CMSVariant,
)
from volvence_zero.memory.cms_math import _clamp, _init_weight, _matvec
from volvence_zero.memory.runtime_evidence import build_runtime_backbone_evidence
from volvence_zero.substrate import SubstrateSnapshot

if TYPE_CHECKING:
    from volvence_zero.prediction.error import PredictionErrorSnapshot


# ATLAS / Titans uplift defaults (see docs/specs/cms-atlas-titans-uplift.md).
# When ``replay_window_sizes`` is None, replay is disabled (K=1 path used,
# numerically equivalent to the pre-uplift CMS).
_REPLAY_DEFAULT_K: dict[str, int] = {
    "online-fast": 8,
    "session-medium": 4,
    "background-slow": 2,
}
_REPLAY_DEFAULT_GAMMA: dict[str, float] = {
    "online-fast": 0.6,
    "session-medium": 0.7,
    "background-slow": 0.8,
}
_REPLAY_HARD_CAP_K: int = 32

# --- M2 (#89 code side): SHADOW→ACTIVE promotion tracker constants ----------
# Exit conditions for the torch band backend, evaluated purely in code; the
# ACTIVE flip itself stays gated on ≥500-turn real-trace evidence (#89 Stage 1).
_CMS_PROMOTION_MIN_COMPARISONS: int = 50
_CMS_PROMOTION_PARITY_FLOOR: float = 0.99
# The torch one-step update must land at least as close to the target as the
# pure manual-backprop step (mean per-update MSE, band outputs live in [0,1]).
_CMS_PROMOTION_MSE_SLACK: float = 1e-6
# Kill condition: torch step consistently lands materially farther from the
# target than the pure step -> recommend rollback to DISABLED.
_CMS_KILL_MSE_DEGRADATION: float = 0.05
# Bounded window for the #89 anti-forgetting gain-curve hooks.
_ANTI_FORGETTING_WINDOW: int = 64


@dataclass(frozen=True)
class CMSBackendPromotionReadout:
    """Code-level promotion verdict for the torch CMS band backend (M2 / #89).

    Report-only. Aggregates the SHADOW dual-run outcomes (forward parity +
    update-outcome comparison pure-vs-torch) plus the bounded anti-forgetting
    window, and evaluates the documented exit / kill conditions. ``promotable``
    means the CODE gate passes; the ACTIVE flip additionally requires the
    real-trace evidence run (#89 Stage 1), which is out of scope here.
    """

    backend: str
    torch_available: bool
    settled_comparisons: int
    parity_checks: int
    parity_pass_rate: float
    pure_update_mse: float
    torch_update_mse: float
    min_comparisons_met: bool
    parity_floor_met: bool
    torch_not_worse: bool
    kill_condition_met: bool
    promotable: bool
    absorption_window_mean: float
    retention_window_mean: float
    anti_forgetting_samples: int
    description: str


def _pure_band_forward(
    *,
    d_in: int,
    d_hidden: int,
    w1_flat: tuple[float, ...],
    w2_flat: tuple[float, ...],
    x: tuple[float, ...],
) -> tuple[float, ...]:
    """Legacy band forward ``y = clamp(x + W1 tanh(W2 x), 0, 1)`` from flat weights."""

    hidden = tuple(
        math.tanh(value) for value in _matvec(list(w2_flat), x, d_hidden, d_in)
    )
    residual = _matvec(list(w1_flat), hidden, d_in, d_hidden)
    return tuple(
        _clamp(x_value + r_value) for x_value, r_value in zip(x, residual, strict=True)
    )


def _mse(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / max(
        len(left), 1
    )


class CMSMemoryCore:
    """Multi-timescale memory core with gradient-style updates.

    Each band has its own learning rate and momentum. Anti-forgetting
    backflow prevents catastrophic overwriting of slow bands.

    Supports two modes:
    - ``"vector"``: fixed-dim vector per band (original behavior)
    - ``"mlp"``: 2-layer residual MLP per band (higher capacity)
    """

    def __init__(
        self,
        *,
        dim: int = 3,
        mode: str = "vector",
        d_in: int = 16,
        d_hidden: int = 32,
        variant: str = "sequential",
        session_cadence: int = 2,
        background_cadence: int = 4,
        online_lr: float = 0.65,
        session_lr: float = 0.3,
        background_lr: float = 0.1,
        momentum_beta: float = 0.9,
        anti_forgetting: float = 0.1,
        # ATLAS / Titans uplift flags (see docs/specs/cms-atlas-titans-uplift.md).
        # Defaults preserve canonical (pre-uplift) behavior so existing wiring
        # is unchanged. SHADOW path turns these on; ACTIVE follows acceptance.
        pe_features_enabled: bool = False,
        replay_window_sizes: Mapping[str, int] | None = None,
        cms_backend: "WiringLevel | None" = None,
    ) -> None:
        if mode not in ("vector", "mlp"):
            raise ValueError(f"mode must be 'vector' or 'mlp', got {mode!r}")
        from volvence_zero.runtime import WiringLevel as _WiringLevel

        # autograd-owner-integration Phase D: torch band gradient kernel routing.
        # DISABLED keeps the pure CMSBandMLP update as the live writer / rollback
        # baseline; SHADOW additionally runs a real torch autograd step for
        # evidence; ACTIVE makes the torch step authoritative for the band W1/W2
        # while keeping the band's pure state/momentum for coherent backflow.
        self._cms_backend = cms_backend if cms_backend is not None else _WiringLevel.DISABLED
        self._latest_cms_backend_evidence: dict | None = None
        # M2 (#89 code side): SHADOW dual-run promotion tracker. Aggregates
        # forward-parity results and the pure-vs-torch update-outcome
        # comparison across band updates so `cms_backend_promotion_readout()`
        # can evaluate the exit / kill conditions in code.
        self._backend_settled_comparisons = 0
        self._backend_parity_checks = 0
        self._backend_parity_passes = 0
        self._backend_pure_mse_sum = 0.0
        self._backend_torch_mse_sum = 0.0
        # M2 (#89): rollback drill for ACTIVE torch write-back. Keyed by
        # band_id; each entry is the band's full pre-update export_params
        # tuple, restorable via `rollback_last_torch_writeback`.
        self._last_pre_writeback_params: dict[str, tuple] = {}
        # M2 (#89): bounded anti-forgetting window (absorption, retention)
        # feeding the gain-curve hooks of the evidence run.
        self._anti_forgetting_window: deque[tuple[float, float]] = deque(
            maxlen=_ANTI_FORGETTING_WINDOW
        )
        self._mode = mode
        self._variant = CMSVariant(variant)
        self._session_cadence = max(session_cadence, 1)
        self._background_cadence = max(background_cadence, 1)
        self._online_lr = online_lr
        self._session_lr = session_lr
        self._background_lr = background_lr
        self._momentum_beta = momentum_beta
        self._anti_forgetting = anti_forgetting
        self._pe_features_enabled = bool(pe_features_enabled)
        # ATLAS / Titans uplift: feature_dim depends on the PE-gating flag.
        # pe_off keeps the legacy ``max(12, d_in)`` layout so the canonical
        # CMS path is bit-equal to pre-uplift behavior. pe_on uses the
        # canonical PE-aware layout (12 base + 4 PE) so the rule learns
        # weights specific to the PE input semantic.
        rule_feature_dim = (
            LEARNED_UPDATE_PE_AWARE_FEATURE_DIM
            if self._pe_features_enabled
            else max(LEARNED_UPDATE_BASE_FEATURE_DIM, d_in if mode == "mlp" else dim)
        )
        self._update_rule = LearnedUpdateRule(
            rule_id="cms-update",
            feature_dim=rule_feature_dim,
            hidden_dim=max(8, (d_hidden // 2) if mode == "mlp" else dim + 2),
        )
        # Replay window per band. ``None`` disables replay entirely; otherwise
        # missing bands fall back to ``_REPLAY_DEFAULT_K`` and unknown band
        # ids are ignored.
        self._atlas_replay_active = replay_window_sizes is not None
        configured_replay = dict(replay_window_sizes or {})
        self._replay_window_sizes: dict[str, int] = {}
        for band_id, default_k in _REPLAY_DEFAULT_K.items():
            requested = configured_replay.get(band_id, default_k if self._atlas_replay_active else 1)
            self._replay_window_sizes[band_id] = max(1, min(int(requested), _REPLAY_HARD_CAP_K))
        self._replay_buffers: dict[str, deque[tuple[float, ...]]] = {
            band_id: deque(maxlen=self._replay_window_sizes[band_id])
            for band_id in _REPLAY_DEFAULT_K
        }
        self._latest_replay_window_size: dict[str, int] = {
            band_id: 0 for band_id in _REPLAY_DEFAULT_K
        }
        self._latest_pe_features: tuple[float, ...] = tuple(
            0.0 for _ in range(LEARNED_UPDATE_PE_FEATURE_DIM)
        )
        self._latest_pe_features_by_band: dict[str, tuple[float, ...]] = {
            band_id: tuple(0.0 for _ in range(LEARNED_UPDATE_PE_FEATURE_DIM))
            for band_id in _REPLAY_DEFAULT_K
        }
        self._latest_update_rule_state: LearnedUpdateRuleState = self._update_rule.export_state()
        self._latest_band_decisions: dict[str, LearnedUpdateDecision] = {}
        self._hope_update_count = 0
        self._hope_last_target_id = ""
        self._hope_generated_learning_rate = 0.0
        self._hope_generated_decay_rate = 0.0
        self._hope_generated_reset_rate = 0.0
        self._hope_last_improvement = 0.0
        self._hope_last_stability = 0.0
        self._hope_last_reward = 0.0
        self._hope_guarded = False
        self._hope_guard_reason = ""
        self._total_observations = 0
        self._total_reflections = 0
        self._last_update_ms = 0
        self._session_observations_since_update = 0
        self._background_observations_since_update = 0
        # #89 anti-forgetting proxies (report-only). Updated at the end of
        # each substrate observation from the actual per-turn band drift.
        # absorption = fast band moved toward the new signal; retention =
        # 1 - slow band drift (the substrate should barely move).
        self._last_new_knowledge_absorption = 0.0
        self._last_old_knowledge_retention = 1.0

        if mode == "mlp":
            self._dim = d_in
            self._d_hidden = d_hidden
            self._online_mlp = CMSBandMLP(
                d_in=d_in, d_hidden=d_hidden,
                learning_rate=online_lr, momentum_beta=momentum_beta,
            )
            self._session_mlp = CMSBandMLP(
                d_in=d_in, d_hidden=d_hidden,
                learning_rate=session_lr, momentum_beta=momentum_beta,
            )
            self._background_mlp = CMSBandMLP(
                d_in=d_in, d_hidden=d_hidden,
                learning_rate=background_lr, momentum_beta=momentum_beta,
            )
            self._session_pending_signal = tuple(0.0 for _ in range(d_in))
            self._background_pending_signal = tuple(0.0 for _ in range(d_in))
            self._nested_session_init_target = tuple(0.0 for _ in range(d_in))
            self._nested_online_init_target = tuple(0.0 for _ in range(d_in))
            self._nested_context_steps = 0
        else:
            self._dim = dim
            self._d_hidden = 0
            self._online_fast = tuple(0.0 for _ in range(dim))
            self._session_medium = tuple(0.0 for _ in range(dim))
            self._background_slow = tuple(0.0 for _ in range(dim))
            self._online_momentum = tuple(0.0 for _ in range(dim))
            self._session_momentum = tuple(0.0 for _ in range(dim))
            self._background_momentum = tuple(0.0 for _ in range(dim))
            self._session_pending_signal = tuple(0.0 for _ in range(dim))
            self._background_pending_signal = tuple(0.0 for _ in range(dim))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def variant(self) -> str:
        return self._variant.value

    def clone_empty(self) -> "CMSMemoryCore":
        return CMSMemoryCore(
            dim=self._dim,
            mode=self._mode,
            d_in=self._dim,
            d_hidden=self._d_hidden,
            variant=self._variant.value,
            session_cadence=self._session_cadence,
            background_cadence=self._background_cadence,
            online_lr=self._online_lr,
            session_lr=self._session_lr,
            background_lr=self._background_lr,
            momentum_beta=self._momentum_beta,
            anti_forgetting=self._anti_forgetting,
            pe_features_enabled=self._pe_features_enabled,
            replay_window_sizes=(
                dict(self._replay_window_sizes) if self._atlas_replay_active else None
            ),
        )

    @property
    def pe_features_enabled(self) -> bool:
        return self._pe_features_enabled

    @property
    def atlas_replay_active(self) -> bool:
        return self._atlas_replay_active

    def _mean_distance(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if not left or not right:
            return 0.0
        n = min(len(left), len(right))
        return sum(abs(left[index] - right[index]) for index in range(n)) / max(n, 1)

    def _decision_features(
        self,
        *,
        current: tuple[float, ...],
        target: tuple[float, ...],
        pending_signal: tuple[float, ...],
        observations_since_update: int,
        cadence_interval: int,
        source_signal: tuple[float, ...],
    ) -> tuple[float, ...]:
        current = self._align_signal_dim(current)
        target = self._align_signal_dim(target)
        pending_signal = self._align_signal_dim(pending_signal)
        source_signal = self._align_signal_dim(source_signal)
        current_norm = self._mean_distance(current, tuple(0.0 for _ in range(self._dim)))
        target_norm = self._mean_distance(target, tuple(0.0 for _ in range(self._dim)))
        delta_norm = self._mean_distance(current, target)
        pending_norm = self._mean_distance(pending_signal, tuple(0.0 for _ in range(self._dim)))
        source_norm = self._mean_distance(source_signal, tuple(0.0 for _ in range(self._dim)))
        cadence_pressure = observations_since_update / max(cadence_interval, 1)
        # Base 12 features (legacy layout).
        base = (
            current_norm,
            target_norm,
            delta_norm,
            pending_norm,
            source_norm,
            _clamp(cadence_pressure),
            _clamp(self._anti_forgetting),
            _clamp(self._online_lr),
            _clamp(self._session_lr),
            _clamp(self._background_lr),
            _clamp(self._momentum_beta),
            1.0 if self._variant is CMSVariant.NESTED else 0.0,
        )
        assert len(base) == LEARNED_UPDATE_BASE_FEATURE_DIM
        # ATLAS / Titans uplift: append PE magnitudes only when the flag is
        # on. When pe_features_enabled is False, return the legacy 12-tuple
        # so the rule sees exactly the pre-uplift feature layout. The rule's
        # ``_align_features`` will modulo-extend it to its own feature_dim
        # (which equals the legacy ``max(12, d_in)`` in this mode), keeping
        # canonical behavior numerically identical to pre-uplift.
        if self._pe_features_enabled:
            return base + tuple(self._latest_pe_features)
        return base

    @staticmethod
    def _pe_features_from_snapshot(
        prediction_error: "PredictionErrorSnapshot | None",
    ) -> tuple[float, ...]:
        """Extract a 4-tuple of PE magnitudes for the LearnedUpdateRule.

        Order: ``(|task_error|, |relationship_error|, |regime_error|,
        |action_error|)``. Returns all-zero when prediction error is missing
        or the turn is bootstrap (no usable signal).
        """
        if prediction_error is None or prediction_error.bootstrap:
            return tuple(0.0 for _ in range(LEARNED_UPDATE_PE_FEATURE_DIM))
        error = prediction_error.error
        return (
            _clamp(abs(error.task_error)),
            _clamp(abs(error.relationship_error)),
            _clamp(abs(error.regime_error)),
            _clamp(abs(error.action_error)),
        )

    def _set_latest_pe_features(
        self,
        prediction_error: "PredictionErrorSnapshot | None",
    ) -> None:
        if self._pe_features_enabled:
            self._latest_pe_features = self._pe_features_from_snapshot(prediction_error)
        else:
            self._latest_pe_features = tuple(
                0.0 for _ in range(LEARNED_UPDATE_PE_FEATURE_DIM)
            )

    def _record_pe_features_for_band(self, band_id: str) -> None:
        if band_id in self._latest_pe_features_by_band:
            self._latest_pe_features_by_band[band_id] = tuple(self._latest_pe_features)

    def _replay_targets_for_band(
        self,
        band_id: str,
        *,
        current_target: tuple[float, ...],
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
        """Return (targets, weights) for a replay update on this band.

        When replay is disabled or the band has no buffer, returns a
        single-target window with weight 1.0, which makes
        ``CMSBandMLP.update_with_replay`` numerically equivalent to the
        legacy ``update`` call.
        """
        normalized = tuple(float(value) for value in current_target)
        if not self._atlas_replay_active or band_id not in self._replay_buffers:
            self._latest_replay_window_size[band_id] = 1
            return ((normalized,), (1.0,))
        buffer = self._replay_buffers[band_id]
        buffer.append(normalized)
        gamma = _REPLAY_DEFAULT_GAMMA.get(band_id, 0.7)
        targets = tuple(buffer)
        K = len(targets)
        weights = tuple(
            gamma ** (K - 1 - index) for index in range(K)
        )
        self._latest_replay_window_size[band_id] = K
        return (targets, weights)

    def _zero_pad_pe_columns_after_restore(self) -> None:
        """Reset PE feature columns of the LearnedUpdateRule projection.

        See ``CMSMemoryCore.restore_state`` for the invariant this protects.
        """
        self._update_rule.zero_input_columns(
            start=LEARNED_UPDATE_BASE_FEATURE_DIM,
            end=LEARNED_UPDATE_PE_AWARE_FEATURE_DIM,
        )
        self._latest_update_rule_state = self._update_rule.export_state()

    def _band_mlp_update(
        self,
        *,
        band_id: str,
        mlp: CMSBandMLP,
        target: tuple[float, ...],
        decision: LearnedUpdateDecision,
    ) -> None:
        targets, weights = self._replay_targets_for_band(
            band_id,
            current_target=target,
        )
        lr_scale = max(0.05, decision.step_scale)
        from volvence_zero.runtime import WiringLevel

        backend_active = self._cms_backend in (WiringLevel.SHADOW, WiringLevel.ACTIVE)
        if not backend_active:
            mlp.update_with_replay(
                targets=targets, weights=weights,
                lr_scale=lr_scale, momentum_gate=decision.momentum_gate,
            )
            return

        from volvence_zero.tensor_backend import is_torch_available

        averaged = self._weighted_average_target(targets, weights)
        # Capture pre-update weights so the torch step starts from the same point
        # as the pure step (apples-to-apples), then run the pure update to keep
        # coherent state/momentum for backflow / mix_from / checkpointing.
        pre = mlp.export_params()  # (state, state_mom, w2, w1, w2_mom, w1_mom)
        mlp.update_with_replay(
            targets=targets, weights=weights,
            lr_scale=lr_scale, momentum_gate=decision.momentum_gate,
        )
        if averaged is None or not is_torch_available():
            self._latest_cms_backend_evidence = {
                "backend": self._cms_backend.value, "parameters_changed": 0,
                "wrote_back": False, "reason": "no-target-or-no-torch",
            }
            return

        from volvence_zero.memory.torch_cms_band import torch_band_update_from_params

        result = torch_band_update_from_params(
            d_in=mlp.d_in, d_hidden=mlp.d_hidden,
            w1_flat=pre[3], w2_flat=pre[2], state=pre[0],
            target=averaged, learning_rate=0.1 * lr_scale,
        )
        wrote_back = self._cms_backend is WiringLevel.ACTIVE
        if wrote_back:
            # M2 (#89): retain the full pre-update params as the rollback
            # point for this write-back (R15 rollback drill contract).
            self._last_pre_writeback_params[band_id] = pre
            post = mlp.export_params()  # state/momentum from the pure update
            mlp.restore_params(
                (post[0], post[1], result.w2_flat, result.w1_flat, post[4], post[5])
            )
        self._latest_cms_backend_evidence = {
            "backend": self._cms_backend.value,
            "band_id": band_id,
            "parameters_changed": result.parameters_changed,
            "parameter_change_rate": result.parameter_change_rate,
            "grad_norm": result.grad_norm,
            "loss": result.loss,
            "wrote_back": wrote_back,
            # CP-08 (GAP-09): last-known anti-forgetting proxies; the
            # end-of-observe enrichment refreshes them with this
            # observation's values. Band updates fired outside a full
            # observe cycle (e.g. consolidation) keep the prior readout
            # instead of dropping the fields.
            "new_knowledge_absorption": _clamp(self._last_new_knowledge_absorption),
            "old_knowledge_retention": _clamp(self._last_old_knowledge_retention),
        }
        if self._cms_backend is WiringLevel.SHADOW:
            # CP-08 (GAP-09): the SHADOW gate is forward parity on the LIVE
            # pre-update band weights (pure vs torch vs legacy), not just
            # the torch step scalars. Runs against the band's own state and
            # this update's averaged target so parity is exercised on real
            # runtime vectors.
            from volvence_zero.memory.torch_cms_band import (
                CMSBandWeights,
                cms_band_shadow_dual_run,
            )

            parity = cms_band_shadow_dual_run(
                weights=CMSBandWeights(
                    d_in=mlp.d_in,
                    d_hidden=mlp.d_hidden,
                    w1=tuple(pre[3]),
                    w2=tuple(pre[2]),
                ),
                inputs=(tuple(pre[0]), averaged),
            )
            self._latest_cms_backend_evidence.update(
                {
                    "forward_parity_max_abs_diff_pure_torch": parity.max_abs_diff_pure_torch,
                    "forward_parity_max_abs_diff_legacy_torch": parity.max_abs_diff_legacy_torch,
                    "forward_parity_within_tolerance": parity.within_tolerance,
                    "forward_parity_tolerance": parity.tolerance,
                    "forward_parity_pure_latency_ms": parity.pure_latency_ms,
                    "forward_parity_torch_latency_ms": parity.torch_latency_ms,
                    "forward_parity_promotable": parity.promotable,
                }
            )
            self._backend_parity_checks += 1
            if parity.within_tolerance:
                self._backend_parity_passes += 1
            # M2 (#89 code side): update-outcome dual-run. Both candidates
            # started from the same pre-update weights and chased the same
            # averaged target; compare where their one-step updates landed
            # (forward on the pre-update state vs the target). This is the
            # apples-to-apples "which update rule learns better" comparison
            # the SHADOW->ACTIVE gate needs, aggregated by the promotion
            # tracker; report-only, no write-back.
            post_pure = mlp.export_params()
            pure_output = _pure_band_forward(
                d_in=mlp.d_in, d_hidden=mlp.d_hidden,
                w1_flat=tuple(post_pure[3]), w2_flat=tuple(post_pure[2]),
                x=tuple(pre[0]),
            )
            torch_output = _pure_band_forward(
                d_in=mlp.d_in, d_hidden=mlp.d_hidden,
                w1_flat=result.w1_flat, w2_flat=result.w2_flat,
                x=tuple(pre[0]),
            )
            pure_update_mse = _mse(pure_output, averaged)
            torch_update_mse = _mse(torch_output, averaged)
            self._backend_settled_comparisons += 1
            self._backend_pure_mse_sum += pure_update_mse
            self._backend_torch_mse_sum += torch_update_mse
            self._latest_cms_backend_evidence.update(
                {
                    "update_outcome_pure_mse": pure_update_mse,
                    "update_outcome_torch_mse": torch_update_mse,
                    "update_outcome_settled_comparisons": (
                        self._backend_settled_comparisons
                    ),
                }
            )

    @staticmethod
    def _weighted_average_target(
        targets: tuple[tuple[float, ...], ...],
        weights: tuple[float, ...],
    ) -> tuple[float, ...] | None:
        """Replicate CMSBandMLP.update_with_replay's normalized weighted target."""

        if not targets:
            return None
        total = sum(max(0.0, w) for w in weights)
        if total <= 1e-9:
            return None
        d = len(targets[0])
        averaged = [0.0] * d
        for target, weight in zip(targets, weights, strict=True):
            nw = max(0.0, weight) / total
            if nw <= 0.0:
                continue
            for i in range(d):
                averaged[i] += nw * target[i]
        return tuple(averaged)

    @property
    def cms_backend(self) -> "WiringLevel":
        """Deploy-wiring readout: the torch band backend level this core runs at."""

        return self._cms_backend

    @property
    def latest_cms_backend_evidence(self) -> dict | None:
        return self._latest_cms_backend_evidence

    def cms_backend_promotion_readout(self) -> CMSBackendPromotionReadout:
        """Evaluate the code-level SHADOW→ACTIVE gate for the torch band backend.

        M2 (#89 code side). Report-only: aggregates the SHADOW dual-run
        evidence collected so far and evaluates the documented exit
        conditions (enough settled comparisons + forward-parity floor +
        torch update-outcome not worse than pure) and the kill condition
        (torch materially worse → recommend staying on / rolling back to the
        pure baseline). ``promotable=True`` means only that the CODE gate
        passes; the ACTIVE flip stays gated on the ≥500-turn real-trace
        evidence run (#89 Stage 1).
        """

        from volvence_zero.tensor_backend import is_torch_available

        comparisons = self._backend_settled_comparisons
        parity_rate = (
            self._backend_parity_passes / self._backend_parity_checks
            if self._backend_parity_checks
            else 0.0
        )
        pure_mse = (
            self._backend_pure_mse_sum / comparisons if comparisons else 0.0
        )
        torch_mse = (
            self._backend_torch_mse_sum / comparisons if comparisons else 0.0
        )
        min_met = comparisons >= _CMS_PROMOTION_MIN_COMPARISONS
        parity_met = (
            self._backend_parity_checks > 0
            and parity_rate >= _CMS_PROMOTION_PARITY_FLOOR
        )
        not_worse = comparisons > 0 and (
            torch_mse <= pure_mse + _CMS_PROMOTION_MSE_SLACK
        )
        kill = comparisons >= _CMS_PROMOTION_MIN_COMPARISONS and (
            torch_mse >= pure_mse + _CMS_KILL_MSE_DEGRADATION
        )
        torch_ok = is_torch_available()
        promotable = torch_ok and min_met and parity_met and not_worse and not kill
        samples = len(self._anti_forgetting_window)
        absorption_mean = (
            sum(item[0] for item in self._anti_forgetting_window) / samples
            if samples
            else 0.0
        )
        retention_mean = (
            sum(item[1] for item in self._anti_forgetting_window) / samples
            if samples
            else 1.0
        )
        return CMSBackendPromotionReadout(
            backend=self._cms_backend.value,
            torch_available=torch_ok,
            settled_comparisons=comparisons,
            parity_checks=self._backend_parity_checks,
            parity_pass_rate=parity_rate,
            pure_update_mse=pure_mse,
            torch_update_mse=torch_mse,
            min_comparisons_met=min_met,
            parity_floor_met=parity_met,
            torch_not_worse=not_worse,
            kill_condition_met=kill,
            promotable=promotable,
            absorption_window_mean=absorption_mean,
            retention_window_mean=retention_mean,
            anti_forgetting_samples=samples,
            description=(
                f"CMS torch backend promotion gate: backend={self._cms_backend.value} "
                f"comparisons={comparisons}/{_CMS_PROMOTION_MIN_COMPARISONS} "
                f"parity_rate={parity_rate:.3f} pure_mse={pure_mse:.5f} "
                f"torch_mse={torch_mse:.5f} kill={kill} promotable={promotable}"
            ),
        )

    def rollback_last_torch_writeback(self, band_id: str) -> None:
        """Restore a band to its state before the last ACTIVE torch write-back.

        M2 (#89): R15 rollback drill for the ACTIVE path. Restores the FULL
        pre-update export_params tuple (state, momentum, and weights), i.e.
        the band returns to exactly where it was before that update ran.
        Fails loudly when no write-back has been recorded for the band.
        """

        record = self._last_pre_writeback_params.get(band_id)
        if record is None:
            raise KeyError(
                f"no torch write-back recorded for band {band_id!r}; "
                "rollback_last_torch_writeback requires a prior ACTIVE update."
            )
        self._mlp_for_band(band_id).restore_params(record)
        del self._last_pre_writeback_params[band_id]

    def _mlp_for_band(self, band_id: str) -> CMSBandMLP:
        if self._mode != "mlp":
            raise ValueError(
                f"band MLP access requires mode='mlp', core is {self._mode!r}."
            )
        bands = {
            "online-fast": self._online_mlp,
            "session-medium": self._session_mlp,
            "background-slow": self._background_mlp,
        }
        if band_id not in bands:
            raise KeyError(f"unknown CMS band {band_id!r}.")
        return bands[band_id]

    def _decide_band_update(
        self,
        *,
        band_id: str,
        current: tuple[float, ...],
        target: tuple[float, ...],
        pending_signal: tuple[float, ...],
        observations_since_update: int,
        cadence_interval: int,
        source_signal: tuple[float, ...],
    ) -> tuple[LearnedUpdateDecision, tuple[float, ...]]:
        features = self._decision_features(
            current=current,
            target=target,
            pending_signal=pending_signal,
            observations_since_update=observations_since_update,
            cadence_interval=cadence_interval,
            source_signal=source_signal,
        )
        decision = self._update_rule.decide(target_id=band_id, features=features)
        self._latest_band_decisions[band_id] = decision
        self._latest_update_rule_state = self._update_rule.export_state()
        # Record the PE features active during this band's decision so the
        # snapshot can publish band-level Titans gating evidence (see
        # ``CMSBandState.pe_feature_summary``).
        self._record_pe_features_for_band(band_id)
        return decision, features

    def _learn_from_band_update(
        self,
        *,
        decision: LearnedUpdateDecision,
        features: tuple[float, ...],
        before: tuple[float, ...],
        after: tuple[float, ...],
        target: tuple[float, ...],
    ) -> None:
        before_error = self._mean_distance(before, target)
        after_error = self._mean_distance(after, target)
        improvement = before_error - after_error
        stability = _clamp(1.0 - self._mean_distance(before, after))
        self._update_rule.learn(
            features=features,
            decision=decision,
            improvement=improvement,
            stability=stability,
        )
        self._latest_update_rule_state = self._update_rule.export_state()
        self._record_hope_self_modification(
            decision=decision,
            improvement=improvement,
            stability=stability,
        )

    def _record_hope_self_modification(
        self,
        *,
        decision: LearnedUpdateDecision,
        improvement: float,
        stability: float,
    ) -> None:
        updater_state = self._latest_update_rule_state
        reward = updater_state.last_reward
        self._hope_update_count += 1
        self._hope_last_target_id = decision.target_id
        self._hope_generated_learning_rate = _clamp(
            updater_state.base_learning_rate
            * max(decision.step_scale, 0.05)
            * max(decision.write_gate, 0.05)
        )
        self._hope_generated_decay_rate = _clamp(
            self._anti_forgetting
            * (0.35 + decision.slow_mix * 0.45)
            * (1.0 + max(-reward, 0.0) * 0.25)
        )
        self._hope_generated_reset_rate = _clamp(
            decision.reset_mix
            * (0.45 + max(0.0, 1.0 - stability) * 0.35)
        )
        self._hope_last_improvement = improvement
        self._hope_last_stability = _clamp(stability)
        self._hope_last_reward = reward
        self._hope_guarded = decision.guard_applied
        self._hope_guard_reason = decision.guard_reason

    def _decision_for(self, band_id: str) -> LearnedUpdateDecision | None:
        return self._latest_band_decisions.get(band_id)

    # ------------------------------------------------------------------
    # #89 anti-forgetting proxies (report-only)
    # ------------------------------------------------------------------

    def _band_representation(self, band_id: str) -> tuple[float, ...]:
        """Current representation vector for a band, mode-agnostic."""
        if self._mode == "mlp":
            mlp = {
                "online-fast": self._online_mlp,
                "session-medium": self._session_mlp,
                "background-slow": self._background_mlp,
            }[band_id]
            return mlp.representation_vector()
        return {
            "online-fast": self._online_fast,
            "session-medium": self._session_medium,
            "background-slow": self._background_slow,
        }[band_id]

    @staticmethod
    def _normalized_drift(
        before: tuple[float, ...], after: tuple[float, ...]
    ) -> float:
        """L2 movement of a band this turn, normalized to [0, 1].

        Normalized by the pre-update magnitude so the proxy is scale-free;
        a band that barely moves reports ~0, one that fully rewrites reports
        ~1. Pure readout — no effect on learning.
        """
        if not before or not after:
            return 0.0
        delta_sq = sum((a - b) * (a - b) for a, b in zip(after, before, strict=False))
        base_sq = sum(b * b for b in before)
        norm = math.sqrt(delta_sq)
        base = math.sqrt(base_sq)
        if base <= 1e-9:
            # Cold start (zero substrate): any movement is "full" absorption.
            return _clamp(norm)
        return _clamp(norm / base)

    def _update_anti_forgetting_proxies(
        self,
        *,
        online_before: tuple[float, ...],
        background_before: tuple[float, ...],
    ) -> None:
        online_drift = self._normalized_drift(
            online_before, self._band_representation("online-fast")
        )
        background_drift = self._normalized_drift(
            background_before, self._band_representation("background-slow")
        )
        self._last_new_knowledge_absorption = online_drift
        self._last_old_knowledge_retention = _clamp(1.0 - background_drift)
        # M2 (#89): feed the bounded gain-curve window so the promotion
        # readout can report windowed absorption/retention aggregates.
        self._anti_forgetting_window.append(
            (
                _clamp(self._last_new_knowledge_absorption),
                _clamp(self._last_old_knowledge_retention),
            )
        )

    # ------------------------------------------------------------------
    # observe_substrate
    # ------------------------------------------------------------------

    def observe_substrate(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot | None,
        timestamp_ms: int,
        prediction_error: "PredictionErrorSnapshot | None" = None,
    ) -> None:
        self._set_latest_pe_features(prediction_error)
        signal = self._signal_from_substrate(substrate_snapshot)
        self._total_observations += 1
        # #89: capture pre-update band reps to measure this turn's drift.
        _proxy_online_before = self._band_representation("online-fast")
        _proxy_background_before = self._band_representation("background-slow")

        if self._mode == "mlp":
            online_before = self._online_mlp.representation_vector()
            online_decision, online_features = self._decide_band_update(
                band_id="online-fast",
                current=online_before,
                target=signal,
                pending_signal=tuple(0.0 for _ in range(self._dim)),
                observations_since_update=0,
                cadence_interval=1,
                source_signal=signal,
            )
            online_target = self._blend_signal(online_before, signal, rate=online_decision.write_gate)
            self._band_mlp_update(
                band_id="online-fast",
                mlp=self._online_mlp,
                target=online_target,
                decision=online_decision,
            )
            self._learn_from_band_update(
                decision=online_decision,
                features=online_features,
                before=online_before,
                after=self._online_mlp.representation_vector(),
                target=online_target,
            )
            online_signal = self._online_mlp.representation_vector()
            if self._variant is CMSVariant.INDEPENDENT:
                session_signal = signal
                background_signal = signal
            elif self._variant is CMSVariant.NESTED:
                session_signal = online_signal
                background_signal = self._session_mlp.representation_vector()
                self._nested_context_steps += 1
            else:
                session_signal = online_signal
                background_signal = self._session_mlp.representation_vector()

            self._session_pending_signal, self._session_observations_since_update = (
                self._integrate_signal_mlp(
                    band_id="session-medium",
                    mlp=self._session_mlp,
                    current_vector=self._session_mlp.representation_vector(),
                    pending_signal=self._session_pending_signal,
                    observations_since_update=self._session_observations_since_update,
                    signal=session_signal,
                    cadence_interval=self._session_cadence,
                )
            )
            self._background_pending_signal, self._background_observations_since_update = (
                self._integrate_signal_mlp(
                    band_id="background-slow",
                    mlp=self._background_mlp,
                    current_vector=self._background_mlp.representation_vector(),
                    pending_signal=self._background_pending_signal,
                    observations_since_update=self._background_observations_since_update,
                    signal=background_signal,
                    cadence_interval=self._background_cadence,
                )
            )

            if self._variant is CMSVariant.NESTED:
                self._update_nested_meta_targets()

            if self._anti_forgetting > 0:
                self._apply_anti_forgetting_mlp()
        else:
            self._online_fast, self._online_momentum = self._gradient_update(
                band_id="online-fast",
                current=self._online_fast,
                target=signal,
                momentum=self._online_momentum,
                lr=self._online_lr,
                pending_signal=tuple(0.0 for _ in range(self._dim)),
                observations_since_update=0,
                cadence_interval=1,
                source_signal=signal,
            )
            (
                self._session_medium,
                self._session_pending_signal,
                self._session_observations_since_update,
                self._session_momentum,
            ) = self._integrate_signal_gradient(
                band_id="session-medium",
                current_vector=self._session_medium,
                pending_signal=self._session_pending_signal,
                observations_since_update=self._session_observations_since_update,
                momentum=self._session_momentum,
                signal=signal,
                lr=self._session_lr,
                cadence_interval=self._session_cadence,
            )
            (
                self._background_slow,
                self._background_pending_signal,
                self._background_observations_since_update,
                self._background_momentum,
            ) = self._integrate_signal_gradient(
                band_id="background-slow",
                current_vector=self._background_slow,
                pending_signal=self._background_pending_signal,
                observations_since_update=self._background_observations_since_update,
                momentum=self._background_momentum,
                signal=signal,
                lr=self._background_lr,
                cadence_interval=self._background_cadence,
            )
            if self._anti_forgetting > 0:
                self._apply_anti_forgetting()

        self._update_anti_forgetting_proxies(
            online_before=_proxy_online_before,
            background_before=_proxy_background_before,
        )
        if self._latest_cms_backend_evidence is not None:
            # CP-08 (GAP-09): the SHADOW evidence surface carries this
            # observation's retention / absorption / drift readouts next to
            # the torch step + parity fields, so the dual-run comparison the
            # plan asks for is one artifact, not a downstream re-join.
            self._latest_cms_backend_evidence.update(
                {
                    "new_knowledge_absorption": _clamp(
                        self._last_new_knowledge_absorption
                    ),
                    "old_knowledge_retention": _clamp(
                        self._last_old_knowledge_retention
                    ),
                }
            )
        self._last_update_ms = timestamp_ms

    # ------------------------------------------------------------------
    # reflect_lessons
    # ------------------------------------------------------------------

    def reflect_lessons(self, *, lesson_count: int, timestamp_ms: int) -> None:
        lesson_signal = tuple(_clamp(lesson_count / (index + 3)) for index in range(self._dim))
        self.apply_tower_consolidation(
            update=CMSTowerConsolidationUpdate(
                session_signal=lesson_signal,
                background_signal=lesson_signal,
                description=f"lesson-count:{lesson_count}",
            ),
            timestamp_ms=timestamp_ms,
        )

    # ------------------------------------------------------------------
    # observe_encoder_feedback
    # ------------------------------------------------------------------

    def observe_encoder_feedback(
        self,
        *,
        encoder_signal: tuple[float, ...],
        timestamp_ms: int,
        prediction_error: "PredictionErrorSnapshot | None" = None,
    ) -> None:
        """Accept metacontroller encoder output as an additional observation."""
        self._set_latest_pe_features(prediction_error)
        if len(encoder_signal) != self._dim:
            projected: tuple[float, ...] = (
                tuple(encoder_signal[i % len(encoder_signal)] for i in range(self._dim))
                if encoder_signal
                else tuple(0.0 for _ in range(self._dim))
            )
        else:
            projected = encoder_signal

        if self._mode == "mlp":
            online_before = self._online_mlp.representation_vector()
            online_decision, online_features = self._decide_band_update(
                band_id="online-fast",
                current=online_before,
                target=projected,
                pending_signal=tuple(0.0 for _ in range(self._dim)),
                observations_since_update=0,
                cadence_interval=1,
                source_signal=projected,
            )
            online_target = self._blend_signal(online_before, projected, rate=online_decision.write_gate)
            self._band_mlp_update(
                band_id="online-fast",
                mlp=self._online_mlp,
                target=online_target,
                decision=online_decision,
            )
            self._learn_from_band_update(
                decision=online_decision,
                features=online_features,
                before=online_before,
                after=self._online_mlp.representation_vector(),
                target=online_target,
            )
            session_signal = (
                projected
                if self._variant is CMSVariant.INDEPENDENT
                else self._online_mlp.representation_vector()
            )
            self._session_pending_signal, self._session_observations_since_update = (
                self._integrate_signal_mlp(
                    band_id="session-medium",
                    mlp=self._session_mlp,
                    current_vector=self._session_mlp.representation_vector(),
                    pending_signal=self._session_pending_signal,
                    observations_since_update=self._session_observations_since_update,
                    signal=session_signal,
                    cadence_interval=self._session_cadence,
                )
            )
        else:
            self._online_fast, self._online_momentum = self._gradient_update(
                band_id="online-fast",
                current=self._online_fast,
                target=projected,
                momentum=self._online_momentum,
                lr=self._online_lr * 0.3,
                pending_signal=tuple(0.0 for _ in range(self._dim)),
                observations_since_update=0,
                cadence_interval=1,
                source_signal=projected,
            )
            (
                self._session_medium,
                self._session_pending_signal,
                self._session_observations_since_update,
                self._session_momentum,
            ) = self._integrate_signal_gradient(
                band_id="session-medium",
                current_vector=self._session_medium,
                pending_signal=self._session_pending_signal,
                observations_since_update=self._session_observations_since_update,
                momentum=self._session_momentum,
                signal=projected,
                lr=self._session_lr * 0.33,
                cadence_interval=self._session_cadence,
            )

        self._last_update_ms = timestamp_ms

    def observe_fast_memory_signal(
        self,
        *,
        signal: tuple[float, ...],
        timestamp_ms: int,
        prediction_error: "PredictionErrorSnapshot | None" = None,
    ) -> None:
        """Accept substrate fast-memory signal as another learned update source."""
        self.observe_encoder_feedback(
            encoder_signal=signal,
            timestamp_ms=timestamp_ms,
            prediction_error=prediction_error,
        )

    # ------------------------------------------------------------------
    # observe_family_signal (MLP mode only)
    # ------------------------------------------------------------------

    def observe_family_signal(
        self,
        *,
        family_centroid: tuple[float, ...],
        family_stability: float,
        timestamp_ms: int,
    ) -> None:
        """Accept action-family observation to enrich session-medium band."""
        if self._mode != "mlp":
            return
        if len(family_centroid) != self._dim:
            projected = tuple(
                family_centroid[i % len(family_centroid)] if family_centroid else 0.0
                for i in range(self._dim)
            )
        else:
            projected = family_centroid
        weighted = tuple(_clamp(projected[i] * _clamp(family_stability)) for i in range(self._dim))
        self._session_pending_signal, self._session_observations_since_update = (
            self._integrate_signal_mlp(
                band_id="session-medium",
                mlp=self._session_mlp,
                current_vector=self._session_mlp.representation_vector(),
                pending_signal=self._session_pending_signal,
                observations_since_update=self._session_observations_since_update,
                signal=weighted,
                cadence_interval=self._session_cadence,
            )
        )
        self._last_update_ms = timestamp_ms

    # ------------------------------------------------------------------
    # reset_context (nested variant)
    # ------------------------------------------------------------------

    def reset_context(self) -> None:
        """Re-initialize fast bands from meta-learned targets (nested CMS).

        In nested mode each slow band meta-learns the *ideal starting
        point* for its faster neighbor.  ``reset_context`` re-initializes
        the fast bands from these learned targets — not by copying the
        slow band's current state, but from targets the slow band has
        been optimizing toward via ``_update_nested_meta_targets``.

        In non-nested modes this is a no-op.
        """
        if self._variant is not CMSVariant.NESTED or self._mode != "mlp":
            return
        session_decision, _ = self._decide_band_update(
            band_id="nested-session-reset",
            current=self._session_mlp.representation_vector(),
            target=self._nested_session_init_target,
            pending_signal=self._session_pending_signal,
            observations_since_update=self._session_observations_since_update,
            cadence_interval=self._session_cadence,
            source_signal=self._nested_session_init_target,
        )
        online_decision, _ = self._decide_band_update(
            band_id="nested-online-reset",
            current=self._online_mlp.representation_vector(),
            target=self._nested_online_init_target,
            pending_signal=tuple(0.0 for _ in range(self._dim)),
            observations_since_update=0,
            cadence_interval=1,
            source_signal=self._nested_online_init_target,
        )
        self._session_mlp.load_representation(
            self._blend_signal(
                self._session_mlp.representation_vector(),
                self._nested_session_init_target,
                rate=session_decision.reset_mix,
            )
        )
        self._online_mlp.load_representation(
            self._blend_signal(
                self._online_mlp.representation_vector(),
                self._nested_online_init_target,
                rate=online_decision.reset_mix,
            )
        )
        self._session_observations_since_update = 0
        self._session_pending_signal = tuple(0.0 for _ in range(self._dim))
        self._nested_context_steps = 0

    def nested_reset_targets(self) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        if self._variant is not CMSVariant.NESTED or self._mode != "mlp":
            return None
        return (self._nested_online_init_target, self._nested_session_init_target)

    def _update_nested_meta_targets(self) -> None:
        """Meta-learn initialization targets for faster bands.

        The slow band observes what the session band converged to and
        moves its meta-target toward that endpoint.  Similarly the
        session band's converged state teaches the online band's target.

        This implements NL Appendix A.5's nested CMS: the i-th layer
        meta-learns the initial state for layer i+1.
        """
        session_converged = self._session_mlp.representation_vector()
        session_decision, session_features = self._decide_band_update(
            band_id="nested-session-target",
            current=self._nested_session_init_target,
            target=session_converged,
            pending_signal=self._session_pending_signal,
            observations_since_update=self._session_observations_since_update,
            cadence_interval=self._session_cadence,
            source_signal=session_converged,
        )
        session_before = self._nested_session_init_target
        self._nested_session_init_target = tuple(
            _clamp(
                self._nested_session_init_target[i]
                + session_decision.step_scale
                * session_decision.write_gate
                * (session_converged[i] - self._nested_session_init_target[i])
            )
            for i in range(self._dim)
        )
        online_converged = self._online_mlp.representation_vector()
        online_decision, online_features = self._decide_band_update(
            band_id="nested-online-target",
            current=self._nested_online_init_target,
            target=online_converged,
            pending_signal=tuple(0.0 for _ in range(self._dim)),
            observations_since_update=0,
            cadence_interval=1,
            source_signal=online_converged,
        )
        online_before = self._nested_online_init_target
        self._nested_online_init_target = tuple(
            _clamp(
                self._nested_online_init_target[i]
                + online_decision.step_scale
                * online_decision.write_gate
                * (online_converged[i] - self._nested_online_init_target[i])
            )
            for i in range(self._dim)
        )
        self._learn_from_band_update(
            decision=session_decision,
            features=session_features,
            before=session_before,
            after=self._nested_session_init_target,
            target=session_converged,
        )
        self._learn_from_band_update(
            decision=online_decision,
            features=online_features,
            before=online_before,
            after=self._nested_online_init_target,
            target=online_converged,
        )

    # ------------------------------------------------------------------
    # export / restore / snapshot
    # ------------------------------------------------------------------

    def export_state(self) -> CMSCheckpointState:
        replay_window_sizes = tuple(
            (band_id, int(self._replay_window_sizes.get(band_id, 1)))
            for band_id in _REPLAY_DEFAULT_K
        )
        if self._mode == "mlp":
            return CMSCheckpointState(
                online_fast=self._online_mlp.representation_vector(),
                session_medium=self._session_mlp.representation_vector(),
                background_slow=self._background_mlp.representation_vector(),
                last_update_ms=self._last_update_ms,
                total_observations=self._total_observations,
                total_reflections=self._total_reflections,
                session_observations_since_update=self._session_observations_since_update,
                background_observations_since_update=self._background_observations_since_update,
                session_pending_signal=self._session_pending_signal,
                background_pending_signal=self._background_pending_signal,
                nested_session_init_target=self._nested_session_init_target,
                nested_online_init_target=self._nested_online_init_target,
                tower_meta_levels=self._export_tower_meta_levels(),
                mode="mlp",
                mlp_params=(
                    self._online_mlp.export_params(),
                    self._session_mlp.export_params(),
                    self._background_mlp.export_params(),
                ),
                update_rule_state=self._latest_update_rule_state,
                hope_self_modification_state=self._hope_state(),
                atlas_replay_active=self._atlas_replay_active,
                titans_pe_gate_active=self._pe_features_enabled,
                replay_window_sizes=replay_window_sizes,
            )
        return CMSCheckpointState(
            online_fast=self._online_fast,
            session_medium=self._session_medium,
            background_slow=self._background_slow,
            last_update_ms=self._last_update_ms,
            total_observations=self._total_observations,
            total_reflections=self._total_reflections,
            session_observations_since_update=self._session_observations_since_update,
            background_observations_since_update=self._background_observations_since_update,
            session_pending_signal=self._session_pending_signal,
            background_pending_signal=self._background_pending_signal,
            tower_meta_levels=self._export_tower_meta_levels(),
            update_rule_state=self._latest_update_rule_state,
            hope_self_modification_state=self._hope_state(),
            atlas_replay_active=self._atlas_replay_active,
            titans_pe_gate_active=self._pe_features_enabled,
            replay_window_sizes=replay_window_sizes,
        )

    def restore_state(self, state: CMSCheckpointState) -> None:
        self._last_update_ms = state.last_update_ms
        self._total_observations = state.total_observations
        self._total_reflections = state.total_reflections
        self._session_observations_since_update = state.session_observations_since_update
        self._background_observations_since_update = state.background_observations_since_update
        self._session_pending_signal = state.session_pending_signal
        self._background_pending_signal = state.background_pending_signal
        if state.update_rule_state is not None:
            self._update_rule.restore_state(state.update_rule_state)
            self._latest_update_rule_state = state.update_rule_state
            # ATLAS / Titans uplift: when restoring a legacy
            # (feature_version=1) state into a PE-aware CMS, the rule's
            # input_projection columns reserved for PE features still carry
            # weights trained for non-PE semantics (modulo-extended legacy
            # features). Reset those columns so the rule starts clean on
            # the PE-gated path. See docs/specs/cms-atlas-titans-uplift.md §5.
            if (
                self._pe_features_enabled
                and state.update_rule_state.feature_version <= 1
                and self._update_rule.feature_dim == LEARNED_UPDATE_PE_AWARE_FEATURE_DIM
            ):
                self._zero_pad_pe_columns_after_restore()
        if state.hope_self_modification_state is not None:
            self._restore_hope_state(state.hope_self_modification_state)
        else:
            self._restore_hope_state(
                CMSHopeSelfModificationState(
                    enabled=True,
                    update_count=0,
                    last_target_id="",
                    generated_learning_rate=0.0,
                    generated_decay_rate=0.0,
                    generated_reset_rate=0.0,
                    last_improvement=0.0,
                    last_stability=0.0,
                    last_reward=0.0,
                    guarded=False,
                )
            )

        if self._mode == "mlp":
            if state.mode == "mlp" and state.mlp_params:
                self._online_mlp.restore_params(state.mlp_params[0])
                self._session_mlp.restore_params(state.mlp_params[1])
                self._background_mlp.restore_params(state.mlp_params[2])
            else:
                self._online_mlp.load_representation(state.online_fast)
                self._session_mlp.load_representation(state.session_medium)
                self._background_mlp.load_representation(state.background_slow)
            if state.nested_session_init_target:
                self._nested_session_init_target = state.nested_session_init_target
            if state.nested_online_init_target:
                self._nested_online_init_target = state.nested_online_init_target
            self._restore_tower_meta_levels(state.tower_meta_levels)
        elif self._mode == "vector":
            self._online_fast = state.online_fast
            self._session_medium = state.session_medium
            self._background_slow = state.background_slow

    def snapshot(self) -> CMSState:
        if self._mode == "mlp":
            return self._snapshot_mlp()
        return self._snapshot_vector()

    def _snapshot_state_extras(self) -> dict[str, object]:
        """Top-level CMSState fields added by the ATLAS / Titans uplift."""
        return {
            "atlas_replay_active": self._atlas_replay_active,
            "titans_pe_gate_active": self._pe_features_enabled,
            "replay_window_sizes": tuple(
                (band_id, int(self._replay_window_sizes.get(band_id, 1)))
                for band_id in _REPLAY_DEFAULT_K
            ),
            # #89 anti-forgetting proxies (report-only).
            "new_knowledge_absorption": _clamp(self._last_new_knowledge_absorption),
            "old_knowledge_retention": _clamp(self._last_old_knowledge_retention),
        }

    def _snapshot_band_extras(self, band_id: str) -> dict[str, object]:
        """Per-band CMSBandState fields added by the ATLAS / Titans uplift."""
        return {
            "replay_window_size": int(self._latest_replay_window_size.get(band_id, 0)),
            "pe_feature_summary": tuple(
                self._latest_pe_features_by_band.get(band_id, ())
            ),
        }

    def _snapshot_vector(self) -> CMSState:
        tower_profile = self._build_tower_profile(
            online_vector=self._online_fast,
            session_vector=self._session_medium,
            background_vector=self._background_slow,
        )
        return CMSState(
            online_fast=CMSBandState(
                name="online-fast",
                vector=self._online_fast,
                last_update_ms=self._last_update_ms,
                cadence_interval=1,
                observations_since_update=0,
                pending_signal=tuple(0.0 for _ in range(self._dim)),
                learning_rate=self._online_lr,
                effective_learning_rate=self._online_lr
                * (self._decision_for("online-fast").step_scale if self._decision_for("online-fast") else 0.0),
                momentum=self._online_momentum,
                anti_forgetting_strength=self._anti_forgetting,
                update_gate=self._decision_for("online-fast").write_gate if self._decision_for("online-fast") else 0.0,
                slow_mix=self._decision_for("online-fast").slow_mix if self._decision_for("online-fast") else 0.0,
                reset_mix=self._decision_for("online-fast").reset_mix if self._decision_for("online-fast") else 0.0,
                confidence=self._decision_for("online-fast").confidence if self._decision_for("online-fast") else 0.0,
                update_summary=self._decision_for("online-fast").description if self._decision_for("online-fast") else "",
                **self._snapshot_band_extras("online-fast"),
            ),
            session_medium=CMSBandState(
                name="session-medium",
                vector=self._session_medium,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._session_cadence,
                observations_since_update=self._session_observations_since_update,
                pending_signal=self._session_pending_signal,
                learning_rate=self._session_lr,
                effective_learning_rate=self._session_lr
                * (self._decision_for("session-medium").step_scale if self._decision_for("session-medium") else 0.0),
                momentum=self._session_momentum,
                anti_forgetting_strength=self._anti_forgetting,
                update_gate=self._decision_for("session-medium").write_gate if self._decision_for("session-medium") else 0.0,
                slow_mix=self._decision_for("session-medium").slow_mix if self._decision_for("session-medium") else 0.0,
                reset_mix=self._decision_for("session-medium").reset_mix if self._decision_for("session-medium") else 0.0,
                confidence=self._decision_for("session-medium").confidence if self._decision_for("session-medium") else 0.0,
                update_summary=self._decision_for("session-medium").description if self._decision_for("session-medium") else "",
                **self._snapshot_band_extras("session-medium"),
            ),
            background_slow=CMSBandState(
                name="background-slow",
                vector=self._background_slow,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._background_cadence,
                observations_since_update=self._background_observations_since_update,
                pending_signal=self._background_pending_signal,
                learning_rate=self._background_lr,
                effective_learning_rate=self._background_lr
                * (self._decision_for("background-slow").step_scale if self._decision_for("background-slow") else 0.0),
                momentum=self._background_momentum,
                anti_forgetting_strength=self._anti_forgetting,
                update_gate=self._decision_for("background-slow").write_gate if self._decision_for("background-slow") else 0.0,
                slow_mix=self._decision_for("background-slow").slow_mix if self._decision_for("background-slow") else 0.0,
                reset_mix=self._decision_for("background-slow").reset_mix if self._decision_for("background-slow") else 0.0,
                confidence=self._decision_for("background-slow").confidence if self._decision_for("background-slow") else 0.0,
                update_summary=self._decision_for("background-slow").description if self._decision_for("background-slow") else "",
                **self._snapshot_band_extras("background-slow"),
            ),
            total_observations=self._total_observations,
            total_reflections=self._total_reflections,
            description=(
                f"CMS core dim={self._dim} with gradient updates, "
                f"online_lr={self._online_lr}, session_lr={self._session_lr}, "
                f"bg_lr={self._background_lr}, anti_forgetting={self._anti_forgetting}."
            ),
            tower_profile=tower_profile,
            tower_depth=len(tower_profile.levels),
            continuum_profile=self._build_continuum_profile(tower_profile),
            update_rule_state=self._latest_update_rule_state,
            hope_self_modification_state=self._hope_state(),
            **self._snapshot_state_extras(),
        )

    def _snapshot_mlp(self) -> CMSState:
        online_rep = self._online_mlp.representation_vector()
        session_rep = self._session_mlp.representation_vector()
        bg_rep = self._background_mlp.representation_vector()
        pc = self._online_mlp.parameter_count()
        tower_profile = self._build_tower_profile(
            online_vector=online_rep,
            session_vector=session_rep,
            background_vector=bg_rep,
        )
        return CMSState(
            online_fast=CMSBandState(
                name="online-fast",
                vector=online_rep,
                last_update_ms=self._last_update_ms,
                cadence_interval=1,
                observations_since_update=0,
                pending_signal=tuple(0.0 for _ in range(self._dim)),
                learning_rate=self._online_lr,
                effective_learning_rate=self._online_lr
                * (self._decision_for("online-fast").step_scale if self._decision_for("online-fast") else 0.0),
                momentum=tuple(self._online_mlp._state_momentum),
                anti_forgetting_strength=self._anti_forgetting,
                update_gate=self._decision_for("online-fast").write_gate if self._decision_for("online-fast") else 0.0,
                slow_mix=self._decision_for("online-fast").slow_mix if self._decision_for("online-fast") else 0.0,
                reset_mix=self._decision_for("online-fast").reset_mix if self._decision_for("online-fast") else 0.0,
                confidence=self._decision_for("online-fast").confidence if self._decision_for("online-fast") else 0.0,
                update_summary=self._decision_for("online-fast").description if self._decision_for("online-fast") else "",
                mode="mlp",
                mlp_param_count=pc,
                **self._snapshot_band_extras("online-fast"),
            ),
            session_medium=CMSBandState(
                name="session-medium",
                vector=session_rep,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._session_cadence,
                observations_since_update=self._session_observations_since_update,
                pending_signal=self._session_pending_signal,
                learning_rate=self._session_lr,
                effective_learning_rate=self._session_lr
                * (self._decision_for("session-medium").step_scale if self._decision_for("session-medium") else 0.0),
                momentum=tuple(self._session_mlp._state_momentum),
                anti_forgetting_strength=self._anti_forgetting,
                update_gate=self._decision_for("session-medium").write_gate if self._decision_for("session-medium") else 0.0,
                slow_mix=self._decision_for("session-medium").slow_mix if self._decision_for("session-medium") else 0.0,
                reset_mix=self._decision_for("session-medium").reset_mix if self._decision_for("session-medium") else 0.0,
                confidence=self._decision_for("session-medium").confidence if self._decision_for("session-medium") else 0.0,
                update_summary=self._decision_for("session-medium").description if self._decision_for("session-medium") else "",
                mode="mlp",
                mlp_param_count=pc,
                **self._snapshot_band_extras("session-medium"),
            ),
            background_slow=CMSBandState(
                name="background-slow",
                vector=bg_rep,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._background_cadence,
                observations_since_update=self._background_observations_since_update,
                pending_signal=self._background_pending_signal,
                learning_rate=self._background_lr,
                effective_learning_rate=self._background_lr
                * (self._decision_for("background-slow").step_scale if self._decision_for("background-slow") else 0.0),
                momentum=tuple(self._background_mlp._state_momentum),
                anti_forgetting_strength=self._anti_forgetting,
                update_gate=self._decision_for("background-slow").write_gate if self._decision_for("background-slow") else 0.0,
                slow_mix=self._decision_for("background-slow").slow_mix if self._decision_for("background-slow") else 0.0,
                reset_mix=self._decision_for("background-slow").reset_mix if self._decision_for("background-slow") else 0.0,
                confidence=self._decision_for("background-slow").confidence if self._decision_for("background-slow") else 0.0,
                update_summary=self._decision_for("background-slow").description if self._decision_for("background-slow") else "",
                mode="mlp",
                mlp_param_count=pc,
                **self._snapshot_band_extras("background-slow"),
            ),
            total_observations=self._total_observations,
            total_reflections=self._total_reflections,
            description=(
                f"CMS core mode=mlp d_in={self._dim} d_hidden={self._d_hidden} "
                f"variant={self._variant.value} params/band={pc}, "
                f"online_lr={self._online_lr}, session_lr={self._session_lr}, "
                f"bg_lr={self._background_lr}, anti_forgetting={self._anti_forgetting}."
            ),
            variant=self._variant.value,
            tower_profile=tower_profile,
            tower_depth=len(tower_profile.levels),
            continuum_profile=self._build_continuum_profile(tower_profile),
            update_rule_state=self._latest_update_rule_state,
            hope_self_modification_state=self._hope_state(),
            **self._snapshot_state_extras(),
        )

    def _hope_state(self) -> CMSHopeSelfModificationState:
        return CMSHopeSelfModificationState(
            enabled=True,
            update_count=self._hope_update_count,
            last_target_id=self._hope_last_target_id,
            generated_learning_rate=self._hope_generated_learning_rate,
            generated_decay_rate=self._hope_generated_decay_rate,
            generated_reset_rate=self._hope_generated_reset_rate,
            last_improvement=self._hope_last_improvement,
            last_stability=self._hope_last_stability,
            last_reward=self._hope_last_reward,
            guarded=self._hope_guarded,
            guard_reason=self._hope_guard_reason,
            description=(
                "Tiny Hope owner-side self-modification state: "
                f"target={self._hope_last_target_id or 'none'} "
                f"updates={self._hope_update_count} "
                f"generated_lr={self._hope_generated_learning_rate:.4f} "
                f"decay={self._hope_generated_decay_rate:.4f} "
                f"reset={self._hope_generated_reset_rate:.4f} "
                f"reward={self._hope_last_reward:.3f} "
                f"guard={self._hope_guard_reason or 'clear'}."
            ),
        )

    def _restore_hope_state(self, state: CMSHopeSelfModificationState) -> None:
        self._hope_update_count = state.update_count
        self._hope_last_target_id = state.last_target_id
        self._hope_generated_learning_rate = state.generated_learning_rate
        self._hope_generated_decay_rate = state.generated_decay_rate
        self._hope_generated_reset_rate = state.generated_reset_rate
        self._hope_last_improvement = state.last_improvement
        self._hope_last_stability = state.last_stability
        self._hope_last_reward = state.last_reward
        self._hope_guarded = state.guarded
        self._hope_guard_reason = state.guard_reason

    def apply_tower_consolidation(
        self,
        *,
        update: CMSTowerConsolidationUpdate,
        timestamp_ms: int,
    ) -> tuple[str, ...]:
        self._total_reflections += 1
        applied: list[str] = []
        online_signal = self._align_signal_dim(update.online_signal)
        session_signal = self._align_signal_dim(update.session_signal)
        background_signal = self._align_signal_dim(update.background_signal)
        nested_transfer_pressure = max(
            self._mean_distance(background_signal, online_signal),
            self._mean_distance(session_signal, online_signal),
        )

        if self._mode == "mlp":
            if any(value > 0.0 for value in online_signal):
                online_before = self._online_mlp.representation_vector()
                online_decision, online_features = self._decide_band_update(
                    band_id="tower-online",
                    current=online_before,
                    target=online_signal,
                    pending_signal=tuple(0.0 for _ in range(self._dim)),
                    observations_since_update=0,
                    cadence_interval=1,
                    source_signal=online_signal,
                )
                online_guidance = self._blend_signal(
                    online_signal,
                    self._session_mlp.representation_vector(),
                    rate=0.16 + online_decision.slow_mix * 0.24,
                )
                if self._variant is CMSVariant.NESTED:
                    online_guidance = self._blend_signal(
                        online_guidance,
                        self._nested_online_init_target,
                        rate=0.08 + online_decision.reset_mix * 0.20,
                    )
                online_target = self._blend_signal(
                    online_before,
                    online_guidance,
                    rate=max(0.05, online_decision.write_gate),
                )
                self._online_mlp.update(
                    target=online_target,
                    lr_scale=max(0.05, online_decision.step_scale),
                    momentum_gate=online_decision.momentum_gate,
                )
                self._learn_from_band_update(
                    decision=online_decision,
                    features=online_features,
                    before=online_before,
                    after=self._online_mlp.representation_vector(),
                    target=online_target,
                )
                applied.append("tower-consolidation:online")
            if any(value > 0.0 for value in session_signal):
                session_before = self._session_mlp.representation_vector()
                session_decision, session_features = self._decide_band_update(
                    band_id="tower-session",
                    current=session_before,
                    target=session_signal,
                    pending_signal=self._session_pending_signal,
                    observations_since_update=self._session_observations_since_update,
                    cadence_interval=self._session_cadence,
                    source_signal=session_signal,
                )
                session_guidance = self._blend_signal(
                    session_signal,
                    self._background_mlp.representation_vector(),
                    rate=0.18 + session_decision.slow_mix * 0.26,
                )
                if self._variant is CMSVariant.NESTED:
                    session_guidance = self._blend_signal(
                        session_guidance,
                        self._nested_session_init_target,
                        rate=0.10 + session_decision.reset_mix * 0.22,
                    )
                session_target = self._blend_signal(
                    session_before,
                    session_guidance,
                    rate=max(0.05, session_decision.write_gate),
                )
                self._session_mlp.update(
                    target=session_target,
                    lr_scale=max(0.05, session_decision.step_scale),
                    momentum_gate=session_decision.momentum_gate,
                )
                self._learn_from_band_update(
                    decision=session_decision,
                    features=session_features,
                    before=session_before,
                    after=self._session_mlp.representation_vector(),
                    target=session_target,
                )
                applied.append("tower-consolidation:session")
            if any(value > 0.0 for value in background_signal):
                background_before = self._background_mlp.representation_vector()
                background_decision, background_features = self._decide_band_update(
                    band_id="tower-background",
                    current=background_before,
                    target=background_signal,
                    pending_signal=self._background_pending_signal,
                    observations_since_update=self._background_observations_since_update,
                    cadence_interval=self._background_cadence,
                    source_signal=background_signal,
                )
                background_guidance = self._blend_signal(
                    background_signal,
                    self._session_mlp.representation_vector(),
                    rate=0.14 + background_decision.slow_mix * 0.18,
                )
                background_target = self._blend_signal(
                    background_before,
                    background_guidance,
                    rate=max(0.05, background_decision.write_gate),
                )
                self._background_mlp.update(
                    target=background_target,
                    lr_scale=max(0.05, background_decision.step_scale),
                    momentum_gate=background_decision.momentum_gate,
                )
                self._learn_from_band_update(
                    decision=background_decision,
                    features=background_features,
                    before=background_before,
                    after=self._background_mlp.representation_vector(),
                    target=background_target,
                )
                applied.append("tower-consolidation:background")
            if update.decay_pressure > 0.0:
                pressure = _clamp(update.decay_pressure)
                background_rep = self._background_mlp.representation_vector()
                self._session_mlp.update(
                    target=self._blend_signal(
                        self._session_mlp.representation_vector(),
                        background_rep,
                        rate=pressure * max(0.05, self._decision_for("anti-forgetting-session").slow_mix)
                        if self._decision_for("anti-forgetting-session")
                        else pressure * 0.32,
                    )
                )
                self._online_mlp.update(
                    target=self._blend_signal(
                        self._online_mlp.representation_vector(),
                        self._session_mlp.representation_vector(),
                        rate=pressure * max(0.05, self._decision_for("anti-forgetting-online").slow_mix)
                        if self._decision_for("anti-forgetting-online")
                        else pressure * 0.24,
                    )
                )
                applied.append("tower-consolidation:decay-pressure")
            if self._variant is CMSVariant.NESTED:
                self._update_nested_meta_targets()
                applied.append("tower-consolidation:meta-targets")
        else:
            if any(value > 0.0 for value in online_signal):
                self._online_fast, self._online_momentum = self._gradient_update(
                    band_id="tower-online",
                    current=self._online_fast,
                    target=self._blend_signal(
                        self._online_fast,
                        self._blend_signal(
                            online_signal,
                            self._session_medium,
                            rate=0.16
                            + (self._decision_for("tower-online").slow_mix if self._decision_for("tower-online") else 0.0)
                            * 0.24,
                        ),
                        rate=max(
                            0.05,
                            self._decision_for("tower-online").write_gate if self._decision_for("tower-online") else 0.55,
                        ),
                    ),
                    momentum=self._online_momentum,
                    lr=self._online_lr,
                    pending_signal=tuple(0.0 for _ in range(self._dim)),
                    observations_since_update=0,
                    cadence_interval=1,
                    source_signal=online_signal,
                )
                applied.append("tower-consolidation:online")
            if any(value > 0.0 for value in session_signal):
                self._session_medium, self._session_momentum = self._gradient_update(
                    band_id="tower-session",
                    current=self._session_medium,
                    target=self._blend_signal(
                        self._session_medium,
                        self._blend_signal(
                            session_signal,
                            self._background_slow,
                            rate=0.18
                            + (self._decision_for("tower-session").slow_mix if self._decision_for("tower-session") else 0.0)
                            * 0.26,
                        ),
                        rate=max(
                            0.05,
                            self._decision_for("tower-session").write_gate if self._decision_for("tower-session") else 0.62,
                        ),
                    ),
                    momentum=self._session_momentum,
                    lr=self._session_lr,
                    pending_signal=self._session_pending_signal,
                    observations_since_update=self._session_observations_since_update,
                    cadence_interval=self._session_cadence,
                    source_signal=session_signal,
                )
                applied.append("tower-consolidation:session")
            if any(value > 0.0 for value in background_signal):
                self._background_slow, self._background_momentum = self._gradient_update(
                    band_id="tower-background",
                    current=self._background_slow,
                    target=self._blend_signal(
                        self._background_slow,
                        self._blend_signal(
                            background_signal,
                            self._session_medium,
                            rate=0.14
                            + (self._decision_for("tower-background").slow_mix if self._decision_for("tower-background") else 0.0)
                            * 0.18,
                        ),
                        rate=max(
                            0.05,
                            self._decision_for("tower-background").write_gate
                            if self._decision_for("tower-background")
                            else 0.7,
                        ),
                    ),
                    momentum=self._background_momentum,
                    lr=self._background_lr,
                    pending_signal=self._background_pending_signal,
                    observations_since_update=self._background_observations_since_update,
                    cadence_interval=self._background_cadence,
                    source_signal=background_signal,
                )
                applied.append("tower-consolidation:background")
            if update.decay_pressure > 0.0:
                pressure = _clamp(update.decay_pressure)
                self._session_medium = self._blend_signal(
                    self._session_medium,
                    self._background_slow,
                    rate=pressure * 0.32,
                )
                self._online_fast = self._blend_signal(
                    self._online_fast,
                    self._session_medium,
                    rate=pressure * 0.22,
                )
                applied.append("tower-consolidation:decay-pressure")

        if (
            self._mode == "mlp"
            and self._variant is CMSVariant.NESTED
            and (update.reset_fast_context or nested_transfer_pressure > 0.10)
        ):
            self.reset_context()
            applied.append("tower-consolidation:nested-reset")
        self._last_update_ms = timestamp_ms
        return tuple(applied)

    # ------------------------------------------------------------------
    # internal helpers — signal extraction
    # ------------------------------------------------------------------

    def _signal_from_substrate(self, substrate_snapshot: SubstrateSnapshot | None) -> tuple[float, ...]:
        return build_runtime_backbone_evidence(
            substrate_snapshot=substrate_snapshot,
            dim=self._dim,
        ).signal

    def _align_signal_dim(self, signal: tuple[float, ...]) -> tuple[float, ...]:
        if len(signal) == self._dim:
            return signal
        if not signal:
            return tuple(0.0 for _ in range(self._dim))
        return tuple(signal[index % len(signal)] for index in range(self._dim))

    def _blend_signal(
        self,
        current: tuple[float, ...],
        incoming: tuple[float, ...],
        *,
        rate: float,
    ) -> tuple[float, ...]:
        current = self._align_signal_dim(current)
        incoming = self._align_signal_dim(incoming)
        return tuple(
            _clamp(current[index] * (1.0 - rate) + incoming[index] * rate)
            for index in range(self._dim)
        )

    def _export_tower_meta_levels(self) -> tuple[tuple[str, tuple[float, ...]], ...]:
        if self._variant is not CMSVariant.NESTED or self._mode != "mlp":
            return ()
        return (
            ("nested-online-prior", self._nested_online_init_target),
            ("nested-session-prior", self._nested_session_init_target),
        )

    def _restore_tower_meta_levels(
        self,
        tower_meta_levels: tuple[tuple[str, tuple[float, ...]], ...],
    ) -> None:
        for level_id, vector in tower_meta_levels:
            if level_id == "nested-online-prior" and vector:
                self._nested_online_init_target = vector
            elif level_id == "nested-session-prior" and vector:
                self._nested_session_init_target = vector

    def _build_tower_profile(
        self,
        *,
        online_vector: tuple[float, ...],
        session_vector: tuple[float, ...],
        background_vector: tuple[float, ...],
    ) -> CMSTowerProfile:
        levels = [
            CMSTowerLevelState(
                level_id="online-fast",
                role="fast-band",
                vector=online_vector,
                cadence_interval=1,
                description="Highest-frequency online adaptation band.",
            ),
            CMSTowerLevelState(
                level_id="session-medium",
                role="session-band",
                vector=session_vector,
                cadence_interval=self._session_cadence,
                source_level_ids=("online-fast",),
                description="Mid-frequency session aggregation band.",
            ),
            CMSTowerLevelState(
                level_id="background-slow",
                role="slow-band",
                vector=background_vector,
                cadence_interval=self._background_cadence,
                source_level_ids=("session-medium",),
                description="Low-frequency background consolidation band.",
            ),
        ]
        weighted_levels: list[tuple[tuple[float, ...], float]] = [
            (online_vector, 0.42),
            (session_vector, 0.28),
            (background_vector, 0.2),
        ]
        if self._variant is CMSVariant.NESTED and self._mode == "mlp":
            levels.extend(
                (
                    CMSTowerLevelState(
                        level_id="nested-online-prior",
                        role="meta-init",
                        vector=self._nested_online_init_target,
                        cadence_interval=self._session_cadence,
                        source_level_ids=("session-medium",),
                        description="Meta-learned prior used to seed online-fast on context reset.",
                    ),
                    CMSTowerLevelState(
                        level_id="nested-session-prior",
                        role="meta-init",
                        vector=self._nested_session_init_target,
                        cadence_interval=self._background_cadence,
                        source_level_ids=("background-slow",),
                        description="Meta-learned prior used to seed session-medium on context reset.",
                    ),
                )
            )
            weighted_levels.extend(
                (
                    (self._nested_online_init_target, 0.06),
                    (self._nested_session_init_target, 0.04),
                )
            )
        total_weight = sum(weight for _, weight in weighted_levels)
        readout_vector = tuple(
            _clamp(
                sum(signal[index] * weight for signal, weight in weighted_levels) / max(total_weight, 1e-6)
            )
            for index in range(self._dim)
        )
        levels.append(
            CMSTowerLevelState(
                level_id="tower-readout",
                role="readout",
                vector=readout_vector,
                cadence_interval=max(self._background_cadence, 1),
                source_level_ids=tuple(level.level_id for level in levels),
                description="Owner-side associative readout over the current nested memory tower.",
            )
        )
        return CMSTowerProfile(
            profile_id=f"{self._mode}:{self._variant.value}:depth{len(levels)}",
            levels=tuple(levels),
            readout_vector=readout_vector,
            description=(
                f"Nested memory tower with {len(levels)} levels, "
                f"mode={self._mode}, variant={self._variant.value}."
            ),
        )

    def _build_continuum_profile(self, tower_profile: CMSTowerProfile) -> CMSContinuumProfile:
        role_defaults = {
            "fast-band": (0.12, 0.36),
            "session-band": (0.48, 0.24),
            "slow-band": (0.88, 0.18),
            "meta-init": (0.72, 0.12),
            "readout": (0.58, 0.28),
        }
        pending_signals = {
            "online-fast": tuple(0.0 for _ in range(self._dim)),
            "session-medium": self._session_pending_signal,
            "background-slow": self._background_pending_signal,
            "nested-online-prior": self._nested_online_init_target
            if self._variant is CMSVariant.NESTED and self._mode == "mlp"
            else tuple(0.0 for _ in range(self._dim)),
            "nested-session-prior": self._nested_session_init_target
            if self._variant is CMSVariant.NESTED and self._mode == "mlp"
            else tuple(0.0 for _ in range(self._dim)),
            "tower-readout": tower_profile.readout_vector,
        }
        bands: list[CMSContinuumBand] = []
        edges: list[CMSContinuumReconstructionEdge] = []
        for level in tower_profile.levels:
            persistence_bias, retrieval_weight = role_defaults.get(level.role, (0.5, 0.2))
            update_frequency = 1.0 / max(level.cadence_interval, 1)
            band = CMSContinuumBand(
                band_id=level.level_id,
                role=level.role,
                vector=level.vector,
                cadence_interval=level.cadence_interval,
                update_frequency=update_frequency,
                persistence_bias=persistence_bias,
                retrieval_weight=retrieval_weight,
                pending_signal=pending_signals.get(level.level_id, tuple(0.0 for _ in range(self._dim))),
                source_band_ids=level.source_level_ids,
                description=level.description,
            )
            bands.append(band)
            for source_band_id in level.source_level_ids:
                transfer_kind = "aggregation"
                strength = 0.45
                if level.role == "meta-init":
                    transfer_kind = "reset-prior"
                    strength = 0.62
                elif level.role == "readout":
                    transfer_kind = "associative-readout"
                    strength = 0.38
                edges.append(
                    CMSContinuumReconstructionEdge(
                        edge_id=f"{source_band_id}->{level.level_id}",
                        source_band_id=source_band_id,
                        target_band_id=level.level_id,
                        transfer_kind=transfer_kind,
                        strength=strength,
                        description=(
                            f"Continuum transfer from {source_band_id} into {level.level_id} "
                            f"with role={level.role}."
                        ),
                    )
                )
        if self._variant is CMSVariant.NESTED and self._mode == "mlp":
            edges.extend(
                (
                    CMSContinuumReconstructionEdge(
                        edge_id="nested-online-prior->online-fast",
                        source_band_id="nested-online-prior",
                        target_band_id="online-fast",
                        transfer_kind="context-reset-reconstruction",
                        strength=0.74,
                        description="Meta-learned online prior can reconstruct the fast band on context reset.",
                    ),
                    CMSContinuumReconstructionEdge(
                        edge_id="nested-session-prior->session-medium",
                        source_band_id="nested-session-prior",
                        target_band_id="session-medium",
                        transfer_kind="context-reset-reconstruction",
                        strength=0.78,
                        description="Meta-learned session prior can reconstruct the session band on context reset.",
                    ),
                    CMSContinuumReconstructionEdge(
                        edge_id="background-slow->session-medium",
                        source_band_id="background-slow",
                        target_band_id="session-medium",
                        transfer_kind="slow-to-fast-reuse",
                        strength=_clamp(self._anti_forgetting + 0.32),
                        description="Slow band re-seeds session memory through anti-forgetting and consolidation.",
                    ),
                    CMSContinuumReconstructionEdge(
                        edge_id="session-medium->online-fast",
                        source_band_id="session-medium",
                        target_band_id="online-fast",
                        transfer_kind="slow-to-fast-reuse",
                        strength=_clamp(self._anti_forgetting + 0.26),
                        description="Session band re-seeds fast memory during reset and consolidation.",
                    ),
                )
            )
        return CMSContinuumProfile(
            profile_id=f"{tower_profile.profile_id}:continuum",
            bands=tuple(bands),
            reconstruction_edges=tuple(edges),
            readout_band_id="tower-readout",
            description=(
                f"Continuum memory profile with {len(bands)} bands and {len(edges)} reconstruction edges "
                f"for mode={self._mode}, variant={self._variant.value}."
            ),
        )

    # ------------------------------------------------------------------
    # internal helpers — vector mode
    # ------------------------------------------------------------------

    def _blend(
        self,
        previous: tuple[float, ...],
        current: tuple[float, ...],
        *,
        rate: float,
    ) -> tuple[float, ...]:
        return tuple(
            _clamp(previous_value * (1.0 - rate) + current_value * rate)
            for previous_value, current_value in zip(previous, current)
        )

    def _gradient_update(
        self,
        *,
        band_id: str,
        current: tuple[float, ...],
        target: tuple[float, ...],
        momentum: tuple[float, ...],
        lr: float,
        pending_signal: tuple[float, ...],
        observations_since_update: int,
        cadence_interval: int,
        source_signal: tuple[float, ...],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Gradient-style update: compute error -> update momentum -> apply."""
        decision, features = self._decide_band_update(
            band_id=band_id,
            current=current,
            target=target,
            pending_signal=pending_signal,
            observations_since_update=observations_since_update,
            cadence_interval=cadence_interval,
            source_signal=source_signal,
        )
        error = tuple(target[i] - current[i] for i in range(self._dim))
        new_momentum = tuple(
            (_clamp(self._momentum_beta * decision.momentum_gate)) * momentum[i]
            + (1.0 - _clamp(self._momentum_beta * decision.momentum_gate)) * error[i]
            for i in range(self._dim)
        )
        effective_lr = lr * max(0.05, decision.step_scale) * max(0.05, decision.write_gate)
        updated = tuple(
            _clamp(current[i] + effective_lr * new_momentum[i] + decision.bias_delta * 0.02)
            for i in range(self._dim)
        )
        self._learn_from_band_update(
            decision=decision,
            features=features,
            before=current,
            after=updated,
            target=target,
        )
        return updated, new_momentum

    def _apply_anti_forgetting(self) -> None:
        """Backflow from slow to fast (vector mode)."""
        online_decision, _ = self._decide_band_update(
            band_id="anti-forgetting-online",
            current=self._online_fast,
            target=self._background_slow,
            pending_signal=tuple(0.0 for _ in range(self._dim)),
            observations_since_update=0,
            cadence_interval=1,
            source_signal=self._background_slow,
        )
        session_decision, _ = self._decide_band_update(
            band_id="anti-forgetting-session",
            current=self._session_medium,
            target=self._background_slow,
            pending_signal=self._session_pending_signal,
            observations_since_update=self._session_observations_since_update,
            cadence_interval=self._session_cadence,
            source_signal=self._background_slow,
        )
        strength = self._anti_forgetting
        self._online_fast = tuple(
            _clamp(
                self._online_fast[i]
                + strength
                * online_decision.slow_mix
                * (self._background_slow[i] - self._online_fast[i])
                * 0.12
            )
            for i in range(self._dim)
        )
        self._session_medium = tuple(
            _clamp(
                self._session_medium[i]
                + strength
                * session_decision.slow_mix
                * (self._background_slow[i] - self._session_medium[i])
                * 0.08
            )
            for i in range(self._dim)
        )

    def _integrate_signal_gradient(
        self,
        *,
        band_id: str,
        current_vector: tuple[float, ...],
        pending_signal: tuple[float, ...],
        observations_since_update: int,
        momentum: tuple[float, ...],
        signal: tuple[float, ...],
        lr: float,
        cadence_interval: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...], int, tuple[float, ...]]:
        """Cadence-gated gradient update for medium/slow bands (vector mode)."""
        pending_signal = self._align_signal_dim(pending_signal)
        signal = self._align_signal_dim(signal)
        next_count = observations_since_update + 1
        next_pending = tuple(
            _clamp((pending_signal[index] * observations_since_update + signal[index]) / next_count)
            for index in range(self._dim)
        )
        if next_count < cadence_interval:
            return (current_vector, next_pending, next_count, momentum)
        updated, new_momentum = self._gradient_update(
            band_id=band_id,
            current=current_vector,
            target=next_pending,
            momentum=momentum,
            lr=lr,
            pending_signal=pending_signal,
            observations_since_update=observations_since_update,
            cadence_interval=cadence_interval,
            source_signal=signal,
        )
        return (updated, tuple(0.0 for _ in range(self._dim)), 0, new_momentum)

    # ------------------------------------------------------------------
    # internal helpers — MLP mode
    # ------------------------------------------------------------------

    def _integrate_signal_mlp(
        self,
        *,
        band_id: str,
        mlp: CMSBandMLP,
        current_vector: tuple[float, ...],
        pending_signal: tuple[float, ...],
        observations_since_update: int,
        signal: tuple[float, ...],
        cadence_interval: int,
    ) -> tuple[tuple[float, ...], int]:
        """Cadence-gated MLP update for medium/slow bands."""
        pending_signal = self._align_signal_dim(pending_signal)
        signal = self._align_signal_dim(signal)
        next_count = observations_since_update + 1
        next_pending = tuple(
            _clamp((pending_signal[i] * observations_since_update + signal[i]) / next_count)
            for i in range(self._dim)
        )
        if next_count < cadence_interval:
            return (next_pending, next_count)
        decision, features = self._decide_band_update(
            band_id=band_id,
            current=current_vector,
            target=next_pending,
            pending_signal=pending_signal,
            observations_since_update=observations_since_update,
            cadence_interval=cadence_interval,
            source_signal=signal,
        )
        target = self._blend_signal(current_vector, next_pending, rate=decision.write_gate)
        self._band_mlp_update(
            band_id=band_id,
            mlp=mlp,
            target=target,
            decision=decision,
        )
        self._learn_from_band_update(
            decision=decision,
            features=features,
            before=current_vector,
            after=mlp.representation_vector(),
            target=target,
        )
        return (tuple(0.0 for _ in range(self._dim)), 0)

    def _apply_anti_forgetting_mlp(self) -> None:
        """Backflow from slow to fast (MLP mode)."""
        online_decision, _ = self._decide_band_update(
            band_id="anti-forgetting-online",
            current=self._online_mlp.representation_vector(),
            target=self._background_mlp.representation_vector(),
            pending_signal=tuple(0.0 for _ in range(self._dim)),
            observations_since_update=0,
            cadence_interval=1,
            source_signal=self._background_mlp.representation_vector(),
        )
        session_decision, _ = self._decide_band_update(
            band_id="anti-forgetting-session",
            current=self._session_mlp.representation_vector(),
            target=self._background_mlp.representation_vector(),
            pending_signal=self._session_pending_signal,
            observations_since_update=self._session_observations_since_update,
            cadence_interval=self._session_cadence,
            source_signal=self._background_mlp.representation_vector(),
        )
        self._online_mlp.mix_from(
            self._background_mlp,
            strength=self._anti_forgetting,
            factor=max(0.01, online_decision.slow_mix * 0.12),
        )
        self._session_mlp.mix_from(
            self._background_mlp,
            strength=self._anti_forgetting,
            factor=max(0.01, session_decision.slow_mix * 0.08),
        )

__all__ = [
    "CMSBandMLP",
    "CMSBandState",
    "CMSCheckpointState",
    "CMSContinuumBand",
    "CMSContinuumProfile",
    "CMSContinuumReconstructionEdge",
    "CMSHopeSelfModificationState",
    "CMSMemoryCore",
    "CMSState",
    "CMSTowerConsolidationUpdate",
    "CMSTowerLevelState",
    "CMSTowerProfile",
    "CMSVariant",
]
