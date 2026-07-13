"""Packet C (long-horizon-closure) — AffordanceModule.

The "future ``AffordanceModule``" referenced in
``lifeform_affordance/snapshot.py`` lands here. It satisfies the
``RuntimeModule[AffordanceSnapshot]`` contract from ``vz-contracts``
so the kernel propagate path can publish an ``affordance`` snapshot
that is **driven by the metacontroller's z_t latent state**, not by
hand-coded routing per descriptor name.

Architecture:

```text
temporal_abstraction snapshot
        |
        v
controller_state.code  (= z_t, low-dim latent action)
        |
        v
score_affordance_candidates(z_t, descriptor_names)
        |
        v   deterministic name -> projection vector via sha256
        v   score = sigmoid(dot(z_t_padded, proj_vector))
        v
AffordanceCandidate(score=...) per descriptor
        |
        v
boundary / regime gating reuses scorer module's _regime_blocked
        |
        v
AffordanceSnapshot.selected = strict argmax (with margin)
```

Hard invariants:

1. **No descriptor-name branching.** ``score_affordance_candidates``
   below treats ``descriptor.name`` as opaque bytes; the projection
   is the same function for every name. There is NO ``if name ==
   "read_file"`` and no ``descriptor_name in {...}`` branch on
   behavior. Static contract test
   ``tests/contracts/test_no_keyword_routing.py`` enforces this at
   the repo level.
2. **Default ``WiringLevel`` is ACTIVE** (long-horizon-closure follow-up).
   The module publishes the ``affordance`` snapshot from its own
   z_t projection so a downstream prompt planner / response
   synthesizer can pick it up without an explicit opt-in. Use
   SHADOW for benchmark ablations that need a quiet module;
   DISABLED to suppress construction entirely.
3. **Reads only public ``temporal_abstraction`` snapshot.** The
   module never reaches into temporal owner internals; it only uses
   ``controller_state.code`` (the public z_t) plus optionally the
   reserved ``MetacontrollerRuntimeState.affordance_selection``
   field (currently always None). When the metacontroller field is
   non-None, AffordanceModule prefers it (single owner principle —
   metacontroller is the legitimate selector, AffordanceModule is
   the renderer).
4. **Cold start.** If z_t is empty (no temporal snapshot or empty
   controller code), every candidate gets score ``_NEUTRAL_SCORE``
   (0.5) and selection is None.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

from volvence_zero.affordance import AffordanceDescriptor, AffordanceKind
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.semantic_state import BoundaryConsentSnapshot
from volvence_zero.temporal_types import TemporalAbstractionSnapshot

from lifeform_affordance.registry import AffordanceRegistry
from lifeform_affordance.snapshot import (
    AffordanceCandidate,
    AffordanceSnapshot,
)


_NEUTRAL_SCORE: float = 0.5
"""Score used when z_t is empty (cold start)."""

_SELECTION_MIN_SCORE: float = 0.55
_SELECTION_MIN_MARGIN: float = 0.05
"""Top candidate is published as ``selected`` only when its score
clears ``_SELECTION_MIN_SCORE`` AND beats the runner-up by at least
``_SELECTION_MIN_MARGIN``. Both thresholds prevent a "confidently
selected nothing-stands-out" snapshot.
"""


def _label_projection_vector(name: str, dim: int) -> tuple[float, ...]:
    """Deterministic descriptor-name -> projection vector.

    Uses SHA-256 so the projection is stable across runs / machines
    and depends on the *full* name string (not just a prefix). Each
    coordinate maps a 4-byte chunk to a signed float in roughly
    ``[-1, 1]``. ``dim`` controls the projection dimensionality.

    The fact that the function is the same for every name is the
    "no keyword routing" guarantee: nothing about ``"read_file"``
    is special compared to ``"write_file"`` — both go through the
    same hash + dot product. Behavior comes from the dot with z_t.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim!r}")
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    out: list[float] = []
    for i in range(dim):
        # 4-byte sliding window with wrap-around for dim > 8 (32 / 4)
        offset = (i * 4) % len(digest)
        chunk = digest[offset:offset + 4]
        if len(chunk) < 4:
            chunk = chunk + digest[: 4 - len(chunk)]
        u = int.from_bytes(chunk, byteorder="big", signed=True)
        out.append(u / float(2 ** 31))
    return tuple(out)


def score_affordance_candidates(
    *,
    descriptor_names: tuple[str, ...],
    z_t: tuple[float, ...],
) -> tuple[tuple[str, float], ...]:
    """Project ``z_t`` onto each descriptor name via deterministic
    hash projection, return ``(name, score)`` pairs in input order.

    Selection invariants:

    - Same function for every descriptor name (no per-name branch);
    - Empty ``z_t`` => every name gets ``_NEUTRAL_SCORE``;
    - Same ``z_t`` + same name => same score (deterministic).

    Score formula::

        score = sigmoid( dot(z_t_padded, proj_vector(name, dim)) )

    where ``dim = max(len(z_t), 1)`` and ``z_t_padded`` is z_t
    truncated/zero-padded to ``dim`` (in practice ``dim == len(z_t)``
    so no padding happens; the explicit pad is just defence-in-depth
    against an empty z_t edge case).
    """
    if not z_t:
        return tuple((name, _NEUTRAL_SCORE) for name in descriptor_names)
    dim = len(z_t)
    out: list[tuple[str, float]] = []
    for name in descriptor_names:
        proj = _label_projection_vector(name, dim)
        raw = sum(z * p for z, p in zip(z_t, proj))
        # Numerically-stable sigmoid clamp to keep scores in
        # ``(eps, 1 - eps)`` so downstream consumers can treat them
        # as "never exactly 0 or 1" without special-casing.
        squashed = 1.0 / (1.0 + math.exp(-_clamp_logit(raw)))
        out.append((name, squashed))
    return tuple(out)


def _clamp_logit(x: float, *, bound: float = 30.0) -> float:
    """Avoid math.exp overflow when z_t has unusually large entries."""
    if x > bound:
        return bound
    if x < -bound:
        return -bound
    return x


def _selection_entropy(scores: tuple[float, ...]) -> float:
    """Normalised Shannon entropy of the score distribution.

    Returns ``1.0`` for a uniform distribution (no preference)
    and ``0.0`` for a one-hot distribution. Used by AffordanceSnapshot
    as a "how confident is the metacontroller about this turn"
    readout. Empty input returns ``0.0``.
    """
    if not scores:
        return 0.0
    total = sum(scores)
    if total <= 0.0:
        return 0.0
    probs = [s / total for s in scores]
    # Shannon, base e.
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    h_max = math.log(len(probs)) if len(probs) > 1 else 1.0
    if h_max <= 0.0:
        return 0.0
    return max(0.0, min(1.0, h / h_max))


def _pick_selected(
    candidates: tuple[AffordanceCandidate, ...],
) -> AffordanceCandidate | None:
    """Strict argmax with min-score and min-margin gating.

    ``selected=None`` is a legitimate output (means "no affordance
    stood out — don't proactively invoke this turn"); the snapshot
    consumer must accept ``selected is None`` without crashing.
    """
    unblocked = [c for c in candidates if not c.is_blocked]
    if not unblocked:
        return None
    unblocked.sort(key=lambda c: c.score, reverse=True)
    top = unblocked[0]
    if top.score < _SELECTION_MIN_SCORE:
        return None
    if len(unblocked) >= 2:
        runner_up = unblocked[1]
        if (top.score - runner_up.score) < _SELECTION_MIN_MARGIN:
            return None
    return top


# CP-04 (intent-alignment W2.E): kinds that act OUTSIDE the kernel. TOOL
# invokes external functions/APIs and SHELL drives deployment-side
# capabilities; ACTION / ORGAN stay inside the kernel and are never
# consent-gated at this layer. This is a protocol-level classification on
# the closed AffordanceKind enum, not keyword routing over names.
_EXTERNAL_ACTION_KINDS: frozenset[AffordanceKind] = frozenset(
    {AffordanceKind.TOOL, AffordanceKind.SHELL}
)


def _consent_blocked_reason(
    descriptor: AffordanceDescriptor,
    *,
    boundary_consent: BoundaryConsentSnapshot | None,
) -> str:
    """CP-04: typed consent gate over the published boundary_consent snapshot.

    When the boundary_consent owner has decided external actions are blocked
    (``external_action_blocked=True``), every external-kind candidate is
    published as typed-blocked instead of being silently dropped, so the
    denial is auditable downstream. The owner's decision is consumed as-is
    (single owner principle); this module never re-derives consent from
    records.
    """
    if boundary_consent is None:
        return ""
    if not boundary_consent.external_action_blocked:
        return ""
    if descriptor.kind not in _EXTERNAL_ACTION_KINDS:
        return ""
    return (
        f"consent_blocked:external_action:kind={descriptor.kind.value}:"
        f"consent={boundary_consent.external_action_consent}"
    )


def _regime_blocked_reason(
    descriptor: AffordanceDescriptor,
    *,
    active_regime_id: str | None,
) -> str:
    """Mirror of ``DescriptorDerivedBoundaryPolicy``'s regime check.

    Consent / confirmation are caller-time concerns and stay in the
    invoker; the snapshot just needs the regime block surface so
    downstream rendering can hide the candidate cleanly.
    """
    if descriptor.excluded_from_runtime_selection:
        return "descriptor_excluded"
    if active_regime_id is None or not active_regime_id:
        return ""
    if active_regime_id in descriptor.safety_model.blocked_in_regimes:
        return (
            f"regime_blocked:{active_regime_id} "
            f"in {sorted(descriptor.safety_model.blocked_in_regimes)!r}"
        )
    return ""


class AffordanceModule(RuntimeModule[AffordanceSnapshot]):
    """Lifeform-side ``RuntimeModule`` that publishes the ``affordance``
    snapshot, scored by the metacontroller's z_t.

    The module is intentionally light: it does not own affordance
    backends (the ``AffordanceInvoker`` does), and it does not own
    selection policy (``score_affordance_candidates`` is a pure
    function in this same file). Its job is to publish a snapshot
    consumers (prompt planner / response synthesizer / dashboards)
    can read without reaching into the registry directly.

    Default wiring level is ``SHADOW`` per the affordance spec:
    publishing happens but downstream consumption is opt-in (currently
    none). When the metacontroller publishes its own
    ``affordance_selection`` (future packet), this module prefers
    the metacontroller-side selection over its local projection.
    """

    slot_name = "affordance"
    owner = "AffordanceModule"
    value_type = AffordanceSnapshot
    dependencies = ("temporal_abstraction", "boundary_consent")
    default_wiring_level = WiringLevel.ACTIVE

    def __init__(
        self,
        *,
        registry: AffordanceRegistry,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._registry = registry

    @property
    def registry(self) -> AffordanceRegistry:
        return self._registry

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[AffordanceSnapshot]:
        temporal_snap = upstream.get("temporal_abstraction")
        regime_snap = upstream.get("regime")
        consent_snap = upstream.get("boundary_consent")

        boundary_consent: BoundaryConsentSnapshot | None = None
        if consent_snap is not None and isinstance(
            consent_snap.value, BoundaryConsentSnapshot
        ):
            boundary_consent = consent_snap.value

        z_t: tuple[float, ...] = ()
        if temporal_snap is not None and isinstance(
            temporal_snap.value, TemporalAbstractionSnapshot
        ):
            z_t = tuple(temporal_snap.value.controller_state.code)

        active_regime_id: str | None = None
        if regime_snap is not None:
            regime_value = regime_snap.value
            active = getattr(regime_value, "active_regime", None)
            if active is not None:
                rid = getattr(active, "regime_id", None)
                if isinstance(rid, str):
                    active_regime_id = rid

        descriptors = tuple(
            d
            for d in self._registry.all_descriptors()
            if not d.excluded_from_runtime_selection
        )
        descriptor_names = tuple(d.name for d in descriptors)

        scores: dict[str, float]
        scoring_source: str
        if (
            temporal_snap is not None
            and isinstance(temporal_snap.value, TemporalAbstractionSnapshot)
            and self._has_metacontroller_selection(temporal_snap.value)
        ):
            # Future-reserved path: metacontroller publishes its own
            # selection. We trust it as the legitimate single owner.
            mc_selection = self._extract_metacontroller_selection(
                temporal_snap.value
            )
            scores = dict(mc_selection.candidate_scores)
            scoring_source = "metacontroller"
        else:
            local = score_affordance_candidates(
                descriptor_names=descriptor_names,
                z_t=z_t,
            )
            scores = dict(local)
            scoring_source = "z_t_projection" if z_t else "neutral_cold_start"

        candidates: list[AffordanceCandidate] = []
        for descriptor in descriptors:
            blocked_reason = _regime_blocked_reason(
                descriptor, active_regime_id=active_regime_id
            )
            if not blocked_reason:
                blocked_reason = _consent_blocked_reason(
                    descriptor, boundary_consent=boundary_consent
                )
            score = scores.get(descriptor.name, _NEUTRAL_SCORE)
            if blocked_reason:
                # Blocked candidates are kept in the snapshot for audit
                # but with score=0 to make the selection invariant
                # trivially satisfied.
                candidates.append(
                    AffordanceCandidate(
                        descriptor_name=descriptor.name,
                        score=0.0,
                        rationale=(
                            f"affordance_module(v1):blocked:{blocked_reason}"
                        ),
                        expected_cost=descriptor.cost_model,
                        blocked_reason=blocked_reason,
                    )
                )
            else:
                candidates.append(
                    AffordanceCandidate(
                        descriptor_name=descriptor.name,
                        score=round(score, 4),
                        rationale=(
                            f"affordance_module(v1):src={scoring_source}:"
                            f"score={score:.4f}:z_dim={len(z_t)}"
                        ),
                        expected_cost=descriptor.cost_model,
                    )
                )

        candidate_tuple = tuple(candidates)
        selected = _pick_selected(candidate_tuple)
        unblocked_score_tuple = tuple(
            c.score for c in candidate_tuple if not c.is_blocked
        )
        entropy = _selection_entropy(unblocked_score_tuple)
        snapshot = AffordanceSnapshot(
            available=descriptors,
            candidates_for_turn=candidate_tuple,
            selected=selected,
            description=(
                f"AffordanceModule(v1) {len(candidate_tuple)} cand / "
                f"{len(unblocked_score_tuple)} unblocked / "
                f"selected={selected.descriptor_name if selected else None!r} / "
                f"src={scoring_source} / entropy={entropy:.3f} / "
                f"regime={active_regime_id!r} / "
                f"consent_external_blocked="
                f"{boundary_consent.external_action_blocked if boundary_consent is not None else None}"
            ),
        )
        return self.publish(snapshot)

    @staticmethod
    def _has_metacontroller_selection(
        temporal_value: TemporalAbstractionSnapshot,
    ) -> bool:
        """Reserved path: detect a populated
        ``MetacontrollerRuntimeState.affordance_selection`` field on
        the temporal snapshot. Currently always False because no
        production temporal owner publishes the metacontroller runtime
        state on ``TemporalAbstractionSnapshot``. The hook is kept
        so the future temporal-owner change can promote selection
        without changing this module's interface.
        """
        runtime_state = getattr(
            temporal_value, "metacontroller_runtime_state", None
        )
        if runtime_state is None:
            return False
        selection = getattr(runtime_state, "affordance_selection", None)
        return selection is not None

    @staticmethod
    def _extract_metacontroller_selection(
        temporal_value: TemporalAbstractionSnapshot,
    ) -> Any:
        """Companion to ``_has_metacontroller_selection``. Returns
        the ``AffordanceSelectionState`` instance; raises if the
        guard above did not check first.
        """
        runtime_state = getattr(
            temporal_value, "metacontroller_runtime_state"
        )
        return runtime_state.affordance_selection


__all__ = [
    "AffordanceModule",
    "score_affordance_candidates",
]
