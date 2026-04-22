from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float, *, magnitude: float = 1.0) -> float:
    limited = max(-magnitude, min(magnitude, value))
    if magnitude <= 1e-8:
        return 0.0
    return limited


def _dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


@dataclass(frozen=True)
class RetrievalControlReadoutInputs:
    regime_id: str
    abstract_action: str | None
    knowledge_domains: tuple[str, ...]
    experience_domains: tuple[str, ...]
    world_weight: float
    self_weight: float
    continuum_position: float
    continuum_frequency: float
    continuum_slow_share: float
    continuum_reconstruction_pressure: float
    delayed_payoff_signal: float
    sequence_payoff_signal: float
    playbook_knowledge_hint: float
    playbook_experience_hint: float
    knowledge_weight_bias: float
    experience_weight_bias: float
    regime_fast_prior_bias: float
    action_fast_prior_bias: float
    family_fast_prior_bias: float
    fast_prior_strength: float = 0.0
    fast_prior_attribution_count: int = 0
    fast_prior_sequence_count: int = 0
    continuum_active_band_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalControlReadout:
    knowledge_domains: tuple[str, ...]
    experience_domains: tuple[str, ...]
    knowledge_weight: float
    experience_weight: float
    description: str


@dataclass(frozen=True)
class RetrievalReadoutCheckpoint:
    checkpoint_id: str
    parameters: "RetrievalControlReadoutParameters"
    confidence: float
    description: str
    source_session_post_job_id: str | None = None


@dataclass(frozen=True)
class RetrievalReadoutFeatures:
    problem_solving_mode: float
    support_mode: float
    repair_mode: float
    world_bias: float
    self_bias: float
    continuum_focus: float
    continuum_frequency: float
    continuum_slow_share: float
    continuum_reconstruction_pressure: float
    delayed_payoff_signal: float
    sequence_payoff_signal: float
    playbook_knowledge_hint: float
    playbook_experience_hint: float
    knowledge_weight_bias: float
    experience_weight_bias: float
    regime_fast_prior_bias: float
    action_fast_prior_bias: float
    family_fast_prior_bias: float
    action_is_stabilize: float
    action_is_task: float


@dataclass(frozen=True)
class RetrievalReadoutHead:
    bias: float
    weights: tuple[tuple[str, float], ...]

    def evaluate(self, features: RetrievalReadoutFeatures) -> float:
        values = features.__dict__
        score = self.bias
        for feature_name, weight in self.weights:
            score += values.get(feature_name, 0.0) * weight
        return score

    def shifted(
        self,
        *,
        bias_delta: float = 0.0,
        weight_deltas: tuple[tuple[str, float], ...] = (),
    ) -> "RetrievalReadoutHead":
        weight_map = {feature_name: weight for feature_name, weight in self.weights}
        for feature_name, delta in weight_deltas:
            weight_map[feature_name] = weight_map.get(feature_name, 0.0) + delta
        return RetrievalReadoutHead(
            bias=self.bias + bias_delta,
            weights=tuple(weight_map.items()),
        )


@dataclass(frozen=True)
class RetrievalControlReadoutParameters:
    knowledge_weight_head: RetrievalReadoutHead
    experience_weight_head: RetrievalReadoutHead
    stabilization_domain_head: RetrievalReadoutHead
    structured_domain_head: RetrievalReadoutHead
    guidance_domain_head: RetrievalReadoutHead

    @classmethod
    def default(cls) -> "RetrievalControlReadoutParameters":
        return cls(
            knowledge_weight_head=RetrievalReadoutHead(
                bias=0.46,
                weights=(
                    ("problem_solving_mode", 0.18),
                    ("support_mode", -0.14),
                    ("repair_mode", -0.08),
                    ("world_bias", 0.16),
                    ("self_bias", -0.10),
                    ("continuum_focus", -0.16),
                    ("continuum_frequency", 0.07),
                    ("continuum_slow_share", -0.10),
                    ("continuum_reconstruction_pressure", -0.05),
                    ("delayed_payoff_signal", 0.06),
                    ("sequence_payoff_signal", 0.05),
                    ("playbook_knowledge_hint", 0.12),
                    ("playbook_experience_hint", -0.08),
                    ("knowledge_weight_bias", 0.22),
                    ("experience_weight_bias", -0.12),
                    ("regime_fast_prior_bias", 0.05),
                    ("action_fast_prior_bias", -0.04),
                    ("family_fast_prior_bias", -0.04),
                    ("action_is_task", 0.03),
                    ("action_is_stabilize", -0.04),
                ),
            ),
            experience_weight_head=RetrievalReadoutHead(
                bias=0.46,
                weights=(
                    ("problem_solving_mode", -0.08),
                    ("support_mode", 0.14),
                    ("repair_mode", 0.08),
                    ("world_bias", -0.08),
                    ("self_bias", 0.12),
                    ("continuum_focus", 0.15),
                    ("continuum_slow_share", 0.11),
                    ("continuum_reconstruction_pressure", 0.10),
                    ("delayed_payoff_signal", 0.03),
                    ("sequence_payoff_signal", 0.05),
                    ("playbook_knowledge_hint", -0.06),
                    ("playbook_experience_hint", 0.12),
                    ("knowledge_weight_bias", -0.12),
                    ("experience_weight_bias", 0.20),
                    ("regime_fast_prior_bias", 0.04),
                    ("action_fast_prior_bias", 0.07),
                    ("family_fast_prior_bias", 0.07),
                    ("action_is_task", -0.02),
                    ("action_is_stabilize", 0.05),
                ),
            ),
            stabilization_domain_head=RetrievalReadoutHead(
                bias=-0.12,
                weights=(
                    ("support_mode", 0.18),
                    ("repair_mode", 0.10),
                    ("self_bias", 0.16),
                    ("continuum_slow_share", 0.16),
                    ("continuum_reconstruction_pressure", 0.10),
                    ("playbook_experience_hint", 0.12),
                    ("action_fast_prior_bias", 0.06),
                    ("family_fast_prior_bias", 0.06),
                    ("action_is_stabilize", 0.12),
                ),
            ),
            structured_domain_head=RetrievalReadoutHead(
                bias=-0.24,
                weights=(
                    ("problem_solving_mode", 0.18),
                    ("world_bias", 0.16),
                    ("continuum_slow_share", 0.10),
                    ("delayed_payoff_signal", 0.08),
                    ("sequence_payoff_signal", 0.08),
                    ("playbook_knowledge_hint", 0.06),
                    ("action_is_task", 0.10),
                ),
            ),
            guidance_domain_head=RetrievalReadoutHead(
                bias=-0.26,
                weights=(
                    ("continuum_reconstruction_pressure", 0.24),
                    ("continuum_slow_share", 0.08),
                    ("experience_weight_bias", 0.06),
                    ("playbook_experience_hint", 0.05),
                ),
            ),
        )

    def updated_from_slow_prior(
        self,
        *,
        strength: float,
        attribution_count: int,
        sequence_count: int,
        regime_bias: float,
        action_bias: float,
        family_bias: float,
        knowledge_weight_bias: float,
        experience_weight_bias: float,
    ) -> "RetrievalControlReadoutParameters":
        bounded_strength = _clamp(strength)
        if bounded_strength <= 1e-6 and attribution_count <= 0 and sequence_count <= 0:
            return self
        evidence_scale = _clamp(min(attribution_count, 6) / 6.0 * 0.7 + min(sequence_count, 4) / 4.0 * 0.3)
        delta_scale = bounded_strength * (0.05 + evidence_scale * 0.07)
        experience_shift = _clamp_signed(experience_weight_bias - knowledge_weight_bias, magnitude=1.0)
        regime_shift = _clamp_signed(regime_bias, magnitude=1.0)
        action_shift = _clamp_signed(action_bias, magnitude=1.0)
        family_shift = _clamp_signed(family_bias, magnitude=1.0)
        return RetrievalControlReadoutParameters(
            knowledge_weight_head=self.knowledge_weight_head.shifted(
                bias_delta=(
                    -experience_shift * delta_scale * 0.6
                    - max(action_shift, 0.0) * delta_scale * 0.25
                    + max(regime_shift, 0.0) * delta_scale * 0.10
                ),
                weight_deltas=(
                    ("knowledge_weight_bias", -experience_shift * delta_scale * 0.35),
                    ("experience_weight_bias", -experience_shift * delta_scale * 0.25),
                    ("action_fast_prior_bias", -max(action_shift, 0.0) * delta_scale * 0.18),
                    ("family_fast_prior_bias", -max(family_shift, 0.0) * delta_scale * 0.14),
                    ("delayed_payoff_signal", max(regime_shift, 0.0) * delta_scale * 0.10),
                    ("sequence_payoff_signal", -max(family_shift, 0.0) * delta_scale * 0.08),
                ),
            ),
            experience_weight_head=self.experience_weight_head.shifted(
                bias_delta=(
                    experience_shift * delta_scale * 0.7
                    + max(action_shift, 0.0) * delta_scale * 0.18
                    + max(family_shift, 0.0) * delta_scale * 0.14
                ),
                weight_deltas=(
                    ("experience_weight_bias", experience_shift * delta_scale * 0.38),
                    ("knowledge_weight_bias", experience_shift * delta_scale * 0.20),
                    ("action_fast_prior_bias", max(action_shift, 0.0) * delta_scale * 0.18),
                    ("family_fast_prior_bias", max(family_shift, 0.0) * delta_scale * 0.16),
                    ("regime_fast_prior_bias", max(regime_shift, 0.0) * delta_scale * 0.12),
                    ("sequence_payoff_signal", max(family_shift, 0.0) * delta_scale * 0.10),
                ),
            ),
            stabilization_domain_head=self.stabilization_domain_head.shifted(
                bias_delta=max(action_shift, 0.0) * delta_scale * 0.45,
                weight_deltas=(
                    ("action_fast_prior_bias", max(action_shift, 0.0) * delta_scale * 0.24),
                    ("family_fast_prior_bias", max(family_shift, 0.0) * delta_scale * 0.14),
                    ("continuum_slow_share", max(regime_shift, 0.0) * delta_scale * 0.10),
                ),
            ),
            structured_domain_head=self.structured_domain_head.shifted(
                bias_delta=max(regime_shift, 0.0) * delta_scale * 0.24,
                weight_deltas=(
                    ("sequence_payoff_signal", max(family_shift, 0.0) * delta_scale * 0.18),
                    ("delayed_payoff_signal", max(regime_shift, 0.0) * delta_scale * 0.18),
                    ("world_bias", max(regime_shift, 0.0) * delta_scale * 0.10),
                ),
            ),
            guidance_domain_head=self.guidance_domain_head.shifted(
                bias_delta=max(experience_shift, 0.0) * delta_scale * 0.12,
                weight_deltas=(
                    ("continuum_reconstruction_pressure", bounded_strength * delta_scale * 0.18),
                    ("experience_weight_bias", max(experience_shift, 0.0) * delta_scale * 0.18),
                    ("playbook_experience_hint", max(family_shift, 0.0) * delta_scale * 0.08),
                ),
            ),
        )


class RetrievalControlReadoutStrategy:
    """Compact owner-side readout seam from ETA/public priors into retrieval control."""

    def __init__(
        self,
        *,
        parameters: RetrievalControlReadoutParameters | None = None,
    ) -> None:
        self._parameters = parameters or RetrievalControlReadoutParameters.default()

    def build(self, inputs: RetrievalControlReadoutInputs) -> RetrievalControlReadout:
        features = self._extract_features(inputs)
        knowledge_weight = _clamp(self._parameters.knowledge_weight_head.evaluate(features))
        experience_weight = _clamp(self._parameters.experience_weight_head.evaluate(features))
        total_weight = knowledge_weight + experience_weight
        if total_weight <= 1e-6:
            knowledge_weight = 0.5
            experience_weight = 0.5
        else:
            knowledge_weight = _clamp(knowledge_weight / total_weight)
            experience_weight = _clamp(experience_weight / total_weight)
        experience_domains = self._adjust_experience_domains(
            experience_domains=inputs.experience_domains,
            features=features,
            parameters=self._parameters,
        )
        return RetrievalControlReadout(
            knowledge_domains=inputs.knowledge_domains,
            experience_domains=experience_domains,
            knowledge_weight=knowledge_weight,
            experience_weight=experience_weight,
            description=(
                f"Retrieval control readout selected {len(inputs.knowledge_domains)} knowledge domains and "
                f"{len(experience_domains)} experience domains from compact ETA/application control using "
                "owner-side parameterized readout. "
                f"bands={','.join(inputs.continuum_active_band_ids) if inputs.continuum_active_band_ids else 'none'} "
                f"regime={inputs.regime_id} abstract_action={inputs.abstract_action or 'none'}."
            ),
        )

    def _extract_features(self, inputs: RetrievalControlReadoutInputs) -> RetrievalReadoutFeatures:
        action_label = (inputs.abstract_action or "").lower()
        problem_solving_mode = 1.0 if inputs.regime_id == "problem_solving" else 0.0
        support_mode = 1.0 if inputs.regime_id == "emotional_support" else 0.0
        repair_mode = 1.0 if inputs.regime_id == "repair_and_deescalation" else 0.0
        return RetrievalReadoutFeatures(
            problem_solving_mode=problem_solving_mode,
            support_mode=support_mode,
            repair_mode=repair_mode,
            world_bias=_clamp(0.5 + (inputs.world_weight - inputs.self_weight) * 0.5),
            self_bias=_clamp(0.5 + (inputs.self_weight - inputs.world_weight) * 0.5),
            continuum_focus=_clamp(inputs.continuum_position),
            continuum_frequency=_clamp(inputs.continuum_frequency),
            continuum_slow_share=_clamp(inputs.continuum_slow_share),
            continuum_reconstruction_pressure=_clamp(inputs.continuum_reconstruction_pressure),
            delayed_payoff_signal=_clamp(inputs.delayed_payoff_signal),
            sequence_payoff_signal=_clamp(inputs.sequence_payoff_signal),
            playbook_knowledge_hint=_clamp(inputs.playbook_knowledge_hint),
            playbook_experience_hint=_clamp(inputs.playbook_experience_hint),
            knowledge_weight_bias=_clamp(0.5 + inputs.knowledge_weight_bias * 0.5),
            experience_weight_bias=_clamp(0.5 + inputs.experience_weight_bias * 0.5),
            regime_fast_prior_bias=_clamp(0.5 + inputs.regime_fast_prior_bias * 0.5),
            action_fast_prior_bias=_clamp(0.5 + inputs.action_fast_prior_bias * 0.5),
            family_fast_prior_bias=_clamp(0.5 + inputs.family_fast_prior_bias * 0.5),
            action_is_stabilize=1.0 if "stabilize" in action_label else 0.0,
            action_is_task=1.0 if "task" in action_label or "structure" in action_label else 0.0,
        )

    def _adjust_experience_domains(
        self,
        *,
        experience_domains: tuple[str, ...],
        features: RetrievalReadoutFeatures,
        parameters: RetrievalControlReadoutParameters,
    ) -> tuple[str, ...]:
        adjusted = experience_domains
        stabilization_score = parameters.stabilization_domain_head.evaluate(features)
        structured_score = parameters.structured_domain_head.evaluate(features)
        guidance_score = parameters.guidance_domain_head.evaluate(features)
        if stabilization_score >= 0.5:
            adjusted = _dedupe(("stabilization_patterns",) + adjusted)
        elif structured_score >= 0.5:
            adjusted = _dedupe(("structured_decision_patterns",) + adjusted)
        if guidance_score >= 0.5 and "general_guidance_patterns" not in adjusted:
            adjusted = _dedupe(adjusted + ("general_guidance_patterns",))
        if (
            stabilization_score > structured_score
            and guidance_score >= 0.5
            and "general_guidance_patterns" in adjusted
        ):
            adjusted = _dedupe(("stabilization_patterns",) + adjusted)
        return adjusted
