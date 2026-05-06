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
    action_family_version: int
    switch_gate: float
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
    knowledge_domain_ranking: tuple[str, ...]
    experience_domain_ranking: tuple[str, ...]
    response_mode_hint: str
    clarification_bias: float
    refer_out_bias: float
    answer_depth_bias: float
    continuum_target_position: float
    ordering_bias: tuple[str, ...]
    ordering_driver: str
    description: str


@dataclass(frozen=True)
class RetrievalReadoutCheckpoint:
    checkpoint_id: str
    parameters: "RetrievalControlReadoutParameters"
    confidence: float
    description: str
    source_session_post_job_id: str | None = None
    source_attribution_ids: tuple[str, ...] = ()
    source_sequence_ids: tuple[str, ...] = ()
    mean_retrieval_mix_alignment: float = 0.0
    mean_regime_alignment: float = 0.0
    mean_action_alignment: float = 0.0
    mean_sequence_payoff: float = 0.0


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
    action_family_activation: float
    switch_gate_signal: float


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
    support_response_head: RetrievalReadoutHead
    clarify_response_head: RetrievalReadoutHead
    structure_response_head: RetrievalReadoutHead
    refer_out_response_head: RetrievalReadoutHead
    answer_depth_head: RetrievalReadoutHead
    continuum_target_head: RetrievalReadoutHead

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
            support_response_head=RetrievalReadoutHead(
                bias=0.32,
                weights=(
                    ("support_mode", 0.22),
                    ("repair_mode", 0.08),
                    ("self_bias", 0.14),
                    ("continuum_focus", 0.12),
                    ("continuum_reconstruction_pressure", 0.08),
                    ("action_is_stabilize", 0.14),
                    ("action_fast_prior_bias", 0.08),
                    ("family_fast_prior_bias", 0.06),
                    ("switch_gate_signal", 0.04),
                ),
            ),
            clarify_response_head=RetrievalReadoutHead(
                bias=0.20,
                weights=(
                    ("continuum_reconstruction_pressure", 0.18),
                    ("continuum_focus", 0.10),
                    ("sequence_payoff_signal", -0.06),
                    ("switch_gate_signal", 0.10),
                    ("action_family_activation", 0.06),
                    ("action_fast_prior_bias", -0.04),
                ),
            ),
            structure_response_head=RetrievalReadoutHead(
                bias=0.22,
                weights=(
                    ("problem_solving_mode", 0.22),
                    ("world_bias", 0.16),
                    ("delayed_payoff_signal", 0.08),
                    ("sequence_payoff_signal", 0.08),
                    ("playbook_knowledge_hint", 0.08),
                    ("action_is_task", 0.12),
                    ("action_family_activation", 0.06),
                ),
            ),
            refer_out_response_head=RetrievalReadoutHead(
                bias=0.08,
                weights=(
                    ("continuum_reconstruction_pressure", 0.22),
                    ("continuum_slow_share", 0.12),
                    ("switch_gate_signal", 0.08),
                    ("family_fast_prior_bias", -0.06),
                    ("sequence_payoff_signal", -0.06),
                ),
            ),
            answer_depth_head=RetrievalReadoutHead(
                bias=0.42,
                weights=(
                    ("world_bias", 0.12),
                    ("problem_solving_mode", 0.10),
                    ("continuum_reconstruction_pressure", 0.16),
                    ("continuum_focus", 0.10),
                    ("switch_gate_signal", 0.08),
                    ("playbook_knowledge_hint", 0.08),
                    ("family_fast_prior_bias", -0.06),
                ),
            ),
            continuum_target_head=RetrievalReadoutHead(
                bias=0.42,
                weights=(
                    ("support_mode", 0.10),
                    ("repair_mode", 0.08),
                    ("problem_solving_mode", -0.12),
                    ("self_bias", 0.08),
                    ("world_bias", -0.08),
                    ("continuum_focus", 0.10),
                    ("continuum_slow_share", 0.08),
                    ("continuum_reconstruction_pressure", 0.08),
                    ("action_is_stabilize", 0.06),
                    ("action_is_task", -0.08),
                    ("action_fast_prior_bias", 0.08),
                    ("action_family_activation", 0.08),
                    ("switch_gate_signal", 0.04),
                    ("family_fast_prior_bias", 0.05),
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
            support_response_head=self.support_response_head.shifted(
                bias_delta=max(action_shift, 0.0) * delta_scale * 0.12,
                weight_deltas=(
                    ("action_fast_prior_bias", max(action_shift, 0.0) * delta_scale * 0.16),
                    ("family_fast_prior_bias", max(family_shift, 0.0) * delta_scale * 0.10),
                ),
            ),
            clarify_response_head=self.clarify_response_head.shifted(
                bias_delta=max(regime_shift, 0.0) * delta_scale * 0.10,
                weight_deltas=(
                    ("continuum_reconstruction_pressure", bounded_strength * delta_scale * 0.16),
                    ("switch_gate_signal", max(action_shift, 0.0) * delta_scale * 0.12),
                ),
            ),
            structure_response_head=self.structure_response_head.shifted(
                bias_delta=max(regime_shift, 0.0) * delta_scale * 0.08,
                weight_deltas=(
                    ("sequence_payoff_signal", max(family_shift, 0.0) * delta_scale * 0.14),
                    ("delayed_payoff_signal", max(regime_shift, 0.0) * delta_scale * 0.12),
                    ("action_family_activation", max(family_shift, 0.0) * delta_scale * 0.10),
                ),
            ),
            refer_out_response_head=self.refer_out_response_head.shifted(
                bias_delta=bounded_strength * delta_scale * 0.08,
                weight_deltas=(
                    ("continuum_reconstruction_pressure", bounded_strength * delta_scale * 0.16),
                    ("switch_gate_signal", max(action_shift, 0.0) * delta_scale * 0.08),
                ),
            ),
            answer_depth_head=self.answer_depth_head.shifted(
                bias_delta=bounded_strength * delta_scale * 0.06,
                weight_deltas=(
                    ("continuum_reconstruction_pressure", bounded_strength * delta_scale * 0.14),
                    ("switch_gate_signal", max(action_shift, 0.0) * delta_scale * 0.08),
                ),
            ),
            continuum_target_head=self.continuum_target_head.shifted(
                bias_delta=(
                    max(action_shift, 0.0) * delta_scale * 0.10
                    - max(regime_shift, 0.0) * delta_scale * 0.06
                ),
                weight_deltas=(
                    ("family_fast_prior_bias", max(family_shift, 0.0) * delta_scale * 0.12),
                    ("action_fast_prior_bias", max(action_shift, 0.0) * delta_scale * 0.10),
                    ("continuum_focus", bounded_strength * delta_scale * 0.10),
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
        effective_parameters = self._parameters.updated_from_slow_prior(
            strength=inputs.fast_prior_strength,
            attribution_count=inputs.fast_prior_attribution_count,
            sequence_count=inputs.fast_prior_sequence_count,
            regime_bias=inputs.regime_fast_prior_bias,
            action_bias=inputs.action_fast_prior_bias,
            family_bias=inputs.family_fast_prior_bias,
            knowledge_weight_bias=inputs.knowledge_weight_bias,
            experience_weight_bias=inputs.experience_weight_bias,
        )
        knowledge_weight = _clamp(effective_parameters.knowledge_weight_head.evaluate(features))
        experience_weight = _clamp(effective_parameters.experience_weight_head.evaluate(features))
        total_weight = knowledge_weight + experience_weight
        if total_weight <= 1e-6:
            knowledge_weight = 0.5
            experience_weight = 0.5
        else:
            knowledge_weight = _clamp(knowledge_weight / total_weight)
            experience_weight = _clamp(experience_weight / total_weight)
        knowledge_domains = self._rank_knowledge_domains(
            knowledge_domains=inputs.knowledge_domains,
            features=features,
            parameters=effective_parameters,
        )
        experience_domains = self._adjust_experience_domains(
            experience_domains=inputs.experience_domains,
            features=features,
            parameters=effective_parameters,
        )
        response_mode_hint, clarification_bias, refer_out_bias, answer_depth_bias = (
            self._response_hints(features=features, parameters=effective_parameters)
        )
        continuum_target_position = _clamp(effective_parameters.continuum_target_head.evaluate(features))
        ordering_bias, ordering_driver = self._ordering_bias(
            response_mode_hint=response_mode_hint,
            continuum_target_position=continuum_target_position,
            experience_domains=experience_domains,
        )
        return RetrievalControlReadout(
            knowledge_domains=knowledge_domains,
            experience_domains=experience_domains,
            knowledge_weight=knowledge_weight,
            experience_weight=experience_weight,
            knowledge_domain_ranking=knowledge_domains,
            experience_domain_ranking=experience_domains,
            response_mode_hint=response_mode_hint,
            clarification_bias=clarification_bias,
            refer_out_bias=refer_out_bias,
            answer_depth_bias=answer_depth_bias,
            continuum_target_position=continuum_target_position,
            ordering_bias=ordering_bias,
            ordering_driver=ordering_driver,
            description=(
                f"Retrieval control readout selected {len(knowledge_domains)} knowledge domains and "
                f"{len(experience_domains)} experience domains from compact ETA/application control using "
                "owner-side parameterized readout with shared retrieval/response/boundary advisories. "
                f"bands={','.join(inputs.continuum_active_band_ids) if inputs.continuum_active_band_ids else 'none'} "
                f"regime={inputs.regime_id} abstract_action={inputs.abstract_action or 'none'}."
            ),
        )

    def _extract_features(self, inputs: RetrievalControlReadoutInputs) -> RetrievalReadoutFeatures:
        # W4 of ssot-cleanup-p0-p4: read regime mode features from
        # ApplicationBrief.task_focus / .support_focus / .repair_focus
        # rather than one-hot ``regime_id == 'X'`` checks. The brief
        # values are continuous (0..1) so a learned regime can express
        # mixed modes; the historical hard one-hot kept regime_id ==
        # "X" -> 1.0, which is the special case
        # ``brief.task_focus >= 0.85`` etc.
        from volvence_zero.regime import application_brief_for_regime

        action_label = (inputs.abstract_action or "").lower()
        brief = application_brief_for_regime(inputs.regime_id)
        problem_solving_mode = brief.task_focus
        support_mode = brief.support_focus
        repair_mode = brief.repair_focus
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
            action_family_activation=_clamp(min(max(inputs.action_family_version, 0), 6) / 6.0),
            switch_gate_signal=_clamp(inputs.switch_gate),
        )

    def _rank_knowledge_domains(
        self,
        *,
        knowledge_domains: tuple[str, ...],
        features: RetrievalReadoutFeatures,
        parameters: RetrievalControlReadoutParameters,
    ) -> tuple[str, ...]:
        if not knowledge_domains:
            return ()
        structured_score = parameters.structured_domain_head.evaluate(features)
        support_score = parameters.support_response_head.evaluate(features)
        if structured_score >= support_score:
            return knowledge_domains
        if len(knowledge_domains) <= 1:
            return knowledge_domains
        return (knowledge_domains[0],) + tuple(sorted(knowledge_domains[1:]))

    def _response_hints(
        self,
        *,
        features: RetrievalReadoutFeatures,
        parameters: RetrievalControlReadoutParameters,
    ) -> tuple[str, float, float, float]:
        support_score = _clamp(parameters.support_response_head.evaluate(features))
        clarify_score = _clamp(parameters.clarify_response_head.evaluate(features))
        structure_score = _clamp(parameters.structure_response_head.evaluate(features))
        refer_out_score = _clamp(parameters.refer_out_response_head.evaluate(features))
        if refer_out_score >= max(clarify_score, structure_score, support_score) and refer_out_score >= 0.56:
            response_mode_hint = "refer-out"
        elif clarify_score >= max(structure_score, support_score) and clarify_score >= 0.48:
            response_mode_hint = "clarify"
        elif structure_score >= support_score:
            response_mode_hint = "structure"
        else:
            response_mode_hint = "support"
        clarification_bias = _clamp(clarify_score * 0.7 + refer_out_score * 0.15)
        refer_out_bias = _clamp(refer_out_score * 0.75 + clarify_score * 0.10)
        answer_depth_bias = _clamp(parameters.answer_depth_head.evaluate(features))
        return response_mode_hint, clarification_bias, refer_out_bias, answer_depth_bias

    def _ordering_bias(
        self,
        *,
        response_mode_hint: str,
        continuum_target_position: float,
        experience_domains: tuple[str, ...],
    ) -> tuple[tuple[str, ...], str]:
        if response_mode_hint == "refer-out":
            return (("stabilize", "bounded_handoff"), "retrieval-refer-out")
        if response_mode_hint == "clarify":
            if continuum_target_position >= 0.66:
                return (("stabilize", "clarify_goal"), "retrieval-support-clarify")
            return (("clarify_goal", "split_axes"), "retrieval-clarify-first")
        if response_mode_hint == "structure":
            return (("structure_options", "smallest_next_step"), "retrieval-structure-first")
        if continuum_target_position >= 0.66 or "stabilization_patterns" in experience_domains:
            return (("stabilize", "acknowledge"), "retrieval-support-first")
        return (("acknowledge", "smallest_next_step"), "retrieval-support-balanced")

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
